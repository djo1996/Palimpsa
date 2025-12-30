# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from palimpsa.ops.palimpsa import chunk_palimpsa, fused_recurrent_palimpsa
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class Palimpsa(nn.Module):
    """
    The layer implementation for Palimpsa.
    Adapted to match fla/flame interface with variable length support.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        reduct_k: float = 1,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        beta_step_rank: int = 128,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        qk_act: str = 'softmax',
        metaplasticity: bool = True,
        finetuning: bool = False,
        **kwargs,
    ) -> Palimpsa:
        super().__init__()

        self.qk_act = qk_act 
        self.metaplasticity = metaplasticity
        self.finetuning = finetuning

        if self.qk_act != 'softmax':
            warnings.warn(
                f"⚠️  Palimpsa is using non-standard query/key activation: '{self.qk_act}'"
            )
        if not self.metaplasticity:
             warnings.warn("⚠️  Palimpsa is running in SimpleGLA mode (Metaplasticity=False).")
        
        if self.finetuning:
             warnings.warn("⚠️  Palimpsa is running in FINETUNING mode (Special b_proj Init Active).")

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.reduct_k = reduct_k

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.beta_step_rank = beta_step_rank

        self.head_k_dim = math.ceil(head_dim / reduct_k)
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(f"Invalid value_dim configuration.")
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.b_rank_proj = nn.Linear(hidden_size, self.beta_step_rank, bias=False)
        self.b_proj = nn.Linear(self.beta_step_rank, self.value_dim, bias=True)
        
        if self.finetuning:
            beta_min = 0.001
            beta_max = 0.1
            beta_init_floor = 1e-4
            b_val = torch.exp(
                torch.rand(self.value_dim) * (math.log(beta_max) - math.log(beta_min))
                + math.log(beta_min),
            )
            b_val = torch.clamp(b_val, min=beta_init_floor)
            inv_b_val = b_val + torch.log(-torch.expm1(-b_val))
            
            with torch.no_grad():
                self.b_proj.bias.copy_(inv_b_val)
            
            self.b_proj.bias._no_weight_decay = True
            self.b_proj.bias._no_reinit = True

        self.bs_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.Ip_log = nn.Parameter(torch.zeros(self.num_v_heads), requires_grad=False)
        self.Ip_log._no_weight_decay = True

        self.dt_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off unless you know what you are doing.",
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix..."

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        dt = F.softplus(self.dt_proj(hidden_states).float() + self.dt_bias)
        A = self.A_log.float().exp()

        if self.qk_act =='softmax':
            q = F.softmax(q, dim=-1)
            k = F.softmax(k, dim=-1)
        elif self.qk_act == 'siluL2':
            q = F.normalize(F.silu(q), p=2, dim=-1)
            k = F.normalize(F.silu(k), p=2, dim=-1)
        elif self.qk_act == 'siluL2softmax':
            q = F.normalize(F.silu(q), p=2, dim=-1)
            k = F.softmax(k, dim=-1)
        elif self.qk_act == 'silu':
            q = F.silu(q)
            k = F.silu(k)
        else:
            raise ValueError(f"qk_act={self.qk_act} not a valid entry.")

        
        bs = torch.sigmoid(self.bs_proj(hidden_states).float())
        bs = bs.to(hidden_states.dtype)
        v = v * bs.unsqueeze(-1)

        if not self.metaplasticity:
            g_log = -dt * A 
            recurrent_state = None
            if last_state is not None:
                recurrent_state = last_state.get('recurrent_state')

            if mode == 'chunk':
                outputs = chunk_simple_gla(
                    q=q, k=k, v=v, g=g_log,
                    scale=1.0, 
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
            elif mode == 'fused_recurrent':
                outputs = fused_recurrent_simple_gla(
                    q=q, k=k, v=v, g=g_log,
                    scale=1.0,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
            else:
                raise NotImplementedError(f"Mode `{mode}` not supported for simple_gla.")
            
            if isinstance(outputs, tuple):
                o = outputs[0]
                final_state = outputs[1]
            else:
                o = outputs
                final_state = None

            recurrent_state = final_state if use_cache else None

        else:
            b = self.b_proj(self.b_rank_proj(hidden_states)).float()
            b = rearrange(b, '... (h d) -> ... h d', d=self.head_v_dim)
            b = torch.sigmoid(b).to(hidden_states.dtype) 
            b = b * bs.unsqueeze(-1)
            

            Ip = torch.exp(self.Ip_log.float())

            active_mu, active_I = None, None
            if last_state is not None:
                state_tuple = last_state.get('recurrent_state')
                if state_tuple is not None:
                    active_mu, active_I = state_tuple

            if mode == 'chunk':
                outputs = chunk_palimpsa(
                    q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip,
                    scale=1.0,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    chunk_size=8,
                )
                if use_cache:
                    o, final_mu, final_I = outputs
                    recurrent_state = (final_mu, final_I)
                else:
                    o = outputs
                    recurrent_state = None
                    
            elif mode == 'fused_recurrent':
                outputs = fused_recurrent_palimpsa(
                    q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip,
                    initial_mu_state=active_mu,
                    initial_I_state=active_I,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    scale=1.0,
                )
                if use_cache:
                    o, final_mu, final_I = outputs
                    recurrent_state = (final_mu, final_I)
                else:
                    o = outputs
                    recurrent_state = None
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
            
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values