# -*- coding: utf-8 -*-
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
import wandb 

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

import torch.distributed as dist

def is_master():
    """Returns True if not in a distributed environment OR if rank is 0."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

class Palimpsa(nn.Module):
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
        gumbel_temp: float=0.5, 
        k_temp: float=1,
        **kwargs,
    ) -> Palimpsa:
        super().__init__()

        self.qk_act = qk_act 
        self.metaplasticity = metaplasticity
        self.finetuning = finetuning

        if self.qk_act not in ['softmax', 'siluL2', 'silu', 'siluL2softmax']:
            warnings.warn(f"⚠️ Palimpsa non-standard query/key activation: '{self.qk_act}'")
        if not self.metaplasticity:
             warnings.warn("⚠️ Palimpsa running in SimpleGLA mode (Metaplasticity=False).")
        if self.finetuning:
             warnings.warn("⚠️ Palimpsa running in FINETUNING mode.")

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.reduct_k = reduct_k
        self.gumbel_temp = gumbel_temp
        self.k_temp = k_temp

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

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.b_rank_proj = nn.Linear(hidden_size, self.beta_step_rank, bias=False)
        self.b_proj = nn.Linear(self.beta_step_rank, self.value_dim, bias=False)

        #New
        b_scale_min, b_scale_max = 0.1, 10
        b_scale = torch.exp(torch.rand(self.num_v_heads) * (math.log(b_scale_max) - math.log(b_scale_min)) + math.log(b_scale_min)).clamp(min=1e-4)
        inv_b_scale = b_scale + torch.log(-torch.expm1(-b_scale))
        self.b_scale = nn.Parameter(inv_b_scale)
        self.b_scale._no_weight_decay = True

        self.bs_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.Ip_log = nn.Parameter(torch.zeros(self.num_v_heads), requires_grad=False)
        self.Ip_log._no_weight_decay = True

        self.dt_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation=None,)
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation=None)
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')

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
        
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        
        if attention_mask is not None:
            # We only pass cu_seqlens to ShortConvolution, NOT mask. 
            # Doing both causes the ValueError you encountered.
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None and last_state.get('conv_state') is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            
            # FLA modules: mask and cu_seqlens are mutually exclusive.
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
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        dt = F.softplus(self.dt_proj(hidden_states).float() + self.dt_bias)
        A = self.A_log.float().exp()

        if self.qk_act == 'softmax':
            q, k = F.softmax(q, dim=-1), F.softmax(k/self.k_temp, dim=-1)
        elif self.qk_act == 'siluL2':
            q, k = F.normalize(F.silu(q), p=2, dim=-1), F.normalize(F.silu(k), p=2, dim=-1)
        elif self.qk_act == 'silu':
            q, k = F.silu(q), F.silu(k)
        elif self.qk_act == 'siluL2softmax':
            q = F.normalize(F.silu(q), p=2, dim=-1)
            k = F.softmax(k/self.k_temp, dim=-1)
        elif self.qk_act == 'gumbel_softmax':
            q = F.softmax(q, dim=-1)
            # hard=True enables the one-hot encoding for the forward pass
            k = F.gumbel_softmax(k, tau=self.gumbel_temp, hard=True, dim=-1)

        bs = torch.sigmoid(self.bs_proj(hidden_states).float()).to(hidden_states.dtype)
        v = v * bs.unsqueeze(-1)

        b = torch.ones(1, device=q.device) 
        if self.metaplasticity:
            b_raw = self.b_proj(self.b_rank_proj(hidden_states)).float()
            b_raw = rearrange(b_raw, '... (h d) -> ... h d', d=self.head_v_dim)
            b = torch.sigmoid(b_raw) * F.softplus(self.b_scale.view(1, 1, -1, 1).float())
            b = (b * bs.unsqueeze(-1)).to(hidden_states.dtype)
        
        # Diagnostic block
        if self.training and not hasattr(self, "_mangled") and self.layer_idx == 0:
            with torch.no_grad():
                br_std = self.b_rank_proj.weight.std().item()
                bp_std = self.b_proj.weight.std().item()
                b_scale = self.b_scale.mean().item()
                # Entropy check for K distribution
                if self.qk_act in ['softmax', 'siluL2softmax', 'gumbel_softmax']:
                    token_entropy = -torch.sum(k * torch.log(k + 1e-9), dim=-1)
                    h_local = token_entropy.mean().item()
                    global_p = k.mean(dim=(0, 1))
                    h_global = -torch.sum(global_p * torch.log(global_p + 1e-9), dim=-1).mean().item()
                else:
                    h_local = h_global = 0.0

                if wandb.run is not None and is_master():
                    wandb.log({
                        "diag/L0_b_rank_proj_std": br_std,
                        "diag/L0_b_proj_std": bp_std,
                        "diag/L0_k_entropy_local": h_local,
                        "diag/L0_k_entropy_global": h_global,
                        "diag/L0_b_output_std": b.std().item(),
                        "diag/L0_b_scale": b_scale,
                        "diag/L0_gumbel_temp": self.gumbel_temp,
                        "diag/L0_k_temp": self.k_temp,
                    }, commit=False)
                self._mangled = True

        Ip = torch.exp(self.Ip_log.float())
        
        recurrent_state = None
        if last_state is not None:
            recurrent_state = last_state.get('recurrent_state') if isinstance(last_state, dict) else last_state[0]

        if not self.metaplasticity:
            g_log = -dt * A 
            if mode == 'chunk':
                outputs = chunk_simple_gla(q=q, k=k, v=v, g=g_log, scale=1.0, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            else:
                outputs = fused_recurrent_simple_gla(q=q, k=k, v=v, g=g_log, scale=1.0, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            o, final_state = outputs if isinstance(outputs, tuple) else (outputs, None)
            recurrent_state = final_state if use_cache else None
        else:
            active_mu = active_I = None
            if recurrent_state is not None and isinstance(recurrent_state, (list, tuple)):
                active_mu, active_I = recurrent_state[0], recurrent_state[1]
            elif recurrent_state is not None:
                active_mu = recurrent_state
            
            if mode == 'chunk':
                outputs = chunk_palimpsa(q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip, scale=1.0, output_final_state=use_cache, cu_seqlens=cu_seqlens, chunk_size=16, initial_mu_state=active_mu, initial_I_state=active_I)
                if use_cache:
                    o, final_mu, final_I = outputs
                    recurrent_state = (final_mu, final_I)
                else:
                    o = outputs
                    recurrent_state = None
            else:
                outputs = fused_recurrent_palimpsa(q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip, initial_mu_state=active_mu, initial_I_state=active_I, output_final_state=use_cache, cu_seqlens=cu_seqlens, scale=1.0)
                if use_cache:
                    o, final_mu, final_I = outputs
                    recurrent_state = (final_mu, final_I)
                else:
                    o = outputs
                    recurrent_state = None

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_gate:
            o = self.o_norm(o, rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim))
        else:
            o = self.o_norm(o)
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values



