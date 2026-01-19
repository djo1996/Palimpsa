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
    """
    Palimpsa Layer.
    Adds Bayesian Metaplasticity terms (Ip, Beta) to a custom Simple GLA architecture.
    Uses Triton-only backend.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        expand_k: float = 1,
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
        metaplasticity: bool = True,
        finetuning: bool = False,
        use_residual: bool = True,
        init_diagnosis: bool = True,
        eval_diagnosis: bool = False,
        **kwargs,
    ) -> Palimpsa:
        super().__init__()

        self.metaplasticity = metaplasticity
        self.finetuning = finetuning
        self.use_residual = use_residual
        self.init_diagnosis = init_diagnosis
        self.eval_diagnosis = eval_diagnosis

        if not self.metaplasticity:
             warnings.warn("⚠️ Palimpsa running in SimpleGLA mode (Metaplasticity=False).")
        if self.finetuning:
             warnings.warn("⚠️ Palimpsa running in FINETUNING mode.")

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.expand_k = expand_k
 
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.beta_step_rank = beta_step_rank

        self.head_k_dim = int(self.head_dim * self.expand_k)
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

    
        self.b_scale = nn.Parameter(torch.ones(self.num_v_heads))
        self.b_scale._no_weight_decay = True

        self.bs_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
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
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
        
        self.D = nn.Parameter(torch.ones(self.num_v_heads))
        self.D._no_weight_decay = True

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _diag_init(self, k, b, b_scale, dt, A):
        """Logs initialization statistics to WandB."""
        if not (wandb.run is not None and is_master()):
            return

        with torch.no_grad():
            br_std = self.b_rank_proj.weight.std().item()
            bp_std = self.b_proj.weight.std().item()
            kp_std = self.k_proj.weight.std().item()
            b_scale_val = b_scale.mean().item()
          
            # Compute N = 1 / (1 - exp(-A*dt))
            decay = torch.exp(-A * dt)
            n_val = 1.0 / (1.0 - decay + 1e-6) 
            n_avg = n_val.mean(dim=(0, 1))

            metrics = {
                f"diag_init/L{self.layer_idx}_b_rank_proj_std": br_std,
                f"diag_init/L{self.layer_idx}_b_proj_std": bp_std,
                f"diag_init/L{self.layer_idx}_k_proj_std": kp_std,
                f"diag_init/L{self.layer_idx}_b_scale": b_scale_val,
            }

            # 2. Add b_std only if b exists
            if b is not None:
                metrics[f"diag_init/L{self.layer_idx}_b_output_std"] = b.std().item()

            # 3. Add N_avg
            for h in range(len(n_avg)):
                metrics[f"diag_init/L{self.layer_idx}_N_avg/H{h}"] = n_avg[h].item()

            wandb.log(metrics, commit=False)

    def _diag_eval(self, final_I, b, dt, A):
        """Logs evaluation statistics for the final State I and b per head."""
        if final_I is None or not (wandb.run is not None and is_master()):
            return
        with torch.no_grad():
            metrics = {}
            H = final_I.shape[1]
            current_b_scales = F.softplus(self.b_scale).detach()
            
            # Compute N = 1 / (1 - exp(-A*dt))
            decay = torch.exp(-A * dt)
            n_val = 1.0 / (1.0 - decay + 1e-6) 
            n_avg = n_val.mean(dim=(0, 1))

            for h in range(H):
                # --- Scalar Metrics ---
                state_h = final_I[:, h, ...] 
                metrics[f"diag_eval/L{self.layer_idx}_I_Range/H{h}"] = (state_h.max() - state_h.min()).item()
                metrics[f"diag_eval/L{self.layer_idx}_I_Mean/H{h}"] = state_h.mean().item()
                metrics[f"diag_eval/L{self.layer_idx}_I_Std/H{h}"] = state_h.std().item()
                
                if b is not None:
                    metrics[f"diag_eval/L{self.layer_idx}_b_std/H{h}"] = b[:, :, h, :].std().item()

                metrics[f"diag_eval/L{self.layer_idx}_b_scale/H{h}"] = current_b_scales[h].item()
                metrics[f"diag_eval/L{self.layer_idx}_N_avg/H{h}"] = n_avg[h].item()
                metrics[f"diag_eval/L{self.layer_idx}_A/H{h}"] = A[h].item()
                metrics[f"diag_eval/L{self.layer_idx}_dt_avg/H{h}"] = dt[:, :, h].mean().item()
            wandb.log(metrics, commit=False)

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
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None and last_state.get('conv_state') is not None:
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
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        x = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        dt = F.softplus(self.dt_proj(hidden_states).float() + self.dt_bias)
        A = self.A_log.float().exp()
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        bs = torch.sigmoid(self.bs_proj(hidden_states).float()).to(hidden_states.dtype)
        v = x * bs.unsqueeze(-1)

        if self.metaplasticity:
            b_raw = self.b_proj(self.b_rank_proj(hidden_states)).float()
            b_raw = rearrange(b_raw, '... (h d) -> ... h d', d=self.head_v_dim)
            b = torch.sigmoid(b_raw) * F.softplus(self.b_scale.view(1, 1, -1, 1).float())
            b = (b * bs.unsqueeze(-1)).to(hidden_states.dtype)
        else: 
            b = None
        
        Ip = torch.exp(self.Ip_log.float())
        # [Diagnostic Init Block]
        if self.init_diagnosis and self.training and not hasattr(self, "_mangled") and self.layer_idx == 0:
            self._diag_init(k, b, self.b_scale, dt, A)
            self._mangled = True

        recurrent_state = None
        if last_state is not None:
            recurrent_state = last_state.get('recurrent_state') if isinstance(last_state, dict) else last_state[0]

        if not self.metaplasticity:
            g_log = -dt * A 
            if mode == 'chunk':
                outputs = chunk_simple_gla(q=q, k=k, v=v, g=g_log, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            else:
                outputs = fused_recurrent_simple_gla(q=q, k=k, v=v, g=g_log, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            o, final_state = outputs if isinstance(outputs, tuple) else (outputs, None)
            recurrent_state = final_state if use_cache else None
        else:
            active_mu = active_I = None
            if recurrent_state is not None and isinstance(recurrent_state, (list, tuple)):
                active_mu, active_I = recurrent_state[0], recurrent_state[1]
            elif recurrent_state is not None:
                active_mu = recurrent_state
            
            if mode == 'chunk':
                outputs = chunk_palimpsa(q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip, output_final_state=use_cache, cu_seqlens=cu_seqlens, chunk_size=16, initial_mu_state=active_mu, initial_I_state=active_I)
                if use_cache:
                    o, final_mu, final_I = outputs
                    recurrent_state = (final_mu, final_I)
                else:
                    o = outputs
                    recurrent_state = None
            else:
                outputs = fused_recurrent_palimpsa(q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip, initial_mu_state=active_mu, initial_I_state=active_I, output_final_state=use_cache, cu_seqlens=cu_seqlens)
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

        # [Diagnostic Eval Block]
        if self.eval_diagnosis and not self.training and self.metaplasticity:
             self._diag_eval(final_I, b, dt, A)

        if self.use_residual:
            o = (o + x * self.D[None, None, :, None])

        if self.use_gate:
            o = self.o_norm(o, rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim))
        else:
            o = self.o_norm(o)
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values



