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
from fla.modules.layernorm_gated import RMSNormGated
from fla.modules.activations import ACT2FN
from palimpsa.ops.palimpsa import chunk_palimpsa, fused_recurrent_palimpsa
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
import wandb 

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class MetaMamba2(nn.Module):
    """
    Meta-Mamba2 Layer.
    Adds Bayesian Metaplasticity terms (Ip, Beta) to the standard Mamba2 architecture.
    Uses Triton-only backend.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int = 64,
        hidden_size: int = 2048,
        state_size: int = 128,
        expand: int = 2,
        n_groups: int = 1,
        conv_kernel: int = 4,
        use_conv_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm: bool = True,
        chunk_size: int = 256,
        time_step_rank: float = 256,
        time_step_limit: tuple[float, float] = (0.0, float("inf")),
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        use_bias: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        metaplasticity: bool = True,
        finetuning: bool = False,
        beta_step_rank: int=128,
        mode: str = 'chunk',
    ) -> MetaMamba2:
        super().__init__()

        self.metaplasticity = metaplasticity
        self.finetuning = finetuning
        self.beta_step_rank = beta_step_rank
        self.mode = mode

        if not self.metaplasticity:
             warnings.warn("⚠️ MetaMamba2 running in Mamba2 mode (Metaplasticity=False).")
        if self.finetuning:
             warnings.warn("⚠️ MetaMamba2 running in FINETUNING mode.")

        #num_heads equivalent to num_v_heads in gated deltanet 
        self.num_heads = num_heads 
        #head_dim equivalent to head_v_dim in gated deltanet 
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        #ssm_state_size equivalent to head_k_dim in gated deltanet
        self.ssm_state_size = state_size
        self.expand = expand
        self.intermediate_size = int(expand * hidden_size)
        #n_groups equivalent to num_k_heads in gated deltanet 
        self.n_groups = n_groups

        self.conv_kernel_size = conv_kernel
        self.use_conv_bias = use_conv_bias
        self.activation = hidden_act
        self.act = ACT2FN[hidden_act]

        self.rms_norm = rms_norm
        self.norm_eps = norm_eps

        self.chunk_size = chunk_size

        self.time_step_rank = int(time_step_rank)
        self.time_step_limit = time_step_limit
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        
        self.conv1d =ShortConvolution(hidden_size=self.conv_dim, kernel_size=conv_kernel, bias=use_conv_bias, activation=self.act)
        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = RMSNormGated(
            self.intermediate_size, eps=self.norm_eps, norm_before_gate=False,
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=use_bias)
        self.use_bias = use_bias

        self.layer_idx = layer_idx

        # Metaplasticity parameters
        self.b_rank_proj = nn.Linear(hidden_size, self.beta_step_rank, bias=False)
        self.b_proj = nn.Linear(self.beta_step_rank, self.intermediate_size, bias=False)
        self.b_scale = nn.Parameter(torch.ones(self.num_heads))
        self.Ip_log = nn.Parameter(torch.zeros(self.num_heads), requires_grad=False)
        self.Ip_log._no_weight_decay = True




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
            
        projected_states = self.in_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        gate, x_B_C, dt = projected_states.split(
            [d_mlp, self.conv_dim, self.num_heads], dim=-1
        )

        # 2. Convolution sequence transformation
        conv_state= None
        if last_state is not None and last_state.get('conv_state') is not None:
            conv_state = last_state['conv_state']

        x_B_C, conv_state = self.conv1d(
            x=x_B_C,
            cache=conv_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,)
        
        x, B, C = torch.split(
            x_B_C,
            [
                self.intermediate_size,
                groups_time_state_size,
                groups_time_state_size,
            ],
            dim=-1,
        )

        C, B = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.ssm_state_size), (C, B))
        x = rearrange(x, '... (h d) -> ... h d', d=self.head_dim)

        if self.num_heads > self.num_groups:
            C, B = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_heads // self.num_groups), (C, B))

        dt = F.softplus(dt.float() + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
        A = self.A_log.float().exp()
        # Operation done in Mamba2
        # dB = dt[..., None] * B[..., None, :]
        # The thing is this could be done on x or B in mamba2 without changing the outcome
        # But in palimpsa doing it on B (equivalent to k) is not equivalent to doing it on x (equivalent to v) because of the importance update (+k**2 * b)
        # So we do the operation on the x and on b. If b is multiply by dt**2 it's as if we multiplied k by dt. Else b is multiply by dt.   


        dx = x * dt.unsqueeze(-1)

        b = torch.ones(1, device=C.device) 
        if self.metaplasticity:
            b_raw = self.b_proj(self.b_rank_proj(hidden_states))
            b_raw = rearrange(b_raw, '... (h d) -> ... h d', d=self.head_dim)
            b = torch.sigmoid(b_raw) * self.b_scale.view(1, 1, -1, 1)
            b = (b * dt.unsqueeze(-1)).to(hidden_states.dtype) #Could be possible to use dt**2 depends on interpratation
        
        # Diagnostic block
        if self.training and not hasattr(self, "_mangled") and self.layer_idx == 0:
            with torch.no_grad():
                #Some other stuff could be plot to see if everything is how it is supposed to be. 
                br_std = self.b_rank_proj.weight.std().item()
                bp_std = self.b_proj.weight.std().item()
                b_scale = self.b_scale.mean().item()
                if wandb.run is not None and torch.distributed.get_rank() == 0:
                    wandb.log({
                        "diag/L0_b_rank_proj_std": br_std,
                        "diag/L0_b_proj_std": bp_std,
                        "diag/L0_b_output_std": b.std().item(),
                        "diag/L0_b_scale": b_scale,
                    }, commit=False)
                self._mangled = True

        Ip = torch.exp(self.Ip_log.float())
        
        recurrent_state = None
        if last_state is not None:
            recurrent_state = last_state.get('recurrent_state') if isinstance(last_state, dict) else last_state[0]

        if not self.metaplasticity:
            # Then the model is equivalent to mamba2
            # We use simple GLA to not have to compile mamba_ssm 
            # It's really important to use scale=1 here since in mamba two the rms_norm is performed usually after the output gating 
            g_log = -dt * A 
            if mode == 'chunk':
                outputs = chunk_simple_gla(q=C, k=B, v=dx, g=g_log, scale=1.0, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            else:
                outputs = fused_recurrent_simple_gla(q=C, k=B, v=dx, g=g_log, scale=1.0, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            o, final_state = outputs if isinstance(outputs, tuple) else (outputs, None)
            recurrent_state = final_state if use_cache else None
        else:
            active_mu = active_I = None
            if recurrent_state is not None and isinstance(recurrent_state, (list, tuple)):
                active_mu, active_I = recurrent_state[0], recurrent_state[1]
            elif recurrent_state is not None:
                active_mu = recurrent_state
            
            if mode == 'chunk':
                outputs = chunk_palimpsa(q=C, k=B, v=dx, b=b, gt=dt, g=A, Ip=Ip, scale=1.0, output_final_state=use_cache, cu_seqlens=cu_seqlens, chunk_size=16, initial_mu_state=active_mu, initial_I_state=active_I)
                if use_cache:
                    o, final_mu, final_I = outputs
                    recurrent_state = (final_mu, final_I)
                else:
                    o = outputs
                    recurrent_state = None
            else:
                outputs = fused_recurrent_palimpsa(q=C, k=B, v=dx, b=b, gt=dt, g=A, Ip=Ip, initial_mu_state=active_mu, initial_I_state=active_I, output_final_state=use_cache, cu_seqlens=cu_seqlens, scale=1.0)
                if use_cache:
                    o, final_mu, final_I = outputs
                    recurrent_state = (final_mu, final_I)
                else:
                    o = outputs
                    recurrent_state = None

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=q_len
            )
        o = (o + x * self.D[None,None,:,None]).to(o.dtype)
        o = self.norm(o, gate)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values


 