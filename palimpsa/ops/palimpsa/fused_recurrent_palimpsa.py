# -*- coding: utf-8 -*-
# Copyright (c) 2025, Djohan Bonnet
# Fused Recurrent Implementation of Bayesian Metaplastic Attention (Palimpsa)
# Strictly typed to match Chunk Kernel vocabulary.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from fla.ops.utils import contiguous
from fla.modules.l2norm import l2norm_fwd
from fla.utils import input_guard

# ----------------------------------------------------------------------------
# Fused Recurrent Forward Kernel (Triton)
# ----------------------------------------------------------------------------
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['initial_mu_state'] is not None,
    'STORE_FINAL_STATE': lambda args: args['final_mu_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'OUTPUT_UNCERTAINTY': lambda args: args['o_var'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_palimpsa_recurrent_fwd_kernel(
    q, k, v, b, gt, g, Ip,              
    o, o_var,                                
    initial_mu_state, initial_I_state,  # [B, H, D_V, D_K]
    final_mu_state, final_I_state,      
    cu_seqlens,                         
    scale, 
    T,                                 
    B: tl.constexpr,
    H: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    OUTPUT_UNCERTAINTY: tl.constexpr
):
    # -----------------------------------------------------------------------
    # Grid: (NV, B * H)
    # -----------------------------------------------------------------------
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    # -----------------------------------------------------------------------
    # Sequence Boundaries & Steps (Fixed Logic)
    # -----------------------------------------------------------------------
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q  = q + (bos * H + i_h) * D_K + o_k
    p_k  = k + (bos * H + i_h) * D_K + o_k
    p_v  = v + (bos * H + i_h) * D_V + o_v
    p_b  = b + (bos * H + i_h) * D_V + o_v
    p_gt = gt + (bos * H + i_h)

    p_o = o + (bos * H + i_h) * D_V + o_v

    # Fix: Initialize p_o_var only if needed
    if OUTPUT_UNCERTAINTY:
        p_o_var = o_var + (bos * H + i_h) * D_V + o_v

    mask_k = o_k < D_K
    mask_v = o_v < D_V
    mask_h = mask_v[:, None] & mask_k[None, :]
    
    # Scalar loads
    b_Ip = tl.load(Ip + i_h)
    b_g = tl.load(g + i_h)

    # -----------------------------------------------------------------------
    # State Initialization (SRAM)
    # -----------------------------------------------------------------------
    # MODIFIED: Renamed b_s -> b_M and b_si -> b_I_bar for clarity
    b_M = tl.zeros([BV, BK], dtype=tl.float32)
    b_I_bar = tl.zeros([BV, BK], dtype=tl.float32) 

    # Load Initial State: (State is NOT flattened, index by i_n)
    if USE_INITIAL_STATE:
        p_s = initial_mu_state + i_nh * D_V*D_K + o_v[:, None] * D_K + o_k[None, :]
        p_si = initial_I_state + i_nh * D_V*D_K + o_v[:, None] * D_K + o_k[None, :]
        
        b_mu_init = tl.load(p_s, mask=mask_h, other=0).to(tl.float32)
        b_I_init = tl.load(p_si, mask=mask_h, other=0).to(tl.float32)
        
        b_M = b_mu_init * b_I_init # M = mu * I
        b_I_bar = b_I_init - b_Ip  # I_bar = I - Ip


    for t in range(T):
        
        b_q = tl.load(p_q, mask=mask_k, other=0)
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_b = tl.load(p_b, mask=mask_v, other=0)
        b_gt = tl.load(p_gt)
        
        # 2. Compute Decay
        decay = tl.exp(-b_gt * b_g)

        # 3. Update States (M and I_bar)
        b_M = b_M * decay + (b_v[:, None] * b_k[None, :])
        
        # Diagonal approximation of precision update (element-wise square of k)
        b_kk = b_k * b_k 
        b_I_bar = b_I_bar * decay + (b_b[:, None] * b_kk[None, :])

        # 4. Compute Output
        current_I_full = b_I_bar + b_Ip
        
        # MODIFIED: Using b_M (Numerator) / I (Denominator) to get Mu
        current_mu = b_M / current_I_full
        
        weighted_mu = current_mu * b_q[None, :] * scale
        out_val = tl.sum(weighted_mu, axis=1)
        tl.store(p_o, out_val, mask=mask_v)
        
        # 5. Advance Pointers for the next token
        p_q += H*D_K
        p_k += H*D_K
        p_v += H*D_V
        p_b += H*D_V
        p_o += H*D_V
        p_gt+=H

        if OUTPUT_UNCERTAINTY:
            weighted_var = b_q[None, :] * b_q[None, :] * scale / current_I_full
            out_var_val = tl.sum(weighted_var, axis=1)
            tl.store(p_o_var, out_var_val, mask=mask_v)
            p_o_var += H*D_V


    # -----------------------------------------------------------------------
    # Store Final State
    # -----------------------------------------------------------------------
    if STORE_FINAL_STATE:
        p_sT = final_mu_state + i_nh * D_V*D_K + o_v[:, None] * D_K + o_k[None, :]
        p_siT = final_I_state + i_nh * D_V*D_K + o_v[:, None] * D_K + o_k[None, :]

        # MODIFIED: Reconstruct final Mu and I from M and I_bar
        final_I = b_I_bar + b_Ip
        final_Mu = b_M / final_I

        tl.store(p_sT, final_Mu.to(p_sT.dtype.element_ty), mask=mask_h)
        tl.store(p_siT, final_I.to(p_siT.dtype.element_ty), mask=mask_h)
       

# ----------------------------------------------------------------------------
# Python Utility Wrapper (Internal use)
# ----------------------------------------------------------------------------
def fused_palimpsa_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    gt: torch.Tensor,
    g: torch.Tensor,
    Ip: torch.Tensor,
    scale: float = None,
    initial_mu_state: torch.Tensor = None,
    initial_I_state: torch.Tensor = None,
    output_final_state: bool = False,
    output_uncertainty: bool = False,
    cu_seqlens: torch.LongTensor = None,
):
    B, T, H, D_K, D_V = *k.shape, v.shape[-1]
    
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # Block Sizes
    BK, BV = triton.next_power_of_2(D_K), min(triton.next_power_of_2(D_V), 8)
    NV = triton.cdiv(D_V, BV)
    num_stages = 3
    num_warps = 1
    # Output and State allocation
    o = torch.empty_like(v, dtype=q.dtype) 
    if output_uncertainty:
        o_var = torch.empty_like(v, dtype=q.dtype) 
    else:
        o_var = None
    
    final_mu_state = q.new_empty(N, H, D_V, D_K, dtype=torch.float32) if output_final_state else None
    final_I_state = q.new_empty(N, H, D_V, D_K, dtype=torch.float32) if output_final_state else None
   
    grid = (NV, N * H) 

    fused_palimpsa_recurrent_fwd_kernel[grid](
        q, k, v, b, gt, g, Ip,
        o, o_var,
        initial_mu_state, initial_I_state,
        final_mu_state, final_I_state,
        cu_seqlens,
        scale,
        T, B,
        H=H, D_K=D_K, D_V=D_V,
        BK=BK, BV=BV,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return o, o_var, final_mu_state, final_I_state


# ----------------------------------------------------------------------------
# Autograd Wrapper (Top-level)
# ----------------------------------------------------------------------------
class FusedRecurrentPalimpsaFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, b, gt, g, Ip, scale, initial_mu_state, initial_I_state, output_final_state, output_uncertainty, cu_seqlens):
        
        # -----------------------------------------------------------------
        # Dtype Management
        # -----------------------------------------------------------------
        ctx.original_v_dtype = v.dtype

        # Cast all inputs to float32 
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        b = b.to(torch.float32)
        gt = gt.to(torch.float32)
        g = g.to(torch.float32)
        Ip = Ip.to(torch.float32)
        
        # Safe handling of optional states
        if initial_mu_state is not None: initial_mu_state = initial_mu_state.to(torch.float32)
        if initial_I_state is not None: initial_I_state = initial_I_state.to(torch.float32)

        # FIX: Added missing output_uncertainty argument here
        o, o_var, final_mu_state, final_I_state = fused_palimpsa_recurrent_fwd(
            q, k, v, b, gt, g, Ip, scale,
            initial_mu_state, initial_I_state, 
            output_final_state,
            output_uncertainty, # <--- THIS WAS MISSING
            cu_seqlens 
        )
        
        o = o.to(ctx.original_v_dtype) 
        return o, o_var, final_mu_state, final_I_state

    @staticmethod
    @input_guard
    def backward(ctx, do, do_var, d_final_mu, d_final_I):
        raise NotImplementedError("Fused recurrent kernel is currently inference-only.")

# ----------------------------------------------------------------------------
# Entry Point Function
# ----------------------------------------------------------------------------
def fused_recurrent_palimpsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    gt: torch.Tensor,
    g : torch.Tensor,
    Ip : torch.Tensor,
    scale: float = None,
    initial_mu_state: Optional[torch.Tensor] = None, 
    initial_I_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    output_uncertainty: bool = False, 
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
        if initial_mu_state is not None and initial_mu_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"Initial states mismatch sequence length.")
        
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, o_var, final_mu_state, final_I_state = FusedRecurrentPalimpsaFunction.apply(
        q, k, v, b, gt, g, Ip, 
        scale,
        initial_mu_state, initial_I_state,
        output_final_state,
        output_uncertainty,
        cu_seqlens
    )
    if output_uncertainty:
        out = (o, o_var)
    else:
        out = o

    if output_final_state:
        return out, final_mu_state, final_I_state
    return out