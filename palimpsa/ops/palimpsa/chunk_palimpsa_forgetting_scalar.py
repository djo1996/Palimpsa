# -*- coding: utf-8 -*-
# Copyright (c) 2025, Djohan Bonnet
# Optimized Implementation of Palimpsa

import torch
import torch.nn.functional as F 
import triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_offsets, prepare_chunk_indices
from fla.utils import use_cuda_graph


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper

# -----------------------------------------------------------------------------
# Autotune Configurations
# -----------------------------------------------------------------------------


FWD_BV_LIST = [16]      
FWD_BK_LIST = [8]      
FWD_ST_LIST = [3]
FWD_WP_LIST = [2]     

BWD_BV_LIST = [16]      
BWD_BK_LIST = [4]      
BWD_ST_LIST = [3]
BWD_WP_LIST = [2]  

@triton.jit
def combine_fn(M_a, I_bar_a, F_a, M_b, I_bar_b, F_b):
    # Standard parallel scan combination
    M_new = M_b + F_b * M_a
    I_bar_new = I_bar_b + F_b * I_bar_a
    F_new = F_a * F_b
    return M_new, I_bar_new, F_new

# ----------------------------------------------------------------------------
# Forward Kernel
# ----------------------------------------------------------------------------
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'STORE_FINAL_STATES': lambda args: args['final_mu_state'] is not None,
    'USE_INITIAL_STATES': lambda args: args['initial_mu_state'] is not None,
    'OUTPUT_UNCERTAINTY': lambda args: args['o_var']is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': bv, 'BK': bk}, num_stages=st, num_warps=wp)
        for bv in FWD_BV_LIST
        for bk in FWD_BK_LIST
        for st in FWD_ST_LIST
        for wp in FWD_WP_LIST
    ],
    key=['D_K', 'D_V'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def chunk_palimpsa_fwd_kernel(
    q, k, v, b, gt, g, Ip, scale,
    o, o_var,
    initial_mu_state, initial_I_state,
    intermediate_mu_state, intermediate_I_state,
    final_mu_state, final_I_state,
    # New Arguments for Varlen
    cu_seqlens, chunk_offsets,
    # Strides
    s_int_l, s_int_h, s_int_v, s_int_k,
    s_final_b, s_final_h,
    # FIX: Add strides for initial states to be safe (reusing final strides is risky if layout differs)
    s_init_b, s_init_h, 
    s_qk_b, s_qk_l, s_qk_h, s_qk_d,
    s_vo_b, s_vo_l, s_vo_h, s_vo_d,
    s_gt_b, s_gt_l, s_gt_h,
    L, H,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    D_K: tl.constexpr, D_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATES: tl.constexpr,
    STORE_FINAL_STATES: tl.constexpr,
    OUTPUT_UNCERTAINTY: tl.constexpr
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    # -----------------------------------------------------------
    # Varlen & Pointer Setup
    # -----------------------------------------------------------
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        T = eos - bos
        start_chunk_idx = tl.load(chunk_offsets + i_b).to(tl.int32)
        
        q_ptr = q + bos * s_qk_l + i_h * s_qk_h
        k_ptr = k + bos * s_qk_l + i_h * s_qk_h
        v_ptr = v + bos * s_vo_l + i_h * s_vo_h
        b_ptr = b + bos * s_vo_l + i_h * s_vo_h
        gt_ptr = gt + bos * s_gt_l + i_h * s_gt_h
        o_ptr = o + bos * s_vo_l + i_h * s_vo_h
        o_var_offset = bos * s_vo_l + i_h * s_vo_h
    else:
        T = L
        # Standard strided logic for fixed length
        start_chunk_idx = i_b * tl.cdiv(L, BT) 
        
        q_ptr = q + i_b * s_qk_b + i_h * s_qk_h
        k_ptr = k + i_b * s_qk_b + i_h * s_qk_h
        v_ptr = v + i_b * s_vo_b + i_h * s_vo_h
        b_ptr = b + i_b * s_vo_b + i_h * s_vo_h
        gt_ptr = gt + i_b * s_gt_b + i_h * s_gt_h
        o_ptr = o + i_b * s_vo_b + i_h * s_vo_h
        o_var_offset = i_b * s_vo_b + i_h * s_vo_h

    num_seq_blocks = tl.cdiv(T, BT)
    NK = tl.cdiv(D_K, BK)

    b_Ip = tl.load(Ip + i_h).to(tl.float32)
    b_g = tl.load(g + i_h).to(tl.float32)

    for nk in range(NK):
        if USE_INITIAL_STATES:
            # FIX: Use s_init strides
            p_initial_mu = tl.make_block_ptr(initial_mu_state + i_b*s_init_b + i_h*s_init_h, (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1,0))
            p_initial_I = tl.make_block_ptr(initial_I_state + i_b*s_init_b + i_h*s_init_h, (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1,0))
            
            # FIX: Cannot use curr_mu in other=... because it's not defined yet. 
            # FIX: Cannot use p_final_mu.dtype because p_final_mu is not defined yet.
            curr_mu = tl.load(p_initial_mu, boundary_check=(0,1)).to(tl.float32)
            curr_I = tl.load(p_initial_I, boundary_check=(0,1)).to(tl.float32)
        else:
            curr_mu = tl.zeros([BV, BK], dtype=tl.float32)
            curr_I = tl.full([BV, BK], b_Ip, dtype=tl.float32)
        
        for seq_blk in range(num_seq_blocks):

            p_q = tl.make_block_ptr(q_ptr, (T, D_K), (s_qk_l, s_qk_d), (seq_blk*BT, nk*BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_ptr, (T, D_K), (s_qk_l, s_qk_d), (seq_blk*BT, nk*BK), (BT, BK), (1, 0))
            p_v = tl.make_block_ptr(v_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1, 0))
            p_b = tl.make_block_ptr(b_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1, 0))
            
            p_gt = tl.make_block_ptr(gt_ptr, (T,), (s_gt_l,), (seq_blk*BT,), (BT,), (0,))
            
            b_q_val = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_k_val = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_v_val = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
            b_b_val = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
            b_gt_raw = tl.load(p_gt, boundary_check=(0,)).to(tl.float32)

            # -----------------------------------------------------------
            # The GT Shift Strategy (Safe & Robust)
            # -----------------------------------------------------------
            p_shift = tl.make_block_ptr(gt_ptr, (T,), (s_gt_l,), (seq_blk*BT + 1,), (BT,), (0,))
            b_gt_shifted = tl.load(p_shift, boundary_check=(0,)).to(tl.float32)
            
            # Mask 
            valid = tl.arange(0, BT) < BT - 1
            b_gt_shifted = tl.where(valid, b_gt_shifted, 0.0)
            
            b_decay = -b_gt_raw * b_g 
            b_decay_shift = -b_gt_shifted * b_g 
            f_t = tl.exp(b_decay)
            f_input = tl.broadcast_to(f_t[:, None, None], (BT, BV, BK))
            b_a_shifted = tl.exp(tl.cumsum(b_decay_shift, axis=0, reverse=True))
            prod_F = tl.exp(tl.sum(b_decay, axis=0))

            input_M = (b_k_val[:, None, :] * b_v_val[:, :, None])
            input_I = ((b_k_val * b_k_val)[:, None, :] * b_b_val[:, :, None])

            abs_chunk_idx = start_chunk_idx + seq_blk
            p_int_mu_ptr = tl.make_block_ptr(
                intermediate_mu_state + abs_chunk_idx * s_int_l + i_h * s_int_h,
                (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1,0)
            )
            p_int_I_ptr = tl.make_block_ptr(
                intermediate_I_state + abs_chunk_idx * s_int_l + i_h * s_int_h,
                (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1,0)
            )
            
            tl.store(p_int_mu_ptr, curr_mu.to(p_int_mu_ptr.dtype.element_ty), boundary_check=(0,1))
            tl.store(p_int_I_ptr, curr_I.to(p_int_I_ptr.dtype.element_ty), boundary_check=(0,1))

            M_scan, I_bar_scan, F_scan = tl.associative_scan((input_M, input_I, f_input), axis=0, combine_fn=combine_fn)

            M_full = M_scan + F_scan * (curr_mu * curr_I)[None, :, :]
            I_bar_full = I_bar_scan + F_scan * (curr_I - b_Ip)[None, :, :]
            I_full = I_bar_full + b_Ip
            Mu = M_full / I_full

            contribution = tl.sum(Mu * b_q_val[:, None, :], axis=2) * scale
            
            p_o = tl.make_block_ptr(o_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1,0))
            curr_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_o, (curr_o + contribution).to(p_o.dtype.element_ty), boundary_check=(0, 1))

            if OUTPUT_UNCERTAINTY:
                contribution_var = tl.sum( b_q_val[:, None, :] * b_q_val[:, None, :] / I_full , axis=2) * scale
                p_o_var = tl.make_block_ptr(o_var + o_var_offset , (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1,0))
                curr_o_var = tl.load(p_o_var, boundary_check=(0, 1)).to(tl.float32)
                tl.store(p_o_var, (curr_o_var + contribution_var).to(p_o.dtype.element_ty), boundary_check=(0, 1))

            new_I = tl.sum(input_I * b_a_shifted[:, None, None], axis=0) + (1 - prod_F) * b_Ip + prod_F * curr_I
            new_mu_num = prod_F * curr_I * curr_mu + tl.sum(input_M * b_a_shifted[:, None, None], axis=0)
            curr_mu = new_mu_num / new_I 
            curr_I = new_I

        if STORE_FINAL_STATES:
            p_final_mu = tl.make_block_ptr(final_mu_state + i_b*s_final_b + i_h*s_final_h, (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1,0))
            p_final_I = tl.make_block_ptr(final_I_state + i_b*s_final_b + i_h*s_final_h, (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1,0))
            tl.store(p_final_mu, curr_mu.to(p_final_mu.dtype.element_ty), boundary_check=(0,1))
            tl.store(p_final_I, curr_I.to(p_final_I.dtype.element_ty), boundary_check=(0,1))

# ----------------------------------------------------------------------------
# Backward Kernel (No changes needed for logic, just kept consistent)
# ----------------------------------------------------------------------------
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'OUTPUT_UNCERTAINTY': lambda args: args['do_var'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': bv, 'BK': bk}, num_stages=st, num_warps=wp)
        for bv in FWD_BV_LIST
        for bk in FWD_BK_LIST
        for st in FWD_ST_LIST
        for wp in FWD_WP_LIST
    ],
    key=['D_K', 'D_V'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def chunk_palimpsa_bwd_kernel(
    do, do_var, q, k, v, b, gt, g, Ip,
    intermediate_mu_state, intermediate_I_state,
    dq, dk, dv, db, dgt, dg, dIp,
    scale,
    cu_seqlens, chunk_offsets,
    s_qk_b, s_qk_l, s_qk_h, s_qk_d,
    s_vo_b, s_vo_l, s_vo_h, s_vo_d,
    s_gt_b, s_gt_l, s_gt_h,
    s_int_l, s_int_h, s_int_v, s_int_k,
    s_dk_v, s_dk_b, s_dk_l, s_dk_h, s_dk_d,
    s_dgt_v, s_dgt_b, s_dgt_l, s_dgt_h,
    s_dg_v, s_dg_b, s_dg_h,
    L, H, EPS,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    D_K: tl.constexpr, D_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    OUTPUT_UNCERTAINTY: tl.constexpr
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        T = eos - bos
        start_chunk_idx = tl.load(chunk_offsets + i_b).to(tl.int32)
        
        q_ptr = q + bos * s_qk_l + i_h * s_qk_h
        k_ptr = k + bos * s_qk_l + i_h * s_qk_h
        v_ptr = v + bos * s_vo_l + i_h * s_vo_h
        b_ptr = b + bos * s_vo_l + i_h * s_vo_h
        do_ptr = do + bos * s_vo_l + i_h * s_vo_h
        do_var_offset =  bos * s_vo_l + i_h * s_vo_h
        gt_ptr = gt + bos * s_gt_l + i_h * s_gt_h
        
        dv_ptr = dv + bos * s_vo_l + i_h * s_vo_h
        db_ptr = db + bos * s_vo_l + i_h * s_vo_h
        
        dq_base = dq + i_v * s_dk_v + bos * s_dk_l + i_h * s_dk_h
        dk_base = dk + i_v * s_dk_v + bos * s_dk_l + i_h * s_dk_h
        dgt_base = dgt + i_v * s_dgt_v + bos * s_dgt_l + i_h * s_dgt_h
    else:
        T = L
        start_chunk_idx = i_b * tl.cdiv(L, BT)
        
        q_ptr = q + i_b * s_qk_b + i_h * s_qk_h
        k_ptr = k + i_b * s_qk_b + i_h * s_qk_h
        v_ptr = v + i_b * s_vo_b + i_h * s_vo_h
        b_ptr = b + i_b * s_vo_b + i_h * s_vo_h
        do_ptr = do + i_b * s_vo_b + i_h * s_vo_h
        do_var_offset = i_b * s_vo_b + i_h * s_vo_h
        gt_ptr = gt + i_b * s_gt_b + i_h * s_gt_h
        
        dv_ptr = dv + i_b * s_vo_b + i_h * s_vo_h
        db_ptr = db + i_b * s_vo_b + i_h * s_vo_h
        
        dq_base = dq + i_v * s_dk_v + i_b * s_dk_b + i_h * s_dk_h
        dk_base = dk + i_v * s_dk_v + i_b * s_dk_b + i_h * s_dk_h
        dgt_base = dgt + i_v * s_dgt_v + i_b * s_dgt_b + i_h * s_dgt_h

    num_seq_blocks = tl.cdiv(T, BT)
    NK = tl.cdiv(D_K, BK)
    b_Ip = tl.load(Ip + i_h).to(tl.float32)
    b_g = tl.load(g + i_h).to(tl.float32)
    
    total_dg = 0.0
    total_dIp = 0.0

    for nk in range(NK):
        curr_dM = tl.zeros([BV, BK], dtype=tl.float32)
        curr_dI = tl.zeros([BV, BK], dtype=tl.float32)
        
        for seq_blk in range(num_seq_blocks - 1, -1, -1):
            
            p_q = tl.make_block_ptr(q_ptr, (T, D_K), (s_qk_l, s_qk_d), (seq_blk*BT, nk*BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_ptr, (T, D_K), (s_qk_l, s_qk_d), (seq_blk*BT, nk*BK), (BT, BK), (1, 0))
            p_v = tl.make_block_ptr(v_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1, 0))
            p_b = tl.make_block_ptr(b_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1, 0))
            p_gt = tl.make_block_ptr(gt_ptr, (T,), (s_gt_l,), (seq_blk*BT,), (BT,), (0,))

            b_q_val = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_k_val = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_v_val = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
            b_b_val = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
            b_do_val = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
            b_gt_raw = tl.load(p_gt, boundary_check=(0,)).to(tl.float32)
            
            p_shift = tl.make_block_ptr(gt_ptr, (T,), (s_gt_l,), (seq_blk*BT + 1,), (BT,), (0,))
            b_gt_shift_raw = tl.load(p_shift, boundary_check=(0,)).to(tl.float32)
            
            valid = tl.arange(0, BT) < BT - 1
            b_gt_shift_raw = tl.where(valid, b_gt_shift_raw, 0.0)

            b_decay = -b_gt_raw * b_g
            b_decay_shift = -b_gt_shift_raw * b_g
            f_t = tl.exp(b_decay)
            f_input = tl.broadcast_to(f_t[:, None, None], (BT, BV, BK))
            f_input_shift = tl.broadcast_to(tl.exp(b_decay_shift)[:, None, None], (BT, BV, BK))
            prod_F = tl.exp(tl.sum(b_decay, axis=0))
            
            abs_chunk_idx = start_chunk_idx + seq_blk
            
            p_int_mu = tl.make_block_ptr(intermediate_mu_state + abs_chunk_idx * s_int_l + i_h * s_int_h,
                                     (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1, 0))
            p_int_I = tl.make_block_ptr(intermediate_I_state + abs_chunk_idx * s_int_l + i_h * s_int_h,
                                    (D_V, D_K), (s_int_v, s_int_k), (i_v*BV, nk*BK), (BV, BK), (1, 0))
            b_mu_prev = tl.load(p_int_mu, boundary_check=(0,1))
            b_I_prev = tl.load(p_int_I, boundary_check=(0,1))
            
            input_M = (b_k_val[:, None, :] * b_v_val[:, :, None])
            input_I = ((b_k_val * b_k_val)[:, None, :] * b_b_val[:, :, None])
            
            M_scan, I_bar_scan, F_scan = tl.associative_scan((input_M, input_I, f_input), axis=0, combine_fn=combine_fn)
            
            M_full = M_scan + F_scan * (b_mu_prev * b_I_prev)[None, :, :]
            I_bar_full = I_bar_scan + F_scan * (b_I_prev - b_Ip)[None, :, :]
            I_full = I_bar_full + b_Ip
            Mu = M_full / I_full

            dmu = b_do_val[:, :, None] * b_q_val[:, None, :] * scale
            dM_local = dmu / I_full
            dI_local = -dmu * M_full / (I_full * I_full)

            grad_q = tl.sum(b_do_val[:, :, None] * Mu, axis=1) * scale

            if OUTPUT_UNCERTAINTY:
                p_do_var = tl.make_block_ptr(do_var + do_var_offset, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1, 0))
                b_do_var_val = tl.load(p_do_var, boundary_check=(0, 1)).to(tl.float32)
                grad_q += 2 * b_q_val * tl.sum(b_do_var_val[:, :, None] / I_full , axis=1) * scale
                dvar = b_do_var_val[:, :, None] * b_q_val[:, None, :] * b_q_val[:, None, :] * scale
                dI_local += -dvar / (I_full * I_full)

            
            p_dq_split = tl.make_block_ptr(dq_base, 
                                           (T, D_K), (s_dk_l, s_dk_d),
                                           (seq_blk*BT, nk*BK), (BT, BK), (1, 0))
            tl.store(p_dq_split, grad_q.to(p_dq_split.dtype.element_ty), boundary_check=(0,1))

            
            dM_rev, dI_rev, F_rev = tl.associative_scan(
                (dM_local, dI_local, f_input_shift), axis=0, combine_fn=combine_fn, reverse=True
            )
            dinput_M = dM_rev + curr_dM * F_rev
            dinput_I = dI_rev + curr_dI * F_rev
            
            grad_k = tl.sum(dinput_M * b_v_val[:, :, None], axis=1) + \
                     2 * tl.sum(dinput_I * b_b_val[:, :, None], axis=1) * b_k_val
            
            p_dk_split = tl.make_block_ptr(dk_base,
                                           (T, D_K), (s_dk_l, s_dk_d),
                                           (seq_blk*BT, nk*BK), (BT, BK), (1, 0))
            tl.store(p_dk_split, grad_k.to(p_dk_split.dtype.element_ty), boundary_check=(0,1))

            grad_v_part = tl.sum(dinput_M * b_k_val[:, None, :], axis=2)
            grad_b_part = tl.sum(dinput_I * (b_k_val*b_k_val)[:, None, :], axis=2)
            
            p_dv = tl.make_block_ptr(dv_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1,0))
            p_db = tl.make_block_ptr(db_ptr, (T, D_V), (s_vo_l, s_vo_d), (seq_blk*BT, i_v*BV), (BT, BV), (1,0))
            
            if nk == 0:
                tl.store(p_dv, grad_v_part.to(p_dv.dtype.element_ty), boundary_check=(0,1))
                tl.store(p_db, grad_b_part.to(p_db.dtype.element_ty), boundary_check=(0,1))
            else:
                curr_dv = tl.load(p_dv, boundary_check=(0,1)).to(tl.float32)
                curr_db = tl.load(p_db, boundary_check=(0,1)).to(tl.float32)
                tl.store(p_dv, (curr_dv + grad_v_part).to(p_dv.dtype.element_ty), boundary_check=(0,1))
                tl.store(p_db, (curr_db + grad_b_part).to(p_db.dtype.element_ty), boundary_check=(0,1))

            M_shift = (M_full - input_M)/(f_input + EPS)
            I_shift = (I_full - input_I - (1-f_input)*b_Ip)/(f_input + EPS)
            grad_a = tl.sum(tl.sum(M_shift * dinput_M, axis=2), axis=1) + \
                     tl.sum(tl.sum((I_shift - b_Ip) * dinput_I, axis=2), axis=1)
            
            grad_decay_term = grad_a * f_t
            grad_gt_part = -grad_decay_term * b_g
            
            total_dg += tl.sum(grad_decay_term * -b_gt_raw)
            total_dIp += tl.sum(dI_local)
            
            # Safe shape (T,)
            p_dgt = tl.make_block_ptr(dgt_base,
                                      (T,), (s_dgt_l,), (seq_blk*BT,), (BT,), (0,))
            if nk == 0:
                 tl.store(p_dgt, grad_gt_part.to(p_dgt.dtype.element_ty), boundary_check=(0,))
            else:
                 curr_dgt = tl.load(p_dgt, boundary_check=(0,)).to(tl.float32)
                 tl.store(p_dgt, (curr_dgt + grad_gt_part).to(p_dgt.dtype.element_ty), boundary_check=(0,))

            curr_dM = tl.sum(dM_local * F_scan, axis=0) + curr_dM * prod_F
            curr_dI = tl.sum(dI_local * F_scan, axis=0) + curr_dI * prod_F

    offset = i_v*s_dg_v + i_b*s_dg_b + i_h*s_dg_h
    tl.store(dg + offset, total_dg.to(dg.dtype.element_ty))
    tl.store(dIp + offset, total_dIp.to(dIp.dtype.element_ty))


class ChunkPalimpsa(torch.autograd.Function):
    @staticmethod
    @contiguous
    @torch.autocast(device_type="cuda")
    def forward(ctx, q, k, v, b, gt, g, Ip, initial_mu_state, initial_I_state, scale, chunk_size, output_final_state, output_uncertainty, cu_seqlens, chunk_offsets, int_mu, int_I):
        # 1. Setup Strides
        if cu_seqlens is not None:
            batch_size = len(cu_seqlens) - 1
            L = q.shape[1]
            s_qk_b, s_qk_l, s_qk_h, s_qk_d = 0, q.stride(1), q.stride(2), q.stride(3)
            s_vo_b, s_vo_l, s_vo_h, s_vo_d = 0, v.stride(1), v.stride(2), v.stride(3)
            s_gt_b, s_gt_l, s_gt_h = 0, gt.stride(1), gt.stride(2)
        else:
            batch_size, L, H, D_K = q.shape
            # For fixed mode, ensure gt has valid strides (padding handled implicitly by shape logic if needed)
            s_qk_b, s_qk_l, s_qk_h, s_qk_d = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
            s_vo_b, s_vo_l, s_vo_h, s_vo_d = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
            s_gt_b, s_gt_l, s_gt_h = gt.stride(0), gt.stride(1), gt.stride(2)

        H, D_K = q.shape[-2], q.shape[-1]
        D_V = v.shape[-1]
        
        ctx.original_dtype = q.dtype
        ctx.scale = scale
        ctx.chunk_size = chunk_size 
        ctx.output_uncertainty = output_uncertainty

        gt, g, Ip = [x.float() for x in [gt, g, Ip]]
        
 
        o = torch.zeros_like(v, dtype=torch.float32)
        # o_var: should I allocate it all the time??
        if output_uncertainty:
            o_var = torch.zeros_like(v, dtype=torch.float32)
        else:
            o_var = None # Passed as None to kernel (needs care in kernel not to dereference)
            
        # Final states
        final_mu = q.new_zeros(batch_size, H, D_V, D_K, dtype=torch.float32)
        final_I = q.new_zeros(batch_size, H, D_V, D_K, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(D_V, META['BV']), batch_size * H)
        
        # Pass strides for flattened intermediate buffers
        s_int_l = int_mu.stride(0)
        s_int_h = int_mu.stride(1)
        s_int_v = int_mu.stride(2)
        s_int_k = int_mu.stride(3)
        
        # FIX: Handle strides for initial state safely
        if initial_mu_state is not None:
             s_init_b = initial_mu_state.stride(0)
             s_init_h = initial_mu_state.stride(1)
        else:
             # Dummies
             s_init_b, s_init_h = 0, 0

        chunk_palimpsa_fwd_kernel[grid](
            q, k, v, b, gt, g, Ip, scale, o, o_var,
            # FIX: MUST PASS INITIAL STATES HERE
            initial_mu_state, initial_I_state,
            int_mu, int_I, 
            final_mu, final_I,
            cu_seqlens, chunk_offsets,
            s_int_l, s_int_h, s_int_v, s_int_k,
            final_mu.stride(0), final_mu.stride(1),
            s_init_b, s_init_h,
            s_qk_b, s_qk_l, s_qk_h, s_qk_d,
            s_vo_b, s_vo_l, s_vo_h, s_vo_d,
            s_gt_b, s_gt_l, s_gt_h,
            L, H,
            BT=chunk_size, D_K=D_K, D_V=D_V,
        )

        # Save everything, including the buffers passed in
        ctx.save_for_backward(q, k, v, b, gt, g, Ip, int_mu, int_I, cu_seqlens, chunk_offsets)
        
        o = o.to(ctx.original_dtype)
        o_var = o_var.to(ctx.original_dtype) if output_uncertainty else None
        
        return o, o_var, final_mu, final_I

    @staticmethod
    @contiguous
    @torch.autocast(device_type="cuda")
    def backward(ctx, do, do_var=None, d_mu=None, d_I=None):
        q, k, v, b, gt, g, Ip, int_mu, int_I, cu_seqlens, chunk_offsets = ctx.saved_tensors
        scale = ctx.scale
        output_uncertainty = ctx.output_uncertainty
        if ctx.output_uncertainty == False:
            do_var = None

        if cu_seqlens is not None:
             batch_size = len(cu_seqlens) - 1
             L = q.shape[1]
             H, D_K = q.shape[2], q.shape[3]
             s_qk_b, s_qk_l, s_qk_h, s_qk_d = 0, q.stride(1), q.stride(2), q.stride(3)
             s_vo_b, s_vo_l, s_vo_h, s_vo_d = 0, v.stride(1), v.stride(2), v.stride(3)
             s_gt_b, s_gt_l, s_gt_h = 0, gt.stride(1), gt.stride(2)
        else:
             batch_size, L, H, D_K = q.shape
             s_qk_b, s_qk_l, s_qk_h, s_qk_d = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
             s_vo_b, s_vo_l, s_vo_h, s_vo_d = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
             s_gt_b, s_gt_l, s_gt_h = gt.stride(0), gt.stride(1), gt.stride(2)

        D_V = v.shape[-1]
        EPS = 1e-12
        
        min_bv = min(BWD_BV_LIST)
        NV_MAX = triton.cdiv(D_V, min_bv)
        
        dq_split = torch.zeros(NV_MAX, *q.shape, dtype=torch.float32, device=q.device)
        dk_split = torch.zeros(NV_MAX, *q.shape, dtype=torch.float32, device=q.device)
        dgt_split = torch.zeros(NV_MAX, *gt.shape, dtype=torch.float32, device=q.device)
        
        dg_split = torch.zeros(NV_MAX, batch_size, H, dtype=torch.float32, device=q.device)
        dIp_split = torch.zeros(NV_MAX, batch_size, H, dtype=torch.float32, device=q.device)
        
        dv = torch.zeros_like(v, dtype=torch.float32)
        db = torch.zeros_like(b, dtype=torch.float32)
        
        grid = lambda META: (triton.cdiv(D_V, META['BV']), batch_size * H)
        
        # Strides for splits
        if cu_seqlens is not None:
             s_dk_v = dq_split.stride(0)
             s_dk_b = 0
             s_dk_l = dq_split.stride(2) 
             s_dk_h = dq_split.stride(3)
             s_dk_d = dq_split.stride(4)

             s_dgt_v = dgt_split.stride(0)
             s_dgt_b = 0
             s_dgt_l = dgt_split.stride(2)
             s_dgt_h = dgt_split.stride(3)

             s_dg_v = dg_split.stride(0)
             s_dg_b = dg_split.stride(1)
             s_dg_h = dg_split.stride(2)
        else:
             s_dk_v = dq_split.stride(0)
             s_dk_b = dq_split.stride(1)
             s_dk_l = dq_split.stride(2)
             s_dk_h = dq_split.stride(3)
             s_dk_d = dq_split.stride(4)
             
             s_dgt_v = dgt_split.stride(0)
             s_dgt_b = dgt_split.stride(1)
             s_dgt_l = dgt_split.stride(2)
             s_dgt_h = dgt_split.stride(3)

             s_dg_v = dg_split.stride(0)
             s_dg_b = dg_split.stride(1)
             s_dg_h = dg_split.stride(2)

        chunk_palimpsa_bwd_kernel[grid](
            do, do_var, q, k, v, b, gt, g, Ip,
            int_mu, int_I,
            dq_split, dk_split, dv, db, dgt_split, dg_split, dIp_split,
            scale,
            cu_seqlens, chunk_offsets,
            s_qk_b, s_qk_l, s_qk_h, s_qk_d,
            s_vo_b, s_vo_l, s_vo_h, s_vo_d,
            s_gt_b, s_gt_l, s_gt_h,
            int_mu.stride(0), int_mu.stride(1), int_mu.stride(2), int_mu.stride(3),
            s_dk_v, s_dk_b, s_dk_l, s_dk_h, s_dk_d,
            s_dgt_v, s_dgt_b, s_dgt_l, s_dgt_h,
            s_dg_v, s_dg_b, s_dg_h,
            L, H, EPS,
            BT=ctx.chunk_size, D_K=D_K, D_V=D_V
        )
        
        dq = dq_split.sum(dim=0)
        dk = dk_split.sum(dim=0)
        dgt = dgt_split.sum(dim=0)
        dg = dg_split.sum(dim=0).sum(dim=0)
        dIp = dIp_split.sum(dim=0).sum(dim=0)
        
        # FIX: The Forward function took 17 arguments. The Backward must return 17 gradients (or Nones).
        # You returned 16 items. The missing item causes an autograd crash.
        return dq.to(ctx.original_dtype), dk.to(ctx.original_dtype), \
               dv.to(ctx.original_dtype), db.to(ctx.original_dtype), \
               dgt.to(ctx.original_dtype), dg.to(ctx.original_dtype), \
               dIp.to(ctx.original_dtype), None, None, None, None, None, None, None, None, None, None
    

def chunk_palimpsa(q, k, v, b, gt, g, Ip, initial_mu_state=None, initial_I_state=None, scale=None, chunk_size=16, output_final_state=False, output_uncertainty=False, cu_seqlens=None):
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # 1. Calculate shapes and offsets using Metadata (Compile-Safe)
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"Batch size must be 1 for Varlen, got {q.shape[0]}")
        
        # We need offsets for the kernel logic
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        
        # KEY FIX: Use prepare_chunk_indices to get length. 
        # This is a metadata operation in Dynamo, so NO Graph Break (unlike .item()).
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        total_chunks = len(chunk_indices)
        
        cu_seqlens = cu_seqlens.to(torch.int32)
        chunk_offsets = chunk_offsets.to(torch.int32)
    else:
        batch_size, L = q.shape[0], q.shape[1]
        # triton.cdiv works on symbolic shapes, so this is also compile-safe
        total_chunks = triton.cdiv(L, chunk_size) * batch_size
        chunk_offsets = None

    # 2. Pre-allocate intermediate buffers
    H, D_K, D_V = q.shape[-2], q.shape[-1], v.shape[-1]
    
    # Allocation happens outside the Autograd function
    int_mu = q.new_zeros(total_chunks, H, D_V, D_K, dtype=torch.float32)
    int_I = q.new_zeros(total_chunks, H, D_V, D_K, dtype=torch.float32)

    o, o_var, final_mu_state, final_I_state = ChunkPalimpsa.apply(
        q, k, v, b, gt, g, Ip, initial_mu_state, initial_I_state,
        scale, chunk_size, output_final_state, output_uncertainty,
        cu_seqlens, chunk_offsets, int_mu, int_I
    )
    if output_uncertainty:
        out = (o, o_var)
    else:
        out = o

    if output_final_state:
        return out, final_mu_state, final_I_state
    return out

def palimpsa_ref(q, k, v, b, gt, g, Ip, initial_mu_state=None, initial_I_state=None, scale=None, output_final_state=False, output_uncertainty=False):
    """
    To Rui:
        we will always use real numbers, so ignore all complex logic in mamba cuda code.
    """

    """
    v:  r(B L H D)
    q:  r(B L H N)
    k:  r(B L H N)
    b: r(B L H D), also in [0, 1]
ok 
    out: r(B D H L)
    last_state (optional): r(B H D dstate) or c(B H D dstate)
    """
    dtype_in = v.dtype
    v = v.float()
    q = q.float()
    k = k.float()
    b = b.float()
    gt = gt.float()
    g = g.float()
    Ip = Ip.float()

    if scale is None:
        scale = q.shape[-1] ** -0.5

   
    B,L,H,D = b.shape
    DK = k.shape[-1]

    y = torch.zeros(B,L,H,D)
    y_var = torch.zeros(B,L,H,D)
    if initial_mu_state == None: 
        Mu_pre = torch.zeros(B,H,D,DK)
        I_pre = Ip[None,:,None,None]
    else:
        Mu_pre = initial_mu_state
        I_pre = initial_I_state

    q = q * scale
    for l in range(L):
    # Cumulative product on reversed
        M = v[:,l,:,:,None]*k[:,l,:,None,:]  + torch.exp(-gt[:,l,:,None,None]*g[None,:,None,None])*Mu_pre*I_pre
        I = b[:,l,:,:,None]*(k[:,l,:,None,:]**2) + (1-torch.exp(-gt[:,l,:,None,None]*g[None,:,None,None]))*Ip[None,:,None,None] + torch.exp(-gt[:,l,:,None,None]*g[None,:,None,None])*I_pre 
        mu = M/I
        Mu_pre = mu
        I_pre = I
        y[:,l] = torch.einsum('bhdn,bhn->bhd', mu,q[:,l])
        y_var[:,l] = torch.einsum('bhdn,bhn->bhd', 1/I,q[:,l]**2)
    final_mu_state = mu
    final_I_state = I
    if output_uncertainty:
        out = (y.to(dtype=dtype_in), y_var.to(dtype=dtype_in))
    else:
        out = y.to(dtype=dtype_in)
    return  out if not output_final_state else (out, final_mu_state, final_I_state)
