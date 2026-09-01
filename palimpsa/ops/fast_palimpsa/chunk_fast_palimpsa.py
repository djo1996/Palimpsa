# -*- coding: utf-8 -*-
# chunk_fast_palimpsa.py
#
# Fast Palimpsa: chunked isotropic-in-D_V approximation of Palimpsa.
#   * State carried across chunks is EXACT, full D_V x D_K.
#   * Inside a chunk, the LOCAL contribution is read with an isotropic precision
#     Ibar_t in R^{D_K} (collapsed over D_V), evolved with full dynamics
#     (alpha decay + (1-f) Ip prior + betabar k^2).
#   * The CARRY contribution (history) is read against the FROZEN boundary state
#     mu_c = M_c / I_c (tl.dot-friendly, the whole point: no per-token DVxDK read).
#
# This file contains:
#   1. fast_palimpsa_ref      -- differentiable PyTorch reference (the contract).
#   2. chunk_fast_palimpsa     -- Triton autograd Function (tiled, any D_K/D_V).
#   3. test_fwd_bwd() / test_ref_shapes() / test_varlen() -- correctness gate,
#      comparing Triton fwd+bwd against the reference above.
#
# Tiling follows fla chunk_h / chunk_gla: state h=[K,V] tiled by (BK,BV) on a
# 3D grid (cdiv(K,BK), cdiv(V,BV), B*H); chunk loop is inside the kernel; reductions
# over D_K are accumulated across BK blocks. Any D_K, D_V are supported.
#
# Memory design (see _fp_compute_state and _FastPalimpsa.forward): the chunk-
# boundary M/I trajectory saved for backward is checkpointed in bf16 where it's
# only ever read through a matmul (M, matching fla's chunk_h/GLA/mesa_net
# states), fp32 where it's read through a division (I, more precision-
# sensitive -- verified empirically, not assumed). fla's usual complementary
# move for this class of kernel -- recomputing the trajectory in backward
# instead of saving it across the autograd boundary -- was tried and measured
# 22-40% *slower* here despite less peak memory (this kernel's per-chunk state
# update is apparently costlier relative to its own backward than in the
# kernels that pattern is standard for), so it stays saved.

from __future__ import annotations
import functools
from functools import lru_cache
import torch


def contiguous(fn):
    # Belt-and-suspenders: _FastPalimpsa.forward/backward already do their own
    # inline .contiguous()/.float() on every tensor they touch, so this is a
    # no-op in the common case -- but it guards the Triton kernels (which index
    # raw pointers with hand-computed strides, not torch's own stride logic)
    # against a non-contiguous input reaching them by some path that doesn't.
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except Exception:
    HAS_TRITON = False

if HAS_TRITON:
    try:
        # fla wraps tl.dot on SM100/SM120 (Blackwell) to work around a Triton
        # compiler bug (TritonGPUHoistTMEMAlloc incorrectly fusing an add into a
        # following dot, triton-lang/triton#8695) that silently miscompiles the
        # accumulator pattern `acc += tl.dot(...)` used throughout this file.
        from fla.ops.utils.op import safe_dot
    except ImportError:
        @triton.jit
        def safe_dot(a, b, allow_tf32: tl.constexpr = None):
            return tl.dot(a, b, allow_tf32=allow_tf32)

    @triton.jit
    def _sdot(a, b):
        # No call site here passes fp32 operands through tf32 tensor cores:
        # every dot in this file is over quantities (states, decay-weighted
        # sums) fed by upstream divisions, so silent tf32 rounding is worth
        # ruling out deliberately rather than relying on tl.dot's default.
        return safe_dot(a, b, allow_tf32=False)

    try:
        from fla.utils._compat import autotune_cache_kwargs
    except ImportError:
        autotune_cache_kwargs = {}

    try:
        from fla.utils import check_shared_mem
    except ImportError:
        def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
            return False

    try:
        # Opt-in hardware fast-math (tldevice.fast_expf) behind FLA_USE_FAST_OPS=1;
        # a plain `exp(x.to(tl.float32))` wrapper otherwise -- i.e. byte-for-byte
        # what every `exp(...)` call in this file already does. Off by default, so
        # this import changes nothing unless the env var is set; the option just
        # didn't exist here before. No tl.log call sites exist in this file's kernels
        # to swap alongside it (the decay is computed directly in log-space via
        # `-gt_c*b_g`, never through an actual tl.log()).
        from fla.ops.utils.op import exp
    except ImportError:
        @triton.jit
        def exp(x):
            return exp(x.to(tl.float32))

    # BK=128 only where it's actually worth attempting: on Hopper+ (232 KiB/block),
    # a K-tiled kernel doing NK=cdiv(D_K,BK) iterations can go from 2 iterations to 1
    # at the 760M shape (D_K=96, next_power_of_2=128) -- on Ada/Ampere it would just
    # burn autotune compile time failing to compile or losing to BK=64, so it's not
    # offered there at all (matches the fla.ops.generalized_delta_rule.dplr idiom,
    # e.g. chunk_o_bwd.py's `BK_LIST = [32, 64, 128] if check_shared_mem() else
    # [16, 32]`). Evaluated once at import time -- fine since this whole block only
    # runs under HAS_TRITON = torch.cuda.is_available(), and this project pins one
    # GPU per job via CUDA_VISIBLE_DEVICES, so device 0 here is the real device.
    _FP_BK_CHOICES = (32, 64, 128) if check_shared_mem('hopper') else (32, 64)

_EPS = 1e-12
CHUNK_C = 32


# =============================================================================
# 1. Reference (differentiable)
# =============================================================================
def _resolve_Ip(Ip, H, DK, dev, dt):
    if not torch.is_tensor(Ip):
        Ip = torch.full((H,), float(Ip), device=dev, dtype=dt)
    Ip = Ip.to(device=dev, dtype=dt)
    if Ip.numel() == H:                       # scalar per head
        return Ip.view(1, H, 1), Ip.view(1, H, 1, 1)        # (1,H,1) key-bcast, (1,H,1,1) vk-bcast
    elif Ip.numel() == H * DK:                # per (head, key-dim)
        return Ip.view(1, H, DK), Ip.view(1, H, 1, DK)
    raise ValueError("Ip must be scalar-per-head (H,) or per-(H,DK).")


def _exact_recurrence(q, k, v, b, gt, g, Ip, scale):
    """Token-exact Palimpsa ground truth. Differentiable."""
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    dev, dt = q.device, q.dtype
    Ip_k, Ipv = _resolve_Ip(Ip, H, DK, dev, dt)
    Ipv_full = Ipv  # (1,H,1,1) or (1,H,1,DK)
    q = q * scale
    M = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I = (Ipv_full.expand(B, H, DV, DK).clone() if Ipv_full.shape[-1] == DK
         else Ipv_full.view(1, H, 1, 1).expand(B, H, DV, DK).clone())
    ys, yvars = [], []
    for t in range(L):
        f = torch.exp(-(gt[:, t] * g.view(1, H))).view(B, H, 1, 1)
        kt = k[:, t].view(B, H, 1, DK)
        vt = v[:, t].view(B, H, DV, 1)
        bt = b[:, t].view(B, H, DV, 1)
        I = f * I + (1 - f) * (Ipv_full if Ipv_full.shape[-1] == DK else Ipv_full) + bt * (kt * kt)
        M = f * M + vt * kt
        mu = M / I
        qt = q[:, t].view(B, H, 1, DK)
        ys.append((mu * qt).sum(-1))
        yvars.append((qt * qt / I).sum(-1))
    return torch.stack(ys, 1), torch.stack(yvars, 1)


def fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C,
                      output_uncertainty=False):
    """Chunked isotropic approximation (frozen carry). Differentiable. The contract."""
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert L % C == 0
    nc = L // C
    if scale is None:
        scale = DK ** -0.5
    dev, dt = q.device, q.dtype
    Ip_k, Ipv = _resolve_Ip(Ip, H, DK, dev, dt)
    perdk = Ipv.shape[-1] == DK

    qs = q * scale
    M_c = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I_c = (Ipv.expand(B, H, DV, DK).clone() if perdk
           else Ipv.view(1, H, 1, 1).expand(B, H, DV, DK).clone())

    y_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)
    yvar_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)

    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        kc, vc, bc, qc, gc = k[:, sl], v[:, sl], b[:, sl], qs[:, sl], gt[:, sl]
        f = torch.exp(-(gc * g.view(1, 1, H)))                  # (B,C,H)
        logf = torch.log(f.clamp_min(1e-30))
        clogf = torch.cumsum(logf, dim=1)                       # (B,C,H)

        Ibar_c = I_c.mean(2)                                    # (B,H,DK)
        abar_c = Ibar_c - Ip_k                                  # (B,H,DK)
        betabar = bc.mean(-1)                                   # (B,C,H)
        ksq = kc * kc

        Ibar = torch.empty(B, C, H, DK, dtype=dt, device=dev)
        a_prev = abar_c
        for t in range(C):
            ft = f[:, t].unsqueeze(-1)
            a_prev = ft * a_prev + betabar[:, t].unsqueeze(-1) * ksq[:, t]
            Ibar[:, t] = Ip_k + a_prev

        # local (isotropic), reader-side scaling Qtil = q/Ibar
        Qtil = qc / Ibar
        score = torch.einsum('bthd,bshd->btsh', Qtil, kc)       # (B,C,C,H)
        Dmask = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1))
        tri = torch.tril(torch.ones(C, C, device=dev, dtype=dt)).view(1, C, C, 1)
        score = score * Dmask * tri
        y_local = torch.einsum('btsh,bshv->bthv', score, vc)    # (B,C,H,DV)

        # carry (exact mu_c, with relative isotropic decay on Q)
        mu_c = M_c / I_c
        Q_carry_til = qc * (Ibar_c.unsqueeze(1) / Ibar)         # (B,C,H,DK)
        base = torch.einsum('bhvd,bthd->bthv', mu_c, Q_carry_til)

        carry_decay = torch.exp(clogf).unsqueeze(-1)
        y_out[:, sl] = y_local + base * carry_decay

        if output_uncertainty:
            yv = torch.einsum('bthd,bthd->bth', qc * qc, 1.0 / Ibar)
            yvar_out[:, sl] = yv.unsqueeze(-1).expand(B, C, H, DV)

        # exact boundary update
        I_new, M_new = I_c.clone(), M_c.clone()
        Ipv_full = (Ipv if perdk else Ipv.view(1, H, 1, 1))
        for t in range(C):
            ft = f[:, t].view(B, H, 1, 1)
            kt = kc[:, t].view(B, H, 1, DK)
            vt = vc[:, t].view(B, H, DV, 1)
            bt = bc[:, t].view(B, H, DV, 1)
            I_new = ft * I_new + (1 - ft) * Ipv_full + bt * (kt * kt)
            M_new = ft * M_new + vt * kt
        I_c, M_c = I_new, M_new

    if output_uncertainty:
        return y_out, yvar_out
    return y_out


def fast_palimpsa_ref_varlen(q, k, v, b, gt, g, Ip, cu_seqlens, scale=None,
                             chunk_size=CHUNK_C):
    """Varlen reference: run `fast_palimpsa_ref` independently per packed sequence.

    q/k/v/b/gt are (1, T_total, H, *) packed tensors, `cu_seqlens` (num_seqs+1,)
    int marks sequence boundaries. Each sub-sequence is zero-padded up to a
    chunk_size multiple (fast_palimpsa_ref requires L % C == 0) -- padding k/v/b
    with 0 and gt with 0 is safe: it contributes no decay (f=1) but zero
    key/value/beta, so the padded tail cannot affect any real output position
    (chunk_fast_palimpsa's carry only ever reads *earlier* real positions).
    Only the first `T_seq` output rows of each padded run are kept, so this is a
    correct-by-construction oracle for the varlen Triton kernel, independent of
    the Triton implementation's own boundary-masking logic.
    """
    assert q.shape[0] == 1, "varlen convention: packed batch dim must be 1"
    C = chunk_size
    num_seqs = cu_seqlens.numel() - 1
    outs = []
    for i in range(num_seqs):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        T_seq = eos - bos
        pad = (-T_seq) % C
        sl = slice(bos, eos)

        def _pad(x, is_gate=False):
            chunk = x[:, sl]
            if pad == 0:
                return chunk
            pad_shape = list(chunk.shape)
            pad_shape[1] = pad
            zeros = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
            return torch.cat([chunk, zeros], dim=1)

        y = fast_palimpsa_ref(_pad(q), _pad(k), _pad(v), _pad(b), _pad(gt), g, Ip,
                              scale=scale, chunk_size=C)
        outs.append(y[:, :T_seq])
    return torch.cat(outs, dim=1)


def fast_palimpsa_vec(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C):
    """Chunk-parallel, fully-vectorized implementation (no Python token loops).

    Same math as fast_palimpsa_ref, but the two intra-chunk `for t in range(C)`
    loops are replaced by closed-form cumulative-decay matmuls, so it runs at
    bmm/einsum efficiency and is differentiable by autograd. Works for ANY
    D_K / D_V with no shared-memory constraint (no full-D_V resident tile), which
    is why it sidesteps the SMEM blow-up that the hand-written Triton backward
    hits at D_V=192. Verified to match the loop reference fwd + all grads to
    machine precision (fp64).
    """
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert L % C == 0, f"L={L} not divisible by C={C}"
    nc = L // C
    if scale is None:
        scale = DK ** -0.5
    dev, dt = q.device, q.dtype
    Ip_k, Ipv = _resolve_Ip(Ip, H, DK, dev, dt)
    perdk = Ipv.shape[-1] == DK
    Ip_k = Ip_k if perdk else Ipv.view(1, H, 1).expand(1, H, DK)

    qs = q * scale
    M_c = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I_c = Ip_k.view(1, H, 1, DK).expand(B, H, DV, DK).clone()

    triCC = torch.tril(torch.ones(C, C, device=dev, dtype=dt)).view(1, C, C, 1)
    y_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)

    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        kc, vc, bc, qc, gc = k[:, sl], v[:, sl], b[:, sl], qs[:, sl], gt[:, sl]
        f = torch.exp(-(gc * g.view(1, 1, H)))                  # (B,C,H)
        logf = torch.log(f.clamp_min(1e-30))
        clogf = torch.cumsum(logf, dim=1)                       # (B,C,H)
        cd = torch.exp(clogf)

        Ibar_c = I_c.mean(2)                                    # (B,H,DK)
        abar_c = Ibar_c - Ip_k                                  # (B,H,DK)
        betabar = bc.mean(-1)                                   # (B,C,H)
        ksq = kc * kc

        # closed-form intra-chunk Ibar scan (replaces the t-loop):
        #   a_t = cd_t*abar_c + sum_{i<=t} (cd_t/cd_i)*betabar_i*ksq_i
        Dm = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1)) * triCC   # (B,C,C,H)[t,i]
        src = betabar.unsqueeze(-1) * ksq                       # (B,C,H,DK) [i]
        a_mat = cd.unsqueeze(-1) * abar_c.unsqueeze(1) + torch.einsum('btih,bihd->bthd', Dm, src)
        Ibar = Ip_k.unsqueeze(1) + a_mat                        # (B,C,H,DK)

        # local (isotropic) reader-side scaling
        Qtil = qc / Ibar
        score = torch.einsum('bthd,bshd->btsh', Qtil, kc)
        Dmask = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1))
        score = score * Dmask * triCC
        y_local = torch.einsum('btsh,bshv->bthv', score, vc)

        # exact-mu carry with relative isotropic decay on Q
        mu_c = M_c / I_c
        Q_carry_til = qc * (Ibar_c.unsqueeze(1) / Ibar)
        base = torch.einsum('bhvd,bthd->bthv', mu_c, Q_carry_til)
        y_out[:, sl] = y_local + base * cd.unsqueeze(-1)

        # closed-form exact boundary update (replaces the t-loop):
        #   M <- prodF*M + sum_t w_t v_t k_t^T ;  w_t = prod_{i>t} f_i
        prodF = torch.exp(clogf[:, -1])                         # (B,H)
        w = torch.exp(clogf[:, -1:].expand(B, C, H) - clogf)    # (B,C,H)
        pf = prodF.view(B, H, 1, 1)
        M_c = pf * M_c + torch.einsum('bthv,bthd->bhvd', w.unsqueeze(-1) * vc, kc)
        I_c = (pf * I_c + (1 - pf) * Ip_k.view(1, H, 1, DK)
               + torch.einsum('bthv,bthd->bhvd', w.unsqueeze(-1) * bc, ksq))

    return y_out


# =============================================================================
# 2. Triton kernels (tiled; any D_K, D_V)
# =============================================================================
if HAS_TRITON:

    def _fp_fwd_state_autotune():
        # Both BK and BV are grid-tiled here (triton.cdiv(D_K,BK)/triton.cdiv(D_V,BV) in
        # the launch grid below), so narrowing either is a legitimate speed/occupancy
        # tradeoff, never a correctness risk -- unlike _fp_ibar_kernel/_fp_output_kernel's
        # BV, which a per-tile D_V-mean reduction requires to cover the full D_V (see
        # their own autotune functions). num_stages capped at 2, matching every other
        # autotuned kernel in this file that has an internal `for c in range(nc)` chunk
        # loop (_fp_bwd_kernel_local_state/_fp_bwd_kernel_scan's _fp_bwd_autotune()) --
        # this kernel has that exact loop shape too.
        return [
            triton.Config({'BK': bk, 'BV': bv}, num_warps=w, num_stages=s)
            for bk in _FP_BK_CHOICES
            for bv in (16, 32, 64)
            for w in (2, 4, 8)
            for s in (1, 2)
        ]

    @triton.heuristics({
        'USE_INITIAL_STATE': lambda args: args['initial_mu_state'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    })
    @triton.autotune(configs=_fp_fwd_state_autotune(), key=['D_K', 'D_V', 'C'],
                     **autotune_cache_kwargs)
    @triton.jit(do_not_specialize=['T'])
    def _fp_state_kernel(
        k, v, b, gt, g, Ip,
        initial_mu_state, initial_I_state,  # (B,H,DV,DK), optional
        M_bound, I_bound,            # (B*H,MAX_NC+1,DV,DK) chunk-ENTRY exact states
        cu_seqlens,                  # (num_seqs+1,) int32, optional
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        MAX_NC: tl.constexpr,
        PERDK: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        # One program per (b, h, k-block, v-block). Walks chunks forward, writes
        # the EXACT chunk-entry M,I for each chunk, then advances exactly.
        # M_bound/I_bound are strided as if every sequence had MAX_NC chunks (the
        # longest one in this launch); a shorter sequence's unused tail slots
        # [nc..MAX_NC] are simply never written past `nc` and never read back by
        # any downstream kernel (each re-derives its own nc the same way).
        i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_b).to(tl.int32)
            eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
            T_seq = eos - bos
            nc = tl.cdiv(T_seq, C)
        else:
            bos = i_b * T
            T_seq = T
            nc = T // C

        o_c = tl.arange(0, C)
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < D_K
        mask_v = o_v < D_V
        mask_kv = mask_v[:, None] & mask_k[None, :]

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)

        if USE_INITIAL_STATE:
            off_init = i_bh * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            mu0 = tl.load(initial_mu_state + off_init, mask=mask_kv, other=0.0).to(tl.float32)
            I0 = tl.load(initial_I_state + off_init, mask=mask_kv, other=0.0).to(tl.float32)
            M = mu0 * I0
            I = I0
        else:
            M = tl.zeros([BV, BK], dtype=tl.float32)
            I = tl.zeros([BV, BK], dtype=tl.float32) + b_Ip[None, :]

        for c in range(nc):
            off = ((i_bh * (MAX_NC + 1) + c) * D_V * D_K
                   + o_v[:, None] * D_K + o_k[None, :])
            tl.store(M_bound + off, M.to(M_bound.dtype.element_ty), mask=mask_kv)
            tl.store(I_bound + off, I.to(I_bound.dtype.element_ty), mask=mask_kv)

            mask_c = o_c < tl.minimum(C, T_seq - c * C)
            base_qk = ((bos + c * C) * H + i_h) * D_K
            base_vo = ((bos + c * C) * H + i_h) * D_V
            base_gt = (bos + c * C) * H + i_h
            # k_ck/gt_c don't depend on o_v -- every i_v program at this (i_k, i_bh)
            # reloads the identical values (gt_c also doesn't depend on o_k, so it's
            # redundant across i_k too), unlike v_cv/b_cv which this one program is the
            # only reader of. evict_last keeps the shared ones warm in L2 for the other
            # concurrent programs; evict_first lets the one-shot ones go without
            # pushing something reusable out.
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_c[:, None] & mask_k[None, :], other=0.0,
                           eviction_policy='evict_last').to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0,
                           eviction_policy='evict_first').to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0,
                           eviction_policy='evict_first').to(tl.float32)
            gt_c = tl.load(gt + base_gt + o_c * H, mask=mask_c, other=0.0,
                           eviction_policy='evict_last').to(tl.float32)
            f_c = exp(-gt_c * b_g)
            logf = -gt_c * b_g
            clogf = tl.cumsum(logf, axis=0)
            clogf_last = tl.sum(tl.where(o_c == (C - 1), clogf, 0.0))
            prodF = exp(clogf_last)                       # prod_t f_t
            # per-token weight w_t = prod_{i>t} f_i = prodF / F_t = exp(clogf_last - clogf_t)
            w = exp(clogf_last - clogf)                   # (C,)
            ksq = k_ck * k_ck
            
            # Matmul forms replacing the sequential token loop
            M = prodF * M + _sdot(tl.trans((w[:, None] * v_cv)).to(tl.float32),
                                   k_ck.to(tl.float32))
            I = (prodF * I + (1.0 - prodF) * b_Ip[None, :]
                 + _sdot(tl.trans((w[:, None] * b_cv)).to(tl.float32),
                          ksq.to(tl.float32)))

        # final state at index nc (this sequence's own chunk count, <= MAX_NC)
        off = ((i_bh * (MAX_NC + 1) + nc) * D_V * D_K
               + o_v[:, None] * D_K + o_k[None, :])
        tl.store(M_bound + off, M.to(M_bound.dtype.element_ty), mask=mask_kv)
        tl.store(I_bound + off, I.to(I_bound.dtype.element_ty), mask=mask_kv)

    def _fp_ibar_autotune():
        # BK only. BV_IB is NOT here and must never be: `o_v = tl.arange(0, BV_IB)` below
        # has no grid dimension and no tiling loop over V at all (unlike BK, which the
        # grid tiles via triton.cdiv(D_K,BK)) -- Ibar_c = sum_v(I_c)/D_V is a genuine
        # full-D_V reduction computed in one shot. Narrowing BV_IB below D_V would
        # silently drop channels from that mean, not just run slower -- the same failure
        # mode as MOPA's BK-truncation bug (see NEXT_mopa_speed_autotune.md), just on the
        # V axis instead of K. BV_IB stays a fixed `triton.next_power_of_2(D_V)` kwarg at
        # the call site, same as before. num_stages capped at 2 -- this kernel has the
        # same `for c in range(nc)` loop shape as the other autotuned chunk-loop kernels.
        return [
            triton.Config({'BK': bk}, num_warps=w, num_stages=s)
            for bk in _FP_BK_CHOICES
            for w in (2, 4, 8)
            for s in (1, 2)
        ]

    @triton.heuristics({
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    })
    @triton.autotune(configs=_fp_ibar_autotune(), key=['D_K', 'D_V', 'C'],
                     **autotune_cache_kwargs)
    @triton.jit(do_not_specialize=['T'])
    def _fp_ibar_kernel(
        k, b, gt, g, Ip, I_bound,
        Ibar_out,                    # (B*H,MAX_NC,C,DK) isotropic per-token precision
        cu_seqlens,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, C: tl.constexpr,
        BV_IB: tl.constexpr, MAX_NC: tl.constexpr,
        PERDK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        # One program per (b,h,k-block). Collapses I_bound over D_V and evolves Ibar_t.
        i_k, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_b).to(tl.int32)
            eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
            T_seq = eos - bos
            nc = tl.cdiv(T_seq, C)
        else:
            bos = i_b * T
            T_seq = T
            nc = T // C
        o_c = tl.arange(0, C)
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < D_K

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)

        o_v = tl.arange(0, BV_IB)
        mask_v = o_v < D_V
        for c in range(nc):
            # collapse I_bound[c] over D_V  -> Ibar_c (BK,)  (vectorized block load)
            off_iv = ((i_bh * (MAX_NC + 1) + c) * D_V * D_K
                      + o_v[:, None] * D_K + o_k[None, :])
            I_c = tl.load(I_bound + off_iv, mask=(mask_v[:, None] & mask_k[None, :]),
                          other=0.0).to(tl.float32)
            Ibar_c = tl.sum(tl.where(mask_v[:, None], I_c, 0.0), axis=0) / D_V
            abar = Ibar_c - b_Ip

            mask_c = o_c < tl.minimum(C, T_seq - c * C)
            base_qk = ((bos + c * C) * H + i_h) * D_K
            base_vo = ((bos + c * C) * H + i_h) * D_V
            base_gt = (bos + c * C) * H + i_h
            # k_ck is this program's own k-tile, one-shot -- evict_first. gt_c/b_cv
            # don't depend on o_k (there's no i_v grid dim here -- o_v is fixed at the
            # full D_V width for every program), so every i_k program at this i_bh
            # reloads them identically -- evict_last.
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_c[:, None] & mask_k[None, :], other=0.0,
                           eviction_policy='evict_first').to(tl.float32)
            gt_c = tl.load(gt + base_gt + o_c * H, mask=mask_c, other=0.0,
                           eviction_policy='evict_last').to(tl.float32)
            logf = -gt_c * b_g
            clogf = tl.cumsum(logf, axis=0)
            cd = exp(clogf)
            ksq = k_ck * k_ck

            # betabar_t = mean over D_V of b  (vectorized block load over D_V)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0,
                           eviction_policy='evict_last').to(tl.float32)
            bbar = tl.sum(b_cv, axis=1) / D_V

            # Ibar_t = b_Ip + a_t,  a_t = f_t a_{t-1} + bbar_t ksq_t,  a_{-1}=abar
            #   a_mat = cd*abar + Dm @ (bbar*ksq),  Dm[t,i]=exp(clogf_t-clogf_i) lower-tri
            Dm = tl.where(o_c[:, None] >= o_c[None, :],
                          exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0)), 0.0)
            a_mat = cd[:, None] * abar[None, :] + _sdot(
                Dm.to(tl.float32), (bbar[:, None] * ksq).to(tl.float32))
            
            Ibar_t = b_Ip[None, :] + a_mat                   # (C,BK)
            off_out = ((i_bh * MAX_NC + c) * C + o_c[:, None]) * D_K + o_k[None, :]
            tl.store(Ibar_out + off_out, Ibar_t, mask=mask_k[None, :])

    def _fp_output_autotune():
        # BK only, same reasoning as _fp_ibar_autotune: this kernel's `Ibar_c =
        # sum_v(I_c)/D_V` (per k-tile, inside the `for i_k in range(NK)` loop) is a
        # full-D_V reduction over the v-BLOCK a program owns -- BV must equal the whole
        # D_V or that mean is silently computed over only part of it (was
        # `_fp_output_block_sizes`'s own hand-picked-heuristic invariant: "BV must cover
        # the full D_V ... so we shrink BK and num_stages instead"). BV stays a fixed
        # `triton.next_power_of_2(D_V)` kwarg. This kernel's own loop is `for i_k in
        # range(NK)` over K-tiles, not chunks, so it doesn't share BSCAN/BREAD's
        # chunk-loop num_stages risk -- but capped at 2 anyway for consistency with
        # every other autotuned kernel in this file; not worth being the one exception.
        return [
            triton.Config({'BK': bk}, num_warps=w, num_stages=s)
            for bk in _FP_BK_CHOICES
            for w in (2, 4, 8)
            for s in (1, 2)
        ]

    @triton.heuristics({
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    })
    @triton.autotune(configs=_fp_output_autotune(), key=['D_K', 'D_V', 'C'],
                     **autotune_cache_kwargs)
    @triton.jit(do_not_specialize=['T'])
    def _fp_output_kernel(
        q, k, v, gt, g, Ip,
        M_bound, I_bound, Ibar,
        o, scale,
        cu_seqlens,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        MAX_NC: tl.constexpr,
        PERDK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        # One program per (b,h,chunk,v-block). Accumulates local score over BK
        # blocks, then local @ v and carry @ q. Grid's chunk axis is padded to
        # MAX_NC across the whole launch; a program whose chunk index doesn't
        # exist for its own (shorter) sequence exits immediately.
        i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_b).to(tl.int32)
            eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
            T_seq = eos - bos
            nc = tl.cdiv(T_seq, C)
        else:
            bos = i_b * T
            T_seq = T
            nc = T // C
        if i_c >= nc:
            return
        mask_c = tl.arange(0, C) < tl.minimum(C, T_seq - i_c * C)
        o_c = tl.arange(0, C)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = o_v < D_V

        base_gt = (bos + i_c * C) * H + i_h
        b_g = tl.load(g + i_h).to(tl.float32)
        gt_c = tl.load(gt + base_gt + o_c * H, mask=mask_c, other=0.0).to(tl.float32)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        carry_decay = exp(clogf)                       # (C,)
        Dmask = exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0))
        causal = o_c[:, None] >= o_c[None, :]
        Dmask = tl.where(causal, Dmask, 0.0)              # (C,C)

        base_qk = ((bos + i_c * C) * H + i_h) * D_K
        base_vo = ((bos + i_c * C) * H + i_h) * D_V

        # v_cv is this program's own v-tile, read once -- evict_first. q_ck/k_ck/ibar/
        # b_Ip_out below don't depend on o_v at all: every i_v program at this (i_k,
        # i_c, i_bh) reloads the identical values, so evict_last keeps them warm in L2
        # for the other concurrent v-tile programs. M_c/I_c genuinely are v-tile
        # specific (depend on o_v) and read once per (i_k, this program) -- evict_first.
        v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                       mask=mask_c[:, None] & mask_v[None, :], other=0.0,
                       eviction_policy='evict_first').to(tl.float32)

        score = tl.zeros([C, C], dtype=tl.float32)
        carry = tl.zeros([C, BV], dtype=tl.float32)
        NK = tl.cdiv(D_K, BK)
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            mask_k = o_k < D_K
            mask_kv = mask_v[:, None] & mask_k[None, :]
            if PERDK:
                b_Ip_out = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0,
                                   eviction_policy='evict_last').to(tl.float32)
            else:
                b_Ip_out = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
            b_Ip_out = tl.where(mask_k, b_Ip_out, 1.0)
            q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_c[:, None] & mask_k[None, :], other=0.0,
                           eviction_policy='evict_last').to(tl.float32) * scale
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_c[:, None] & mask_k[None, :], other=0.0,
                           eviction_policy='evict_last').to(tl.float32)
            ibar = tl.load(Ibar + ((i_bh * MAX_NC + i_c) * C + o_c[:, None]) * D_K + o_k[None, :],
                           mask=mask_k[None, :], other=1.0,
                           eviction_policy='evict_last').to(tl.float32)
            ibar = tl.maximum(ibar, b_Ip_out[None, :])
            Qtil = q_ck / ibar
            score += _sdot(Qtil.to(tl.float32), tl.trans(k_ck))

            # carry term: Qtil @ M_c^T over this k-block
            off_st = ((i_bh * (MAX_NC + 1) + i_c) * D_V * D_K
                      + o_v[:, None] * D_K + o_k[None, :])
            M_c = tl.load(M_bound + off_st, mask=mask_kv, other=0.0,
                         eviction_policy='evict_first').to(tl.float32)
            I_c = tl.load(I_bound + off_st, mask=mask_kv, other=1.0,
                         eviction_policy='evict_first').to(tl.float32)
            I_c = tl.maximum(I_c, b_Ip_out[None, :])

            # Since BV covers all of D_V, we safely reduce directly in SRAM/Registers
            Ibar_c = tl.sum(tl.where(mask_v[:, None], I_c, 0.0), axis=0) / D_V
            Q_carry_til = q_ck * (Ibar_c[None, :] / ibar)

            mu_c = M_c / I_c
            carry += _sdot(Q_carry_til.to(tl.float32), tl.trans(mu_c))

        score = score * Dmask
        y_local = _sdot(score.to(tl.float32), v_cv)
        y = y_local + carry * carry_decay[:, None]
        tl.store(o + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                 y.to(o.dtype.element_ty), mask=mask_c[:, None] & mask_v[None, :])

    def _fp_compute_state(k, v, b, gt, g, Ip, initial_mu_state, initial_I_state, cu_seqlens,
                          T, H, D_K, D_V, C, MAX_NC, perdk, num_bh, dev):
        """Chunk-boundary M/I state trajectory + per-token Ibar.

        Called once, from `_FastPalimpsa.forward`. (Not shared with the backward: an
        earlier design recomputed this in `_fast_palimpsa_bwd_triton` instead of saving
        it across the autograd boundary, measured 22-40% *slower* -- see forward's own
        docstring note -- and was reverted; the trajectory is saved and reused instead.)

        BK/BV are autotuned per kernel now (`_fp_fwd_state_autotune`/`_fp_ibar_autotune`),
        not passed in -- both kernels tile D_K via the grid (`triton.cdiv(D_K, META['BK'])`
        below), and `_fp_state_kernel` additionally tiles D_V the same way, so narrowing
        either is a speed tradeoff, not a correctness one. `_fp_ibar_kernel`'s D_V axis is
        the opposite case -- `BV_IB` has no grid or loop tiling it at all (a genuine
        full-D_V reduction), so it stays a fixed, non-autotuned `next_power_of_2(D_V)`.
        """
        M_bound = torch.empty(num_bh, MAX_NC + 1, D_V, D_K, device=dev, dtype=torch.bfloat16)
        I_bound = torch.empty(num_bh, MAX_NC + 1, D_V, D_K, device=dev, dtype=torch.float32)
        Ibar = torch.empty(num_bh, MAX_NC, C, D_K, device=dev, dtype=torch.float32)
        grid_state = lambda META: (triton.cdiv(D_K, META['BK']), triton.cdiv(D_V, META['BV']), num_bh)
        _fp_state_kernel[grid_state](
            k, v, b, gt, g, Ip, initial_mu_state, initial_I_state, M_bound, I_bound, cu_seqlens,
            T, H=H, D_K=D_K, D_V=D_V, C=C, MAX_NC=MAX_NC, PERDK=perdk)
        grid_ibar = lambda META: (triton.cdiv(D_K, META['BK']), num_bh)
        _fp_ibar_kernel[grid_ibar](
            k, b, gt, g, Ip, I_bound, Ibar, cu_seqlens,
            T, H=H, D_K=D_K, D_V=D_V, C=C,
            BV_IB=triton.next_power_of_2(D_V), MAX_NC=MAX_NC, PERDK=perdk)
        return M_bound, I_bound, Ibar

    def _fp_bwd_autotune():
        # BK up to 128 on Hopper+ only -- see _FP_BK_CHOICES. Both kernels this serves
        # (_fp_bwd_kernel_local_state, _fp_bwd_kernel_scan) grid-tile K via
        # triton.cdiv(D_K, META['BK']) (program_id(0) = i_k), so a wider BK is a
        # legitimate fewer-iterations tradeoff there too, same as the forward kernels.
        return [
            triton.Config({'BK': bk, 'BV': bv}, num_warps=w, num_stages=s)
            for bk in _FP_BK_CHOICES
            for bv in (16, 32, 64)
            for w in (2, 4, 8)
            for s in (1, 2)
        ]

    @triton.heuristics({
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    })
    @triton.autotune(configs=_fp_bwd_autotune(), key=['D_K', 'D_V', 'C'], **autotune_cache_kwargs)
    @triton.jit(do_not_specialize=['T'])
    def _fp_bwd_kernel_local_state(
        q, k, v, b, gt, g, Ip,
        M_bound, I_bound, do,
        dM_local_out, dI_local_out, scale,
        cu_seqlens,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        MAX_NC: tl.constexpr,
        PERDK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        """Pass 1/3 (Parallel), BV-tiled. grid=(NK, MAX_NC, num_bh).

        Two sweeps over the D_V tiles: sweep A accumulates the D_V-reductions
        (Ibar_c, bbar, dsc, dQtil_carry); the middle [C,*] algebra is computed
        once (no D_V axis, fully resident); sweep B writes the per-(D_V-block)
        out_dM/out_dI. No full-D_V residency -> fits SMEM for any D_V.
        """
        i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_b).to(tl.int32)
            eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
            T_seq = eos - bos
            nc = tl.cdiv(T_seq, C)
        else:
            bos = i_b * T
            T_seq = T
            nc = T // C
        c = i_c
        if i_c >= nc:
            return
        mask_c = tl.arange(0, C) < tl.minimum(C, T_seq - c * C)

        o_c = tl.arange(0, C)
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < D_K
        NV = tl.cdiv(D_V, BV)

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_Ip = tl.where(mask_k, b_Ip, 1.0)
        b_g = tl.load(g + i_h).to(tl.float32)

        base_qk = ((bos + c * C) * H + i_h) * D_K
        base_vo = ((bos + c * C) * H + i_h) * D_V
        base_gt = (bos + c * C) * H + i_h

        q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        gt_c = tl.load(gt + base_gt + o_c * H, mask=mask_c, other=0.0).to(tl.float32)

        qs = q_ck * scale
        f_c = exp(-gt_c * b_g)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        cd = exp(clogf)
        ksq = k_ck * k_ck

        Dm_full = exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0))
        tri = o_c[:, None] >= o_c[None, :]
        Dm = tl.where(tri, Dm_full, 0.0)

        # ---- sweep A: accumulate D_V-reductions over value-tiles ----
        Ibar_c = tl.zeros([BK], dtype=tl.float32)              # sum_v I_c
        bbar = tl.zeros([C], dtype=tl.float32)                 # sum_v b
        dsc = tl.zeros([C, C], dtype=tl.float32)               # do @ v^T
        dQtil_carry = tl.zeros([C, BK], dtype=tl.float32)      # (do*cd) @ mu
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (MAX_NC + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            mu_v = M_cv / I_cv
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            Ibar_c += tl.sum(tl.where(mask_v[:, None], I_cv, 0.0), axis=0)
            bbar += tl.sum(tl.where(mask_v[None, :], b_cv, 0.0), axis=1)
            dbase = do_cv * cd[:, None]
            dsc += _sdot(do_cv.to(tl.float32), tl.trans(v_cv).to(tl.float32))
            dQtil_carry += _sdot(dbase.to(tl.float32), mu_v.to(tl.float32))
        Ibar_c = Ibar_c / D_V
        bbar = bbar / D_V

        # ---- middle: [C,*] algebra, no D_V axis ----
        abar = tl.where(mask_k, Ibar_c - b_Ip, 0.0)
        src = bbar[:, None] * ksq
        a_mat = cd[:, None] * abar[None, :] + _sdot(Dm.to(tl.float32), src.to(tl.float32))
        Ibar_mat = tl.maximum(b_Ip[None, :] + a_mat, b_Ip[None, :])
        Qtil = qs / Ibar_mat

        dsc_raw = dsc * Dm
        dQtil_local = _sdot(dsc_raw.to(tl.float32), k_ck.to(tl.float32))
        dIbar = -(dQtil_local * qs / (Ibar_mat * Ibar_mat)) \
                - (dQtil_carry * qs * Ibar_c[None, :] / (Ibar_mat * Ibar_mat))
        dIbar_c = tl.sum(dQtil_carry * (qs / Ibar_mat), axis=0)        # (BK,)
        Wt = tl.where(o_c[None, :] >= o_c[:, None], exp(tl.minimum(clogf[None, :] - clogf[:, None], 0.0)), 0.0)
        D = _sdot(Wt.to(tl.float32), dIbar.to(tl.float32))
        D0 = tl.sum(tl.where(o_c[:, None] == 0, D, 0.0), axis=0)
        f0 = tl.sum(tl.where(o_c == 0, f_c, 0.0))
        dabar = f0 * D0                                                # (BK,)
        bdry_dI = (dIbar_c + dabar) / D_V                              # broadcast over v

        Q_carry_til = qs * (Ibar_c[None, :] / Ibar_mat)

        # ---- sweep B: per-block out_dM / out_dI ----
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (MAX_NC + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            dbase = do_cv * cd[:, None]
            dmu = _sdot(tl.trans(dbase).to(tl.float32), Q_carry_til.to(tl.float32))   # (BV,BK)
            out_dM = dmu / I_cv
            out_dI = -dmu * M_cv / (I_cv * I_cv) + tl.where(mask_v[:, None], bdry_dI[None, :], 0.0)
            off_out = (i_bh * MAX_NC + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            tl.store(dM_local_out + off_out, out_dM, mask=mask_kv)
            tl.store(dI_local_out + off_out, out_dI, mask=mask_kv)


    @triton.heuristics({
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    })
    @triton.autotune(configs=_fp_bwd_autotune(), key=['D_K', 'D_V', 'C'], **autotune_cache_kwargs)
    @triton.jit(do_not_specialize=['T'])
    def _fp_bwd_kernel_scan(
        dM_local, dI_local, gt, g,
        dM_bound, dI_bound, cu_seqlens,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        MAX_NC: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        """Pass 2/3 (Sequential over chunks), BV/BK-tiled. grid=(NK, NV, num_bh).
        Pure elementwise recurrence (Flast scalar per chunk) -> tiles trivially."""
        i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_b).to(tl.int32)
            eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
            T_seq = eos - bos
            nc = tl.cdiv(T_seq, C)
        else:
            bos = i_b * T
            T_seq = T
            nc = T // C

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < D_K
        mask_v = o_v < D_V
        mask_kv = mask_v[:, None] & mask_k[None, :]

        dM = tl.zeros([BV, BK], dtype=tl.float32)
        dI = tl.zeros([BV, BK], dtype=tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)

        off_bound_nc = (i_bh * (MAX_NC + 1) + nc) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
        tl.store(dM_bound + off_bound_nc, dM, mask=mask_kv)
        tl.store(dI_bound + off_bound_nc, dI, mask=mask_kv)

        for cc in range(nc):
            c = nc - 1 - cc

            mask_c = tl.arange(0, C) < tl.minimum(C, T_seq - c * C)
            base_gt = (bos + c * C) * H + i_h
            o_c = tl.arange(0, C)
            gt_c = tl.load(gt + base_gt + o_c * H, mask=mask_c, other=0.0).to(tl.float32)
            logf = -gt_c * b_g
            clogf = tl.cumsum(logf, axis=0)
            clogf_last = tl.sum(tl.where(o_c == (C - 1), clogf, 0.0))
            Flast = exp(clogf_last)

            off_local = (i_bh * MAX_NC + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            dMl = tl.load(dM_local + off_local, mask=mask_kv, other=0.0).to(tl.float32)
            dIl = tl.load(dI_local + off_local, mask=mask_kv, other=0.0).to(tl.float32)

            dM = Flast * dM + dMl
            dI = Flast * dI + dIl

            off_bound = (i_bh * (MAX_NC + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            tl.store(dM_bound + off_bound, dM, mask=mask_kv)
            tl.store(dI_bound + off_bound, dI, mask=mask_kv)

    @triton.heuristics({
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    })
    @triton.autotune(
        configs=[triton.Config({'BV': bv}, num_warps=w, num_stages=s)
                 for bv in (16, 32, 64) for w in (2, 4, 8) for s in (1, 2)],
        key=['D_K', 'D_V', 'C'], **autotune_cache_kwargs)
    @triton.jit(do_not_specialize=['T'])
    def _fp_bwd_kernel_intra(
        q, k, v, b, gt, g, Ip,
        M_bound, I_bound, do,
        dM_bound, dI_bound,
        dq, dk, dv, db, dgt, scale,
        cu_seqlens,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        MAX_NC: tl.constexpr,
        PERDK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        """Pass 3/3 (Parallel), BV-tiled with BK full-resident. grid=(MAX_NC, num_bh).

        BK = next_pow2(D_K) is kept resident (it fits; only BV=next_pow2(D_V) blew
        SMEM). Two sweeps over D_V tiles: sweep A accumulates the D_V-reductions
        feeding the [C,*] algebra (Ibar_c, bbar, dM/dI-contractions, dsc,
        dQtil_carry, dcd, pM, pI, cM, cI); sweep B writes per-block dv, db and
        adds the per-block dk/df contributions. dq/dk/dgt have full D_K resident
        so no cross-block accumulation is needed.

        Ragged tail (varlen): every per-token load below is masked by `mask_c`, so
        every downstream quantity built from a padded row (dsc, dclogf, dIbar, D,
        ...) is provably zero there -- a padded row can never leak gradient into a
        real one purely through this masking (no explicit re-derivation of the
        boundary terms was needed). The three per-token *stores* (dq, dk, dgt, dv,
        db) are separately masked with `mask_c` too -- that one IS load-bearing:
        those tensors are the physical packed buffers, so an unmasked write to a
        padded tail position would silently corrupt the START of the NEXT
        sequence's gradient.
        """
        i_c, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_b).to(tl.int32)
            eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
            T_seq = eos - bos
            nc = tl.cdiv(T_seq, C)
        else:
            bos = i_b * T
            T_seq = T
            nc = T // C
        c = i_c
        if i_c >= nc:
            return
        mask_c = tl.arange(0, C) < tl.minimum(C, T_seq - c * C)

        o_c = tl.arange(0, C)
        o_k = tl.arange(0, BK)
        mask_k = o_k < D_K
        NV = tl.cdiv(D_V, BV)

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_Ip = tl.where(mask_k, b_Ip, 1.0)
        b_g = tl.load(g + i_h).to(tl.float32)

        base_qk = ((bos + c * C) * H + i_h) * D_K
        base_vo = ((bos + c * C) * H + i_h) * D_V
        base_gt = (bos + c * C) * H + i_h

        q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        gt_c = tl.load(gt + base_gt + o_c * H, mask=mask_c, other=0.0).to(tl.float32)

        qs = q_ck * scale
        f_c = exp(-gt_c * b_g)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        cd = exp(clogf)
        ksq = k_ck * k_ck
        Dm_full = exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0))
        tri = o_c[:, None] >= o_c[None, :]
        Dm = tl.where(tri, Dm_full, 0.0)
        clogf_last = tl.sum(tl.where(o_c == (C - 1), clogf, 0.0))
        sca = exp(clogf_last - clogf)

        # ---- sweep A: all D_V-reductions ----
        Ibar_c = tl.zeros([BK], dtype=tl.float32)
        bbar = tl.zeros([C], dtype=tl.float32)
        dsc = tl.zeros([C, C], dtype=tl.float32)               # do @ v^T
        dQtil_carry = tl.zeros([C, BK], dtype=tl.float32)      # (do*cd) @ mu
        dcd = tl.zeros([C], dtype=tl.float32)                  # sum_v do*base
        pM = tl.zeros([C], dtype=tl.float32)
        pI = tl.zeros([C], dtype=tl.float32)
        cM = tl.zeros([1], dtype=tl.float32)
        cI = tl.zeros([1], dtype=tl.float32)
        dk_from_state = tl.zeros([C, BK], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (MAX_NC + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            mu_v = M_cv / I_cv
            off_dn = ((i_bh * (MAX_NC + 1) + c + 1) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            dM_v = tl.load(dM_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            dI_v = tl.load(dI_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            Ibar_c += tl.sum(tl.where(mask_v[:, None], I_cv, 0.0), axis=0)
            bbar += tl.sum(tl.where(mask_v[None, :], b_cv, 0.0), axis=1)
            vdM = _sdot(v_cv.to(tl.float32), dM_v.to(tl.float32))   # (C,BK)
            bdI = _sdot(b_cv.to(tl.float32), dI_v.to(tl.float32))   # (C,BK)
            pM += tl.sum(vdM * k_ck, axis=1)
            pI += tl.sum(bdI * ksq, axis=1)
            cM += tl.sum(M_cv * dM_v)
            cI += tl.sum((I_cv - b_Ip[None, :]) * dI_v)
            dk_from_state += sca[:, None] * (2.0 * k_ck * bdI + vdM)
            dbase = do_cv * cd[:, None]
            dsc += _sdot(do_cv.to(tl.float32), tl.trans(v_cv).to(tl.float32))
            dQtil_carry += _sdot(dbase.to(tl.float32), mu_v.to(tl.float32))
        Ibar_c = Ibar_c / D_V
        bbar = bbar / D_V

        # ---- middle [C,*] algebra ----
        abar = tl.where(mask_k, Ibar_c - b_Ip, 0.0)
        src = bbar[:, None] * ksq
        a_mat = cd[:, None] * abar[None, :] + _sdot(Dm.to(tl.float32), src.to(tl.float32))
        Ibar_mat = tl.maximum(b_Ip[None, :] + a_mat, b_Ip[None, :])
        Qtil = qs / Ibar_mat
        sc_raw = _sdot(Qtil.to(tl.float32), tl.trans(k_ck).to(tl.float32))
        sc = sc_raw * Dm
        Q_carry_til = qs * (Ibar_c[None, :] / Ibar_mat)

        dq_acc = tl.zeros([C, BK], dtype=tl.float32)
        dk_acc = dk_from_state
        df = tl.zeros([C], dtype=tl.float32)
        dclogf = tl.zeros([C], dtype=tl.float32)

        # df from state (pM/pI/cM/cI)
        clogf_prev = clogf - logf
        Fprev = exp(clogf_prev)
        W = tl.where(o_c[:, None] > o_c[None, :], exp(tl.minimum(clogf_prev[:, None] - clogf[None, :], 0.0)), 0.0)
        dfM = Fprev * tl.sum(cM) + tl.sum(W * pM[None, :], axis=1)
        dfI = Fprev * tl.sum(cI) + tl.sum(W * pI[None, :], axis=1)
        df += sca * (dfI + dfM)

        dclogf += dcd * cd  # dcd is 0 here; folded below via the carry path
        # carry-path dcd: dcd_t = sum_v do*base ; base = Q_carry_til @ mu^T.
        # base needs mu (D_V) -> recompute in sweep B and accumulate dcd there.

        dsc_raw = dsc * Dm
        dDm = dsc * sc_raw
        dexp = tl.where(tri, dDm, 0.0)
        dclogf += tl.sum(dexp * Dm_full, axis=1)
        dclogf += -tl.sum(dexp * Dm_full, axis=0)
        dQtil_local = _sdot(dsc_raw.to(tl.float32), k_ck.to(tl.float32))
        dk_acc += _sdot(tl.trans(dsc_raw).to(tl.float32), Qtil.to(tl.float32))

        dqs = (dQtil_local / Ibar_mat) + (dQtil_carry * Ibar_c[None, :] / Ibar_mat)
        dIbar = -(dQtil_local * qs / (Ibar_mat * Ibar_mat)) \
                - (dQtil_carry * qs * Ibar_c[None, :] / (Ibar_mat * Ibar_mat))
        dq_acc += dqs * scale

        Wt = tl.where(o_c[None, :] >= o_c[:, None], exp(tl.minimum(clogf[None, :] - clogf[:, None], 0.0)), 0.0)
        D = _sdot(Wt.to(tl.float32), dIbar.to(tl.float32))
        
        Dm_prev = tl.where(o_c[:, None] > o_c[None, :], exp(tl.minimum(clogf_prev[:, None] - clogf[None, :], 0.0)), 0.0)
        a_prev = Fprev[:, None] * abar[None, :] + _sdot(Dm_prev.to(tl.float32), src.to(tl.float32))
        df += tl.sum(a_prev * D, axis=1)
        dk_acc += bbar[:, None] * 2.0 * k_ck * D
        kD = tl.sum(ksq * D, axis=1) / D_V    # (C,) -> broadcast into db per block

        # ---- sweep B: per-block dv, db, and dcd accumulation ----
        dcd_acc = tl.zeros([C], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (MAX_NC + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            mu_v = M_cv / I_cv
            off_dn = ((i_bh * (MAX_NC + 1) + c + 1) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            dM_v = tl.load(dM_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            dI_v = tl.load(dI_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            # dcd contribution: base = Q_carry_til @ mu^T  (C,BV)
            base_blk = _sdot(Q_carry_til.to(tl.float32), tl.trans(mu_v).to(tl.float32))
            dcd_acc += tl.sum(do_cv * base_blk, axis=1)
            # dv, db per block
            dv_blk = sca[:, None] * _sdot(k_ck.to(tl.float32), tl.trans(dM_v).to(tl.float32))
            dv_blk += _sdot(tl.trans(sc).to(tl.float32), do_cv.to(tl.float32))
            db_blk = sca[:, None] * _sdot(ksq.to(tl.float32), tl.trans(dI_v).to(tl.float32))
            db_blk += kD[:, None] * tl.where(mask_v[None, :], 1.0, 0.0)
            tl.store(dv + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], dv_blk,
                     mask=mask_c[:, None] & mask_v[None, :])
            tl.store(db + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], db_blk,
                     mask=mask_c[:, None] & mask_v[None, :])

        # fold dcd (carry-decay path) into dclogf, finalize df, dgt
        dclogf += dcd_acc * cd
        csum = tl.cumsum(dclogf, axis=0)
        total = tl.sum(dclogf)
        dlogf = total - csum + dclogf
        dgt_c = (df * f_c + dlogf) * (-b_g)

        tl.store(dq + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dq_acc,
                 mask=mask_c[:, None] & mask_k[None, :])
        tl.store(dk + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dk_acc,
                 mask=mask_c[:, None] & mask_k[None, :])
        tl.store(dgt + base_gt + o_c * H, dgt_c, mask=mask_c)

    @lru_cache(maxsize=None)
    def _fp_clamp_C(D_K, D_V, C, dev):
        """Largest power-of-two chunk <= C that actually runs (forward AND
        backward) on this device without exceeding shared memory.

        Checked with a real, minimal (B=1,H=1,T=2*c) forward+backward dry run
        through `_FastPalimpsa.apply` directly, catching any compile/launch
        failure -- not a guessed formula. Forward and backward kernels are all
        `@triton.autotune`d now, each with independent shared-memory footprints
        that don't reduce to one closed-form estimate in D_K, D_V, C alone --
        which one binds first flips depending on whether D_K or D_V dominates --
        so this asks the real kernels directly instead of estimating. Side
        effect: this also pre-warms triton's own autotune cache for the
        shape, so the real subsequent call doesn't repeat the search.

        Cached: a live device-properties query plus Python-side branching on
        every forward call is host-side work that competes with issuing the
        next layer's FSDP unshard/all-gather prefetch on the same thread --
        same reasoning as every other per-device/per-shape memoization in this
        file. FLOOR IS 16: tl.dot requires the contraction dim K>=16, so the
        kernel cannot run below C=16. If even C=16 won't fit, keep 16 and let
        the real launch raise a clear OutOfResources rather than emit a broken
        C=8.
        """
        def runs(c):
            try:
                T = 2 * c
                mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32, requires_grad=True)
                q, k, v = mk(1, T, 1, D_K), mk(1, T, 1, D_K), mk(1, T, 1, D_V)
                b = (torch.rand(1, T, 1, D_V, device=dev) * 1.5 + 0.1).requires_grad_(True)
                gt = (torch.rand(1, T, 1, device=dev) * 0.1).requires_grad_(True)
                g = torch.rand(1, device=dev) * 0.5 + 0.5
                Ip = torch.rand(1, device=dev) + 0.5
                o, _, _ = _FastPalimpsa.apply(q, k, v, b, gt, g, Ip, D_K ** -0.5, c,
                                              None, None, None, False)
                o.sum().backward()
                torch.cuda.synchronize()
                return True
            except Exception:
                return False

        c = C
        while c > 16 and not runs(c):
            c //= 2
        return c

    def _fast_palimpsa_bwd_triton(do, q, k, v, b, gt, g, Ip, scale, C,
                                  M_bound, I_bound, Ibar, cu_seqlens=None, MAX_NC=None):
        B, T, H, D_K = q.shape
        D_V = v.shape[-1]
        dev = q.device
        is_varlen = cu_seqlens is not None
        if is_varlen:
            num_bh = (cu_seqlens.numel() - 1) * H
            assert MAX_NC is not None
        else:
            num_bh = B * H
            if MAX_NC is None:
                MAX_NC = T // C
        nc = MAX_NC
        BK_full = triton.next_power_of_2(D_K)
        perdk = torch.is_tensor(Ip) and Ip.numel() == H * D_K

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        db = torch.zeros_like(b, dtype=torch.float32)
        dgt = torch.zeros_like(gt, dtype=torch.float32)

        Ipf = (Ip.float().contiguous() if torch.is_tensor(Ip)
               else torch.full((H,), float(Ip), device=dev, dtype=torch.float32))

        qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]
        gf = g.float().contiguous()
        dof = do.contiguous().float()

        dM_local = torch.empty(num_bh, nc, D_V, D_K, device=dev, dtype=torch.float32)
        dI_local = torch.empty(num_bh, nc, D_V, D_K, device=dev, dtype=torch.float32)
        dM_bound = torch.empty(num_bh, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
        dI_bound = torch.empty(num_bh, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)

        # 1. local-state (BV/BK tiled, autotuned). grid=(NK, MAX_NC, num_bh)
        grid_ls = lambda META: (triton.cdiv(D_K, META['BK']), nc, num_bh)
        _fp_bwd_kernel_local_state[grid_ls](
            qc, kc, vc, bc, gtc, gf, Ipf,
            M_bound, I_bound, dof,
            dM_local, dI_local, scale, cu_seqlens,
            T, H=H, D_K=D_K, D_V=D_V, C=C, MAX_NC=nc, PERDK=perdk,
        )

        # 2. sequential scan over chunks (BV/BK tiled, autotuned). grid=(NK, NV, num_bh)
        grid_sc = lambda META: (triton.cdiv(D_K, META['BK']), triton.cdiv(D_V, META['BV']), num_bh)
        _fp_bwd_kernel_scan[grid_sc](
            dM_local, dI_local, gtc, gf,
            dM_bound, dI_bound, cu_seqlens,
            T, H=H, D_K=D_K, D_V=D_V, C=C, MAX_NC=nc,
        )

        # 3. intra (BV tiled, BK full-resident, autotuned). grid=(MAX_NC, num_bh)
        grid_in = lambda META: (nc, num_bh)
        _fp_bwd_kernel_intra[grid_in](
            qc, kc, vc, bc, gtc, gf, Ipf,
            M_bound, I_bound, dof,
            dM_bound, dI_bound,
            dq, dk, dv, db, dgt, scale, cu_seqlens,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK_full, C=C, MAX_NC=nc, PERDK=perdk,
        )

        return dq, dk, dv, db, dgt

    class _FastPalimpsa(torch.autograd.Function):
        @staticmethod
        @contiguous
        @torch.autocast(device_type="cuda")
        def forward(ctx, q, k, v, b, gt, g, Ip, scale, C, initial_mu_state, initial_I_state,
                    cu_seqlens, recompute_state):
            B, T, H, D_K = q.shape
            D_V = v.shape[-1]
            dev = q.device
            is_varlen = cu_seqlens is not None
            if is_varlen:
                cu_seqlens = cu_seqlens.to(device=dev, dtype=torch.int32)
                num_bh = (cu_seqlens.numel() - 1) * H
                seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                MAX_NC = int(triton.cdiv(int(seqlens.max().item()), C))
            else:
                num_bh = B * H
                MAX_NC = T // C
            # BV_out: _fp_output_kernel's own Ibar_c = sum_v(I_c)/D_V reduction needs the
            # full D_V per program (see _fp_output_autotune's comment) -- fixed, not
            # autotuned, same as _fp_ibar_kernel's BV_IB above.
            BV_out = triton.next_power_of_2(D_V)
            perdk = torch.is_tensor(Ip) and Ip.numel() == H * D_K
            Ipf = (Ip.float().contiguous() if torch.is_tensor(Ip)
                   else torch.full((H,), float(Ip), device=dev, dtype=torch.float32))
            gf = g.float().contiguous()
            init_mu = (initial_mu_state.float().contiguous()
                       if initial_mu_state is not None else None)
            init_I = (initial_I_state.float().contiguous()
                      if initial_I_state is not None else None)
            o = torch.empty(B, T, H, D_V, device=dev, dtype=q.dtype)
            qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]

            # M_bound: chunk-boundary M, bf16 checkpoint (read only via a matmul
            # against q, like fla's GLA/mesa_net states -- see _fp_compute_state).
            # I_bound stays fp32: read through a division (mu_c = M_c / I_c, and
            # again inside _fp_ibar_kernel's Ibar), and measured empirically
            # (test_fwd_bwd across D_K/D_V/C) to push db's relative error past
            # this file's 5e-3 gate in bf16 -- more precision-sensitive than
            # GLA's plain matmul state, as expected for a division-read state.
            #
            # M_bound/I_bound/Ibar are saved across the autograd boundary for backward to
            # reuse BY DEFAULT (as opposed to fla's usual recompute-in-backward pattern
            # for this class of kernel): measured on this GPU (before this session's
            # forward-kernel autotune work), recompute made fwd+bwd wall-clock 22-40%
            # *slower* despite ~20-27% less peak memory -- this kernel's per-chunk state
            # update is apparently proportionally more expensive than GLA/mesa_net's, so
            # recompute's "quite cheap" premise (their own comment) doesn't transfer.
            # `recompute_state=True` re-enables the option now that the forward kernels
            # this recomputes are themselves faster -- re-measure before trusting it
            # (the old rejection was never confirmed stale, just flagged as
            # possibly-stale; see improve_fast_palimpsa.md's addendum).
            M_bound, I_bound, Ibar = _fp_compute_state(
                kc, vc, bc, gtc, gf, Ipf, init_mu, init_I, cu_seqlens,
                T, H, D_K, D_V, C, MAX_NC, perdk, num_bh, dev)

            _fp_output_kernel[(triton.cdiv(D_V, BV_out), MAX_NC, num_bh)](
                qc, kc, vc, gtc, gf, Ipf, M_bound, I_bound, Ibar, o, scale, cu_seqlens,
                T, H=H, D_K=D_K, D_V=D_V, BV=BV_out, C=C, MAX_NC=MAX_NC, PERDK=perdk)

            ctx.recompute_state = recompute_state
            if recompute_state:
                # Only the small per-token inputs (+ initial state, if any) cross the
                # boundary -- M_bound/I_bound/Ibar (the O(nc)-scaled trajectory) are
                # recomputed in .backward() instead, via the same _fp_compute_state call
                # used above. init_mu/init_I are small (B,H,D_V,D_K), not nc-scaled, and
                # carry no gradient (see the state-boundary note below) -- plain ctx
                # attributes rather than save_for_backward, same as scale/C/MAX_NC.
                ctx.save_for_backward(q, k, v, b, gt, gf, Ipf, cu_seqlens)
                ctx.init_mu, ctx.init_I = init_mu, init_I
            else:
                ctx.save_for_backward(q, k, v, b, gt, gf, Ipf, M_bound, I_bound, Ibar, cu_seqlens)
            ctx.scale, ctx.C, ctx.MAX_NC = scale, C, MAX_NC
            # Final chunk-exit state per sequence, exact (same M,I the carry recurrence
            # already materializes at each sequence's own chunk count -- no extra kernel).
            # Gradients are not propagated through the state boundary, matching exact
            # Palimpsa's own ChunkPalimpsa (returns None for initial_mu/I_state's grad
            # slot too): this is a truncated-BPTT boundary, not a differentiable input.
            if is_varlen:
                nc_per_seq = (seqlens + C - 1) // C
                num_seqs = cu_seqlens.numel() - 1
                idx = nc_per_seq.view(num_seqs, 1, 1, 1, 1).expand(-1, H, 1, D_V, D_K)
                final_M = M_bound.view(num_seqs, H, MAX_NC + 1, D_V, D_K).gather(2, idx).squeeze(2)
                final_I = I_bound.view(num_seqs, H, MAX_NC + 1, D_V, D_K).gather(2, idx).squeeze(2)
            else:
                final_M = M_bound[:, MAX_NC].reshape(B, H, D_V, D_K)
                final_I = I_bound[:, MAX_NC].reshape(B, H, D_V, D_K)
            # Exported recurrent state: upcast before dividing so the bf16 checkpoint
            # storage above doesn't degrade the state a caller resumes from.
            final_mu = final_M.float() / final_I.float()
            return o, final_mu, final_I.float()

        @staticmethod
        def backward(ctx, do, d_final_mu, d_final_I):
            if ctx.recompute_state:
                q, k, v, b, gt, g, Ip, cu_seqlens = ctx.saved_tensors
                B, T, H, D_K = q.shape
                D_V = v.shape[-1]
                dev = q.device
                num_bh = ((cu_seqlens.numel() - 1) * H if cu_seqlens is not None
                          else B * H)
                perdk = Ip.numel() == H * D_K
                M_bound, I_bound, Ibar = _fp_compute_state(
                    k.contiguous(), v.contiguous(), b.contiguous(), gt.contiguous(), g,
                    Ip, ctx.init_mu, ctx.init_I, cu_seqlens,
                    T, H, D_K, D_V, ctx.C, ctx.MAX_NC, perdk, num_bh, dev)
            else:
                q, k, v, b, gt, g, Ip, M_bound, I_bound, Ibar, cu_seqlens = ctx.saved_tensors
            dq, dk, dv, db, dgt = _fast_palimpsa_bwd_triton(
                do, q, k, v, b, gt, g, Ip, ctx.scale, ctx.C, M_bound, I_bound, Ibar,
                cu_seqlens=cu_seqlens, MAX_NC=ctx.MAX_NC)
            return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                    db.to(b.dtype), dgt.to(gt.dtype), None, None, None, None, None, None,
                    None, None)

    @torch.compiler.disable
    def chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C,
                            backend="triton", initial_mu_state=None, initial_I_state=None,
                            output_final_state=False, cu_seqlens=None,
                            recompute_state=False):
        """`recompute_state` (default False, `backend='triton'` only): save
        M_bound/I_bound/Ibar across the autograd boundary (current default), or
        True to re-derive them in `.backward()` from q/k/v/b/gt/g/Ip instead (fla's
        usual pattern for this class of kernel) -- lower peak memory, previously
        measured slower here (see the comment above `_fp_compute_state`'s call in
        `_FastPalimpsa.forward`); re-measure at your shape before trusting either
        direction, the old number predates this session's forward-kernel autotune
        work.
        """
        if scale is None:
            scale = q.shape[-1] ** -0.5
        D_K = q.shape[-1]
        D_V = v.shape[-1]
        # 'vec': chunk-parallel autograd path. General D_K/D_V, no SMEM limit,
        #        grads via autograd (correct by construction). Use this for training.
        # 'triton': the hand-written Triton fwd/bwd (currently blows SMEM at large D_V).
        if backend == "vec":
            if (initial_mu_state is not None or initial_I_state is not None
                    or output_final_state or cu_seqlens is not None):
                raise NotImplementedError(
                    "chunk_fast_palimpsa: initial/final state and cu_seqlens are only "
                    "implemented for backend='triton'.")
            return fast_palimpsa_vec(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
        # The clamp below exists only to keep the *backward* kernel inside shared
        # memory. Applying it when no backward will run makes the operator itself
        # device-dependent -- Fast Palimpsa is chunk-LOCAL, so C is part of the
        # function, not a tuning knob -- and an eval of the same checkpoint then
        # returns different NLL on an H100 (227 KiB SMEM, C=32 survives) than on a
        # Blackwell RTX PRO 6000 (99 KiB, C falls to 16). Under inference we skip it,
        # so a checkpoint reads the same everywhere.
        needs_bwd = torch.is_grad_enabled() and any(
            t.requires_grad for t in (q, k, v, b, gt))
        if q.is_cuda and needs_bwd:
            C_eff = _fp_clamp_C(D_K, D_V, chunk_size, q.device)
            if C_eff != chunk_size:
                import warnings
                warnings.warn(
                    f"chunk_fast_palimpsa: chunk_size={chunk_size} exceeds the backward "
                    f"shared-memory limit for D_K={D_K}, D_V={D_V} on this GPU; "
                    f"using chunk_size={C_eff}. Training will use C={C_eff}; note this "
                    f"changes the operator, so a checkpoint trained at C=32 must be "
                    f"evaluated at C=32.")
        else:
            C_eff = chunk_size

        if cu_seqlens is not None:
            # Ragged per-sequence chunks are handled natively inside the kernels
            # (grid padded to the longest sequence's chunk count, boundary-masked --
            # see _fp_state_kernel et al.), so no whole-tensor tail padding here.
            if q.shape[0] != 1:
                raise ValueError(
                    "chunk_fast_palimpsa: cu_seqlens requires the packed-batch "
                    f"convention (q.shape[0] == 1), got batch size {q.shape[0]}.")
            o, final_mu, final_I = _FastPalimpsa.apply(
                q, k, v, b, gt, g, Ip, scale, C_eff, initial_mu_state, initial_I_state,
                cu_seqlens, recompute_state)
            if output_final_state:
                return o, final_mu, final_I
            return o

        # T need not be a multiple of C_eff: pad the tail chunk with zeros rather than
        # coupling C_eff to T (that used to shrink the chunk size itself, silently
        # changing the operator -- exactly what the SMEM-clamp comment above says not to
        # do). A padded step has gt=0 (decay f=exp(0)=1, i.e. carries the boundary state
        # through unchanged) and k=v=b=0 (contributes nothing new), so it is an exact
        # no-op on every real position's output and on the state anything after it would
        # read -- not an approximation. Causality guarantees no leakage into real
        # positions either way. Verified in test_ragged_length: real-position outputs
        # are bit-identical whether the tail beyond T is zero-padding or arbitrary data.
        T = q.shape[1]
        pad = (-T) % C_eff
        if pad:
            def _pad(x):
                shape = list(x.shape)
                shape[1] = pad
                return torch.cat([x, x.new_zeros(shape)], dim=1)
            q, k, v, b, gt = (_pad(x) for x in (q, k, v, b, gt))

        o, final_mu, final_I = _FastPalimpsa.apply(
            q, k, v, b, gt, g, Ip, scale, C_eff, initial_mu_state, initial_I_state, None,
            recompute_state)
        if pad:
            o = o[:, :T]
        if output_final_state:
            return o, final_mu, final_I
        return o


# =============================================================================
# 3. Test: Triton fwd & bwd vs reference fwd & bwd
# =============================================================================
def test_fwd_bwd(B=2, T=64, H=3, D_K=48, D_V=40, C=16, seed=0, dtype=torch.float32):
    assert HAS_TRITON, "CUDA/Triton required for the Triton test."
    torch.manual_seed(seed)
    dev = "cuda"
    mk = lambda *s: torch.randn(*s, device=dev, dtype=dtype)
    q = mk(B, T, H, D_K).requires_grad_(True)
    k = torch.nn.functional.normalize(mk(B, T, H, D_K), dim=-1).detach().requires_grad_(True)
    v = mk(B, T, H, D_V).requires_grad_(True)
    b = (torch.rand(B, T, H, D_V, device=dev, dtype=dtype) * 1.5 + 0.1).requires_grad_(True)
    gt = (torch.rand(B, T, H, device=dev, dtype=dtype) * 0.1).requires_grad_(True)
    g = torch.rand(H, device=dev, dtype=dtype) * 0.5 + 0.5
    Ip = torch.rand(H, device=dev, dtype=dtype) + 0.5
    scale = D_K ** -0.5

    # --- forward ---
    o_tri = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C,
                                backend="triton")
    o_ref = fast_palimpsa_ref(q.detach(), k.detach(), v.detach(), b.detach(),
                              gt.detach(), g, Ip, scale=scale, chunk_size=C)
    fwd_err = (o_tri - o_ref).abs().max().item()
    print(f"[fwd] D_K={D_K} D_V={D_V}  max|triton-ref| = {fwd_err:.3e}")

    # --- backward ---
    do = torch.randn_like(o_tri)
    grads_tri = torch.autograd.grad(o_tri, (q, k, v, b, gt), do, retain_graph=False)

    q2, k2, v2, b2, gt2 = [t.detach().clone().requires_grad_(True) for t in (q, k, v, b, gt)]
    o_ref2 = fast_palimpsa_ref(q2, k2, v2, b2, gt2, g, Ip, scale=scale, chunk_size=C)
    grads_ref = torch.autograd.grad(o_ref2, (q2, k2, v2, b2, gt2), do)

    names = ["dq", "dk", "dv", "db", "dgt"]
    # vec backend matches the reference up to fp accumulation only; judge by rel err.
    fwd_rel = fwd_err / (o_ref.abs().max().item() + 1e-12)
    print(f"      (fwd rel err = {fwd_rel:.3e})")
    # Triton fp32 tl.dot (tensor cores) vs fp32 loop ref: ~1-3e-3 rel is expected.
    tol = 5e-3 if dtype == torch.float32 else 1e-5
    ok = fwd_rel < tol
    for n, gtri, gref in zip(names, grads_tri, grads_ref):
        e = (gtri - gref).abs().max().item()
        rel = e / (gref.abs().max().item() + 1e-12)
        print(f"[bwd] {n}: max abs err {e:.3e}  rel {rel:.3e}")
        ok = ok and rel < tol
    print("PASS" if ok else "FAIL")
    return ok


def test_ref_shapes():
    """CPU-only: exercises the reference across odd D_K/D_V and Ip layouts."""
    for (DK, DV) in [(128, 16), (48, 40), (33, 7), (64, 128)]:
        for ip_kind in ("scalar", "perdk"):
            torch.manual_seed(0)
            B, L, H, C = 1, 32, 2, 16
            q = torch.randn(B, L, H, DK, dtype=torch.float64)
            k = torch.nn.functional.normalize(torch.randn(B, L, H, DK, dtype=torch.float64), dim=-1)
            v = torch.randn(B, L, H, DV, dtype=torch.float64)
            b = torch.rand(B, L, H, DV, dtype=torch.float64) * 1.5 + 0.1
            gt = torch.rand(B, L, H, dtype=torch.float64) * 0.1
            g = torch.rand(H, dtype=torch.float64) * 0.5 + 0.5
            Ip = (torch.rand(H, dtype=torch.float64) + 0.5 if ip_kind == "scalar"
                  else torch.rand(H, DK, dtype=torch.float64) + 0.5)
            scale = DK ** -0.5
            y = fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C)
            assert y.shape == (B, L, H, DV)
            # TRUE invariant for frozen-carry mode: with NO carried history (a
            # single chunk, L == C) AND beta constant across D_V (so the isotropic
            # collapse is lossless), the approximation is EXACT. (With history or
            # anisotropic beta it is genuinely approximate -- that is the method.)
            qs, ks, vs, gts = q[:, :C], k[:, :C], v[:, :C], gt[:, :C]
            bs = b[:, :C].mean(-1, keepdim=True).expand(B, C, H, DV).contiguous()
            ye, _ = _exact_recurrence(qs, ks, vs, bs, gts, g, Ip, scale)
            ya = fast_palimpsa_ref(qs, ks, vs, bs, gts, g, Ip, scale=scale, chunk_size=C)
            err = (ya - ye).abs().max().item()
            print(f"DK={DK:3d} DV={DV:3d} Ip={ip_kind:6s}: shape ok, single-chunk const-beta exactness {err:.2e}")
            assert err < 1e-9, f"single-chunk const-beta should be exact, got {err}"


def test_varlen(H=3, D_K=48, D_V=40, C=16, seed=0, tol=5e-3):
    """cu_seqlens (packed, ragged multi-sequence) fwd+bwd vs fast_palimpsa_ref_varlen.

    Covers: lengths that are not multiples of C (ragged last chunk), a
    length-1 sequence (the extreme ragged case), and state export/import
    (output_final_state / initial_mu_state) across a CHUNK-ALIGNED split --
    the only split point this chunk-local operator can reproduce exactly,
    since a mid-chunk split changes which tokens share a local-attention
    chunk (verified: a non-chunk-aligned split legitimately differs from the
    continuous computation, matching how e.g. exact_palimpsa or GDN's own
    chunked kernels only resume cleanly at chunk boundaries too -- that is
    not tested here since it is not a correctness bug).
    """
    assert HAS_TRITON, "CUDA/Triton required for the Triton test."
    torch.manual_seed(seed)
    dev = "cuda"
    lens = [37, 16, 61, 3, 100, 17]
    total = sum(lens)
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                              dtype=torch.int32, device=dev)
    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32)
    q = mk(1, total, H, D_K).requires_grad_(True)
    k = torch.nn.functional.normalize(mk(1, total, H, D_K), dim=-1).detach().requires_grad_(True)
    v = mk(1, total, H, D_V).requires_grad_(True)
    b = (torch.rand(1, total, H, D_V, device=dev) * 1.5 + 0.1).requires_grad_(True)
    gt = (torch.rand(1, total, H, device=dev) * 0.1).requires_grad_(True)
    g = torch.rand(H, device=dev) * 0.5 + 0.5
    Ip = torch.rand(H, device=dev) + 0.5
    scale = D_K ** -0.5

    # --- forward, ragged multi-sequence ---
    o_tri = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C,
                                cu_seqlens=cu_seqlens)
    o_ref = fast_palimpsa_ref_varlen(q.detach(), k.detach(), v.detach(), b.detach(),
                                     gt.detach(), g, Ip, cu_seqlens, chunk_size=C)
    fwd_rel = (o_tri - o_ref).abs().max().item() / (o_ref.abs().max().item() + 1e-12)
    print(f"[varlen fwd] lens={lens}  rel err = {fwd_rel:.3e}")
    ok = fwd_rel < tol

    # --- backward, ragged multi-sequence ---
    do = torch.randn_like(o_tri)
    grads_tri = torch.autograd.grad(o_tri, (q, k, v, b, gt), do, retain_graph=False)
    q2, k2, v2, b2, gt2 = [t.detach().clone().requires_grad_(True) for t in (q, k, v, b, gt)]
    o_ref2 = fast_palimpsa_ref_varlen(q2, k2, v2, b2, gt2, g, Ip, cu_seqlens, chunk_size=C)
    grads_ref = torch.autograd.grad(o_ref2, (q2, k2, v2, b2, gt2), do)
    for n, gtri, gref in zip(["dq", "dk", "dv", "db", "dgt"], grads_tri, grads_ref):
        rel = (gtri - gref).abs().max().item() / (gref.abs().max().item() + 1e-12)
        print(f"[varlen bwd] {n}: rel {rel:.3e}")
        ok = ok and rel < tol

    # --- state export/import across a chunk-aligned split ---
    mids = [min((l // C) * C, l) for l in lens]
    keep, cu1, cu2 = [], [0], [0]
    parts1 = {n: [] for n in "qkvbg"}
    parts2 = {n: [] for n in "qkvbg"}
    tensors = {"q": q.detach(), "k": k.detach(), "v": v.detach(), "b": b.detach(), "g": gt.detach()}
    for i, l in enumerate(lens):
        m = mids[i]
        if m == 0 or m == l:
            continue
        keep.append(i)
        bos = int(cu_seqlens[i])
        for n, t in tensors.items():
            parts1[n].append(t[:, bos:bos + m])
            parts2[n].append(t[:, bos + m:bos + l])
        cu1.append(cu1[-1] + m)
        cu2.append(cu2[-1] + (l - m))
    cat = lambda parts: torch.cat(parts, dim=1)
    cu1 = torch.tensor(cu1, dtype=torch.int32, device=dev)
    cu2 = torch.tensor(cu2, dtype=torch.int32, device=dev)
    _, mid_mu, mid_I = chunk_fast_palimpsa(
        cat(parts1["q"]), cat(parts1["k"]), cat(parts1["v"]), cat(parts1["b"]), cat(parts1["g"]),
        g, Ip, scale=scale, chunk_size=C, cu_seqlens=cu1, output_final_state=True)
    o2 = chunk_fast_palimpsa(
        cat(parts2["q"]), cat(parts2["k"]), cat(parts2["v"]), cat(parts2["b"]), cat(parts2["g"]),
        g, Ip, scale=scale, chunk_size=C, cu_seqlens=cu2,
        initial_mu_state=mid_mu, initial_I_state=mid_I)
    state_ok = True
    for j, i in enumerate(keep):
        l, m, bos, bos2 = lens[i], mids[i], int(cu_seqlens[i]), int(cu2[j])
        seg_ref = o_ref[:, bos + m: bos + l]
        seg_chain = o2[:, bos2: bos2 + (l - m)]
        rel = (seg_chain - seg_ref).abs().max().item() / (seg_ref.abs().max().item() + 1e-12)
        state_ok = state_ok and rel < tol
    print(f"[varlen state chain, chunk-aligned split] {'ok' if state_ok else 'FAIL'}")
    ok = ok and state_ok

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    print("== reference shape/exactness sweep (CPU) ==")
    test_ref_shapes()
    if HAS_TRITON:
        print("\n== TRITON fwd/bwd vs reference (GPU) ==")
        print("   (test forces backend='triton'; BV-tiled kernels, any D_K/D_V)")
        for C_ in (16, 32):
            print(f"\n-- chunk_size C={C_} --")
            for (dk, dv) in [(128, 16), (48, 40), (64, 128), (33, 24), (96, 192)]:
                test_fwd_bwd(D_K=dk, D_V=dv, T=4 * C_, C=C_)
        # also exercise fp32 at the real training shape with a couple of chunks
        print("\n-- real shape sanity (D_K=96, D_V=192, longer seq) --")
        test_fwd_bwd(B=1, T=256, H=4, D_K=96, D_V=192, C=16)
        print("\n== TRITON varlen (cu_seqlens) fwd/bwd/state vs reference (GPU) ==")
        test_varlen()
    else:
        print("\n(no CUDA/Triton here -> ran reference checks only)")