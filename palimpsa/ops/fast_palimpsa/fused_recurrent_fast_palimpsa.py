# -*- coding: utf-8 -*-
# Fused recurrent kernel for Fast Palimpsa.
#
# Fast Palimpsa's chunked kernel approximates the read of *within-chunk* history with
# an isotropic-in-D_V precision Ibar = I_c.mean(dim=D_V) (see chunk_fast_palimpsa.py's
# header). At a chunk size of 1, "within-chunk" history is empty, so there is nothing to
# collapse across value channels for: the isotropic read and the exact per-(D_V,D_K) read
# of the token-exact recurrence coincide, PROVIDED the precision state stays isotropic
# across D_V at every step. That holds by induction whenever the beta gate `b` is
# scalar-per-head (uniform across D_V): starting from I_0 = Ip (isotropic by
# construction), I_t = f*I_{t-1} + (1-f)*Ip + b*k_t^2 stays isotropic at every t because
# every term on the right is isotropic. Verified numerically (test_fused_recurrent_fast,
# 2026-08-28): chunk_fast_palimpsa's own C=1 algorithm matches the token-exact recurrence
# to 3.5e-7 (fp32 floor) with scalar beta.
#
# `fused_recurrent_palimpsa` (exact Palimpsa's recurrent kernel) is unconditionally exact
# regardless of beta's shape -- it never performs Fast Palimpsa's isotropic collapse at
# all. That is NOT "the same approximation the chunked kernel uses" when beta varies
# across D_V (`_uses_vector_beta=True` on the Palimpsa layer, the default whenever
# kernel='fast' and metaplasticity=True): the same numerical test shows Fast Palimpsa's
# own chunked algorithm at C=1 disagrees with the token-exact recurrence by 0.28 (max
# abs, not noise) once beta varies across D_V. Reusing the exact recurrent kernel there
# would silently swap in different, MORE exact numerics than whatever a vector-beta Fast
# Palimpsa checkpoint was trained under -- a real train/inference mismatch, not a
# matching approximation. So this wrapper only reuses fused_recurrent_palimpsa when beta
# is (numerically) scalar-per-head; vector beta is refused rather than silently wrong,
# mirroring how chunk_fast_palimpsa itself refuses cu_seqlens/use_cache. A genuine
# isotropic-collapsing recurrent kernel for the vector-beta case does not exist yet.
#
# No separate Triton kernel is written here for the scalar-beta case: it is a thin
# re-export of exact Palimpsa's own recurrent kernel.

from __future__ import annotations

import torch

from palimpsa.ops.palimpsa.fused_recurrent_palimpsa import fused_recurrent_palimpsa

_SCALAR_BETA_ATOL = 1e-6


def _is_scalar_beta(b: torch.Tensor) -> bool:
    if b.shape[-1] == 1:
        return True
    return bool((b - b[..., :1]).abs().max() <= _SCALAR_BETA_ATOL)


def fused_recurrent_fast_palimpsa(
    q, k, v, b, gt, g, Ip, scale=None,
    initial_mu_state=None, initial_I_state=None,
    output_final_state=False, output_uncertainty=False,
    cu_seqlens=None,
):
    """Fast Palimpsa's fused recurrent kernel.

    Exact reuse of exact Palimpsa's recurrent kernel, valid only where the two
    algorithms provably coincide: `b` (beta) uniform across D_V. Raises otherwise --
    see the module header for why vector beta cannot use this path.
    """
    if torch.is_tensor(b) and b.dim() > 0 and not _is_scalar_beta(b):
        raise ValueError(
            "fused_recurrent_fast_palimpsa: beta varies across D_V (vector beta). Fast "
            "Palimpsa's chunked kernel and the token-exact recurrence provably coincide "
            "only for scalar-per-head beta (see this module's header) -- reusing exact "
            "Palimpsa's recurrent kernel here would silently score a vector-beta "
            "checkpoint with different numerics than it was trained under. Use "
            "kernel='exact' for recurrent/cached generation with vector beta instead."
        )
    return fused_recurrent_palimpsa(
        q, k, v, b, gt, g, Ip, scale=scale,
        initial_mu_state=initial_mu_state, initial_I_state=initial_I_state,
        output_final_state=output_final_state, output_uncertainty=output_uncertainty,
        cu_seqlens=cu_seqlens,
    )


__all__ = ['fused_recurrent_fast_palimpsa']
