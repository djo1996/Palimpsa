# -*- coding: utf-8 -*-
# Fused recurrent kernel for Fast Palimpsa.
#
# This is `fast_palimpsa_vec` called with `chunk_size=1` -- Fast Palimpsa's own
# isotropic-in-D_V approximation at its smallest, most-exact granularity, NOT a
# reuse of exact Palimpsa's unrelated recurrence. An earlier version of this file
# reused `fused_recurrent_palimpsa` (exact Palimpsa's kernel) for scalar-per-head
# beta and refused vector beta outright. That was wrong to ship: at C=1 there is
# still a real, provable, and previously-measured (0.28 max abs, not noise)
# difference between Fast Palimpsa's own math and the token-exact recurrence
# whenever beta varies across D_V -- see `fast_palimpsa_vec`'s docstring on why
# chunk_size is part of the operator, not just a tiling knob. Falling back to a
# genuinely different operator for that case was a worse train/inference mismatch
# than just using Fast Palimpsa's own C=1 limit, which is the closest
# self-consistent approximation this operator family actually has to a streaming
# form -- the same way any chunked-kernel model's decode path is never bit-exact
# to its training-time chunk size, by construction, not a special case here.
#
# For scalar-per-head beta specifically, C=1 *is* exactly the token-exact
# recurrence (verified in `test_fused_recurrent_fast`, matches to 3.5e-7, the
# fp32 floor) -- so nothing is lost there, and vector beta now works too instead
# of being refused.

from __future__ import annotations

from palimpsa.ops.fast_palimpsa.chunk_fast_palimpsa import fast_palimpsa_vec


def fused_recurrent_fast_palimpsa(
    q, k, v, b, gt, g, Ip, scale=None,
    initial_mu_state=None, initial_I_state=None,
    output_final_state=False, output_uncertainty=False,
    cu_seqlens=None,
):
    """Fast Palimpsa's fused recurrent kernel: `fast_palimpsa_vec(chunk_size=1)`.

    Works for any beta shape (scalar-per-head or vector-per-channel) -- unlike
    exact Palimpsa's recurrence, this is not a separate implementation to keep in
    sync, it is the same chunked math this package's training-time kernel uses,
    just at C=1. See the module header for why that is the right choice, not a
    workaround.
    """
    if cu_seqlens is not None:
        raise NotImplementedError(
            "fused_recurrent_fast_palimpsa: cu_seqlens (varlen packing) is not "
            "implemented yet -- fast_palimpsa_vec has no padding-free varlen path "
            "(see fast_palimpsa_ref_varlen's padding-based approach for the "
            "chunked kernel's own varlen story, not yet ported to the C=1 "
            "recurrent case)."
        )
    if output_uncertainty:
        raise NotImplementedError(
            "fused_recurrent_fast_palimpsa: output_uncertainty is not implemented."
        )
    return fast_palimpsa_vec(
        q, k, v, b, gt, g, Ip, scale=scale, chunk_size=1,
        initial_mu_state=initial_mu_state, initial_I_state=initial_I_state,
        output_final_state=output_final_state,
    )


__all__ = ['fused_recurrent_fast_palimpsa']
