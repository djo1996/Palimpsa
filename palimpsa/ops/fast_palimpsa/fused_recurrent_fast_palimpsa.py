# -*- coding: utf-8 -*-
# Fused recurrent kernel for Fast Palimpsa: exact Palimpsa's own recurrent kernel,
# reused directly, for any beta shape (scalar-per-head or vector-per-channel).
#
# Fast Palimpsa's chunked kernel carries state ACROSS chunk boundaries exactly: the
# boundary state update (chunk_fast_palimpsa.py's M_c/I_c recursion) uses the full
# per-channel beta (`bc`, not a D_V-collapsed `betabar`), so M_c/I_c at any chunk
# boundary is bit-for-bit the same (D_V,D_K) state exact Palimpsa's own token-by-token
# recursion would have at that same position. Only the WITHIN-chunk *read* (the
# isotropic Ibar machinery) approximates -- and that machinery exists purely to avoid
# an O(chunk_size x D_V x D_K) cost when processing several tokens per chunk at once.
# It buys nothing when generating one token at a time: there is no "next chunk" to
# amortize over, so continuing from an exact boundary state via exact Palimpsa's own
# per-token recursion is not a different operator bolted on -- it is what Fast
# Palimpsa's own chunk-boundary math already computes, just without the now-pointless
# within-chunk shortcut.

from __future__ import annotations

from palimpsa.ops.palimpsa.fused_recurrent_palimpsa import fused_recurrent_palimpsa


def fused_recurrent_fast_palimpsa(
    q, k, v, b, gt, g, Ip, scale=None,
    initial_mu_state=None, initial_I_state=None,
    output_final_state=False, output_uncertainty=False,
    cu_seqlens=None,
):
    """Fast Palimpsa's fused recurrent kernel: exact Palimpsa's own recurrent
    kernel, reused directly -- see the module header for why that is correct
    (not an approximation swap) rather than a workaround.
    """
    return fused_recurrent_palimpsa(
        q, k, v, b, gt, g, Ip, scale=scale,
        initial_mu_state=initial_mu_state, initial_I_state=initial_I_state,
        output_final_state=output_final_state, output_uncertainty=output_uncertainty,
        cu_seqlens=cu_seqlens,
    )


__all__ = ['fused_recurrent_fast_palimpsa']
