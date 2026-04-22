/* conv/path_selection.c — shape-aware path-selection heuristics for conv2d.

   k.2 split: extracted from src/core/conv.c. forward.c picks one of
   six conv paths per call (winograd, direct 3x3 s1, direct 3x3 s2,
   direct small-Cin, implicit gemm, im2col+gemm). most of the
   per-path predicates live next to their kernel — can_direct_3x3
   in direct.c, prefer_winograd_f23 in winograd.c. only the implicit
   GEMM predicate doesn't have a natural home (the implicit-gemm
   kernel itself lives in the compute backend, not here), so it
   gets its own tiny tu rather than being shoehorned into one of the
   above. */

#include "internal.h"
#include <stdint.h>

/* implicit gemm: useful when K is large (typically C_in >= 64 with 3x3+ kernels)
   AND M is large enough that the gemm dominates over the gather overhead. */
/* implicit GEMM (gather-on-the-fly into pack_b) wins when:
   (a) both K and M are large enough to amortize the per-element gather
       overhead (the original heuristic), OR
   (b) explicit im2col would materialize >8 MB per sample and K is at least
       moderate (≥512). avoiding the full im2col write+read saves ~2× the
       buffer's bytes of memory traffic; for cbr_64x112x112_128x3 this is
       ~58 MB per sample (28.9 MB im2col write + 28.9 MB pack_b read of
       the same data). K threshold of 512 keeps gather overhead ≤ ~25%
       of inner loop time. */
bool ax_conv_prefer_implicit_gemm(int64_t K, int64_t M)
{
    /* implicit gemm gathers im2col patches on-the-fly inside pack_b,
       avoiding the large materialized [K, M] column buffer. wins when
       the buffer would exceed L3 (or a meaningful chunk of it) so the
       explicit im2col write+read pair doubles bandwidth.
       per-sample im2col bytes = K*M*4. 6 MB is above L1+L2 on typical
       desktops, low enough to catch shapes like 576*3136*4 = 7.2 MB
       (the conv_64x112_128_s2 case). */
    if (K >= 1024 && M >= 256) return true;
    int64_t im2col_bytes = K * M * (int64_t)sizeof(float);
    return im2col_bytes > (int64_t)(6 * 1024 * 1024) && K >= 256 && M >= 256;
}
