/* conv/winograd.c — Winograd F(2x2, 3x3) forward kernel.

   k.2 split: extracted from src/core/conv.c. winograd is one of the
   six conv paths the dispatcher in forward.c picks between (winograd,
   direct 3x3, direct 3x3 stride-2, small-Cin direct, implicit gemm,
   im2col+gemm). it lives in its own tu because the F(2,3) algorithm
   is an algebraic transform — input → 4x4 patch * B^T·B basis,
   weights → 4x4 G·G^T basis, do 16 small GEMMs, then output → 2x2
   via A^T·A — that has no shared state with the other paths.

   F(2,3) means "F(output tile, kernel)" = 2x2 output from 3x3 kernel
   using a 4x4 input tile. theoretical 2.25x speedup on multiply count
   (16 muls per tile vs 36 for direct), wins when Cin/Cout are large
   enough to amortise the input/kernel/output transforms (~32+) and
   spatial output is at least 4x4. only handles 3x3 stride=1 pad=1 —
   F(2,3) doesn't apply to 1x1, 5x5, 7x7 (different variants), nor
   stride=2 (different algo).

   the V buffer ([16, C_in, num_tiles]) caps shape eligibility: at
   large num_tiles the per-tile scatter spans Cin*num_tiles*sizeof(float)
   of memory per ij stride, thrashing all cache levels. ax_conv_prefer_
   winograd_f23 enforces a ~32 MB slot cap based on measured wins/losses. */

#include "internal.h"
#include "axiom/compute.h"
#include "../../compute/backends/simd_defs.h"
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

bool ax_conv_prefer_winograd_f23(int kh, int kw, int sh, int sw, int ph, int pw,
                                  int64_t N, int64_t C_in, int64_t C_out,
                                  int64_t out_h, int64_t out_w)
{
    (void)C_out;
    if (!(kh == 3 && kw == 3 && sh == 1 && sw == 1 && ph == 1 && pw == 1)) return false;
    if (C_in < 32 || C_out < 32) return false;
    if (out_h < 4 || out_w < 4) return false;
    /* V buffer size: 16 * C_in * num_tiles floats, with num_tiles = N * Ty * Tx.
       cap at ~32 MB worth of slots (8M floats) so the input-transform scatter
       stays manageable. measured: V=51 MB wins +38%, V=102 MB wins +15%, but
       V=205 MB loses -6% and V=411 MB loses -34% as cache thrashing dominates. */
    int64_t Ty = (out_h + 1) / 2;
    int64_t Tx = (out_w + 1) / 2;
    int64_t v_slots = 16 * C_in * N * Ty * Tx;
    if (v_slots > (int64_t)(32 * 1024 * 1024)) return false;
    return true;
}

/* winograd F(2,3) transform matrices (precomputed constants):
     B^T (input transform, 4x4):       G (kernel transform, 4x3):    A^T (output xform, 2x4):
       [ 1,  0, -1,  0]                 [  1,    0,    0 ]            [ 1,  1,  1,  0]
       [ 0,  1,  1,  0]                 [ 1/2,  1/2,  1/2]            [ 0,  1, -1, -1]
       [ 0, -1,  1,  0]                 [ 1/2, -1/2,  1/2]
       [ 0,  1,  0, -1]                 [  0,    0,    1 ]
   each transform is small enough that the constants are unrolled below as
   straight-line code (no matrix multiply at runtime). */

void ax_conv_wino_input_transform_tile(const float *d, float *v)
{
    /* t = B^T * d (4 rows × 4 cols) */
    float t[16];
    for (int j = 0; j < 4; j++) {
        const float d0j = d[0*4+j], d1j = d[1*4+j], d2j = d[2*4+j], d3j = d[3*4+j];
        t[0*4+j] =  d0j - d2j;
        t[1*4+j] =  d1j + d2j;
        t[2*4+j] = -d1j + d2j;
        t[3*4+j] =  d1j - d3j;
    }
    /* v = t * B (B = (B^T)^T). same row pattern applied per row of t. */
    for (int i = 0; i < 4; i++) {
        const float t0 = t[i*4+0], t1 = t[i*4+1], t2 = t[i*4+2], t3 = t[i*4+3];
        v[i*4+0] =  t0 - t2;
        v[i*4+1] =  t1 + t2;
        v[i*4+2] = -t1 + t2;
        v[i*4+3] =  t1 - t3;
    }
}

void ax_conv_wino_kernel_transform_filter(const float *g, float *u)
{
    /* t = G * g (4 rows × 3 cols) */
    float t[12];
    for (int j = 0; j < 3; j++) {
        const float g0j = g[0*3+j], g1j = g[1*3+j], g2j = g[2*3+j];
        t[0*3+j] = g0j;
        t[1*3+j] = 0.5f * ( g0j + g1j + g2j);
        t[2*3+j] = 0.5f * ( g0j - g1j + g2j);
        t[3*3+j] = g2j;
    }
    /* u = t * G^T (4 rows × 4 cols) */
    for (int i = 0; i < 4; i++) {
        const float t0 = t[i*3+0], t1 = t[i*3+1], t2 = t[i*3+2];
        u[i*4+0] = t0;
        u[i*4+1] = 0.5f * ( t0 + t1 + t2);
        u[i*4+2] = 0.5f * ( t0 - t1 + t2);
        u[i*4+3] = t2;
    }
}

void ax_conv_wino_output_transform_tile(const float *m, float *y)
{
    /* t = A^T * m (2 rows × 4 cols) */
    float t[8];
    for (int j = 0; j < 4; j++) {
        const float m0j = m[0*4+j], m1j = m[1*4+j], m2j = m[2*4+j], m3j = m[3*4+j];
        t[0*4+j] = m0j + m1j + m2j;
        t[1*4+j] = m1j - m2j - m3j;
    }
    /* y = t * A (2 rows × 2 cols) */
    for (int i = 0; i < 2; i++) {
        const float t0 = t[i*4+0], t1 = t[i*4+1], t2 = t[i*4+2], t3 = t[i*4+3];
        y[i*2+0] = t0 + t1 + t2;
        y[i*2+1] = t1 - t2 - t3;
    }
}

/* winograd F(2,3) forward kernel for the whole batch.
   layout of scratch:
     U (kernel transformed): [16, C_out, C_in], filled fresh from current weights
       (weights change every train step; cheap relative to input transform).
     V (input transformed):  [16, C_in, num_tiles] where num_tiles = N * Ty * Tx,
       Ty = ceil(out_h/2), Tx = ceil(out_w/2).
     M (gemm output):        [16, C_out, num_tiles]
   pipeline:
     1. transform kernel:    G * g[co,ci] * G^T  → U[*,*, co, ci]   (small matmul × 16 entries)
     2. transform input:     B^T * d[n,ci, ty*2-1:+3, tx*2-1:+3] * B → V[*,*, ci, tile]
     3. 16 gemms:            U[ij,*,*] @ V[ij,*,*]  → M[ij,*,*]
     4. transform output:    A^T * m[*,*, co, tile] * A → 2x2 output, plus bias.
   bounds: tiles at the right/bottom edge may map a 2x2 output that extends
   past out_h/out_w; the writeback masks those positions. */
int ax_conv_winograd_f23_forward(
    struct ax_conv_scratch *s,
    const float *id,    /* [N, C_in, H, W] */
    const float *wd,    /* [C_out, C_in, 3, 3] */
    const float *bias,  /* [C_out] or NULL */
    float *od,          /* [N, C_out, out_h, out_w] */
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t out_h, int64_t out_w, int T)
{
    (void)T;
    const int64_t Ty = (out_h + 1) / 2;
    const int64_t Tx = (out_w + 1) / 2;
    const int64_t num_tiles = N * Ty * Tx;
    if (num_tiles != s->wino_num_tiles || !s->wino_U || !s->wino_V || !s->wino_M)
        return -1;  /* scratch shape mismatch — caller falls back */

    float *U = (float *)s->wino_U->storage->data;  /* [16, C_out, C_in] */
    float *V = (float *)s->wino_V->storage->data;  /* [16, C_in, num_tiles] */
    float *M = (float *)s->wino_M->storage->data;  /* [16, C_out, num_tiles] */

    const int64_t U_stride_ij = C_out * C_in;
    const int64_t V_stride_ij = C_in * num_tiles;
    const int64_t M_stride_ij = C_out * num_tiles;

    /* 1. kernel transform: U[ij, co, ci] from weight[co, ci, 3x3].
       parallel over (co, ci); each filter is 9 inputs → 16 outputs. */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(static) collapse(2)
    #endif
    for (int64_t co = 0; co < C_out; co++) {
        for (int64_t ci = 0; ci < C_in; ci++) {
            float u[16];
            ax_conv_wino_kernel_transform_filter(wd + (co * C_in + ci) * 9, u);
            for (int ij = 0; ij < 16; ij++)
                U[ij * U_stride_ij + co * C_in + ci] = u[ij];
        }
    }

    /* 2. input transform: V[ij, ci, tile] from input 4x4 patch.
       tile (n, ty, tx) → input rows [ty*2-1 : ty*2+3], cols [tx*2-1 : tx*2+3].
       pad=1 means out-of-bound reads are zero. parallel over (n, ci, ty, tx). */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(static) collapse(3)
    #endif
    for (int64_t n = 0; n < N; n++) {
        for (int64_t ci = 0; ci < C_in; ci++) {
            for (int64_t ty = 0; ty < Ty; ty++) {
                const float *ic_plane = id + (n * C_in + ci) * H * W;
                for (int64_t tx = 0; tx < Tx; tx++) {
                    float d[16];
                    /* extract 4x4 patch with implicit zero-pad */
                    for (int di = 0; di < 4; di++) {
                        const int64_t ih = ty * 2 + di - 1;  /* pad=1 → -1 origin */
                        if (ih < 0 || ih >= H) {
                            d[di*4+0] = d[di*4+1] = d[di*4+2] = d[di*4+3] = 0.0f;
                            continue;
                        }
                        const float *irow = ic_plane + ih * W;
                        for (int dj = 0; dj < 4; dj++) {
                            const int64_t iw = tx * 2 + dj - 1;
                            d[di*4+dj] = (iw >= 0 && iw < W) ? irow[iw] : 0.0f;
                        }
                    }
                    float v[16];
                    ax_conv_wino_input_transform_tile(d, v);
                    /* scatter v[ij] to V[ij, ci, tile_idx] */
                    const int64_t tile_idx = (n * Ty + ty) * Tx + tx;
                    for (int ij = 0; ij < 16; ij++)
                        V[ij * V_stride_ij + ci * num_tiles + tile_idx] = v[ij];
                }
            }
        }
    }

    /* 3. 16 GEMMs: M[ij] = U[ij] @ V[ij] using stack tensor views.
       each gemm is [C_out, C_in] @ [C_in, num_tiles] → [C_out, num_tiles].
       opt_gemm zero-fills the output internally; no per-call memset. */
    for (int ij = 0; ij < 16; ij++) {
        ax_storage_t a_st, b_st, c_st;
        ax_tensor_t  a_tv, b_tv, c_tv;
        ax_conv_make_stack_view(&a_tv, &a_st, U + ij * U_stride_ij, C_out, C_in);
        ax_conv_make_stack_view(&b_tv, &b_st, V + ij * V_stride_ij, C_in,  num_tiles);
        ax_conv_make_stack_view(&c_tv, &c_st, M + ij * M_stride_ij, C_out, num_tiles);
        ax_compute_gemm(&a_tv, &b_tv, &c_tv);
    }

    /* 4. output transform: y = A^T * m_tile * A → 2x2 + bias, write to od.
       parallel over (n, co, ty, tx); each tile writes a disjoint 2x2 region.
       last-row/last-col tiles may have ty*2+1 == out_h or tx*2+1 == out_w
       (i.e. the 2x2 falls partly outside the valid output) — masked below. */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(static) collapse(3)
    #endif
    for (int64_t n = 0; n < N; n++) {
        for (int64_t co = 0; co < C_out; co++) {
            for (int64_t ty = 0; ty < Ty; ty++) {
                const float bias_val = bias ? bias[co] : 0.0f;
                float *oc_plane = od + (n * C_out + co) * out_h * out_w;
                for (int64_t tx = 0; tx < Tx; tx++) {
                    float m[16];
                    const int64_t tile_idx = (n * Ty + ty) * Tx + tx;
                    for (int ij = 0; ij < 16; ij++)
                        m[ij] = M[ij * M_stride_ij + co * num_tiles + tile_idx];
                    float y[4];
                    ax_conv_wino_output_transform_tile(m, y);
                    /* writeback with bounds check (last partial tile). */
                    for (int oi = 0; oi < 2; oi++) {
                        const int64_t oh = ty * 2 + oi;
                        if (oh >= out_h) break;
                        for (int oj = 0; oj < 2; oj++) {
                            const int64_t ow = tx * 2 + oj;
                            if (ow >= out_w) break;
                            oc_plane[oh * out_w + ow] = y[oi*2+oj] + bias_val;
                        }
                    }
                }
            }
        }
    }

    return 0;
}
