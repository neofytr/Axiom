/* conv/winograd.c — Winograd F(2x2, 3x3) forward + backward-dX kernels.

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

   perf-critical layout choices:
   * forward and backward run inside a single omp parallel region
     (one fork-join vs the naive 4+16 = 20 fork-joins).
   * input/output transforms scatter/gather through a tile-batched
     temp buffer (WINO_TB tiles at a time, ~4 KB in L1) so each V[ij]
     plane gets a contiguous memcpy instead of per-tile strided writes
     that thrash L2.
   * the 16 GEMMs are distributed across threads via omp-for rather
     than running 16 sequential multi-threaded GEMMs.

   backward dX: dX = conv(grad_out, W_rot180, s=1, p=1). since this
   is a standard 3x3 s1 p1 convolution, the same winograd F(2,3)
   pipeline applies. the kernel flip is folded into the transform
   (gf[i] = g[8-i]) with no separate buffer. the forward's wino_U/V/M
   scratch buffers are reused with swapped roles. */

#include "internal.h"
#include "axiom/compute.h"
#include "../../compute/backends/simd_defs.h"
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* tile batch size for scatter/gather: temp[16][WINO_TB] = 4 KB, fits L1 */
#define WINO_TB 64

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

/* winograd F(2,3) forward: single omp parallel region, tile-batched
   scatter/gather, parallel GEMM distribution across 16 ij values. */
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

#ifdef _OPENMP
    #pragma omp parallel
    {
#endif

    /* step 1: kernel transform — U[ij, co, ci] from weight[co, ci, 3x3].
       parallel over flat (co * C_in + ci) index. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t idx = 0; idx < C_out * C_in; idx++) {
        int64_t co = idx / C_in, ci = idx % C_in;
        float u[16];
        ax_conv_wino_kernel_transform_filter(wd + (co * C_in + ci) * 9, u);
        for (int ij = 0; ij < 16; ij++)
            U[ij * U_stride_ij + co * C_in + ci] = u[ij];
    }

    /* step 2: input transform — V[ij, ci, tile] from input 4x4 patch.
       tile-batched scatter: compute WINO_TB tiles into a 4KB L1-resident
       temp, then memcpy each ij plane contiguously. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_in * Ty; flat++) {
        int64_t n  = flat / (C_in * Ty);
        int64_t ci = (flat / Ty) % C_in;
        int64_t ty = flat % Ty;
        const float *ic_plane = id + (n * C_in + ci) * H * W;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO_TB) {
            int ntx = (int)((tx0 + WINO_TB <= Tx) ? WINO_TB : Tx - tx0);
            float tmp[16][WINO_TB];

            for (int dt = 0; dt < ntx; dt++) {
                int64_t tx = tx0 + dt;
                float d[16];
                for (int di = 0; di < 4; di++) {
                    int64_t ih = ty * 2 + di - 1;
                    if (ih < 0 || ih >= H) {
                        d[di*4] = d[di*4+1] = d[di*4+2] = d[di*4+3] = 0.0f;
                        continue;
                    }
                    const float *irow = ic_plane + ih * W;
                    for (int dj = 0; dj < 4; dj++) {
                        int64_t iw = tx * 2 + dj - 1;
                        d[di*4+dj] = (iw >= 0 && iw < W) ? irow[iw] : 0.0f;
                    }
                }
                float v[16];
                ax_conv_wino_input_transform_tile(d, v);
                for (int ij = 0; ij < 16; ij++)
                    tmp[ij][dt] = v[ij];
            }

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 16; ij++)
                memcpy(V + ij * V_stride_ij + ci * num_tiles + tb,
                       tmp[ij], (size_t)ntx * sizeof(float));
        }
    }

    /* step 3: 16 GEMMs distributed across threads — each thread runs
       single-threaded GEMMs for its assigned ij values (omp_in_parallel
       true → GEMM uses 1 thread). 8 threads × 2 ij = 16 GEMMs with
       8-way concurrency, zero extra fork-joins. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int ij = 0; ij < 16; ij++) {
        ax_storage_t a_st, b_st, c_st;
        ax_tensor_t  a_tv, b_tv, c_tv;
        ax_conv_make_stack_view(&a_tv, &a_st, U + ij * U_stride_ij, C_out, C_in);
        ax_conv_make_stack_view(&b_tv, &b_st, V + ij * V_stride_ij, C_in,  num_tiles);
        ax_conv_make_stack_view(&c_tv, &c_st, M + ij * M_stride_ij, C_out, num_tiles);
        ax_compute_gemm(&a_tv, &b_tv, &c_tv);
    }

    /* step 4: output transform — y = A^T * m_tile * A → 2x2 + bias.
       tile-batched gather: memcpy WINO_TB tiles from each M[ij] plane
       into a 4KB temp, transform, writeback. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_out * Ty; flat++) {
        int64_t n  = flat / (C_out * Ty);
        int64_t co = (flat / Ty) % C_out;
        int64_t ty = flat % Ty;
        const float bias_val = bias ? bias[co] : 0.0f;
        float *oc_plane = od + (n * C_out + co) * out_h * out_w;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO_TB) {
            int ntx = (int)((tx0 + WINO_TB <= Tx) ? WINO_TB : Tx - tx0);
            float tmp[16][WINO_TB];

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 16; ij++)
                memcpy(tmp[ij], M + ij * M_stride_ij + co * num_tiles + tb,
                       (size_t)ntx * sizeof(float));

            for (int dt = 0; dt < ntx; dt++) {
                float m[16];
                for (int ij = 0; ij < 16; ij++)
                    m[ij] = tmp[ij][dt];
                float y[4];
                ax_conv_wino_output_transform_tile(m, y);
                int64_t tx = tx0 + dt;
                for (int oi = 0; oi < 2; oi++) {
                    int64_t oh = ty * 2 + oi;
                    if (oh >= out_h) break;
                    for (int oj = 0; oj < 2; oj++) {
                        int64_t ow = tx * 2 + oj;
                        if (ow >= out_w) break;
                        oc_plane[oh * out_w + ow] = y[oi*2+oj] + bias_val;
                    }
                }
            }
        }
    }

#ifdef _OPENMP
    }
#endif

    return 0;
}

/* winograd F(2,3) backward dX: dX = conv(grad_out, W_rot180, s=1, p=1).
   reuses forward's wino_U/V/M buffers with swapped roles:
     forward U [16, C_out, C_in]     → backward U (same total, reinterpreted as [16, C_in, C_out])
     forward M [16, C_out, num_tiles] → backward V (same total)
     forward V [16, C_in, num_tiles]  → backward M (same total)
   kernel flip is folded into the transform: gf[i] = g[8-i]. */
int ax_conv_winograd_f23_backward_dx(
    struct ax_conv_scratch *s,
    const float *grad_out,  /* [N, C_out, out_h, out_w] */
    const float *wd,        /* [C_out, C_in, 3, 3] original weight */
    float *dx,              /* [N, C_in, H, W] — accumulating */
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t out_h, int64_t out_w, int T)
{
    (void)T;
    const int64_t Ty = (H + 1) / 2;
    const int64_t Tx = (W + 1) / 2;
    const int64_t num_tiles = N * Ty * Tx;
    if (num_tiles != s->wino_num_tiles || !s->wino_U || !s->wino_V || !s->wino_M)
        return -1;

    float *bU = (float *)s->wino_U->storage->data;  /* [16, C_in, C_out] */
    float *bV = (float *)s->wino_M->storage->data;  /* [16, C_out, num_tiles] */
    float *bM = (float *)s->wino_V->storage->data;  /* [16, C_in, num_tiles] */

    const int64_t bU_stride = C_in * C_out;
    const int64_t bV_stride = C_out * num_tiles;
    const int64_t bM_stride = C_in * num_tiles;

#ifdef _OPENMP
    #pragma omp parallel
    {
#endif

    /* step 1: flipped kernel transform.
       W_rot180[ci, co, r, s] = W[co, ci, 2-r, 2-s] → gf[i] = g[8-i]. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t idx = 0; idx < C_in * C_out; idx++) {
        int64_t ci = idx / C_out, co = idx % C_out;
        const float *g = wd + (co * C_in + ci) * 9;
        float gf[9];
        for (int i = 0; i < 9; i++) gf[i] = g[8 - i];
        float u[16];
        ax_conv_wino_kernel_transform_filter(gf, u);
        for (int ij = 0; ij < 16; ij++)
            bU[ij * bU_stride + ci * C_out + co] = u[ij];
    }

    /* step 2: input transform of grad_out [N, C_out, out_h, out_w]. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_out * Ty; flat++) {
        int64_t n  = flat / (C_out * Ty);
        int64_t co = (flat / Ty) % C_out;
        int64_t ty = flat % Ty;
        const float *gc_plane = grad_out + (n * C_out + co) * out_h * out_w;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO_TB) {
            int ntx = (int)((tx0 + WINO_TB <= Tx) ? WINO_TB : Tx - tx0);
            float tmp[16][WINO_TB];

            for (int dt = 0; dt < ntx; dt++) {
                int64_t tx = tx0 + dt;
                float d[16];
                for (int di = 0; di < 4; di++) {
                    int64_t ih = ty * 2 + di - 1;
                    if (ih < 0 || ih >= out_h) {
                        d[di*4] = d[di*4+1] = d[di*4+2] = d[di*4+3] = 0.0f;
                        continue;
                    }
                    const float *grow = gc_plane + ih * out_w;
                    for (int dj = 0; dj < 4; dj++) {
                        int64_t iw = tx * 2 + dj - 1;
                        d[di*4+dj] = (iw >= 0 && iw < out_w) ? grow[iw] : 0.0f;
                    }
                }
                float v[16];
                ax_conv_wino_input_transform_tile(d, v);
                for (int ij = 0; ij < 16; ij++)
                    tmp[ij][dt] = v[ij];
            }

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 16; ij++)
                memcpy(bV + ij * bV_stride + co * num_tiles + tb,
                       tmp[ij], (size_t)ntx * sizeof(float));
        }
    }

    /* step 3: 16 GEMMs — [C_in, C_out] @ [C_out, num_tiles] → [C_in, num_tiles]. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int ij = 0; ij < 16; ij++) {
        ax_storage_t a_st, b_st, c_st;
        ax_tensor_t  a_tv, b_tv, c_tv;
        ax_conv_make_stack_view(&a_tv, &a_st, bU + ij * bU_stride, C_in,  C_out);
        ax_conv_make_stack_view(&b_tv, &b_st, bV + ij * bV_stride, C_out, num_tiles);
        ax_conv_make_stack_view(&c_tv, &c_st, bM + ij * bM_stride, C_in,  num_tiles);
        ax_compute_gemm(&a_tv, &b_tv, &c_tv);
    }

    /* step 4: output transform → dX (accumulating into dx). */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_in * Ty; flat++) {
        int64_t n  = flat / (C_in * Ty);
        int64_t ci = (flat / Ty) % C_in;
        int64_t ty = flat % Ty;
        float *dc_plane = dx + (n * C_in + ci) * H * W;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO_TB) {
            int ntx = (int)((tx0 + WINO_TB <= Tx) ? WINO_TB : Tx - tx0);
            float tmp[16][WINO_TB];

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 16; ij++)
                memcpy(tmp[ij], bM + ij * bM_stride + ci * num_tiles + tb,
                       (size_t)ntx * sizeof(float));

            for (int dt = 0; dt < ntx; dt++) {
                float m[16];
                for (int ij = 0; ij < 16; ij++)
                    m[ij] = tmp[ij][dt];
                float y[4];
                ax_conv_wino_output_transform_tile(m, y);
                int64_t tx = tx0 + dt;
                for (int oi = 0; oi < 2; oi++) {
                    int64_t oh = ty * 2 + oi;
                    if (oh >= H) break;
                    for (int oj = 0; oj < 2; oj++) {
                        int64_t ow = tx * 2 + oj;
                        if (ow >= W) break;
                        dc_plane[oh * W + ow] += y[oi*2+oj];
                    }
                }
            }
        }
    }

#ifdef _OPENMP
    }
#endif

    return 0;
}

/* winograd F(4,3) — 4x4 output tiles from 3x3 kernels via 6x6 transforms.
   fewer tiles than F(2,3) for the same spatial size (4x output per tile vs 2x),
   reducing total GEMM work at the cost of larger transforms. */

#define WINO43_TB 32

bool ax_conv_prefer_winograd_f43(int kh, int kw, int sh, int sw, int ph, int pw,
                                  int64_t N, int64_t C_in, int64_t C_out,
                                  int64_t out_h, int64_t out_w)
{
    if (!(kh == 3 && kw == 3 && sh == 1 && sw == 1 && ph == 1 && pw == 1)) return false;
    if (C_in < 32 || C_out < 32) return false;
    if (out_h < 8 || out_w < 8) return false;
    int64_t Ty = (out_h + 3) / 4;
    int64_t Tx = (out_w + 3) / 4;
    int64_t v_slots = 36 * C_in * N * Ty * Tx;
    if (v_slots > (int64_t)(16 * 1024 * 1024)) return false;
    return true;
}

void ax_conv_wino43_kernel_transform_filter(const float *g, float *u)
{
    float t[18];
    for (int j = 0; j < 3; j++) {
        const float g0 = g[0*3+j], g1 = g[1*3+j], g2 = g[2*3+j];
        t[0*3+j] = g0 * 0.25f;
        t[1*3+j] = -(g0 + g1 + g2) / 6.0f;
        t[2*3+j] = (-g0 + g1 - g2) / 6.0f;
        t[3*3+j] = (g0 + 2.0f * g1 + 4.0f * g2) / 24.0f;
        t[4*3+j] = (g0 - 2.0f * g1 + 4.0f * g2) / 24.0f;
        t[5*3+j] = g2;
    }
    for (int i = 0; i < 6; i++) {
        const float t0 = t[i*3+0], t1 = t[i*3+1], t2 = t[i*3+2];
        u[i*6+0] = t0 * 0.25f;
        u[i*6+1] = -(t0 + t1 + t2) / 6.0f;
        u[i*6+2] = (-t0 + t1 - t2) / 6.0f;
        u[i*6+3] = (t0 + 2.0f * t1 + 4.0f * t2) / 24.0f;
        u[i*6+4] = (t0 - 2.0f * t1 + 4.0f * t2) / 24.0f;
        u[i*6+5] = t2;
    }
}

void ax_conv_wino43_input_transform_tile(const float *d, float *v)
{
    float t[36];
    for (int j = 0; j < 6; j++) {
        const float d0 = d[0*6+j], d1 = d[1*6+j], d2 = d[2*6+j];
        const float d3 = d[3*6+j], d4 = d[4*6+j], d5 = d[5*6+j];
        t[0*6+j] = 4.0f*d0 - 5.0f*d2 + d4;
        t[1*6+j] = -4.0f*d1 - 4.0f*d2 + d3 + d4;
        t[2*6+j] =  4.0f*d1 - 4.0f*d2 - d3 + d4;
        t[3*6+j] = -2.0f*d1 - d2 + 2.0f*d3 + d4;
        t[4*6+j] =  2.0f*d1 - d2 - 2.0f*d3 + d4;
        t[5*6+j] = 4.0f*d1 - 5.0f*d3 + d5;
    }
    for (int i = 0; i < 6; i++) {
        const float t0 = t[i*6+0], t1 = t[i*6+1], t2 = t[i*6+2];
        const float t3 = t[i*6+3], t4 = t[i*6+4], t5 = t[i*6+5];
        v[i*6+0] = 4.0f*t0 - 5.0f*t2 + t4;
        v[i*6+1] = -4.0f*t1 - 4.0f*t2 + t3 + t4;
        v[i*6+2] =  4.0f*t1 - 4.0f*t2 - t3 + t4;
        v[i*6+3] = -2.0f*t1 - t2 + 2.0f*t3 + t4;
        v[i*6+4] =  2.0f*t1 - t2 - 2.0f*t3 + t4;
        v[i*6+5] = 4.0f*t1 - 5.0f*t3 + t5;
    }
}

void ax_conv_wino43_output_transform_tile(const float *m, float *y)
{
    float t[24];
    for (int j = 0; j < 6; j++) {
        const float m0 = m[0*6+j], m1 = m[1*6+j], m2 = m[2*6+j];
        const float m3 = m[3*6+j], m4 = m[4*6+j], m5 = m[5*6+j];
        t[0*6+j] = m0 + m1 + m2 +       m3 +       m4;
        t[1*6+j] =      m1 - m2 + 2.0f*m3 - 2.0f*m4;
        t[2*6+j] =      m1 + m2 + 4.0f*m3 + 4.0f*m4;
        t[3*6+j] =      m1 - m2 + 8.0f*m3 - 8.0f*m4 + m5;
    }
    for (int i = 0; i < 4; i++) {
        const float t0 = t[i*6+0], t1 = t[i*6+1], t2 = t[i*6+2];
        const float t3 = t[i*6+3], t4 = t[i*6+4], t5 = t[i*6+5];
        y[i*4+0] = t0 + t1 + t2 +       t3 +       t4;
        y[i*4+1] =      t1 - t2 + 2.0f*t3 - 2.0f*t4;
        y[i*4+2] =      t1 + t2 + 4.0f*t3 + 4.0f*t4;
        y[i*4+3] =      t1 - t2 + 8.0f*t3 - 8.0f*t4 + t5;
    }
}

int ax_conv_winograd_f43_forward(
    struct ax_conv_scratch *s,
    const float *id,
    const float *wd,
    const float *bias,
    float *od,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t out_h, int64_t out_w, int T)
{
    (void)T;
    const int64_t Ty = (out_h + 3) / 4;
    const int64_t Tx = (out_w + 3) / 4;
    const int64_t num_tiles = N * Ty * Tx;
    if (num_tiles != s->wino43_num_tiles || !s->wino43_U || !s->wino43_V || !s->wino43_M)
        return -1;

    float *U = (float *)s->wino43_U->storage->data;
    float *V = (float *)s->wino43_V->storage->data;
    float *M = (float *)s->wino43_M->storage->data;

    const int64_t U_stride_ij = C_out * C_in;
    const int64_t V_stride_ij = C_in * num_tiles;
    const int64_t M_stride_ij = C_out * num_tiles;

#ifdef _OPENMP
    #pragma omp parallel
    {
#endif

#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t idx = 0; idx < C_out * C_in; idx++) {
        int64_t co = idx / C_in, ci = idx % C_in;
        float u[36];
        ax_conv_wino43_kernel_transform_filter(wd + (co * C_in + ci) * 9, u);
        for (int ij = 0; ij < 36; ij++)
            U[ij * U_stride_ij + co * C_in + ci] = u[ij];
    }

#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_in * Ty; flat++) {
        int64_t n  = flat / (C_in * Ty);
        int64_t ci = (flat / Ty) % C_in;
        int64_t ty = flat % Ty;
        const float *ic_plane = id + (n * C_in + ci) * H * W;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO43_TB) {
            int ntx = (int)((tx0 + WINO43_TB <= Tx) ? WINO43_TB : Tx - tx0);
            float tmp[36][WINO43_TB];

            for (int dt = 0; dt < ntx; dt++) {
                int64_t tx = tx0 + dt;
                float d[36];
                for (int di = 0; di < 6; di++) {
                    int64_t ih = ty * 4 + di - 1;
                    if (ih < 0 || ih >= H) {
                        for (int dj = 0; dj < 6; dj++)
                            d[di*6+dj] = 0.0f;
                        continue;
                    }
                    const float *irow = ic_plane + ih * W;
                    for (int dj = 0; dj < 6; dj++) {
                        int64_t iw = tx * 4 + dj - 1;
                        d[di*6+dj] = (iw >= 0 && iw < W) ? irow[iw] : 0.0f;
                    }
                }
                float v[36];
                ax_conv_wino43_input_transform_tile(d, v);
                for (int ij = 0; ij < 36; ij++)
                    tmp[ij][dt] = v[ij];
            }

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 36; ij++)
                memcpy(V + ij * V_stride_ij + ci * num_tiles + tb,
                       tmp[ij], (size_t)ntx * sizeof(float));
        }
    }

#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int ij = 0; ij < 36; ij++) {
        ax_storage_t a_st, b_st, c_st;
        ax_tensor_t  a_tv, b_tv, c_tv;
        ax_conv_make_stack_view(&a_tv, &a_st, U + ij * U_stride_ij, C_out, C_in);
        ax_conv_make_stack_view(&b_tv, &b_st, V + ij * V_stride_ij, C_in,  num_tiles);
        ax_conv_make_stack_view(&c_tv, &c_st, M + ij * M_stride_ij, C_out, num_tiles);
        ax_compute_gemm(&a_tv, &b_tv, &c_tv);
    }

#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_out * Ty; flat++) {
        int64_t n  = flat / (C_out * Ty);
        int64_t co = (flat / Ty) % C_out;
        int64_t ty = flat % Ty;
        const float bias_val = bias ? bias[co] : 0.0f;
        float *oc_plane = od + (n * C_out + co) * out_h * out_w;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO43_TB) {
            int ntx = (int)((tx0 + WINO43_TB <= Tx) ? WINO43_TB : Tx - tx0);
            float tmp[36][WINO43_TB];

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 36; ij++)
                memcpy(tmp[ij], M + ij * M_stride_ij + co * num_tiles + tb,
                       (size_t)ntx * sizeof(float));

            for (int dt = 0; dt < ntx; dt++) {
                float m[36];
                for (int ij = 0; ij < 36; ij++)
                    m[ij] = tmp[ij][dt];
                float y[16];
                ax_conv_wino43_output_transform_tile(m, y);
                int64_t tx = tx0 + dt;
                for (int oi = 0; oi < 4; oi++) {
                    int64_t oh = ty * 4 + oi;
                    if (oh >= out_h) break;
                    for (int oj = 0; oj < 4; oj++) {
                        int64_t ow = tx * 4 + oj;
                        if (ow >= out_w) break;
                        oc_plane[oh * out_w + ow] = y[oi*4+oj] + bias_val;
                    }
                }
            }
        }
    }

#ifdef _OPENMP
    }
#endif

    return 0;
}

/* winograd F(4,3) backward dX: dX = conv(grad_out, W_rot180, s=1, p=1).
   reuses forward's wino43_U/V/M buffers with swapped roles:
     forward U [36, C_out, C_in]     -> backward U (reinterpreted as [36, C_in, C_out])
     forward M [36, C_out, num_tiles] -> backward V (input transform of grad_out)
     forward V [36, C_in, num_tiles]  -> backward M (GEMM output)
   kernel flip folded into transform: gf[i] = g[8-i]. */
int ax_conv_winograd_f43_backward_dx(
    struct ax_conv_scratch *s,
    const float *grad_out,  /* [N, C_out, out_h, out_w] */
    const float *wd,        /* [C_out, C_in, 3, 3] original weight */
    float *dx,              /* [N, C_in, H, W] -- accumulating */
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t out_h, int64_t out_w, int T)
{
    (void)T;
    const int64_t Ty = (H + 3) / 4;
    const int64_t Tx = (W + 3) / 4;
    const int64_t num_tiles = N * Ty * Tx;
    if (num_tiles != s->wino43_num_tiles || !s->wino43_U || !s->wino43_V || !s->wino43_M)
        return -1;

    float *bU = (float *)s->wino43_U->storage->data;  /* [36, C_in, C_out] */
    float *bV = (float *)s->wino43_M->storage->data;  /* [36, C_out, num_tiles] */
    float *bM = (float *)s->wino43_V->storage->data;  /* [36, C_in, num_tiles] */

    const int64_t bU_stride = C_in * C_out;
    const int64_t bV_stride = C_out * num_tiles;
    const int64_t bM_stride = C_in * num_tiles;

#ifdef _OPENMP
    #pragma omp parallel
    {
#endif

    /* step 1: flipped kernel transform.
       W_rot180[ci, co, r, s] = W[co, ci, 2-r, 2-s] -> gf[i] = g[8-i]. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t idx = 0; idx < C_in * C_out; idx++) {
        int64_t ci = idx / C_out, co = idx % C_out;
        const float *g = wd + (co * C_in + ci) * 9;
        float gf[9];
        for (int i = 0; i < 9; i++) gf[i] = g[8 - i];
        float u[36];
        ax_conv_wino43_kernel_transform_filter(gf, u);
        for (int ij = 0; ij < 36; ij++)
            bU[ij * bU_stride + ci * C_out + co] = u[ij];
    }

    /* step 2: input transform of grad_out [N, C_out, out_h, out_w]. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_out * Ty; flat++) {
        int64_t n  = flat / (C_out * Ty);
        int64_t co = (flat / Ty) % C_out;
        int64_t ty = flat % Ty;
        const float *gc_plane = grad_out + (n * C_out + co) * out_h * out_w;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO43_TB) {
            int ntx = (int)((tx0 + WINO43_TB <= Tx) ? WINO43_TB : Tx - tx0);
            float tmp[36][WINO43_TB];

            for (int dt = 0; dt < ntx; dt++) {
                int64_t tx = tx0 + dt;
                float d[36];
                for (int di = 0; di < 6; di++) {
                    int64_t ih = ty * 4 + di - 1;
                    if (ih < 0 || ih >= out_h) {
                        for (int dj = 0; dj < 6; dj++)
                            d[di*6+dj] = 0.0f;
                        continue;
                    }
                    const float *grow = gc_plane + ih * out_w;
                    for (int dj = 0; dj < 6; dj++) {
                        int64_t iw = tx * 4 + dj - 1;
                        d[di*6+dj] = (iw >= 0 && iw < out_w) ? grow[iw] : 0.0f;
                    }
                }
                float v[36];
                ax_conv_wino43_input_transform_tile(d, v);
                for (int ij = 0; ij < 36; ij++)
                    tmp[ij][dt] = v[ij];
            }

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 36; ij++)
                memcpy(bV + ij * bV_stride + co * num_tiles + tb,
                       tmp[ij], (size_t)ntx * sizeof(float));
        }
    }

    /* step 3: 36 GEMMs -- [C_in, C_out] @ [C_out, num_tiles] -> [C_in, num_tiles]. */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int ij = 0; ij < 36; ij++) {
        ax_storage_t a_st, b_st, c_st;
        ax_tensor_t  a_tv, b_tv, c_tv;
        ax_conv_make_stack_view(&a_tv, &a_st, bU + ij * bU_stride, C_in,  C_out);
        ax_conv_make_stack_view(&b_tv, &b_st, bV + ij * bV_stride, C_out, num_tiles);
        ax_conv_make_stack_view(&c_tv, &c_st, bM + ij * bM_stride, C_in,  num_tiles);
        ax_compute_gemm(&a_tv, &b_tv, &c_tv);
    }

    /* step 4: output transform -> dX (accumulating into dx). */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int64_t flat = 0; flat < N * C_in * Ty; flat++) {
        int64_t n  = flat / (C_in * Ty);
        int64_t ci = (flat / Ty) % C_in;
        int64_t ty = flat % Ty;
        float *dc_plane = dx + (n * C_in + ci) * H * W;
        int64_t tile_row = (n * Ty + ty) * Tx;

        for (int64_t tx0 = 0; tx0 < Tx; tx0 += WINO43_TB) {
            int ntx = (int)((tx0 + WINO43_TB <= Tx) ? WINO43_TB : Tx - tx0);
            float tmp[36][WINO43_TB];

            int64_t tb = tile_row + tx0;
            for (int ij = 0; ij < 36; ij++)
                memcpy(tmp[ij], bM + ij * bM_stride + ci * num_tiles + tb,
                       (size_t)ntx * sizeof(float));

            for (int dt = 0; dt < ntx; dt++) {
                float m[36];
                for (int ij = 0; ij < 36; ij++)
                    m[ij] = tmp[ij][dt];
                float y[16];
                ax_conv_wino43_output_transform_tile(m, y);
                int64_t tx = tx0 + dt;
                for (int oi = 0; oi < 4; oi++) {
                    int64_t oh = ty * 4 + oi;
                    if (oh >= H) break;
                    for (int oj = 0; oj < 4; oj++) {
                        int64_t ow = tx * 4 + oj;
                        if (ow >= W) break;
                        dc_plane[oh * W + ow] += y[oi*4+oj];
                    }
                }
            }
        }
    }

#ifdef _OPENMP
    }
#endif

    return 0;
}
