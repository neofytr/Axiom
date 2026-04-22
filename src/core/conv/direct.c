/* conv/direct.c — direct (im2col-free) convolution kernels.

   k.2 split: extracted from src/core/conv.c. these three kernels skip
   the gemm/im2col detour entirely, useful when:

   * 3x3 stride-1 pad-1, small Cin*9 kernel that stays hot in L1
     (gated off by default — see ax_conv_can_direct_3x3 for the
     measurement note explaining why)
   * 3x3 stride-2 pad-1 — the same fast-path with the stride-2
     gather via shuffle+permute
   * very small Cin (1..4), typical first conv layer of an image
     net (RGB), where im2col would copy 9× the input but the resulting
     GEMM with K=27 is too small to amortise pack overhead. tf has the
     same specialised first-layer kernel for this reason.

   each kernel is per-sample; the dispatcher in forward.c parallelises
   the outer (sample) loop and picks one kernel based on the
   ax_conv_can_direct_* predicates below. */

#include "internal.h"
#include "../../compute/backends/simd_defs.h"
#include <stdint.h>

/* direct 3x3 stride=1 pad=1 conv for a single sample.
   skips the gemm/im2col detour when the kernel is small enough to stay
   hot in L1. output shape [C_out, H, W] (pad=1 keeps spatial size).
   bias is folded into the row init. */
void ax_conv_direct_3x3_sample(
    const float *in_n,
    const float *wd,
    const float *bias,
    float *out_n,
    int64_t C_in, int64_t C_out,
    int64_t H, int64_t W)
{
    const int64_t HW = H * W;
    const int64_t K9 = C_in * 9;

    for (int64_t co = 0; co < C_out; co++) {
        float bias_val = bias ? bias[co] : 0.0f;
        ax_vf32 vb = ax_vf32_set1(bias_val);
        const float *wco = wd + co * K9;
        float *out_co = out_n + co * HW;

        for (int64_t y = 0; y < H; y++) {
            float *out_row = out_co + y * W;
            int64_t xi = 0, vend_init = W - (W % AX_VF32_WIDTH);
            for (; xi < vend_init; xi += AX_VF32_WIDTH)
                ax_vf32_storeu(out_row + xi, vb);
            for (; xi < W; xi++) out_row[xi] = bias_val;

            for (int64_t ci = 0; ci < C_in; ci++) {
                const float *win = wco + ci * 9;
                const float *in_ci = in_n + ci * HW;

                for (int ky = 0; ky < 3; ky++) {
                    int64_t in_y = y + ky - 1;
                    if (in_y < 0 || in_y >= H) continue;
                    const float *in_row = in_ci + in_y * W;

                    for (int kx = 0; kx < 3; kx++) {
                        float wv = win[ky * 3 + kx];
                        ax_vf32 vw = ax_vf32_set1(wv);
                        int64_t shift = (int64_t)kx - 1;

                        int64_t x_lo = (kx == 0) ? 1 : 0;
                        int64_t x_hi = (kx == 2) ? (W - 1) : W;
                        int64_t x = x_lo;
                        int64_t span = x_hi - x_lo;
                        int64_t xvec_end = x_lo + (span - (span % AX_VF32_WIDTH));
                        for (; x < xvec_end; x += AX_VF32_WIDTH) {
                            ax_vf32 vi = ax_vf32_loadu(in_row + x + shift);
                            ax_vf32 vo = ax_vf32_loadu(out_row + x);
                            ax_vf32_storeu(out_row + x, ax_vf32_fmadd(vi, vw, vo));
                        }
                        for (; x < x_hi; x++)
                            out_row[x] += in_row[x + shift] * wv;
                    }
                }
            }
        }
    }
}

/* direct 3x3 stride=2 pad=1 conv for a single sample.
   skips im2col + gemm entirely. output shape:
     out_h = (H + 2*1 - 3) / 2 + 1 = (H + 1) / 2  (when H is even)
     out_w = (W + 1) / 2
   structure mirrors ax_conv_direct_3x3_sample (stride=1 variant) above —
   write bias to out, then for each (ci, ky, kx) accumulate into out via
   broadcast-FMA. the only delta is the stride-2 input gather which we
   handle via the same shuffle+permute trick as im2col_into's stride_w==2
   fast path: load 16 contiguous floats, _mm256_shuffle_ps(a,b,(2,0,2,0))
   collects evens within each 128-bit lane, _mm256_permutevar8x32_ps
   reorders cross-lane → 8 stride-2 floats. */
void ax_conv_direct_3x3_s2_sample(
    const float *in_n,
    const float *wd,
    const float *bias,
    float *out_n,
    int64_t C_in, int64_t C_out,
    int64_t H, int64_t W,
    int64_t out_h, int64_t out_w)
{
    const int64_t HW = H * W;
    const int64_t out_HW = out_h * out_w;
    const int64_t K9 = C_in * 9;

#if defined(AX_SIMD_AVX2) && AX_VF32_WIDTH == 8
    const __m256i extract_evens = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
#endif

    for (int64_t co = 0; co < C_out; co++) {
        float bias_val = bias ? bias[co] : 0.0f;
        ax_vf32 vb = ax_vf32_set1(bias_val);
        const float *wco = wd + co * K9;
        float *out_co = out_n + co * out_HW;

        /* init output to bias */
        for (int64_t y = 0; y < out_h; y++) {
            float *out_row = out_co + y * out_w;
            int64_t xi = 0, vend = out_w - (out_w % AX_VF32_WIDTH);
            for (; xi < vend; xi += AX_VF32_WIDTH)
                ax_vf32_storeu(out_row + xi, vb);
            for (; xi < out_w; xi++) out_row[xi] = bias_val;
        }

        /* accumulate Cin × 9 contributions */
        for (int64_t ci = 0; ci < C_in; ci++) {
            const float *win = wco + ci * 9;
            const float *in_ci = in_n + ci * HW;

            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    float wv = win[ky * 3 + kx];
                    ax_vf32 vw = ax_vf32_set1(wv);

                    for (int64_t oh = 0; oh < out_h; oh++) {
                        int64_t ih = oh * 2 - 1 + ky;
                        if (ih < 0 || ih >= H) continue;
                        const float *in_row = in_ci + ih * W;
                        float *out_row = out_co + oh * out_w;

                        /* ow bounds for in-image accesses:
                             iw = ow*2 - 1 + kx must be in [0, W)
                             → ow_lo = ceil((1 - kx) / 2) when (1-kx) > 0
                             → ow_hi = floor((W - 1 + 1 - kx) / 2) + 1 */
                        int64_t ow_lo = (kx == 0) ? 1 : 0;
                        int64_t ow_hi = out_w;
                        while (ow_hi > 0 && (ow_hi - 1) * 2 - 1 + kx >= W) ow_hi--;

                        int64_t ow = ow_lo;
                        int64_t span = ow_hi - ow_lo;
                        int64_t ow_vec_end = ow_lo + (span - (span % AX_VF32_WIDTH));

#if defined(AX_SIMD_AVX2) && AX_VF32_WIDTH == 8
                        /* SIMD: 8 outputs per iter; stride-2 gather via shuffle */
                        for (; ow < ow_vec_end; ow += AX_VF32_WIDTH) {
                            int64_t iw_first = ow * 2 - 1 + kx;
                            /* must have iw_first + 16 <= W to load 16 floats */
                            if (iw_first + 16 > W) break;
                            __m256 a = _mm256_loadu_ps(in_row + iw_first);
                            __m256 b = _mm256_loadu_ps(in_row + iw_first + 8);
                            __m256 t = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
                            __m256 vi = _mm256_permutevar8x32_ps(t, extract_evens);
                            __m256 vo = _mm256_loadu_ps(out_row + ow);
                            _mm256_storeu_ps(out_row + ow, _mm256_fmadd_ps(vi, vw, vo));
                        }
#endif
                        /* scalar tail / right edge */
                        for (; ow < ow_hi; ow++) {
                            int64_t iw = ow * 2 - 1 + kx;
                            out_row[ow] += in_row[iw] * wv;
                        }
                    }
                }
            }
        }
    }
}

bool ax_conv_can_direct_3x3_s2(int kh, int kw, int sh, int sw,
                                int ph, int pw, int64_t C_in,
                                int64_t H, int64_t W)
{
    /* match exactly the strided 3x3-pad1 case the kernel handles */
    return kh == 3 && kw == 3 && sh == 2 && sw == 2 && ph == 1 && pw == 1
           && H >= 4 && W >= 16 && C_in >= 1;
}

/* shape-aware path selection. measurement note: in practice the BLIS-style
   tiled gemm with explicit im2col beats this hand-rolled direct conv on
   typical mnist conv shapes (74s vs 137s at T=1) because the micro-kernel
   hits ~96 GFLOPS while the direct loop is bandwidth-bound on the bias init
   and the per-tap fmadd. direct 3x3 is kept here as available infrastructure
   for workloads where im2col cost truly dominates (very small spatial dims
   with many channels), but the default predicate disables it. flip
   `AX_USE_DIRECT_3X3=1` at compile time to opt back in. */
#ifndef AX_USE_DIRECT_3X3
#define AX_USE_DIRECT_3X3 0
#endif
bool ax_conv_can_direct_3x3(int kh, int kw, int sh, int sw, int ph, int pw, int64_t C_in)
{
#if AX_USE_DIRECT_3X3
    return kh == 3 && kw == 3 && sh == 1 && sw == 1 && ph == 1 && pw == 1 && (C_in * 9) < 512;
#else
    (void)kh; (void)kw; (void)sh; (void)sw; (void)ph; (void)pw; (void)C_in;
    return false;
#endif
}

/* direct conv for very-small input channels (typical first layer: C_in=3 RGB).
   im2col would copy 9× the input for K=3, then run a GEMM with K_gemm=27 too
   small to amortize pack overhead. tf has a specialized first-layer kernel
   for the same reason. this kernel keeps the per-output-element accumulator
   in a vector register (one bias-init, K*K*C_in fmadds, one store per
   AX_VF32_WIDTH outputs). three regions split out: left-pad scalar, middle
   pure SIMD with no bounds check (the fast path that covers >97% of pixels
   on typical 224×224 inputs with pad=1), right-pad scalar. */
void ax_conv_direct_smallcin_sample(
    const float *in_n, const float *wd, const float *bias, float *out_n,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W,
    int kh, int kw, int sh, int sw, int ph, int pw,
    int64_t H_out, int64_t W_out)
{
    const int64_t K_per_co = C_in * kh * kw;
    /* middle region of ow where every accessed iw is in [0, W).
       iw = ow*sw + kx - pw, ranges over kx in [0, kw).
       safe iff (ow*sw - pw) >= 0  AND  (ow*sw + kw-1 - pw) < W
            iff  ow >= ceil(pw/sw)  AND  ow <= floor((W-1 - (kw-1) + pw) / sw).
       the SIMD load reads AX_VF32_WIDTH consecutive columns when sw==1, so
       we additionally need (ow + AX_VF32_WIDTH - 1)*sw + kw-1 - pw < W. */
    int64_t ow_safe_lo = (pw + sw - 1) / sw;        /* ceil(pw/sw) */
    int64_t last_in = W - 1 - (kw - 1) + pw;        /* iw_base for last safe ow */
    int64_t ow_safe_hi = last_in >= 0 ? (last_in / sw) + 1 : 0;
    if (ow_safe_hi > W_out) ow_safe_hi = W_out;
    if (ow_safe_lo > ow_safe_hi) ow_safe_lo = ow_safe_hi;
    /* SIMD safety needs the last vector lane to also be in range when sw==1 */
    int64_t ow_simd_hi = (sw == 1) ? (ow_safe_hi - (AX_VF32_WIDTH - 1)) : ow_safe_lo;
    if (ow_simd_hi < ow_safe_lo) ow_simd_hi = ow_safe_lo;

    /* 4-co tiling path: amortizes input loads 4× across output channels.
       Each inner (ci, ky, kx) iter does 2 input loads (shared across 4 co)
       + 4 weight broadcasts + 8 fmadds. Fits cleanly in 14 AVX2 ymm regs:
       8 acc + 2 vi + 4 vw. C_out remainder (mod 4) falls through to
       single-co path below. */
    int64_t co = 0;
    if (sw == 1) {
        const int64_t TW = AX_VF32_WIDTH;
        const int64_t TILE_OW = 2 * TW;
        for (; co + 4 <= C_out; co += 4) {
            const float bv0 = bias ? bias[co + 0] : 0.0f;
            const float bv1 = bias ? bias[co + 1] : 0.0f;
            const float bv2 = bias ? bias[co + 2] : 0.0f;
            const float bv3 = bias ? bias[co + 3] : 0.0f;
            const ax_vf32 vb0 = ax_vf32_set1(bv0);
            const ax_vf32 vb1 = ax_vf32_set1(bv1);
            const ax_vf32 vb2 = ax_vf32_set1(bv2);
            const ax_vf32 vb3 = ax_vf32_set1(bv3);
            const float *wco0 = wd + (co + 0) * K_per_co;
            const float *wco1 = wd + (co + 1) * K_per_co;
            const float *wco2 = wd + (co + 2) * K_per_co;
            const float *wco3 = wd + (co + 3) * K_per_co;
            float *oco0 = out_n + (co + 0) * H_out * W_out;
            float *oco1 = out_n + (co + 1) * H_out * W_out;
            float *oco2 = out_n + (co + 2) * H_out * W_out;
            float *oco3 = out_n + (co + 3) * H_out * W_out;

            for (int64_t oh = 0; oh < H_out; oh++) {
                const int64_t ih_base = oh * sh - ph;
                float *out_row0 = oco0 + oh * W_out;
                float *out_row1 = oco1 + oh * W_out;
                float *out_row2 = oco2 + oh * W_out;
                float *out_row3 = oco3 + oh * W_out;
                int64_t ow = 0;

                /* left pad: scalar over the 4-co block */
                for (; ow < ow_safe_lo && ow < W_out; ow++) {
                    float a0 = bv0, a1 = bv1, a2 = bv2, a3 = bv3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                int64_t iw = iw_base + kx;
                                if (iw < 0 || iw >= W) continue;
                                float v = in_row[iw];
                                a0 += wco0[wbase + kx] * v;
                                a1 += wco1[wbase + kx] * v;
                                a2 += wco2[wbase + kx] * v;
                                a3 += wco3[wbase + kx] * v;
                            }
                        }
                    }
                    out_row0[ow] = a0;
                    out_row1[ow] = a1;
                    out_row2[ow] = a2;
                    out_row3[ow] = a3;
                }

                /* SIMD core: 2 ow tiles × 4 co channels = 8 accumulators */
                int64_t v2end = ow_simd_hi - ((ow_simd_hi - ow) % TILE_OW);
                for (; ow + TILE_OW <= v2end; ow += TILE_OW) {
                    ax_vf32 a00 = vb0, a01 = vb0;
                    ax_vf32 a10 = vb1, a11 = vb1;
                    ax_vf32 a20 = vb2, a21 = vb2;
                    ax_vf32 a30 = vb3, a31 = vb3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 v0 = ax_vf32_loadu(in_row + iw_base + kx);
                                ax_vf32 v1 = ax_vf32_loadu(in_row + iw_base + TW + kx);
                                ax_vf32 vw0 = ax_vf32_set1(wco0[wbase + kx]);
                                ax_vf32 vw1 = ax_vf32_set1(wco1[wbase + kx]);
                                ax_vf32 vw2 = ax_vf32_set1(wco2[wbase + kx]);
                                ax_vf32 vw3 = ax_vf32_set1(wco3[wbase + kx]);
                                a00 = ax_vf32_fmadd(vw0, v0, a00);
                                a01 = ax_vf32_fmadd(vw0, v1, a01);
                                a10 = ax_vf32_fmadd(vw1, v0, a10);
                                a11 = ax_vf32_fmadd(vw1, v1, a11);
                                a20 = ax_vf32_fmadd(vw2, v0, a20);
                                a21 = ax_vf32_fmadd(vw2, v1, a21);
                                a30 = ax_vf32_fmadd(vw3, v0, a30);
                                a31 = ax_vf32_fmadd(vw3, v1, a31);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row0 + ow,      a00);
                    ax_vf32_storeu(out_row0 + ow + TW, a01);
                    ax_vf32_storeu(out_row1 + ow,      a10);
                    ax_vf32_storeu(out_row1 + ow + TW, a11);
                    ax_vf32_storeu(out_row2 + ow,      a20);
                    ax_vf32_storeu(out_row2 + ow + TW, a21);
                    ax_vf32_storeu(out_row3 + ow,      a30);
                    ax_vf32_storeu(out_row3 + ow + TW, a31);
                }

                /* 1-tile SIMD tail (still 4-co) */
                int64_t vend = ow_simd_hi - ((ow_simd_hi - ow) % TW);
                for (; ow < vend; ow += TW) {
                    ax_vf32 a0 = vb0, a1 = vb1, a2 = vb2, a3 = vb3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 vi = ax_vf32_loadu(in_row + iw_base + kx);
                                a0 = ax_vf32_fmadd(ax_vf32_set1(wco0[wbase + kx]), vi, a0);
                                a1 = ax_vf32_fmadd(ax_vf32_set1(wco1[wbase + kx]), vi, a1);
                                a2 = ax_vf32_fmadd(ax_vf32_set1(wco2[wbase + kx]), vi, a2);
                                a3 = ax_vf32_fmadd(ax_vf32_set1(wco3[wbase + kx]), vi, a3);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row0 + ow, a0);
                    ax_vf32_storeu(out_row1 + ow, a1);
                    ax_vf32_storeu(out_row2 + ow, a2);
                    ax_vf32_storeu(out_row3 + ow, a3);
                }

                /* right-pad / scalar tail (4-co) */
                for (; ow < W_out; ow++) {
                    float a0 = bv0, a1 = bv1, a2 = bv2, a3 = bv3;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            int64_t wbase = (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                int64_t iw = iw_base + kx;
                                if (iw < 0 || iw >= W) continue;
                                float v = in_row[iw];
                                a0 += wco0[wbase + kx] * v;
                                a1 += wco1[wbase + kx] * v;
                                a2 += wco2[wbase + kx] * v;
                                a3 += wco3[wbase + kx] * v;
                            }
                        }
                    }
                    out_row0[ow] = a0;
                    out_row1[ow] = a1;
                    out_row2[ow] = a2;
                    out_row3[ow] = a3;
                }
            }
        }
    }

    /* remainder co (1..3 channels): single-co fallback path */
    for (; co < C_out; co++) {
        const float bv = bias ? bias[co] : 0.0f;
        const ax_vf32 vb = ax_vf32_set1(bv);
        const float *wco = wd + co * K_per_co;
        float *oco = out_n + co * H_out * W_out;

        for (int64_t oh = 0; oh < H_out; oh++) {
            const int64_t ih_base = oh * sh - ph;
            float *out_row = oco + oh * W_out;
            int64_t ow = 0;

            /* left pad: scalar, bounded loop */
            for (; ow < ow_safe_lo && ow < W_out; ow++) {
                float acc = bv;
                int64_t iw_base = ow * sw - pw;
                for (int64_t ci = 0; ci < C_in; ci++) {
                    const float *in_ci = in_n + ci * H * W;
                    for (int ky = 0; ky < kh; ky++) {
                        int64_t ih = ih_base + ky;
                        if (ih < 0 || ih >= H) continue;
                        const float *in_row = in_ci + ih * W;
                        const float *wky = wco + (ci * kh + ky) * kw;
                        for (int kx = 0; kx < kw; kx++) {
                            int64_t iw = iw_base + kx;
                            if (iw >= 0 && iw < W) acc += wky[kx] * in_row[iw];
                        }
                    }
                }
                out_row[ow] = acc;
            }

            /* middle region: pure SIMD, no per-pixel bounds check.
               4-way ow register tiling to hide fma latency. with C_in=3 and
               K=3 the inner (ci, ky, kx) loop is 27 sequential fmadds; a
               single accumulator chains all 27 dependencies through one
               register and stalls the pipeline (4-cycle fma latency × 27 =
               108 cycles, vs 14 cycles at throughput). 4 parallel
               accumulators turn that into 4 independent chains, fully
               saturating the FMA units. only oh-edge ky values still need
               the ih bounds check. */
            if (sw == 1) {
                const int64_t TW = AX_VF32_WIDTH;
                const int64_t TILE4 = 4 * TW;
                /* 4-tile path: process 4 vector blocks per outer iter */
                int64_t v4end = ow_simd_hi - ((ow_simd_hi - ow) % TILE4);
                for (; ow + TILE4 <= v4end; ow += TILE4) {
                    ax_vf32 a0 = vb, a1 = vb, a2 = vb, a3 = vb;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            const float *wky = wco + (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 vw = ax_vf32_set1(wky[kx]);
                                ax_vf32 v0 = ax_vf32_loadu(in_row + iw_base + kx);
                                ax_vf32 v1 = ax_vf32_loadu(in_row + iw_base + TW + kx);
                                ax_vf32 v2 = ax_vf32_loadu(in_row + iw_base + 2*TW + kx);
                                ax_vf32 v3 = ax_vf32_loadu(in_row + iw_base + 3*TW + kx);
                                a0 = ax_vf32_fmadd(vw, v0, a0);
                                a1 = ax_vf32_fmadd(vw, v1, a1);
                                a2 = ax_vf32_fmadd(vw, v2, a2);
                                a3 = ax_vf32_fmadd(vw, v3, a3);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row + ow,          a0);
                    ax_vf32_storeu(out_row + ow + TW,     a1);
                    ax_vf32_storeu(out_row + ow + 2 * TW, a2);
                    ax_vf32_storeu(out_row + ow + 3 * TW, a3);
                }
                /* 1-tile tail */
                int64_t vend = ow_simd_hi - ((ow_simd_hi - ow) % TW);
                for (; ow < vend; ow += TW) {
                    ax_vf32 acc = vb;
                    int64_t iw_base = ow - pw;
                    for (int64_t ci = 0; ci < C_in; ci++) {
                        const float *in_ci = in_n + ci * H * W;
                        for (int ky = 0; ky < kh; ky++) {
                            int64_t ih = ih_base + ky;
                            if (ih < 0 || ih >= H) continue;
                            const float *in_row = in_ci + ih * W;
                            const float *wky = wco + (ci * kh + ky) * kw;
                            for (int kx = 0; kx < kw; kx++) {
                                ax_vf32 vw = ax_vf32_set1(wky[kx]);
                                ax_vf32 vi = ax_vf32_loadu(in_row + iw_base + kx);
                                acc = ax_vf32_fmadd(vw, vi, acc);
                            }
                        }
                    }
                    ax_vf32_storeu(out_row + ow, acc);
                }
            }

            /* tail / strided: scalar */
            for (; ow < W_out; ow++) {
                float acc = bv;
                int64_t iw_base = ow * sw - pw;
                for (int64_t ci = 0; ci < C_in; ci++) {
                    const float *in_ci = in_n + ci * H * W;
                    for (int ky = 0; ky < kh; ky++) {
                        int64_t ih = ih_base + ky;
                        if (ih < 0 || ih >= H) continue;
                        const float *in_row = in_ci + ih * W;
                        const float *wky = wco + (ci * kh + ky) * kw;
                        for (int kx = 0; kx < kw; kx++) {
                            int64_t iw = iw_base + kx;
                            if (iw >= 0 && iw < W) acc += wky[kx] * in_row[iw];
                        }
                    }
                }
                out_row[ow] = acc;
            }
        }
    }
}

/* eligibility for the small-C_in direct path. covers RGB image first-layer
   convs (the regression case): C_in ∈ {1,2,3,4}, kernel up to 7, any stride
   and padding. SIMD path requires the spatial output to be wide enough that
   middle-region SIMD covers more than the (left-pad + tail) overhead, so
   require W_out >= AX_VF32_WIDTH * 2 to be conservative. */
bool ax_conv_can_direct_smallcin(int kh, int kw, int sh, int sw,
                                  int64_t C_in, int64_t W_out)
{
    (void)sh; (void)sw;
    return C_in <= 4 && kh <= 7 && kw <= 7
           && W_out >= (int64_t)AX_VF32_WIDTH * 2;
}
