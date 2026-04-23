/* attention/layout.c — head-interleave / deinterleave layout transforms
   used by the multi-head attention forward + backward.

   data layouts:
     [B, S, D]      = "flat"      — what the user sees (rank-3 input)
     [B, S, H, dk]  = "split"     — heads alongside the sequence dim
     [B, H, S, dk]  = "head-major"— heads separated; sdpa input layout
     [BH, S, dk]    = same as head-major but with B*H flattened

   the qkv-fused variants operate on a wider [rows, 3*D] buffer so the
   forward QKV projection ([rows, D] @ [D, 3D]) can feed sdpa in one
   pass without materialising intermediate [rows, D] Qf / Kf / Vf
   tensors. all transforms are pure memcpy patterns plus optional
   bias-add — no compute. parallelism is openmp parallel-for over
   (b, h) pairs. */

#include "internal.h"
#include "../../compute/backends/simd_defs.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void ax_attn_head_interleave(const float *src, float *dst,
                              int64_t B, int64_t S, int64_t H, int64_t dk)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + (((b * S) + s) * H + h) * dk;
                float *dst_row       = dst + (((b * H) + h) * S + s) * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

void ax_attn_head_deinterleave(const float *src, float *dst,
                                int64_t B, int64_t S, int64_t H, int64_t dk)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                const float *src_row = src + (((b * H) + h) * S + s) * dk;
                float *dst_row       = dst + (((b * S) + s) * H + h) * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

void ax_attn_head_deinterleave_slot(const float *src, float *dst,
                                     int64_t B, int64_t S, int64_t H, int64_t dk,
                                     int64_t dst_cols, int64_t col_off)
{
    (void)H;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                const float *src_row = src + (((b * H) + h) * S + s) * dk;
                float *dst_row       = dst + ((b * S) + s) * dst_cols + col_off + h * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

void ax_attn_head_deinterleave_qkv_merge(const float *srcQ, const float *srcK,
                                          const float *srcV, float *dst,
                                          int64_t B, int64_t S, int64_t H,
                                          int64_t dk, int64_t D)
{
    int64_t dst_cols = 3 * D;
    size_t dk_bytes = (size_t)dk * sizeof(float);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t src_off = (((b * H) + h) * S + s) * dk;
                int64_t dst_base = ((b * S) + s) * dst_cols + h * dk;
                memcpy(dst + dst_base,         srcQ + src_off, dk_bytes);
                memcpy(dst + dst_base + D,     srcK + src_off, dk_bytes);
                memcpy(dst + dst_base + 2*D,   srcV + src_off, dk_bytes);
            }
        }
    }
}

void ax_attn_head_interleave_from_slot(const float *src, float *dst,
                                        int64_t B, int64_t S, int64_t H, int64_t dk,
                                        int64_t src_cols, int64_t col_off)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + ((b * S) + s) * src_cols + col_off + h * dk;
                float *dst_row       = dst + (((b * H) + h) * S + s) * dk;
                memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
            }
        }
    }
}

void ax_attn_head_interleave_qkv_split(const float *src,
                                        float *dstQ, float *dstK, float *dstV,
                                        int64_t B, int64_t S, int64_t H, int64_t dk,
                                        int64_t D)
{
    int64_t src_cols = 3 * D;
    size_t dk_bytes = (size_t)dk * sizeof(float);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + ((b * S) + s) * src_cols + h * dk;
                int64_t dst_off = (((b * H) + h) * S + s) * dk;
                memcpy(dstQ + dst_off, src_row,           dk_bytes);
                memcpy(dstK + dst_off, src_row + D,       dk_bytes);
                memcpy(dstV + dst_off, src_row + 2 * D,   dk_bytes);
            }
        }
    }
}

void ax_attn_head_interleave_qkv_split_bias(const float *src, const float *bias,
                                             float *dstQ, float *dstK, float *dstV,
                                             int64_t B, int64_t S, int64_t H, int64_t dk,
                                             int64_t D)
{
    int64_t src_cols = 3 * D;
    int64_t TC = AX_VF32_WIDTH;
    int64_t ve = dk - (dk % TC);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            const float *bQ_h = bias + h * dk;             /* bq slice for this head */
            const float *bK_h = bias + D + h * dk;
            const float *bV_h = bias + 2 * D + h * dk;
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + ((b * S) + s) * src_cols + h * dk;
                int64_t dst_off = (((b * H) + h) * S + s) * dk;
                float *dQ = dstQ + dst_off;
                float *dK = dstK + dst_off;
                float *dV = dstV + dst_off;
                int64_t i = 0;
                for (; i < ve; i += TC) {
                    ax_vf32_storeu(dQ + i, ax_vf32_add(ax_vf32_loadu(src_row +       i), ax_vf32_loadu(bQ_h + i)));
                    ax_vf32_storeu(dK + i, ax_vf32_add(ax_vf32_loadu(src_row + D   + i), ax_vf32_loadu(bK_h + i)));
                    ax_vf32_storeu(dV + i, ax_vf32_add(ax_vf32_loadu(src_row + 2*D + i), ax_vf32_loadu(bV_h + i)));
                }
                for (; i < dk; i++) {
                    dQ[i] = src_row[i] + bQ_h[i];
                    dK[i] = src_row[D + i] + bK_h[i];
                    dV[i] = src_row[2*D + i] + bV_h[i];
                }
            }
        }
    }
}
