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
   (b, h) pairs.

   T3.1: streaming stores. for layouts whose dst footprint exceeds the
   per-thread L2 share, we use SIMD non-temporal stores so the write
   bypasses cache. saves ~15-25 % of L3 bandwidth on bigger shapes
   (B=2 S=1024 dst ~12 MB) without polluting cache lines that the
   subsequent attention forward will reload anyway. small shapes (dst
   < L2_per_thread * 2) keep the regular memcpy path so the write
   stays cache-resident for the immediate downstream reader.

   alignment requirements: NT stores need 32-byte alignment on AVX2
   (_mm256_stream_ps) and 64-byte alignment on AVX-512 (_mm512_stream_ps).
   axiom's tensor allocator guarantees 64-byte alignment, and dk is
   always a multiple of AX_VF32_WIDTH on AVX2/AVX-512 — but per-row
   destinations may sit on non-aligned offsets if dk * sizeof(float)
   isn't 64-byte-aligned. we check alignment per-row at runtime; the
   fast path takes pure NT, slow path falls back to memcpy. */

#include "internal.h"
#include "../../compute/backends/simd_defs.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(AX_HAS_SIMD)
/* per-row streaming copy: writes len floats from src to dst with NT
   stores. dst must be AX_VF32_WIDTH * 4-byte aligned and len must be
   a multiple of AX_VF32_WIDTH. caller wraps in a loop and ensures
   sfence at the end of the parallel region. */
static inline void ax_stream_copy_floats(float *dst, const float *src, int64_t len) {
    int64_t i = 0;
    int64_t ve = len - (len % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 v = ax_vf32_loadu(src + i);
        ax_vf32_stream(dst + i, v);
    }
    for (; i < len; i++) dst[i] = src[i];
}

/* sfence emitter — required after a batch of NT stores so subsequent
   loads in the same thread see the data. only meaningful on AVX2 / AVX-512. */
static inline void ax_stream_fence(void) {
#if defined(AX_SIMD_AVX512) || defined(AX_SIMD_AVX2)
    _mm_sfence();
#endif
}

/* aligned-enough check — NT stores need 32-byte (AVX2) / 64-byte (AVX-512)
   alignment on the destination. AX_VF32_WIDTH * sizeof(float) gives the
   alignment in bytes (32 for AVX2, 64 for AVX-512). */
static inline int ax_dst_nt_alignable(const void *p, int64_t row_floats) {
    const size_t want = (size_t)AX_VF32_WIDTH * sizeof(float);
    if (((uintptr_t)p % want) != 0) return 0;
    /* row stride in bytes must keep subsequent rows aligned too */
    if (((size_t)row_floats * sizeof(float) % want) != 0) return 0;
    return 1;
}
#endif

/* heuristic: does this transform's dst footprint exceed cache enough
   that NT writes are a net win? the threshold is 4 MB which is bigger
   than typical per-thread L2 share (2 MB / nt = 128 KB at 16 threads,
   or up to 512 KB on Apple M) but smaller than L3 — exactly the regime
   where dst spills L2 but the immediate-downstream reader can refetch
   from L3. above L3 size, NT is strictly better; below, conventional
   memcpy keeps the data hot for the next stage. */
static inline int ax_layout_use_nt(int64_t dst_total_bytes) {
    return dst_total_bytes >= (4 * 1024 * 1024);
}

void ax_attn_head_interleave(const float *src, float *dst,
                              int64_t B, int64_t S, int64_t H, int64_t dk)
{
    int64_t total_bytes = B * H * S * dk * (int64_t)sizeof(float);
    int use_nt = 0;
#if defined(AX_HAS_SIMD)
    use_nt = ax_layout_use_nt(total_bytes) && ax_dst_nt_alignable(dst, dk);
#else
    (void)total_bytes;
#endif
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + (((b * S) + s) * H + h) * dk;
                float *dst_row       = dst + (((b * H) + h) * S + s) * dk;
#if defined(AX_HAS_SIMD)
                if (use_nt) {
                    ax_stream_copy_floats(dst_row, src_row, dk);
                } else
#endif
                {
                    memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
                }
            }
        }
    }
#if defined(AX_HAS_SIMD)
    if (use_nt) ax_stream_fence();
#endif
}

void ax_attn_head_deinterleave(const float *src, float *dst,
                                int64_t B, int64_t S, int64_t H, int64_t dk)
{
    int64_t total_bytes = B * S * H * dk * (int64_t)sizeof(float);
    int use_nt = 0;
#if defined(AX_HAS_SIMD)
    use_nt = ax_layout_use_nt(total_bytes) && ax_dst_nt_alignable(dst, dk);
#else
    (void)total_bytes;
#endif
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                const float *src_row = src + (((b * H) + h) * S + s) * dk;
                float *dst_row       = dst + (((b * S) + s) * H + h) * dk;
#if defined(AX_HAS_SIMD)
                if (use_nt) {
                    ax_stream_copy_floats(dst_row, src_row, dk);
                } else
#endif
                {
                    memcpy(dst_row, src_row, (size_t)dk * sizeof(float));
                }
            }
        }
    }
#if defined(AX_HAS_SIMD)
    if (use_nt) ax_stream_fence();
#endif
}

void ax_attn_head_deinterleave_slot(const float *src, float *dst,
                                     int64_t B, int64_t S, int64_t H, int64_t dk,
                                     int64_t dst_cols, int64_t col_off)
{
    (void)H;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
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
    #pragma omp parallel for collapse(2) schedule(static)
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
    #pragma omp parallel for collapse(2) schedule(static)
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
    /* total dst footprint = 3 dst tensors each [B, H, S, dk]. NT triggers
       when collectively > 4 MB so the L3-spill regime gets the savings. */
    int64_t per_dst_bytes = B * H * S * dk * (int64_t)sizeof(float);
    int use_nt = 0;
#if defined(AX_HAS_SIMD)
    use_nt = ax_layout_use_nt(3 * per_dst_bytes)
          && ax_dst_nt_alignable(dstQ, dk)
          && ax_dst_nt_alignable(dstK, dk)
          && ax_dst_nt_alignable(dstV, dk);
#else
    (void)per_dst_bytes;
#endif
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                const float *src_row = src + ((b * S) + s) * src_cols + h * dk;
                int64_t dst_off = (((b * H) + h) * S + s) * dk;
#if defined(AX_HAS_SIMD)
                if (use_nt) {
                    ax_stream_copy_floats(dstQ + dst_off, src_row,         dk);
                    ax_stream_copy_floats(dstK + dst_off, src_row + D,     dk);
                    ax_stream_copy_floats(dstV + dst_off, src_row + 2 * D, dk);
                } else
#endif
                {
                    memcpy(dstQ + dst_off, src_row,           dk_bytes);
                    memcpy(dstK + dst_off, src_row + D,       dk_bytes);
                    memcpy(dstV + dst_off, src_row + 2 * D,   dk_bytes);
                }
            }
        }
    }
#if defined(AX_HAS_SIMD)
    if (use_nt) ax_stream_fence();
#endif
}

void ax_attn_head_interleave_qkv_split_bias(const float *src, const float *bias,
                                             float *dstQ, float *dstK, float *dstV,
                                             int64_t B, int64_t S, int64_t H, int64_t dk,
                                             int64_t D)
{
    int64_t src_cols = 3 * D;
    int64_t TC = AX_VF32_WIDTH;
    int64_t ve = dk - (dk % TC);
    int64_t per_dst_bytes = B * H * S * dk * (int64_t)sizeof(float);
    int use_nt = 0;
#if defined(AX_HAS_SIMD)
    use_nt = ax_layout_use_nt(3 * per_dst_bytes)
          && ax_dst_nt_alignable(dstQ, dk)
          && ax_dst_nt_alignable(dstK, dk)
          && ax_dst_nt_alignable(dstV, dk);
#else
    (void)per_dst_bytes;
#endif
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
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
#if defined(AX_HAS_SIMD)
                if (use_nt) {
                    /* fused add + NT store. compiler reorders the loads
                       to overlap; sfence at the parallel-region exit. */
                    for (; i < ve; i += TC) {
                        ax_vf32_stream(dQ + i, ax_vf32_add(ax_vf32_loadu(src_row +       i), ax_vf32_loadu(bQ_h + i)));
                        ax_vf32_stream(dK + i, ax_vf32_add(ax_vf32_loadu(src_row + D   + i), ax_vf32_loadu(bK_h + i)));
                        ax_vf32_stream(dV + i, ax_vf32_add(ax_vf32_loadu(src_row + 2*D + i), ax_vf32_loadu(bV_h + i)));
                    }
                } else
#endif
                {
                    for (; i < ve; i += TC) {
                        ax_vf32_storeu(dQ + i, ax_vf32_add(ax_vf32_loadu(src_row +       i), ax_vf32_loadu(bQ_h + i)));
                        ax_vf32_storeu(dK + i, ax_vf32_add(ax_vf32_loadu(src_row + D   + i), ax_vf32_loadu(bK_h + i)));
                        ax_vf32_storeu(dV + i, ax_vf32_add(ax_vf32_loadu(src_row + 2*D + i), ax_vf32_loadu(bV_h + i)));
                    }
                }
                for (; i < dk; i++) {
                    dQ[i] = src_row[i] + bQ_h[i];
                    dK[i] = src_row[D + i] + bK_h[i];
                    dV[i] = src_row[2*D + i] + bV_h[i];
                }
            }
        }
    }
#if defined(AX_HAS_SIMD)
    if (use_nt) ax_stream_fence();
#endif
}
