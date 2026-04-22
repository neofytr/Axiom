/* quantize.h — int8 weight quantization for inference.

   step 7: per-channel symmetric int8 quantization of fp32 weight tensors.
   intended for the embedded niche: halves weight storage, ~2x speedup on
   bandwidth-bound shapes (Linear with K large), and unblocks deployment to
   targets where holding the fp32 model in memory is too expensive.

   convention: W8A32 (int8 weights, fp32 activations).
     - quantization is symmetric (no zero-point).
     - scale is fp32, one per output channel (matches the way Linear and
       Conv outputs are reduced over K).
     - dequant: w_fp32 = q_int8 * scale[output_channel]

   the kernel can either (a) dequant on-the-fly inside the FMA loop —
   trades per-K-iter int8→fp32 conversion for 4× less weight bandwidth —
   or (b) pre-compute fp32 weights once and call the existing fp32 GEMM
   (no BW win, but uses the JIT'd kernel). path (a) is what we ship
   for the inference fast path. */

#ifndef AX_QUANTIZE_H
#define AX_QUANTIZE_H

/* ownership: ax_qweight_create_from_fp32 returns a heap qweight owned
   by the caller; release with ax_qweight_destroy. ax_qweight_dequantize
   writes into a caller-supplied fp32 buffer; no allocation.

   error handling: ax_qweight_create_from_fp32 returns NULL on alloc
   failure. ax_qgemm_w8a32 returns AX_ERR_INVALID for shape mismatch,
   AX_ERR_NULL_ARG for null inputs, AX_OK on success.

   thread-safety: qweight contents are immutable post-construction —
   ax_qgemm_w8a32 is concurrent-safe across distinct (a, out) buffers
   on the same qweight. ax_qweight_dequantize is read-only on the
   qweight and writes a unique buffer per call. */

#include "types.h"
#include "tensor.h"
#include "error.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* int8-quantized weight matrix shaped [N, K] (output channels × input dim).
   stored row-major. each row has its own fp32 scale s.t.
     w_fp32[n, k] ≈ qw->data[n*K + k] * qw->scales[n]
   the scale is computed at quantization time as max(|w_fp32|) / 127. */
typedef struct {
    int8_t  *data;     /* [N, K] int8 weights */
    float   *scales;   /* [N] fp32 per-output-channel scale */
    int64_t  N;        /* output channels */
    int64_t  K;        /* input dim */
} ax_qweight_t;

/* quantize an [N, K] fp32 weight matrix to int8 with per-channel scales.
   returns NULL on alloc failure. caller owns the result; free with
   ax_qweight_destroy. */
ax_qweight_t *ax_qweight_create_from_fp32(const float *fp32_w, int64_t N, int64_t K);

/* free a qweight. */
void ax_qweight_destroy(ax_qweight_t *qw);

/* dequantize back to fp32 (for testing / fallback). out must be [N, K]. */
void ax_qweight_dequantize(const ax_qweight_t *qw, float *out_fp32);

/* W8A32 GEMM: out[M, N] = a[M, K] @ qw->data[N, K]^T   (with per-row scale).
   shape contract:
     - a is row-major [M, K] fp32
     - qw is [N, K] int8 with [N] scales
     - out is row-major [M, N] fp32
   semantics matches: out[m, n] = sum_k a[m, k] * (qw->data[n, k] * qw->scales[n])
                                = qw->scales[n] * sum_k a[m, k] * qw->data[n, k] */
ax_status_t ax_qgemm_w8a32(
    const float *a,
    const ax_qweight_t *qw,
    float *out,
    int64_t M);

#ifdef __cplusplus
}
#endif

#endif /* AX_QUANTIZE_H */
