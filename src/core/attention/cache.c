/* attention/cache.c — fused QKV weight + bias cache management.

   the layer keeps three separate weight tensors (Wq/Wk/Wv [D, D] each)
   so autograd can track them as discrete parameters. the forward pass
   prefers a single fused [D, 3D] gemm over three D-wide gemms — saves
   2/3 of the weight-matrix traffic. this file caches the concatenated
   panel and rebuilds it on demand.

   staleness check uses ax_storage_t's monotonic generation counter:
   any in-place write to Wq/Wk/Wv (e.g. an optimizer step) bumps the
   counter via ax_storage_touch, so the cache invalidates without us
   needing to wrap every mutator. biases (3*D floats) are cheap to
   rebuild so they're refreshed unconditionally. */

#include "internal.h"
#include "axiom/tensor.h"
#include "axiom/internal/cuda_extension.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void ax_attn_refresh_fused_qkv(ax_mha_t *m)
{
    uint64_t gq = m->Wq->storage->generation;
    uint64_t gk = m->Wk->storage->generation;
    uint64_t gv = m->Wv->storage->generation;

    int64_t D = m->d_model;

    /* bump weight panel if anything changed OR cache doesn't exist */
    bool w_stale = !m->Wqkv_cache || gq != m->Wq_gen || gk != m->Wk_gen || gv != m->Wv_gen;
    if (w_stale) {
        if (!m->Wqkv_cache) {
            int64_t sh[] = {D, 3 * D};
            m->Wqkv_cache = ax_tensor_create(sh, 2, AX_FLOAT32);
        }
        float *dst = (float *)m->Wqkv_cache->storage->data;
        const float *wq = (const float *)m->Wq->storage->data;
        const float *wk = (const float *)m->Wk->storage->data;
        const float *wv = (const float *)m->Wv->storage->data;
        /* k.4: dispatch on device through the cuda extension registry. */
        const ax_cuda_extension_t *cu = ax_compute_get_cuda_extension();
        bool cache_on_cuda = (m->Wqkv_cache->storage->device != AX_DEVICE_CPU);
        if (cache_on_cuda && cu && cu->qkv_cache_build) {
            cu->qkv_cache_build(dst, wq, wk, wv, D);
        } else {
            /* [D, 3D] row-major: row i -> [wq[i,:] | wk[i,:] | wv[i,:]] */
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < D; i++) {
                memcpy(dst + i * 3 * D,            wq + i * D, (size_t)D * sizeof(float));
                memcpy(dst + i * 3 * D + D,        wk + i * D, (size_t)D * sizeof(float));
                memcpy(dst + i * 3 * D + 2 * D,    wv + i * D, (size_t)D * sizeof(float));
            }
        }
        ax_storage_touch(m->Wqkv_cache->storage);
        m->Wq_gen = gq; m->Wk_gen = gk; m->Wv_gen = gv;
    }

    /* biases: rebuild unconditionally when used — cheap (3D floats) */
    if (m->use_bias) {
        if (!m->bqkv_cache) {
            int64_t sh[] = {3 * D};
            m->bqkv_cache = ax_tensor_create(sh, 1, AX_FLOAT32);
        }
        float *dst = (float *)m->bqkv_cache->storage->data;
        /* k.4: dispatch on device through the cuda extension registry. */
        const ax_cuda_extension_t *cu = ax_compute_get_cuda_extension();
        bool b_on_cuda = (m->bqkv_cache->storage->device != AX_DEVICE_CPU);
        if (b_on_cuda && cu && cu->bias_cache_build) {
            cu->bias_cache_build(dst,
                (const float *)m->bq->storage->data,
                (const float *)m->bk->storage->data,
                (const float *)m->bv->storage->data, D);
        } else {
            memcpy(dst,         m->bq->storage->data, (size_t)D * sizeof(float));
            memcpy(dst + D,     m->bk->storage->data, (size_t)D * sizeof(float));
            memcpy(dst + 2 * D, m->bv->storage->data, (size_t)D * sizeof(float));
        }
        ax_storage_touch(m->bqkv_cache->storage);
    }
}
