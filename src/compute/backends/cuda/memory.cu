/* memory.cu — device memory allocation + host<->device transfer hooks.
   implementations of the memory and memcpy_* function pointers in
   ax_backend_ops_t. core routes every storage_alloc, storage_free,
   and cross-device memcpy through these via the device registry in
   dispatch.c. no cuda api ever leaks into src/core/ anymore. */

#include "internal.h"
#include <stdio.h>
#include <stdlib.h>

/* alloc/free counters for leak diagnosis. query from test code via
   ax_cuda_alloc_stats. compiled unconditionally; zero cost when not read. */
static int64_t g_cuda_allocs = 0;
static int64_t g_cuda_frees  = 0;
static int64_t g_cuda_bytes_alloc = 0;
static int64_t g_cuda_bytes_free  = 0;

extern "C" {

void ax_cuda_alloc_stats(int64_t *allocs, int64_t *frees,
                          int64_t *bytes_alloc, int64_t *bytes_free) {
    if (allocs) *allocs = g_cuda_allocs;
    if (frees) *frees = g_cuda_frees;
    if (bytes_alloc) *bytes_alloc = g_cuda_bytes_alloc;
    if (bytes_free) *bytes_free = g_cuda_bytes_free;
}
void ax_cuda_reset_alloc_stats(void) {
    g_cuda_allocs = g_cuda_frees = g_cuda_bytes_alloc = g_cuda_bytes_free = 0;
}

/* explicit device memory model.

   we allocate with cudaMalloc (not cudaMallocManaged). this means the
   returned pointer is device-only — dereferencing it from the host
   segfaults. every host-side touch must go through memcpy_h2d /
   memcpy_d2h, and every op that needs to read or write tensor data
   must dispatch through the backend vtable.

   this is the right model for real training throughput on discrete
   gpus. managed memory silently page-faults on every cpu access,
   which is convenient for development but kills perf under load.
   core/tensor.c's host helpers (tensor_fill_value,
   tensor_write_from_host, ax_tensor_print, ax_tensor_get_f32) all
   route through the backend registry, so host-visible apis keep
   working at the cost of an explicit transfer per call. */

void *cuda_storage_alloc_hook(size_t size_bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, size_bytes) != cudaSuccess) return NULL;
    if (cudaMemset(p, 0, size_bytes) != cudaSuccess) {
        cudaFree(p);
        return NULL;
    }
    g_cuda_allocs++;
    g_cuda_bytes_alloc += (int64_t)size_bytes;
    if (getenv("AX_TRACE_CUDA_ALLOC"))
        fprintf(stderr, "  CUDA_ALLOC %p size=%zu\n", p, size_bytes);
    return p;
}

void cuda_storage_free_hook(void *ptr) {
    if (ptr) {
        g_cuda_frees++;
        if (getenv("AX_TRACE_CUDA_ALLOC"))
            fprintf(stderr, "  CUDA_FREE  %p\n", ptr);
        cudaFree(ptr);
    }
}

ax_status_t cuda_memcpy_h2d_hook(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy H2D failed: %s",
                   cudaGetErrorString(cudaGetLastError()));
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

ax_status_t cuda_memcpy_d2h_hook(void *dst, const void *src, size_t bytes) {
    /* cudaMemcpy D2H on the default stream is synchronous w.r.t. the
       host and already observes in-flight kernels, so no explicit
       barrier needed. */
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy D2H failed: %s",
                   cudaGetErrorString(cudaGetLastError()));
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

ax_status_t cuda_memcpy_d2d_hook(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy D2D failed: %s",
                   cudaGetErrorString(cudaGetLastError()));
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

} /* extern "C" */
