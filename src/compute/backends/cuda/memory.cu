/* memory.cu — device memory allocation + host<->device transfer hooks.
   implementations of the memory and memcpy_* function pointers in
   ax_backend_ops_t. core routes every storage_alloc, storage_free,
   and cross-device memcpy through these via the device registry in
   dispatch.c. no cuda api ever leaks into src/core/ anymore. */

#include "internal.h"

extern "C" {

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
    /* zero by default so ax_tensor_zeros semantics hold even if the
       caller never touches the tensor through fill. cudaMalloc does
       not guarantee zero-initialised memory. */
    if (cudaMemset(p, 0, size_bytes) != cudaSuccess) {
        cudaFree(p);
        return NULL;
    }
    return p;
}

void cuda_storage_free_hook(void *ptr) {
    if (ptr) cudaFree(ptr);
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
