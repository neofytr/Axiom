/* memory.cu — device memory allocation + host<->device transfer hooks.
   implementations of the memory and memcpy_* function pointers in
   ax_backend_ops_t. core routes every storage_alloc, storage_free,
   and cross-device memcpy through these via the device registry in
   dispatch.c. no cuda api ever leaks into src/core/ anymore. */

#include "internal.h"

extern "C" {

/* phase 0 note: we still use cudaMallocManaged here for compatibility
   with the existing behaviour where cpu code may page-fault into gpu
   memory. phase 2 switches this to cudaMalloc + explicit transfers. */

void *cuda_storage_alloc_hook(size_t size_bytes) {
    void *p = NULL;
    if (cudaMallocManaged(&p, size_bytes, cudaMemAttachGlobal) != cudaSuccess)
        return NULL;
    return p;
}

void cuda_storage_free_hook(void *ptr) {
    if (ptr) cudaFree(ptr);
}

ax_status_t cuda_memcpy_h2d_hook(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy H2D failed");
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

ax_status_t cuda_memcpy_d2h_hook(void *dst, const void *src, size_t bytes) {
    /* unified memory is coherent after a device sync; cudaMemcpy
       handles the sync implicitly, but an explicit barrier here is
       cheap insurance while we still use managed memory. */
    cudaDeviceSynchronize();
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy D2H failed");
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

ax_status_t cuda_memcpy_d2d_hook(void *dst, const void *src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        ax_err_set(AX_ERR_BACKEND, "cudaMemcpy D2D failed");
        return AX_ERR_BACKEND;
    }
    return AX_OK;
}

} /* extern "C" */
