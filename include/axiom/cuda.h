/* axiom/cuda.h — legacy cuda-specific helpers.
   kept as a thin compatibility shim over axiom/device.h. prefer the
   generic ax_device_* api for new code; these names may be removed in
   a future release. */

#ifndef AX_CUDA_H
#define AX_CUDA_H

#include "types.h"
#include "tensor.h"
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* equivalent to ax_device_count(AX_DEVICE_CUDA). */
static inline int ax_cuda_device_count(void) {
    return ax_device_count(AX_DEVICE_CUDA);
}

/* equivalent to ax_device_synchronize(AX_DEVICE_CUDA). */
static inline void ax_cuda_synchronize(void) {
    ax_device_synchronize(AX_DEVICE_CUDA);
}

#ifdef AX_HAVE_CUDA
/* allocate a zero tensor directly on gpu without going through cpu first. */
ax_tensor_t *ax_tensor_cuda_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype);
#endif

#ifdef __cplusplus
}
#endif

#endif /* AX_CUDA_H */
