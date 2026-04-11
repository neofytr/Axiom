/* axiom/cuda.h — public API for the NVIDIA GPU backend */

#ifndef AX_CUDA_H
#define AX_CUDA_H

#include "types.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef AX_HAVE_CUDA

/* returns the number of available CUDA devices (0 if none) */
int ax_cuda_device_count(void);

/* block until all pending GPU operations on the default stream complete */
void ax_cuda_synchronize(void);

/* allocate a zero tensor directly on GPU without going through CPU first */
ax_tensor_t *ax_tensor_cuda_zeros(const int64_t *shape, int ndim, ax_dtype_t dtype);

#endif /* AX_HAVE_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* AX_CUDA_H */
