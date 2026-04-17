/* axiom/types.h — fundamental type definitions for the axiom framework */

#ifndef AX_TYPES_H
#define AX_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* max tensor dimensions — covers all practical cases */
#define AX_MAX_DIMS 8

/* data types supported by tensors */
typedef enum {
    AX_FLOAT32 = 0,
    AX_FLOAT64,
    AX_INT32,
    AX_INT64,
    AX_UINT8,
    AX_BOOL,
    AX_DTYPE_COUNT
} ax_dtype_t;

/* returns byte size for a given dtype */
static inline size_t ax_dtype_size(ax_dtype_t dtype) {
    static const size_t sizes[] = {
        [AX_FLOAT32] = sizeof(float),
        [AX_FLOAT64] = sizeof(double),
        [AX_INT32]   = sizeof(int32_t),
        [AX_INT64]   = sizeof(int64_t),
        [AX_UINT8]   = sizeof(uint8_t),
        [AX_BOOL]    = sizeof(uint8_t),
    };
    return (dtype < AX_DTYPE_COUNT) ? sizes[dtype] : 0;
}

/* returns human-readable name for a dtype */
static inline const char *ax_dtype_name(ax_dtype_t dtype) {
    static const char *names[] = {
        [AX_FLOAT32] = "float32",
        [AX_FLOAT64] = "float64",
        [AX_INT32]   = "int32",
        [AX_INT64]   = "int64",
        [AX_UINT8]   = "uint8",
        [AX_BOOL]    = "bool",
    };
    return (dtype < AX_DTYPE_COUNT) ? names[dtype] : "unknown";
}

/* status codes returned by all axiom operations */
typedef enum {
    AX_OK = 0,
    AX_ERR_NULL_ARG,
    AX_ERR_ALLOC,
    AX_ERR_SHAPE_MISMATCH,
    AX_ERR_DTYPE_MISMATCH,
    AX_ERR_INVALID_AXIS,
    AX_ERR_INVALID_SHAPE,
    AX_ERR_INVALID_DTYPE,
    AX_ERR_OUT_OF_BOUNDS,
    AX_ERR_BACKEND,
    AX_ERR_NOT_IMPLEMENTED,
    AX_ERR_INTERNAL,
    AX_STATUS_COUNT
} ax_status_t;

/* device placement for tensors */
typedef enum {
    AX_DEVICE_CPU = 0,
    AX_DEVICE_CUDA,
    AX_DEVICE_COUNT
} ax_device_t;

/* memory layout hint for 4D image tensors (N, C, H, W). meaningful only for
   ndim==4 tensors flowing through conv/pool/bn/etc.; ignored for other ranks.
   defaults to NCHW for backward compat. NHWC enables better SIMD for
   pool/bn/elementwise where channels are the innermost dim, and is preferred
   on cpu for vision workloads with large channel counts (≥32). */
typedef enum {
    AX_LAYOUT_NCHW = 0,  /* default; [N, C, H, W] for 4D tensors */
    AX_LAYOUT_NHWC = 1,  /* [N, H, W, C] — channels innermost */
    AX_LAYOUT_COUNT
} ax_layout_t;

static inline const char *ax_layout_name(ax_layout_t l) {
    static const char *names[] = {
        [AX_LAYOUT_NCHW] = "NCHW",
        [AX_LAYOUT_NHWC] = "NHWC",
    };
    return (l < AX_LAYOUT_COUNT) ? names[l] : "unknown";
}

/* returns human-readable name for a status code */
static inline const char *ax_status_name(ax_status_t s) {
    static const char *names[] = {
        [AX_OK]                 = "ok",
        [AX_ERR_NULL_ARG]       = "null argument",
        [AX_ERR_ALLOC]          = "allocation failed",
        [AX_ERR_SHAPE_MISMATCH] = "shape mismatch",
        [AX_ERR_DTYPE_MISMATCH] = "dtype mismatch",
        [AX_ERR_INVALID_AXIS]   = "invalid axis",
        [AX_ERR_INVALID_SHAPE]  = "invalid shape",
        [AX_ERR_INVALID_DTYPE]  = "invalid dtype",
        [AX_ERR_OUT_OF_BOUNDS]  = "out of bounds",
        [AX_ERR_BACKEND]        = "backend error",
        [AX_ERR_NOT_IMPLEMENTED]= "not implemented",
        [AX_ERR_INTERNAL]       = "internal error",
    };
    return (s < AX_STATUS_COUNT) ? names[s] : "unknown error";
}

#endif /* AX_TYPES_H */
