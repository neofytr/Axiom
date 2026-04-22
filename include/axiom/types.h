/* axiom/types.h — fundamental type definitions for the axiom framework. */

#ifndef AX_TYPES_H
#define AX_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* version number — bumped per the SemVer rules:
     MAJOR  — incompatible api change (re-tag all functions abi-stable
              again at the next major).
     MINOR  — new functionality added in a backward-compatible manner.
     PATCH  — backward-compatible bug fixes.
   the integer constant AX_VERSION packs MM*10000 + mm*100 + pp so
   compile-time `#if AX_VERSION >= 1000` checks work. */
#define AX_VERSION_MAJOR  0
#define AX_VERSION_MINOR  10
#define AX_VERSION_PATCH  0
#define AX_VERSION (AX_VERSION_MAJOR * 10000 + AX_VERSION_MINOR * 100 + AX_VERSION_PATCH)
#define AX_VERSION_STRING "0.10.0"

/* abi stability tag. every public function in include/axiom/<name>.h
   (excluding axiom/internal/) is implicitly tagged AX_ABI_STABLE_SINCE("0.10.0").
   future minor versions may not break or remove a tagged symbol; only
   a major bump (1.0.0 -> 2.0.0) may. removals must go through a
   deprecation cycle:
       1) tag the symbol AX_DEPRECATED with a hint to its replacement;
       2) keep both names available for at least one minor release;
       3) remove the old name only at the next major bump.
   the AX_DEPRECATED macro emits a compiler warning at every call site
   so users can migrate at their own pace. */
#if defined(__GNUC__) || defined(__clang__)
  #define AX_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
  #define AX_DEPRECATED(msg) __declspec(deprecated(msg))
#else
  #define AX_DEPRECATED(msg)
#endif

/* purely documentary — does not affect codegen. usage:
       AX_ABI_STABLE_SINCE("0.10.0") ax_status_t ax_compute_init(void);
   most public functions inherit the implicit baseline (0.10.0) above
   and don't need this tag explicitly; mark only when the symbol's
   guarantee differs from the baseline (e.g. "stable since 0.11.0"). */
#define AX_ABI_STABLE_SINCE(version)

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
