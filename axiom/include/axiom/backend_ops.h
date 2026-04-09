/* axiom/backend_ops.h — vtable interface that every compute backend must implement */

#ifndef AX_BACKEND_OPS_H
#define AX_BACKEND_OPS_H

#include "types.h"

/* forward declaration — tensor is defined in tensor.h */
typedef struct ax_tensor ax_tensor_t;

/* backend operation vtable.
   each backend (cpu naive, cpu simd, cuda, etc.) provides one of these
   with function pointers filled in for every supported operation. */
typedef struct {
    const char *name;   /* human-readable backend name, e.g. "cpu_naive" */

    /* --- element-wise binary ops --- */
    /* all binary ops support broadcasting */
    ax_status_t (*add)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*sub)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*mul)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*div_op)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* --- element-wise unary ops --- */
    ax_status_t (*neg)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*abs_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*exp_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*log_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*sqrt_op)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*square)(const ax_tensor_t *in, ax_tensor_t *out);

    /* --- scalar ops --- */
    ax_status_t (*add_scalar)(const ax_tensor_t *in, double scalar, ax_tensor_t *out);
    ax_status_t (*mul_scalar)(const ax_tensor_t *in, double scalar, ax_tensor_t *out);

    /* --- matrix ops --- */
    /* general matrix multiply: out = a @ b */
    ax_status_t (*gemm)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* --- reduction ops --- */
    /* axis=-1 means reduce all dimensions */
    ax_status_t (*sum)(const ax_tensor_t *in, int axis, ax_tensor_t *out);
    ax_status_t (*mean)(const ax_tensor_t *in, int axis, ax_tensor_t *out);
    ax_status_t (*max_op)(const ax_tensor_t *in, int axis, ax_tensor_t *out);
    ax_status_t (*min_op)(const ax_tensor_t *in, int axis, ax_tensor_t *out);

    /* --- comparison ops --- */
    ax_status_t (*equal)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*greater)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);

    /* --- data movement --- */
    /* fill all elements with a scalar value */
    ax_status_t (*fill)(ax_tensor_t *t, double value);
    /* copy data from src to dst (same shape required) */
    ax_status_t (*copy)(const ax_tensor_t *src, ax_tensor_t *dst);

    /* --- activation functions (fused for performance) --- */
    ax_status_t (*relu)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*sigmoid)(const ax_tensor_t *in, ax_tensor_t *out);
    ax_status_t (*tanh_op)(const ax_tensor_t *in, ax_tensor_t *out);

} ax_backend_ops_t;

#endif /* AX_BACKEND_OPS_H */
