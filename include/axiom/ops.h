/* axiom/ops.h — high-level tensor operations.
   these handle output allocation, broadcasting, and (later) autograd recording.
   this is what user code should call, not the compute layer directly. */

#ifndef AX_OPS_H
#define AX_OPS_H

#include "tensor.h"

/* binary ops (with broadcasting) */

ax_tensor_t *ax_add(ax_tensor_t *a, ax_tensor_t *b);
ax_tensor_t *ax_sub(ax_tensor_t *a, ax_tensor_t *b);
ax_tensor_t *ax_mul(ax_tensor_t *a, ax_tensor_t *b);
ax_tensor_t *ax_div(ax_tensor_t *a, ax_tensor_t *b);

/* unary ops */

ax_tensor_t *ax_neg(ax_tensor_t *a);
ax_tensor_t *ax_abs(ax_tensor_t *a);
ax_tensor_t *ax_exp(ax_tensor_t *a);
ax_tensor_t *ax_log(ax_tensor_t *a);
ax_tensor_t *ax_sqrt(ax_tensor_t *a);
ax_tensor_t *ax_square(ax_tensor_t *a);

/* scalar ops */

ax_tensor_t *ax_add_scalar(ax_tensor_t *a, double s);
ax_tensor_t *ax_mul_scalar(ax_tensor_t *a, double s);

/* matrix multiply */
/* for 2d tensors: standard matmul.
   both inputs must be 2d for now. */
ax_tensor_t *ax_matmul(ax_tensor_t *a, ax_tensor_t *b);

/* fused matmul + bias: out = (a @ b) + bias.
   single output tensor + single grad_fn. backward computes dA, dB, and
   dBias in one traversal step. eliminates the extra tensor allocation +
   extra autograd node that a separate ax_add(matmul_result, bias) would
   create. bias may be NULL for matmul-only (equivalent to ax_matmul). */
ax_tensor_t *ax_matmul_bias(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *bias);

/* fused matmul + bias + relu: out = relu((a @ b) + bias).
   when the backend provides gemm_relu, the relu is applied during the
   gemm writeback step — saving one full memory pass over the output.
   falls back to matmul_bias + separate relu when gemm_relu is absent.
   bias may be NULL. */
ax_tensor_t *ax_matmul_bias_relu(ax_tensor_t *a, ax_tensor_t *b, ax_tensor_t *bias);

/* reductions */
/* axis = -1 reduces everything to a scalar */

ax_tensor_t *ax_sum(ax_tensor_t *a, int axis);
ax_tensor_t *ax_mean(ax_tensor_t *a, int axis);
ax_tensor_t *ax_max(ax_tensor_t *a, int axis);
ax_tensor_t *ax_min(ax_tensor_t *a, int axis);

/* activations */

ax_tensor_t *ax_relu(ax_tensor_t *a);
ax_tensor_t *ax_sigmoid(ax_tensor_t *a);
ax_tensor_t *ax_tanh_op(ax_tensor_t *a);

/* in-place activation variants. these mutate the input buffer and return
   the same tensor pointer, avoiding one allocation + one full pass.
   preconditions: input storage refcount must be 1 (no shared views).
   if refcount > 1 these fall back to the non-inplace variant for safety.
   autograd-compatible: relu/sigmoid/tanh backward all work with the
   post-activation values (output), which inplace preserves. */

ax_tensor_t *ax_relu_inplace(ax_tensor_t *a);
ax_tensor_t *ax_sigmoid_inplace(ax_tensor_t *a);
ax_tensor_t *ax_tanh_inplace(ax_tensor_t *a);

/* comparison (returns float tensor with 0/1 values) */

ax_tensor_t *ax_equal(ax_tensor_t *a, ax_tensor_t *b);
ax_tensor_t *ax_greater(ax_tensor_t *a, ax_tensor_t *b);

#endif /* AX_OPS_H */
