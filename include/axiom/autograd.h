/* axiom/autograd.h — reverse-mode automatic differentiation engine.
   records operations as they happen, then walks backwards
   through the computation graph to accumulate gradients. */

#ifndef AX_AUTOGRAD_H
#define AX_AUTOGRAD_H

#include "tensor.h"

/* max number of inputs a single operation can have.
   most ops are unary or binary so 2 is fine;
   bump this if we ever need ternary ops */
#define AX_GRAD_MAX_INPUTS 2

/* max saved tensors per backward fn (some ops need to
   stash their inputs/outputs for the backward pass) */
#define AX_GRAD_MAX_SAVED 3

/* forward declaration so the fn pointer type can reference it */
typedef struct ax_grad_fn ax_grad_fn_t;

/* backward function type: given the output gradient,
   compute and accumulate gradients for each input */
typedef void (*ax_backward_fn_t)(ax_grad_fn_t *self, ax_tensor_t *grad_output);

/* grad_fn: one of these is attached to every tensor that
   was produced by a differentiable operation */
struct ax_grad_fn
{
    ax_backward_fn_t backward;

    /* inputs to the forward op (we need these to route gradients back) */
    ax_tensor_t *inputs[AX_GRAD_MAX_INPUTS];
    int n_inputs;

    /* tensors saved during forward that backward needs
       (e.g., matmul saves both inputs, sigmoid saves output) */
    ax_tensor_t *saved[AX_GRAD_MAX_SAVED];
    int n_saved;

    /* extra scalar context some ops need (e.g., mul_scalar stores the scalar) */
    double scalar_ctx;
    int int_ctx; /* e.g., axis for sum */

    /* opaque pointer for ops that need more context (e.g., conv2d stores its layer) */
    void *ctx;
};

/* run backward pass starting from this tensor.
   tensor should be a scalar (1 element).
   computes gradients for all tensors in the graph that
   have requires_grad=true. */
ax_status_t ax_backward(ax_tensor_t *loss);

/* zero out the gradient of a tensor (call before each training step) */
void ax_zero_grad(ax_tensor_t *t);

/* free the computation graph reachable from this tensor.
   destroys all intermediate (non-leaf) tensors created during the forward pass.
   leaf tensors (those without grad_fn, i.e. parameters and user-created tensors)
   are NOT destroyed. call this after backward to prevent memory leaks. */
void ax_graph_cleanup(ax_tensor_t *root);

/* gradient context: disable/enable gradient tracking.
   useful during inference when you don't want the overhead. */
void ax_no_grad(void);
void ax_enable_grad(void);
bool ax_grad_enabled(void);

/* allocate a grad_fn (just a malloc + zeroing) */
ax_grad_fn_t *ax_grad_fn_create(ax_backward_fn_t fn);

/* numerical gradient check for testing.
   compares analytical gradient against finite-difference approximation.
   returns the max absolute difference. */
double ax_grad_check(
    ax_tensor_t *(*forward_fn)(ax_tensor_t *input),
    ax_tensor_t *input,
    double eps);

#endif /* AX_AUTOGRAD_H */
