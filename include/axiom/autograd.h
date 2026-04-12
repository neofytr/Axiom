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

    /* tensors saved during forward that backward needs.
       saved_owned[i] indicates whether the grad_fn owns saved[i] and
       should destroy it on cleanup (true for helper tensors like the
       softmax output in cross-entropy, false for tensors that are
       also graph nodes or user-owned). */
    ax_tensor_t *saved[AX_GRAD_MAX_SAVED];
    bool saved_owned[AX_GRAD_MAX_SAVED];
    int n_saved;

    /* extra scalar context some ops need (e.g., mul_scalar stores the scalar) */
    double scalar_ctx;
    int int_ctx; /* e.g., axis for sum */

    /* opaque pointer for ops that need more context (e.g., conv2d stores its layer).
       ctx_cleanup is called to free ctx if backward was never called. if backward
       handles its own cleanup, it should set ctx = NULL afterward. */
    void *ctx;
    void (*ctx_cleanup)(void *ctx);
};

#ifndef AX_INFERENCE_ONLY

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
   are NOT destroyed. the root tensor itself is NOT destroyed — the caller owns it.

   typical training loop usage:
     ax_backward(loss);
     ax_optimizer_step(opt);
     ax_graph_cleanup(loss);   // frees intermediates, detaches root from graph
     ax_tensor_destroy(loss);  // frees the root (scalar loss) itself

   if ax_tensor_destroy(loss) is called without ax_graph_cleanup first,
   all intermediate tensors from the forward pass will leak. */
void ax_graph_cleanup(ax_tensor_t *root);

/* gradient context: disable/enable gradient tracking.
   useful during inference when you don't want the overhead. */
void ax_no_grad(void);
void ax_enable_grad(void);
bool ax_grad_enabled(void);

/* allocate a grad_fn (uses thread-local slab free-list) */
ax_grad_fn_t *ax_grad_fn_create(ax_backward_fn_t fn);

/* free a grad_fn (returns to thread-local slab free-list) */
void ax_grad_fn_destroy(ax_grad_fn_t *gf);

#else /* AX_INFERENCE_ONLY */

/* inference-only stubs: grad tracking is permanently disabled. forward
   files compile unchanged — the compiler constant-folds the returned
   false and dce's all grad_fn setup branches. ax_grad_fn_create is
   provided as a NULL stub so the forward files' unused grad_fn code
   paths still typecheck but never execute. */
static inline bool  ax_grad_enabled(void)       { return false; }
static inline void  ax_no_grad(void)            { }
static inline void  ax_enable_grad(void)        { }
static inline ax_grad_fn_t *ax_grad_fn_create(ax_backward_fn_t fn) {
    (void)fn; return NULL;
}
static inline void  ax_grad_fn_destroy(ax_grad_fn_t *gf) { (void)gf; }

/* training-only entry points: declared but undefined. any inference-only
   user code that calls these gets a link error, which is the clearest
   signal that this build doesn't support training. */

#endif /* AX_INFERENCE_ONLY */

/* get the thread-local backward scratch arena. backward functions can
   use this for temporary buffers that are freed in bulk after ax_backward().
   do NOT store arena pointers in grad_fn->saved[] — they become stale after reset. */
#include "memory.h"
ax_arena_t *ax_backward_arena(void);

/* get the thread-local forward scratch arena. forward fns can use this for
   tensors that must outlive the forward pass and stay alive across backward
   (e.g. batchnorm's saved x_hat / inv_std). reset by ax_graph_cleanup, NOT by
   ax_backward, so saved tensors remain valid through the backward walk.
   caveat: arena spans multiple forward calls and is reset only at cleanup, so
   if two forwards are issued without an intervening cleanup the second forward's
   allocations coexist with the first — that's fine for normal training where
   each step does forward -> backward -> cleanup before the next forward. */
ax_arena_t *ax_forward_arena(void);

/* numerical gradient check for testing.
   compares analytical gradient against finite-difference approximation.
   returns the max absolute difference. */
double ax_grad_check(
    ax_tensor_t *(*forward_fn)(ax_tensor_t *input),
    ax_tensor_t *input,
    double eps);

#endif /* AX_AUTOGRAD_H */
