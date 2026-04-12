import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/architecture/autograd')({
  component: AutogradPage,
})

function AutogradPage() {
  return (
    <>
      <h1>Autograd Engine</h1>
      <p>
        Axiom implements reverse-mode automatic differentiation. During the forward pass, every
        differentiable operation records itself into a computation graph. The backward pass walks
        this graph in reverse to accumulate gradients. This page covers how that works internally.
      </p>

      <h2>The computation graph</h2>
      <p>
        When gradient tracking is enabled and an operation's inputs have <code>requires_grad = true</code>,
        the op creates a <code>grad_fn</code> node and attaches it to the output tensor.
        Each <code>grad_fn</code> contains:
      </p>
      <pre><code className="language-c">{`struct ax_grad_fn {
    ax_backward_fn_t backward;   // the backward function pointer

    // inputs to the forward op (for routing gradients back)
    ax_tensor_t *inputs[AX_GRAD_MAX_INPUTS];  // max 2
    int n_inputs;

    // tensors saved during forward that backward needs
    ax_tensor_t *saved[AX_GRAD_MAX_SAVED];    // max 4
    bool saved_owned[AX_GRAD_MAX_SAVED];      // cleanup ownership
    int n_saved;

    // extra context
    double scalar_ctx;  // e.g., the scalar in mul_scalar
    int int_ctx;        // e.g., the axis for sum
    void *ctx;          // opaque pointer for complex ops (conv2d layer)
    void (*ctx_cleanup)(void *ctx);
};`}</code></pre>
      <p>
        The graph is implicit: it's a DAG of tensors connected through their <code>grad_fn</code>
        pointers. Leaf tensors (parameters, user-created tensors) have <code>grad_fn = NULL</code>.
        Intermediate tensors (results of ops) have a <code>grad_fn</code> whose <code>inputs[]</code>
        point back to the op's input tensors.
      </p>

      <h2>How backward works</h2>
      <p>
        <code>ax_backward(loss)</code> does three things:
      </p>
      <ol>
        <li>
          <strong>Topological sort.</strong> A DFS from the loss tensor, following{' '}
          <code>grad_fn&gt;inputs[]</code> edges, builds a reverse topological ordering.
          A visited set (pointer comparison) prevents revisiting shared subgraphs.
        </li>
        <li>
          <strong>Seed the gradient.</strong> The loss tensor's grad is set to 1.0 (the derivative
          of the loss with respect to itself).
        </li>
        <li>
          <strong>Walk backward.</strong> For each node in reverse topological order, call{' '}
          <code>grad_fn&gt;backward(grad_fn, output_grad)</code>. The backward function computes
          the local Jacobian-vector product and accumulates the result into each input's{' '}
          <code>.grad</code> tensor.
        </li>
      </ol>

      <h2>Gradient accumulation</h2>
      <p>
        When a tensor is used in multiple operations (e.g., a weight matrix used in two different
        matmuls), its gradient is accumulated: each backward function <em>adds</em> to the existing
        grad rather than overwriting it. Fresh gradients (first write) skip the accumulate step
        and write directly, avoiding an unnecessary read-add-write pass.
      </p>

      <h2>Memory management: slab allocator</h2>
      <p>
        Creating a <code>grad_fn</code> for every operation during training would normally mean one
        malloc per op per forward pass. Axiom avoids this with a thread-local slab free-list:
      </p>
      <pre><code className="language-c">{`// freed grad_fns are pushed onto a free-list
static _Thread_local ax_grad_fn_t *grad_fn_freelist = NULL;

ax_grad_fn_t *ax_grad_fn_create(ax_backward_fn_t fn) {
    ax_grad_fn_t *gf = grad_fn_freelist;
    if (gf) {
        grad_fn_freelist = *(ax_grad_fn_t **)gf;  // pop
        memset(gf, 0, sizeof(ax_grad_fn_t));
        return gf;
    }
    return calloc(1, sizeof(ax_grad_fn_t));  // cold path
}

void ax_grad_fn_destroy(ax_grad_fn_t *gf) {
    *(ax_grad_fn_t **)gf = grad_fn_freelist;  // push
    grad_fn_freelist = gf;
}`}</code></pre>
      <p>
        After the first epoch, the free-list is warm and every <code>grad_fn_create</code> is just
        a pointer pop + memset. No syscalls, no lock contention (thread-local), no fragmentation.
      </p>

      <h2>Scratch arenas</h2>
      <p>
        The autograd system provides two bump arenas for temporary tensor allocations:
      </p>
      <ul>
        <li>
          <strong>Backward arena</strong> (<code>ax_backward_arena()</code>) — for scratch buffers
          during the backward pass. 16 MB default. Reset after <code>ax_backward()</code> completes.
          Backward functions use this for temporary gradient tensors that don't need to survive
          beyond the current backward step.
        </li>
        <li>
          <strong>Forward arena</strong> (<code>ax_forward_arena()</code>) — for tensors that the
          forward pass saves for backward (batchnorm's x_hat, inv_std, etc). These must survive
          the forward pass and stay valid through the backward walk. Reset only by{' '}
          <code>ax_graph_cleanup()</code>, not by <code>ax_backward()</code>.
        </li>
      </ul>
      <p>
        Arena allocation is a pointer bump with no per-object free. A single{' '}
        <code>ax_arena_reset()</code> call invalidates everything. For a training step with hundreds
        of intermediate tensors, this saves hundreds of free() calls.
      </p>
      <p>
        Important: arena pointers must never be stored in <code>grad_fn&gt;saved[]</code> from the
        backward arena, because they become stale after reset. The forward arena is safe for
        saved tensors since it outlives the backward pass.
      </p>

      <h2>Nestable no-grad</h2>
      <p>
        The no-grad context uses a depth counter, not a boolean toggle:
      </p>
      <pre><code className="language-c">{`static _Thread_local int no_grad_depth = 0;

void ax_no_grad(void)    { no_grad_depth++; }
void ax_enable_grad(void) {
    if (no_grad_depth > 0) no_grad_depth--;
}`}</code></pre>
      <p>
        This means library code can safely call <code>ax_no_grad()</code> without clobbering
        the caller's no-grad scope. Each <code>ax_no_grad()</code> must be paired with an{' '}
        <code>ax_enable_grad()</code>.
      </p>

      <h2>Graph cleanup</h2>
      <p>
        After backward and optimizer step, call <code>ax_graph_cleanup(loss)</code> to free the
        computation graph. This destroys all intermediate (non-leaf) tensors created during the
        forward pass. Leaf tensors (parameters) are not touched. The typical training loop:
      </p>
      <pre><code className="language-c">{`ax_backward(loss);
ax_optimizer_step(opt);
ax_graph_cleanup(loss);   // frees intermediates, resets forward arena
ax_tensor_destroy(loss);  // frees the scalar loss tensor itself`}</code></pre>
      <p>
        If you call <code>ax_tensor_destroy(loss)</code> without <code>ax_graph_cleanup</code>,
        all intermediate tensors from the forward pass will leak.
      </p>

      <h2>Inference-only builds</h2>
      <p>
        When compiled with <code>AX_INFERENCE_ONLY</code>, the autograd engine is completely
        stripped. <code>ax_grad_enabled()</code> is an inline function that returns false, so
        the compiler constant-folds away all grad_fn setup branches in the forward ops.{' '}
        <code>ax_grad_fn_create()</code> returns NULL. Any code that calls{' '}
        <code>ax_backward()</code> gets a link error.
      </p>
    </>
  )
}
