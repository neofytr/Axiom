import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/autograd')({
  component: AutogradApiPage,
})

function AutogradApiPage() {
  return (
    <>
      <h1>Autograd API</h1>
      <p>
        Reverse-mode automatic differentiation. Records operations during the forward pass
        and walks the graph backward to compute gradients. For the internal design, see the{' '}
        <a href="/docs/architecture/autograd">Autograd Engine</a> architecture page.
      </p>
      <p>Header: <code>axiom/autograd.h</code></p>

      <h2>Backward pass</h2>

      <h3>ax_backward</h3>
      <pre><code className="language-c">{`ax_status_t ax_backward(ax_tensor_t *loss);`}</code></pre>
      <p>
        Runs the backward pass starting from a scalar tensor (1 element). Computes gradients
        for all tensors in the graph that have <code>requires_grad = true</code>. After this
        call, every parameter tensor's <code>.grad</code> field contains d(loss)/d(param).
      </p>

      <h3>ax_graph_cleanup</h3>
      <pre><code className="language-c">{`void ax_graph_cleanup(ax_tensor_t *root);`}</code></pre>
      <p>
        Frees the computation graph reachable from the root tensor. Destroys all intermediate
        (non-leaf) tensors from the forward pass. Leaf tensors (parameters) are not touched.
        The root tensor itself is not destroyed; the caller owns it.
      </p>
      <p>
        <strong>Important:</strong> always call <code>ax_graph_cleanup(loss)</code> before{' '}
        <code>ax_tensor_destroy(loss)</code>. Otherwise, all intermediate tensors leak.
      </p>

      <h2>Gradient management</h2>

      <h3>ax_zero_grad</h3>
      <pre><code className="language-c">{`void ax_zero_grad(ax_tensor_t *t);`}</code></pre>
      <p>
        Zeros out the gradient tensor of t. Call before each training step to prevent gradient
        accumulation from previous steps. The optimizer's <code>ax_optimizer_zero_grad()</code>
        calls this on all parameters.
      </p>

      <h3>Gradient context</h3>
      <pre><code className="language-c">{`void ax_no_grad(void);
void ax_enable_grad(void);
bool ax_grad_enabled(void);`}</code></pre>
      <p>
        Disable and re-enable gradient tracking. Uses a depth counter, so nested calls are safe.
        When disabled, ops skip the grad_fn creation entirely, saving memory and time.
      </p>
      <pre><code className="language-c">{`ax_no_grad();
ax_tensor_t *pred = ax_layer_forward(net, input);  // no graph recorded
ax_enable_grad();`}</code></pre>
      <p>
        The model's <code>ax_model_predict()</code> wraps the forward pass in no_grad
        automatically.
      </p>

      <h2>Gradient checking</h2>
      <pre><code className="language-c">{`double ax_grad_check(
    ax_tensor_t *(*forward_fn)(ax_tensor_t *input),
    ax_tensor_t *input,
    double eps);`}</code></pre>
      <p>
        Numerical gradient check for testing. Compares the analytical gradient (from autograd)
        against a finite-difference approximation. Returns the maximum absolute difference.
        A difference above ~1e-4 usually indicates a bug in the backward function.
      </p>
      <pre><code className="language-c">{`// check gradient of a custom forward function
double max_diff = ax_grad_check(my_forward_fn, input, 1e-5);
assert(max_diff < 1e-4);`}</code></pre>

      <h2>Typical training loop</h2>
      <pre><code className="language-c">{`for (int epoch = 0; epoch < epochs; epoch++) {
    // zero gradients
    ax_optimizer_zero_grad(opt);

    // forward pass (graph is recorded)
    ax_tensor_t *pred = ax_layer_forward(net, input);
    ax_tensor_t *loss = ax_cross_entropy_loss(pred, target);

    // backward pass (fills .grad on all parameters)
    ax_backward(loss);

    // update weights
    ax_optimizer_step(opt);

    // cleanup: free intermediates, then the loss
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
}`}</code></pre>

      <h2>requires_grad</h2>
      <p>
        Set <code>t&gt;requires_grad = true</code> on any tensor to track its gradient.
        Layer parameters have this set automatically at creation. User tensors (inputs, targets)
        normally don't need gradients.
      </p>
      <p>
        An operation only records a grad_fn if at least one of its inputs has{' '}
        <code>requires_grad = true</code> and gradient tracking is enabled.
      </p>

      <h2>Arenas</h2>
      <pre><code className="language-c">{`ax_arena_t *ax_backward_arena(void);
ax_arena_t *ax_forward_arena(void);`}</code></pre>
      <p>
        Thread-local bump arenas for scratch tensor allocation during forward/backward.
        The backward arena is reset after <code>ax_backward()</code>. The forward arena is
        reset by <code>ax_graph_cleanup()</code>. See the{' '}
        <a href="/docs/architecture/autograd">architecture page</a> for details.
      </p>

      <h2>Inference-only mode</h2>
      <p>
        When built with <code>AX_INFERENCE_ONLY</code>:
      </p>
      <ul>
        <li><code>ax_grad_enabled()</code> returns false (inline, constant-folded)</li>
        <li><code>ax_grad_fn_create()</code> returns NULL</li>
        <li><code>ax_backward()</code> and <code>ax_graph_cleanup()</code> are undefined (link error if called)</li>
        <li>All grad_fn setup in forward ops is dead-code-eliminated by the compiler</li>
      </ul>
    </>
  )
}
