import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/activations')({
  component: ActivationsApiPage,
})

function ActivationsApiPage() {
  return (
    <>
      <h1>Activations API</h1>
      <p>
        Activation functions as tensor operations. These all support autograd (gradients flow
        through them during backward). For use in sequential models, see the layer wrappers
        in the <a href="/docs/api/layers">Layers API</a>.
      </p>
      <p>
        Headers: <code>axiom/ops.h</code>, <code>axiom/activations.h</code>
      </p>

      <h2>ReLU</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_relu(ax_tensor_t *a);
ax_tensor_t *ax_relu_inplace(ax_tensor_t *a);`}</code></pre>
      <p>
        <code>max(0, x)</code>. The in-place variant mutates the input buffer directly, avoiding
        one allocation and one full memory pass. The in-place version falls back to the
        out-of-place path if the storage refcount is greater than 1 (shared views).
        Autograd-compatible: ReLU backward only needs the post-activation output, which
        in-place preserves.
      </p>

      <h2>Sigmoid</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_sigmoid(ax_tensor_t *a);
ax_tensor_t *ax_sigmoid_inplace(ax_tensor_t *a);`}</code></pre>
      <p>
        <code>1 / (1 + exp(-x))</code>. Maps inputs to (0, 1). The backward pass needs the
        sigmoid output: <code>grad = output * (1 - output) * grad_output</code>.
      </p>

      <h2>Tanh</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_tanh_op(ax_tensor_t *a);
ax_tensor_t *ax_tanh_inplace(ax_tensor_t *a);`}</code></pre>
      <p>
        Hyperbolic tangent. Maps inputs to (-1, 1). Named <code>ax_tanh_op</code> to avoid
        conflicting with the C standard library's <code>tanh</code>.
      </p>

      <h2>LeakyReLU</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_leaky_relu(ax_tensor_t *a, float alpha);`}</code></pre>
      <p>
        <code>max(alpha * x, x)</code> where alpha is typically 0.01. Allows a small gradient
        for negative inputs, which can help with dying ReLU problems.
      </p>

      <h2>ELU</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_elu(ax_tensor_t *a, float alpha);`}</code></pre>
      <p>
        <code>x</code> if x &gt; 0, <code>alpha * (exp(x) - 1)</code> otherwise.
        Smooth for negative inputs, which can speed up convergence.
      </p>

      <h2>SELU</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_selu(ax_tensor_t *a);`}</code></pre>
      <p>
        Scaled ELU with specific constants (lambda ~1.0507, alpha ~1.6733) that make the
        activation self-normalizing: the output mean and variance tend toward 0 and 1 through
        the network, without needing batch normalization.
      </p>

      <h2>GELU</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_gelu(ax_tensor_t *a);`}</code></pre>
      <p>
        Gaussian Error Linear Unit. Uses the tanh approximation:{' '}
        <code>x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))</code>.
        This is the standard activation in transformers and modern architectures.
      </p>

      <h2>Swish (SiLU)</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_swish(ax_tensor_t *a);`}</code></pre>
      <p>
        <code>x * sigmoid(x)</code>. Smooth, non-monotonic activation used in EfficientNet
        and other modern architectures.
      </p>

      <h2>Softplus</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_softplus(ax_tensor_t *a);`}</code></pre>
      <p>
        <code>log(1 + exp(x))</code>. A smooth approximation of ReLU. Always positive.
      </p>

      <h2>Mish</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_mish(ax_tensor_t *a);`}</code></pre>
      <p>
        <code>x * tanh(softplus(x))</code>. Another smooth, self-regularizing activation.
      </p>

      <h2>Softmax</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_softmax(ax_tensor_t *a, int axis);`}</code></pre>
      <p>
        Softmax along the given axis. Numerically stable (subtracts the max first to prevent
        overflow). <code>axis = -1</code> means the last dimension. For classification, apply
        to the logits along the class dimension.
      </p>
      <p>
        Note: <code>ax_cross_entropy_loss</code> applies log-softmax internally, so don't
        apply softmax to your logits before passing them to the loss function.
      </p>

      <h2>SIMD and fusion</h2>
      <p>
        All activations have SIMD-optimized implementations in the cpu_opt backend. The
        backend also provides fused ops:
      </p>
      <ul>
        <li>
          <code>gemm_relu</code> — applies ReLU during the GEMM writeback while the output
          tile is still in registers. Used automatically when Dense is followed by ReLU in
          a sequential model.
        </li>
        <li>
          <code>add_relu</code> — fused <code>relu(a + b)</code> in a single pass.
        </li>
        <li>
          <code>softmax_rowwise</code> — row-wise softmax on 2D tensors with the
          max-subtract trick, all in one kernel.
        </li>
      </ul>
    </>
  )
}
