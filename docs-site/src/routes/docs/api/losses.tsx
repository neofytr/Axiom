import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/losses')({
  component: LossesApiPage,
})

function LossesApiPage() {
  return (
    <>
      <h1>Losses API</h1>
      <p>
        Loss functions for training. Each returns a <strong>scalar tensor</strong> (1 element)
        suitable for passing to <code>ax_backward()</code>.
      </p>
      <p>Header: <code>axiom/losses.h</code></p>

      <h2>MSE Loss</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_mse_loss(ax_tensor_t *pred, ax_tensor_t *target);`}</code></pre>
      <p>
        Mean squared error: <code>(1/n) * sum((pred - target)^2)</code>. The classic regression
        loss. Use when your output is a continuous value.
      </p>
      <pre><code className="language-c">{`// regression example
ax_tensor_t *loss = ax_mse_loss(predictions, targets);
ax_backward(loss);`}</code></pre>

      <h2>MAE Loss</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_mae_loss(ax_tensor_t *pred, ax_tensor_t *target);`}</code></pre>
      <p>
        Mean absolute error: <code>(1/n) * sum(|pred - target|)</code>. More robust to outliers
        than MSE because it doesn't square large errors.
      </p>

      <h2>Cross-Entropy Loss</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_cross_entropy_loss(ax_tensor_t *pred, ax_tensor_t *target);`}</code></pre>
      <p>
        Cross-entropy for multi-class classification. Takes raw logits (not softmax'd). Applies
        log-softmax internally for numerical stability.
      </p>
      <ul>
        <li><code>pred</code>: [batch, num_classes] — raw logits</li>
        <li><code>target</code>: [batch, num_classes] — one-hot encoded</li>
      </ul>
      <p>
        The loss is: <code>-sum(target * log_softmax(pred)) / batch_size</code>
      </p>
      <p>
        The backward pass computes <code>softmax(pred) - target</code> in a single fused SIMD
        pass, which is much more efficient than computing the full softmax Jacobian.
      </p>
      <pre><code className="language-c">{`// classification example
ax_model_compile(model, opt, ax_cross_entropy_loss);
float loss = ax_model_train_step(model, batch_x, batch_y_onehot);`}</code></pre>

      <h2>BCE with Logits</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_bce_with_logits_loss(ax_tensor_t *pred, ax_tensor_t *target);`}</code></pre>
      <p>
        Binary cross-entropy with logits. Takes raw logits (before sigmoid). Target is 0 or 1.
        Uses the numerically stable formulation:
      </p>
      <pre><code className="language-text">{`loss = mean(max(pred, 0) - pred*target + log(1 + exp(-|pred|)))`}</code></pre>
      <p>Use for binary classification or multi-label classification.</p>

      <h2>Huber Loss</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_huber_loss(ax_tensor_t *pred, ax_tensor_t *target, float delta);`}</code></pre>
      <p>
        Smooth L1 loss. Quadratic for small errors, linear for large. The delta parameter
        controls the transition point:
      </p>
      <pre><code className="language-text">{`loss = 0.5 * x^2                     if |x| <= delta
       delta * (|x| - 0.5 * delta)   otherwise`}</code></pre>
      <p>
        Combines the best of MSE (stable gradients near zero) and MAE (robustness to outliers).
        Common in reinforcement learning and regression with noisy data.
      </p>

      <h2>Using losses with the model API</h2>
      <p>
        Loss functions match the <code>ax_loss_fn_t</code> type signature, so you can pass them
        directly to <code>ax_model_compile</code>:
      </p>
      <pre><code className="language-c">{`// function pointer type
typedef ax_tensor_t *(*ax_loss_fn_t)(ax_tensor_t *pred, ax_tensor_t *target);

// MSE for regression
ax_model_compile(model, opt, ax_mse_loss);

// cross-entropy for classification
ax_model_compile(model, opt, ax_cross_entropy_loss);`}</code></pre>
      <p>
        Note: <code>ax_huber_loss</code> takes an extra parameter (delta), so it doesn't match
        the two-argument <code>ax_loss_fn_t</code> signature directly. Use it in a manual training
        loop instead.
      </p>
    </>
  )
}
