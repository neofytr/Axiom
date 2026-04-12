import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/optimizers')({
  component: OptimizersApiPage,
})

function OptimizersApiPage() {
  return (
    <>
      <h1>Optimizers API</h1>
      <p>
        Optimizers update model parameters using their gradients. Each optimizer holds per-parameter
        state (momentum buffers, Adam moments, etc.) and implements the <code>step()</code> function.
      </p>
      <p>Header: <code>axiom/optim.h</code></p>

      <h2>SGD</h2>
      <pre><code className="language-c">{`ax_optimizer_t *ax_sgd_create(ax_tensor_t **params, int n_params,
                              float lr, float momentum, float weight_decay,
                              bool nesterov);`}</code></pre>
      <p>
        Stochastic gradient descent with optional momentum and weight decay. Set{' '}
        <code>momentum = 0</code> for vanilla SGD. Nesterov momentum computes the gradient at
        the "lookahead" position, which often converges faster.
      </p>
      <pre><code className="language-c">{`// vanilla SGD
ax_optimizer_t *opt = ax_sgd_create(params, n, 0.01f, 0, 0, false);

// SGD with momentum
ax_optimizer_t *opt = ax_sgd_create(params, n, 0.01f, 0.9f, 0, false);

// SGD with nesterov momentum and L2 regularization
ax_optimizer_t *opt = ax_sgd_create(params, n, 0.01f, 0.9f, 1e-4f, true);`}</code></pre>

      <h2>Adam</h2>
      <pre><code className="language-c">{`ax_optimizer_t *ax_adam_create(ax_tensor_t **params, int n_params,
                               float lr, float beta1, float beta2,
                               float eps, float weight_decay);`}</code></pre>
      <p>
        Adaptive moment estimation. Maintains per-parameter running estimates of the first moment
        (mean) and second moment (uncentered variance) of the gradient. Bias-corrected for the
        initial steps. This is the default choice for most training tasks.
      </p>
      <p>Typical hyperparameters: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8.</p>
      <pre><code className="language-c">{`ax_optimizer_t *opt = ax_adam_create(
    m&gt;params, m&gt;n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0);`}</code></pre>

      <h2>AdamW</h2>
      <pre><code className="language-c">{`ax_optimizer_t *ax_adamw_create(ax_tensor_t **params, int n_params,
                                float lr, float beta1, float beta2,
                                float eps, float weight_decay);`}</code></pre>
      <p>
        Adam with decoupled weight decay. Unlike Adam with L2 regularization (which adds the
        penalty to the gradient before the adaptive scaling), AdamW applies weight decay
        directly to the weights after the Adam step. This is the correct formulation and
        works better in practice, especially with learning rate scheduling.
      </p>
      <pre><code className="language-c">{`ax_optimizer_t *opt = ax_adamw_create(
    m&gt;params, m&gt;n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);`}</code></pre>

      <h2>RMSprop</h2>
      <pre><code className="language-c">{`ax_optimizer_t *ax_rmsprop_create(ax_tensor_t **params, int n_params,
                                  float lr, float rho, float eps,
                                  float weight_decay);`}</code></pre>
      <p>
        Maintains a moving average of squared gradients and divides the gradient by its root.
        Adapts the learning rate per-parameter. Typical rho=0.99.
      </p>

      <h2>Adagrad</h2>
      <pre><code className="language-c">{`ax_optimizer_t *ax_adagrad_create(ax_tensor_t **params, int n_params,
                                  float lr, float eps, float weight_decay);`}</code></pre>
      <p>
        Accumulates the sum of squared gradients and scales the learning rate inversely.
        Good for sparse gradients. The accumulated sum grows monotonically, so the effective
        learning rate decays over time.
      </p>

      <h2>Common operations</h2>
      <pre><code className="language-c">{`// update all parameters using their gradients
void ax_optimizer_step(ax_optimizer_t *opt);

// zero out gradients (call before each training step)
void ax_optimizer_zero_grad(ax_optimizer_t *opt);

// adjust learning rate
void ax_optimizer_set_lr(ax_optimizer_t *opt, float lr);
float ax_optimizer_get_lr(ax_optimizer_t *opt);

// free optimizer state (does NOT free the parameter tensors)
void ax_optimizer_destroy(ax_optimizer_t *opt);`}</code></pre>

      <h2>Per-parameter state</h2>
      <p>
        Optimizer state is allocated <strong>lazily</strong> on the first call to{' '}
        <code>ax_optimizer_step()</code>. The state struct per parameter:
      </p>
      <pre><code className="language-c">{`typedef struct {
    ax_tensor_t *m;     // first moment (momentum / adam m)
    ax_tensor_t *v;     // second moment (adam v / rmsprop cache)
    int64_t step_count; // for adam bias correction
} ax_param_state_t;`}</code></pre>
      <p>
        Adam/AdamW use both m and v. SGD with momentum uses only m. Adagrad uses only v.
        The global step counter is shared across all parameters for bias correction.
      </p>

      <h2>Learning rate scheduling</h2>
      <p>Header: <code>axiom/lr_scheduler.h</code></p>
      <pre><code className="language-c">{`// step decay: lr *= gamma every step_size epochs
ax_lr_scheduler_t *ax_sched_step_decay(ax_optimizer_t *opt,
                                        int step_size, float gamma);

// exponential: lr *= gamma every epoch
ax_lr_scheduler_t *ax_sched_exponential(ax_optimizer_t *opt, float gamma);

// cosine annealing between initial_lr and min_lr
ax_lr_scheduler_t *ax_sched_cosine(ax_optimizer_t *opt,
                                    int total_steps, float min_lr);

// warmup + cosine (standard transformer schedule)
ax_lr_scheduler_t *ax_sched_warmup_cosine(ax_optimizer_t *opt,
                                           int warmup_steps, int total_steps,
                                           float min_lr);

// call after each epoch/step
void ax_sched_step(ax_lr_scheduler_t *sched);

// query and cleanup
float ax_sched_get_lr(ax_lr_scheduler_t *sched);
void ax_sched_destroy(ax_lr_scheduler_t *sched);`}</code></pre>

      <h2>Fused GPU updates</h2>
      <p>
        When running on the CUDA backend, the optimizer dispatches to fused GPU kernels
        (<code>adam_update</code>, <code>sgd_update</code>) that read weight, gradient, and moment
        tensors and write back the updated values in a single kernel launch. This avoids multiple
        kernel launches and multiple memory passes.
      </p>

      <h2>Parallelism</h2>
      <p>
        On CPU, optimizer updates are parallelized with OpenMP across parameters using dynamic
        scheduling. A serial pre-pass allocates any needed state tensors (first step only),
        then the parallel loop handles the actual updates.
      </p>
    </>
  )
}
