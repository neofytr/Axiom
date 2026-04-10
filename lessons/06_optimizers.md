# Unit 6: Optimizers

## Why This Matters

Vanilla gradient descent is slow and fragile. Modern optimizers make training
faster, more stable, and less sensitive to hyperparameters. Understanding them
means understanding why Adam usually "just works" and when to use something else.


## 6.1 The Update Rule

Every optimizer follows the same pattern:

    w_new = w_old - lr * update(gradient, state)

They differ in how they compute `update` from the gradient and accumulated state.


## 6.2 SGD (Stochastic Gradient Descent)

The simplest optimizer. "Stochastic" because we use mini-batches, not the full dataset.

**Without momentum:**

    w = w - lr * grad

**With momentum** (velocity term `v`):

    v = momentum * v + grad
    w = w - lr * v

Momentum accumulates past gradients like a rolling ball. It smooths out noisy
gradients and accelerates through flat regions.

**With Nesterov momentum** (look-ahead):

    v = momentum * v + grad
    w = w - lr * (grad + momentum * v)

Nesterov computes the gradient at the "look-ahead" position, giving better
convergence in practice.

Axiom implementation (`optim.c`):

```c
if (opt->momentum > 0.0f) {
    for (int64_t j = 0; j < n; j++) {
        vd[j] = opt->momentum * vd[j] + gd[j];
        if (opt->nesterov)
            wd[j] -= opt->lr * (gd[j] + opt->momentum * vd[j]);
        else
            wd[j] -= opt->lr * vd[j];
    }
}
```


## 6.3 Adam (Adaptive Moment Estimation)

The default choice for most problems. Maintains two running averages per parameter:

**First moment** (mean of gradients, like momentum):

    m = beta1 * m + (1 - beta1) * grad

**Second moment** (mean of squared gradients, per-parameter learning rate):

    v = beta2 * v + (1 - beta2) * grad^2

**Bias correction** (early steps are biased toward zero):

    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)

**Update:**

    w = w - lr * m_hat / (sqrt(v_hat) + eps)

Default hyperparameters: beta1 = 0.9, beta2 = 0.999, eps = 1e-8.

Why it works:
- First moment: momentum-like smoothing of gradient direction.
- Second moment: normalizes the step size per-parameter. Parameters with large
  gradients get smaller steps; sparse parameters get larger steps.
- Bias correction: accounts for the fact that m and v are initialized to zero.

Axiom:
```c
md[j] = opt->beta1 * md[j] + (1.0f - opt->beta1) * g;
vd[j] = opt->beta2 * vd[j] + (1.0f - opt->beta2) * g * g;
float m_hat = md[j] / bc1;
float v_hat = vd[j] / bc2;
wd[j] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
```


## 6.4 AdamW (Adam with Decoupled Weight Decay)

Standard Adam applies weight decay through the gradient:

    grad += weight_decay * w    // L2 regularization
    // then do the Adam update with this modified gradient

AdamW applies it directly to the weights:

    w *= (1 - lr * weight_decay)    // decoupled decay
    // then do the normal Adam update with unmodified gradient

Why this matters: in standard Adam, the adaptive learning rate also scales the
weight decay term, weakening regularization for parameters with large second
moments. Decoupling fixes this. AdamW is now the standard for training
transformers.

Axiom uses the same function with a boolean flag:

```c
static void adam_step(ax_optimizer_t *opt, bool decoupled_decay) {
    if (decoupled_decay && opt->weight_decay > 0.0f) {
        wd[j] *= (1.0f - opt->lr * opt->weight_decay);  // AdamW
    }
    if (!decoupled_decay && opt->weight_decay > 0.0f) {
        gd[j] += opt->weight_decay * wd[j];              // Adam L2
    }
    // ... rest of Adam update
}
```


## 6.5 RMSProp

A precursor to Adam. Maintains only the second moment (no first moment):

    v = rho * v + (1 - rho) * grad^2
    w = w - lr * grad / (sqrt(v) + eps)

Default: rho = 0.99, eps = 1e-8.

RMSProp adapts the learning rate per-parameter but lacks momentum. Think of it
as "Adam without the first moment." It's simpler but generally outperformed by
Adam. Still useful for recurrent networks.


## 6.6 Adagrad

Accumulates all past squared gradients:

    v = v + grad^2
    w = w - lr * grad / (sqrt(v) + eps)

Key property: the effective learning rate monotonically decreases. Parameters
that receive frequent large gradients quickly get smaller learning rates. Good
for sparse data (NLP, recommendations), bad for deep networks (learning rate
goes to zero too fast).


## 6.7 Weight Decay (L2 Regularization)

Weight decay penalizes large weights by adding a term to the loss:

    L_total = L_data + (weight_decay / 2) * ||w||^2

This adds `weight_decay * w` to the gradient for each parameter. It prevents
the network from relying too heavily on any single weight, improving
generalization.

All Axiom optimizers support weight decay:

```c
if (opt->weight_decay > 0.0f) {
    for (int64_t j = 0; j < n; j++)
        gd[j] += opt->weight_decay * wd[j];
}
```


## 6.8 Per-Parameter State

Each optimizer maintains per-parameter state (momentum buffer, second moment, etc.)
in `ax_param_state_t`:

```c
typedef struct {
    ax_tensor_t *m;         // first moment (Adam, SGD momentum)
    ax_tensor_t *v;         // second moment (Adam, RMSProp, Adagrad)
    int64_t step_count;     // number of updates applied
} ax_param_state_t;
```

State tensors are lazily allocated on first use:

```c
static void ensure_state_m(ax_param_state_t *s, ax_tensor_t *p) {
    if (!s->m)
        s->m = ax_tensor_zeros(p->shape, p->ndim, p->dtype);
}
```


## 6.9 The Common Pattern

All optimizers in Axiom follow the same loop:

```c
for (int i = 0; i < opt->n_params; i++) {
    ax_tensor_t *p = opt->params[i];
    if (!p->grad) continue;           // 1. skip if no gradient

    // 2. apply weight decay
    // 3. update momentum/moment state
    // 4. compute step and apply to weights

    opt->state[i].step_count++;       // 5. increment step count
}
```


## 6.10 Which Optimizer to Use?

| Situation                    | Optimizer       | Why                                    |
|-----------------------------|-----------------|-----------------------------------------|
| Default / first try          | Adam            | Works well out of the box               |
| Training transformers        | AdamW           | Decoupled weight decay is important     |
| When Adam overfits           | SGD + momentum  | Better generalization on some problems  |
| Sparse gradients             | Adagrad         | Larger steps for rare features          |
| RNNs                         | RMSProp         | Historically the go-to for RNNs        |

In practice, Adam with lr=0.001 is a safe starting point for almost anything.


## Key Takeaways

1. Momentum smooths gradient noise and accelerates convergence.
2. Adam adapts the learning rate per-parameter using first and second moments.
3. AdamW decouples weight decay from the adaptive learning rate.
4. Weight decay (L2 regularization) prevents overfitting by penalizing large weights.
5. Start with Adam at lr=0.001. Switch to SGD+momentum for fine-tuning if needed.
