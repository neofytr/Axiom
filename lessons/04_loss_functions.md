# Unit 4: Loss Functions -- Measuring Error

## Why This Matters

A loss function measures how wrong the network's predictions are. It's the quantity
we minimize during training. The choice of loss function defines what "good" means
for your model. Pick the wrong one and the model optimizes for the wrong thing.


## 4.1 The General Idea

A loss function takes two inputs:
- **Predictions** (what the network outputs)
- **Targets** (what we want it to output, the ground truth)

It returns a single scalar: the loss. Lower is better.

    L = loss_fn(predictions, targets)

Training = finding weights that minimize L.


## 4.2 Mean Squared Error (MSE)

The most intuitive loss for regression:

    MSE = (1/n) * sum((pred_i - target_i)^2)

Properties:
- Always non-negative (squared terms)
- Zero only when predictions exactly match targets
- Penalizes large errors quadratically (an error of 10 costs 100x more than an error of 1)
- Gradient: `dL/dpred_i = 2 * (pred_i - target_i) / n`

In Axiom, MSE is composed from basic ops so autograd handles it automatically:

```c
ax_tensor_t *ax_mse_loss(ax_tensor_t *pred, ax_tensor_t *target) {
    ax_tensor_t *diff = ax_sub(pred, target);     // pred - target
    ax_tensor_t *sq = ax_square(diff);             // (pred - target)^2
    ax_tensor_t *loss = ax_mean(sq, -1);           // mean over all elements
    return loss;
}
```

No custom backward needed. The chain rule flows through sub -> square -> mean.


## 4.3 Mean Absolute Error (MAE / L1 Loss)

    MAE = (1/n) * sum(|pred_i - target_i|)

Properties:
- Linear penalty (error of 10 costs 10x more than error of 1)
- More robust to outliers than MSE
- Gradient: `dL/dpred_i = sign(pred_i - target_i) / n`
- Not differentiable at zero (gradient is defined as 0 there)

Axiom uses a custom backward for MAE because `abs()` doesn't have autograd:

```c
float sign = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
pg[i] += g * sign * scale;
```


## 4.4 Cross-Entropy Loss

The standard loss for classification. If the network outputs class probabilities
and the target is a one-hot vector:

    CE = -(1/N) * sum_over_batch(sum_over_classes(target_c * log(softmax(pred_c))))

This is the workhouse of classification. The key insight: cross-entropy with softmax
has an incredibly clean gradient:

    dL/dpred = softmax(pred) - target

This is why cross-entropy + softmax is always used together. The derivative simplifies
beautifully.

Axiom computes this with the log-softmax trick for numerical stability:

```c
// log_softmax(x)_i = x_i - max(x) - log(sum(exp(x_j - max(x))))
float log_sm = (pd[idx] - mx) - log_sum;
total_loss -= td[idx] * log_sm;
```

The backward stores the softmax output and uses the simple gradient formula:

```c
// gradient of cross-entropy with softmax: softmax(pred) - target
pg[idx] += (sd[idx] - td[idx]) * scale;
```


## 4.5 Binary Cross-Entropy with Logits

For binary classification (two classes):

    BCE = -(1/n) * sum(t * log(sigma(x)) + (1-t) * log(1 - sigma(x)))

Where sigma is the sigmoid function. Numerically stable form:

    BCE = (1/n) * sum(max(x, 0) - x*t + log(1 + exp(-|x|)))

This avoids computing log(sigmoid(x)) directly, which underflows for large negative x.

Gradient:

    dL/dpred = (sigmoid(pred) - target) / n

Axiom implements the stable version in `losses.c`:

```c
float mx = x > 0.0f ? x : 0.0f;
float ax = x > 0.0f ? x : -x;
total += mx - x * t + logf(1.0f + expf(-ax));
```


## 4.6 Huber Loss

A hybrid of MSE and MAE that transitions at a threshold delta:

    L = 0.5 * d^2           if |d| <= delta
    L = delta * (|d| - 0.5 * delta)  otherwise

Where d = pred - target.

Properties:
- Quadratic for small errors (like MSE — smooth, fast convergence near optimum)
- Linear for large errors (like MAE — robust to outliers)
- The delta parameter controls the transition point

Gradient:

    dL/dpred = d                        if |d| <= delta
    dL/dpred = delta * sign(d)          if |d| > delta

This is sometimes called "smooth L1 loss" and is popular in object detection.


## 4.7 Which Loss to Use?

| Task                    | Loss Function          | Output Activation |
|------------------------|----------------------|------------------|
| Regression             | MSE                  | None (linear)    |
| Regression (outliers)  | Huber or MAE         | None (linear)    |
| Binary classification  | BCE with logits      | Sigmoid          |
| Multi-class (exclusive)| Cross-entropy        | Softmax          |


## 4.8 Composing Losses from Basic Ops

Axiom takes two approaches to implementing losses:

1. **Composed from differentiable ops** (MSE): build the loss from existing
   autograd-aware operations. No custom backward needed. Simple code, but
   intermediate tensors stay alive until cleanup.

2. **Custom backward** (MAE, cross-entropy, BCE, Huber): implement the forward
   manually and attach a custom `ax_grad_fn_t`. More code, but often more
   numerically stable and memory-efficient.

The custom backward approach is used when:
- The composed version would be numerically unstable (cross-entropy)
- Some operation in the chain doesn't have autograd (abs in MAE)
- The fused gradient is much simpler than the composed gradient


## Key Takeaways

1. MSE for regression, cross-entropy for classification. Start there.
2. Cross-entropy + softmax has the clean gradient `softmax(pred) - target`.
3. Numerical stability matters: log-softmax trick, stable BCE formula.
4. Huber loss combines the best of MSE (smooth) and MAE (robust).
5. Losses can be composed from basic ops (autograd handles the rest) or
   implemented with custom backward functions for stability.
