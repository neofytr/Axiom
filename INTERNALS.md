# Axiom Internals: Deep Learning from Scratch

This document teaches you deep learning — the math, the intuition, and the
implementation — from zero. It's structured as a course. Each unit builds
on the last. By the end, you'll understand every line of code in this
project and the theory behind it.

This is a living document. New units are added as the project grows.


# Part 1: Foundations


## Unit 1: Vectors, Matrices, and Tensors

Everything in deep learning is a tensor operation. If you understand
tensors, you understand the data structures. If you understand the
operations on tensors, you understand the algorithms.

### Scalars, vectors, matrices

A **scalar** is a single number. Temperature: 23.5. Price: 49.99.

A **vector** is an ordered list of numbers. It represents a point in space,
or a direction, or a collection of features.

```
v = [height, weight, age] = [170, 65, 25]
```

This vector lives in 3-dimensional space. Each element is a **component**.
The number of components is the **dimension** of the vector.

A **matrix** is a 2D grid of numbers. Rows and columns.

```
M = [[1, 2, 3],
     [4, 5, 6]]
```

This is a 2x3 matrix (2 rows, 3 columns). We write its **shape** as [2, 3].

A **tensor** generalizes this to any number of dimensions:
- 0D tensor = scalar: `42`
- 1D tensor = vector: `[1, 2, 3]`
- 2D tensor = matrix: `[[1, 2], [3, 4]]`
- 3D tensor = a "cube" of numbers. An RGB image is 3D: [height, width, 3]
- 4D tensor: a batch of images is [batch_size, height, width, channels]

In Axiom, `ax_tensor_t` handles all of these. The `ndim` field tells you
how many dimensions, and `shape[i]` tells you the size along dimension `i`.

### Why matrices matter: the dot product

The **dot product** of two vectors is the sum of element-wise products:

```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

Geometrically, the dot product measures how much two vectors point in the
same direction. If they're perpendicular, the dot product is zero. If they
point the same way, it's large and positive. Opposite: large and negative.

This matters because a **neuron** computes a dot product. The input is a
vector, the weights are a vector, and the neuron measures "how much does
this input match these weights?"

### Matrix multiplication

**Matrix multiplication** is many dot products at once.

If A is [m, k] and B is [k, n], then C = A @ B is [m, n], where:

```
C[i, j] = sum over p from 0 to k-1 of: A[i, p] * B[p, j]
```

Each element of C is a dot product of a row of A with a column of B.

Worked example:

```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]

C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
C[1,1] = 3*6 + 4*8 = 18 + 32 = 50

C = [[19, 22],
     [43, 50]]
```

In our code, this is `ax_compute_gemm()` in `src/compute/backends/cpu_naive.c`.
The naive implementation is three nested loops, exactly matching the formula.

### Why matrix multiply is the core operation

A neural network layer computes:

```
output = input @ weight + bias
```

If input is [batch_size, input_features] and weight is [input_features, output_features],
then output is [batch_size, output_features].

Each sample in the batch gets transformed. Each row of the weight matrix defines
one output feature — it's a set of "what to look for" coefficients. The matrix
multiply applies all these detectors to all samples simultaneously.

This is why GPUs are fast for deep learning: they're designed for massively
parallel matrix multiplication.

### Strides: how tensors live in memory

A 2D tensor is stored as a flat 1D array in memory. To find element [i, j],
you need to know how many elements to skip per row. This is the **stride**.

For a [2, 3] tensor stored in **row-major** order (C convention):

```
Memory: [a, b, c, d, e, f]

Element [0, 0] is at position 0*3 + 0 = 0  -> a
Element [0, 2] is at position 0*3 + 2 = 2  -> c
Element [1, 1] is at position 1*3 + 1 = 4  -> e

strides = [3, 1]   (skip 3 elements to advance one row, 1 to advance one column)
```

The formula: `element[i, j] = data[i * stride[0] + j * stride[1]]`

For N dimensions: `element[i0, i1, ..., in] = data[i0*s0 + i1*s1 + ... + in*sn]`

Why do we care? Because strides enable **zero-copy operations**:

**Transpose**: instead of physically rearranging the data, just swap the strides.
A [2, 3] matrix with strides [3, 1] becomes a [3, 2] matrix with strides [1, 3].
Same data in memory, read differently.

**Reshape**: if the data is contiguous, just change the shape and recompute strides.
A [2, 3] tensor becomes [6] or [3, 2] without moving any data.

**Slicing**: take a subset by adjusting the offset and shape. Still no copy.

In Axiom, see `ax_tensor_transpose()` in `src/core/tensor.c` — it literally just
swaps two shape and stride values and returns.

### Broadcasting: making shapes compatible

What happens when you add a [2, 3] tensor and a [3] tensor? The shapes don't match,
but the operation should still make sense — add the [3] vector to each row.

Broadcasting rules (same as NumPy):

1. Right-align the shapes:
   ```
   [2, 3]
      [3]
   ```

2. Pad the shorter one with 1s on the left:
   ```
   [2, 3]
   [1, 3]
   ```

3. For each dimension, they must be equal or one of them must be 1:
   ```
   2 vs 1 -> ok, the 1 stretches to 2
   3 vs 3 -> ok, equal
   ```

4. Result shape: [2, 3]

The "stretching" doesn't copy data. When accessing element [i, j] of the
broadcast tensor, if a dimension has size 1, the index for that dimension
is always 0. See `bcast_get_f32()` in `cpu_naive.c`.

This is critical for the bias add in a dense layer: bias is [output_features]
and gets broadcast across the [batch_size, output_features] output.

### Storage and reference counting

In Axiom, the actual data lives in an `ax_storage_t`:

```c
typedef struct {
    void *data;
    size_t size_bytes;
    int refcount;
    ax_device_t device;
} ax_storage_t;
```

Multiple tensors can share the same storage. When you transpose a tensor,
the new tensor points to the same storage with different strides. The refcount
tracks how many tensors share it. When the last one is destroyed, the data
is freed.

This is the same pattern used by PyTorch (`torch.Storage`) and NumPy
(`ndarray.base`).


## Unit 2: Calculus for Deep Learning

You need exactly three things from calculus: derivatives, partial derivatives,
and the chain rule. That's it. Everything else in deep learning's math follows
from these three.

### Derivatives: the rate of change

The derivative of f(x) at a point tells you how fast f is changing there.

```
f(x) = x^2

f'(x) = 2x

At x=3: f'(3) = 6. If x increases by a tiny amount ε, f increases by about 6ε.
```

Key derivatives you need to know:

```
f(x) = x^n     ->  f'(x) = n * x^(n-1)
f(x) = e^x     ->  f'(x) = e^x           (exponential is its own derivative!)
f(x) = ln(x)   ->  f'(x) = 1/x
f(x) = 1/x     ->  f'(x) = -1/x^2
```

The derivative tells us the **direction of increase**. If f'(x) > 0, f is
increasing at x. If f'(x) < 0, f is decreasing. This is how we'll minimize
the loss function: follow the negative derivative.

### Partial derivatives: multiple inputs

When f depends on multiple variables, the partial derivative with respect to
one variable treats all others as constants.

```
f(x, y) = x^2 + 3xy + y^2

∂f/∂x = 2x + 3y      (differentiate w.r.t. x, treat y as constant)
∂f/∂y = 3x + 2y      (differentiate w.r.t. y, treat x as constant)
```

The **gradient** is the vector of all partial derivatives:

```
∇f = [∂f/∂x, ∂f/∂y] = [2x + 3y, 3x + 2y]
```

The gradient points in the direction of steepest increase. To minimize f,
go in the opposite direction: -∇f.

### The chain rule: derivatives of compositions

If y = f(g(x)), then:

```
dy/dx = (dy/dg) * (dg/dx)
```

"The derivative of the outside times the derivative of the inside."

Example:

```
y = (3x + 2)^2

Let g = 3x + 2, so y = g^2.

dy/dg = 2g
dg/dx = 3

dy/dx = 2g * 3 = 2(3x + 2) * 3 = 6(3x + 2)
```

With more steps, you just keep multiplying:

```
y = f(g(h(x)))

dy/dx = (dy/df) * (df/dg) * (dg/dh) * (dh/dx)
```

**This is the entire basis of backpropagation.** A neural network is a
chain of functions. The chain rule lets us compute how the loss changes
with respect to any weight, no matter how deep the network.


## Unit 3: The Neuron

### What a single neuron computes

A neuron takes a vector of inputs, computes a weighted sum, adds a bias,
and passes the result through an activation function:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b    (weighted sum + bias)
a = σ(z)                                (activation function)
```

In vector notation: `z = w · x + b`, then `a = σ(z)`.

The **weights** (w) determine how much each input matters.
The **bias** (b) shifts the decision boundary.
The **activation** (σ) introduces nonlinearity.

### Why activation functions?

Without activation functions, a neural network is just a sequence of linear
transformations. And the composition of linear functions is still linear:

```
f(x) = W₂(W₁x + b₁) + b₂ = W₂W₁x + W₂b₁ + b₂ = W'x + b'
```

No matter how many layers, it collapses to a single linear transformation.
You can't learn XOR, you can't learn curves, you can't learn anything
that a straight line can't represent.

Activation functions break this linearity. They let the network learn
arbitrary nonlinear functions.

### Sigmoid

```
σ(z) = 1 / (1 + e^(-z))
```

Squashes any real number into (0, 1). Historically popular because:
- Output looks like a probability
- Smooth and differentiable everywhere
- Derivative has a nice form: σ'(z) = σ(z) * (1 - σ(z))

Problems: for very large or very small z, the derivative is nearly zero.
This causes the **vanishing gradient problem** in deep networks.

In Axiom: `cpu_sigmoid()` in `cpu_naive.c`, `sigmoid_backward()` in `autograd_ops.c`.

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

Dead simple. If positive, pass through. If negative, output zero.

Derivative: 1 if z > 0, 0 if z < 0, undefined at z = 0 (we use 0).

Why it works so well:
- No vanishing gradient for positive values (derivative is always 1)
- Computationally cheap (just a comparison)
- Induces sparsity (many neurons output 0, which is efficient)

Problem: "dying ReLU" — if a neuron's output is always negative (due to a
large negative bias), its gradient is always 0 and it never updates.

Variants that fix this:
- **Leaky ReLU**: `max(αx, x)` where α is small (0.01). Negative side has
  a small slope instead of zero.
- **ELU**: `x if x > 0, α(e^x - 1) otherwise`. Smooth near zero.
- **GELU**: `x * Φ(x)` where Φ is the standard normal CDF. The activation
  used in GPT and BERT. Smooth approximation of ReLU that allows small
  negative values.

In Axiom: `ax_relu()` in `ops.c`, `ax_leaky_relu()`, `ax_gelu()` etc. in `activations.c`.


## Unit 4: Loss Functions — Measuring Error

The loss function quantifies how wrong the model is. Training minimizes it.
The choice of loss function shapes how the model learns.

### MSE (Mean Squared Error) for Regression

```
L = (1/n) Σ (pred_i - target_i)²
```

You square the errors and average them. Properties:
- Always non-negative (squared values are positive)
- Zero only when predictions are perfect
- Large errors are penalized much more than small errors (quadratic growth)
- Differentiable everywhere (smooth)

Gradient: `∂L/∂pred_i = 2(pred_i - target_i) / n`

This is proportional to the error itself. Large errors produce large gradients,
which produce large updates. The model fixes its biggest mistakes first.

In Axiom: `ax_mse_loss()` in `losses.c`. It's composed from `ax_sub → ax_square → ax_mean`,
so autograd handles the gradient automatically through the chain rule.

### Cross-Entropy for Classification

Classification is fundamentally different from regression. The model outputs
a probability distribution over classes. We need a loss that measures how
different two probability distributions are.

**Softmax** first converts raw logits (unnormalized scores) into probabilities:

```
softmax(x_i) = exp(x_i) / Σⱼ exp(x_j)
```

All outputs are positive and sum to 1. The largest logit gets the highest
probability, but all classes get some probability (never exactly 0 or 1).

Numerical stability trick: subtract max(x) from all logits before exp.
This doesn't change the result (the constant cancels in the fraction)
but prevents exp from overflowing.

```
softmax(x_i) = exp(x_i - max(x)) / Σⱼ exp(x_j - max(x))
```

**Cross-entropy** then measures the difference:

```
L = -(1/batch) Σ_samples Σ_classes target_c * log(softmax(pred_c))
```

If target is one-hot (e.g., [0, 0, 1, 0] for class 2), only the true
class contributes. The loss is `-log(predicted probability of true class)`.

Think about what this means:
- If model predicts 0.9 for the correct class: loss = -log(0.9) = 0.105 (small)
- If model predicts 0.5: loss = -log(0.5) = 0.693 (medium)
- If model predicts 0.01: loss = -log(0.01) = 4.605 (huge!)

Cross-entropy punishes confident wrong answers very harshly. This is exactly
the behavior we want.

The gradient has one of the most elegant results in all of deep learning:

```
∂L/∂pred = (softmax(pred) - target) / batch_size
```

"What you predicted minus what you should have predicted." Simple, clean,
numerically stable. This is why cross-entropy + softmax is the standard
for classification — the gradient is trivial.

In Axiom: `ax_cross_entropy_loss()` in `losses.c`. We compute softmax and
cross-entropy in a single fused pass for numerical stability, and store the
softmax output for the backward pass.

### Why MSE is bad for classification

With MSE on classification, gradients can be tiny even when predictions are
very wrong. Consider sigmoid output + MSE:

```
pred = sigmoid(z) = 0.99    (very confident, but target is 0!)
MSE gradient ∝ (0.99 - 0) * sigmoid'(z)

sigmoid'(z) at z≈4.6 is about 0.01

So gradient ∝ 0.99 * 0.01 = 0.0099   (tiny!)
```

The model is confidently wrong, but the gradient is almost zero because
sigmoid saturates. Cross-entropy doesn't have this problem because
log(0.01) = -4.6 produces a huge loss and gradient.

### Binary Cross-Entropy

For binary (yes/no) problems with a single output:

```
L = -mean(target * log(σ(pred)) + (1 - target) * log(1 - σ(pred)))
```

Numerically stable version avoids computing log(sigmoid) directly:

```
L = mean(max(pred, 0) - pred * target + log(1 + exp(-|pred|)))
```

Gradient: `σ(pred) - target`, same elegant form as multiclass cross-entropy.

In Axiom: `ax_bce_with_logits_loss()` in `losses.c`.


## Unit 5: Gradient Descent and Backpropagation

This is where everything connects. We have a model (sequence of tensor
operations), a loss function (measures error), and calculus (chain rule).
Now we put them together to make the model learn.

### The optimization problem

We want to find weights W that minimize the loss:

```
W* = argmin_W L(model(X; W), Y)
```

Where X is input data, Y is target labels, and model(X; W) is the prediction.

The loss landscape is a surface in high-dimensional space (one dimension per weight).
We're trying to find the lowest point on this surface. For a network with
millions of parameters, this surface has millions of dimensions.

### Gradient descent

We can't find the minimum analytically (the function is too complex).
Instead, we use an iterative approach:

1. Start at a random point (random weight initialization)
2. Compute the gradient of the loss with respect to each weight
3. Take a small step in the direction that reduces the loss
4. Repeat

```
w = w - lr * ∂L/∂w
```

The **learning rate** (lr) controls step size:
- Too large: we overshoot the minimum and oscillate or diverge
- Too small: we converge painfully slowly
- Just right: we smoothly descend to a good minimum

Typical values: 0.001 for Adam, 0.01 - 0.1 for SGD.

### Backpropagation: computing gradients efficiently

For a network with layers `f₁, f₂, ..., fₙ` and loss L:

```
input -> f₁ -> f₂ -> ... -> fₙ -> loss
  x      z₁    z₂          zₙ      L
```

We need ∂L/∂w for every weight in every layer. The chain rule gives us:

```
∂L/∂w_in_layer_i = ∂L/∂zₙ * ∂zₙ/∂zₙ₋₁ * ... * ∂z_{i+1}/∂z_i * ∂z_i/∂w
```

Computing this naively for each weight would repeat a lot of work.
**Backpropagation** avoids this by working backwards:

```
1. Forward pass: compute z₁, z₂, ..., zₙ, L  (save all intermediate values)

2. Backward pass:
   ∂L/∂zₙ = (direct from loss function)
   ∂L/∂zₙ₋₁ = ∂L/∂zₙ * ∂zₙ/∂zₙ₋₁
   ∂L/∂zₙ₋₂ = ∂L/∂zₙ₋₁ * ∂zₙ₋₁/∂zₙ₋₂
   ...
```

Each step reuses the result from the previous step. The total cost is
proportional to one forward pass — very efficient.

### Worked example: 2-layer network

Let's trace through a concrete example. Network:

```
z₁ = x @ W₁ + b₁     (linear layer 1)
a₁ = relu(z₁)         (activation)
z₂ = a₁ @ W₂ + b₂    (linear layer 2)
L = MSE(z₂, target)   (loss)
```

Forward pass (all values are computed and stored):
```
x = [1, 2] (input)
W₁ = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] (shape [2, 3])
b₁ = [0, 0, 0]

z₁ = [1, 2] @ W₁ + b₁ = [0.9, 1.2, 1.5]
a₁ = relu(z₁) = [0.9, 1.2, 1.5]  (all positive, so relu is identity)

W₂ = [[0.1], [0.2], [0.3]] (shape [3, 1])
b₂ = [0]

z₂ = [0.9, 1.2, 1.5] @ W₂ + b₂ = [0.9*0.1 + 1.2*0.2 + 1.5*0.3] = [0.78]

target = [1.0]
L = (0.78 - 1.0)² = 0.0484
```

Backward pass:
```
∂L/∂z₂ = 2 * (0.78 - 1.0) = -0.44

∂L/∂W₂ = a₁ᵀ @ ∂L/∂z₂ = [[0.9], [1.2], [1.5]] * (-0.44)
        = [[-0.396], [-0.528], [-0.66]]

∂L/∂a₁ = ∂L/∂z₂ @ W₂ᵀ = (-0.44) * [0.1, 0.2, 0.3] = [-0.044, -0.088, -0.132]

∂L/∂z₁ = ∂L/∂a₁ * relu'(z₁) = [-0.044, -0.088, -0.132] * [1, 1, 1]
        = [-0.044, -0.088, -0.132]   (all z₁ were positive, so relu' = 1)

∂L/∂W₁ = xᵀ @ ∂L/∂z₁ = [[1], [2]] @ [[-0.044, -0.088, -0.132]]
        = [[-0.044, -0.088, -0.132],
           [-0.088, -0.176, -0.264]]
```

Now we update:
```
W₁ = W₁ - lr * ∂L/∂W₁
W₂ = W₂ - lr * ∂L/∂W₂
```

Every weight gets pushed in the direction that reduces the loss.

### How Axiom implements this

In Axiom, you don't compute any of this by hand. The autograd engine does it:

```c
ax_tensor_t *z1 = ax_matmul(x, w1);      // records: z1 = matmul(x, w1)
ax_tensor_t *a1 = ax_relu(z1);            // records: a1 = relu(z1)
ax_tensor_t *z2 = ax_matmul(a1, w2);      // records: z2 = matmul(a1, w2)
ax_tensor_t *loss = ax_mse_loss(z2, target);

ax_backward(loss);  // computes ALL gradients automatically
// w1->grad and w2->grad are now filled in
```

Each `ax_*` operation creates a tensor with a `grad_fn` attached — a struct
containing the backward function and saved values. When `ax_backward` runs,
it topologically sorts these nodes and calls each backward function in
reverse order, accumulating gradients.

See `autograd.c` for the backward engine and `autograd_ops.c` for the
backward function of every operation.


## Unit 6: Optimizers — Better Than Plain Gradient Descent

Plain SGD works but has problems. Modern optimizers fix them.

### The problem with plain SGD

Consider a loss surface shaped like a narrow valley:

```
        /  steep walls  \
       /                 \
      /    shallow floor  \
     /____________________\
```

SGD oscillates across the steep walls (large gradients there) while making
slow progress along the shallow floor (small gradients). You can't just
increase the learning rate because the steep dimensions would diverge.

### Momentum: remember past gradients

Instead of using the raw gradient, maintain a **velocity** that accumulates:

```
v = β * v + gradient           (β is typically 0.9)
w = w - lr * v
```

The velocity averages out oscillations across the steep dimensions (they
alternate sign and cancel out) while accumulating in the consistent direction
(same sign adds up). Like a heavy ball rolling: it smooths the zigzag.

### Adam: adaptive learning rates

Different parameters need different learning rates. A weight connecting to
a frequently active input needs smaller updates than one connecting to a
rarely active input.

Adam tracks two things per parameter:

```
m = β₁ * m + (1 - β₁) * g            (1st moment: running mean of gradients)
v = β₂ * v + (1 - β₂) * g²           (2nd moment: running mean of squared gradients)
```

The update:

```
w = w - lr * m / (√v + ε)
```

Think about what this does:
- m smooths the gradient direction (like momentum)
- √v estimates the typical gradient magnitude for this parameter
- Dividing by √v normalizes the step: parameters with large gradients get
  smaller effective learning rates, parameters with small gradients get larger ones

There's a subtlety: m and v are initialized to zero, so they're biased toward
zero in early steps. **Bias correction** fixes this:

```
m_hat = m / (1 - β₁ᵗ)
v_hat = v / (1 - β₂ᵗ)
```

At step t=1 with β₁=0.9: `1 - 0.9¹ = 0.1`, so `m_hat = m / 0.1 = 10 * m`.
This corrects for the fact that m has only accumulated one gradient so far.
As t grows, `β₁ᵗ → 0` and the correction vanishes.

Default values (β₁=0.9, β₂=0.999, ε=1e-8) work well for almost everything.
This is why Adam is the default choice in most deep learning.

In Axiom: `adam_step()` in `optim.c`. The implementation follows the formulas
exactly. Per-parameter state (m, v) is stored in `ax_param_state_t`.

### AdamW: fixing weight decay

Weight decay (L2 regularization) adds a penalty for large weights:

```
L_total = L + λ * Σ w²
```

This is equivalent to adding `λ * w` to the gradient. But with Adam,
the adaptive learning rate interferes with this — large-gradient parameters
have their weight decay attenuated.

AdamW decouples them by applying weight decay directly to the weights:

```
w = w * (1 - lr * λ) - lr * m_hat / (√v_hat + ε)
```

The first term shrinks weights, the second term is the normal Adam update.
They don't interfere with each other. This gives better generalization
in practice and is the standard for training transformers.

In Axiom: `adam_step(opt, true)` for AdamW vs `adam_step(opt, false)` for Adam.
The boolean controls whether weight decay is decoupled.


## Unit 7: Weight Initialization — Why It Matters

### The variance problem

Consider a layer: `z = Σᵢ wᵢ * xᵢ` (sum of n terms).

If weights and inputs are independent with zero mean, then:

```
Var(z) = n * Var(w) * Var(x)
```

(This follows from the variance of a product of independent variables.)

If Var(w) = 1 and n = 1000 (a wide layer), then Var(z) = 1000 * Var(x).
After k layers: Var(output) = 1000^k * Var(input). Exponential explosion!

If Var(w) = 0.001, then Var(z) = 1 * Var(x). Looks fine for one layer.
But after k layers: 0.001^k * ... → 0. Everything vanishes.

We need `n * Var(w) = 1`, so `Var(w) = 1/n`. This keeps variance stable.

### Xavier/Glorot initialization

For a layer with `fan_in` inputs and `fan_out` outputs, Xavier balances
both forward and backward variance:

```
Var(w) = 2 / (fan_in + fan_out)
```

For uniform distribution: `w ~ U[-limit, limit]` where `limit = √(6 / (fan_in + fan_out))`

(Because Var(U[-a, a]) = a²/3, and we want a²/3 = 2/(fan_in + fan_out),
so a = √(6/(fan_in + fan_out)).)

This works well for sigmoid and tanh activations.

### He/Kaiming initialization

ReLU sets half the values to zero (those where z < 0). This halves the
variance at each layer. To compensate:

```
Var(w) = 2 / fan_in
```

The factor of 2 accounts for ReLU killing half the signal.

For uniform: `limit = √(6 / fan_in)`
For normal: `std = √(2 / fan_in)`

This is the default in Axiom when you call `ax_dense_create()`. The dense
layer constructor calls `ax_init_kaiming_uniform()`.

### Why zero initialization doesn't work

If all weights start at zero, all neurons in a layer produce the same output.
During backprop, they all get the same gradient. They all update the same way.
They stay identical forever. This is called the **symmetry problem** — the
network has many neurons but they all learn the same thing.

Random initialization breaks this symmetry. Each neuron starts at a different
point and learns different features.


## Unit 8: The Layer System and Model API

### Why layers?

Without layers, building a model means manually managing dozens of tensors:

```c
ax_tensor_t *w1 = ax_tensor_rand(shape1, 2, -1, 1);
ax_tensor_t *b1 = ax_tensor_zeros(shape_b1, 1, AX_FLOAT32);
w1->requires_grad = true;
b1->requires_grad = true;
// ... repeat for every layer, manually wire forward pass, manually collect params
```

With layers:

```c
ax_layer_t *model = ax_sequential_create();
ax_sequential_add(model, ax_dense_create(784, 128, true));
ax_sequential_add(model, ax_relu_layer_create());
ax_sequential_add(model, ax_dense_create(128, 10, true));

ax_tensor_t *out = ax_layer_forward(model, input);
```

Layers encapsulate: weight allocation, initialization, the forward computation,
and parameter tracking. The model API adds training on top.

### C polymorphism: vtables without classes

C has no classes, but we achieve polymorphism with the same technique used
by the Linux kernel, GLib, and SQLite.

Every layer type is a struct with `ax_layer_t` as its first field:

```c
struct ax_layer {
    ax_layer_ops_t ops;    // function pointers: forward(), destroy()
    ax_layer_type_t type;
    ax_tensor_t *params[AX_LAYER_MAX_PARAMS];
    int n_params;
};

typedef struct {
    ax_layer_t base;       // MUST be first field
    ax_tensor_t *weight;
    ax_tensor_t *bias;
} ax_dense_t;
```

Because `base` is at memory offset 0, a pointer to `ax_dense_t` is also
a valid pointer to `ax_layer_t`:

```c
ax_dense_t *dense = malloc(sizeof(ax_dense_t));
dense->base.ops.forward = dense_forward;  // set the vtable

ax_layer_t *layer = (ax_layer_t *)dense;  // safe cast
layer->ops.forward(layer, input);          // calls dense_forward
```

This is literally how C++ vtables work under the hood, minus the compiler
doing it for you.

### The complete training loop

Putting everything together:

```c
// 1. build the model
ax_layer_t *net = ax_sequential_create();
ax_sequential_add(net, ax_dense_create(2, 8, true));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_dense_create(8, 1, true));

// 2. create model and compile
ax_model_t *model = ax_model_create(net);
ax_optimizer_t *opt = ax_adam_create(model->params, model->n_params,
                                     0.001, 0.9, 0.999, 1e-8, 0);
ax_model_compile(model, opt, ax_mse_loss);

// 3. train
ax_model_fit(model, train_x, train_y, 1000, 100);

// 4. predict
ax_tensor_t *pred = ax_model_predict(model, test_x);
```

Inside `ax_model_train_step()`, this happens:

```
1. ax_optimizer_zero_grad(opt)       — clear all .grad tensors to zero
2. ax_layer_forward(net, input)      — forward pass (records computation graph)
3. loss_fn(pred, target)             — compute scalar loss
4. ax_backward(loss)                 — backward pass (fills all .grad tensors)
5. ax_optimizer_step(opt)            — update weights using gradients
```

This is the heartbeat of deep learning. Every framework does exactly this
loop, with different syntax.


## Unit 9: Code Map

Where everything lives in the project:

```
include/axiom/
  axiom.h          master include
  types.h          dtypes (float32, int32, ...), error codes, device enum
  error.h          thread-safe error handling, AX_CHECK macros
  memory.h         arena allocator for temporaries, aligned alloc for SIMD
  tensor.h         N-dim tensor: creation, reshape, transpose, views
  broadcast.h      numpy-style broadcast shape computation
  backend_ops.h    vtable that every compute backend fills in
  compute.h        dispatch layer: routes math ops to active backend
  ops.h            user-facing operations: ax_add, ax_matmul, ax_relu, etc.
  autograd.h       backward pass engine, grad_fn struct, gradient checking
  activations.h    leaky relu, elu, selu, gelu, swish, softplus, mish, softmax
  losses.h         mse, mae, cross-entropy, bce, huber
  optim.h          sgd, adam, adamw, rmsprop, adagrad
  init.h           xavier, kaiming, lecun, uniform, normal initialization
  layer.h          layer interface, dense, activation wrappers, sequential
  model.h          model container, compile, train_step, predict, fit

src/core/
  error.c          thread-local error state with formatted messages
  memory.c         arena (bump allocator + linked block list), aligned malloc
  tensor.c         tensor lifecycle, storage refcounting, zero-copy shape ops
  broadcast.c      broadcast shape computation (right-align, match dimensions)
  ops.c            allocates outputs, checks shapes, records autograd, dispatches
  autograd.c       backward(): topo sort via DFS, then calls grad_fns in reverse
  autograd_ops.c   backward fn for every differentiable op (the derivative cheat sheet)
  activations.c    activation forward + backward (leaky relu through softmax)
  losses.c         loss forward + custom backward where composition doesn't work
  optim.c          optimizer state management + update rules (sgd through adagrad)
  init.c           random number generation + scaled initialization schemes
  layer.c          dense forward, activation wrappers, sequential container
  model.c          model compile, train loop, predict with no_grad

src/compute/
  dispatch.c       single global pointer to active backend, dispatch functions
  backends/
    cpu_naive.c    reference implementation: pure C loops for every operation

tests/
  test.h              assert macros, test runner
  test_error.c        error state tests
  test_memory.c       arena allocator tests
  test_tensor.c       tensor creation, reshape, transpose, refcount tests
  test_compute.c      every compute op tested individually
  test_ops.c          broadcasting, chaining, shape inference tests
  test_autograd.c     gradient correctness for every differentiable op
  test_activations.c  activation forward values + gradients
  test_losses.c       loss values + gradients
  test_optim.c        optimizer convergence on f(x)=x², init sanity checks
  test_layer.c        dense, sequential, param collection, XOR training end-to-end
```


## Coming Next

- **Unit 10**: Serialization — saving and loading models for deployment
- **Unit 11**: Convolutions — what they are, why they work for images, the im2col trick
- **Unit 12**: Batch normalization — stabilizing deep network training
- **Unit 13**: Dropout — regularization by random disabling
- **Unit 14**: Recurrent networks — processing sequences, LSTM gates
- **Unit 15**: Attention and transformers — the mechanism behind modern AI
- **Unit 16**: SIMD optimization — processing 8 numbers at once
- **Unit 17**: Quantization — INT8 inference for embedded deployment
