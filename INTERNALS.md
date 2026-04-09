# Axiom Internals: A Complete Guide

This document teaches you everything inside Axiom, from the ground up.
Each lesson builds on the previous one. By the end, you'll understand
every line of code in this project and the math behind it.

This is a living document. As new modules are added, new lessons appear.

---

## Lesson 1: Tensors (What They Are and Why They Matter)

### The concept

A **tensor** is just a multi-dimensional array of numbers. That's it.

- A scalar (single number) is a 0-dimensional tensor: `42`
- A vector is a 1D tensor: `[1, 2, 3]`
- A matrix is a 2D tensor: `[[1, 2], [3, 4]]`
- An image is a 3D tensor: height x width x channels (e.g., 224x224x3)
- A batch of images is 4D: batch x height x width x channels

Neural networks do nothing but multiply, add, and transform tensors.
Every weight, every input, every output is a tensor.

### Shape and strides

A tensor's **shape** tells you its dimensions: `[2, 3]` means 2 rows, 3 columns.

**Strides** tell you how many elements to skip to move one step along each dimension.
For a `[2, 3]` tensor stored in row-major order (C layout):

```
Data in memory: [a, b, c, d, e, f]

shape   = [2, 3]
strides = [3, 1]   (skip 3 to move down a row, skip 1 to move right a column)

a b c     [0,0]=a  [0,1]=b  [0,2]=c
d e f     [1,0]=d  [1,1]=e  [1,2]=f
```

To access element `[i, j]`, you compute: `data[i * stride[0] + j * stride[1]]`

Why strides? Because they let us do **zero-copy operations**:
- **Transpose**: just swap the strides. `strides = [1, 3]` instead of `[3, 1]`.
  The data doesn't move in memory, we just read it differently.
- **Reshape**: if the data is contiguous, just change the shape and recompute strides.
  Again, no data is copied.
- **Slicing/views**: adjust the offset and strides. Still no copy.

This is how NumPy and PyTorch work internally.

### In our code

Look at `include/axiom/tensor.h`:

```c
typedef struct ax_tensor {
    ax_storage_t *storage;     // the actual data buffer (shared across views)
    int64_t shape[AX_MAX_DIMS];
    int64_t strides[AX_MAX_DIMS];
    int ndim;
    ax_dtype_t dtype;
    size_t offset;             // where our data starts within storage
    ...
} ax_tensor_t;
```

The `storage` is reference-counted. When you transpose a tensor, the new tensor
points to the same storage with different strides. When both tensors are destroyed,
the storage is freed only when the last reference goes away.

See `src/core/tensor.c`: `ax_tensor_transpose()` just swaps shape and strides,
increments the storage refcount, and returns. No data is ever copied.


## Lesson 2: Memory Management (Arenas and Alignment)

### Why not just malloc/free everywhere?

During a neural network's forward pass, you create hundreds of temporary tensors
(intermediate results). Each one needs a malloc and a free. This is:
1. Slow (malloc has overhead per call)
2. Fragmenting (lots of small allocations scatter your data in memory)

### Arena allocator

An arena is a bulk allocator. You grab a big chunk of memory upfront, then
"allocate" from it by just bumping a pointer forward. When you're done,
you reset the whole thing at once. No individual frees.

```
Before any allocations:
[                           big block                            ]
^
used = 0

After alloc(128):
[####128 bytes####|                                              ]
                  ^
                  used = 128

After alloc(64):
[####128 bytes####|##64##|                                       ]
                         ^
                         used = 192

After reset():
[                           big block                            ]
^
used = 0    (memory is reusable, not freed)
```

See `src/core/memory.c`. If an allocation doesn't fit in the current block,
a new block is allocated and linked into the chain.

### Alignment

SIMD instructions (AVX, SSE) require data to be aligned to specific boundaries
(16, 32, or 64 bytes). Even without SIMD, aligned data is faster because of
how CPU caches work. We default to 64-byte alignment for everything.

`ax_aligned_alloc()` over-allocates, aligns the pointer, and stashes the
original pointer just before the aligned address so `ax_aligned_free()` can
find it and pass it to `free()`.


## Lesson 3: The Compute Backend System

### The problem

You want `ax_matmul(a, b)` to work. But the actual multiplication could happen
in many ways:
- Pure C triple loop (slow but correct)
- AVX2 SIMD (fast on modern x86 CPUs)
- cuBLAS on GPU (very fast for large matrices)
- OpenBLAS (optimized CPU BLAS library)

### The solution: a vtable

Every math operation is a function pointer in a struct:

```c
typedef struct {
    const char *name;
    ax_status_t (*add)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*gemm)(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out);
    ax_status_t (*relu)(const ax_tensor_t *in, ax_tensor_t *out);
    // ... every operation
} ax_backend_ops_t;
```

Each backend fills in this struct with its implementations:

```c
// in cpu_naive.c:
const ax_backend_ops_t ax_cpu_naive_ops = {
    .name = "cpu_naive",
    .add  = cpu_add,
    .gemm = cpu_gemm,
    ...
};
```

At runtime, one backend is active. All operations route through it:

```c
// in dispatch.c:
static const ax_backend_ops_t *active_ops = &ax_cpu_naive_ops;

ax_status_t ax_compute_add(a, b, out) {
    return active_ops->add(a, b, out);
}
```

This is one function pointer dereference. The cost is negligible.
But it means everything above (autograd, layers, models) doesn't
care what's doing the actual math. Swap in a CUDA backend and
the same training code runs on GPU.

### How the naive backend does things

`src/compute/backends/cpu_naive.c` is the reference implementation.
It uses macros to avoid repeating boilerplate:

```c
#define DEFINE_BINOP(name, op_expr) \
static ax_status_t cpu_##name(const ax_tensor_t *a, const ax_tensor_t *b, ax_tensor_t *out) { \
    int64_t n = tensor_numel(out); \
    for (int64_t i = 0; i < n; i++) { \
        float va = bcast_get_f32(a, out, i); \
        float vb = bcast_get_f32(b, out, i); \
        tensor_set_f32(out, i, (op_expr)); \
    } \
    return AX_OK; \
}

DEFINE_BINOP(add, va + vb)
DEFINE_BINOP(mul, va * vb)
```

The `bcast_get_f32` function handles broadcasting: it maps an output index
back to the input index, collapsing dimensions that were broadcast.


## Lesson 4: Broadcasting

### The rules (same as NumPy)

When you add a `[2, 3]` tensor and a `[3]` tensor, broadcasting makes them
compatible by "stretching" the smaller one:

```
a: [2, 3]    b: [3]

Step 1: right-align the shapes
  a: 2  3
  b:    3

Step 2: pad with 1s
  a: 2  3
  b: 1  3

Step 3: for each dimension, they must be equal or one of them must be 1
  2 vs 1 -> ok (1 broadcasts to 2)
  3 vs 3 -> ok (equal)

Result shape: [2, 3]
```

The key insight: **broadcasting never copies data**. The "stretched" values
are computed on the fly by repeating indices. When dimension size is 1,
that index is always 0 regardless of the output position.

### In the gradient computation

Broadcasting creates a problem for gradients. If `a` has shape `[3]` and was
broadcast to `[2, 3]` during addition, the gradient that flows back is `[2, 3]`.
But `a`'s gradient should be `[3]`. So we **sum along the broadcast dimensions**
to collapse it back. This is what `accumulate_grad()` in `autograd_ops.c` does.


## Lesson 5: Automatic Differentiation (The Core Idea)

This is the most important lesson. Understanding autograd means
understanding how neural networks learn.

### What we're trying to do

We have a loss function `L(w)` where `w` are the network's weights.
We want to compute `dL/dw` for every weight so we can update them
to make the loss smaller (gradient descent).

### The chain rule

If `L = f(g(h(w)))`, then:

```
dL/dw = (dL/df) * (df/dg) * (dg/dh) * (dh/dw)
```

Each factor is a local derivative. Reverse-mode autodiff computes
these efficiently from output to input.

### How it works in Axiom

Every operation records itself when gradient tracking is enabled.
When you write:

```c
ax_tensor_t *z = ax_matmul(x, w);    // records: z came from matmul(x, w)
ax_tensor_t *a = ax_relu(z);          // records: a came from relu(z)
ax_tensor_t *loss = ax_sum(a, -1);    // records: loss came from sum(a)
```

Each result tensor gets a `grad_fn` that knows:
1. What operation created it (backward function pointer)
2. What the inputs were (so it can route gradients back)
3. Any saved values needed for gradient computation

When you call `ax_backward(loss)`:

```
1. Set loss.grad = 1.0  (dL/dL = 1, this seeds the process)

2. Topological sort: [loss, a, z]  (reverse order)

3. Walk backwards:
   - loss.grad_fn->backward(grad=1.0)
     -> This is sum_backward: broadcast 1.0 to a's shape
     -> a.grad = [1, 1, 1, ...]

   - a.grad_fn->backward(grad=a.grad)
     -> This is relu_backward: multiply by (z > 0 ? 1 : 0)
     -> z.grad = a.grad * relu'(z)

   - z.grad_fn->backward(grad=z.grad)
     -> This is matmul_backward:
     -> x.grad += z.grad @ w^T
     -> w.grad += x^T @ z.grad
```

That's it. Every weight now has its gradient. One forward pass,
one backward pass, and you have all the information needed to
update every parameter in the network.

### The topological sort

The computation forms a directed acyclic graph (DAG). We need to
process nodes in reverse order (outputs before inputs) so that
when we compute a node's backward, all the gradients flowing
into it have already been accumulated.

We use DFS with post-order traversal. The last node visited
(deepest in the graph) gets processed first during backward.

See `src/core/autograd.c`, the `topo_sort_dfs` function.


## Lesson 6: Derivative Cheat Sheet

Here are the derivatives used in `autograd_ops.c`.
`g` is the gradient flowing in from above (grad_output).

### Element-wise ops

| Forward          | Backward (gradient w.r.t. input)          |
|-----------------|-------------------------------------------|
| `y = a + b`     | `da = g`, `db = g`                        |
| `y = a - b`     | `da = g`, `db = -g`                       |
| `y = a * b`     | `da = g * b`, `db = g * a`                |
| `y = a / b`     | `da = g / b`, `db = -g * a / b^2`         |
| `y = -a`        | `da = -g`                                 |
| `y = exp(a)`    | `da = g * exp(a)` (= `g * y`)             |
| `y = log(a)`    | `da = g / a`                              |
| `y = sqrt(a)`   | `da = g / (2 * sqrt(a))` (= `g / (2*y)`) |
| `y = a^2`       | `da = g * 2a`                             |
| `y = a * c`     | `da = g * c`  (c is scalar constant)      |

### Activations

| Forward          | Backward                                  |
|-----------------|-------------------------------------------|
| `y = relu(a)`    | `da = g * (a > 0 ? 1 : 0)`               |
| `y = sigmoid(a)` | `da = g * y * (1 - y)`                    |
| `y = tanh(a)`    | `da = g * (1 - y^2)`                      |

Notice sigmoid and tanh backward use the **output** `y`, not the input `a`.
This is why we save the output tensor in the grad_fn.

### Matrix multiply

```
y = a @ b    (where a is [m,k] and b is [k,n])

da = g @ b^T     (g is [m,n], b^T is [n,k], result is [m,k])
db = a^T @ g     (a^T is [k,m], g is [m,n], result is [k,n])
```

Why? Think about it dimensionally. If `y[i,j] = sum_k(a[i,k] * b[k,j])`,
then `dy/da[i,k] = b[k,j]` and we need to sum over `j` (the output dimension),
which is exactly what `g @ b^T` does.

### Reductions

| Forward          | Backward                                  |
|-----------------|-------------------------------------------|
| `y = sum(a)`     | `da = g * ones_like(a)` (broadcast g)     |
| `y = mean(a)`    | `da = g * (1/N) * ones_like(a)`           |

Sum backward just broadcasts the gradient back to the input shape.
Mean backward does the same but scales by 1/N.


## Lesson 7: The Forward-Backward Pattern

Putting it all together. Here's what happens when you train:

```c
// 1. Forward pass: compute predictions and loss
ax_tensor_t *h = ax_matmul(input, w1);     // hidden = input @ w1
ax_tensor_t *a = ax_relu(h);                // activated = relu(hidden)
ax_tensor_t *pred = ax_matmul(a, w2);       // predictions = activated @ w2
ax_tensor_t *diff = ax_sub(pred, target);   // error = predictions - target
ax_tensor_t *sq = ax_square(diff);           // squared error
ax_tensor_t *loss = ax_mean(sq, -1);         // mean squared error

// 2. Backward pass: compute all gradients
ax_backward(loss);

// 3. Update weights: w -= learning_rate * w.grad
// (this is what optimizers will do in Phase 3)
```

During step 1, each operation records itself in the computation graph.
During step 2, `ax_backward` walks the graph in reverse and fills in
`.grad` for every tensor that has `requires_grad = true`.
Step 3 uses those gradients to update the weights.

This is the heartbeat of neural network training. Every framework
(PyTorch, TensorFlow, JAX) does exactly this, just with different
syntax and optimizations.


## Lesson 8: Code Map

Where everything lives:

```
include/axiom/
  types.h          data types, error codes, device enum
  error.h          error handling API (AX_CHECK macros)
  memory.h         arena allocator, aligned alloc
  tensor.h         the tensor struct and all creation/manipulation functions
  broadcast.h      numpy-style broadcast shape computation
  backend_ops.h    the vtable every backend must fill in
  compute.h        dispatch layer (routes ops to active backend)
  ops.h            user-facing operations (ax_add, ax_matmul, etc.)
  autograd.h       backward pass engine, grad_fn, gradient checking
  axiom.h          master include (pulls in everything)

src/core/
  error.c          thread-local error state
  memory.c         arena and aligned allocation implementations
  tensor.c         tensor creation, storage refcounting, shape ops
  broadcast.c      broadcast shape computation
  ops.c            user-facing ops (allocates output, records autograd)
  autograd.c       backward engine (topo sort, backward traversal)
  autograd_ops.c   backward functions for every differentiable op
  activations.c    leaky relu, elu, selu, gelu, swish, softplus, mish, softmax
  losses.c         mse, mae, cross-entropy, bce, huber loss
  optim.c          sgd, adam, adamw, rmsprop, adagrad
  init.c           xavier, kaiming, lecun, uniform, normal initialization

src/compute/
  dispatch.c       routes calls to active backend
  backends/
    cpu_naive.c    pure C reference implementation of all ops

tests/
  test.h              minimal test harness (assert macros)
  test_error.c        error handling tests
  test_memory.c       arena allocator tests
  test_tensor.c       tensor creation and shape manipulation tests
  test_compute.c      compute backend tests (every op individually)
  test_ops.c          high-level ops tests (broadcasting, chaining)
  test_autograd.c     gradient correctness tests for every differentiable op
  test_activations.c  activation function forward + gradient tests
  test_losses.c       loss function value and gradient tests
  test_optim.c        optimizer convergence tests + init tests
```


## Lesson 9: Loss Functions (Measuring How Wrong You Are)

A loss function measures the gap between your model's predictions and
the true answer. Training is about minimizing this number.

### MSE (Mean Squared Error)

```
L = (1/n) * sum((pred_i - target_i)^2)
```

Squares the errors, so large errors are penalized much more than small ones.
The gradient is `2*(pred - target) / n`. This pulls predictions toward
their targets proportionally to the error.

Good for regression. Bad for classification (doesn't understand probabilities).

### Cross-Entropy (The Classification Loss)

This is the most important loss for classification and understanding it
requires understanding softmax first.

**Softmax** converts raw logits (any real numbers) into probabilities:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

The outputs are all positive and sum to 1. The largest logit gets the
highest probability. Numerically, we subtract max(x) before exp to
avoid overflow:

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**Cross-entropy** then measures how different the predicted distribution is
from the target distribution:

```
L = -(1/batch) * sum(target_i * log(softmax(pred_i)))
```

If target is one-hot (e.g., [0, 0, 1, 0] for class 2), only the true
class contributes to the loss. High confidence in the correct class = low loss.

The beautiful thing: the gradient of cross-entropy + softmax simplifies to:

```
dL/d(pred) = (softmax(pred) - target) / batch_size
```

This is one of the cleanest results in deep learning. No complicated
chain rule needed. The gradient is just "what you predicted minus what
you should have predicted." See `ce_backward` in `losses.c`.

### Binary Cross-Entropy

For binary (yes/no) classification:

```
L = -mean(target * log(sigmoid(pred)) + (1-target) * log(1-sigmoid(pred)))
```

Numerically stable version avoids computing log(sigmoid) directly:

```
L = mean(max(pred, 0) - pred*target + log(1 + exp(-|pred|)))
```

Gradient: `sigmoid(pred) - target` (same elegant simplification).

### Huber Loss

A blend of MSE and MAE. Quadratic for small errors (smooth, easy to optimize),
linear for large errors (robust to outliers):

```
L = 0.5 * x^2           if |x| <= delta
    delta * (|x| - 0.5*delta)  otherwise
```


## Lesson 10: Optimizers (How to Use Gradients)

Once you have gradients from backward, you need to update the weights.
The simplest approach is `w = w - lr * grad`, but modern optimizers
do much better.

### SGD (Stochastic Gradient Descent)

```
w = w - lr * grad
```

Simple, but can oscillate in narrow valleys and gets stuck at saddle points.

### SGD with Momentum

Imagine a ball rolling down a hill. It accumulates speed in consistent directions
and dampens oscillations.

```
v = momentum * v + grad        (velocity accumulates past gradients)
w = w - lr * v                 (update using velocity, not raw gradient)
```

Typical momentum = 0.9 means 90% of the previous velocity is retained.
This smooths out the zigzag and accelerates in consistent directions.

### Adam (Adaptive Moment Estimation)

The most popular optimizer. Combines momentum with per-parameter learning rates.

```
m = beta1 * m + (1 - beta1) * grad          (1st moment: mean of gradients)
v = beta2 * v + (1 - beta2) * grad^2        (2nd moment: mean of squared gradients)
m_hat = m / (1 - beta1^t)                    (bias correction for early steps)
v_hat = v / (1 - beta2^t)                    (bias correction for early steps)
w = w - lr * m_hat / (sqrt(v_hat) + eps)     (update)
```

Why this works:
- `m` is like momentum (smooths gradient direction)
- `v` estimates the variance of gradients (how noisy each parameter is)
- Dividing by `sqrt(v)` gives each parameter its own effective learning rate:
  noisy parameters get smaller updates, stable ones get larger updates
- Bias correction handles the fact that m and v start at zero

Default: beta1=0.9, beta2=0.999, eps=1e-8.

### AdamW (Decoupled Weight Decay)

Standard Adam applies weight decay through the gradient: `grad += wd * w`.
AdamW applies it directly to the weights: `w *= (1 - lr * wd)`.

This seems trivial but makes a real difference. With standard Adam, the
adaptive learning rate interferes with weight decay. Decoupling them
gives better generalization.

### RMSProp

A precursor to Adam. Keeps a running average of squared gradients:

```
v = rho * v + (1 - rho) * grad^2
w = w - lr * grad / (sqrt(v) + eps)
```

Like Adam without the momentum (first moment). Parameters with large
recent gradients get smaller effective learning rates.


## Lesson 11: Weight Initialization (Why It Matters)

### The problem

If weights are too large, activations explode exponentially through layers.
If too small, they shrink to zero and gradients vanish (the network can't learn).

We need the variance of activations to stay roughly constant as data flows
through the network.

### Xavier/Glorot initialization

Analyzed for linear layers with sigmoid/tanh activation.

For a layer with `fan_in` inputs and `fan_out` outputs, we want:
```
Var(output) = Var(input)
```

This gives us:
```
Var(weight) = 2 / (fan_in + fan_out)
```

Uniform version: sample from `U[-limit, limit]` where `limit = sqrt(6 / (fan_in + fan_out))`
Normal version: sample from `N(0, std)` where `std = sqrt(2 / (fan_in + fan_out))`

### He/Kaiming initialization

ReLU kills half the values (sets negatives to 0), so Xavier underestimates the needed variance by a factor of 2.

```
Var(weight) = 2 / fan_in
```

This is the standard for networks using ReLU.

### Why only fan_in?

When doing forward propagation, output variance depends on fan_in.
When doing backward propagation, gradient variance depends on fan_out.
He init focuses on forward stability. Xavier balances both.


## Lesson 12: Layers and the Module System

### Why layers?

Without layers, building a model looks like this:

```c
ax_tensor_t *w1 = ax_tensor_rand(...);
ax_tensor_t *b1 = ax_tensor_zeros(...);
ax_tensor_t *w2 = ax_tensor_rand(...);
// ... manually init each, manually track each, manually wire forward pass
```

With layers:

```c
ax_layer_t *model = ax_sequential_create();
ax_sequential_add(model, ax_dense_create(784, 128, true));
ax_sequential_add(model, ax_relu_layer_create());
ax_sequential_add(model, ax_dense_create(128, 10, true));

ax_tensor_t *output = ax_layer_forward(model, input);
```

Layers encapsulate weights, initialization, and the forward computation.
Sequential chains them. The model API adds optimizer and loss on top.

### How it works in C (polymorphism without classes)

C doesn't have classes, but we fake it with the same trick the Linux kernel uses:
embed a "base struct" as the first field of every concrete struct, then cast between them.

```c
/* base */
struct ax_layer {
    ax_layer_ops_t ops;    /* vtable: forward(), destroy() */
    ax_layer_type_t type;
    ax_tensor_t *params[AX_LAYER_MAX_PARAMS];
    int n_params;
    ...
};

/* concrete */
typedef struct {
    ax_layer_t base;       /* MUST be first field */
    ax_tensor_t *weight;
    ax_tensor_t *bias;
} ax_dense_t;
```

Because `base` is at offset 0, a pointer to `ax_dense_t` is also a valid
pointer to `ax_layer_t`. We can pass it to any function expecting a layer:

```c
ax_dense_t *d = ...;
ax_layer_forward((ax_layer_t *)d, input);  /* calls d->base.ops.forward */
```

The `ops.forward` function pointer was set to `dense_forward` during creation,
so the right implementation runs. This is exactly how C++ vtables work under
the hood, minus the compiler magic.

### The Dense layer

The fundamental building block. Computes:

```
output = input @ weight + bias
```

Where `input` is `[batch, in_features]`, `weight` is `[in_features, out_features]`,
and `bias` is `[out_features]`. The bias gets broadcast across the batch dimension.

Weight is initialized with Kaiming uniform (good for ReLU networks).
Bias is initialized to zeros.

### Sequential container

Just an array of layer pointers. Forward pass chains them:

```c
x = input;
for each layer:
    x = layer.forward(x);
return x;
```

Parameter collection recurses into sublayers, so `ax_layer_get_params`
on a sequential gives you every trainable tensor in the whole model.

### The Model API

Wraps a network + optimizer + loss into a single object:

```c
ax_model_t *m = ax_model_create(net);
ax_model_compile(m, optimizer, ax_mse_loss);
ax_model_fit(m, train_x, train_y, epochs, print_every);
```

`train_step` does the full loop: zero grad -> forward -> loss -> backward -> step.
`predict` switches to eval mode and disables gradient tracking.

### Embedded considerations

All arrays are fixed-size (no malloc during forward pass for layer bookkeeping).
`AX_SEQ_MAX_LAYERS` (64) and `AX_MODEL_MAX_PARAMS` (256) are compile-time
constants that can be tuned down for memory-constrained targets.
No strings, no dynamic dispatch beyond the function pointer call.


## Coming Next

As the project grows, new lessons will be added:

- **Lesson 13**: Serialization (saving/loading models for deployment)
- **Lesson 14**: Convolutions (im2col trick, how CNNs see)
- **Lesson 15**: Batch normalization (why deep networks need it)
- **Lesson 16**: Recurrent networks (LSTM gates, backprop through time)
- **Lesson 17**: Attention and transformers (the mechanism behind modern AI)
- **Lesson 18**: SIMD optimization (how AVX makes math 8x faster)
- **Lesson 19**: Quantization (INT8 inference for embedded deployment)
- **Lesson 20**: GPU compute (CUDA kernels, memory coalescing)
