# Unit 3: The Neuron

## Why This Matters

A neural network is made of neurons. Understanding one neuron — what it computes,
why it needs an activation function, how its gradients flow — gives you the intuition
for understanding the whole network.


## 3.1 The Biological Inspiration (Briefly)

A biological neuron receives signals from other neurons through dendrites, sums
them up in the cell body, and fires an output signal through the axon if the total
exceeds a threshold. The artificial neuron is a loose mathematical analogy.

Don't take the analogy too far. Artificial neurons are computing machines that
happen to share a naming convention with biology.


## 3.2 The Mathematical Neuron

A single neuron computes:

    z = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
    a = f(z)

Where:
- `x_1, ..., x_n` are the inputs (a vector)
- `w_1, ..., w_n` are the weights (learned parameters)
- `b` is the bias (another learned parameter)
- `z` is the pre-activation (weighted sum)
- `f` is the activation function
- `a` is the output (activation)

In vector notation:

    z = w^T * x + b = dot(w, x) + b
    a = f(z)


## 3.3 Why the Bias?

Without the bias, the neuron's decision boundary always passes through the origin.
The bias shifts it:

    z = w*x + b

With b = 0, z = 0 when x = 0 (always). With b != 0, the threshold shifts. The bias
gives the neuron freedom to activate even when inputs are zero, or to not activate
even when inputs are large.

In Axiom's dense layer, bias is optional:

```c
ax_layer_t *ax_dense_create(int64_t in_features, int64_t out_features, bool use_bias);
```


## 3.4 From One Neuron to a Layer

A single dense layer is just many neurons computed in parallel. Instead of one
weight vector, you have a weight matrix:

    Z = X @ W + b

Where:
- X is `[batch_size, in_features]` — a batch of input vectors
- W is `[in_features, out_features]` — each column is one neuron's weights
- b is `[out_features]` — one bias per neuron (broadcast across the batch)
- Z is `[batch_size, out_features]` — the output

This is why matrix multiplication is the fundamental operation. One matmul computes
all neurons in a layer simultaneously.

Axiom's dense forward pass (`layer.c`):

```c
static ax_tensor_t *dense_forward(ax_layer_t *self, ax_tensor_t *input) {
    ax_dense_t *d = (ax_dense_t *)self;
    ax_tensor_t *out = ax_matmul(input, d->weight);    // X @ W
    if (d->use_bias && d->bias) {
        ax_tensor_t *biased = ax_add(out, d->bias);    // + b (broadcast)
        return biased;
    }
    return out;
}
```


## 3.5 Activation Functions

Without an activation function, stacking layers is pointless:

    layer2(layer1(x)) = W2 @ (W1 @ x) = (W2 @ W1) @ x = W_combined @ x

It collapses to a single linear transformation. The activation function introduces
**nonlinearity**, giving the network the ability to learn curved, complex boundaries.

Common activations implemented in Axiom (`activations.c`):

**ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`
- Most popular. Simple, fast, works well in practice.
- Problem: "dying ReLU" — neurons that always output 0 stop learning.

**Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- Squashes output to (0, 1). Useful for probabilities.
- Problem: gradients vanish for large |x| (saturation).

**Tanh**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- Squashes to (-1, 1). Zero-centered, often better than sigmoid.
- Same saturation problem as sigmoid.

**Leaky ReLU**: `f(x) = x if x > 0, else alpha * x`  (alpha ~ 0.01)
- Fixes dying ReLU by allowing a small gradient when x < 0.

**ELU**: `f(x) = x if x > 0, else alpha * (e^x - 1)`
- Smooth version of leaky ReLU. Negative values saturate at -alpha.

**GELU**: `f(x) = x * Phi(x)` where Phi is the standard normal CDF
- Used in transformers (BERT, GPT). Smooth, non-monotonic.
- Axiom uses the tanh approximation for speed.

**Swish (SiLU)**: `f(x) = x * sigmoid(x)`
- Self-gated. Discovered by neural architecture search.
- `f'(x) = swish(x) + sigmoid(x) * (1 - swish(x))`

**Softmax**: `f(x_i) = e^(x_i) / sum(e^(x_j))`
- Converts a vector of logits into probabilities that sum to 1.
- Always used as the final layer for classification.
- Numerically stabilized by subtracting max(x) before exponentiating.


## 3.6 The Softmax Stability Trick

Naive softmax overflows easily. `e^(1000)` is infinity in float32. The fix:

    softmax(x)_i = e^(x_i - max(x)) / sum(e^(x_j - max(x)))

Subtracting max(x) from all elements doesn't change the output (the factors cancel
in the ratio) but keeps the exponents in a safe range. Axiom implements this in
`activations.c`:

```c
float mx = -FLT_MAX;
for (int64_t i = 0; i < n; i++) {
    float v = ad[a->offset + i];
    if (v > mx) mx = v;
}
// ... exp(x_i - mx) / sum(exp(x_j - mx))
```


## 3.7 Layer Wrapping in Axiom

Activations can be used as standalone functions or wrapped as layers for use in
sequential models. The layer wrapper is a thin shim:

```c
static ax_tensor_t *relu_forward(ax_layer_t *self, ax_tensor_t *input) {
    return ax_relu(input);
}

ax_layer_t *ax_relu_layer_create(void) {
    return make_activation(AX_LAYER_RELU, relu_forward);
}
```

Parameterized activations (leaky ReLU, ELU) store their parameter in the layer struct:

```c
typedef struct {
    ax_layer_t base;
    float alpha;    // for leaky relu, elu
    int axis;       // for softmax
} ax_activation_layer_t;
```


## Key Takeaways

1. A neuron computes: weighted sum of inputs + bias, then activation function.
2. A dense layer is many neurons computed in parallel via matrix multiplication.
3. Without nonlinear activations, stacking layers collapses to a single linear transform.
4. ReLU is the default choice; GELU/Swish for transformers; sigmoid/softmax for outputs.
5. Softmax must be numerically stabilized by subtracting the max.
