# Unit 7: Weight Initialization

## Why This Matters

Before training starts, every weight needs an initial value. Initialize them wrong
and training fails immediately — activations explode to infinity or collapse to zero.
Proper initialization sets the scale so information flows smoothly through the network
from the very first step.


## 7.1 The Problem

Consider a dense layer: `z = W @ x`. If W has n inputs, then:

    z_j = sum_{i=1}^{n} w_{ij} * x_i

The variance of z_j depends on the variances of the weights and inputs:

    Var(z_j) = n * Var(w) * Var(x)

If Var(w) is too large, Var(z) grows exponentially with depth. If too small, it
shrinks exponentially. After 50 layers, the signal is either infinity or zero.

The fix: choose Var(w) so that Var(z) = Var(x). That means:

    Var(w) = 1 / n

This is the core idea behind all initialization schemes.


## 7.2 Xavier/Glorot Initialization

Designed for sigmoid and tanh activations (symmetric, roughly linear near zero).

**Xavier Uniform:**

    w ~ Uniform(-limit, limit)  where limit = sqrt(6 / (fan_in + fan_out))

**Xavier Normal:**

    w ~ Normal(0, std)  where std = sqrt(2 / (fan_in + fan_out))

Where:
- `fan_in` = number of input connections (in_features for dense)
- `fan_out` = number of output connections (out_features for dense)

The averaging of fan_in and fan_out preserves variance in both the forward and
backward passes.

Axiom:
```c
void ax_init_xavier_uniform(ax_tensor_t *t, int64_t fan_in, int64_t fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    // fill with Uniform(-limit, limit)
}
```


## 7.3 Kaiming/He Initialization

Xavier assumes the activation is linear. ReLU kills half the values (everything < 0
becomes 0), halving the variance. Kaiming compensates:

**Kaiming Uniform:**

    w ~ Uniform(-limit, limit)  where limit = sqrt(6 / fan_in)

**Kaiming Normal:**

    w ~ Normal(0, std)  where std = sqrt(2 / fan_in)

The factor of 2 (instead of 1) accounts for ReLU zeroing half the activations.
Only uses fan_in because the forward pass variance matters most.

This is Axiom's default for dense and conv layers:

```c
void ax_init_kaiming_uniform(ax_tensor_t *t, int64_t fan_in) {
    float limit = sqrtf(6.0f / (float)fan_in);
    // fill with Uniform(-limit, limit)
}
```

In `ax_dense_create`:
```c
ax_init_kaiming_uniform(d->weight, in_features);
```

In `ax_conv2d_create_ex`:
```c
ax_init_kaiming_uniform(c->weight, in_ch * kh * kw);
```

For convolutions, fan_in = in_channels * kernel_h * kernel_w (total number of
input connections to one output neuron).


## 7.4 LeCun Initialization

For SELU (self-normalizing) networks:

    w ~ Normal(0, std)  where std = sqrt(1 / fan_in)

Same as Kaiming but without the factor of 2. SELU activations have a built-in
self-normalizing property that assumes this exact variance.

```c
void ax_init_lecun_normal(ax_tensor_t *t, int64_t fan_in) {
    float std = sqrtf(1.0f / (float)fan_in);
    // fill with Normal(0, std)
}
```


## 7.5 Bias Initialization

Biases are almost always initialized to zero. They don't suffer from the
symmetry-breaking problem (weights need to be different from each other; biases
just shift the output).

Axiom:
```c
if (use_bias) {
    d->bias = ax_tensor_zeros(b_shape, 1, AX_FLOAT32);
}
```


## 7.6 The Math: Why These Formulas Work

For a single layer with no activation, input x with E[x_i] = 0:

    z_j = sum_{i=1}^{fan_in} w_{ij} * x_i

Since w and x are independent:

    Var(z_j) = fan_in * Var(w) * Var(x)

To preserve variance (Var(z) = Var(x)):

    Var(w) = 1 / fan_in

For the backward pass, similar analysis gives:

    Var(w) = 1 / fan_out

Xavier averages these: `Var(w) = 2 / (fan_in + fan_out)`.
Kaiming uses only fan_in: `Var(w) = 2 / fan_in` (the factor 2 accounts for ReLU).

For a uniform distribution U(-a, a), `Var = a^2/3`, so:

    a^2/3 = 2/fan_in  =>  a = sqrt(6/fan_in)    [Kaiming]
    a^2/3 = 2/(fan_in+fan_out)  =>  a = sqrt(6/(fan_in+fan_out))  [Xavier]


## 7.7 Random Number Generation

Axiom uses the Box-Muller transform to generate normally distributed random numbers
from uniform samples:

```c
static float rand_normal(float mean, float std) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / (float)RAND_MAX;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    return mean + std * z;
}
```

Box-Muller: given two uniform samples u1, u2 in (0, 1]:

    z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)

z is standard normal N(0, 1). Scale by std and shift by mean to get N(mean, std^2).

The RNG is seeded lazily on first use with `srand(time(NULL))`. Not
cryptographically secure (not needed for ML), but sufficient for initialization.


## 7.8 Available Initializers

| Function                    | Distribution              | Use Case           |
|-----------------------------|---------------------------|--------------------|
| `ax_init_xavier_uniform`    | U(-sqrt(6/(fi+fo)), ...)  | Sigmoid/Tanh       |
| `ax_init_xavier_normal`     | N(0, sqrt(2/(fi+fo)))     | Sigmoid/Tanh       |
| `ax_init_kaiming_uniform`   | U(-sqrt(6/fi), ...)       | ReLU (default)     |
| `ax_init_kaiming_normal`    | N(0, sqrt(2/fi))          | ReLU               |
| `ax_init_lecun_normal`      | N(0, sqrt(1/fi))          | SELU               |
| `ax_init_uniform`           | U(low, high)              | Custom range       |
| `ax_init_normal`            | N(mean, std)              | Custom distribution|
| `ax_init_zeros`             | 0                         | Biases             |
| `ax_init_ones`              | 1                         | Scale parameters   |
| `ax_init_constant`          | c                         | Fixed value        |


## Key Takeaways

1. Weight scale must match the layer width to prevent signal explosion/vanishing.
2. Kaiming (He) for ReLU networks. Xavier (Glorot) for sigmoid/tanh.
3. Biases are initialized to zero. Weights are never all-zero (breaks symmetry).
4. For convolutions, fan_in = in_channels * kernel_height * kernel_width.
5. Axiom defaults to Kaiming uniform for both dense and conv layers.
