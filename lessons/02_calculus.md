# Unit 2: Calculus for Deep Learning

## Why This Matters

Training a neural network means finding the weights that minimize a loss function.
Calculus — specifically derivatives — tells us which direction to adjust each weight.
You don't need to be a calculus expert, but you need to understand derivatives, the
chain rule, and partial derivatives. That's it. Those three ideas power all of
backpropagation.


## 2.1 Derivatives: Rate of Change

The derivative of f(x) measures how fast f changes when x changes by a tiny amount:

    f'(x) = lim[h->0] (f(x + h) - f(x)) / h

Intuition: if f(x) = x^2, then f'(x) = 2x. At x = 3, the derivative is 6, meaning
a tiny increase in x produces ~6x that increase in f.

Key derivatives you'll see in deep learning:

    f(x) = x^n       =>  f'(x) = n * x^(n-1)
    f(x) = e^x       =>  f'(x) = e^x
    f(x) = ln(x)     =>  f'(x) = 1/x
    f(x) = 1/(1+e^-x) =>  f'(x) = f(x) * (1 - f(x))    [sigmoid]
    f(x) = max(0, x) =>  f'(x) = 1 if x > 0, else 0     [relu]


## 2.2 Partial Derivatives

When f depends on multiple variables, the **partial derivative** with respect to one
variable treats all others as constants:

    f(x, y) = x^2 + xy + y^2

    df/dx = 2x + y       (treating y as constant)
    df/dy = x + 2y       (treating x as constant)

In a neural network with millions of weights, the loss L depends on all of them.
We compute dL/dw_i for each weight w_i independently.


## 2.3 The Chain Rule

The chain rule is the single most important formula in deep learning. If you have
composed functions:

    z = f(g(x))

Then:

    dz/dx = f'(g(x)) * g'(x)

Or more compactly, if y = g(x) and z = f(y):

    dz/dx = (dz/dy) * (dy/dx)

This generalizes to any number of composed functions:

    dL/dw = (dL/da_n) * (da_n/da_{n-1}) * ... * (da_2/da_1) * (da_1/dw)

This is exactly what backpropagation computes. A neural network is a chain of
operations (matmul, add bias, activate, matmul, ...). The chain rule lets us compute
how the loss changes with respect to any weight by multiplying local derivatives
backwards through the chain.


## 2.4 The Gradient

The **gradient** of a scalar function with respect to a vector is the vector of all
partial derivatives:

    grad_w L = [dL/dw_1, dL/dw_2, ..., dL/dw_n]

The gradient points in the direction of steepest increase. To minimize the loss,
we move in the opposite direction (gradient descent).

In Axiom, every tensor can have an associated `grad` tensor of the same shape:

```c
typedef struct ax_tensor {
    ...
    bool requires_grad;        // track gradients for this tensor?
    struct ax_tensor *grad;    // accumulated gradient (same shape)
    void *grad_fn;             // backward function that produced this tensor
} ax_tensor_t;
```


## 2.5 Numerical Gradient Checking

How do you verify your analytical gradients are correct? Compute them numerically
using the definition of the derivative:

    df/dx ~ (f(x + eps) - f(x - eps)) / (2 * eps)

This is the centered finite difference formula. It's O(eps^2) accurate (much better
than the one-sided version).

Axiom implements this in `ax_grad_check()`:

```c
double ax_grad_check(
    ax_tensor_t *(*forward_fn)(ax_tensor_t *input),
    ax_tensor_t *input,
    double eps)
{
    // 1. Compute analytical gradient via backward()
    // 2. For each element, perturb by +eps and -eps
    // 3. Compare numerical gradient to analytical gradient
    // 4. Return max absolute difference
}
```

If `ax_grad_check` returns a value < 1e-5, your gradients are almost certainly correct.
If it returns > 1e-2, something is wrong.


## 2.6 Derivatives of Common Activations

Here are the derivatives used in Axiom's activation backward functions:

**ReLU:** `f(x) = max(0, x)`

    f'(x) = 1 if x > 0, else 0

**Sigmoid:** `f(x) = 1 / (1 + e^(-x))`

    f'(x) = f(x) * (1 - f(x))

**Tanh:** `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

    f'(x) = 1 - f(x)^2

**Leaky ReLU:** `f(x) = x if x > 0, else alpha * x`

    f'(x) = 1 if x > 0, else alpha

**ELU:** `f(x) = x if x > 0, else alpha * (e^x - 1)`

    f'(x) = 1 if x > 0, else f(x) + alpha

**GELU (approximate):** `f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

    f'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
    where t = tanh(sqrt(2/pi) * (x + 0.044715*x^3))

**Swish (SiLU):** `f(x) = x * sigmoid(x)`

    f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
           = swish(x) + sigmoid(x) * (1 - swish(x))

These are all implemented in `src/core/activations.c`.


## 2.7 Multivariate Chain Rule and Jacobians

When functions map vectors to vectors (like a layer), the chain rule generalizes
using the **Jacobian matrix** — the matrix of all partial derivatives:

    J[i][j] = d(output_i) / d(input_j)

For backpropagation, we don't actually compute the full Jacobian. Instead, we compute
**vector-Jacobian products** (VJPs): given the gradient of the loss with respect to
the output, compute the gradient with respect to the input. This is much cheaper
because we only need one row of the Jacobian at a time.

In code, each backward function takes `grad_out` (the upstream gradient) and produces
`grad_in` (the downstream gradient). This is the VJP.


## Key Takeaways

1. The derivative tells you how a function changes as its input changes.
2. The chain rule lets you compose derivatives through a sequence of operations.
3. The gradient is a vector pointing toward steepest ascent; negate it to descend.
4. Every activation function has a known derivative used in backpropagation.
5. Gradient checking (numerical vs analytical) is the primary debugging tool.
