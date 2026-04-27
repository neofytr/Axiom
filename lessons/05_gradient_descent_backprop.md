# Unit 5: Gradient Descent and Backpropagation

## Why This Matters

Gradient descent is how neural networks learn. Backpropagation is how they compute
the gradients needed for gradient descent. Together, they form the training algorithm
that powers all of modern deep learning.


## 5.1 The Optimization Problem

Training a neural network means solving:

    w* = argmin_w L(w)

Find the weights w that minimize the loss L. The loss surface is a function of
potentially millions of parameters — we can't search exhaustively.


## 5.2 Gradient Descent

The gradient of L with respect to w points uphill. Move the opposite direction:

    w_new = w_old - lr * grad_w(L)

Where `lr` (learning rate) controls the step size. This is the simplest form of
gradient descent. Repeat until convergence.

- **lr too large**: overshoots the minimum, loss oscillates or diverges.
- **lr too small**: converges too slowly, gets stuck in local minima.
- **lr just right**: smooth convergence to a good minimum.

Typical starting values: 0.001 for Adam, 0.01 for SGD with momentum.


## 5.3 Backpropagation: The Algorithm

Backpropagation is just the chain rule applied systematically. Given a computation
graph (forward pass), backprop computes gradients by walking backward:

1. Compute the forward pass: input -> layer1 -> layer2 -> ... -> loss
2. Seed the gradient: dL/dL = 1
3. For each operation in reverse order:
   - Take the upstream gradient (from the output side)
   - Multiply by the local derivative (how this op's output depends on its input)
   - Pass the result downstream (toward the input side)

For a chain: `loss = f(g(h(x)))`:

    dL/dx = dL/df * df/dg * dg/dh * dh/dx

Each operation only needs to know its own local gradient. The chain rule handles
the rest.


## 5.4 Axiom's Autograd Engine

Axiom implements reverse-mode automatic differentiation in `autograd.c`. Every
differentiable operation:

1. Computes the forward result
2. Creates a `grad_fn` that knows how to compute the backward pass
3. Saves any tensors needed for the backward computation
4. Attaches the `grad_fn` to the output tensor

The grad_fn structure:

```c
typedef struct ax_grad_fn {
    ax_backward_fn_t backward;           // the backward function
    ax_tensor_t *inputs[AX_GRAD_MAX_INPUTS];   // input tensors
    int n_inputs;
    ax_tensor_t *saved[AX_GRAD_MAX_SAVED];     // saved for backward
    int n_saved;
    double scalar_ctx;                   // extra scalar context
    void *ctx;                           // extra pointer context
} ax_grad_fn_t;
```

This creates a directed acyclic graph (DAG) where each tensor points to the
operation that created it, and that operation points to its inputs.


## 5.5 Topological Sort and Backward Walk

`ax_backward()` traverses this graph in reverse topological order:

1. **Build the order**: iterative DFS from the loss tensor, collect nodes in
   post-order (children before parents).
2. **Seed**: set loss.grad = 1.0 (dL/dL = 1).
3. **Walk backwards**: for each node in reverse post-order, call its backward
   function with the accumulated gradient.

The DFS is iterative (explicit stack), not recursive, to avoid stack overflow on
deep computation graphs:

```c
#define MAX_GRAPH_NODES 4096

typedef struct {
    ax_tensor_t *node;
    int child_idx;
} dfs_frame_t;

static void topo_sort_dfs(ax_tensor_t *t, topo_list_t *visited, topo_list_t *order) {
    dfs_frame_t stack[MAX_GRAPH_NODES];
    int stack_top = 0;
    stack[stack_top++] = (dfs_frame_t){ .node = t, .child_idx = 0 };

    while (stack_top > 0) {
        // ... iterative DFS, append to order on exit
    }
}
```


## 5.6 Gradient Accumulation

Gradients are **accumulated**, not replaced. If a tensor is used in multiple
operations, each backward call adds to its gradient. This correctly handles
cases where a variable appears multiple times in an expression.

Before each training step, gradients must be zeroed:

```c
ax_optimizer_zero_grad(opt);  // zeros all param gradients
```


## 5.7 The Training Loop

A single training step in Axiom (`model.c`):

```c
float ax_model_train_step(ax_model_t *model, ax_tensor_t *input, ax_tensor_t *target) {
    // 1. Zero gradients from previous step
    ax_optimizer_zero_grad(model->opt);

    // 2. Forward pass
    ax_tensor_t *pred = ax_layer_forward(model->net, input);

    // 3. Compute loss
    ax_tensor_t *loss = model->loss_fn(pred, target);
    float loss_val = ax_tensor_get_f32(loss, (int64_t[]){0});

    // 4. Backward pass (compute all gradients)
    ax_backward(loss);

    // 5. Update weights
    ax_optimizer_step(model->opt);

    // 6. Clean up computation graph
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);

    return loss_val;
}
```


## 5.8 Graph Cleanup

After backward, the computation graph is no longer needed. `ax_graph_cleanup()`
destroys intermediate tensors (those with a `grad_fn`) but not leaf tensors
(parameters, user-created tensors):

```c
void ax_graph_cleanup(ax_tensor_t *root) {
    // Collect all nodes via DFS
    // Destroy non-leaf intermediates (node != root && node->grad_fn != NULL)
    // Detach root from graph
}
```

This is critical for memory management. Without cleanup, each training step would
leak all intermediate tensors.


## 5.9 Grad Enable/Disable

During inference (prediction), we don't need gradients. Disabling them saves memory
(no intermediate tensors saved) and computation:

```c
ax_no_grad();      // disable gradient tracking
// ... inference code ...
ax_enable_grad();  // re-enable
```

Axiom's `ax_model_predict()` does this automatically:

```c
ax_tensor_t *ax_model_predict(ax_model_t *model, ax_tensor_t *input) {
    ax_layer_eval(model->net);
    ax_no_grad();
    ax_tensor_t *out = ax_layer_forward(model->net, input);
    ax_enable_grad();
    return out;
}
```


## 5.10 Concrete Example: Backprop Through MSE + Dense

Forward:
```
z = x @ W + b          (dense layer)
loss = mean((z - y)^2)  (MSE loss)
```

Backward:
```
d_loss/d_loss = 1.0                          (seed)
d_loss/d_sq = 1/n * ones                     (mean backward)
d_loss/d_diff = 2 * diff * d_loss/d_sq       (square backward)
d_loss/d_z = d_loss/d_diff * 1               (sub backward, pred side)
d_loss/d_W = x^T @ d_loss/d_z                (matmul backward, weight)
d_loss/d_b = sum(d_loss/d_z, axis=0)         (add backward, bias)
d_loss/d_x = d_loss/d_z @ W^T               (matmul backward, input)
```

Each step multiplies the upstream gradient by the local derivative. The chain
rule handles the composition automatically.


## Key Takeaways

1. Gradient descent: move weights in the direction opposite to the gradient.
2. Backpropagation is the chain rule applied to a computation graph.
3. Axiom uses iterative DFS (not recursion) to traverse the graph safely.
4. Gradients accumulate — always zero them before each training step.
5. Graph cleanup after backward prevents memory leaks from intermediate tensors.
6. Disable gradients during inference for efficiency.
