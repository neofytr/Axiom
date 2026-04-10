# Unit 9: Code Map

## Why This Matters

This unit is your reference guide to the Axiom codebase. When you want to find
where something is implemented or understand how the pieces connect, start here.


## 9.1 Directory Structure

```
neoNN/
  include/axiom/       Public headers (the API)
    types.h             Fundamental types: dtype, status codes, device enum
    error.h             Error reporting: ax_err_set, ax_err_last_message
    tensor.h            Tensor struct, creation, shape ops, element access
    memory.h            Arena allocator, aligned alloc
    ops.h               Element-wise math ops: add, sub, mul, matmul, relu, etc.
    activations.h       Advanced activations: leaky_relu, elu, gelu, swish, softmax
    autograd.h          Backward pass engine: ax_backward, ax_grad_fn_t, grad check
    init.h              Weight initialization: xavier, kaiming, lecun
    layer.h             Layer abstraction: dense, activation wrappers, sequential
    model.h             Model container: create, compile, train_step, predict, fit
    optim.h             Optimizers: SGD, Adam, AdamW, RMSProp, Adagrad
    losses.h            Loss functions: MSE, MAE, cross-entropy, BCE, Huber
    broadcast.h         Broadcasting logic for element-wise ops
    compute.h           Compute dispatch (backend abstraction)
    backend_ops.h       Backend function pointer table
    serialize.h         Model save/load (binary format)
    data.h              Dataset, dataloader, transforms
    conv.h              Conv2d, pooling, flatten, im2col
    norm.h              Normalization layers (batch norm, layer norm)
    axiom.h             Meta-header that includes everything

  src/core/             Implementation files
    tensor.c            Storage management, tensor creation, reshape, transpose
    ops.c               Element-wise ops with autograd support
    activations.c       Activation functions with backward
    autograd.c          Backward engine: topological sort, gradient propagation
    init.c              Initialization algorithms (Box-Muller for normal)
    layer.c             Dense layer, activation layers, sequential model
    model.c             Model API: train_step, predict, fit, summary
    optim.c             Optimizer implementations
    losses.c            Loss function implementations
    broadcast.c         Shape broadcasting logic
    memory.c            Arena allocator, aligned malloc
    error.c             Thread-local error state
    serialize.c         Binary serialization for .axm and .axt files
    data.c              Dataset/dataloader/transform implementations
    conv.c              Convolution, im2col, pooling, flatten
    norm.c              Normalization layer implementations

  src/compute/          Backend dispatch
    dispatch.c          Selects CPU backend, calls function pointers
    backends/
      cpu_naive.c       Naive CPU implementations (reference)

  examples/
    xor.c              XOR problem — minimal working example

  tests/               Unit and integration tests
  INTERNALS.md         This course document
  SECURITY.md          Security policy and conventions
  CMakeLists.txt       Build configuration
```


## 9.2 Data Flow: Training

Here's how data flows through a single training step:

```
User calls: ax_model_train_step(model, input, target)

  1. ax_optimizer_zero_grad(opt)
     -> zeros all param.grad tensors

  2. ax_layer_forward(model->net, input)
     -> sequential_forward()
        -> dense_forward(): ax_matmul(input, W), ax_add(result, bias)
           -> each op creates a grad_fn and saves tensors for backward
        -> relu_forward(): ax_relu(z)
           -> creates grad_fn, saves input
        -> dense_forward(): ax_matmul(...), ax_add(...)
        -> returns final output (pred)

  3. loss_fn(pred, target)
     -> e.g. ax_mse_loss: sub -> square -> mean
        -> each op hooks into the graph
     -> returns scalar loss tensor

  4. ax_backward(loss)
     -> topo_sort_dfs: builds reverse topological order
     -> seeds loss.grad = 1.0
     -> walks backward: calls each grad_fn->backward(gf, node->grad)
        -> mean_backward, square_backward, sub_backward, ...
        -> matmul_backward: computes dL/dW, dL/db, dL/dx
        -> relu_backward: masks gradient where input < 0
     -> gradients now stored in each param.grad

  5. ax_optimizer_step(opt)
     -> for each param with a gradient:
        -> update moments (Adam) or velocity (SGD)
        -> update weight values

  6. ax_graph_cleanup(loss)
     -> destroys intermediate tensors (those with grad_fn)
     -> detaches loss from graph
  7. ax_tensor_destroy(loss)
```


## 9.3 Data Flow: Inference

```
User calls: ax_model_predict(model, input)

  1. ax_layer_eval(model->net)    // set eval mode
  2. ax_no_grad()                 // disable gradient tracking
  3. ax_layer_forward(model->net, input)
     -> sequential_forward()
        -> each layer runs forward, no grad_fn created
        -> intermediates freed immediately after next layer runs
  4. ax_enable_grad()             // re-enable
  5. return output
```


## 9.4 Module Dependency Graph

```
types.h (no deps)
  |
  error.h
  |
  tensor.h (-> types.h, error.h)
  |
  +-- memory.h
  +-- ops.h (-> tensor.h)
  +-- activations.h (-> tensor.h)
  +-- autograd.h (-> tensor.h)
  +-- broadcast.h (-> tensor.h)
  +-- init.h (-> tensor.h)
  |
  layer.h (-> tensor.h)
  |
  +-- optim.h (-> tensor.h)
  +-- losses.h (-> tensor.h, ops.h)
  |
  model.h (-> layer.h, optim.h, losses.h)
  |
  serialize.h (-> layer.h, model.h)
  data.h (-> tensor.h)
  conv.h (-> layer.h)
  norm.h (-> layer.h)
  |
  compute.h / backend_ops.h (-> tensor.h)
  |
  axiom.h (includes everything)
```


## 9.5 Key Constants

| Constant               | Value | Location    | Purpose                           |
|-----------------------|-------|-------------|-----------------------------------|
| `AX_MAX_DIMS`         | 8     | types.h     | Max tensor dimensions             |
| `AX_LAYER_MAX_PARAMS` | 4     | layer.h     | Max params per layer              |
| `AX_SEQ_MAX_LAYERS`   | 64    | layer.h     | Max layers in sequential          |
| `AX_MODEL_MAX_PARAMS` | 256   | model.h     | Max params across all layers      |
| `MAX_GRAPH_NODES`     | 4096  | autograd.c  | Max nodes in computation graph    |
| `AX_DEFAULT_ALIGNMENT`| 64    | memory.h    | Memory alignment (SIMD-ready)     |
| `AX_ARENA_DEFAULT_BLOCK_SIZE` | 1MB | memory.h | Arena block size            |
| `AX_MAGIC`            | 0x41584F4E | serialize.h | Model file magic ("AXON")   |
| `AX_FORMAT_VERSION`   | 1     | serialize.h | Serialization format version      |


## 9.6 Error Handling Pattern

Axiom does not use exceptions (C doesn't have them) or setjmp/longjmp. Instead:

1. Functions that can fail return either a status code (`ax_status_t`) or a
   pointer (NULL on failure).
2. Before returning an error, they call `ax_err_set()` with a human-readable message.
3. The caller can retrieve the message with `ax_err_last_message()`.

```c
ax_tensor_t *t = ax_tensor_create(shape, ndim, dtype);
if (!t) {
    fprintf(stderr, "Error: %s\n", ax_err_last_message());
    return;
}
```

No `exit()` calls in library code. The application decides how to handle errors.


## 9.7 Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

With sanitizers for development:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DAX_SANITIZE=ON
make -j$(nproc)
ctest --output-on-failure
```


## Key Takeaways

1. Headers in `include/axiom/`, implementations in `src/core/`.
2. Training flow: zero_grad -> forward -> loss -> backward -> step -> cleanup.
3. Inference flow: eval mode -> no_grad -> forward -> done.
4. Fixed-size arrays and no dynamic allocation in hot paths.
5. Error handling through return codes + `ax_err_set` message.
