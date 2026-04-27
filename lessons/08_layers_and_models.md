# Unit 8: The Layer System and Model API

## Why This Matters

Individual neurons and operations are the atoms. Layers and models are the molecules
and organisms. The layer abstraction lets you compose networks declaratively — stack
layers, attach an optimizer and loss, and train. This unit covers how Axiom organizes
this from the layer interface up to the model API.


## 8.1 The Layer Abstraction

In Axiom, a layer is anything that:
1. Has a forward function (takes a tensor, returns a tensor)
2. Can be destroyed (frees its resources)
3. Optionally has trainable parameters

The base layer struct (`layer.h`):

```c
struct ax_layer {
    ax_layer_ops_t ops;                        // vtable: forward, destroy
    ax_layer_type_t type;                      // Dense, ReLU, Sequential, etc.
    bool training;                             // train vs eval mode

    ax_tensor_t *params[AX_LAYER_MAX_PARAMS];  // parameter pointers (flat array)
    int n_params;                              // how many params

    int64_t input_features;                    // for model summary
    int64_t output_features;
};
```

Key design decisions for embedded:
- **No heap-allocated arrays**: params is a fixed-size array (`AX_LAYER_MAX_PARAMS = 4`).
- **No strings**: layer type is an enum, not a string name.
- **C polymorphism**: concrete layers embed `ax_layer_t` as their first field,
  allowing pointer casting between the base and derived types.


## 8.2 C Polymorphism Pattern

This is the classic C "inheritance" pattern:

```c
// Base layer
struct ax_layer { ax_layer_ops_t ops; ... };

// Dense layer "inherits" by embedding ax_layer_t as first field
typedef struct {
    ax_layer_t base;           // MUST be first
    ax_tensor_t *weight;
    ax_tensor_t *bias;
    bool use_bias;
} ax_dense_t;

// Cast between types:
ax_dense_t *d = (ax_dense_t *)layer;     // downcast
ax_layer_t *l = (ax_layer_t *)d;          // upcast (or just &d->base)
```

The vtable pattern (`ax_layer_ops_t`) provides virtual dispatch:

```c
typedef struct {
    ax_tensor_t *(*forward)(ax_layer_t *self, ax_tensor_t *input);
    void (*destroy)(ax_layer_t *self);
} ax_layer_ops_t;

// Call forward on any layer type:
ax_tensor_t *ax_layer_forward(ax_layer_t *layer, ax_tensor_t *input) {
    return layer->ops.forward(layer, input);
}
```


## 8.3 The Dense Layer

The fundamental building block. Computes `out = input @ weight + bias`:

```c
ax_layer_t *ax_dense_create(int64_t in_features, int64_t out_features, bool use_bias);
```

On creation:
1. Allocates the dense struct
2. Creates weight tensor `[in_features, out_features]`
3. Initializes with Kaiming uniform
4. Sets `weight->requires_grad = true`
5. Creates bias `[out_features]` initialized to zeros (if use_bias)
6. Registers params in the flat array

Forward:
```c
static ax_tensor_t *dense_forward(ax_layer_t *self, ax_tensor_t *input) {
    ax_dense_t *d = (ax_dense_t *)self;
    ax_tensor_t *out = ax_matmul(input, d->weight);    // X @ W
    if (d->use_bias && d->bias) {
        ax_tensor_t *biased = ax_add(out, d->bias);    // + b
        if (!ax_grad_enabled() && !out->grad_fn)
            ax_tensor_destroy(out);  // free intermediate in inference mode
        return biased;
    }
    return out;
}
```

Note the memory management: in inference mode, intermediates are freed immediately.
In training mode, they're kept alive for backpropagation.


## 8.4 Activation Layers

Stateless wrappers around activation functions:

```c
ax_layer_t *ax_relu_layer_create(void);
ax_layer_t *ax_sigmoid_layer_create(void);
ax_layer_t *ax_tanh_layer_create(void);
ax_layer_t *ax_gelu_layer_create(void);
ax_layer_t *ax_swish_layer_create(void);
ax_layer_t *ax_leaky_relu_layer_create(float alpha);
ax_layer_t *ax_elu_layer_create(float alpha);
ax_layer_t *ax_softmax_layer_create(int axis);
```

These have no parameters and no trainable state. They exist so you can put them
in a sequential model alongside dense layers.


## 8.5 The Sequential Container

Chains layers together. Forward pass runs input through each layer in order:

```c
ax_layer_t *seq = ax_sequential_create();
ax_sequential_add(seq, ax_dense_create(784, 128, true));
ax_sequential_add(seq, ax_relu_layer_create());
ax_sequential_add(seq, ax_dense_create(128, 10, true));
```

The sequential struct:

```c
typedef struct {
    ax_layer_t base;
    ax_layer_t *layers[AX_SEQ_MAX_LAYERS];   // fixed-size, no realloc
    int n_layers;
} ax_sequential_t;
```

Forward pass:
```c
static ax_tensor_t *sequential_forward(ax_layer_t *self, ax_tensor_t *input) {
    ax_sequential_t *seq = (ax_sequential_t *)self;
    ax_tensor_t *x = input;
    ax_tensor_t *prev = NULL;

    for (int i = 0; i < seq->n_layers; i++) {
        ax_tensor_t *next = ax_layer_forward(seq->layers[i], x);
        // In inference mode, free previous intermediate
        if (prev && !ax_grad_enabled() && !prev->grad_fn)
            ax_tensor_destroy(prev);
        prev = next;
        x = next;
    }
    return x;
}
```


## 8.6 Parameter Collection

The model needs to know all trainable parameters across all layers (to pass them
to the optimizer). `ax_layer_get_params` recursively collects them:

```c
int ax_layer_get_params(ax_layer_t *layer, ax_tensor_t **params, int max_params) {
    if (layer->type == AX_LAYER_SEQUENTIAL) {
        // Recurse into sublayers
        ax_sequential_t *seq = (ax_sequential_t *)layer;
        int total = 0;
        for (int i = 0; i < seq->n_layers && total < max_params; i++)
            total += ax_layer_get_params(seq->layers[i], params + total, max_params - total);
        return total;
    }
    // Copy params from this layer's flat array
    int count = 0;
    for (int i = 0; i < layer->n_params && count < max_params; i++)
        params[count++] = layer->params[i];
    return count;
}
```


## 8.7 Train vs Eval Mode

Layers can be in training or evaluation mode. Currently this only affects memory
management (inference mode frees intermediates), but it's designed for future
layers that behave differently in train vs eval (dropout, batch normalization).

```c
void ax_layer_train(ax_layer_t *layer);  // recursive for sequential
void ax_layer_eval(ax_layer_t *layer);   // recursive for sequential
```


## 8.8 The Model API

The model bundles everything together:

```c
typedef struct {
    ax_layer_t *net;                          // the network
    ax_optimizer_t *opt;                      // optimizer
    ax_loss_fn_t loss_fn;                     // loss function

    ax_tensor_t *params[AX_MODEL_MAX_PARAMS]; // collected params (max 256)
    int n_params;
} ax_model_t;
```

Usage:

```c
// 1. Build the network
ax_layer_t *net = ax_sequential_create();
ax_sequential_add(net, ax_dense_create(2, 16, true));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_dense_create(16, 1, true));

// 2. Create the model (collects params automatically)
ax_model_t *model = ax_model_create(net);

// 3. Compile: attach optimizer and loss
ax_optimizer_t *opt = ax_adam_create(model->params, model->n_params,
                                     0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
ax_model_compile(model, opt, ax_mse_loss);

// 4. Train
ax_model_fit(model, train_x, train_y, 1000, 100);

// 5. Predict
ax_tensor_t *pred = ax_model_predict(model, test_x);

// 6. Clean up
ax_model_destroy(model);  // frees net, optimizer, everything
```


## 8.9 Model Summary

```c
ax_model_summary(model);
```

Prints:
```
Sequential model (3 layers, 65 params)
#    type           shape                params
0    Dense          (-1, 2) -> (-1, 16)     48
1    ReLU                                    0
2    Dense          (-1, 16) -> (-1, 1)     17
total params: 65
```


## 8.10 Memory Ownership

Clear ownership rules prevent leaks:
- The **model** owns the network (layer tree) and optimizer.
- The **sequential** layer owns its sublayers.
- Each **layer** owns its weight and bias tensors.
- The **optimizer** owns its per-parameter state (momentum buffers).
- `ax_model_destroy` cascades: destroys optimizer, then layer tree (which
  destroys sublayers, which destroy their weights/biases).


## Key Takeaways

1. Layers use C polymorphism: base struct + vtable + first-field casting.
2. Sequential chains layers; forward pass runs them in order.
3. Parameter collection is recursive (sequential -> sublayers -> params).
4. Train mode keeps intermediates for backprop; eval mode frees them.
5. The model bundles network + optimizer + loss for a clean training API.
6. Fixed-size arrays everywhere — no realloc needed, embedded-friendly.
