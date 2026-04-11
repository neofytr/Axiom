/* axiom/layer.h — layer abstraction for composing models.

   layers are the building blocks. each one has a forward function,
   knows its parameters, and can switch between train/eval mode.

   the design is intentionally minimal for embedded targets:
   no heap-allocated strings, no dynamic arrays where avoidable,
   just function pointers and flat structs. */

#ifndef AX_LAYER_H
#define AX_LAYER_H

#include "tensor.h"

/* max parameters a single layer can have (weights + bias = 2 for dense) */
#define AX_LAYER_MAX_PARAMS 4

/* max non-trainable buffers a layer can have (e.g. batchnorm running stats) */
#define AX_LAYER_MAX_BUFFERS 4

/* max layers in a sequential model */
#define AX_SEQ_MAX_LAYERS 64

/* layer types — every layer MUST have its own value here.
   adding a new layer type requires touching exactly three places:
     1. this enum (add the new value before AX_LAYER_TYPE_COUNT)
     2. layer_type_name() in layer.c (add a case returning the display name)
     3. serialize.c: ax_model_save() layer header write + ax_model_load()
        layer reconstruction (add cases for the new type in both)
   missing any one of these will cause serialization to silently skip or
   misidentify the new layer type. */
typedef enum
{
    AX_LAYER_DENSE = 0,
    AX_LAYER_RELU,
    AX_LAYER_SIGMOID,
    AX_LAYER_TANH,
    AX_LAYER_LEAKY_RELU,
    AX_LAYER_ELU,
    AX_LAYER_GELU,
    AX_LAYER_SWISH,
    AX_LAYER_SOFTMAX,
    AX_LAYER_CONV2D,
    AX_LAYER_MAXPOOL2D,
    AX_LAYER_AVGPOOL2D,
    AX_LAYER_GLOBAL_AVGPOOL2D,
    AX_LAYER_FLATTEN,
    AX_LAYER_BATCHNORM,
    AX_LAYER_LAYERNORM,
    AX_LAYER_DROPOUT,
    AX_LAYER_SEQUENTIAL,
    AX_LAYER_CONV_BN_RELU,   /* fused conv2d+batchnorm+relu (see conv.h) */
    AX_LAYER_TYPE_COUNT,
} ax_layer_type_t;

/* forward declaration */
typedef struct ax_layer ax_layer_t;

/* layer vtable: every layer type implements these */
typedef struct
{
    ax_tensor_t *(*forward)(ax_layer_t *self, ax_tensor_t *input);
    void (*destroy)(ax_layer_t *self);
} ax_layer_ops_t;

/* base layer struct. concrete layers embed this as the first field
   so you can cast between them (classic c polymorphism). */
struct ax_layer
{
    ax_layer_ops_t ops;
    ax_layer_type_t type;
    bool training;             /* train vs eval mode (affects dropout, batchnorm) */

    /* parameter tracking — flat array, no heap allocation */
    ax_tensor_t *params[AX_LAYER_MAX_PARAMS];
    int n_params;

    /* non-trainable buffer tracking (e.g. batchnorm running stats) */
    ax_tensor_t *buffers[AX_LAYER_MAX_BUFFERS];
    int n_buffers;

    /* shape info for model summary */
    int64_t input_features;
    int64_t output_features;
};


/* dense (fully connected) layer: out = input @ weight + bias
   the workhorse of neural networks. */

typedef struct
{
    ax_layer_t base;
    ax_tensor_t *weight;   /* [input_features, output_features] */
    ax_tensor_t *bias;     /* [output_features] or NULL if no bias */
    bool use_bias;
} ax_dense_t;

/* create a dense layer. initializes weights with kaiming by default. */
ax_layer_t *ax_dense_create(int64_t in_features, int64_t out_features, bool use_bias);


/* activation layers — the struct is public so serialization can access alpha/axis */

typedef struct
{
    ax_layer_t base;
    float alpha;    /* for leaky relu, elu */
    int axis;       /* for softmax */
} ax_activation_layer_t;


/* activation layers — stateless, no parameters.
   you could just call ax_relu() directly, but wrapping them as layers
   lets you put them in a sequential model. */

ax_layer_t *ax_relu_layer_create(void);
ax_layer_t *ax_sigmoid_layer_create(void);
ax_layer_t *ax_tanh_layer_create(void);
ax_layer_t *ax_leaky_relu_layer_create(float alpha);
ax_layer_t *ax_elu_layer_create(float alpha);
ax_layer_t *ax_gelu_layer_create(void);
ax_layer_t *ax_swish_layer_create(void);
ax_layer_t *ax_softmax_layer_create(int axis);


/* sequential model: chains layers together.
   forward pass runs input through each layer in order. */

typedef struct
{
    ax_layer_t base;
    ax_layer_t *layers[AX_SEQ_MAX_LAYERS];
    int n_layers;
} ax_sequential_t;

/* create an empty sequential model */
ax_layer_t *ax_sequential_create(void);

/* add a layer to the sequential model. returns the model for chaining. */
ax_layer_t *ax_sequential_add(ax_layer_t *seq, ax_layer_t *layer);


/* common layer operations */

/* run forward pass through the layer */
ax_tensor_t *ax_layer_forward(ax_layer_t *layer, ax_tensor_t *input);

/* get all trainable parameters from a layer (and sublayers for sequential).
   fills params array, returns count. */
int ax_layer_get_params(ax_layer_t *layer, ax_tensor_t **params, int max_params);

/* get all non-trainable buffers from a layer (and sublayers for sequential).
   fills buffers array, returns count. */
int ax_layer_get_buffers(ax_layer_t *layer, ax_tensor_t **buffers, int max_buffers);

/* set train/eval mode (recursive for sequential) */
void ax_layer_train(ax_layer_t *layer);
void ax_layer_eval(ax_layer_t *layer);

/* print model summary: layer types, shapes, param counts */
void ax_layer_summary(ax_layer_t *layer);

/* count total trainable parameters */
int64_t ax_layer_param_count(ax_layer_t *layer);

/* free a layer and all its owned tensors */
void ax_layer_destroy(ax_layer_t *layer);

#endif /* AX_LAYER_H */
