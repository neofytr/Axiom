/* layer.c — layer implementations.
   dense layer, activation wrappers, and sequential container. */

#include "axiom/layer.h"
#include "axiom/ops.h"
#include "axiom/activations.h"
#include "axiom/init.h"
#include "axiom/compute.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* layer type names for printing */
static const char *layer_type_name(ax_layer_type_t type)
{
    switch (type)
    {
        case AX_LAYER_DENSE:      return "Dense";
        case AX_LAYER_RELU:       return "ReLU";
        case AX_LAYER_SIGMOID:    return "Sigmoid";
        case AX_LAYER_TANH:       return "Tanh";
        case AX_LAYER_LEAKY_RELU: return "LeakyReLU";
        case AX_LAYER_ELU:        return "ELU";
        case AX_LAYER_GELU:       return "GELU";
        case AX_LAYER_SWISH:      return "Swish";
        case AX_LAYER_SOFTMAX:    return "Softmax";
        case AX_LAYER_SEQUENTIAL: return "Sequential";
        default: return "Unknown";
    }
}


/* dense layer */

static ax_tensor_t *dense_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_dense_t *d = (ax_dense_t *)self;

    /* out = input @ weight */
    ax_tensor_t *out = ax_matmul(input, d->weight);
    if (!out) return NULL;

    /* out = out + bias (if present) */
    if (d->use_bias && d->bias)
    {
        ax_tensor_t *biased = ax_add(out, d->bias);
        /* can't destroy out here if autograd is tracking it...
           same issue as in losses.c. leave it for now. */
        if (biased) return biased;
    }
    return out;
}

static void dense_destroy(ax_layer_t *self)
{
    ax_dense_t *d = (ax_dense_t *)self;
    if (d->weight) ax_tensor_destroy(d->weight);
    if (d->bias) ax_tensor_destroy(d->bias);
    free(d);
}

ax_layer_t *ax_dense_create(int64_t in_features, int64_t out_features, bool use_bias)
{
    ax_dense_t *d = calloc(1, sizeof(ax_dense_t));
    if (!d) return NULL;

    d->base.ops.forward = dense_forward;
    d->base.ops.destroy = dense_destroy;
    d->base.type = AX_LAYER_DENSE;
    d->base.training = true;
    d->base.input_features = in_features;
    d->base.output_features = out_features;
    d->use_bias = use_bias;

    /* allocate weight [in, out] and init with kaiming */
    int64_t w_shape[] = {in_features, out_features};
    d->weight = ax_tensor_create(w_shape, 2, AX_FLOAT32);
    if (!d->weight) { free(d); return NULL; }

    ax_init_kaiming_uniform(d->weight, in_features);
    d->weight->requires_grad = true;

    d->base.params[0] = d->weight;
    d->base.n_params = 1;

    /* bias initialized to zeros */
    if (use_bias)
    {
        int64_t b_shape[] = {out_features};
        d->bias = ax_tensor_zeros(b_shape, 1, AX_FLOAT32);
        if (!d->bias) { ax_tensor_destroy(d->weight); free(d); return NULL; }

        d->bias->requires_grad = true;
        d->base.params[1] = d->bias;
        d->base.n_params = 2;
    }

    return (ax_layer_t *)d;
}


/* activation layers — thin wrappers so they fit in a sequential model.
   they have no parameters and no state. */

typedef struct
{
    ax_layer_t base;
    float alpha;    /* for leaky relu, elu */
    int axis;       /* for softmax */
} ax_activation_layer_t;

static ax_tensor_t *relu_forward(ax_layer_t *self, ax_tensor_t *input) { return ax_relu(input); }
static ax_tensor_t *sigmoid_forward(ax_layer_t *self, ax_tensor_t *input) { return ax_sigmoid(input); }
static ax_tensor_t *tanh_forward(ax_layer_t *self, ax_tensor_t *input) { return ax_tanh_op(input); }
static ax_tensor_t *gelu_forward(ax_layer_t *self, ax_tensor_t *input) { return ax_gelu(input); }
static ax_tensor_t *swish_forward(ax_layer_t *self, ax_tensor_t *input) { return ax_swish(input); }

static ax_tensor_t *leaky_relu_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_activation_layer_t *a = (ax_activation_layer_t *)self;
    return ax_leaky_relu(input, a->alpha);
}

static ax_tensor_t *elu_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_activation_layer_t *a = (ax_activation_layer_t *)self;
    return ax_elu(input, a->alpha);
}

static ax_tensor_t *softmax_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_activation_layer_t *a = (ax_activation_layer_t *)self;
    return ax_softmax(input, a->axis);
}

static void activation_destroy(ax_layer_t *self) { free(self); }

/* factory for stateless activations */
static ax_layer_t *make_activation(ax_layer_type_t type,
                                    ax_tensor_t *(*fwd)(ax_layer_t *, ax_tensor_t *))
{
    ax_activation_layer_t *a = calloc(1, sizeof(ax_activation_layer_t));
    if (!a) return NULL;
    a->base.ops.forward = fwd;
    a->base.ops.destroy = activation_destroy;
    a->base.type = type;
    a->base.training = true;
    return (ax_layer_t *)a;
}

ax_layer_t *ax_relu_layer_create(void) { return make_activation(AX_LAYER_RELU, relu_forward); }
ax_layer_t *ax_sigmoid_layer_create(void) { return make_activation(AX_LAYER_SIGMOID, sigmoid_forward); }
ax_layer_t *ax_tanh_layer_create(void) { return make_activation(AX_LAYER_TANH, tanh_forward); }
ax_layer_t *ax_gelu_layer_create(void) { return make_activation(AX_LAYER_GELU, gelu_forward); }
ax_layer_t *ax_swish_layer_create(void) { return make_activation(AX_LAYER_SWISH, swish_forward); }

ax_layer_t *ax_leaky_relu_layer_create(float alpha)
{
    ax_activation_layer_t *a = (ax_activation_layer_t *)make_activation(AX_LAYER_LEAKY_RELU, leaky_relu_forward);
    if (a) a->alpha = alpha;
    return (ax_layer_t *)a;
}

ax_layer_t *ax_elu_layer_create(float alpha)
{
    ax_activation_layer_t *a = (ax_activation_layer_t *)make_activation(AX_LAYER_ELU, elu_forward);
    if (a) a->alpha = alpha;
    return (ax_layer_t *)a;
}

ax_layer_t *ax_softmax_layer_create(int axis)
{
    ax_activation_layer_t *a = (ax_activation_layer_t *)make_activation(AX_LAYER_SOFTMAX, softmax_forward);
    if (a) a->axis = axis;
    return (ax_layer_t *)a;
}


/* sequential model */

static ax_tensor_t *sequential_forward(ax_layer_t *self, ax_tensor_t *input)
{
    ax_sequential_t *seq = (ax_sequential_t *)self;
    ax_tensor_t *x = input;

    for (int i = 0; i < seq->n_layers; i++)
    {
        x = ax_layer_forward(seq->layers[i], x);
        if (!x) return NULL;
    }
    return x;
}

static void sequential_destroy(ax_layer_t *self)
{
    ax_sequential_t *seq = (ax_sequential_t *)self;
    for (int i = 0; i < seq->n_layers; i++)
    {
        ax_layer_destroy(seq->layers[i]);
    }
    free(seq);
}

ax_layer_t *ax_sequential_create(void)
{
    ax_sequential_t *seq = calloc(1, sizeof(ax_sequential_t));
    if (!seq) return NULL;

    seq->base.ops.forward = sequential_forward;
    seq->base.ops.destroy = sequential_destroy;
    seq->base.type = AX_LAYER_SEQUENTIAL;
    seq->base.training = true;
    seq->n_layers = 0;
    return (ax_layer_t *)seq;
}

ax_layer_t *ax_sequential_add(ax_layer_t *seq_base, ax_layer_t *layer)
{
    ax_sequential_t *seq = (ax_sequential_t *)seq_base;
    if (seq->n_layers >= AX_SEQ_MAX_LAYERS)
    {
        /* could happen on a microcontroller with a deep model —
           bump AX_SEQ_MAX_LAYERS if needed */
        return NULL;
    }
    seq->layers[seq->n_layers++] = layer;
    return seq_base;
}


/* common operations */

ax_tensor_t *ax_layer_forward(ax_layer_t *layer, ax_tensor_t *input)
{
    if (!layer || !layer->ops.forward) return NULL;
    return layer->ops.forward(layer, input);
}

int ax_layer_get_params(ax_layer_t *layer, ax_tensor_t **params, int max_params)
{
    if (!layer) return 0;

    /* for sequential: recurse into sublayers */
    if (layer->type == AX_LAYER_SEQUENTIAL)
    {
        ax_sequential_t *seq = (ax_sequential_t *)layer;
        int total = 0;
        for (int i = 0; i < seq->n_layers && total < max_params; i++)
        {
            total += ax_layer_get_params(seq->layers[i], params + total, max_params - total);
        }
        return total;
    }

    /* for regular layers: copy params from the flat array */
    int count = 0;
    for (int i = 0; i < layer->n_params && count < max_params; i++)
    {
        params[count++] = layer->params[i];
    }
    return count;
}

void ax_layer_train(ax_layer_t *layer)
{
    if (!layer) return;
    layer->training = true;
    if (layer->type == AX_LAYER_SEQUENTIAL)
    {
        ax_sequential_t *seq = (ax_sequential_t *)layer;
        for (int i = 0; i < seq->n_layers; i++)
            ax_layer_train(seq->layers[i]);
    }
}

void ax_layer_eval(ax_layer_t *layer)
{
    if (!layer) return;
    layer->training = false;
    if (layer->type == AX_LAYER_SEQUENTIAL)
    {
        ax_sequential_t *seq = (ax_sequential_t *)layer;
        for (int i = 0; i < seq->n_layers; i++)
            ax_layer_eval(seq->layers[i]);
    }
}

int64_t ax_layer_param_count(ax_layer_t *layer)
{
    if (!layer) return 0;

    if (layer->type == AX_LAYER_SEQUENTIAL)
    {
        ax_sequential_t *seq = (ax_sequential_t *)layer;
        int64_t total = 0;
        for (int i = 0; i < seq->n_layers; i++)
            total += ax_layer_param_count(seq->layers[i]);
        return total;
    }

    int64_t total = 0;
    for (int i = 0; i < layer->n_params; i++)
        total += ax_tensor_numel(layer->params[i]);
    return total;
}

void ax_layer_summary(ax_layer_t *layer)
{
    if (!layer) return;

    if (layer->type == AX_LAYER_SEQUENTIAL)
    {
        ax_sequential_t *seq = (ax_sequential_t *)layer;
        printf("Sequential model (%d layers, %ld params)\n",
               seq->n_layers, ax_layer_param_count(layer));
        printf("%-4s %-14s %-20s %s\n", "#", "type", "shape", "params");

        for (int i = 0; i < seq->n_layers; i++)
        {
            ax_layer_t *l = seq->layers[i];
            int64_t pc = ax_layer_param_count(l);

            if (l->type == AX_LAYER_DENSE)
            {
                printf("%-4d %-14s (%ld, %ld) -> (%ld, %ld)     %ld\n",
                       i, layer_type_name(l->type),
                       (long)-1, l->input_features,
                       (long)-1, l->output_features,
                       pc);
            }
            else
            {
                printf("%-4d %-14s                         %ld\n",
                       i, layer_type_name(l->type), pc);
            }
        }

        printf("total params: %ld\n", ax_layer_param_count(layer));
        return;
    }

    /* single layer */
    printf("%s: %ld params\n", layer_type_name(layer->type),
           ax_layer_param_count(layer));
}

void ax_layer_destroy(ax_layer_t *layer)
{
    if (!layer) return;
    if (layer->ops.destroy)
        layer->ops.destroy(layer);
}
