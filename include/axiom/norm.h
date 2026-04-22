/* axiom/norm.h — normalization, dropout, and gradient clipping.

   ownership/error-handling/thread-safety follow the conventions in
   axiom/axiom.h. all ax_*_create functions return a heap layer (NULL
   on alloc failure); release with ax_layer_destroy. layers own their
   parameter tensors (gamma, beta) and running buffers. */

#ifndef AX_NORM_H
#define AX_NORM_H

#include "layer.h"

/* batch normalization layer.
   normalizes each feature across the batch to zero mean, unit variance,
   then applies learnable scale (gamma) and shift (beta).
   maintains running stats for inference.
   input shape: [batch, features] */
typedef struct
{
    ax_layer_t base;
    int64_t num_features;
    float eps;
    float momentum;
    ax_tensor_t *gamma;
    ax_tensor_t *beta;
    ax_tensor_t *running_mean;
    ax_tensor_t *running_var;
} ax_batchnorm_t;

ax_layer_t *ax_batchnorm_create(int64_t num_features, float eps, float momentum);

/* layer normalization.
   normalizes across features for each sample independently.
   no running stats needed. input shape: [batch, features] */
typedef struct
{
    ax_layer_t base;
    int64_t num_features;
    float eps;
    ax_tensor_t *gamma;
    ax_tensor_t *beta;
} ax_layernorm_t;

ax_layer_t *ax_layernorm_create(int64_t num_features, float eps);

/* dropout layer.
   training: zeros elements with probability p, scales rest by 1/(1-p).
   eval: identity. */
typedef struct
{
    ax_layer_t base;
    float p;
} ax_dropout_t;

ax_layer_t *ax_dropout_create(float p);

/* gradient clipping */
void ax_clip_grad_value(ax_tensor_t **params, int n_params, float max_val);
float ax_clip_grad_norm(ax_tensor_t **params, int n_params, float max_norm);

#endif /* AX_NORM_H */
