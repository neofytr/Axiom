/* axiom/optim.h — optimizers for updating model parameters.
   each optimizer holds per-parameter state (momentum, adam moments, etc.)
   and implements the step() function to update weights using gradients. */

#ifndef AX_OPTIM_H
#define AX_OPTIM_H

#include "tensor.h"

/* optimizer types */
typedef enum
{
    AX_OPTIM_SGD = 0,
    AX_OPTIM_ADAM,
    AX_OPTIM_ADAMW,
    AX_OPTIM_RMSPROP,
    AX_OPTIM_ADAGRAD,
} ax_optim_type_t;

/* per-parameter state (adam moments, momentum buffer, etc.) */
typedef struct
{
    ax_tensor_t *m;     /* first moment (momentum / adam m) */
    ax_tensor_t *v;     /* second moment (adam v / rmsprop cache) */
    int64_t step_count; /* number of steps taken (for adam bias correction) */
} ax_param_state_t;

/* the optimizer struct */
typedef struct
{
    ax_optim_type_t type;

    /* parameters we're optimizing */
    ax_tensor_t **params;
    int n_params;

    /* per-parameter state (allocated lazily on first step) */
    ax_param_state_t *state;

    /* global step counter — incremented once per ax_optimizer_step() call.
       used for adam/adamw bias correction so all parameters share the same t. */
    int64_t global_step;

    /* hyperparameters */
    float lr;           /* learning rate */
    float beta1;        /* adam/adamw first moment decay (default 0.9) */
    float beta2;        /* adam/adamw second moment decay (default 0.999) */
    float eps;          /* adam/adamw/rmsprop epsilon (default 1e-8) */
    float weight_decay; /* l2 regularization / adamw decoupled decay */
    float momentum;     /* sgd momentum factor */
    bool nesterov;      /* sgd nesterov momentum */
    float rho;          /* rmsprop smoothing constant (default 0.99) */
} ax_optimizer_t;

/* create optimizers. params is an array of tensor pointers (the weights).
   the optimizer does NOT own the tensors, just references them. */

ax_optimizer_t *ax_sgd_create(ax_tensor_t **params, int n_params,
                              float lr, float momentum, float weight_decay,
                              bool nesterov);

ax_optimizer_t *ax_adam_create(ax_tensor_t **params, int n_params,
                               float lr, float beta1, float beta2,
                               float eps, float weight_decay);

ax_optimizer_t *ax_adamw_create(ax_tensor_t **params, int n_params,
                                float lr, float beta1, float beta2,
                                float eps, float weight_decay);

ax_optimizer_t *ax_rmsprop_create(ax_tensor_t **params, int n_params,
                                  float lr, float rho, float eps,
                                  float weight_decay);

ax_optimizer_t *ax_adagrad_create(ax_tensor_t **params, int n_params,
                                  float lr, float eps, float weight_decay);

/* update all parameters using their gradients */
void ax_optimizer_step(ax_optimizer_t *opt);

/* zero out gradients of all parameters */
void ax_optimizer_zero_grad(ax_optimizer_t *opt);

/* set learning rate (useful for manual scheduling) */
void ax_optimizer_set_lr(ax_optimizer_t *opt, float lr);

/* get current learning rate */
float ax_optimizer_get_lr(ax_optimizer_t *opt);

/* free optimizer and its internal state (does NOT free the param tensors) */
void ax_optimizer_destroy(ax_optimizer_t *opt);

#endif /* AX_OPTIM_H */
