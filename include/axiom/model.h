/* axiom/model.h — high-level model container.
   wraps a layer (usually sequential) with an optimizer and loss function
   to provide a simple train/predict interface.

   designed to be usable on embedded: no heap-allocated arrays for params,
   uses a fixed-size buffer. bump AX_MODEL_MAX_PARAMS if you need more. */

#ifndef AX_MODEL_H
#define AX_MODEL_H

#include "layer.h"
#include "optim.h"
#include "losses.h"

/* max trainable parameters across all layers */
#define AX_MODEL_MAX_PARAMS 256

/* loss function type (function pointer matching our loss api) */
typedef ax_tensor_t *(*ax_loss_fn_t)(ax_tensor_t *pred, ax_tensor_t *target);

/* the model: bundles a network, optimizer, and loss together */
typedef struct
{
    ax_layer_t *net;                          /* the network (usually sequential) */
    ax_optimizer_t *opt;                      /* optimizer (created by user, owned by model) */
    ax_loss_fn_t loss_fn;                     /* loss function */

    ax_tensor_t *params[AX_MODEL_MAX_PARAMS]; /* collected parameter pointers */
    int n_params;
} ax_model_t;

/* create a model from a network layer.
   collects all trainable params from the layer tree. */
ax_model_t *ax_model_create(ax_layer_t *net);

/* compile: attach optimizer and loss function.
   call this before training. the model takes ownership of the optimizer. */
void ax_model_compile(ax_model_t *model, ax_optimizer_t *opt, ax_loss_fn_t loss_fn);

/* single training step: forward, loss, backward, optimizer step.
   returns the loss value as a float. */
float ax_model_train_step(ax_model_t *model, ax_tensor_t *input, ax_tensor_t *target);

/* forward pass only (inference). switches to eval mode, runs forward,
   switches back. does not track gradients. */
ax_tensor_t *ax_model_predict(ax_model_t *model, ax_tensor_t *input);

/* train for multiple epochs. prints loss every print_every epochs.
   simple but gets the job done for small datasets that fit in memory. */
void ax_model_fit(ax_model_t *model,
                  ax_tensor_t *train_x, ax_tensor_t *train_y,
                  int epochs, int print_every);

/* print model summary */
void ax_model_summary(ax_model_t *model);

/* destroy model (frees optimizer, net, everything) */
void ax_model_destroy(ax_model_t *model);

#endif /* AX_MODEL_H */
