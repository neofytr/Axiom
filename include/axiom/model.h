/* axiom/model.h — high-level model container.

   wraps a layer (usually sequential) with an optimizer and loss
   function to provide a simple train/predict interface.

   designed to be usable on embedded: no heap-allocated arrays for
   params, uses a fixed-size buffer. bump AX_MODEL_MAX_PARAMS if you
   need more.

   ownership: ax_model_create takes ownership of the supplied network
   layer (do not destroy it separately). ax_model_compile takes
   ownership of the optimizer. ax_model_destroy releases the network,
   the optimizer, and any state it allocated.

   error handling: ax_model_create returns NULL on alloc failure or
   parameter overflow (more than AX_MODEL_MAX_PARAMS).
   ax_model_train_step returns NaN on backward / forward failure
   (ax_err_last_message describes); successful steps return the loss
   value as a float.

   thread-safety: not concurrent-safe on a single model. one thread per
   model instance for training; predict is also single-threaded but
   independent models may be evaluated in parallel. */

#ifndef AX_MODEL_H
#define AX_MODEL_H

#include "layer.h"
#ifndef AX_INFERENCE_ONLY
#include "optim.h"
#include "losses.h"
#else
/* inference-only: the full optim/losses headers aren't compiled into the
   library. provide a minimal forward declaration so the ax_model_t struct
   still typechecks — the opt/loss_fn fields are simply unused. */
typedef struct ax_optimizer ax_optimizer_t;
#endif

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

#ifndef AX_INFERENCE_ONLY
/* compile: attach optimizer and loss function.
   call this before training. the model takes ownership of the optimizer. */
void ax_model_compile(ax_model_t *model, ax_optimizer_t *opt, ax_loss_fn_t loss_fn);

/* single training step: forward, loss, backward, optimizer step.
   returns the loss value as a float. */
float ax_model_train_step(ax_model_t *model, ax_tensor_t *input, ax_tensor_t *target);
#endif

/* forward pass only (inference). switches to eval mode, runs forward,
   switches back. does not track gradients. */
ax_tensor_t *ax_model_predict(ax_model_t *model, ax_tensor_t *input);

#ifndef AX_INFERENCE_ONLY
/* train for multiple epochs. prints loss every print_every epochs.
   simple but gets the job done for small datasets that fit in memory. */
void ax_model_fit(ax_model_t *model,
                  ax_tensor_t *train_x, ax_tensor_t *train_y,
                  int epochs, int print_every);
#endif

/* print model summary */
void ax_model_summary(ax_model_t *model);

/* destroy model (frees optimizer, net, everything) */
void ax_model_destroy(ax_model_t *model);

#endif /* AX_MODEL_H */
