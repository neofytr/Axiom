/* axiom/losses.h — loss functions for training.
   each returns a scalar tensor (1 element) suitable for ax_backward(). */

#ifndef AX_LOSSES_H
#define AX_LOSSES_H

#include "tensor.h"

/* mean squared error: (1/n) * sum((pred - target)^2)
   the classic regression loss. */
ax_tensor_t *ax_mse_loss(ax_tensor_t *pred, ax_tensor_t *target);

/* mean absolute error: (1/n) * sum(|pred - target|) */
ax_tensor_t *ax_mae_loss(ax_tensor_t *pred, ax_tensor_t *target);

/* cross-entropy loss for classification.
   pred should be raw logits (NOT softmax'd). we apply log-softmax internally
   for numerical stability. target is one-hot or class indices.

   for the one-hot version:
     loss = -sum(target * log_softmax(pred)) / batch_size

   pred: [batch, classes]  target: [batch, classes] (one-hot) */
ax_tensor_t *ax_cross_entropy_loss(ax_tensor_t *pred, ax_tensor_t *target);

/* binary cross-entropy with logits.
   pred is raw logits (before sigmoid). applies sigmoid internally.
   target is 0 or 1.

   loss = -mean(target * log(sigmoid(pred)) + (1-target) * log(1-sigmoid(pred)))

   numerically stable formulation:
   loss = mean(max(pred, 0) - pred*target + log(1 + exp(-|pred|))) */
ax_tensor_t *ax_bce_with_logits_loss(ax_tensor_t *pred, ax_tensor_t *target);

/* huber loss (smooth l1): quadratic for small errors, linear for large.
   delta controls the transition point.
   loss = 0.5 * x^2           if |x| <= delta
          delta * (|x| - 0.5 * delta)  otherwise */
ax_tensor_t *ax_huber_loss(ax_tensor_t *pred, ax_tensor_t *target, float delta);

#endif /* AX_LOSSES_H */
