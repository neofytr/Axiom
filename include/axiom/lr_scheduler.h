/* axiom/lr_scheduler.h — learning rate scheduling.

   learning rate is the most important hyperparameter. starting high
   and decaying it during training is almost always better than
   a constant lr. these schedulers modify the optimizer's lr based on
   the current epoch/step.

   ownership: scheduler borrows the optimizer (does NOT own it).
   ax_sched_destroy frees only the scheduler. release the optimizer
   separately with ax_optimizer_destroy.

   error handling: ax_sched_*_create returns NULL on alloc failure.
   ax_sched_step / ax_sched_get_lr are tolerant of NULL (no-op /
   return 0) but will warn via ax_err_set in debug builds.

   thread-safety: not concurrent-safe on the same scheduler. */

#ifndef AX_LR_SCHEDULER_H
#define AX_LR_SCHEDULER_H

#include "optim.h"

typedef struct ax_lr_scheduler ax_lr_scheduler_t;

/* scheduler types */
typedef enum
{
    AX_SCHED_STEP_DECAY,
    AX_SCHED_EXPONENTIAL,
    AX_SCHED_COSINE,
    AX_SCHED_WARMUP_COSINE,
} ax_sched_type_t;

struct ax_lr_scheduler
{
    ax_sched_type_t type;
    ax_optimizer_t *opt;
    float initial_lr;
    int current_step;

    /* type-specific params */
    int step_size;       /* for step decay: decay every N epochs */
    float gamma;         /* for step/exponential: multiply lr by gamma */
    int total_steps;     /* for cosine: total training steps */
    float min_lr;        /* for cosine: minimum lr */
    int warmup_steps;    /* for warmup_cosine: warmup phase length */
};

/* step decay: lr = initial_lr * gamma^(epoch / step_size)
   e.g., step_size=30, gamma=0.1 means lr drops by 10x every 30 epochs */
ax_lr_scheduler_t *ax_sched_step_decay(ax_optimizer_t *opt,
                                        int step_size, float gamma);

/* exponential decay: lr = initial_lr * gamma^epoch */
ax_lr_scheduler_t *ax_sched_exponential(ax_optimizer_t *opt, float gamma);

/* cosine annealing: lr oscillates between initial_lr and min_lr
   following a cosine curve over total_steps.
   lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * step / total)) */
ax_lr_scheduler_t *ax_sched_cosine(ax_optimizer_t *opt,
                                    int total_steps, float min_lr);

/* warmup + cosine: linear warmup from 0 to initial_lr over warmup_steps,
   then cosine decay to min_lr over the remaining steps.
   this is the standard schedule for transformer training. */
ax_lr_scheduler_t *ax_sched_warmup_cosine(ax_optimizer_t *opt,
                                           int warmup_steps, int total_steps,
                                           float min_lr);

/* call after each epoch (or step) to update the lr */
void ax_sched_step(ax_lr_scheduler_t *sched);

/* get current lr (for logging) */
float ax_sched_get_lr(ax_lr_scheduler_t *sched);

/* free the scheduler (does NOT free the optimizer) */
void ax_sched_destroy(ax_lr_scheduler_t *sched);

#endif /* AX_LR_SCHEDULER_H */
