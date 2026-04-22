/* lr_scheduler.c — learning rate scheduling implementations. */

#include "axiom/lr_scheduler.h"
#include "axiom/error.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static ax_lr_scheduler_t *sched_alloc(ax_sched_type_t type, ax_optimizer_t *opt)
{
    if (!opt) {
        ax_err_set(AX_ERR_NULL_ARG, "ax_sched_*: optimizer is NULL");
        return NULL;
    }
    ax_lr_scheduler_t *s = calloc(1, sizeof(ax_lr_scheduler_t));
    AX_RETURN_NULL_IF_ALLOC_FAIL(s, "ax_sched_*");
    s->type = type;
    s->opt = opt;
    s->initial_lr = opt->lr;
    s->current_step = 0;
    return s;
}

ax_lr_scheduler_t *ax_sched_step_decay(ax_optimizer_t *opt,
                                        int step_size, float gamma)
{
    ax_lr_scheduler_t *s = sched_alloc(AX_SCHED_STEP_DECAY, opt);
    if (!s) return NULL;
    s->step_size = step_size > 0 ? step_size : 10;
    s->gamma = gamma > 0 ? gamma : 0.1f;
    return s;
}

ax_lr_scheduler_t *ax_sched_exponential(ax_optimizer_t *opt, float gamma)
{
    ax_lr_scheduler_t *s = sched_alloc(AX_SCHED_EXPONENTIAL, opt);
    if (!s) return NULL;
    s->gamma = gamma > 0 ? gamma : 0.95f;
    return s;
}

ax_lr_scheduler_t *ax_sched_cosine(ax_optimizer_t *opt,
                                    int total_steps, float min_lr)
{
    ax_lr_scheduler_t *s = sched_alloc(AX_SCHED_COSINE, opt);
    if (!s) return NULL;
    s->total_steps = total_steps > 0 ? total_steps : 100;
    s->min_lr = min_lr >= 0 ? min_lr : 0.0f;
    return s;
}

ax_lr_scheduler_t *ax_sched_warmup_cosine(ax_optimizer_t *opt,
                                           int warmup_steps, int total_steps,
                                           float min_lr)
{
    ax_lr_scheduler_t *s = sched_alloc(AX_SCHED_WARMUP_COSINE, opt);
    if (!s) return NULL;
    s->warmup_steps = warmup_steps > 0 ? warmup_steps : 10;
    s->total_steps = total_steps > 0 ? total_steps : 100;
    s->min_lr = min_lr >= 0 ? min_lr : 0.0f;
    return s;
}

void ax_sched_step(ax_lr_scheduler_t *sched)
{
    if (!sched || !sched->opt) return;

    sched->current_step++;
    float new_lr;

    switch (sched->type)
    {
        case AX_SCHED_STEP_DECAY:
            /* drop lr by gamma every step_size epochs */
            new_lr = sched->initial_lr *
                powf(sched->gamma, (float)(sched->current_step / sched->step_size));
            break;

        case AX_SCHED_EXPONENTIAL:
            new_lr = sched->initial_lr *
                powf(sched->gamma, (float)sched->current_step);
            break;

        case AX_SCHED_COSINE: {
            /* clamp step to total_steps */
            int step = sched->current_step;
            if (step > sched->total_steps) step = sched->total_steps;
            float progress = (float)step / (float)sched->total_steps;
            new_lr = sched->min_lr +
                0.5f * (sched->initial_lr - sched->min_lr) *
                (1.0f + cosf((float)M_PI * progress));
            break;
        }

        case AX_SCHED_WARMUP_COSINE: {
            int step = sched->current_step;
            if (step <= sched->warmup_steps)
            {
                /* linear warmup: ramp from 0 to initial_lr */
                new_lr = sched->initial_lr *
                    ((float)step / (float)sched->warmup_steps);
            }
            else
            {
                /* cosine decay from initial_lr to min_lr */
                int decay_step = step - sched->warmup_steps;
                int decay_total = sched->total_steps - sched->warmup_steps;
                if (decay_total <= 0) decay_total = 1;
                if (decay_step > decay_total) decay_step = decay_total;
                float progress = (float)decay_step / (float)decay_total;
                new_lr = sched->min_lr +
                    0.5f * (sched->initial_lr - sched->min_lr) *
                    (1.0f + cosf((float)M_PI * progress));
            }
            break;
        }

        default:
            return;
    }

    ax_optimizer_set_lr(sched->opt, new_lr);
}

float ax_sched_get_lr(ax_lr_scheduler_t *sched)
{
    if (!sched || !sched->opt) return 0.0f;
    return ax_optimizer_get_lr(sched->opt);
}

void ax_sched_destroy(ax_lr_scheduler_t *sched)
{
    free(sched);
}
