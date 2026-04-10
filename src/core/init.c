/* init.c — weight initialization implementations.

   the core idea: if weights are too large, activations explode.
   too small, they vanish. the "right" scale depends on the
   number of inputs (fan_in) and outputs (fan_out) of each layer,
   and which activation function you're using.

   xavier/glorot analyzed this for linear/sigmoid/tanh.
   he/kaiming analyzed it for relu.
   the math is in INTERNALS.md. */

#include "axiom/init.h"
#include "axiom/tensor.h"
#include "axiom/compute.h"
#include "axiom/rng.h"
#include <stdlib.h>
#include <math.h>


void ax_init_xavier_uniform(ax_tensor_t *t, int64_t fan_in, int64_t fan_out)
{
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = ax_rng_uniform(-limit, limit);
}

void ax_init_xavier_normal(ax_tensor_t *t, int64_t fan_in, int64_t fan_out)
{
    float std = sqrtf(2.0f / (float)(fan_in + fan_out));
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = ax_rng_normal_params(0.0f, std);
}

void ax_init_kaiming_uniform(ax_tensor_t *t, int64_t fan_in)
{
    float limit = sqrtf(6.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = ax_rng_uniform(-limit, limit);
}

void ax_init_kaiming_normal(ax_tensor_t *t, int64_t fan_in)
{
    float std = sqrtf(2.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = ax_rng_normal_params(0.0f, std);
}

void ax_init_lecun_normal(ax_tensor_t *t, int64_t fan_in)
{
    float std = sqrtf(1.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = ax_rng_normal_params(0.0f, std);
}

void ax_init_uniform(ax_tensor_t *t, float low, float high)
{
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = ax_rng_uniform(low, high);
}

void ax_init_normal(ax_tensor_t *t, float mean, float std)
{
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = ax_rng_normal_params(mean, std);
}

void ax_init_zeros(ax_tensor_t *t)    { ax_compute_fill(t, 0.0); }
void ax_init_ones(ax_tensor_t *t)     { ax_compute_fill(t, 1.0); }
void ax_init_constant(ax_tensor_t *t, float value) { ax_compute_fill(t, (double)value); }
