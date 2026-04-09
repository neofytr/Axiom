/* init.c — weight initialization implementations.

   the core idea: if weights are too large, activations explode.
   too small, they vanish. the "right" scale depends on the
   number of inputs (fan_in) and outputs (fan_out) of each layer,
   and which activation function you're using.

   xavier/glorot analyzed this for linear/sigmoid/tanh.
   he/kaiming analyzed it for relu.
   the math is in INTERNALS.md. */

#include "axiom/init.h"
#include "axiom/compute.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

static bool seeded = false;

static void seed_once(void)
{
    if (!seeded)
    {
        srand((unsigned)time(NULL));
        seeded = true;
    }
}

/* generate uniform random float in [low, high) */
static float rand_uniform(float low, float high)
{
    return low + ((float)rand() / (float)RAND_MAX) * (high - low);
}

/* box-muller transform for normal distribution.
   generates two normal samples from two uniform samples. */
static float rand_normal(float mean, float std)
{
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / (float)RAND_MAX;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    return mean + std * z;
}


void ax_init_xavier_uniform(ax_tensor_t *t, int64_t fan_in, int64_t fan_out)
{
    seed_once();
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = rand_uniform(-limit, limit);
}

void ax_init_xavier_normal(ax_tensor_t *t, int64_t fan_in, int64_t fan_out)
{
    seed_once();
    float std = sqrtf(2.0f / (float)(fan_in + fan_out));
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = rand_normal(0.0f, std);
}

void ax_init_kaiming_uniform(ax_tensor_t *t, int64_t fan_in)
{
    seed_once();
    float limit = sqrtf(6.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = rand_uniform(-limit, limit);
}

void ax_init_kaiming_normal(ax_tensor_t *t, int64_t fan_in)
{
    seed_once();
    float std = sqrtf(2.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = rand_normal(0.0f, std);
}

void ax_init_lecun_normal(ax_tensor_t *t, int64_t fan_in)
{
    seed_once();
    float std = sqrtf(1.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = rand_normal(0.0f, std);
}

void ax_init_uniform(ax_tensor_t *t, float low, float high)
{
    seed_once();
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = rand_uniform(low, high);
}

void ax_init_normal(ax_tensor_t *t, float mean, float std)
{
    seed_once();
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++)
        d[t->offset + i] = rand_normal(mean, std);
}

void ax_init_zeros(ax_tensor_t *t)    { ax_compute_fill(t, 0.0); }
void ax_init_ones(ax_tensor_t *t)     { ax_compute_fill(t, 1.0); }
void ax_init_constant(ax_tensor_t *t, float value) { ax_compute_fill(t, (double)value); }
