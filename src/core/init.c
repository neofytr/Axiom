/* init.c — weight initialization implementations.

   the core idea: if weights are too large, activations explode.
   too small, they vanish. the "right" scale depends on the
   number of inputs (fan_in) and outputs (fan_out) of each layer,
   and which activation function you're using.

   xavier/glorot analyzed this for linear/sigmoid/tanh.
   he/kaiming analyzed it for relu.
   the math is in INTERNALS.md.

   device note: every initializer writes random values into the
   tensor. for cpu tensors we write straight into storage->data.
   for non-cpu tensors (cuda, etc.) we fill a small host scratch
   buffer and then dispatch through the owning backend's memcpy_h2d
   hook. init is a one-shot operation so the extra transfer cost
   is irrelevant. */

#include "axiom/init.h"
#include "axiom/tensor.h"
#include "axiom/compute.h"
#include "axiom/device.h"
#include "axiom/internal/backend_ops.h"
#include "axiom/internal/compute_internal.h"
#include "axiom/rng.h"
#include "axiom/error.h"
#include <stdlib.h>
#include <math.h>

/* acquire a writable host-side float buffer sized to the tensor's
   numel. for cpu tensors this is just `t->storage->data + t->offset`;
   for non-cpu tensors this mallocs a scratch buffer that must be
   released via init_host_commit.

   *is_scratch_out is set true when the caller must free the pointer
   after committing. */
static float *init_host_acquire(ax_tensor_t *t, int64_t n, bool *is_scratch_out) {
    if (t->storage->device == AX_DEVICE_CPU) {
        *is_scratch_out = false;
        return (float *)t->storage->data + t->offset;
    }
    *is_scratch_out = true;
    if (n <= 0 || (size_t)n > SIZE_MAX / sizeof(float)) return NULL;
    return (float *)malloc((size_t)n * sizeof(float));
}

/* commit (transfer h2d) and release a scratch buffer returned by
   init_host_acquire. no-op for cpu tensors. */
static void init_host_commit(ax_tensor_t *t, float *buf, int64_t n, bool is_scratch) {
    if (!is_scratch) return;
    const ax_backend_ops_t *ops = ax_backend_for_device(t->storage->device);
    if (ops && ops->memcpy_h2d) {
        void *dst = (char *)t->storage->data + (size_t)t->offset * sizeof(float);
        ops->memcpy_h2d(dst, buf, (size_t)n * sizeof(float));
    }
    free(buf);
}


void ax_init_xavier_uniform(ax_tensor_t *t, int64_t fan_in, int64_t fan_out)
{
    if (!t || !t->storage) return;
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    int64_t n = ax_tensor_numel(t);
    bool scratch;
    float *d = init_host_acquire(t, n, &scratch);
    if (!d) return;
    for (int64_t i = 0; i < n; i++) d[i] = ax_rng_uniform(-limit, limit);
    init_host_commit(t, d, n, scratch);
}

void ax_init_xavier_normal(ax_tensor_t *t, int64_t fan_in, int64_t fan_out)
{
    if (!t || !t->storage) return;
    float std = sqrtf(2.0f / (float)(fan_in + fan_out));
    int64_t n = ax_tensor_numel(t);
    bool scratch;
    float *d = init_host_acquire(t, n, &scratch);
    if (!d) return;
    for (int64_t i = 0; i < n; i++) d[i] = ax_rng_normal_params(0.0f, std);
    init_host_commit(t, d, n, scratch);
}

void ax_init_kaiming_uniform(ax_tensor_t *t, int64_t fan_in)
{
    if (!t || !t->storage) return;
    float limit = sqrtf(6.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    bool scratch;
    float *d = init_host_acquire(t, n, &scratch);
    if (!d) return;
    for (int64_t i = 0; i < n; i++) d[i] = ax_rng_uniform(-limit, limit);
    init_host_commit(t, d, n, scratch);
}

void ax_init_kaiming_normal(ax_tensor_t *t, int64_t fan_in)
{
    if (!t || !t->storage) return;
    float std = sqrtf(2.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    bool scratch;
    float *d = init_host_acquire(t, n, &scratch);
    if (!d) return;
    for (int64_t i = 0; i < n; i++) d[i] = ax_rng_normal_params(0.0f, std);
    init_host_commit(t, d, n, scratch);
}

void ax_init_lecun_normal(ax_tensor_t *t, int64_t fan_in)
{
    if (!t || !t->storage) return;
    float std = sqrtf(1.0f / (float)fan_in);
    int64_t n = ax_tensor_numel(t);
    bool scratch;
    float *d = init_host_acquire(t, n, &scratch);
    if (!d) return;
    for (int64_t i = 0; i < n; i++) d[i] = ax_rng_normal_params(0.0f, std);
    init_host_commit(t, d, n, scratch);
}

void ax_init_uniform(ax_tensor_t *t, float low, float high)
{
    if (!t || !t->storage) return;
    int64_t n = ax_tensor_numel(t);
    bool scratch;
    float *d = init_host_acquire(t, n, &scratch);
    if (!d) return;
    for (int64_t i = 0; i < n; i++) d[i] = ax_rng_uniform(low, high);
    init_host_commit(t, d, n, scratch);
}

void ax_init_normal(ax_tensor_t *t, float mean, float std)
{
    if (!t || !t->storage) return;
    int64_t n = ax_tensor_numel(t);
    bool scratch;
    float *d = init_host_acquire(t, n, &scratch);
    if (!d) return;
    for (int64_t i = 0; i < n; i++) d[i] = ax_rng_normal_params(mean, std);
    init_host_commit(t, d, n, scratch);
}

/* zeros/ones/constant route through the owning backend's fill hook
   for non-cpu tensors, and through ax_compute_fill (active backend)
   for cpu tensors. this makes init_zeros on a cuda tensor safe even
   if the caller hasn't set the active backend to cuda. */
static void init_fill(ax_tensor_t *t, double value) {
    if (!t || !t->storage) return;
    if (t->storage->device == AX_DEVICE_CPU) {
        ax_compute_fill(t, value);
        return;
    }
    const ax_backend_ops_t *ops = ax_backend_for_device(t->storage->device);
    if (ops && ops->fill) ops->fill(t, value);
}

void ax_init_zeros(ax_tensor_t *t)    { init_fill(t, 0.0); }
void ax_init_ones(ax_tensor_t *t)     { init_fill(t, 1.0); }
void ax_init_constant(ax_tensor_t *t, float value) { init_fill(t, (double)value); }
