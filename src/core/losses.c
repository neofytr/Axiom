/* losses.c — loss function implementations.
   most are composed from existing differentiable ops
   so autograd handles the backward pass automatically.

   vectorized with the simd abstraction in compute/backends/simd_defs.h.
   forward paths detect contiguity and fall back to scalar when needed.
   backward paths run on saved tensors that are guaranteed contiguous
   (ax_ensure_contiguous is called at save-time), so they always vectorize. */

#include "axiom/losses.h"
#include "axiom/ops.h"
#include "axiom/autograd.h"
#include "axiom/activations.h"
#include "axiom/error.h"
#include "../compute/backends/simd_defs.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>


/* mse: mean((pred - target)^2)
   d/dpred = 2*(pred - target) / n */

static void mse_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];
    int64_t n = ax_tensor_numel(pred);
    float g = ((float *)grad_out->storage->data)[grad_out->offset];
    float scale = 2.0f * g / (float)n;

    if (!pred->requires_grad) return;
    if (!pred->grad)
        pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!pred->grad) return;

    /* saved tensors are contiguous with offset 0; grad is freshly zeroed */
    float *pg = (float *)pred->grad->storage->data + pred->grad->offset;
    float *pd = (float *)pred->storage->data + pred->offset;
    float *td = (float *)target->storage->data + target->offset;

    int64_t i = 0;
    ax_vf32 vs = ax_vf32_set1(scale);
    int64_t ve = n - (n % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 vp = ax_vf32_loadu(pd + i);
        ax_vf32 vt = ax_vf32_loadu(td + i);
        ax_vf32 vg = ax_vf32_loadu(pg + i);
        ax_vf32 vd = ax_vf32_sub(vp, vt);
        /* pg += diff * scale  =>  fmadd(diff, scale, pg) */
        ax_vf32_storeu(pg + i, ax_vf32_fmadd(vd, vs, vg));
    }
    for (; i < n; i++) {
        float diff = pd[i] - td[i];
        pg[i] += diff * scale;
    }
}

ax_tensor_t *ax_mse_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    if (!pred || !target)
    {
        ax_err_set(AX_ERR_NULL_ARG, "mse_loss: NULL tensor");
        return NULL;
    }
    if (pred->dtype != AX_FLOAT32 || target->dtype != AX_FLOAT32)
    {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "mse_loss: only float32 supported");
        return NULL;
    }
    if (ax_tensor_numel(pred) != ax_tensor_numel(target))
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "mse_loss: pred and target element count differ");
        return NULL;
    }

    int64_t n = ax_tensor_numel(pred);
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    double total = 0.0;
    bool pc = ax_tensor_is_contiguous(pred);
    bool tc = ax_tensor_is_contiguous(target);

    if (pc && tc) {
        /* 4-way simd accumulators to hide fp latency */
        float *pdp = pd + pred->offset;
        float *tdp = td + target->offset;
        ax_vf32 a0 = ax_vf32_zero(), a1 = ax_vf32_zero();
        ax_vf32 a2 = ax_vf32_zero(), a3 = ax_vf32_zero();
        int64_t i = 0;
        int64_t step = AX_VF32_WIDTH * 4;
        int64_t ve = n - (n % step);
        for (; i < ve; i += step) {
            ax_vf32 d0 = ax_vf32_sub(ax_vf32_loadu(pdp + i),
                                     ax_vf32_loadu(tdp + i));
            ax_vf32 d1 = ax_vf32_sub(ax_vf32_loadu(pdp + i + AX_VF32_WIDTH),
                                     ax_vf32_loadu(tdp + i + AX_VF32_WIDTH));
            ax_vf32 d2 = ax_vf32_sub(ax_vf32_loadu(pdp + i + 2*AX_VF32_WIDTH),
                                     ax_vf32_loadu(tdp + i + 2*AX_VF32_WIDTH));
            ax_vf32 d3 = ax_vf32_sub(ax_vf32_loadu(pdp + i + 3*AX_VF32_WIDTH),
                                     ax_vf32_loadu(tdp + i + 3*AX_VF32_WIDTH));
            a0 = ax_vf32_fmadd(d0, d0, a0);
            a1 = ax_vf32_fmadd(d1, d1, a1);
            a2 = ax_vf32_fmadd(d2, d2, a2);
            a3 = ax_vf32_fmadd(d3, d3, a3);
        }
        /* tail single-lane simd */
        int64_t ve2 = n - (n % AX_VF32_WIDTH);
        for (; i < ve2; i += AX_VF32_WIDTH) {
            ax_vf32 d = ax_vf32_sub(ax_vf32_loadu(pdp + i),
                                    ax_vf32_loadu(tdp + i));
            a0 = ax_vf32_fmadd(d, d, a0);
        }
        ax_vf32 asum = ax_vf32_add(ax_vf32_add(a0, a1), ax_vf32_add(a2, a3));
        total = (double)ax_vf32_hsum(asum);
        for (; i < n; i++) {
            double d = (double)pdp[i] - (double)tdp[i];
            total += d * d;
        }
    } else {
        /* non-contiguous fallback via strided indexing; only 1d/2d common */
        for (int64_t i = 0; i < n; i++)
        {
            double d = (double)pd[pred->offset + i] - (double)td[target->offset + i];
            total += d * d;
        }
    }

    ax_tensor_t *loss = ax_tensor_scalar((float)(total / (double)n));

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(mse_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        ax_tensor_t *pred_safe = ax_ensure_contiguous(pred);
        ax_tensor_t *target_safe = ax_ensure_contiguous(target);
        gf->saved[0] = pred_safe;
        gf->saved_owned[0] = (pred_safe != pred);
        gf->saved[1] = target_safe;
        gf->saved_owned[1] = (target_safe != target);
        gf->n_saved = 2;
        loss->grad_fn = gf;
    }
    return loss;
}


/* mae: mean(|pred - target|)
   abs isn't differentiable at 0 but we handle it manually here. */

static void mae_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];
    int64_t n = ax_tensor_numel(pred);
    float scale = 1.0f / (float)n;

    float g = ((float *)grad_out->storage->data)[grad_out->offset];

    if (pred->requires_grad)
    {
        if (!pred->grad)
            pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);

        float *pg = (float *)pred->grad->storage->data + pred->grad->offset;
        float *pd = (float *)pred->storage->data + pred->offset;
        float *td = (float *)target->storage->data + target->offset;

        float coef = g * scale;
        ax_vf32 vcoef = ax_vf32_set1(coef);
        ax_vf32 vzero = ax_vf32_zero();
        int64_t i = 0;
        int64_t ve = n - (n % AX_VF32_WIDTH);
        for (; i < ve; i += AX_VF32_WIDTH) {
            ax_vf32 vd = ax_vf32_sub(ax_vf32_loadu(pd + i),
                                     ax_vf32_loadu(td + i));
            /* sign = (d>0 ? 1 : 0) - (d<0 ? 1 : 0) */
            ax_vf32 pos = ax_vf32_cmpgt(vd, vzero);
            ax_vf32 neg = ax_vf32_cmpgt(vzero, vd);
            ax_vf32 sgn = ax_vf32_sub(pos, neg);
            ax_vf32 vg = ax_vf32_loadu(pg + i);
            ax_vf32_storeu(pg + i, ax_vf32_fmadd(sgn, vcoef, vg));
        }
        for (; i < n; i++) {
            float d = pd[i] - td[i];
            float sign = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
            pg[i] += coef * sign;
        }
    }
}

ax_tensor_t *ax_mae_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    if (!pred || !target)
    {
        ax_err_set(AX_ERR_NULL_ARG, "mae_loss: NULL tensor");
        return NULL;
    }
    if (pred->dtype != AX_FLOAT32 || target->dtype != AX_FLOAT32)
    {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "mae_loss: only float32 supported");
        return NULL;
    }
    if (ax_tensor_numel(pred) != ax_tensor_numel(target))
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "mae_loss: pred and target element count differ");
        return NULL;
    }

    ax_tensor_t *diff = ax_sub(pred, target);
    if (!diff) return NULL;

    ax_tensor_t *absd = ax_abs(diff);
    ax_tensor_destroy(diff);
    if (!absd) return NULL;

    /* we can't just use ax_mean here because abs doesn't have autograd.
       so we compute the mean manually and attach a custom backward. */
    int64_t n = ax_tensor_numel(absd);
    double total = 0.0;
    float *ad = (float *)absd->storage->data;

    /* absd from ax_abs is a fresh tensor so contiguous with offset 0,
       but stay defensive */
    if (ax_tensor_is_contiguous(absd)) {
        float *adp = ad + absd->offset;
        ax_vf32 a0 = ax_vf32_zero(), a1 = ax_vf32_zero();
        ax_vf32 a2 = ax_vf32_zero(), a3 = ax_vf32_zero();
        int64_t i = 0;
        int64_t step = AX_VF32_WIDTH * 4;
        int64_t ve = n - (n % step);
        for (; i < ve; i += step) {
            a0 = ax_vf32_add(a0, ax_vf32_loadu(adp + i));
            a1 = ax_vf32_add(a1, ax_vf32_loadu(adp + i + AX_VF32_WIDTH));
            a2 = ax_vf32_add(a2, ax_vf32_loadu(adp + i + 2*AX_VF32_WIDTH));
            a3 = ax_vf32_add(a3, ax_vf32_loadu(adp + i + 3*AX_VF32_WIDTH));
        }
        int64_t ve2 = n - (n % AX_VF32_WIDTH);
        for (; i < ve2; i += AX_VF32_WIDTH) {
            a0 = ax_vf32_add(a0, ax_vf32_loadu(adp + i));
        }
        ax_vf32 asum = ax_vf32_add(ax_vf32_add(a0, a1), ax_vf32_add(a2, a3));
        total = (double)ax_vf32_hsum(asum);
        for (; i < n; i++) total += (double)adp[i];
    } else {
        for (int64_t i = 0; i < n; i++)
            total += (double)ad[absd->offset + i];
    }

    ax_tensor_t *loss = ax_tensor_scalar((float)(total / (double)n));
    ax_tensor_destroy(absd);

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(mae_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        ax_tensor_t *pred_safe = ax_ensure_contiguous(pred);
        ax_tensor_t *target_safe = ax_ensure_contiguous(target);
        gf->saved[0] = pred_safe;
        gf->saved_owned[0] = (pred_safe != pred);
        gf->saved[1] = target_safe;
        gf->saved_owned[1] = (target_safe != target);
        gf->n_saved = 2;
        loss->grad_fn = gf;
    }
    return loss;
}


/* cross-entropy loss with logits.
   does log-softmax internally for numerical stability.

   log_softmax(x)_i = x_i - log(sum(exp(x_j)))
                     = x_i - max(x) - log(sum(exp(x_j - max(x))))

   loss = -sum(target * log_softmax(pred)) / batch_size */

static void ce_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];
    ax_tensor_t *softmax_out = self->saved[2]; /* we precomputed this */

    if (!pred->requires_grad) return;
    if (!pred->grad)
        pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);

    float g = ((float *)grad_out->storage->data)[grad_out->offset];
    int64_t batch = pred->shape[0];
    int64_t classes = pred->shape[1];
    float scale = g / (float)batch;

    float *pg = (float *)pred->grad->storage->data;
    float *sd = (float *)softmax_out->storage->data;
    float *td = (float *)target->storage->data;

    /* gradient of cross-entropy with softmax is simply: softmax(pred) - target
       this is one of the cleanest results in all of deep learning.
       note: pred->grad and softmax_out have their own (contiguous) strides;
       target may also have non-default strides. if every stride[1]==1
       we simd the inner loop, otherwise fall back to scalar strided. */
    bool inner_unit =
        pred->grad->strides[1] == 1 &&
        softmax_out->strides[1] == 1 &&
        target->strides[1] == 1;

    if (inner_unit) {
        ax_vf32 vs = ax_vf32_set1(scale);
        for (int64_t b = 0; b < batch; b++) {
            float *pgb = pg + pred->grad->offset + b * pred->grad->strides[0];
            float *sdb = sd + softmax_out->offset + b * softmax_out->strides[0];
            float *tdb = td + target->offset + b * target->strides[0];
            int64_t c = 0;
            int64_t ve = classes - (classes % AX_VF32_WIDTH);
            for (; c < ve; c += AX_VF32_WIDTH) {
                ax_vf32 vsm = ax_vf32_loadu(sdb + c);
                ax_vf32 vtg = ax_vf32_loadu(tdb + c);
                ax_vf32 vg = ax_vf32_loadu(pgb + c);
                /* pg += (sm - tg) * scale */
                ax_vf32 vdiff = ax_vf32_sub(vsm, vtg);
                ax_vf32_storeu(pgb + c, ax_vf32_fmadd(vdiff, vs, vg));
            }
            for (; c < classes; c++)
                pgb[c] += (sdb[c] - tdb[c]) * scale;
        }
    } else {
        for (int64_t b = 0; b < batch; b++)
        {
            for (int64_t c = 0; c < classes; c++)
            {
                int64_t grad_idx = b * pred->grad->strides[0] + c * pred->grad->strides[1];
                int64_t sm_idx = b * softmax_out->strides[0] + c * softmax_out->strides[1];
                int64_t tgt_idx = b * target->strides[0] + c * target->strides[1];
                pg[pred->grad->offset + grad_idx] +=
                    (sd[softmax_out->offset + sm_idx] - td[target->offset + tgt_idx]) * scale;
            }
        }
    }
}

/* gpu backward for cross-entropy: grad_input += (softmax - target) * scale.
   all ops go through dispatch so they run on the active device. */
static void ce_backward_device(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];
    ax_tensor_t *softmax_out = self->saved[2];
    if (!pred->requires_grad) return;
    if (!pred->grad)
        pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!pred->grad) return;

    int64_t i0[] = {0};
    float g = ax_tensor_get_f32(grad_out, i0);
    float scale = g / (float)pred->shape[0];

    /* diff = softmax - target */
    ax_tensor_t *diff = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!diff) return;
    ax_compute_sub(softmax_out, target, diff);
    /* pred->grad += scale * diff */
    ax_compute_axpy(diff, scale, pred->grad);
    ax_tensor_destroy(diff);
}

/* gpu forward for cross-entropy: composes from existing dispatch ops.
   avoids all raw pointer access. the cpu path below stays unchanged. */
static ax_tensor_t *cross_entropy_loss_device(ax_tensor_t *pred, ax_tensor_t *target)
{
    int64_t batch = pred->shape[0];

    /* softmax */
    ax_tensor_t *sm = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!sm) return NULL;
    if (ax_compute_softmax_rowwise(pred, sm) != AX_OK) {
        ax_tensor_destroy(sm);
        return NULL;
    }

    /* log(softmax) */
    ax_tensor_t *log_sm = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!log_sm) { ax_tensor_destroy(sm); return NULL; }
    ax_compute_log(sm, log_sm);

    /* element-wise target * log_sm */
    ax_tensor_t *prod = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!prod) { ax_tensor_destroy(sm); ax_tensor_destroy(log_sm); return NULL; }
    ax_compute_mul(target, log_sm, prod);
    ax_tensor_destroy(log_sm);

    /* sum all elements */
    int64_t one_shape[] = {1};
    ax_tensor_t *total = ax_tensor_zeros(one_shape, 1, AX_FLOAT32);
    if (!total) { ax_tensor_destroy(sm); ax_tensor_destroy(prod); return NULL; }
    ax_compute_sum(prod, -1, total);
    ax_tensor_destroy(prod);

    /* loss = -total / batch */
    ax_tensor_t *loss = ax_tensor_zeros(one_shape, 1, AX_FLOAT32);
    if (!loss) { ax_tensor_destroy(sm); ax_tensor_destroy(total); return NULL; }
    ax_compute_mul_scalar(total, -1.0 / (double)batch, loss);
    ax_tensor_destroy(total);

    if (ax_grad_enabled() && pred->requires_grad) {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(ce_backward_device);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        gf->saved[0] = pred;
        gf->saved_owned[0] = false;
        gf->saved[1] = target;
        gf->saved_owned[1] = false;
        gf->saved[2] = sm;
        gf->saved_owned[2] = true;
        gf->n_saved = 3;
        loss->grad_fn = gf;
    } else {
        ax_tensor_destroy(sm);
    }
    return loss;
}

ax_tensor_t *ax_cross_entropy_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    if (!pred || !target)
    {
        ax_err_set(AX_ERR_NULL_ARG, "cross_entropy_loss: NULL tensor");
        return NULL;
    }
    if (pred->dtype != AX_FLOAT32 || target->dtype != AX_FLOAT32)
    {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "cross_entropy_loss: only float32 supported");
        return NULL;
    }
    if (pred->ndim != 2 || target->ndim != 2)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "cross_entropy_loss: needs [batch, classes] tensors");
        return NULL;
    }
    if (pred->shape[0] != target->shape[0] || pred->shape[1] != target->shape[1])
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH,
                   "cross_entropy_loss: pred shape [%lld,%lld] != target shape [%lld,%lld]",
                   (long long)pred->shape[0], (long long)pred->shape[1],
                   (long long)target->shape[0], (long long)target->shape[1]);
        return NULL;
    }

    /* gpu fast path: compose from dispatch ops, no raw pointer access */
    if (pred->storage->device != AX_DEVICE_CPU) {
        return cross_entropy_loss_device(pred, target);
    }

    /* cpu path below: character-for-character identical to before */
    int64_t batch = pred->shape[0];
    int64_t classes = pred->shape[1];
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    /* compute log-softmax and cross entropy in one pass */
    double total_loss = 0.0;

    /* also compute softmax for the backward pass */
    ax_tensor_t *sm = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);
    if (!sm) return NULL;
    float *sd = (float *)sm->storage->data;

    /* simd inner loops only when inner stride is unit for all three tensors.
       sm is fresh from ax_tensor_zeros so its strides[1]==1 always. */
    bool inner_unit = pred->strides[1] == 1 && target->strides[1] == 1;

    for (int64_t b = 0; b < batch; b++)
    {
        int64_t p_row = pred->offset + b * pred->strides[0];
        int64_t s_row = sm->offset + b * sm->strides[0];
        int64_t t_row = target->offset + b * target->strides[0];

        /* find max for numerical stability */
        float mx = -FLT_MAX;
        if (inner_unit) {
            float *pdr = pd + p_row;
            ax_vf32 vmx = ax_vf32_set1(-FLT_MAX);
            int64_t c = 0;
            int64_t ve = classes - (classes % AX_VF32_WIDTH);
            for (; c < ve; c += AX_VF32_WIDTH)
                vmx = ax_vf32_max(vmx, ax_vf32_loadu(pdr + c));
            mx = ax_vf32_hmax(vmx);
            for (; c < classes; c++)
                if (pdr[c] > mx) mx = pdr[c];
        } else {
            for (int64_t c = 0; c < classes; c++) {
                float v = pd[p_row + c * pred->strides[1]];
                if (v > mx) mx = v;
            }
        }

        /* compute exp(x - mx), store in sm, accumulate sum.
           we use scalar expf here to preserve precision (the fast simd exp
           has ~1e-4 relative error which compounds through the log and
           per-sample accumulation). classes is typically small so the
           scalar exp loop isn't the bottleneck anyway. */
        double sum_exp = 0.0;
        if (inner_unit) {
            float *pdr = pd + p_row;
            float *sdr = sd + s_row;
            for (int64_t c = 0; c < classes; c++) {
                float e = expf(pdr[c] - mx);
                sdr[c] = e;
                sum_exp += (double)e;
            }
        } else {
            for (int64_t c = 0; c < classes; c++) {
                float e = expf(pd[p_row + c * pred->strides[1]] - mx);
                sd[s_row + c * sm->strides[1]] = e;
                sum_exp += (double)e;
            }
        }

        /* normalize to get softmax and accumulate loss.
           loss contribution is -target * log_softmax = -target * ((x - mx) - log_sum). */
        double log_sum = log(sum_exp);
        float inv_sum = (float)(1.0 / sum_exp);
        float bias = (float)mx + (float)log_sum;

        if (inner_unit) {
            float *pdr = pd + p_row;
            float *sdr = sd + s_row;
            float *tdr = td + t_row;

            /* simd normalize: sd[c] *= inv_sum */
            ax_vf32 vinv = ax_vf32_set1(inv_sum);
            int64_t c = 0;
            int64_t ve = classes - (classes % AX_VF32_WIDTH);
            for (; c < ve; c += AX_VF32_WIDTH)
                ax_vf32_storeu(sdr + c, ax_vf32_mul(ax_vf32_loadu(sdr + c), vinv));
            for (; c < classes; c++) sdr[c] *= inv_sum;

            /* accumulate -target * (pred - bias) as a dot product.
               use double for the per-batch accumulation to match original. */
            double row_loss = 0.0;
            for (c = 0; c < classes; c++) {
                double log_sm = (double)pdr[c] - (double)bias;
                row_loss -= (double)tdr[c] * log_sm;
            }
            total_loss += row_loss;
        } else {
            for (int64_t c = 0; c < classes; c++) {
                int64_t sm_idx = s_row + c * sm->strides[1];
                sd[sm_idx] = (float)((double)sd[sm_idx] / sum_exp);
                double log_sm = ((double)pd[p_row + c * pred->strides[1]] - (double)mx) - log_sum;
                total_loss -= (double)td[t_row + c * target->strides[1]] * log_sm;
            }
        }
    }

    ax_tensor_t *loss = ax_tensor_scalar((float)(total_loss / (double)batch));

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(ce_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        ax_tensor_t *pred_safe = ax_ensure_contiguous(pred);
        ax_tensor_t *target_safe = ax_ensure_contiguous(target);
        gf->saved[0] = pred_safe;
        gf->saved_owned[0] = (pred_safe != pred);
        gf->saved[1] = target_safe;
        gf->saved_owned[1] = (target_safe != target);
        gf->saved[2] = sm;      /* owned, helper we created */
        gf->saved_owned[2] = true;
        gf->n_saved = 3;
        loss->grad_fn = gf;
    }
    else
    {
        ax_tensor_destroy(sm);
    }

    return loss;
}


/* binary cross-entropy with logits.
   numerically stable: loss = max(x, 0) - x*t + log(1 + exp(-|x|)) */

static void bce_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *pred = self->saved[0];
    ax_tensor_t *target = self->saved[1];

    if (!pred->requires_grad) return;
    if (!pred->grad)
        pred->grad = ax_tensor_zeros(pred->shape, pred->ndim, pred->dtype);

    float g = ((float *)grad_out->storage->data)[grad_out->offset];
    int64_t n = ax_tensor_numel(pred);
    float scale = g / (float)n;

    float *pg = (float *)pred->grad->storage->data + pred->grad->offset;
    float *pd = (float *)pred->storage->data + pred->offset;
    float *td = (float *)target->storage->data + target->offset;

    /* d/dpred of bce = sigmoid(pred) - target.
       we keep scalar expf here to preserve bit-level fidelity with the
       original implementation (simd sigmoid has ~1e-4 relative error
       which is noticeable vs the test tolerances). */
    int64_t i = 0;
    for (; i < n; i++)
    {
        float x = pd[i];
        float sig = 1.0f / (1.0f + expf(-x));
        float t = td[i];
        pg[i] += (sig - t) * scale;
    }
}

ax_tensor_t *ax_bce_with_logits_loss(ax_tensor_t *pred, ax_tensor_t *target)
{
    if (!pred || !target)
    {
        ax_err_set(AX_ERR_NULL_ARG, "bce_with_logits_loss: NULL tensor");
        return NULL;
    }
    if (pred->dtype != AX_FLOAT32 || target->dtype != AX_FLOAT32)
    {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "bce_with_logits_loss: only float32 supported");
        return NULL;
    }
    if (ax_tensor_numel(pred) != ax_tensor_numel(target))
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "bce_with_logits_loss: pred and target element count differ");
        return NULL;
    }

    int64_t n = ax_tensor_numel(pred);
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    double total = 0.0;
    /* forward uses expf/logf which dominate; keep scalar for bit fidelity
       and full fp precision. the outer loop has no data dependencies so
       the compiler autovectorizes the arithmetic parts freely. */
    for (int64_t i = 0; i < n; i++)
    {
        float x = pd[pred->offset + i];
        float t = td[target->offset + i];
        /* max(x,0) - x*t + log(1 + exp(-|x|)) */
        float absx = x > 0.0f ? x : -x;
        total += (double)(x > 0.0f ? x : 0.0f) - (double)x * (double)t
               + (double)logf(1.0f + expf(-absx));
    }

    ax_tensor_t *loss = ax_tensor_scalar((float)(total / (double)n));

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(bce_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        ax_tensor_t *pred_safe = ax_ensure_contiguous(pred);
        ax_tensor_t *target_safe = ax_ensure_contiguous(target);
        gf->saved[0] = pred_safe;
        gf->saved_owned[0] = (pred_safe != pred);
        gf->saved[1] = target_safe;
        gf->saved_owned[1] = (target_safe != target);
        gf->n_saved = 2;
        loss->grad_fn = gf;
    }
    return loss;
}


/* huber loss: smooth transition from quadratic to linear at delta.
   grad is diff clipped to [-delta, delta] (the clean closed form). */

static void huber_backward(ax_grad_fn_t *self, ax_tensor_t *grad_out)
{
    ax_tensor_t *p = self->saved[0];
    ax_tensor_t *t = self->saved[1];
    float dl = (float)self->scalar_ctx;
    int64_t n = ax_tensor_numel(p);
    float g = ((float *)grad_out->storage->data)[grad_out->offset];
    float sc = g / (float)n;

    if (!p->requires_grad) return;
    if (!p->grad)
        p->grad = ax_tensor_zeros(p->shape, p->ndim, p->dtype);

    float *pg = (float *)p->grad->storage->data + p->grad->offset;
    float *pd = (float *)p->storage->data + p->offset;
    float *td = (float *)t->storage->data + t->offset;

    ax_vf32 vdl_pos = ax_vf32_set1(dl);
    ax_vf32 vdl_neg = ax_vf32_set1(-dl);
    ax_vf32 vsc = ax_vf32_set1(sc);

    int64_t i = 0;
    int64_t ve = n - (n % AX_VF32_WIDTH);
    for (; i < ve; i += AX_VF32_WIDTH) {
        ax_vf32 vp = ax_vf32_loadu(pd + i);
        ax_vf32 vt = ax_vf32_loadu(td + i);
        ax_vf32 vd = ax_vf32_sub(vp, vt);
        /* deriv = clamp(diff, -dl, dl) */
        ax_vf32 vder = ax_vf32_max(ax_vf32_min(vd, vdl_pos), vdl_neg);
        ax_vf32 vg = ax_vf32_loadu(pg + i);
        ax_vf32_storeu(pg + i, ax_vf32_fmadd(vder, vsc, vg));
    }
    for (; i < n; i++) {
        float diff = pd[i] - td[i];
        float deriv;
        float adiff = diff > 0.0f ? diff : -diff;
        if (adiff <= dl)
            deriv = diff;
        else
            deriv = dl * (diff > 0.0f ? 1.0f : -1.0f);
        pg[i] += deriv * sc;
    }
}

ax_tensor_t *ax_huber_loss(ax_tensor_t *pred, ax_tensor_t *target, float delta)
{
    if (!pred || !target)
    {
        ax_err_set(AX_ERR_NULL_ARG, "huber_loss: NULL tensor");
        return NULL;
    }
    if (pred->dtype != AX_FLOAT32 || target->dtype != AX_FLOAT32)
    {
        ax_err_set(AX_ERR_DTYPE_MISMATCH, "huber_loss: only float32 supported");
        return NULL;
    }
    if (ax_tensor_numel(pred) != ax_tensor_numel(target))
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH, "huber_loss: pred and target element count differ");
        return NULL;
    }

    int64_t n = ax_tensor_numel(pred);
    float *pd = (float *)pred->storage->data;
    float *td = (float *)target->storage->data;

    double total = 0.0;
    bool pc = ax_tensor_is_contiguous(pred);
    bool tc = ax_tensor_is_contiguous(target);

    if (pc && tc) {
        float *pdp = pd + pred->offset;
        float *tdp = td + target->offset;

        /* per-element: if |d|<=delta -> 0.5*d^2, else delta*(|d| - 0.5*delta).
           closed form: let c = clamp(|d|, 0, delta).
             contribution = 0.5*c*c + delta*(|d| - c)
           verify:
             |d|<=delta: c=|d| -> 0.5*d^2 + 0 = 0.5*d^2. ok
             |d|>delta:  c=delta -> 0.5*delta^2 + delta*(|d|-delta) = delta*(|d|-0.5*delta). ok */
        ax_vf32 vdelta = ax_vf32_set1(delta);
        ax_vf32 vzero = ax_vf32_zero();
        ax_vf32 vhalf = ax_vf32_set1(0.5f);

        ax_vf32 a0 = ax_vf32_zero(), a1 = ax_vf32_zero();
        int64_t i = 0;
        int64_t step = AX_VF32_WIDTH * 2;
        int64_t ve = n - (n % step);
        for (; i < ve; i += step) {
            ax_vf32 d0 = ax_vf32_sub(ax_vf32_loadu(pdp + i),
                                     ax_vf32_loadu(tdp + i));
            ax_vf32 d1 = ax_vf32_sub(ax_vf32_loadu(pdp + i + AX_VF32_WIDTH),
                                     ax_vf32_loadu(tdp + i + AX_VF32_WIDTH));
            ax_vf32 ad0 = ax_vf32_abs(d0);
            ax_vf32 ad1 = ax_vf32_abs(d1);
            ax_vf32 c0 = ax_vf32_min(ad0, vdelta);
            ax_vf32 c1 = ax_vf32_min(ad1, vdelta);
            /* 0.5*c*c */
            ax_vf32 q0 = ax_vf32_mul(ax_vf32_mul(vhalf, c0), c0);
            ax_vf32 q1 = ax_vf32_mul(ax_vf32_mul(vhalf, c1), c1);
            /* + delta*(|d| - c) */
            ax_vf32 lin0 = ax_vf32_mul(vdelta, ax_vf32_sub(ad0, c0));
            ax_vf32 lin1 = ax_vf32_mul(vdelta, ax_vf32_sub(ad1, c1));
            a0 = ax_vf32_add(a0, ax_vf32_add(q0, lin0));
            a1 = ax_vf32_add(a1, ax_vf32_add(q1, lin1));
        }
        int64_t ve2 = n - (n % AX_VF32_WIDTH);
        for (; i < ve2; i += AX_VF32_WIDTH) {
            ax_vf32 d = ax_vf32_sub(ax_vf32_loadu(pdp + i),
                                    ax_vf32_loadu(tdp + i));
            ax_vf32 ad = ax_vf32_abs(d);
            ax_vf32 c = ax_vf32_min(ad, vdelta);
            ax_vf32 q = ax_vf32_mul(ax_vf32_mul(vhalf, c), c);
            ax_vf32 lin = ax_vf32_mul(vdelta, ax_vf32_sub(ad, c));
            a0 = ax_vf32_add(a0, ax_vf32_add(q, lin));
        }
        (void)vzero;
        ax_vf32 asum = ax_vf32_add(a0, a1);
        total = (double)ax_vf32_hsum(asum);
        for (; i < n; i++) {
            float d = pdp[i] - tdp[i];
            float ad = d > 0.0f ? d : -d;
            if (ad <= delta)
                total += 0.5 * (double)d * (double)d;
            else
                total += (double)delta * ((double)ad - 0.5 * (double)delta);
        }
    } else {
        for (int64_t i = 0; i < n; i++)
        {
            float d = pd[pred->offset + i] - td[target->offset + i];
            float ad = d > 0.0f ? d : -d;
            if (ad <= delta)
                total += 0.5 * (double)d * (double)d;
            else
                total += (double)delta * ((double)ad - 0.5 * (double)delta);
        }
    }

    ax_tensor_t *loss = ax_tensor_scalar((float)(total / (double)n));

    if (ax_grad_enabled() && pred->requires_grad)
    {
        loss->requires_grad = true;
        ax_grad_fn_t *gf = ax_grad_fn_create(huber_backward);
        gf->inputs[0] = pred;
        gf->n_inputs = 1;
        ax_tensor_t *pred_safe = ax_ensure_contiguous(pred);
        ax_tensor_t *target_safe = ax_ensure_contiguous(target);
        gf->saved[0] = pred_safe;
        gf->saved_owned[0] = (pred_safe != pred);
        gf->saved[1] = target_safe;
        gf->saved_owned[1] = (target_safe != target);
        gf->n_saved = 2;
        gf->scalar_ctx = (double)delta;
        loss->grad_fn = gf;
    }
    return loss;
}
