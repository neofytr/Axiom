/* model.c — high-level model container and training loop. */

#include "axiom/model.h"
#include "axiom/autograd.h"
#include "axiom/compute.h"
#include <stdlib.h>
#include <stdio.h>

ax_model_t *ax_model_create(ax_layer_t *net)
{
    if (!net) return NULL;

    ax_model_t *m = calloc(1, sizeof(ax_model_t));
    if (!m) return NULL;

    m->net = net;
    m->n_params = ax_layer_get_params(net, m->params, AX_MODEL_MAX_PARAMS);
    return m;
}

#ifndef AX_INFERENCE_ONLY
void ax_model_compile(ax_model_t *model, ax_optimizer_t *opt, ax_loss_fn_t loss_fn)
{
    if (!model) return;
    model->opt = opt;
    model->loss_fn = loss_fn;

    /* point the optimizer at our collected params.
       this is a bit sneaky — we're replacing the optimizer's param array
       with ours. works because the optimizer just stores a pointer. */
    if (opt)
    {
        opt->params = model->params;
        opt->n_params = model->n_params;
    }
}

float ax_model_train_step(ax_model_t *model, ax_tensor_t *input, ax_tensor_t *target)
{
    if (!model || !model->opt || !model->loss_fn) return -1.0f;

    /* zero gradients from previous step */
    ax_optimizer_zero_grad(model->opt);

    /* forward */
    ax_layer_train(model->net);
    ax_enable_grad();
    ax_tensor_t *pred = ax_layer_forward(model->net, input);
    if (!pred) return -1.0f;

    /* compute loss */
    ax_tensor_t *loss = model->loss_fn(pred, target);
    if (!loss)
    {
        ax_graph_cleanup(pred);
        ax_tensor_destroy(pred);
        return -1.0f;
    }

    /* read loss value before backward potentially messes with things */
    int64_t i0[] = {0};
    float loss_val = ax_tensor_get_f32(loss, i0);

    /* backward */
    if (ax_backward(loss) != AX_OK)
    {
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss);
        return -1.0f;
    }

    /* update weights */
    ax_optimizer_step(model->opt);

    /* free the computation graph (intermediates from this step).
       parameters and their grads survive because they have no grad_fn.
       cleanup frees intermediates but not the root, so we destroy it after. */
    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);

    return loss_val;
}
#endif

ax_tensor_t *ax_model_predict(ax_model_t *model, ax_tensor_t *input)
{
    if (!model) return NULL;

    ax_layer_eval(model->net);
    ax_no_grad();

    ax_tensor_t *out = ax_layer_forward(model->net, input);

    ax_enable_grad();
    return out;
}

#ifndef AX_INFERENCE_ONLY
void ax_model_fit(ax_model_t *model,
                  ax_tensor_t *train_x, ax_tensor_t *train_y,
                  int epochs, int print_every)
{
    if (!model) return;

    for (int e = 0; e < epochs; e++)
    {
        float loss = ax_model_train_step(model, train_x, train_y);

        if (print_every > 0 && (e % print_every == 0 || e == epochs - 1))
        {
            printf("epoch %d/%d  loss: %.6f\n", e + 1, epochs, loss);
        }
    }
}
#endif

void ax_model_summary(ax_model_t *model)
{
    if (!model || !model->net) return;
    ax_layer_summary(model->net);
}

void ax_model_destroy(ax_model_t *model)
{
    if (!model) return;
#ifndef AX_INFERENCE_ONLY
    if (model->opt) ax_optimizer_destroy(model->opt);
#endif
    if (model->net) ax_layer_destroy(model->net);
    free(model);
}
