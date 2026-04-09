/* autograd.c — the backward pass engine.
   does a reverse topological sort of the computation graph
   then calls each op's backward function in order. */

#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/error.h"
#include <stdlib.h>
#include <string.h>

/* global gradient tracking state */
static bool grad_tracking = true;

void ax_no_grad(void)    { grad_tracking = false; }
void ax_enable_grad(void){ grad_tracking = true; }
bool ax_grad_enabled(void){ return grad_tracking; }

ax_grad_fn_t *ax_grad_fn_create(ax_backward_fn_t fn)
{
    ax_grad_fn_t *gf = calloc(1, sizeof(ax_grad_fn_t));
    if (!gf) return NULL;
    gf->backward = fn;
    return gf;
}

void ax_zero_grad(ax_tensor_t *t)
{
    if (!t) return;
    if (t->grad)
    {
        ax_compute_fill(t->grad, 0.0);
    }
}

/* we need to collect all nodes in reverse topological order.
   classic approach: dfs with visited set, append on exit.

   keeping it simple with a flat array. for the kinds of graphs
   we'll be building (hundreds, not millions of nodes), this is fine. */

#define MAX_GRAPH_NODES 4096

typedef struct
{
    ax_tensor_t *nodes[MAX_GRAPH_NODES];
    int count;
} topo_list_t;

/* visited set — just linear scan, it's small */
static bool in_list(topo_list_t *list, ax_tensor_t *t)
{
    for (int i = 0; i < list->count; i++)
    {
        if (list->nodes[i] == t) return true;
    }
    return false;
}

/* dfs to build reverse topological order */
static void topo_sort_dfs(ax_tensor_t *t, topo_list_t *visited, topo_list_t *order)
{
    if (!t || in_list(visited, t)) return;
    if (visited->count >= MAX_GRAPH_NODES) return; /* bail if graph is huge */

    visited->nodes[visited->count++] = t;

    /* recurse into inputs */
    ax_grad_fn_t *gf = (ax_grad_fn_t *)t->grad_fn;
    if (gf)
    {
        for (int i = 0; i < gf->n_inputs; i++)
        {
            topo_sort_dfs(gf->inputs[i], visited, order);
        }
    }

    /* post-order: append after visiting children */
    if (order->count < MAX_GRAPH_NODES)
    {
        order->nodes[order->count++] = t;
    }
}

ax_status_t ax_backward(ax_tensor_t *loss)
{
    if (!loss)
    {
        ax_err_set(AX_ERR_NULL_ARG, "backward called on null tensor");
        return AX_ERR_NULL_ARG;
    }

    if (ax_tensor_numel(loss) != 1)
    {
        ax_err_set(AX_ERR_SHAPE_MISMATCH,
                   "backward expects scalar loss, got %ld elements",
                   ax_tensor_numel(loss));
        return AX_ERR_SHAPE_MISMATCH;
    }

    /* seed the loss gradient with 1.0 (dloss/dloss = 1) */
    if (!loss->grad)
    {
        loss->grad = ax_tensor_ones(loss->shape, loss->ndim, loss->dtype);
    }
    else
    {
        ax_compute_fill(loss->grad, 1.0);
    }

    /* build topological order */
    topo_list_t visited = { .count = 0 };
    topo_list_t order = { .count = 0 };
    topo_sort_dfs(loss, &visited, &order);

    /* walk in reverse (from loss toward leaves) and call backward fns */
    for (int i = order.count - 1; i >= 0; i--)
    {
        ax_tensor_t *node = order.nodes[i];
        ax_grad_fn_t *gf = (ax_grad_fn_t *)node->grad_fn;

        if (!gf || !gf->backward) continue;
        if (!node->grad) continue;

        gf->backward(gf, node->grad);
    }

    return AX_OK;
}

/* gradient checking utility: compare analytical vs numerical gradient.
   forward_fn should take an input tensor and return a scalar loss.
   we perturb each element of input by +/- eps and measure the change. */
double ax_grad_check(
    ax_tensor_t *(*forward_fn)(ax_tensor_t *input),
    ax_tensor_t *input,
    double eps)
{
    if (!input || !forward_fn) return -1.0;

    int64_t n = ax_tensor_numel(input);
    float *data = (float *)input->storage->data;
    double max_diff = 0.0;

    /* compute analytical gradients first */
    input->requires_grad = true;
    ax_zero_grad(input);

    ax_enable_grad();
    ax_tensor_t *loss = forward_fn(input);
    ax_backward(loss);

    /* now compute numerical gradients and compare */
    for (int64_t i = 0; i < n; i++)
    {
        float original = data[input->offset + i];

        /* f(x + eps) */
        data[input->offset + i] = original + (float)eps;
        ax_no_grad();
        ax_tensor_t *loss_plus = forward_fn(input);
        float f_plus = ((float *)loss_plus->storage->data)[loss_plus->offset];

        /* f(x - eps) */
        data[input->offset + i] = original - (float)eps;
        ax_tensor_t *loss_minus = forward_fn(input);
        float f_minus = ((float *)loss_minus->storage->data)[loss_minus->offset];

        /* restore */
        data[input->offset + i] = original;
        ax_enable_grad();

        /* numerical gradient: (f(x+e) - f(x-e)) / 2e */
        double numerical = (double)(f_plus - f_minus) / (2.0 * eps);

        /* analytical gradient from backward */
        /* flat index into grad tensor */
        int64_t remaining = i;
        int64_t indices[AX_MAX_DIMS];
        for (int d = input->ndim - 1; d >= 0; d--)
        {
            indices[d] = remaining % input->shape[d];
            remaining /= input->shape[d];
        }
        double analytical = (double)ax_tensor_get_f32(input->grad, indices);

        double diff = analytical - numerical;
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;

        ax_tensor_destroy(loss_plus);
        ax_tensor_destroy(loss_minus);
    }

    ax_tensor_destroy(loss);
    return max_diff;
}
