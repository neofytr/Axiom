/* autograd.c — the backward pass engine.
   does a reverse topological sort of the computation graph
   then calls each op's backward function in order. */

#include "axiom/autograd.h"
#include "axiom/compute.h"
#include "axiom/memory.h"
#include "axiom/error.h"
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

/* thread-local gradient tracking — safe for concurrent inference */
static _Thread_local bool grad_tracking = true;

/* thread-local arena for backward pass scratch allocations.
   created lazily, reset after each ax_backward() call.
   backward functions can use ax_backward_arena() to get fast
   bump-allocated scratch that's freed in bulk. */
static _Thread_local ax_arena_t *backward_arena = NULL;

ax_arena_t *ax_backward_arena(void)
{
    /* when the default device is non-cpu (e.g. cuda), return NULL so
       arena callers fall back to the device-aware heap path (tensor.c's
       ax_tensor_zeros / ax_tensor_create route through the owning
       backend's storage_alloc). the cpu arena only holds host memory
       and can't be used for device tensors. */
    if (ax_get_default_device() != AX_DEVICE_CPU)
        return NULL;
    if (!backward_arena)
        backward_arena = ax_arena_create(0);
    return backward_arena;
}

/* thread-local arena for tensors that the forward pass must save for backward
   (e.g. batchnorm x_hat / inv_std). lifetime: spans forward + backward of one
   training step. reset only by ax_graph_cleanup, never by ax_backward, so the
   saved tensors stay valid through the backward walk. */
static _Thread_local ax_arena_t *forward_arena = NULL;

ax_arena_t *ax_forward_arena(void)
{
    if (ax_get_default_device() != AX_DEVICE_CPU)
        return NULL;
    if (!forward_arena)
        forward_arena = ax_arena_create(0);
    return forward_arena;
}

/* nestable no-grad: ax_no_grad increments a depth counter,
   ax_enable_grad decrements. grad is enabled only at depth 0.
   this lets library code safely disable grad without clobbering
   the caller's no-grad scope. */
static _Thread_local int no_grad_depth = 0;

void ax_no_grad(void)    { no_grad_depth++; grad_tracking = false; }
void ax_enable_grad(void){
    if (no_grad_depth > 0) no_grad_depth--;
    if (no_grad_depth == 0) grad_tracking = true;
}
bool ax_grad_enabled(void){ return grad_tracking; }

/* thread-local slab allocator for ax_grad_fn_t — eliminates per-op calloc.
   freed grad_fns are pushed onto a free-list; new allocations pop from it. */
static _Thread_local ax_grad_fn_t *grad_fn_freelist = NULL;

static ax_grad_fn_t *slab_grad_fn_alloc(void)
{
    ax_grad_fn_t *gf = grad_fn_freelist;
    if (gf) {
        grad_fn_freelist = *(ax_grad_fn_t **)gf; /* next pointer in first slot */
        memset(gf, 0, sizeof(ax_grad_fn_t));
        return gf;
    }
    return (ax_grad_fn_t *)calloc(1, sizeof(ax_grad_fn_t));
}

static void slab_grad_fn_free(ax_grad_fn_t *gf)
{
    if (!gf) return;
    *(ax_grad_fn_t **)gf = grad_fn_freelist;
    grad_fn_freelist = gf;
}

ax_grad_fn_t *ax_grad_fn_create(ax_backward_fn_t fn)
{
    ax_grad_fn_t *gf = slab_grad_fn_alloc();
    if (!gf) return NULL;
    gf->backward = fn;
    return gf;
}

void ax_grad_fn_destroy(ax_grad_fn_t *gf)
{
    slab_grad_fn_free(gf);
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

   both lists are heap-allocated and grow dynamically to avoid
   blowing the stack on embedded targets (typical limit: 8-64KB). */

#define TOPO_INIT_CAP 4096

/* open-addressing hash set on pointer values.
   O(1) amortized lookup instead of O(n) linear scan. */
typedef struct {
    uintptr_t *buckets;
    int capacity;
    int count;
} ptr_set_t;

static bool ps_init(ptr_set_t *s, int cap)
{
    s->buckets = (uintptr_t *)calloc((size_t)cap, sizeof(uintptr_t));
    if (!s->buckets) return false;
    s->capacity = cap;
    s->count = 0;
    return true;
}

/* fast reset: zero buckets in place, preserve capacity */
static void ps_reset(ptr_set_t *s)
{
    if (s->buckets && s->capacity > 0)
        memset(s->buckets, 0, (size_t)s->capacity * sizeof(uintptr_t));
    s->count = 0;
}

/* fibonacci hashing for pointer -> slot */
static inline int ps_slot(uintptr_t key, int cap)
{
    return (int)((key * 0x9e3779b97f4a7c15ULL) >> 32) & (cap - 1);
}

static bool ps_grow(ptr_set_t *s)
{
    int new_cap = s->capacity * 2;
    uintptr_t *old = s->buckets;
    int old_cap = s->capacity;

    s->buckets = (uintptr_t *)calloc((size_t)new_cap, sizeof(uintptr_t));
    if (!s->buckets) { s->buckets = old; return false; }
    s->capacity = new_cap;
    s->count = 0;

    for (int i = 0; i < old_cap; i++)
    {
        if (old[i] != 0)
        {
            int slot = ps_slot(old[i], new_cap);
            while (s->buckets[slot] != 0) slot = (slot + 1) & (new_cap - 1);
            s->buckets[slot] = old[i];
            s->count++;
        }
    }
    free(old);
    return true;
}

/* returns true if key was already present */
static bool ps_insert(ptr_set_t *s, void *ptr)
{
    uintptr_t key = (uintptr_t)ptr;
    if (key == 0) return false; /* 0 is sentinel */

    /* grow at 70% load */
    if (s->count * 10 >= s->capacity * 7)
    {
        if (!ps_grow(s)) return false;
    }

    int slot = ps_slot(key, s->capacity);
    while (s->buckets[slot] != 0)
    {
        if (s->buckets[slot] == key) return true; /* already present */
        slot = (slot + 1) & (s->capacity - 1);
    }
    s->buckets[slot] = key;
    s->count++;
    return false;
}

static bool ps_contains(const ptr_set_t *s, void *ptr)
{
    uintptr_t key = (uintptr_t)ptr;
    if (key == 0) return false;
    int slot = ps_slot(key, s->capacity);
    while (s->buckets[slot] != 0)
    {
        if (s->buckets[slot] == key) return true;
        slot = (slot + 1) & (s->capacity - 1);
    }
    return false;
}

/* dynamic array for topo order */
typedef struct
{
    ax_tensor_t **nodes;
    int count;
    int capacity;
} topo_list_t;

static bool tl_init(topo_list_t *l)
{
    l->nodes = (ax_tensor_t **)malloc(TOPO_INIT_CAP * sizeof(ax_tensor_t *));
    if (!l->nodes) return false;
    l->count = 0;
    l->capacity = TOPO_INIT_CAP;
    return true;
}

/* fast reset: preserve capacity, just clear count */
static void tl_reset(topo_list_t *l) { l->count = 0; }

static bool tl_push(topo_list_t *l, ax_tensor_t *t)
{
    if (l->count >= l->capacity)
    {
        int new_cap = l->capacity * 2;
        ax_tensor_t **nn = (ax_tensor_t **)realloc(l->nodes,
                                                    (size_t)new_cap * sizeof(ax_tensor_t *));
        if (!nn) return false;
        l->nodes = nn;
        l->capacity = new_cap;
    }
    l->nodes[l->count++] = t;
    return true;
}

/* iterative dfs to build reverse topological order.
   avoids stack overflow on deep computation graphs. */
typedef struct {
    ax_tensor_t *node;
    int child_idx;
} dfs_frame_t;

/* thread-local persistent state for backward/cleanup — reused across calls.
   reset between uses instead of free+calloc. eliminates ~3 x (32KB + 32KB + 32KB)
   of metadata malloc traffic per training step. */
static _Thread_local ptr_set_t tl_visited = {0};
static _Thread_local topo_list_t tl_order = {0};
static _Thread_local dfs_frame_t *tl_dfs_stack = NULL;
static _Thread_local int tl_dfs_stack_cap = 0;

static bool ensure_autograd_state(void)
{
    if (!tl_visited.buckets) {
        if (!ps_init(&tl_visited, TOPO_INIT_CAP)) return false;
    }
    if (!tl_order.nodes) {
        if (!tl_init(&tl_order)) return false;
    }
    if (!tl_dfs_stack) {
        tl_dfs_stack_cap = TOPO_INIT_CAP;
        tl_dfs_stack = (dfs_frame_t *)malloc((size_t)tl_dfs_stack_cap * sizeof(dfs_frame_t));
        if (!tl_dfs_stack) { tl_dfs_stack_cap = 0; return false; }
    }
    return true;
}

static void topo_sort_dfs(ax_tensor_t *t, ptr_set_t *visited, topo_list_t *order)
{
    if (!t) return;

    int stack_top = 0;
    tl_dfs_stack[stack_top++] = (dfs_frame_t){ .node = t, .child_idx = 0 };

    while (stack_top > 0)
    {
        ax_tensor_t *node = tl_dfs_stack[stack_top - 1].node;
        int cidx = tl_dfs_stack[stack_top - 1].child_idx;

        if (cidx == 0)
        {
            if (ps_insert(visited, node)) { stack_top--; continue; } /* already visited */
        }

        ax_grad_fn_t *gf = (ax_grad_fn_t *)node->grad_fn;
        int n_children = gf ? gf->n_inputs : 0;

        if (cidx < n_children)
        {
            tl_dfs_stack[stack_top - 1].child_idx++;
            ax_tensor_t *child = gf->inputs[cidx];

            if (child && !ps_contains(visited, child))
            {
                if (stack_top >= tl_dfs_stack_cap)
                {
                    int new_cap = tl_dfs_stack_cap * 2;
                    dfs_frame_t *ns = (dfs_frame_t *)realloc(tl_dfs_stack,
                                          (size_t)new_cap * sizeof(dfs_frame_t));
                    if (!ns)
                    {
                        ax_err_set(AX_ERR_ALLOC, "topo_sort_dfs: stack growth failed");
                        break;
                    }
                    tl_dfs_stack = ns;
                    tl_dfs_stack_cap = new_cap;
                }
                tl_dfs_stack[stack_top++] = (dfs_frame_t){ .node = child, .child_idx = 0 };
            }
        }
        else
        {
            tl_push(order, node);
            stack_top--;
        }
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
                   "backward expects scalar loss, got %" PRId64 " elements",
                   ax_tensor_numel(loss));
        return AX_ERR_SHAPE_MISMATCH;
    }

    /* seed the loss gradient with 1.0 (dloss/dloss = 1) */
    if (!loss->grad)
    {
        loss->grad = ax_tensor_ones(loss->shape, loss->ndim, loss->dtype);
        if (!loss->grad)
        {
            ax_err_set(AX_ERR_ALLOC, "failed to allocate loss gradient");
            return AX_ERR_ALLOC;
        }
    }
    else
    {
        ax_compute_fill(loss->grad, 1.0);
    }

    /* use persistent thread-local state — no per-call calloc */
    if (!ensure_autograd_state())
    {
        ax_err_set(AX_ERR_ALLOC, "ax_backward: state alloc failed");
        return AX_ERR_ALLOC;
    }
    ps_reset(&tl_visited);
    tl_reset(&tl_order);

    topo_sort_dfs(loss, &tl_visited, &tl_order);

    /* walk in reverse (from loss toward leaves) and call backward fns */
    for (int i = tl_order.count - 1; i >= 0; i--)
    {
        ax_tensor_t *node = tl_order.nodes[i];
        ax_grad_fn_t *gf = (ax_grad_fn_t *)node->grad_fn;

        if (!gf || !gf->backward) continue;
        if (!node->grad) continue;

        gf->backward(gf, node->grad);
    }

    /* reset backward arena — frees all scratch from this pass in bulk */
    if (backward_arena)
        ax_arena_reset(backward_arena);

    return AX_OK;
}

void ax_graph_cleanup(ax_tensor_t *root)
{
    if (!root) return;

    if (!ensure_autograd_state()) return;
    ps_reset(&tl_visited);
    tl_reset(&tl_order);

    topo_sort_dfs(root, &tl_visited, &tl_order);

    /* two-pass cleanup. the topo order is leaf-first, so a tensor A may
       be destroyed (pass 1, old code) before a later node B's saved[j]
       reference to A has its retained refcount released. that leaves A's
       storage at refcount 1 (the retained ref) which is never freed.

       fix: pass 1 releases all saved refs + frees grad_fns but does NOT
       destroy tensor structs. pass 2 destroys the now-fully-released
       intermediates. a tombstone sentinel distinguishes cleaned-up
       intermediates (grad_fn set to tombstone in pass 1) from leaf
       params (grad_fn was always NULL). */
#define AX_GRAD_FN_TOMBSTONE ((void *)(uintptr_t)1)

    /* pass 1: release saved refs, free grad_fn structs, mark for pass 2 */
    for (int i = 0; i < tl_order.count; i++)
    {
        ax_tensor_t *node = tl_order.nodes[i];
        if (node == root || !node->grad_fn) continue;

        ax_grad_fn_t *gf = (ax_grad_fn_t *)node->grad_fn;
        if (gf->ctx && gf->ctx_cleanup)
        {
            gf->ctx_cleanup(gf->ctx);
            gf->ctx = NULL;
        }
        for (int j = 0; j < gf->n_saved; j++)
        {
            if (gf->saved_owned[j] && gf->saved[j])
            {
                ax_tensor_destroy(gf->saved[j]);
                gf->saved[j] = NULL;
            }
            else if (gf->saved_retained[j] && gf->saved[j] && gf->saved[j]->storage)
            {
                ax_storage_release(gf->saved[j]->storage);
                gf->saved_retained[j] = false;
            }
        }
        slab_grad_fn_free(gf);
        node->grad_fn = AX_GRAD_FN_TOMBSTONE;
    }

    /* pass 2: destroy tombstoned intermediates (retained refs already released) */
    for (int i = 0; i < tl_order.count; i++)
    {
        ax_tensor_t *node = tl_order.nodes[i];
        if (node->grad_fn == AX_GRAD_FN_TOMBSTONE)
        {
            node->grad_fn = NULL;
            ax_tensor_destroy(node);
        }
    }

    /* detach root from the graph without destroying it — caller owns it */
    if (root->grad_fn)
    {
        ax_grad_fn_t *gf = (ax_grad_fn_t *)root->grad_fn;
        if (gf->ctx && gf->ctx_cleanup)
        {
            gf->ctx_cleanup(gf->ctx);
            gf->ctx = NULL;
        }
        for (int i = 0; i < gf->n_saved; i++)
        {
            if (gf->saved_owned[i] && gf->saved[i])
            {
                ax_tensor_destroy(gf->saved[i]);
                gf->saved[i] = NULL;
            }
            else if (gf->saved_retained[i] && gf->saved[i] && gf->saved[i]->storage)
            {
                ax_storage_release(gf->saved[i]->storage);
                gf->saved_retained[i] = false;
            }
        }
        slab_grad_fn_free(gf);
        root->grad_fn = NULL;
    }

    /* reset forward arena now that the graph is gone — any arena-temp tensors
       that were saved on grad_fns have already been "destroyed" (no-op) above.
       safe to bulk free here. */
    if (forward_arena)
        ax_arena_reset(forward_arena);

    /* persistent state is reused across calls — no free here */
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

    ax_graph_cleanup(loss);
    ax_tensor_destroy(loss);
    return max_diff;
}
