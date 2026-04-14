/* bench_parity.c — numerical correctness gate vs tf_parity.py.

   both sides use a shared PRNG seed and identical shapes.  Axiom runs
   its version of the op, saves the output to a .axt tensor file via
   ax_tensor_save.  tf_parity.py loads the same inputs, runs the TF
   reference op, and compares the two outputs element-wise.

   the workflow step runs this, then runs tf_parity.py, which reports
   max_err per op.  any op where max_err exceeds 1e-3 fails the gate. */

#include "axiom/axiom.h"
#include "bench_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tiny deterministic PRNG — matches tf_parity.py seeding so both sides
   see the same input data without needing to share it on disk. */
static uint64_t g_state = 0x1234567890abcdefULL;
static void seed(uint64_t s) { g_state = s ? s : 1; }
static float randf(void) {
    g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17;
    return (float)(g_state >> 40) / 16777216.0f - 0.5f;
}

static void save(ax_tensor_t *t, const char *path) {
    if (ax_tensor_save(t, path) != AX_OK)
        fprintf(stderr, "save %s failed\n", path);
}

static ax_tensor_t *make_rand(const int64_t *sh, int nd) {
    ax_tensor_t *t = ax_tensor_create(sh, nd, AX_FLOAT32);
    if (!t) return NULL;
    int64_t n = ax_tensor_numel(t);
    float *d = (float *)t->storage->data;
    for (int64_t i = 0; i < n; i++) d[i] = randf();
    return t;
}

static void save_inputs(const char *tag, ax_tensor_t **in, int n) {
    for (int i = 0; i < n; i++) {
        char p[256]; snprintf(p, sizeof(p), "build/parity_%s_in%d.axt", tag, i);
        save(in[i], p);
    }
}

static void save_output(const char *tag, ax_tensor_t *out) {
    char p[256]; snprintf(p, sizeof(p), "build/parity_%s_out.axt", tag);
    save(out, p);
}

static void parity_matmul(void) {
    seed(1);
    int64_t a_sh[] = {256, 512}, b_sh[] = {512, 384};
    ax_tensor_t *A = make_rand(a_sh, 2);
    ax_tensor_t *B = make_rand(b_sh, 2);
    save_inputs("matmul", (ax_tensor_t*[]){A, B}, 2);
    ax_tensor_t *C = ax_matmul(A, B);
    save_output("matmul", C);
    ax_tensor_destroy(A); ax_tensor_destroy(B); ax_tensor_destroy(C);
}

static void parity_softmax(void) {
    seed(2);
    int64_t sh[] = {64, 512};
    ax_tensor_t *X = make_rand(sh, 2);
    save_inputs("softmax", &X, 1);
    ax_tensor_t *Y = ax_softmax(X, 1);
    save_output("softmax", Y);
    ax_tensor_destroy(X); ax_tensor_destroy(Y);
}

static void parity_gelu(void) {
    seed(3);
    int64_t sh[] = {32, 1024};
    ax_tensor_t *X = make_rand(sh, 2);
    save_inputs("gelu", &X, 1);
    ax_tensor_t *Y = ax_gelu(X);
    save_output("gelu", Y);
    ax_tensor_destroy(X); ax_tensor_destroy(Y);
}

static void parity_layernorm(void) {
    seed(4);
    int64_t sh[] = {32, 256};
    ax_tensor_t *X = make_rand(sh, 2);
    save_inputs("layernorm", &X, 1);
    ax_layer_t *ln = ax_layernorm_create(256, 1e-5f);
    ax_no_grad();
    ax_layer_eval(ln);
    ax_tensor_t *Y = ax_layer_forward(ln, X);
    save_output("layernorm", Y);
    ax_enable_grad();
    ax_tensor_destroy(X); ax_tensor_destroy(Y);
    ax_layer_destroy(ln);
}

static void parity_sdpa(void) {
    seed(5);
    int BH = 4, S = 64, dk = 32;
    int64_t sh[] = {BH, S, dk};
    ax_tensor_t *Q = make_rand(sh, 3);
    ax_tensor_t *K = make_rand(sh, 3);
    ax_tensor_t *V = make_rand(sh, 3);
    save_inputs("sdpa", (ax_tensor_t*[]){Q, K, V}, 3);

    ax_tensor_t *Out = ax_tensor_create(sh, 3, AX_FLOAT32);
    float scale = 1.0f / 5.656854f; /* 1/sqrt(32) */
    ax_fused_attention_fwd(
        (float*)Q->storage->data, (float*)K->storage->data, (float*)V->storage->data,
        (float*)Out->storage->data, NULL,
        BH, S, dk, scale);
    save_output("sdpa", Out);
    ax_tensor_destroy(Q); ax_tensor_destroy(K); ax_tensor_destroy(V);
    ax_tensor_destroy(Out);
}

int main(void) {
    ax_init();
    parity_matmul();
    parity_softmax();
    parity_gelu();
    parity_layernorm();
    parity_sdpa();
    printf("# parity inputs/outputs written to build/parity_*.axt\n");
    ax_shutdown();
    return 0;
}
