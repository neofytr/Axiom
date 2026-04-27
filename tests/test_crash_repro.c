#include "axiom/axiom.h"
#include "axiom/autograd.h"
#include <stdio.h>

int main(void)
{
    ax_init();
    ax_layer_t *conv = ax_conv2d_create_ex(256, 512, 3, 3, 1, 1, 1, 1, true);
    int64_t in_sh[] = {1, 256, 4, 4};
    ax_tensor_t *x = ax_tensor_rand(in_sh, 4, -0.5f, 0.5f);
    x->requires_grad = true;
    ax_layer_train(conv);

    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "iter %d\n", i);
        ax_tensor_t *y = ax_layer_forward(conv, x);
        ax_tensor_t *loss = ax_sum(y, -1);
        ax_backward(loss);
        ax_graph_cleanup(loss);
        ax_tensor_destroy(loss);
        ax_tensor_destroy(y);
    }

    printf("ok\n");
    ax_tensor_destroy(x);
    ax_layer_destroy(conv);
    return 0;
}
