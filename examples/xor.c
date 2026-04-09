/* xor.c — trains a small network to learn the xor function.
   the simplest possible non-linear problem. if this works, the
   framework works: autograd, optimizers, layers, the whole chain. */

#include "axiom/axiom.h"
#include <stdio.h>

int main(void)
{
    ax_init();

    /* xor truth table:
       0 xor 0 = 0
       0 xor 1 = 1
       1 xor 0 = 1
       1 xor 1 = 0 */
    float x_data[] = {0, 0,
                      0, 1,
                      1, 0,
                      1, 1};
    float y_data[] = {0,
                      1,
                      1,
                      0};

    int64_t x_shape[] = {4, 2};
    int64_t y_shape[] = {4, 1};
    ax_tensor_t *train_x = ax_tensor_from_array(x_data, x_shape, 2, AX_FLOAT32);
    ax_tensor_t *train_y = ax_tensor_from_array(y_data, y_shape, 2, AX_FLOAT32);

    /* network: 2 -> 16 -> relu -> 16 -> relu -> 1 */
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 16, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 16, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(16, 1, true));

    ax_model_t *model = ax_model_create(net);

    printf("model architecture:\n");
    ax_model_summary(model);
    printf("\n");

    /* adam optimizer */
    ax_optimizer_t *opt = ax_adam_create(
        model->params, model->n_params,
        0.01f,     /* lr */
        0.9f,      /* beta1 */
        0.999f,    /* beta2 */
        1e-8f,     /* eps */
        0.0f       /* weight decay */
    );
    ax_model_compile(model, opt, ax_mse_loss);

    /* train */
    printf("training on xor...\n\n");
    int epochs = 2000;
    for (int e = 0; e < epochs; e++)
    {
        float loss = ax_model_train_step(model, train_x, train_y);
        if (e % 200 == 0 || e == epochs - 1)
            printf("  epoch %4d  loss: %.6f\n", e + 1, loss);
    }

    /* test: run each input through the trained model */
    printf("\nresults:\n");
    for (int i = 0; i < 4; i++)
    {
        float in[2] = {x_data[i * 2], x_data[i * 2 + 1]};
        int64_t in_shape[] = {1, 2};
        ax_tensor_t *input = ax_tensor_from_array(in, in_shape, 2, AX_FLOAT32);

        ax_tensor_t *pred = ax_model_predict(model, input);
        int64_t idx[] = {0, 0};
        float val = ax_tensor_get_f32(pred, idx);

        printf("  %d xor %d = %.4f  (expected %d)\n",
               (int)in[0], (int)in[1], val, (int)y_data[i]);

        ax_tensor_destroy(input);
    }

    /* save the trained model */
    ax_model_save(model, "xor_model.axm");
    printf("\nmodel saved to xor_model.axm\n");

    /* load it back and verify */
    ax_model_t *loaded = ax_model_load("xor_model.axm");
    if (loaded)
    {
        printf("\nloaded model predictions:\n");
        for (int i = 0; i < 4; i++)
        {
            float in[2] = {x_data[i * 2], x_data[i * 2 + 1]};
            int64_t in_shape[] = {1, 2};
            ax_tensor_t *input = ax_tensor_from_array(in, in_shape, 2, AX_FLOAT32);

            ax_tensor_t *pred = ax_model_predict(loaded, input);
            int64_t idx[] = {0, 0};
            float val = ax_tensor_get_f32(pred, idx);

            printf("  %d xor %d = %.4f\n", (int)in[0], (int)in[1], val);
            ax_tensor_destroy(input);
        }
        ax_model_destroy(loaded);
    }

    /* cleanup */
    ax_tensor_destroy(train_x);
    ax_tensor_destroy(train_y);
    ax_model_destroy(model);
    ax_shutdown();

    return 0;
}
