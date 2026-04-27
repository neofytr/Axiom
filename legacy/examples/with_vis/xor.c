#include "../../nn_visualizer.h"

ELEMENT_TYPE xor_data[] = {
    // Input pairs | Expected output
    1, 0, 1,
    0, 1, 1,
    1, 1, 0,
    0, 0, 0};

int main(void)
{
    size_t arch[] = {2, 2, 1};

    Mat *training_input = &(Mat){
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .es = &xor_data[0]};

    Mat *training_output = &(Mat){
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .es = &xor_data[2]};

    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    if (!nn)
    {
        fprintf(stderr, "Failed to allocate neural network\n");
        return 1;
    }

    randomize_parameters_NN(nn, 0, 0);

    Visualizer *vis = init_visualizer();
    if (!vis)
    {
        fprintf(stderr, "Failed to initialize visualizer\n");
        // nn_free(nn);
        return 1;
    }

    printf("Training XOR gate neural network...\n");
    printf("Press ESC to stop training early\n");

    learn_with_visualization(
        nn,
        1e-2,
        1e-2,
        10000,
        training_input,
        training_output,
        vis);

    printf("\nFinal XOR predictions:\n");
    for (size_t a1 = 0; a1 < 2; a1++)
    {
        for (size_t a0 = 0; a0 < 2; a0++)
        {
            ELEMENT_TYPE input[] = {
                (ELEMENT_TYPE)a1,
                (ELEMENT_TYPE)a0,
            };
            Mat input_mat = {
                .rows = 1,
                .cols = 2,
                .es = input};
            *(NN_INPUT(nn)) = input_mat;
            forward_NN(nn);
            printf("%zu + %zu = %f\n",
                   a1, a0,
                   NN_OUTPUT(nn)->es[0]);
        }
    }

    cleanup_visualizer(vis);

    return 0;
}