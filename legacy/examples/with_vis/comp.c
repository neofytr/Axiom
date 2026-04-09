#define NN_IMPLEMENTATION_
#include "../../nn_visualizer.h"
#include <stdio.h>

ELEMENT_TYPE xor_and_data[] = {
    0, 0, 0, 0, // A XOR B = 0, 0 AND C = 0
    0, 0, 1, 0,
    0, 1, 0, 0, // A XOR B = 1, 1 AND C = 0
    0, 1, 1, 1,
    1, 0, 0, 0, // A XOR B = 1, 1 AND C = 0
    1, 0, 1, 1,
    1, 1, 0, 0, // A XOR B = 0, 0 AND C = 0
    1, 1, 1, 0};

int main()
{
    size_t arch[] = {3, 4, 1};

    Mat *training_input = &(Mat){
        .rows = 8,             // Number of training examples
        .cols = 3,             // Three inputs for XOR-AND gate
        .stride = 4,           // Stride in the xor_and_data array (input: 3 elements, output: 1 element per row)
        .es = &xor_and_data[0] // Pointer to the input data
    };
    Mat *training_output = &(Mat){
        .rows = 8,             // Number of training examples
        .cols = 1,             // One output for XOR-AND gate
        .stride = 4,           // Stride in the xor_and_data array (input: 3 elements, output: 1 element per row)
        .es = &xor_and_data[3] // Pointer to the output data
    };

    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    randomize_parameters_NN(nn, 0, 0);

    Visualizer *vis = init_visualizer();
    if (!vis)
    {
        fprintf(stderr, "Failed to initialize visualizer\n");
        // nn_free(nn);
        return 1;
    }

    printf("Training neural network...\n");
    printf("Press ESC to stop training early\n");

    learn_with_visualization(
        nn,
        1e-2,
        1e-2,
        10000,
        training_input,
        training_output,
        vis);

    for (size_t c = 0; c < 2; c++)
    {
        for (size_t b = 0; b < 2; b++)
        {
            for (size_t a = 0; a < 2; a++)
            {
                ELEMENT_TYPE input[] = {
                    (ELEMENT_TYPE)a,
                    (ELEMENT_TYPE)b,
                    (ELEMENT_TYPE)c,
                };
                Mat input_mat = {
                    .rows = 1,
                    .cols = 3,
                    .es = input};
                *(NN_INPUT(nn)) = input_mat;
                forward_NN(nn);
                printf("%zu XOR %zu AND %zu = %f\n",
                       a, b, c,
                       NN_OUTPUT(nn)->es[0]);
            }
        }
    }

    // nn_free(nn);

    return 0;
}
