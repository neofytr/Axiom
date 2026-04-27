#define NN_IMPLEMENTATION_
#include "../../nn.h"
ELEMENT_TYPE xor [] = {
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
};
int main()
{
    size_t arch[] = {2, 2, 1};
    Mat *training_input = &(Mat){.rows = 4, .cols = 2, .stride = 3, .es = &xor[0]};
    Mat *training_output = &(Mat){.rows = 4, .cols = 1, .stride = 3, .es = &xor[2]};
    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    randomize_parameters_NN(nn, 0, 0);
    learn(nn,
          1e-1,
          1e-1,
          1000000,
          training_input,
          training_output);
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
}