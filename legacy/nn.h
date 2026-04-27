#ifndef NN_H_
#define NN_H_

#define GRAD_DESC 100
#define DIFF 200

#ifndef DECAY_RATE
#define DECAY_RATE 1e-1
#endif

#ifndef MODEL
#define MODEL GRAD_DESC
#endif // MODEL

#ifndef NN_ASSERT
#define NN_ASSERT assert
#endif // NN_ASSERT

#ifndef ELEMENT_TYPE
#define ELEMENT_TYPE float
#endif // ELEMENT_TYPE

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef MAX_RAND_ELEMENT
#define MAX_RAND_ELEMENT 1.f
#endif // MAX_RAND_ELEMENT

#ifndef MIN_RAND_ELEMENT
#define MIN_RAND_ELEMENT 0.f
#endif // MIN_RAND_ELEMENT

#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#if ELEMENT_TYPE == float
#define EXP expf
#elif ELEMENT_TYPE == double
#define EXP exp
#endif // EXP

#define SUCCESS 1
#define FAILURE 0
#define MAT_AT(a, i, j) a->es[(i) * (a->stride) + (j)]
#define MAT_AT_POINTER(a, i, j) &(a->es[(i) * (a->stride) + (j)])
#define MAT_PRINT(m) mat_print(m, #m, 0) // #m converts the tokens in m into a string
#define OUTPUT_AT(nn, input_set_no, output_set_col) (nn->model_output[input_set_no * NEURONS_IN_LAYER(nn, 0) + output_set_col])
#define NN_PRINT(n) nn_print(n, #n, 4)
#define NUMS(a) (sizeof(a) / sizeof(a[0]))
#define NEURONS_IN_LAYER(nn, layer_no) ((nn->arch[layer_no]))
#define DS_OF_LAYER(nn, layer_no) (nn->ds[layer_no])
#define ARRAY_LEN(xs) (sizeof(xs) / sizeof(xs[0]))

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    ELEMENT_TYPE *es;
} Mat;

Mat *mat_alloc(size_t rows, size_t cols);
void mat_dealloc(Mat *a);
void mat_rand(Mat *a, ELEMENT_TYPE high, ELEMENT_TYPE low);
void mat_dot(Mat *dst, Mat *a, Mat *b);
void mat_sum(Mat *dst, Mat *a);
void mat_print(Mat *a, const char *name, size_t padding);
void mat_sig(Mat *a);
static inline ELEMENT_TYPE rand_element(ELEMENT_TYPE high, ELEMENT_TYPE low);
static inline ELEMENT_TYPE sigmoid(ELEMENT_TYPE element);
static inline Mat mat_row(Mat *a, size_t row);
static inline void mat_copy(Mat *dst, Mat *src);

typedef struct
{
    size_t arch_count;
    size_t *arch;
    ELEMENT_TYPE *model_output;
    Mat **ws; // Weights
    Mat **bs; // Biases
    Mat **as; // Activations
    Mat **ds; // Deltas
} NN;

#define NN_INPUT(n) (n->as[0])
#define NN_OUTPUT(n) (n->as[n->arch_count - 1])

void nn_print(NN *nn, const char *name, size_t padding);
NN *nn_alloc(size_t *arch, size_t arch_count);
void nn_delloc(NN *nn);
void randomize_parameters_NN(NN *nn, int high, int low);
void forward(NN *nn);
ELEMENT_TYPE cost_NN(NN *nn, Mat *training_input, Mat *training_output);
void gradient_descent(NN *nn, ELEMENT_TYPE learning_rate, size_t epoch);
void diff(NN *nn, Mat *training_input, Mat *training_output, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate, Mat *temp_para, ELEMENT_TYPE cost);
void delta(NN *nn, Mat *training_input, Mat *training_output, ELEMENT_TYPE learning_rate);
void learn(NN *nn, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate, size_t learning_iterations, Mat *training_input, Mat *training_output);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION_

Mat *mat_alloc(size_t rows, size_t cols)
{
    Mat *a = NN_MALLOC(sizeof(Mat));
    if (!a)
    {
        perror("mat_alloc");
        exit(EXIT_FAILURE);
    }

    a->rows = rows;
    a->cols = cols;
    a->es = calloc(rows * cols, sizeof(ELEMENT_TYPE));
    a->stride = cols;
    if (!a->es)
    {
        perror("mat_alloc");
        exit(EXIT_FAILURE);
    }

    return a;
}

void mat_print(Mat *a, const char *name, size_t padding)
{
    printf("%*s%s: [\n", (int)padding, "", name);
    for (size_t i = 0; i < a->rows; i++)
    {
        printf("%*s", (int)padding, "");
        for (size_t j = 0; j < a->cols; j++)
        {
            printf("    %8f ", MAT_AT(a, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void mat_rand(Mat *a, ELEMENT_TYPE high, ELEMENT_TYPE low)
{
    srand(time(NULL));
    if (high == 0 && low == 0)
    {
        high = MAX_RAND_ELEMENT;
        low = MIN_RAND_ELEMENT;
    }

    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a->rows; i++)
    {
        for (size_t j = 0; j < a->cols; j++)
        {
            MAT_AT(a, i, j) = rand_element(high, low);
        }
    }
}

static inline ELEMENT_TYPE rand_element(ELEMENT_TYPE high, ELEMENT_TYPE low)
{
    return (((float)rand() / (float)RAND_MAX) * (high - low) + low);
}

void mat_dealloc(Mat *a)
{
    free(a->es);
    free(a);
    a = NULL;
}

void mat_sum(Mat *dst, Mat *a)
{
    NN_ASSERT(dst->rows == a->rows);
    NN_ASSERT(dst->cols == a->cols);

    size_t rows = a->rows;
    size_t cols = a->cols;

    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_fill(Mat *a, ELEMENT_TYPE element)
{
    size_t rows = a->rows;
    size_t cols = a->cols;

    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            MAT_AT(a, i, j) = element;
        }
    }
}

void mat_dot(Mat *dst, Mat *a, Mat *b)
{
    NN_ASSERT(a->cols == b->rows);
    size_t n = a->cols;
    NN_ASSERT(dst->rows == a->rows);
    NN_ASSERT(dst->cols == b->cols);

    size_t rows = dst->rows;
    size_t cols = dst->cols;

    // #pragma omp parallel for
    //  #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

static inline ELEMENT_TYPE sigmoid(ELEMENT_TYPE element)
{
    return ((ELEMENT_TYPE)1 / ((ELEMENT_TYPE)1 + EXP(-element)));
}

void mat_sig(Mat *a)
{
    size_t rows = a->rows;
    size_t cols = a->cols;

    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            MAT_AT(a, i, j) = sigmoid(MAT_AT(a, i, j));
        }
    }
}

static inline Mat mat_row(Mat *a, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = a->cols,
        .stride = a->cols,
        .es = &MAT_AT(a, row, 0),
    };
}

static inline void mat_copy(Mat *dst, Mat *src)
{
    NN_ASSERT(dst->rows == src->rows);
    NN_ASSERT(dst->cols == src->cols);

    size_t rows = dst->rows;
    size_t cols = dst->cols;

    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void nn_print(NN *nn, const char *name, size_t padding)
{
    char buffer[64];
    printf("%s: [\n", name);
    for (size_t i = 1; i < nn->arch_count; i++)
    {
        sprintf(buffer, "Weigth Matrix, Layer No. %zu", i);
        mat_print(nn->ws[i], buffer, padding);
        sprintf(buffer, "Bias Matrix, Layer No. %zu", i);
        mat_print(nn->bs[i], buffer, padding);
    }
    printf("]\n");
}

NN *nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN_ASSERT(arch != NULL);
    NN *nn = malloc(sizeof(NN));
    if (!nn)
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    nn->ws = malloc(sizeof(Mat *) * (arch_count));
    if (!(nn->ws))
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    nn->bs = malloc(sizeof(Mat *) * (arch_count));
    if (!(nn->bs))
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    nn->as = malloc(sizeof(Mat *) * (arch_count));
    if (!(nn->as))
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    nn->ds = malloc(sizeof(Mat *) * arch_count);
    if (!(nn->ds))
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < arch_count; i++)
    {
        if (i == 0)
        {
            nn->ws[i] = NULL;
            nn->bs[i] = NULL;
            nn->as[i] = mat_alloc(1, arch[i]);
            nn->ds[i] = mat_alloc(1, arch[i]);
        }
        else
        {
            nn->as[i] = mat_alloc(1, arch[i]);
            nn->ws[i] = mat_alloc(arch[i - 1], arch[i]);
            nn->bs[i] = mat_alloc(1, arch[i]);
            nn->ds[i] = mat_alloc(1, arch[i]);
        }
    }

    nn->arch = arch;
    nn->arch_count = arch_count;
    return nn;
}

void randomize_parameters_NN(NN *nn, int high, int low)
{
    NN_ASSERT(nn != NULL);
    size_t arch_count = nn->arch_count;

    // #pragma omp parallel for
    for (size_t i = 1; i < arch_count; i++)
    {
        mat_rand(nn->ws[i], high, low);
        mat_rand(nn->bs[i], high, low);
    }
}

void forward_NN(NN *nn)
{
    NN_ASSERT(nn != NULL);
    size_t arch_count = nn->arch_count;
    for (size_t i = 1; i < arch_count; i++)
    {
        mat_dot(nn->as[i], nn->as[i - 1], nn->ws[i]);
        mat_sum(nn->as[i], nn->bs[i]);
        mat_sig(nn->as[i]);
    }
}

ELEMENT_TYPE cost_NN(NN *nn, Mat *training_input, Mat *training_output)
{
    NN_ASSERT(nn != NULL);
    NN_ASSERT(training_input != NULL);
    NN_ASSERT(training_output != NULL);
    assert(training_input->rows == training_output->rows);
    assert(NN_OUTPUT(nn)->cols == training_output->cols);

    size_t rows = training_input->rows;
    size_t cols = training_output->cols;

    ELEMENT_TYPE result = (ELEMENT_TYPE)0;
    // #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < rows; i++)
    {
        Mat x = mat_row(training_input, i);
        Mat y = mat_row(training_output, i);
        Mat *y_ = &y;

        *(NN_INPUT(nn)) = x;
        forward_NN(nn);

        for (size_t j = 0; j < cols; j++)
        {
            ELEMENT_TYPE d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y_, 0, j); // Correct row access
            result += d * d;
        }
    }
    return (result / rows);
}

void diff(NN *nn, Mat *training_input, Mat *training_output, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate, Mat *temp_para, ELEMENT_TYPE c)
{
    ELEMENT_TYPE saved;
    size_t rows = temp_para->rows, cols = temp_para->cols;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            saved = MAT_AT(temp_para, i, j);
            *MAT_AT_POINTER(temp_para, i, j) += eps;
            ELEMENT_TYPE d = (cost_NN(nn, training_input, training_output) - c) / eps;
            *MAT_AT_POINTER(temp_para, i, j) = saved;
            *MAT_AT_POINTER(temp_para, i, j) -= d * learning_rate;
        }
    }
}

void delta(NN *nn, Mat *training_input, Mat *training_output, ELEMENT_TYPE learning_rate)
{
    size_t arch_count = nn->arch_count;
    size_t neurons_in_last_layer = NEURONS_IN_LAYER(nn, arch_count - 1);
    size_t neurons_in_first_layer = NEURONS_IN_LAYER(nn, 0);
    NN_ASSERT(neurons_in_first_layer == training_input->cols);

    for (size_t j = 0; j < training_input->rows; j++)
    {
        Mat x = mat_row(training_input, j);
        *(NN_INPUT(nn)) = x;
        forward_NN(nn);

        // #pragma omp parallel for
        for (size_t k = 0; k < neurons_in_last_layer; k++)
        {
            ELEMENT_TYPE a = MAT_AT(NN_OUTPUT(nn), 0, k);
            ELEMENT_TYPE error = a - MAT_AT(training_output, j, k);
            MAT_AT(DS_OF_LAYER(nn, arch_count - 1), 0, k) = (2 * error * a * (1 - a)) / neurons_in_last_layer;
        }

        for (size_t layer = arch_count - 2; layer > 0; layer--)
        {
            for (size_t neuron = 0; neuron < nn->arch[layer]; neuron++)
            {
                ELEMENT_TYPE delta = (ELEMENT_TYPE)0;
                for (size_t next_neuron = 0; next_neuron < nn->arch[layer + 1]; next_neuron++)
                {
                    delta += MAT_AT(DS_OF_LAYER(nn, layer + 1), 0, next_neuron) * MAT_AT(nn->ws[layer + 1], neuron, next_neuron);
                }
                ELEMENT_TYPE a = MAT_AT(nn->as[layer], 0, neuron);
                MAT_AT(DS_OF_LAYER(nn, layer), 0, neuron) = delta * a * (1 - a);
            }
        }

        gradient_descent(nn, learning_rate, j);
    }
}

void gradient_descent(NN *nn, ELEMENT_TYPE learning_rate, size_t epoch)
{
    size_t arch_count = nn->arch_count;
    // #pragma omp parallel for collapse(1)
    for (size_t layer = 1; layer < arch_count; layer++)
    {
        for (size_t neuron = 0; neuron < nn->arch[layer]; neuron++)
        {
            for (size_t weight = 0; weight < nn->arch[layer - 1]; weight++)
            {
                ELEMENT_TYPE a = MAT_AT(nn->as[layer - 1], 0, weight);
                ELEMENT_TYPE delta = MAT_AT(DS_OF_LAYER(nn, layer), 0, neuron);
                MAT_AT(nn->ws[layer], weight, neuron) -= learning_rate * delta * a;
            }
            MAT_AT(nn->bs[layer], 0, neuron) -= learning_rate * MAT_AT(DS_OF_LAYER(nn, layer), 0, neuron);
        }
    }
}

void learn(NN *nn, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate, size_t learning_iterations, Mat *training_input, Mat *training_output)
{

    for (size_t i = 0; i < learning_iterations; i++)
    {
#if MODEL == DIFF
        size_t arch_count = nn->arch_count;
        ELEMENT_TYPE c = cost_NN(nn, training_input, training_output);
        for (size_t j = 1; j < arch_count; j++)
        {
            diff(nn, training_input, training_output, eps, learning_rate, nn->ws[j], c);
            diff(nn, training_input, training_output, eps, learning_rate, nn->bs[j], c);
        }
#elif MODEL == GRAD_DESC
        delta(nn, training_input, training_output, learning_rate);
#endif // MODEL
    }
    ELEMENT_TYPE cost = cost_NN(nn, training_input, training_output);
    printf("Final Cost: %f\n", cost);
}

#endif // NN_IMPLEMENTATION_