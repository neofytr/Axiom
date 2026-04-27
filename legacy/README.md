# 🧠 Simple Neural Network Library in C

A lightweight, efficient neural network implementation in C supporting both gradient descent and numerical differentiation approaches for training. This library provides a flexible architecture for creating and training feed-forward neural networks with sigmoid activation functions.

## 📋 Table of Contents
- [Features](#-features)
- [Technical Details](#️-technical-details)
- [API Reference](#-api-reference)
- [Comprehensive Usage Example](#-comprehensive-usage-example)
- [Configuration](#️-configuration)
- [Installation](#-installation)
- [Performance Considerations](#-performance-considerations)
- [Math Behind the Implementation](#-math-behind-the-implementation)
- [Common Issues & Solutions](#-common-issues--solutions)
- [License](#-license)
- [Contributing](#-contributing)
- [References](#-references)
- [Visualizer](#-Visualier)

## ✨ Features

- 🚀 Feed-forward neural network implementation
- 📊 Support for both gradient descent and numerical differentiation
- 💪 Flexible network architecture configuration
- 🔧 Configurable element type (float/double)
- 📈 Sigmoid activation function
- 🧮 Matrix operations library included
- 💻 OpenMP support for parallel processing
- 🎯 Easy-to-use API
- 🔄 Batch learning support
- 📉 MSE (Mean Squared Error) cost function
- 🔍 Dynamic memory allocation
- 🛡️ Error handling and input validation

## 🛠️ Technical Details

### Neural Network Architecture

The library implements a feed-forward neural network with:
- Fully connected layers
- Sigmoid activation function: f(x) = 1/(1 + e^(-x))
- Mean Squared Error (MSE) cost function: C = (1/n)Σ(y - ŷ)²
- Support for both gradient descent and numerical differentiation training methods
- Batch processing capability
- Configurable learning rate and epsilon values
- OpenMP parallel processing support

### Core Data Structures

#### Matrix Structure
```c
typedef struct {
    size_t rows;       // Number of rows in matrix
    size_t cols;       // Number of columns in matrix
    size_t stride;     // Stride for memory access optimization
    ELEMENT_TYPE *es;  // Pointer to matrix elements
} Mat;
```

#### Neural Network Structure
```c
typedef struct {
    size_t arch_count;          // Number of layers
    size_t *arch;               // Array containing neurons per layer
    ELEMENT_TYPE *model_output; // Network output buffer
    Mat **ws;                   // Weight matrices
    Mat **bs;                   // Bias matrices
    Mat **as;                   // Activation matrices
    Mat **ds;                   // Delta matrices for backpropagation
} NN;
```

## 📚 API Reference

### Matrix Operations

#### `Mat *mat_alloc(size_t rows, size_t cols)`
Allocates a new matrix with specified dimensions.
- Parameters:
  - `rows`: Number of rows
  - `cols`: Number of columns
- Returns: Pointer to allocated matrix
- Error handling: Exits with error message if allocation fails

#### `void mat_dealloc(Mat *a)`
Deallocates a matrix.
- Parameters:
  - `a`: Pointer to matrix to deallocate
- Memory safety: Sets pointer to NULL after deallocation

#### `void mat_rand(Mat *a, ELEMENT_TYPE high, ELEMENT_TYPE low)`
Initializes matrix with random values.
- Parameters:
  - `a`: Target matrix
  - `high`: Upper bound for random values
  - `low`: Lower bound for random values
- Uses: Weight and bias initialization

#### `void mat_dot(Mat *dst, Mat *a, Mat *b)`
Matrix multiplication with OpenMP optimization.
- Parameters:
  - `dst`: Destination matrix
  - `a`: First input matrix
  - `b`: Second input matrix
- Validation: Asserts compatible dimensions

#### `void mat_sum(Mat *dst, Mat *a)`
Adds matrices element-wise.
- Parameters:
  - `dst`: Destination matrix (also first operand)
  - `a`: Matrix to add
- Validation: Asserts matching dimensions

#### `void mat_sig(Mat *a)`
Applies sigmoid activation.
- Parameters:
  - `a`: Matrix to transform
- Formula: f(x) = 1/(1 + e^(-x))

### Neural Network Operations

#### `NN *nn_alloc(size_t *arch, size_t arch_count)`
Creates neural network.
- Parameters:
  - `arch`: Array defining layer sizes
  - `arch_count`: Number of layers
- Returns: Initialized network
- Example:
  ```c
  size_t arch[] = {2, 3, 1};  // 2 inputs, 3 hidden, 1 output
  NN *nn = nn_alloc(arch, 3);
  ```

#### `void nn_delloc(NN *nn)`
Cleanup network resources.
- Parameters:
  - `nn`: Network to deallocate
- Memory safety: Handles all internal matrices

#### `void forward(NN *nn)`
Forward propagation.
- Parameters:
  - `nn`: Network to process
- Process:
  1. Matrix multiplication with weights
  2. Bias addition
  3. Sigmoid activation

#### `ELEMENT_TYPE cost_NN(NN *nn, Mat *training_input, Mat *training_output)`
Calculates MSE cost.
- Parameters:
  - `nn`: Network
  - `training_input`: Input data
  - `training_output`: Expected output
- Returns: Cost value
- Formula: C = (1/n)Σ(y - ŷ)²

## 🚀 Comprehensive Usage Example

Here's a detailed example implementing binary addition without carry. This example demonstrates the full capabilities of the library:

```c
#define NN_IMPLEMENTATION_  // Enable implementation
#define MODEL GRAD_DESC     // Use gradient descent (alternative: DIFF)
#define ELEMENT_TYPE float  // Use float precision
#include "nn.h"

// Training data for 2-bit binary addition
ELEMENT_TYPE td[] = {
    // Format: A1, A0, B1, B0, S2, S1, S0
    // Input bits: A1A0 (first number) and B1B0 (second number)
    // Output bits: S2S1S0 (sum without carry)
    0, 0, 0, 0, 0, 0, 0, // 00 + 00 = 00
    0, 0, 0, 1, 0, 0, 1, // 00 + 01 = 01
    0, 0, 1, 0, 0, 1, 0, // 00 + 10 = 10
    0, 0, 1, 1, 0, 1, 1, // 00 + 11 = 11
    0, 1, 0, 0, 0, 0, 1, // 01 + 00 = 01
    0, 1, 0, 1, 0, 1, 0, // 01 + 01 = 10
    0, 1, 1, 0, 0, 1, 1, // 01 + 10 = 11
    0, 1, 1, 1, 1, 0, 0, // 01 + 11 = 00 (no carry considered, just the lower 2 bits)
    1, 0, 0, 0, 0, 1, 0, // 10 + 00 = 10
    1, 0, 0, 1, 0, 1, 1, // 10 + 01 = 11
    1, 0, 1, 0, 1, 0, 0, // 10 + 10 = 00 (no carry considered)
    1, 0, 1, 1, 0, 0, 1, // 10 + 11 = 01
    1, 1, 0, 0, 0, 1, 1, // 11 + 00 = 11
    1, 1, 0, 1, 1, 0, 0, // 11 + 01 = 00 (no carry considered)
    1, 1, 1, 0, 0, 0, 1, // 11 + 10 = 01
    1, 1, 1, 1, 1, 1, 0  // 11 + 11 = 10 (no carry considered) 
};

int main() {
    // 1. Define network architecture
    size_t arch[] = {4, 4, 3};  // 4 inputs, 4 hidden neurons, 3 outputs
    
    // 2. Create training data matrices
    Mat *training_input = &(Mat){
        .rows = 16,     // 16 training examples
        .cols = 4,      // 4 input bits
        .stride = 7,    // Skip to next training example
        .es = training_data
    };
    
    Mat *training_output = &(Mat){
        .rows = 16,     // 16 training examples
        .cols = 3,      // 3 output bits
        .stride = 7,    // Skip to next training example
        .es = &training_data[4]  // Point to output part of data
    };
    
    // 3. Create and initialize neural network
    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    randomize_parameters_NN(nn, 0, 0);  // Initialize weights/biases
    
    // 4. Train the network
    learn(nn, 
        1e-1,       // epsilon (for numerical differentiation)
        1e-1,       // learning rate
        1000000,    // iterations
        training_input, 
        training_output
    );
    
    // 5. Test the network
    for (size_t a1 = 0; a1 < 2; a1++) {
        for (size_t a0 = 0; a0 < 2; a0++) {
            for (size_t b1 = 0; b1 < 2; b1++) {
                for (size_t b0 = 0; b0 < 2; b0++) {
                    // Create input
                    ELEMENT_TYPE input[] = {
                        (ELEMENT_TYPE)a1,
                        (ELEMENT_TYPE)a0,
                        (ELEMENT_TYPE)b1,
                        (ELEMENT_TYPE)b0
                    };
                    
                    // Set input and forward propagate
                    Mat input_mat = {
                        .rows = 1,
                        .cols = 4,
                        .es = input
                    };
                    *(NN_INPUT(nn)) = input_mat;
                    forward_NN(nn);
                    
                    // Print results
                    printf("%zu%zu + %zu%zu = %f%f%f\n",
                        a1, a0, b1, b0,
                        NN_OUTPUT(nn)->es[0],
                        NN_OUTPUT(nn)->es[1],
                        NN_OUTPUT(nn)->es[2]
                    );
                }
            }
        }
    }
    
    return EXIT_SUCCESS;
}
```

### Explanation of the Example

1. **Configuration**
   - Uses gradient descent (`MODEL GRAD_DESC`)
   - Uses float precision (`ELEMENT_TYPE float`)
   - Implements binary addition without carry

2. **Network Architecture**
   - Input layer: 4 neurons (A1, A0, B1, B0)
   - Hidden layer: 4 neurons
   - Output layer: 3 neurons (S2, S1, S0)

3. **Training Data**
   - 16 examples covering all possible 2-bit additions
   - Each example has 7 values (4 input bits, 3 output bits)
   - Data arranged for efficient stride-based access

4. **Training Process**
   - 1,000,000 iterations
   - Learning rate: 0.1
   - Epsilon: 0.1 (for numerical differentiation if used)

5. **Testing**
   - Tests all possible input combinations
   - Prints binary addition results
   - Output values are sigmoid-activated (between 0 and 1)

## ⚙️ Configuration Options

```c
// Training method selection
#define GRAD_DESC 100  // Use gradient descent
#define DIFF 200       // Use numerical differentiation
#define MODEL GRAD_DESC

// Learning parameters
#define DECAY_RATE 1e-1

// Data type configuration
#define ELEMENT_TYPE float  // or double

// Memory allocation
#define NN_MALLOC malloc
#define NN_ASSERT assert

// Random initialization range
#define MAX_RAND_ELEMENT 1.f
#define MIN_RAND_ELEMENT 0.f
```

## 🧮 Math Behind the Implementation

### Forward Propagation
1. **Layer Activation**
   ```
   z[l] = w[l]a[l-1] + b[l]
   a[l] = σ(z[l])
   ```
   where σ is the sigmoid function

### Gradient Descent
1. **Cost Function**
   ```
   C = (1/n)Σ(y - ŷ)²
   ```

2. **Backpropagation**
   ```
   δ[L] = ∇C ⊙ σ'(z[L])
   δ[l] = ((w[l+1])^T δ[l+1]) ⊙ σ'(z[l])
   ```

### Numerical Differentiation
```
∂C/∂w ≈ (C(w + ε) - C(w)) / ε
```

## 🔍 Common Issues & Solutions

1. **Vanishing Gradients**
   - Symptom: Training stalls
   - Solution: Adjust learning rate or use numerical differentiation

2. **Exploding Gradients**
   - Symptom: NaN values
   - Solution: Reduce learning rate, adjust weight initialization

3. **Poor Convergence**
   - Symptom: High final cost
   - Solution: Increase network size or training iterations

## 🔧 Installation

1. Copy `nn.h` to your project
2. Include with implementation:
```c
#define NN_IMPLEMENTATION_
#include "nn.h"
```

## 🎯 Performance Considerations

1. **OpenMP Parallelization**
   - Matrix operations are parallelized
   - Enable with `-fopenmp` compiler flag

2. **Memory Access**
   - Stride-based matrix access for better cache usage
   - Efficient memory allocation patterns

3. **Data Type Selection**
   - `float`: Faster, less precise
   - `double`: Slower, more precise

# Visualizer

The visualizer is designed to provide a graphical representation of your neural network during the training process. It allows you to see the architecture of the network, the weights of connections between neurons, and the cost over iterations.

## Features:
- **Neural Network Visualization**: Displays the structure of the neural network, including neurons and their connections. The color of the connections changes based on the weight, transitioning from green (low weight) to pink (high weight).
- **Training Cost Graph**: Shows the cost (loss) over iterations in a separate window, helping you to track the training progress visually.

## How to Use:
1. **Initialization**: Create an instance of the visualizer before starting the training.
2. **Draw the Network**: After each training iteration, call the function to redraw the network with updated weights and activations.
3. **Update Cost Graph**: Continuously update the cost graph with the current training cost after each iteration.
4. **Run the Training Loop**: Use your existing training loop to train the model while updating the visualizer.

## Example Usage:
```c
Visualizer *vis = init_visualizer();
learn_with_visualization(nn, eps, learning_rate, iterations, training_input, training_output, vis);
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

Guidelines:
- Follow existing code style
- Add tests for new features
- Update documentation
- Maintain thread safety