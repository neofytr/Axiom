# Axiom Internals: Deep Learning from Scratch

This document teaches you deep learning — the math, the intuition, and the
implementation — from zero. It's structured as a course. Each unit builds
on the last. By the end, you'll understand every line of code in this
project and the theory behind it.

This is a living document. New units are added as the project grows.


## Table of Contents

### Part 1: Foundations

| Unit | Topic | File |
|------|-------|------|
| 1 | Vectors, Matrices, and Tensors | [lessons/01_vectors_matrices_tensors.md](lessons/01_vectors_matrices_tensors.md) |
| 2 | Calculus for Deep Learning | [lessons/02_calculus.md](lessons/02_calculus.md) |
| 3 | The Neuron | [lessons/03_the_neuron.md](lessons/03_the_neuron.md) |
| 4 | Loss Functions — Measuring Error | [lessons/04_loss_functions.md](lessons/04_loss_functions.md) |
| 5 | Gradient Descent and Backpropagation | [lessons/05_gradient_descent_backprop.md](lessons/05_gradient_descent_backprop.md) |
| 6 | Optimizers | [lessons/06_optimizers.md](lessons/06_optimizers.md) |
| 7 | Weight Initialization | [lessons/07_weight_initialization.md](lessons/07_weight_initialization.md) |
| 8 | The Layer System and Model API | [lessons/08_layers_and_models.md](lessons/08_layers_and_models.md) |
| 9 | Code Map | [lessons/09_code_map.md](lessons/09_code_map.md) |

> **Note:** The `lessons/` directory is gitignored and exists only locally.


## Coming Next

- **Unit 10**: Serialization — saving and loading models for deployment
- **Unit 11**: Convolutions — what they are, why they work for images, the im2col trick
- **Unit 12**: Batch normalization — stabilizing deep network training
- **Unit 13**: Dropout — regularization by random disabling
- **Unit 14**: Recurrent networks — processing sequences, LSTM gates
- **Unit 15**: Attention and transformers — the mechanism behind modern AI
- **Unit 16**: SIMD optimization — processing 8 numbers at once
- **Unit 17**: Quantization — INT8 inference for embedded deployment
