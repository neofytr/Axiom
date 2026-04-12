# Axiom

A deep learning framework written from scratch in C. No Python runtime, no BLAS dependency, no framework overhead. Just tensors, autograd, and optimized compute kernels that run on CPUs and GPUs.

Axiom started as a question: how fast can a neural network train if you strip away every abstraction layer? The answer, it turns out, is faster than TensorFlow on the same hardware.

## What it does

Axiom trains and runs neural networks. It has everything you need for the full pipeline:

- **Tensors** with arbitrary dimensions, views, slicing, and broadcasting
- **Reverse-mode autograd** that records a computation graph and walks it backwards to compute gradients
- **Layers**: Dense, Conv2D, BatchNorm, LayerNorm, Dropout, MaxPool, AvgPool, GlobalAvgPool, Flatten
- **Activations**: ReLU, Sigmoid, Tanh, GELU, Swish, LeakyReLU, ELU, Softmax
- **Losses**: MSE, Cross-Entropy
- **Optimizers**: SGD (with momentum/nesterov), Adam, AdamW, RMSprop, Adagrad
- **LR schedulers**: Step decay, Exponential, Cosine annealing, Warmup+Cosine
- **Model save/load** in a compact binary format
- **Data loading** with batching, shuffling, and train/test splits
- **CUDA GPU backend** with cuBLAS GEMM, fused element-wise kernels, and explicit device memory

## Performance

Axiom is benchmarked against TensorFlow (with oneDNN/MKL) on identical hardware in CI. These numbers are from GitHub Actions runners (no cherry-picking):

**x86_64 (AMD EPYC, 4 threads):**

| Model | Params | Axiom | TensorFlow | Speedup |
|-------|--------|-------|------------|---------|
| 784->1024->512->256->10 | 1.46M | 4.13s | 4.36s | 1.05x |
| 784->2048->1024->512->10 | 4.2M | 11.88s | 11.98s | 1.01x |
| 784->4096->2048->512->10 | 12.6M | 35.0s | 42.4s | **1.21x** |

**ARM (Graviton3 Neoverse-N2, 4 threads):**

| Model | Params | Axiom | TensorFlow | Speedup |
|-------|--------|-------|------------|---------|
| 784->1024->512->256->10 | 1.46M | 3.10s | 3.76s | **1.21x** |

The advantage grows with model size because Axiom's tiled GEMM kernels scale better with large matrices.

## How it's fast

No magic. Just doing the obvious things well:

- **BLIS-style tiled GEMM** with MR x NR micro-kernels: 6x16 (AVX2), 14x32 (AVX-512), 8x12 (NEON)
- **Fused ops**: Dense+ReLU executes as one kernel call. The output stays cache-hot through both the matmul and activation
- **Direct-write backward**: when a gradient tensor is fresh (never accumulated into), the backward pass writes directly instead of allocating a temp and adding
- **Adaptive JC tiling**: auto-shrinks GEMM tile width when there aren't enough tiles to keep all threads busy
- **Runtime ISA dispatch**: builds AVX-512, AVX2, and scalar variants, picks the best at startup
- **Cache auto-tuning**: reads L1d and L2 sizes from the OS and adjusts tile dimensions to fit
- **HT dedup**: detects hyperthreading siblings and pins threads to physical cores
- **Pack-b caching**: reuses the transposed weight panel across batches (generation counter invalidation)

All of this is general-purpose. No microarchitecture-specific tuning, no hand-written assembly. The same binary runs on Zen3, Ice Lake, Graviton, or a Raspberry Pi.

## Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### Build profiles

| Profile | Flag | What it does |
|---------|------|-------------|
| Desktop (default) | none | Full framework, OpenMP, all features |
| Embedded Linux | `-DAX_PROFILE=embedded-linux` | Strips inference-only, smaller buffers |
| Baremetal | `-DAX_PROFILE=embedded-baremetal` | No stdio, no threads, no dynamic alloc |
| Inference only | `-DAX_INFERENCE_ONLY=ON` | Removes autograd, optimizers, training code |
| ISA dispatch | `-DAX_CPU_ISA_DISPATCH=ON` | Builds AVX-512 + AVX2 + scalar, picks at runtime |
| CUDA | `-DAX_CUDA=ON` | Enables GPU backend (needs CUDA toolkit) |

## Quick example

```c
#include "axiom/axiom.h"

int main(void) {
    ax_init();

    // build a model
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(784, 128, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(128, 10, true));

    ax_model_t *model = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(
        model->params, model->n_params,
        1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(model, opt, ax_cross_entropy_loss);

    // train
    for (int i = 0; i < 1000; i++)
        ax_model_train_step(model, train_x, train_y);

    // save
    ax_model_save(model, "model.axm");

    ax_model_destroy(model);
    ax_shutdown();
}
```

See `examples/` for complete working programs (XOR, MNIST MLP, MNIST CNN, deep MLP).

## Project structure

```
include/axiom/    public headers (tensor.h, autograd.h, layer.h, ...)
src/core/         tensor, autograd, ops, layers, optimizers, losses
src/compute/      dispatch layer + backend implementations
  backends/       cpu_naive.c, cpu_opt.c (SIMD kernels), simd_defs.h, cuda/
tests/            22 test binaries, ~5700 lines of tests
examples/         training examples (xor, mnist, deep mlp)
benchmarks/       performance comparisons vs tensorflow
```

## What's different from TensorFlow/PyTorch

Axiom is not trying to replace TensorFlow. It's a different tool for different constraints:

- **Zero dependencies**. No Python, no BLAS, no protobuf, no build system gymnastics. One C library, one header directory.
- **Embedded targets**. The baremetal profile compiles for Cortex-M with no heap, no stdio, no threads. You can run inference on a microcontroller.
- **Predictable performance**. No JIT, no graph optimization passes, no "sometimes fast sometimes slow". The same code path runs every time.
- **Small binary**. The full framework is ~19K lines of C. A static build with inference-only strips to under 100KB on ARM.

The tradeoff is obvious: TensorFlow has thousands of ops, distributed training, TPU support, a massive ecosystem. Axiom has a handful of layer types and trains on one machine. If you need ResNet-152 on a TPU pod, use TensorFlow. If you need a 100KB inference engine on an STM32, or you want to understand exactly what your training loop is doing, Axiom is the tool.

## Testing

22 test binaries covering:
- Tensor ops, broadcasting, views, slicing
- Every activation forward + backward
- Every optimizer converges on XOR
- SIMD kernel correctness (cross-checked against naive backend)
- Conv2D, BatchNorm, LayerNorm, Dropout gradient flow
- Large model stress (270K-12.6M params)
- Fused Dense+ReLU gradient correctness
- No-grad nesting, memory pressure, odd-dimension edge cases
- Model serialization roundtrip
- Numerical gradient verification (finite differences vs analytical)

CI runs on x86_64 (with AVX-512), ARM64 (with NEON), and includes ASan + inference-only builds.

## Roadmap

- [ ] Quantization (INT8 inference with calibration)
- [ ] Attention / transformer layers
- [ ] ONNX import (run models trained elsewhere)
- [ ] Multi-GPU data parallelism
- [ ] More conv variants (depthwise separable, dilated)
- [ ] RISC-V vector extension support

## License

MIT
