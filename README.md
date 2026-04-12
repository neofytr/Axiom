<p align="center">
  <img src="https://img.shields.io/badge/lang-C-555?style=flat-square&logo=c&logoColor=white" alt="C">
  <img src="https://img.shields.io/badge/deps-zero-2ea44f?style=flat-square" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/inference_binary-<100KB-blue?style=flat-square" alt="Under 100KB">
</p>

# Axiom

Deep learning framework in C. Trains neural networks from scratch with no external dependencies, no Python runtime, no BLAS library. Written to run on everything from cloud servers to microcontrollers.

It's also faster than TensorFlow on CPU. Not by gaming benchmarks or picking favorable shapes, but on real training workloads head to head on the same hardware.

## Benchmarks

All numbers from GitHub Actions CI. Same model, same dataset, same thread count, same machine.

**Training (MNIST, 60K samples, Adam optimizer, batch 256)**

| | x86 (EPYC, 4 threads) | ARM (Graviton3, 4 threads) |
|---|---|---|
| **12.6M params** | Axiom 69s / TF 87s | |
| **4.2M params** | Axiom 34s / TF 36s | Axiom 26s / TF 30s |
| **1.46M params** | Axiom 16s / TF 17s | Axiom 16s / TF 20s |

**Inference (1.46M params, batch 256, 1000 forward passes)**

| | Axiom | TensorFlow |
|---|---|---|
| Throughput | **51,226 img/s** | 22,185 img/s |
| Per batch | 4.99ms | 11.54ms |

2.3x faster on inference. The gap comes from TF's Python dispatch overhead and graph executor, which matter more when the compute per op is small.

## What's in here

Everything you need to train and deploy small-to-medium neural nets.

**The basics**: tensors with arbitrary dims, views, slicing, broadcasting. Reverse-mode autograd that records a computation graph and backpropagates through it.

**Layers**: Dense, Conv2D, BatchNorm, LayerNorm, Dropout, MaxPool, AvgPool, GlobalAvgPool, Flatten.

**Activations**: ReLU, Sigmoid, Tanh, GELU, Swish, LeakyReLU, ELU, Softmax.

**Training**: MSE and cross-entropy losses. SGD (momentum + nesterov), Adam, AdamW, RMSprop, Adagrad. Cosine annealing, step decay, exponential, and warmup LR schedules. Gradient clipping. Data batching and shuffling.

**I/O**: Model save/load in a compact binary format. Works across platforms.

**GPU**: CUDA backend with cuBLAS GEMM, fused element-wise kernels, explicit device memory management.

## Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest --output-on-failure
```

That's it. No package manager, no `pip install`, no downloading pretrained weights. The compiler is the only dependency.

### Build options

| Flag | What it does |
|------|-------------|
| `-DAX_CPU_ISA_DISPATCH=ON` | Builds AVX-512 + AVX2 + scalar, picks fastest at runtime |
| `-DAX_CUDA=ON` | Enables GPU backend (needs CUDA toolkit) |
| `-DAX_INFERENCE_ONLY=ON` | Strips all training code. Binary under 100KB on ARM |
| `-DAX_PROFILE=embedded-linux` | Smaller buffers, trimmed for embedded Linux targets |
| `-DAX_PROFILE=embedded-baremetal` | No stdio, no heap, no threads. Runs on Cortex-M |

## Example

```c
#include "axiom/axiom.h"

int main(void) {
    ax_init();

    // 784 -> 128 -> 10 classifier
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(784, 128, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(128, 10, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(
        m->params, m->n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_cross_entropy_loss);

    for (int i = 0; i < 1000; i++)
        ax_model_train_step(m, train_x, train_y);

    ax_model_save(m, "model.axm");
    ax_model_destroy(m);
    ax_shutdown();
}
```

There are more complete examples in `examples/` (XOR, MNIST MLP, MNIST CNN, deep MLP with batchnorm + dropout + LR scheduling).

## How it's fast

The short version: Axiom does the same math as TensorFlow but with less stuff between you and the hardware.

The longer version:

**GEMM kernels** are the heart of it. Axiom uses BLIS-style tiled matrix multiplication with micro-kernels sized to the register file: 14x32 on AVX-512 (uses all 32 ZMM registers), 6x16 on AVX2, 8x12 on NEON. The tiles are auto-tuned to the CPU's L1 and L2 cache sizes at startup.

**Op fusion** eliminates memory round-trips. Dense + ReLU compiles into a single kernel call where the activation runs while the matmul output is still in cache. The cross-entropy backward pass computes `softmax - target` in one SIMD pass instead of materializing the full softmax Jacobian.

**Autograd overhead is minimal.** Grad nodes come from a thread-local slab allocator (no malloc per op). Fresh gradients skip the accumulate step and write directly. Lazy zero_grad skips the memset entirely and just marks the generation counter.

**Threading is careful.** Axiom detects hyperthreading siblings and pins workers to physical cores. The JC tile loop auto-shrinks when there aren't enough tiles to fill all threads. Pack buffers are per-thread with no lock contention.

None of this is microarchitecture-specific. No hand-written assembly, no Zen4-only codepaths. The same binary runs on any x86 with AVX2, any aarch64 with NEON, or falls back to scalar C.

## Project layout

```
include/axiom/     24 public headers
src/core/          tensor, autograd, ops, layers, optimizers, losses
src/compute/       dispatch + backends (cpu_naive, cpu_opt, cuda/)
tests/             22 test binaries covering everything
examples/          xor, mnist, mnist_cnn, deep_mlp
benchmarks/        perf comparisons, TF baselines
docs/              embedded deployment guide
```

About 19K lines of C for the whole framework.

## Testing

22 test binaries. CI runs on x86 (AVX-512 Xeon), ARM (Graviton3 NEON), plus ASan and inference-only builds. The test suite covers tensor math, autograd correctness, every activation and optimizer, SIMD-vs-naive cross-checks, gradient numerical verification, models up to 12.6M parameters, odd-dimension edge cases, memory pressure under repeated train/cleanup cycles, and model serialization roundtrips.

## Why not just use PyTorch

You should, probably. PyTorch and TensorFlow are mature, battle-tested, and have enormous ecosystems.

Axiom exists for the cases where they don't fit:

- You need inference on a microcontroller with 256KB of flash and no OS
- You want to understand exactly what your training loop does, down to the SIMD instruction
- You need a self-contained C library with zero transitive dependencies
- You're building something where a 500MB Python runtime isn't an option

If none of those apply, the mainstream frameworks will serve you better.

## Roadmap

- INT8 quantization with calibration
- Attention / transformer layers
- ONNX model import
- Multi-GPU data parallelism
- Depthwise separable and dilated convolutions

## License

MIT
