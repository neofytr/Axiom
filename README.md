<p align="center">
  <img src="https://img.shields.io/badge/lang-C-555?style=flat-square&logo=c&logoColor=white" alt="C">
  <img src="https://img.shields.io/badge/deps-zero-2ea44f?style=flat-square" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/inference_binary-<100KB-blue?style=flat-square" alt="Under 100KB">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT license">
  <img src="https://img.shields.io/badge/version-0.10.0-orange?style=flat-square" alt="v0.10.0">
</p>

# axiom

deep learning framework in pure C. trains and serves neural networks with no
external runtime — no python, no BLAS, no protobuf. one library, one compiler,
runs from cloud servers down to cortex-m microcontrollers.

axiom beats tensorflow on most cpu paths and matches or beats it on cuda
where cublas is the floor. headline numbers (intel i5-12500h, avx2, tf 2.21
with onednn, 5-run medians):

- mha forward (transformer attention): **+25 % to +70 %** vs tf
- raw sdpa + causal sdpa: **+66 % to +500 %**
- kv-cache attend (llm decode step): **+320 % to +500 %**
- transformer encoder block end-to-end: **+119 %**
- elementwise ops (relu, gelu, layernorm, softmax): **+40 % to +19 000 %**
- bs=1 mlp inference (skinny-m route): **4.4× faster than tf**

full table with hardware, methodology, and the one shape we still lose on
(`mha_train`, an apples-to-oranges bench) lives in
[docs/PERF_REPORT.md](docs/PERF_REPORT.md).

## quickstart

```bash
git clone https://github.com/neofytr/neoNN axiom && cd axiom
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/example_mnist          # trains the bundled mlp on mnist
```

`example_mnist` needs the idx files under `examples/data/`; see the comment
at the top of `examples/mnist.c` for the urls. `xor` and `spiral` examples
are self-contained and run with no extra data.

## install

axiom installs as a system library with cmake config + pkg-config support:

```bash
cmake --install build --prefix /usr/local
```

drops:

- `lib/libaxiom.{a,so.0.10.0}` (with versioned symlinks)
- `include/axiom/*.h` (public headers only — internal headers are excluded)
- `lib/cmake/axiom/axiomConfig.cmake` (`find_package(axiom)`)
- `lib/pkgconfig/axiom.pc` (`pkg-config --cflags --libs axiom`)

downstream cmake:

```cmake
find_package(axiom 0.10 REQUIRED)
target_link_libraries(my_app PRIVATE axiom::axiom_shared)
```

downstream make:

```makefile
CFLAGS  += $(shell pkg-config --cflags axiom)
LDFLAGS += $(shell pkg-config --libs   axiom)
```

## build modes

axiom is one source tree with several profiles. defaults are tuned for a
hosted desktop / server. flip flags or pick a profile to retarget.

| flag | default | what it does |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | Release | `-O3` + hardening flags. use `Debug` for sanitizers. |
| `-DAX_PROFILE=desktop` | desktop | full training, autotuner on, openmp on. |
| `-DAX_PROFILE=embedded-linux` | — | smaller buffers, no autotuner, no jit. |
| `-DAX_PROFILE=embedded-baremetal` | — | inference-only, no stdio, no heap, no threads. cortex-m target. |
| `-DAX_INFERENCE_ONLY=ON` | OFF | strip autograd / optimizers / losses / dataloader. <100 kb on arm. |
| `-DAX_NO_AUTOTUNE=ON` | OFF | skip ~150 ms startup tile/thread calibration. |
| `-DAX_OPENMP=ON\|OFF` | profile | thread-parallel kernels (default on for desktop, off for baremetal). |
| `-DAX_SINGLE_THREADED=ON` | OFF | hard-disable any threading runtime. |
| `-DAX_NO_STDIO=ON` | OFF | drop `stderr`/`fprintf` diagnostics. needed for baremetal. |
| `-DAX_CPU_ISA_DISPATCH=ON` | OFF | build avx2 + scalar variants, pick at runtime via `__builtin_cpu_supports`. |
| `-DAX_CUDA=ON` | OFF | nvidia gpu backend (cublas + custom kernels). needs cuda toolkit. |
| `-DAX_SANITIZE=ON` | OFF | address + ub sanitizers (debug builds). |
| `-DAX_TSAN=ON` | OFF | thread sanitizer (mutually exclusive with asan). |

extras passed via `add_compile_definitions` rather than cmake options:

- `-DAX_NO_JIT` — disable runtime kernel emission on x86_64 / aarch64. handy
  for w^x environments that forbid `mmap(PROT_EXEC)`.

build profile files live under `cmake/profiles/`. a baremetal toolchain
file recipe is in [docs/embedded.md](docs/embedded.md).

## benchmarks summary

post-phase-i suite-level numbers vs tensorflow 2.21 on the same hardware:

| suite | cases | axiom wins | median axiom advantage |
|---|---|---|---|
| gemm | 27 | 18 / 27 | up to +880 % (small skinny) |
| ops (relu, gelu, layernorm, softmax, …) | 25+ | 25 / 25 | +40 % to +19 000 % |
| mha / sdpa fwd | 5 | 4 / 5 (1 tie) | +25 % to +70 % |
| mha / sdpa raw + causal | 10 | 10 / 10 | +66 % to +500 % |
| kv cache attend (llm decode) | 5 | 5 / 5 | +320 % to +500 % |
| mha training (apples-to-oranges, see report) | 5 | 0 / 5 | tf +10 % to +32 % |
| transformer encoder block | 1 | 1 / 1 | +119 % |
| cuda gemm vs tf gpu | 14 | 14 / 14 | +7 % to +1556 % |
| cuda mha forward vs tf gpu | 5 | 5 / 5 | +17 % to +39 % |

the `mha_train` row is a methodology mismatch — tf's bench skips the
dx-through-qkv gradient via xla pruning, axiom does not. see
[docs/PERF_REPORT.md](docs/PERF_REPORT.md) for the full table, raw
latencies, hardware, and tail variance notes.

## what it gives you

- **zero deps** at runtime. one `libaxiom.{a,so}`. no python, no blas,
  no onnx runtime.
- **tensors** with arbitrary dims, views, slicing, broadcasting; reverse-
  mode autograd with a thread-local slab allocator for grad nodes.
- **layers**: dense, conv2d, batchnorm, layernorm, dropout, maxpool,
  avgpool, globalavgpool, flatten, multi-head attention.
- **activations**: relu, sigmoid, tanh, gelu, swish, leakyrelu, elu, softmax.
- **training**: mse + cross-entropy losses; sgd (momentum + nesterov),
  adam, adamw, rmsprop, adagrad; cosine / step / exponential / warmup
  lr schedules; gradient clipping; data batching + shuffling.
- **i/o**: compact `.axm` binary model format, portable across endianness.
- **cpu backends**: avx-512, avx2, neon, scalar — auto-selected at runtime
  with `AX_CPU_ISA_DISPATCH=ON`. blis-style 5-loop tiled gemm with jit-
  emitted micro-kernels and per-host tile autocalibration.
- **cuda backend**: cublas gemm with tf32 tensor cores on sm ≥ 8.0,
  custom fused softmax / sdpa / layout kernels, opt-in winograd f(2,3) for
  3×3 stride-1 conv (`AX_CUDA_WINOGRAD=1`).
- **embedded story**: `AX_INFERENCE_ONLY` + `AX_NO_AUTOTUNE` + `AX_NO_STDIO`
  → <100 kb inference binary on arm. `embedded-baremetal` profile drops
  heap / threads / stdio entirely so the library runs on cortex-m.

about 27 k loc of c. 30 test binaries (29 registered with ctest), passing
on cpu and cuda builds.

## links

- [docs/PERF_REPORT.md](docs/PERF_REPORT.md) — full perf report (cpu + cuda
  vs tensorflow, methodology, variance).
- [docs/PRODUCTION_PLAN.md](docs/PRODUCTION_PLAN.md) — roadmap to v1.0
  (api hardening, distribution, ci, docs).
- [docs/embedded.md](docs/embedded.md) — embedded deployment guide.
- [docs/index.html](docs/index.html) — html api reference (open in a
  browser; doxygen build is in flight, see PRODUCTION_PLAN N.2).
- [docs/architecture.html](docs/architecture.html) — architecture overview.
  a markdown one-pager covering backend dispatch and tensor lifecycle is
  planned (PRODUCTION_PLAN N.3).
- [examples/](examples/) — runnable: `xor.c`, `mnist.c`, `mnist_cnn.c`,
  `deep_mlp.c`, `spiral.c`.
- [include/axiom/axiom.h](include/axiom/axiom.h) — master include with
  the conventions block (error handling, ownership, thread safety, naming).
- [CHANGELOG.md](CHANGELOG.md) — release notes.

## minimal example

```c
#include "axiom/axiom.h"

int main(void) {
    ax_init();

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

## license

mit. see [LICENSE.txt](LICENSE.txt).
