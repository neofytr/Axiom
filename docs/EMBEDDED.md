# axiom on cortex-m and similar embedded targets

a cookbook for getting axiom inference onto a microcontroller. covers
the build flags, the runtime contract you owe the framework, a worked
sketch of a baremetal inference loop, footprint measurement, and the
gotchas that bit the early adopters.

if you're targeting a cortex-a sbc running linux (raspberry pi, jetson,
i.mx 8), you want `docs/embedded.md` (lowercase) — that doc covers the
hosted-linux cross-compile path. this doc is for the harder case:
baremetal, freertos, zephyr, nuttx, or any environment without a posix
runtime.


## 1. what axiom gives you on embedded

| property | value |
| --- | --- |
| static archive | `libaxiom.a`, ~80–100 kb inference-only after `--gc-sections` |
| dependencies | none. links against libm + your libc, that's it. |
| dynamic allocation | one-shot at model load + first gemm; storage pool recycles after that. |
| weight quantization | int8 per-channel symmetric (W8A32). 4× smaller weights than fp32. |
| training on device | not supported. train on host, deploy a frozen model. |
| stdio | compiled out entirely under `AX_NO_STDIO`. no `printf`, `fopen`, `fwrite` symbols leak into the firmware. |
| threading | single-threaded under the embedded profile. no openmp, no tls outside the user's libc. |

axiom occupies the niche between hand-written kernels and a full
training framework. tflite micro is comparable on inference footprint
(~50 kb library) but provides no training path; you'd train in tf
python and convert. axiom lets you train with the same C api on the
host (`AX_PROFILE=desktop`), then static-link the inference subset
into the firmware. one repo, two profiles, identical math.

what the inference build does **not** cost you in flash:

- autograd graph walker (~28 kb under -O3)
- adam / sgd / sgd-momentum optimizers (~12 kb)
- cross-entropy / mse / nll loss kernels (~8 kb)
- dataloader, dataset, lr scheduler (~14 kb)
- model_save / model_load file i/o paths (~6 kb)
- benchmark harness, autotuner, calibration probes (~9 kb)

these are excluded from the archive itself, not just gc'd from the
binary. trying to call them from user code gets a clean compile error
at the include site (the symbols are gated by `#ifndef AX_INFERENCE_ONLY`
in `axiom/axiom.h`) rather than a buried linker error two minutes into
the build.


## 2. building for cortex-m / arm-none-eabi

install the toolchain.

```
sudo apt install gcc-arm-none-eabi
```

or grab the latest from
<https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads>.

axiom ships a starter toolchain file at
`cmake/toolchains/arm-none-eabi.cmake`. it defaults to cortex-m4f with
hardware float; edit one line for your part. the common choices are
listed in the file's comment block. minimal configure:

```
cmake -B build-stm32 \
      -DAX_PROFILE=embedded-baremetal \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake \
      -DCMAKE_C_FLAGS="-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb"
cmake --build build-stm32 --target axiom_static
```

the resulting `build-stm32/libaxiom.a` is what you link into your
firmware. the test and example targets are skipped automatically —
they assume a hosted libc and won't compile under arm-none-eabi.

what each flag does, if you want to tune piecewise instead of using the
profile:

| flag | what it strips |
| --- | --- |
| `AX_OPENMP=OFF` | every `#pragma omp` expands to nothing. no libgomp link dep. |
| `AX_SINGLE_THREADED=ON` | `_Thread_local` collapses to plain `static`. required for toolchains without tls (newlib-nano, some rtos). |
| `AX_NO_AUTOTUNE=ON` | `ax_autotune_threads()` becomes `return 1`. no `sched_setaffinity`, no `sysconf`. |
| `AX_NO_STDIO=ON` | every `fprintf(stderr, ...)` diagnostic compiles out. |
| `AX_INFERENCE_ONLY=ON` | autograd, optim, losses, lr_scheduler, data, serialize i/o all dropped from the build. |

`AX_PROFILE=embedded-baremetal` flips all five at once and shrinks the
default gemm tile sizes. you can override any individual one — the
profile sets cache defaults, your `-D` overrides win.

### tile sizes

the baremetal profile sets:

```
AX_GEMM_DEFAULT_MC=24
AX_GEMM_DEFAULT_NC=32
AX_GEMM_DEFAULT_KC=64
```

pack_a is `MC*KC*4 = 6 kb`, pack_b is `NC*KC*4 = 8 kb`. fits a 64 kb
sram with room for stack, model, and your scratch buffers. if your
target has 256 kb sram or more, bump them up — the rule of thumb is
`KC * NC * sizeof(float) ≈ half your data cache`. example for an mcu
with 192 kb of l1 d-cache:

```
cmake -B build-stm32 -DAX_PROFILE=embedded-baremetal \
      -DCMAKE_C_FLAGS="-mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard \
                       -DAX_GEMM_DEFAULT_MC=48 \
                       -DAX_GEMM_DEFAULT_NC=64 \
                       -DAX_GEMM_DEFAULT_KC=128"
```

these are compile-time only on baremetal. the env-var overrides
(`AX_GEMM_MC=...`) work on hosted builds but require `getenv`, which
the embedded profile assumes you don't have.

### writing your own toolchain file

if your target isn't covered, copy `cmake/toolchains/arm-none-eabi.cmake`
and adjust. the bare minimum is six lines:

```cmake
# cmake/toolchains/my-target.cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER my-cc)
set(CMAKE_C_FLAGS_INIT "-mcpu=... -mfpu=... -mfloat-abi=hard -mthumb")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
```

the `STATIC_LIBRARY` target type for `try_compile` is what lets cmake
probe the compiler without trying to link a full executable — bare
metal toolchains don't have a default crt0 on the search path and the
default exe-link probe fails.


## 3. what gets included / excluded under inference-only

**included** (compiled into `libaxiom.a`):

- `tensor.c`, `memory.c`, `error.c`, `init.c` — core data structures
- `ops.c`, `activations.c`, `broadcast.c` — elementwise + reductions
- `layer.c`, `model.c` — sequential model + dense / activation layers
- `conv/` — conv2d forward (im2col, direct, winograd selection)
- `norm.c` — batchnorm + layernorm forward
- `attention.c`, `attention/`, `fused_attention.c` — mha forward + kv cache
- `quantize.c` — int8 quantized gemm (W8A32)
- `dispatch.c`, `cpu_naive.c`, `cpu_opt.c` — backend dispatch + simd kernels
- `serialize.c` — model file format (read path; write path is i/o gated)
- `rng.c` — seedable rng (used by test scaffolding; not pulled by inference unless called)

**excluded** (training code, dropped from the source list under
`AX_INFERENCE_ONLY=ON`):

- `autograd.c`, `autograd_ops.c` — graph walker + grad fns
- `losses.c` — mse, ce, nll, kld
- `optim.c` — sgd, sgd-momentum, adam, adamw
- `data.c` — dataset / dataloader / batcher
- `lr_scheduler.c` — step, cosine, exp decay schedules

the public umbrella header `axiom/axiom.h` skips the corresponding
`#include` lines under `AX_INFERENCE_ONLY`, so calling `ax_mse_loss`
or `ax_adam_create` from your firmware code won't even parse — you
get a "declared nowhere" error at the call site, not a link-time
"undefined reference".

`stdio` symbols deserve a callout. CI runs `nm libaxiom.a | grep -E ' U
(printf|fprintf|fopen|fwrite)'` after every embedded build and fails the
job on any match. if your firmware suddenly grows by ~20 kb after a
library bump, the stdio-leak check is the first thing to look at.

`x86`-only intrinsics get the same treatment via
`scripts/check_portability.py`: any `__builtin_ia32_*`, `_mm_*`,
`_mm256_*`, `_mm512_*`, `__rdtsc` outside an `#if defined(__x86_64__)`
guard fails the lint job. the embedded build won't see x86 intrinsics
at all because the jit_x64 / jit_gemm_avx2 / jit_gemm_avx512 sources
are conditionally added to the cmake source list only on x86 hosts.


## 4. runtime contract — what the caller owes the framework

axiom asks for very little. all of it is one-time setup at boot.

1. **a heap.** axiom calls `ax_aligned_alloc` for tensor data buffers
   and the gemm pack scratch. on hosted builds this resolves to
   `posix_memalign` or `aligned_alloc` from your libc. on baremetal
   you need to provide one of these symbols — most rtos ports do; if
   yours doesn't, write a five-line wrapper around your heap allocator
   (newlib's `_sbrk`, freertos `pvPortMalloc`, etc.):

   ```c
   /* example: bridge axiom -> freertos heap_4 */
   void *aligned_alloc(size_t alignment, size_t size) {
       /* freertos heap_4 already 8-byte aligned; round up size for >8 */
       size_t pad = alignment > 8 ? alignment - 1 : 0;
       void *raw = pvPortMalloc(size + pad + sizeof(void *));
       if (!raw) return NULL;
       uintptr_t aligned = ((uintptr_t)raw + sizeof(void *) + pad)
                         & ~(uintptr_t)(alignment - 1);
       ((void **)aligned)[-1] = raw;
       return (void *)aligned;
   }
   void free(void *p) {
       if (p) vPortFree(((void **)p)[-1]);
   }
   ```

   axiom never calls `malloc` or `free` directly — only the aligned
   variants. heap fragmentation is bounded by the storage pool (see
   point 6 below).

2. **stack space.** the gemm pack buffers live on the heap, not the
   stack, so the per-call stack draw is small (single-digit kb for
   most layers). but several internal kernels stack-allocate small
   tiles up to ~4 kb. budget at least **16 kb** of stack for whichever
   thread runs `ax_layer_forward`. if you run inference from an rtos
   task, make sure that task's stack is sized accordingly.

3. **`ax_compute_init()` exactly once at boot.** before any tensor
   operation. a single call:

   ```c
   #include <axiom/axiom.h>
   void app_main(void) {
       ax_init();           /* wraps ax_compute_init */
       /* ... model load + inference loop ... */
   }
   ```

   on baremetal `ax_init` does very little — registers the cpu_opt
   vtable, calls `ax_cpu_opt_tune_init` to size the pack buffers from
   the compile-time `AX_GEMM_DEFAULT_*` macros, and returns. no probe,
   no calibration, no thread spawn. the autotune / calibration paths
   are compiled out by `AX_NO_AUTOTUNE=ON`.

4. **error callback (optional).** axiom signals failures via a
   per-thread error tls. if you want a sink for them — to log to your
   uart, set a fault led, etc. — register a callback:

   ```c
   static void on_axiom_error(ax_status_t s, const char *msg, void *ud) {
       (void)ud;
       my_uart_printf("axiom: %s: %s\n", ax_status_name(s), msg);
   }
   ax_err_set_callback(on_axiom_error, NULL);
   ```

   without a callback you can still poll `ax_err_last_status()` and
   `ax_err_last_message()` after every call that returns `NULL` or
   non-`AX_OK`. on baremetal the error message string lives in
   thread-local storage (or plain `static` if `AX_SINGLE_THREADED=ON`)
   so you don't need to dup it before the next axiom call — but the
   buffer is reused, so log it before the next failure.


## 5. worked example — quantized linear inference on baremetal

a sketch of the smallest possible inference loop: load a quantized
weight blob from flash, multiply by an input vector, write a single
prediction back through the user's uart. real firmware would chain
several layers and apply an activation; the math primitive is the same.

```c
/* main.c — baremetal inference sketch.
   assumes the firmware has linked libaxiom.a (built with
   AX_PROFILE=embedded-baremetal) and provides aligned_alloc/free. */

#include <axiom/axiom.h>
#include <axiom/quantize.h>
#include <stdint.h>

/* the user's serial sink — you provide this from the bsp. */
extern void uart_write_f32(float v);
extern void uart_write_str(const char *s);

/* the model: 64-input -> 32-output dense layer with int8 weights.
   weights and scales were extracted on the training host and burned
   into the firmware image as a const blob. order: int8 data row-major
   [N, K], then fp32 scales [N]. */
extern const int8_t model_qw_data[];     /* N*K = 32*64 = 2048 bytes */
extern const float  model_qw_scales[];   /* N = 32 floats */
#define MODEL_N 32
#define MODEL_K 64

/* input vector — populated from sensors; here we just fake one. */
static float input[MODEL_K];
static float output[MODEL_N];

static void on_axiom_error(ax_status_t s, const char *msg, void *ud) {
    (void)ud;
    uart_write_str("axiom err: ");
    uart_write_str(ax_status_name(s));
    uart_write_str(": ");
    uart_write_str(msg);
    uart_write_str("\n");
}

void app_main(void) {
    /* one-time setup. ax_init wraps ax_compute_init. */
    if (ax_init() != AX_OK) {
        uart_write_str("axiom init failed\n");
        return;
    }
    ax_err_set_callback(on_axiom_error, NULL);

    /* wrap the flash-resident weights in an ax_qweight_t. we don't
       use ax_qweight_create_from_fp32 — that allocates and fills from
       fp32 source. instead build the struct in stack scratch with
       borrowed pointers; do NOT call ax_qweight_destroy on it. */
    ax_qweight_t qw = {
        .data   = (int8_t *)model_qw_data,
        .scales = (float  *)model_qw_scales,
        .N      = MODEL_N,
        .K      = MODEL_K,
    };

    /* main loop */
    for (;;) {
        /* fill input from your sensor / dma buffer */
        for (int i = 0; i < MODEL_K; i++) input[i] = (float)i * 0.01f;

        /* M=1 batch (one input vector). out = input @ qw_data^T * scales. */
        ax_status_t s = ax_qgemm_w8a32(input, &qw, output, /*M=*/1);
        if (s != AX_OK) continue;  /* error already reported via callback */

        /* argmax of output for a classification head */
        int best = 0;
        for (int i = 1; i < MODEL_N; i++)
            if (output[i] > output[best]) best = i;

        uart_write_str("class = ");
        uart_write_f32((float)best);
        uart_write_str("\n");

        /* sleep / wait for next sensor frame here */
    }
}
```

three things to notice:

- no `printf`, no `fopen`, no `getenv`. all stdio is dispatched through
  the user's uart sink.
- `ax_qweight_t` is plain c struct with three pointers; you can build
  it on the stack with borrowed pointers into flash-resident data.
  no malloc on the hot path.
- the inference loop is allocation-free after the first call. the
  storage pool warms up on the first `ax_qgemm_w8a32` and recycles
  the buffers from then on.

for a multi-layer model you'd build an `ax_sequential_t` of `ax_dense_create`
+ activation layers and call `ax_layer_forward(seq, input_tensor)`. the
quantized path doesn't have a layer wrapper yet — it's a raw kernel.
that's a roadmap item; for now treat it as the inference primitive you
build your own layer struct around.


## 6. memory profiling — what your firmware actually pays

measure the static archive footprint:

```
$ size build-stm32/libaxiom.a
   text    data     bss     dec     hex filename
   ...
```

aggregate `text` across the rows is the upper bound on what gets
linked into your firmware. but `--gc-sections` (which the embedded
profile enables via `-ffunction-sections -fdata-sections` +
`-Wl,--gc-sections`) only pulls the functions you actually call. real
footprints from a release build of the inference subset on cortex-m4f:

| firmware uses | typical text footprint |
| --- | --- |
| dense + relu only | ~38 kb |
| above + softmax + argmax | ~46 kb |
| + conv2d (im2col path) | ~72 kb |
| + batchnorm forward | ~88 kb |
| + winograd conv (large weights) | ~110 kb |
| + mha forward (transformer head) | ~135 kb |

bss / data are dominated by the pack scratch buffers (sized from the
compile-time `AX_GEMM_DEFAULT_*` macros) plus the user's model
weights. with the default `24/32/64` tiles, scratch is ~14 kb total
(pack_a + pack_b). bumping kc to 128 doubles pack_b to ~16 kb.

to verify the linker has stripped what it should, run:

```
arm-none-eabi-objdump -h your_firmware.elf | grep -E '\.text|\.rodata|\.bss'
arm-none-eabi-nm --size-sort your_firmware.elf | tail -30
```

the `nm --size-sort` tail tells you what's actually still in the
binary, sorted by size descending. if you see a symbol you didn't
expect (e.g. `ax_adam_step` in an inference build), something pulled
the autograd path back in — most likely a stray `#include
<axiom/optim.h>` or a layer constructor that compiled with
`AX_INFERENCE_ONLY=0`.


## 7. pitfalls

each one of these has bitten somebody. read this section before your
first deployment.

### heap exhaustion on first conv

the storage pool is lazy: it allocates on the first call and recycles
from then on. the first `ax_conv2d_forward` on a fresh pool can
allocate **all** the im2col scratch in one go (input tile + pack_a +
pack_b + output tile + winograd transform buffers if applicable).
size your heap for the worst case, not the steady-state.

worst-case rule of thumb for one conv layer:

```
heap_bytes_for_conv ≈ (C_in * kh * kw * out_h * out_w * 4)   /* im2col */
                    + (MC * KC * 4) + (NC * KC * 4)          /* pack */
                    + (C_out * out_h * out_w * 4)            /* output */
```

if you're feeding a 32×32×3 image through a conv with 32 output
channels and 3×3 kernel, im2col alone is 32*32*3*9*4 = ~110 kb. a
cortex-m with 128 kb sram will not fit this. either downsize the input
in a preprocessing layer or use the direct conv path (axiom auto-selects
the smaller-memory path; see `src/core/conv/path_selection.c` for the
heuristic).

once the first call returns, those buffers stay in the storage pool
and subsequent forwards re-use them with zero allocation.

### thread-local storage on baremetal

axiom uses `_Thread_local` for the pack buffers and the per-thread
storage pool. on toolchains that support tls (gcc-arm-none-eabi 10+,
picolibc), this works as expected. on toolchains that don't (some
older newlib-nano builds, certain rtos ports), the build fails with
"`__emutls_get_address` undefined".

`AX_SINGLE_THREADED=ON` (default in the embedded profile) collapses
`_Thread_local` to plain `static`. this loses concurrent-safety —
which doesn't matter on a single-thread baremetal target — but it
removes the toolchain dependency. if your firmware is multi-threaded
(rtos with several inference tasks) you have two choices:

- keep `AX_SINGLE_THREADED=ON` and serialize inference calls behind a
  mutex. simplest, fine for sub-100 hz inference rates.
- set `AX_SINGLE_THREADED=OFF` and ensure your toolchain has tls
  support. pack buffers will be per-thread; concurrent calls on
  separate tensors are safe.

### no fpu on cortex-m0 / m0+ / m23

axiom's hot kernels are all `float`. cortex-m0 and m0+ have no fpu —
every `float` op compiles to a soft-float library call, and inference
runs roughly 30–80× slower than on an m4f. the framework will run, but
the only sane targets for axiom are:

- cortex-m4f, m7, m33, m55, m85 (single-precision fpu)
- cortex-m7 with double-precision fpu (uses the dp path automatically)
- any cortex-a (always has neon-fp)

if you're on m0/m0+ and absolutely need to run a model, look at the
quantized-only inference path (`ax_qgemm_w8a32`) — int8 mul + int32
accumulate doesn't need an fpu. you still need fp32 for the per-channel
scale at the end, but the bulk of the work stays integer.

### `ax_compute_init` and constructor attributes

the cpu_opt backend's tile sizes are set inside `ax_cpu_opt_tune_init`,
which is normally invoked via a gcc `__attribute__((constructor))`.
some embedded crt0 scripts (notably hand-rolled vector tables that
predate `.init_array` walking) skip the constructor pass. axiom guards
against this by also calling `ax_cpu_opt_tune_init` explicitly from
`ax_compute_init`, so as long as you call `ax_init()` at boot you're
covered. if you skip `ax_init` and try to use a tensor op directly,
you'll get a `NULL` backend dispatch and an immediate failure. always
call `ax_init` first.

### serialize.h and file i/o

`ax_model_load` and `ax_model_save` use `fopen`/`fread`/`fwrite`. they
compile under `AX_INFERENCE_ONLY` (the parsing logic is reusable) but
they pull stdio, which the embedded profile compiles out elsewhere.
so on baremetal you do **not** use `ax_model_load`. instead:

- save the model on the host with `ax_model_save` to a `.axm` file
- convert the `.axm` to a `const uint8_t[]` (e.g. via `xxd -i model.axm > model.h`)
- include the array in your firmware as a flash-resident blob
- parse it in-memory with the manual format walker (see `src/core/serialize.c`
  for the on-disk layout — header + layer descriptors + parameter data)

a future release will add an `ax_model_load_from_buffer` that reads the
same format from a memory pointer; for now you wire the layers manually
in code and copy the parameter bytes into the layer's weight tensors
via `ax_tensor_from_array`.

quantized weights are easier — they're three plain c arrays (`int8_t
data[]`, `float scales[]`, plus N and K), and the example in section 5
shows how to point an `ax_qweight_t` at them directly.

### libgomp and "undefined reference to `__gomp_*`"

if you accidentally leave `AX_OPENMP=ON` and your toolchain has
libgomp, the build succeeds but the firmware grows by ~40 kb of unused
runtime. if your toolchain doesn't have libgomp the link fails with
"undefined reference to `GOMP_parallel`". fix: `-DAX_OPENMP=OFF`, or
just use the `embedded-baremetal` profile which sets it for you.


## further reading

- `docs/embedded.md` — embedded-linux profile (raspberry pi, jetson, etc.)
- `docs/PRODUCTION_PLAN.md` — roadmap for `ax_model_load_from_buffer`,
  cortex-m helium (mve) simd, and embedded quantization extensions
- `cmake/profiles/embedded-baremetal.cmake` — the source of truth for
  what flags the profile flips
- `cmake/toolchains/arm-none-eabi.cmake` — starter toolchain file with
  `-mcpu`/`-mfpu` recipes for the common cortex-m parts
- `include/axiom/quantize.h` — int8 W8A32 api
- `include/axiom/error.h` — error tls + callback contract
- `scripts/check_portability.py` — the lint that catches x86-only
  intrinsic regressions before they reach a cross-compile build
