# Axiom on embedded targets

Axiom builds and runs on three tiers of hardware. You pick one via the
`AX_PROFILE` CMake variable and (for cross-compile) a toolchain file.
You do **not** need to know anything about the specific chip — just
the class of environment.

## Profiles

| `AX_PROFILE=` | Target class | OMP | Autotune | Stdio | TLS | GEMM tiles (MC/NC/KC) |
|---|---|---|---|---|---|---|
| `desktop` (default) | x86 / arm desktop + laptop, cloud vm | on | on | on | yes | 72/256/256 |
| `embedded-linux` | Raspberry Pi, Jetson, i.MX, Zynq running Linux | on | on | on | yes | 48/128/128 |
| `embedded-baremetal` | FreeRTOS, Zephyr, NuttX, bare metal, Cortex-M RTOS | off | off | off | no | 24/32/64 |

Each profile sets sane cache defaults for every option; you can still
override any single one on the command line:

```bash
cmake -DAX_PROFILE=embedded-linux -DAX_OPENMP=OFF ..
```

## Building for the three classes

### Desktop (native)

```bash
mkdir build && cd build
cmake ..
cmake --build .
ctest
```

This is also the zero-config case: `AX_PROFILE` defaults to `desktop`.

### Embedded Linux (Raspberry Pi 4/5, Jetson, iMX, Zynq)

Install the cross toolchain:

```bash
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu   # aarch64
sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf # armv7
```

Configure with the matching toolchain file:

```bash
mkdir build-pi && cd build-pi
cmake -DAX_PROFILE=embedded-linux \
      -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/aarch64-linux-gnu.cmake \
      ..
cmake --build .
```

Copy the resulting `libaxiom.a` (or `libaxiom.so`) to the target and
link as usual. NEON is picked up automatically via `simd_defs.h`.
OpenMP is still on — if your target's libgomp is missing, pass
`-DAX_OPENMP=OFF`.

### Embedded baremetal / RTOS (Cortex-M, Cortex-A standalone, FreeRTOS, Zephyr)

Install the bare-metal toolchain:

```bash
sudo apt install gcc-arm-none-eabi
```

or grab the latest Arm release from
<https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads>.

Edit `cmake/toolchains/arm-none-eabi.cmake` to match your MCU (the
file has a list of common `-mcpu`/`-mfpu` settings in a comment block),
then configure:

```bash
mkdir build-bm && cd build-bm
cmake -DAX_PROFILE=embedded-baremetal \
      -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/arm-none-eabi.cmake \
      ..
cmake --build . --target axiom_static
```

Only `axiom_static` is meaningful on baremetal — the test/example
binaries assume a hosted libc and will fail to link. Drop
`libaxiom.a` into your firmware project's link step.

## What the baremetal profile actually changes

If you're curious what `embedded-baremetal` flips under the hood:

- `AX_OPENMP=OFF` — all `#pragma omp ...` expands to nothing; library
  runs serial.
- `AX_SINGLE_THREADED=1` — `_Thread_local` in `cpu_opt.c` collapses to
  plain `static`. Works on toolchains without TLS support (old
  newlib-nano, some RTOS ports).
- `AX_NO_AUTOTUNE=1` — `ax_autotune_threads()` compiles to a single
  `return 1`. No `sched_setaffinity`, no `sysconf`.
- `AX_NO_STDIO=1` — every `fprintf(stderr, ...)` diagnostic compiles
  out, so you don't pull in ~20 KB of stdio machinery if your crt
  doesn't have it.
- `AX_GEMM_DEFAULT_MC=24 NC=32 KC=64` — pack buffers shrink to
  `24*64*4=6 KB` (pack_a) and `32*64*4=8 KB` (pack_b). Fits in a 64 KB
  SRAM with room for stack and model.
- `-ffunction-sections -fdata-sections` + `-Wl,--gc-sections` — the
  linker garbage-collects anything unreachable from your application's
  `main`, so the baremetal firmware only pays for the ops it actually
  calls.
- `-fno-unwind-tables -fno-asynchronous-unwind-tables` — strips
  `.eh_frame` that GCC emits by default.

You can override any of these individually without switching profiles,
e.g. `-DAX_OPENMP=ON` if your RTOS ships libgomp.

## Tuning the GEMM tile sizes for your target

The default MC/NC/KC for each profile are starting points. If your
target has a larger L2 than the default assumed, bump them up; if the
compile fails with "out of memory" in the micro-kernel pack buffers,
shrink them:

```bash
cmake -DAX_PROFILE=embedded-baremetal \
      -DCMAKE_C_FLAGS="-DAX_GEMM_DEFAULT_MC=16 -DAX_GEMM_DEFAULT_NC=32 -DAX_GEMM_DEFAULT_KC=32" \
      ..
```

Rule of thumb: pick KC * NC * 4 bytes so that it's ~half your L2
cache. Then pick MC as a small multiple of 6 (AVX2) / 4 (NEON) that
fits the rest.

On hosted Linux builds you also get runtime override via env vars:

```bash
AX_GEMM_MC=64 AX_GEMM_NC=128 AX_GEMM_KC=128 ./your_app
```

These are disabled on baremetal (no `getenv` in a clean crt) and you
must set them at compile time instead.

## What still doesn't work on baremetal

- **Int8/int16 quantized inference.** fp32 only. For very constrained
  targets (< 1 MB RAM) this is a real limitation; roadmapped for a
  separate quantization round.
- **Training.** The autograd machinery, optimizers, loss functions,
  and serialization are all part of `libaxiom.a` but inference-only
  firmware should use `--gc-sections` to strip them. If you need a
  true inference-only build with no autograd at all, that's a future
  build-system option (`-DAX_INFERENCE_ONLY=ON`) — not yet plumbed.
- **Cortex-M Helium (MVE) SIMD.** Cortex-M55/M85 have MVE but axiom
  doesn't emit it yet — falls back to the scalar path which is
  correct but much slower than what the hardware can do.

## Supported cross-compile tuples (templates provided)

| Toolchain file | Target |
|---|---|
| `cmake/toolchains/aarch64-linux-gnu.cmake` | 64-bit Arm Linux (Pi 4/5 aarch64 os, Jetson, iMX 8, Zynq US+) |
| `cmake/toolchains/arm-linux-gnueabihf.cmake` | 32-bit Arm Linux hardfp (Pi 2/3 armv7, Pi Zero 2, BBB, legacy boards) |
| `cmake/toolchains/arm-none-eabi.cmake` | Bare metal / RTOS Cortex-A/M (any MCU or Cortex-A without Linux) |

All three files are editable templates — adjust `-mcpu` /`-mfpu` /
`-march` to your specific chip. The existing entries are conservative
baselines that run on nearly any part in the family.

For CPUs not in this list (RISC-V, Xtensa, PowerPC embedded): write a
new toolchain file following the same pattern. The axiom backend
itself is portable — only SIMD takes a per-ISA codepath, and the
scalar fallback in `simd_defs.h` covers anything without AVX2 or NEON.
