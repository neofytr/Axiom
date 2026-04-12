# axiom embedded-linux profile: arm sbcs running a real linux kernel.
# raspberry pi 4/5, jetson nano/orin, nxp i.mx 8, rockchip rk3588,
# xilinx zynq ultrascale+, beaglebone, orange pi, etc.
#
# assumptions: linux kernel + glibc (or musl), mb-scale ram, neon if the
# cpu has it (most cortex-a do), openmp available as libgomp. full posix.
#
# use this with a cross-compile toolchain file, e.g.:
#   cmake -DAX_PROFILE=embedded-linux \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake ..

message(STATUS "axiom profile: embedded-linux")

# openmp is standard on linux-on-arm. disable only if your distro lacks
# libgomp (uncommon; musl can still link gcc's libgomp).
set(AX_OPENMP ON CACHE BOOL "enable openmp multi-core parallelism")

# autotuner is safe here — sched_setaffinity and sysconf are available
# on every linux build. ~200ms probe at first use.
set(AX_NO_AUTOTUNE OFF CACHE BOOL "disable runtime cpu autotuner")

# tls works normally on glibc/musl. keep the tls-backed pack buffers.
set(AX_SINGLE_THREADED OFF CACHE BOOL "disable thread-local storage and omp")

# linux has stderr, keep diagnostics.
set(AX_NO_STDIO OFF CACHE BOOL "drop fprintf(stderr,...) diagnostics")

# smaller gemm tile defaults tuned for l2 caches typical of cortex-a:
# 48/128/128 -> pack_a 24 kb, pack_b 64 kb. fits in a 256 kb l2 with
# margin. users with larger l2 (jetson orin has 2 mb) can still bump
# via runtime AX_GEMM_* env vars.
add_compile_definitions(
    AX_GEMM_DEFAULT_MC=48
    AX_GEMM_DEFAULT_NC=128
    AX_GEMM_DEFAULT_KC=128
)
