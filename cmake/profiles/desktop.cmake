# axiom desktop profile: hosted x86_64 / aarch64 linux, macos, windows.
# assumes full glibc, openmp available, plenty of ram and cache.
# this is the default profile when AX_PROFILE is not set.

message(STATUS "axiom profile: desktop")

# parallelism on by default. every modern desktop has multi-core + libgomp.
set(AX_OPENMP ON CACHE BOOL "enable openmp multi-core parallelism" FORCE)

# runtime hybrid-cpu autotuner on by default. probes per-core speed at
# first use and pins omp workers to the fast cores on alder lake style
# hybrid cpus. ~200ms calibration budget, opt out via AX_NO_AUTOTUNE=1 env.
set(AX_NO_AUTOTUNE OFF CACHE BOOL "disable runtime cpu autotuner" FORCE)

# tls-backed per-thread pack buffers for gemm. requires _Thread_local
# support from the toolchain (all modern desktop toolchains have it).
set(AX_SINGLE_THREADED OFF CACHE BOOL "disable thread-local storage and omp" FORCE)

# stderr logging on. users see autotuner diagnostics and gemm tile warnings.
set(AX_NO_STDIO OFF CACHE BOOL "drop fprintf(stderr,...) diagnostics" FORCE)

# gemm tile defaults are the compile-time baseline. the runtime
# AX_GEMM_MC/NC/KC env vars still override at process startup.
# 72/256/256 targets a ~256 kb l2 typical on modern desktop x86.
add_compile_definitions(
    AX_GEMM_DEFAULT_MC=72
    AX_GEMM_DEFAULT_NC=256
    AX_GEMM_DEFAULT_KC=256
)
