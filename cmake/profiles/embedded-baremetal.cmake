# axiom embedded-baremetal profile: bare metal, freertos, zephyr, nuttx.
# cortex-a without linux (zynq standalone), cortex-m4f/m7f/m33/m55/m85
# running an rtos, nxp i.mx rt, stm32h7, ambiq apollo, nordic nrf53, etc.
#
# assumptions: no full posix, often no stdio, often no threading runtime,
# may have no tls support from the toolchain, kb- to mb-scale ram.
#
# use with an arm-none-eabi or picolibc-based toolchain file:
#   cmake -DAX_PROFILE=embedded-baremetal \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake ..
#
# override individual defaults on the command line if your target is
# atypical (e.g. you DO have freertos threads and want omp-style
# parallelism via a ported libgomp).

message(STATUS "axiom profile: embedded-baremetal")

# no openmp — most baremetal/rtos targets have no libgomp.
set(AX_OPENMP OFF CACHE BOOL "enable openmp multi-core parallelism")

# no autotuner — no sched_setaffinity, no sysconf, no way to probe.
# the code compiles out entirely with this flag.
set(AX_NO_AUTOTUNE ON CACHE BOOL "disable runtime cpu autotuner")

# single-threaded: plain `static` instead of `_Thread_local` for the
# gemm pack buffers. required for toolchains without tls support
# (newlib-nano, some rtos ports).
set(AX_SINGLE_THREADED ON CACHE BOOL "disable thread-local storage and omp")

# no stderr — all fprintf(stderr, ...) diagnostics compile out.
# saves ~20 kb of pulled-in stdio machinery on flash-constrained targets.
set(AX_NO_STDIO ON CACHE BOOL "drop fprintf(stderr,...) diagnostics")

# tiny gemm tiles. pack_a 24*64*4 = 6 kb, pack_b 32*64*4 = 8 kb.
# fits comfortably in a 64 kb sram with room for stack, model, and
# scratch. users with more sram can override at configure time:
#   cmake -DAX_GEMM_DEFAULT_KC=128 ...
add_compile_definitions(
    AX_GEMM_DEFAULT_MC=24
    AX_GEMM_DEFAULT_NC=32
    AX_GEMM_DEFAULT_KC=64
)

# reduce binary bloat: place each function/data in its own section so
# the linker can garbage-collect anything unreachable from main.
# saves significant flash on deployments that use only the inference path.
add_compile_options(-ffunction-sections -fdata-sections -fno-common)

# no exceptions or unwind info — axiom is C only so there are no c++
# exceptions to handle, but gcc still generates .eh_frame unless told
# not to. the flag below strips it on any supporting target.
add_compile_options(-fno-unwind-tables -fno-asynchronous-unwind-tables)

# let the linker drop unused sections (paired with -ffunction-sections)
add_link_options(-Wl,--gc-sections)
