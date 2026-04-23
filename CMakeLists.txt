cmake_minimum_required(VERSION 3.18)
# version is the single source of truth for the released build artefacts
# (cmake project version, axiom.pc, library SOVERSION). it MUST track the
# AX_VERSION_* constants in include/axiom/types.h.
project(axiom
        VERSION 0.10.0
        DESCRIPTION "axiom — pure C deep learning framework (zero-deps, embedded-ready)"
        HOMEPAGE_URL "https://github.com/neofytr/neoNN"
        LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# build type: Debug (sanitizers), Release (optimized), or default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# build profile: picks sane defaults for a target class (desktop vs
# linux-embedded vs baremetal). each profile file under cmake/profiles/
# sets cache defaults for AX_OPENMP, AX_NO_AUTOTUNE, AX_SINGLE_THREADED,
# AX_NO_STDIO, and the compile-time GEMM tile sizes. command-line
# overrides (e.g. -DAX_OPENMP=OFF) still win. see docs/embedded.md.
if(NOT AX_PROFILE)
    set(AX_PROFILE "desktop" CACHE STRING
        "build profile: desktop | embedded-linux | embedded-baremetal")
endif()
set_property(CACHE AX_PROFILE PROPERTY STRINGS
    desktop embedded-linux embedded-baremetal)

set(_ax_profile_file "${CMAKE_CURRENT_SOURCE_DIR}/cmake/profiles/${AX_PROFILE}.cmake")
if(NOT EXISTS "${_ax_profile_file}")
    message(FATAL_ERROR "unknown AX_PROFILE '${AX_PROFILE}' — expected one of "
                        "desktop / embedded-linux / embedded-baremetal")
endif()
include("${_ax_profile_file}")

# base warnings — strict by default
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter
                    -Wformat=2 -Wformat-security -Wconversion
                    -Wshadow -Wstrict-prototypes
                    -fstack-protector-strong)

# sanitizer build: cmake -DCMAKE_BUILD_TYPE=Debug -DAX_SANITIZE=ON ..
option(AX_SANITIZE "enable address + undefined behavior sanitizers" OFF)
option(AX_TSAN "enable thread sanitizer (mutually exclusive with AX_SANITIZE)" OFF)

if(AX_SANITIZE AND AX_TSAN)
    message(FATAL_ERROR "AX_SANITIZE and AX_TSAN are mutually exclusive")
endif()

if(AX_SANITIZE)
    message(STATUS "sanitizers enabled (asan + ubsan)")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

if(AX_TSAN)
    message(STATUS "thread sanitizer enabled")
    message(STATUS "  note: run TSan tests via 'setarch -R ctest' on linux with high-entropy ASLR")
    message(STATUS "  PIE/ASLR remain enabled in the binary; setarch only affects test runtime")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=thread)
endif()

# CUDA GPU backend: cmake -DAX_CUDA=ON ..
option(AX_CUDA "enable NVIDIA GPU backend via CUDA + cuBLAS" OFF)
if(AX_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # multi-arch fat binary: covers maxwell through ada lovelace. the
    # user can narrow to a single arch for faster compile + smaller
    # binary via -DCMAKE_CUDA_ARCHITECTURES=86 etc.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "52;60;70;75;80;86;89")
    endif()

    message(STATUS "CUDA enabled (toolkit ${CUDAToolkit_VERSION}, "
                   "architectures: ${CMAKE_CUDA_ARCHITECTURES})")
    add_compile_definitions(AX_HAVE_CUDA)
endif()

# openmp parallelism (default set by AX_PROFILE; override via -DAX_OPENMP=ON|OFF)
if(AX_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        message(STATUS "openmp enabled (${OpenMP_C_VERSION})")
        add_compile_definitions(AX_HAVE_OPENMP)
    else()
        message(STATUS "openmp requested but not found — building single-threaded")
        set(AX_OPENMP OFF)
    endif()
endif()

# tier-2 baremetal gates. each one disables a feature that assumes
# a hosted os / glibc / threading runtime. all propagate into the code
# as preprocessor defines consumed by cpu_opt.c, dispatch.c, etc.
if(AX_NO_AUTOTUNE)
    message(STATUS "runtime cpu autotuner disabled")
    add_compile_definitions(AX_NO_AUTOTUNE=1)
endif()
if(AX_SINGLE_THREADED)
    message(STATUS "single-threaded build (_Thread_local -> static)")
    add_compile_definitions(AX_SINGLE_THREADED=1)
endif()
if(AX_NO_STDIO)
    message(STATUS "stderr diagnostics disabled (AX_NO_STDIO)")
    add_compile_definitions(AX_NO_STDIO=1)
endif()

# inference-only mode: strip all training code from the library.
# excludes autograd, optimizers, loss functions, lr schedulers, and the
# dataset/dataloader from AXIOM_SOURCES. the forward files (conv/norm/
# ops/activations/layer) still compile unchanged — ax_grad_enabled() is
# stubbed to a constant false in the header, so the compiler dce's every
# grad_fn setup branch and never emits references to the excluded code.
# default off; the embedded-baremetal profile flips it on.
if(AX_INFERENCE_ONLY)
    message(STATUS "inference-only build (training code excluded)")
    add_compile_definitions(AX_INFERENCE_ONLY=1)
endif()

# hardening flags for release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -fno-math-errno -funroll-loops -fno-trapping-math -D_FORTIFY_SOURCE=2)
endif()

# library sources — split into always-on (forward + shared infra) and
# training-only (autograd, optimizers, losses, dataloader, lr schedules).
# under AX_INFERENCE_ONLY the training group is dropped from the build.
# runtime isa dispatch for cpu_opt. when on, cpu_opt.c is compiled
# twice: once with -mavx2 -mfma (AX_CPU_OPT_SUFFIX=_avx2) and once
# with vector extensions disabled (AX_CPU_OPT_SUFFIX=_scalar). the
# two object files produce distinct vtable symbols and dispatch.c
# picks one at ax_compute_init time via __builtin_cpu_supports.
# single binary that gracefully degrades on pre-avx2 x86_64.
# default off — costs nothing on a fixed-target deployment.
option(AX_CPU_ISA_DISPATCH "build cpu_opt twice (avx2 + scalar) and select at runtime" OFF)

set(AXIOM_SOURCES
    src/core/error.c
    src/core/memory.c
    src/core/tensor.c
    src/core/broadcast.c
    src/core/ops.c
    src/core/activations.c
    src/core/init.c
    src/core/layer.c
    src/core/model.c
    src/core/serialize.c
    src/core/conv/pool.c
    src/core/conv/conv_bn_relu.c
    src/core/conv/winograd.c
    src/core/conv/direct.c
    src/core/conv/im2col.c
    src/core/conv/path_selection.c
    src/core/conv/backward.c
    src/core/conv/forward.c
    src/core/norm.c
    src/core/attention.c
    src/core/attention/layout.c
    src/core/attention/cache.c
    src/core/attention/train_step.c
    src/core/attention/train_step_fused.c
    src/core/fused_attention.c
    src/core/rng.c
    src/core/quantize.c
    src/compute/dispatch.c
    src/compute/backends/cpu_naive.c
)

# JIT encoders. x86_64 path emits VEX/EVEX, arm64 path emits A64.
# the encoders themselves are pure C (no SIMD intrinsics) — they just
# write machine bytes. ALL encoders compile on every host so byte-level
# tests can verify encodings even when the bytes won't execute. the
# kernel-emitter wrappers (jit_gemm_*) compile per host ISA so the
# (per-thread, pthread_once-protected) cached fn-ptr only points at
# bytes the host can actually run.
list(APPEND AXIOM_SOURCES src/compute/backends/jit_arm64.c)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    list(APPEND AXIOM_SOURCES
        src/compute/backends/jit_x64.c
        src/compute/backends/jit_gemm_avx2.c
        src/compute/backends/jit_gemm_avx512.c
    )
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    list(APPEND AXIOM_SOURCES src/compute/backends/jit_gemm_neon.c)
endif()
# cpu_opt goes in directly under single-build, or via dual object
# libraries under AX_CPU_ISA_DISPATCH. see the isa-dispatch block
# below.
if(NOT AX_CPU_ISA_DISPATCH)
    list(APPEND AXIOM_SOURCES src/compute/backends/cpu_opt.c)
endif()

if(NOT AX_INFERENCE_ONLY)
    list(APPEND AXIOM_SOURCES
        src/core/autograd.c
        src/core/autograd_ops.c
        src/core/losses.c
        src/core/optim.c
        src/core/data.c
        src/core/lr_scheduler.c
    )
endif()

if(AX_CUDA)
    # cuda backend is a self-contained module under backends/cuda/
    set(AX_CUDA_SOURCES
        src/compute/backends/cuda/backend.cu
        src/compute/backends/cuda/memory.cu
        src/compute/backends/cuda/ops_elementwise.cu
        src/compute/backends/cuda/ops_reduce.cu
        src/compute/backends/cuda/ops_gemm.cu
        src/compute/backends/cuda/ops_conv.cu
        src/compute/backends/cuda/ops_winograd.cu
        src/compute/backends/cuda/ops_optim.cu
        src/compute/backends/cuda/ops_fused.cu
        src/compute/backends/cuda/ops_activations.cu
        src/compute/backends/cuda/ops_attention.cu
    )
    list(APPEND AXIOM_SOURCES ${AX_CUDA_SOURCES})
endif()

# enable SIMD for the optimized backend on x86. single-build path:
# use the detected target ISA (-mavx2 -mfma on x86_64). the runtime-
# isa-dispatch path below overrides this by compiling two object
# libraries with their own flags.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86" AND NOT AX_CPU_ISA_DISPATCH)
    set_source_files_properties(src/compute/backends/cpu_opt.c
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()

# SIMD flags for every core/*.c file that includes simd_defs.h. without
# this the compiler doesn't predefine __AVX2__/__FMA__ and AX_HAS_SIMD
# falls back to the scalar path, destroying throughput on GELU, BN, LN,
# Adam, etc.  under AX_CPU_ISA_DISPATCH we still apply the baseline flags
# (AVX2+FMA) to these TUs — they aren't dispatched per-ISA, only the
# cpu_opt backend is.  so we target the machines that matter (AVX2+) and
# let the top-level x86 baseline take care of pre-AVX2 fallbacks through
# the compiler's -march setting.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86")
    set_source_files_properties(
        src/core/activations.c
        src/core/attention.c
        src/core/autograd_ops.c
        src/core/conv/backward.c
        src/core/conv/conv_bn_relu.c
        src/core/conv/direct.c
        src/core/conv/forward.c
        src/core/conv/im2col.c
        src/core/conv/pool.c
        src/core/conv/winograd.c
        src/core/losses.c
        src/core/norm.c
        src/core/optim.c
        src/core/quantize.c
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()

# runtime isa dispatch: build two cpu_opt object libraries with
# distinct symbol suffixes and different compile flags, then link
# both into axiom_static/axiom_shared. dispatch.c selects the
# active vtable at ax_compute_init via __builtin_cpu_supports.
if(AX_CPU_ISA_DISPATCH)
    message(STATUS "cpu_opt runtime isa dispatch enabled (avx512 + avx2 + scalar variants)")

    # avx-512 variant — fastest on xeon / ice lake+.
    add_library(axiom_cpu_opt_avx512 OBJECT src/compute/backends/cpu_opt.c)
    target_include_directories(axiom_cpu_opt_avx512 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE src/compute/backends)
    target_compile_definitions(axiom_cpu_opt_avx512 PRIVATE AX_CPU_OPT_SUFFIX=_avx512)
    target_compile_options(axiom_cpu_opt_avx512 PRIVATE -mavx512f -mavx2 -mfma)
    set_property(TARGET axiom_cpu_opt_avx512 PROPERTY POSITION_INDEPENDENT_CODE ON)
    if(AX_OPENMP AND OpenMP_C_FOUND)
        target_link_libraries(axiom_cpu_opt_avx512 PUBLIC OpenMP::OpenMP_C)
    endif()

    # avx2+fma variant — fast on most x86_64 since haswell (2013).
    add_library(axiom_cpu_opt_avx2 OBJECT src/compute/backends/cpu_opt.c)
    target_include_directories(axiom_cpu_opt_avx2 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE src/compute/backends)
    target_compile_definitions(axiom_cpu_opt_avx2 PRIVATE AX_CPU_OPT_SUFFIX=_avx2)
    target_compile_options(axiom_cpu_opt_avx2 PRIVATE -mavx2 -mfma)
    # pic required because the object files are linked into both the
    # static and the shared axiom libraries.
    set_property(TARGET axiom_cpu_opt_avx2 PROPERTY POSITION_INDEPENDENT_CODE ON)
    if(AX_OPENMP AND OpenMP_C_FOUND)
        target_link_libraries(axiom_cpu_opt_avx2 PUBLIC OpenMP::OpenMP_C)
    endif()

    # scalar variant — no vector extensions beyond sse2 baseline.
    # runs on any x86_64 cpu including very old ones; simd_defs.h
    # falls through to its scalar branch when __AVX2__ isn't set.
    add_library(axiom_cpu_opt_scalar OBJECT src/compute/backends/cpu_opt.c)
    target_include_directories(axiom_cpu_opt_scalar PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE src/compute/backends)
    target_compile_definitions(axiom_cpu_opt_scalar PRIVATE AX_CPU_OPT_SUFFIX=_scalar)
    target_compile_options(axiom_cpu_opt_scalar PRIVATE -mno-avx2 -mno-fma -mno-avx)
    set_property(TARGET axiom_cpu_opt_scalar PROPERTY POSITION_INDEPENDENT_CODE ON)
    if(AX_OPENMP AND OpenMP_C_FOUND)
        target_link_libraries(axiom_cpu_opt_scalar PUBLIC OpenMP::OpenMP_C)
    endif()

    # core compiled units get the dispatch flag so dispatch.c swaps
    # the extern decls + runtime selection in.
    add_compile_definitions(AX_CPU_ISA_DISPATCH=1)
endif()

# static library
add_library(axiom_static STATIC ${AXIOM_SOURCES})
target_include_directories(axiom_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(axiom_static PUBLIC m)
if(AX_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(axiom_static PUBLIC OpenMP::OpenMP_C)
endif()
if(AX_CPU_ISA_DISPATCH)
    target_link_libraries(axiom_static PUBLIC
        axiom_cpu_opt_avx512 axiom_cpu_opt_avx2 axiom_cpu_opt_scalar)
endif()
if(AX_CUDA)
    target_link_libraries(axiom_static PUBLIC CUDA::cudart CUDA::cublas)
    # suppress C-only warning flags when nvcc compiles cuda sources as C++
    set_source_files_properties(${AX_CUDA_SOURCES}
        PROPERTIES COMPILE_FLAGS "-Xcompiler -Wno-pedantic,-Wno-strict-prototypes")
endif()
set_target_properties(axiom_static PROPERTIES OUTPUT_NAME axiom)

# shared library
add_library(axiom_shared SHARED ${AXIOM_SOURCES})
target_include_directories(axiom_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(axiom_shared PUBLIC m)
if(AX_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(axiom_shared PUBLIC OpenMP::OpenMP_C)
endif()
if(AX_CPU_ISA_DISPATCH)
    target_link_libraries(axiom_shared PUBLIC
        axiom_cpu_opt_avx2 axiom_cpu_opt_scalar)
endif()
if(AX_CUDA)
    target_link_libraries(axiom_shared PUBLIC CUDA::cudart CUDA::cublas)
endif()
set_target_properties(axiom_shared PROPERTIES OUTPUT_NAME axiom)

# tests
enable_testing()

# always-available tests (exercise only forward-path + tensor machinery).
# these must NOT call ax_backward, optimizers, losses, or model_save —
# those symbols are stripped from inference-only builds.
set(TEST_SOURCES
    tests/test_error.c
    tests/test_memory.c
    tests/test_tensor.c
    tests/test_compute.c
    tests/test_property.c
    tests/test_ops.c
    tests/test_conv.c
    tests/test_norm.c
    tests/test_cuda.c
)

# JIT correctness tests. byte-level tests run on every host (independent
# of execution); execution tests only on x86_64 since this dev box can run
# AVX2 emitted code.
list(APPEND TEST_SOURCES tests/test_jit_arm64_bytes.c tests/test_quantize.c)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    list(APPEND TEST_SOURCES
        tests/test_jit_x64.c
        tests/test_jit_gemm.c
        tests/test_jit_avx512_bytes.c
    )
endif()

# training-only tests: anything that calls ax_backward, optimizers,
# losses, model_compile, model_train_step, tensor_save, or model_save.
if(NOT AX_INFERENCE_ONLY)
    list(APPEND TEST_SOURCES
        tests/test_activations.c
        tests/test_attention.c
        tests/test_layer.c
        tests/test_serialize.c
        tests/test_backend_opt.c
        tests/test_autograd.c
        tests/test_losses.c
        tests/test_optim.c
        tests/test_data.c
        tests/test_integration.c
        tests/test_grad_verify.c
        tests/test_serialize_all.c
        tests/test_lr_sched.c
        tests/test_inplace.c
        tests/test_stress.c
        tests/test_fuzz_serialize.c
    )
endif()

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE tests)
    target_link_libraries(${test_name} PRIVATE axiom_static)
    if(AX_TSAN)
        # tsan needs reduced ASLR; setarch -R disables address space randomization
        # for the test process only (host system unchanged)
        add_test(NAME ${test_name}
                 COMMAND setarch ${CMAKE_SYSTEM_PROCESSOR} -R $<TARGET_FILE:${test_name}>)
    else()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
    # baseline test environment:
    #   AX_NO_AUTOTUNE=1   skip thread-count probe (~150 ms / binary)
    #   AX_GEMM_CALIBRATE=0 skip gemm tile probe (~500 ms / binary)
    # tests don't need autotuned thresholds — they need correctness. each
    # test binary previously paid ~6-8 s of startup; this drops it to ~1 s
    # so the full ctest suite finishes in under a minute.
    set(_ax_test_env "AX_NO_AUTOTUNE=1;AX_GEMM_CALIBRATE=0")
    # tests intentionally don't clean up every intermediate tensor in the
    # computation graph (that's the model api's job via ax_graph_cleanup).
    # suppress leak detection for tests so asan only catches real bugs.
    if(AX_SANITIZE)
        list(APPEND _ax_test_env "ASAN_OPTIONS=detect_leaks=0")
    endif()
    if(AX_TSAN)
        list(APPEND _ax_test_env
             "TSAN_OPTIONS=ignore_noninstrumented_modules=1 suppressions=${CMAKE_SOURCE_DIR}/.tsan_suppressions")
    endif()
    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "${_ax_test_env}")
endforeach()

# examples — training examples are excluded under AX_INFERENCE_ONLY
# since they all call ax_backward / optimizers.
if(NOT AX_INFERENCE_ONLY)
    add_executable(example_mnist examples/mnist.c)
    target_link_libraries(example_mnist PRIVATE axiom_static)

    add_executable(mnist_cnn examples/mnist_cnn.c)
    target_link_libraries(mnist_cnn PRIVATE axiom_static)

    add_executable(deep_mlp examples/deep_mlp.c)
    target_link_libraries(deep_mlp PRIVATE axiom_static)

    # benchmarks
    add_executable(bench_backend benchmarks/bench_backend.c)
    target_link_libraries(bench_backend PRIVATE axiom_static)

    add_executable(bench_optimizations benchmarks/bench_optimizations.c)
    target_link_libraries(bench_optimizations PRIVATE axiom_static)

    if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/bench_large.c")
        add_executable(bench_large benchmarks/bench_large.c)
        target_link_libraries(bench_large PRIVATE axiom_static)
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/bench_wide.c")
        add_executable(bench_wide benchmarks/bench_wide.c)
        target_link_libraries(bench_wide PRIVATE axiom_static)
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/bench_vgg.c")
        add_executable(bench_vgg benchmarks/bench_vgg.c)
        target_link_libraries(bench_vgg PRIVATE axiom_static)
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/bench_mha.c")
        add_executable(bench_mha benchmarks/bench_mha.c)
        target_include_directories(bench_mha PRIVATE benchmarks)
        target_link_libraries(bench_mha PRIVATE axiom_static)
    endif()
    # ax_* training benchmarks mirroring the tf_*.py scripts one-to-one.
    # each is gated on file existence so optional / in-development benches
    # don't break the build.
    foreach(ax_bench ax_mnist ax_train_save ax_infer_load)
        if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/${ax_bench}.c")
            add_executable(${ax_bench} benchmarks/${ax_bench}.c)
            target_link_libraries(${ax_bench} PRIVATE axiom_static)
        endif()
    endforeach()

    # comprehensive suites vs TF. each has a tf_<name>.py sidecar and
    # benchmarks/summarize.py merges RESULT lines from both sides.
    foreach(suite_bench
            bench_gemm_suite bench_ops_suite bench_conv_suite
            bench_transformer bench_stress bench_parity bench_qgemm)
        if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/${suite_bench}.c")
            add_executable(${suite_bench} benchmarks/${suite_bench}.c)
            target_include_directories(${suite_bench} PRIVATE benchmarks)
            target_link_libraries(${suite_bench} PRIVATE axiom_static)
        endif()
    endforeach()
endif()

# convenience target: run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAMES}
    COMMENT "running all axiom tests"
)

# ---------------------------------------------------------------------------
# install rules (Phase L.1)
#
# `cmake --install build --prefix /usr/local` lays out:
#   ${prefix}/lib/libaxiom.{a,so.0.10.0}                    (libraries)
#   ${prefix}/lib/libaxiom.so -> libaxiom.so.0              (versioned symlinks)
#   ${prefix}/include/axiom/*.h                             (public headers)
#   ${prefix}/lib/cmake/axiom/axiomConfig.cmake             (find_package support)
#   ${prefix}/lib/cmake/axiom/axiomConfigVersion.cmake      (semver match)
#   ${prefix}/lib/pkgconfig/axiom.pc                        (pkg-config)
#
# the internal headers (include/axiom/internal/) are intentionally NOT
# installed — they are part of the framework's implementation detail and
# carry no abi promise. user code must include only axiom/<name>.h.
# ---------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# library SOVERSION = MAJOR; FULLVERSION embeds MAJOR.MINOR.PATCH so the
# linker resolves libaxiom.so -> libaxiom.so.0 -> libaxiom.so.0.10.0 the
# usual way. callers link against the SONAME (libaxiom.so.0) and don't
# need to be rebuilt for patch / minor updates.
set_target_properties(axiom_shared PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

# the multi-isa cpu_opt sub-libraries are OBJECT libs linked privately
# into axiom_static / axiom_shared. they have no separate install artefact
# but cmake EXPORT requires them in the export set so the linker ordering
# can be reproduced downstream.
set(_axiom_export_targets axiom_static axiom_shared)
if(AX_CPU_ISA_DISPATCH)
    list(APPEND _axiom_export_targets
         axiom_cpu_opt_avx2 axiom_cpu_opt_scalar)
    if(TARGET axiom_cpu_opt_avx512)
        list(APPEND _axiom_export_targets axiom_cpu_opt_avx512)
    endif()
endif()

install(TARGETS ${_axiom_export_targets}
    EXPORT axiomTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install only the public headers — drop axiom/internal/.
install(DIRECTORY include/axiom/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/axiom
    FILES_MATCHING PATTERN "*.h"
    PATTERN "internal" EXCLUDE)

# cmake package config — lets downstream `find_package(axiom)` work.
install(EXPORT axiomTargets
    FILE axiomTargets.cmake
    NAMESPACE axiom::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axiom)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/axiomConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/axiomConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axiom)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/axiomConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/axiomConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/axiomConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axiom)

# pkg-config file (Phase L.2). populated from axiom.pc.in via
# configure_file so the version and prefix track the cmake config.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/axiom.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/axiom.pc"
    @ONLY)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/axiom.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
