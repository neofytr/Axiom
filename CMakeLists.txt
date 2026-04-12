cmake_minimum_required(VERSION 3.18)
project(axiom C)

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
    message(STATUS "CUDA enabled (toolkit ${CUDAToolkit_VERSION})")
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

# hardening flags for release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -fno-math-errno -funroll-loops -fno-trapping-math -D_FORTIFY_SOURCE=2)
endif()

# library sources
set(AXIOM_SOURCES
    src/core/error.c
    src/core/memory.c
    src/core/tensor.c
    src/core/broadcast.c
    src/core/ops.c
    src/core/autograd.c
    src/core/autograd_ops.c
    src/core/activations.c
    src/core/losses.c
    src/core/optim.c
    src/core/init.c
    src/core/layer.c
    src/core/model.c
    src/core/serialize.c
    src/core/data.c
    src/core/conv.c
    src/core/norm.c
    src/core/lr_scheduler.c
    src/core/rng.c
    src/compute/dispatch.c
    src/compute/backends/cpu_naive.c
    src/compute/backends/cpu_opt.c
)

if(AX_CUDA)
    # cuda backend is a self-contained module under backends/cuda/
    set(AX_CUDA_SOURCES
        src/compute/backends/cuda/backend.cu
        src/compute/backends/cuda/memory.cu
        src/compute/backends/cuda/ops_elementwise.cu
        src/compute/backends/cuda/ops_reduce.cu
        src/compute/backends/cuda/ops_gemm.cu
        src/compute/backends/cuda/ops_conv.cu
    )
    list(APPEND AXIOM_SOURCES ${AX_CUDA_SOURCES})
endif()

# enable SIMD for the optimized backend on x86
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86")
    set_source_files_properties(src/compute/backends/cpu_opt.c
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()

# static library
add_library(axiom_static STATIC ${AXIOM_SOURCES})
target_include_directories(axiom_static PUBLIC include)
target_link_libraries(axiom_static PUBLIC m)
if(AX_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(axiom_static PUBLIC OpenMP::OpenMP_C)
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
target_include_directories(axiom_shared PUBLIC include)
target_link_libraries(axiom_shared PUBLIC m)
if(AX_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(axiom_shared PUBLIC OpenMP::OpenMP_C)
endif()
if(AX_CUDA)
    target_link_libraries(axiom_shared PUBLIC CUDA::cudart CUDA::cublas)
endif()
set_target_properties(axiom_shared PROPERTIES OUTPUT_NAME axiom)

# tests
enable_testing()

set(TEST_SOURCES
    tests/test_error.c
    tests/test_memory.c
    tests/test_tensor.c
    tests/test_compute.c
    tests/test_autograd.c
    tests/test_ops.c
    tests/test_activations.c
    tests/test_losses.c
    tests/test_optim.c
    tests/test_layer.c
    tests/test_serialize.c
    tests/test_data.c
    tests/test_conv.c
    tests/test_norm.c
    tests/test_integration.c
    tests/test_grad_verify.c
    tests/test_serialize_all.c
    tests/test_lr_sched.c
    tests/test_backend_opt.c
    tests/test_inplace.c
    tests/test_cuda.c
)

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
    # tests intentionally don't clean up every intermediate tensor in the
    # computation graph (that's the model api's job via ax_graph_cleanup).
    # suppress leak detection for tests so asan only catches real bugs.
    if(AX_SANITIZE)
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endif()
    if(AX_TSAN)
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT "TSAN_OPTIONS=ignore_noninstrumented_modules=1 suppressions=${CMAKE_SOURCE_DIR}/.tsan_suppressions")
    endif()
endforeach()

# examples (leak detection stays on for these — they use the full api)
add_executable(example_mnist examples/mnist.c)
target_link_libraries(example_mnist PRIVATE axiom_static)

add_executable(mnist_cnn examples/mnist_cnn.c)
target_link_libraries(mnist_cnn PRIVATE axiom_static)

add_executable(deep_mlp examples/deep_mlp.c)
target_link_libraries(deep_mlp PRIVATE axiom_static)

add_executable(bench_backend examples/bench_backend.c)
target_link_libraries(bench_backend PRIVATE axiom_static)

add_executable(bench_optimizations examples/bench_optimizations.c)
target_link_libraries(bench_optimizations PRIVATE axiom_static)

# convenience target: run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAMES}
    COMMENT "running all axiom tests"
)
