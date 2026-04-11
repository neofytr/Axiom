cmake_minimum_required(VERSION 3.16)
project(axiom C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# build type: Debug (sanitizers), Release (optimized), or default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

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

# openmp parallelism: cmake -DAX_OPENMP=ON .. (default ON for desktop)
option(AX_OPENMP "enable openmp multi-core parallelism" ON)
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

# hardening flags for release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2 -D_FORTIFY_SOURCE=2)
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
set_target_properties(axiom_static PROPERTIES OUTPUT_NAME axiom)

# shared library
add_library(axiom_shared SHARED ${AXIOM_SOURCES})
target_include_directories(axiom_shared PUBLIC include)
target_link_libraries(axiom_shared PUBLIC m)
if(AX_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(axiom_shared PUBLIC OpenMP::OpenMP_C)
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
add_executable(example_xor examples/xor.c)
target_link_libraries(example_xor PRIVATE axiom_static)

add_executable(example_mnist examples/mnist.c)
target_link_libraries(example_mnist PRIVATE axiom_static)

add_executable(example_spiral examples/spiral.c)
target_link_libraries(example_spiral PRIVATE axiom_static)

add_executable(example_regression examples/regression.c)
target_link_libraries(example_regression PRIVATE axiom_static)

add_executable(bench_backend examples/bench_backend.c)
target_link_libraries(bench_backend PRIVATE axiom_static)

add_executable(mnist_cnn examples/mnist_cnn.c)
target_link_libraries(mnist_cnn PRIVATE axiom_static)

# convenience target: run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAMES}
    COMMENT "running all axiom tests"
)
