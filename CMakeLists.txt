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
if(AX_SANITIZE)
    message(STATUS "sanitizers enabled (asan + ubsan)")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
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
    src/compute/dispatch.c
    src/compute/backends/cpu_naive.c
)

# static library
add_library(axiom_static STATIC ${AXIOM_SOURCES})
target_include_directories(axiom_static PUBLIC include)
target_link_libraries(axiom_static PUBLIC m)
set_target_properties(axiom_static PROPERTIES OUTPUT_NAME axiom)

# shared library
add_library(axiom_shared SHARED ${AXIOM_SOURCES})
target_include_directories(axiom_shared PUBLIC include)
target_link_libraries(axiom_shared PUBLIC m)
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
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE tests)
    target_link_libraries(${test_name} PRIVATE axiom_static)
    add_test(NAME ${test_name} COMMAND ${test_name})
    # tests intentionally don't clean up every intermediate tensor in the
    # computation graph (that's the model api's job via ax_graph_cleanup).
    # suppress leak detection for tests so asan only catches real bugs
    # (use-after-free, buffer overflow, etc.)
    if(AX_SANITIZE)
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endif()
endforeach()

# examples (leak detection stays on for these — they use the full api)
add_executable(example_xor examples/xor.c)
target_link_libraries(example_xor PRIVATE axiom_static)

# convenience target: run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAMES}
    COMMENT "running all axiom tests"
)
