cmake_minimum_required(VERSION 3.16)
project(axiom C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

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
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE tests)
    target_link_libraries(${test_name} PRIVATE axiom_static)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# convenience target: run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAMES}
    COMMENT "running all axiom tests"
)
