# cross-compile toolchain for bare-metal / rtos arm cortex-m and
# cortex-a targets WITHOUT a hosted operating system.
#
# works for: freertos, zephyr, nuttx, bare metal, mbed os, stm32cube
# templates, picolibc-based projects.
#
# install on ubuntu/debian:
#   sudo apt install gcc-arm-none-eabi
#
# or grab the latest arm toolchain release from
# https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
#
# usage:
#   cmake -DAX_PROFILE=embedded-baremetal \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake ..
#
# you almost certainly need to adjust -mcpu / -mfpu for your specific
# target. the default here is cortex-m4 with hardware float — edit
# this file (or override at configure time via -DCMAKE_C_FLAGS="...")
# for other targets. common choices:
#
#   cortex-m4f      -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
#   cortex-m7f      -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard
#   cortex-m33      -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard
#   cortex-m55      -mcpu=cortex-m55 -mfpu=auto -mfloat-abi=hard
#   cortex-a9       -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard
#                   (zynq ps7 standalone etc.)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# skip the compiler link-test — bare-metal toolchains fail the default
# test because there's no crt0 on the search path for a linker probe.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_AR arm-none-eabi-ar)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP arm-none-eabi-objdump)

# default to cortex-m4f. override this line for your target.
set(CMAKE_C_FLAGS_INIT
    "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb")

# don't pull in hosted stdio by default — the embedded-baremetal profile
# already defines AX_NO_STDIO so axiom's own diagnostics are compiled out.
# your application still links its own libc (usually picolibc or
# newlib-nano via --specs=nano.specs), set in your application's
# CMakeLists after including axiom.

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
