# cross-compile toolchain for 64-bit arm linux targets.
# raspberry pi 4/5 (64-bit os), jetson nano/orin, most modern sbcs.
#
# install on ubuntu/debian:
#   sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
#
# usage:
#   cmake -DAX_PROFILE=embedded-linux \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# cortex-a72 baseline is safe for pi 4, jetson nano, zynq ultrascale+.
# override to -mcpu=cortex-a76 for pi 5 / jetson orin for slightly better
# codegen, or leave as-is — the difference is small.
set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-a72 -mtune=cortex-a72")

# search for libraries / headers under the toolchain sysroot, not the host
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
