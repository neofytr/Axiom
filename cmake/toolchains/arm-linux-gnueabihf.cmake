# cross-compile toolchain for 32-bit arm linux targets with hardfp.
# raspberry pi zero 2, pi 3 (32-bit os), beaglebone, older allwinner
# and rockchip boards, armv7 imx. anywhere you need a 32-bit armv7a
# binary with neon.
#
# install on ubuntu/debian:
#   sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
#
# usage:
#   cmake -DAX_PROFILE=embedded-linux \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-linux-gnueabihf.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# cortex-a7 baseline covers pi 2, pi zero 2, beaglebone black.
# neon + vfpv4 is standard on these. bump to cortex-a15 / cortex-a53
# for faster chips.
set(CMAKE_C_FLAGS_INIT "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mtune=cortex-a7")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
