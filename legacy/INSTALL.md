To use the visualizer on Ubuntu, building raylib is required

1. Initialize the submodules: git submodule update --init --recursive

2. Install the dependencies:

sudo apt install libasound2-dev libx11-dev libxrandr-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev libwayland-dev libxkbcommon-dev

3. Build the static version of the library:

cd raylib/src/
make PLATFORM=PLATFORM_DESKTOP 

