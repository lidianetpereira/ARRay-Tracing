# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/lidiane/Downloads/CLion-2019.3.5/clion-2019.3.5/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/lidiane/Downloads/CLion-2019.3.5/clion-2019.3.5/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lidiane/CLionProjects/optix/SDK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug

# Include any dependencies generated for this target.
include sutil/CMakeFiles/sutil_sdk.dir/depend.make

# Include the progress variables for this target.
include sutil/CMakeFiles/sutil_sdk.dir/progress.make

# Include the compile flags for this target's objects.
include sutil/CMakeFiles/sutil_sdk.dir/flags.make

sutil/CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.o: ../sutil/rply-1.01/rply.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object sutil/CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.o   -c /home/lidiane/CLionProjects/optix/SDK/sutil/rply-1.01/rply.c

sutil/CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/rply-1.01/rply.c > CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.i

sutil/CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/rply-1.01/rply.c -o CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.s

sutil/CMakeFiles/sutil_sdk.dir/Arcball.cpp.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/Arcball.cpp.o: ../sutil/Arcball.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object sutil/CMakeFiles/sutil_sdk.dir/Arcball.cpp.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutil_sdk.dir/Arcball.cpp.o -c /home/lidiane/CLionProjects/optix/SDK/sutil/Arcball.cpp

sutil/CMakeFiles/sutil_sdk.dir/Arcball.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutil_sdk.dir/Arcball.cpp.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/Arcball.cpp > CMakeFiles/sutil_sdk.dir/Arcball.cpp.i

sutil/CMakeFiles/sutil_sdk.dir/Arcball.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutil_sdk.dir/Arcball.cpp.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/Arcball.cpp -o CMakeFiles/sutil_sdk.dir/Arcball.cpp.s

sutil/CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.o: ../sutil/HDRLoader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object sutil/CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.o -c /home/lidiane/CLionProjects/optix/SDK/sutil/HDRLoader.cpp

sutil/CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/HDRLoader.cpp > CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.i

sutil/CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/HDRLoader.cpp -o CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.s

sutil/CMakeFiles/sutil_sdk.dir/Mesh.cpp.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/Mesh.cpp.o: ../sutil/Mesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object sutil/CMakeFiles/sutil_sdk.dir/Mesh.cpp.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutil_sdk.dir/Mesh.cpp.o -c /home/lidiane/CLionProjects/optix/SDK/sutil/Mesh.cpp

sutil/CMakeFiles/sutil_sdk.dir/Mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutil_sdk.dir/Mesh.cpp.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/Mesh.cpp > CMakeFiles/sutil_sdk.dir/Mesh.cpp.i

sutil/CMakeFiles/sutil_sdk.dir/Mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutil_sdk.dir/Mesh.cpp.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/Mesh.cpp -o CMakeFiles/sutil_sdk.dir/Mesh.cpp.s

sutil/CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.o: ../sutil/OptiXMesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object sutil/CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.o -c /home/lidiane/CLionProjects/optix/SDK/sutil/OptiXMesh.cpp

sutil/CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/OptiXMesh.cpp > CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.i

sutil/CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/OptiXMesh.cpp -o CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.s

sutil/CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.o: ../sutil/PPMLoader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object sutil/CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.o -c /home/lidiane/CLionProjects/optix/SDK/sutil/PPMLoader.cpp

sutil/CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/PPMLoader.cpp > CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.i

sutil/CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/PPMLoader.cpp -o CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.s

sutil/CMakeFiles/sutil_sdk.dir/sutil.cpp.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/sutil.cpp.o: ../sutil/sutil.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object sutil/CMakeFiles/sutil_sdk.dir/sutil.cpp.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutil_sdk.dir/sutil.cpp.o -c /home/lidiane/CLionProjects/optix/SDK/sutil/sutil.cpp

sutil/CMakeFiles/sutil_sdk.dir/sutil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutil_sdk.dir/sutil.cpp.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/sutil.cpp > CMakeFiles/sutil_sdk.dir/sutil.cpp.i

sutil/CMakeFiles/sutil_sdk.dir/sutil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutil_sdk.dir/sutil.cpp.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/sutil.cpp -o CMakeFiles/sutil_sdk.dir/sutil.cpp.s

sutil/CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.o: ../sutil/tinyobjloader/tiny_obj_loader.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object sutil/CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.o -c /home/lidiane/CLionProjects/optix/SDK/sutil/tinyobjloader/tiny_obj_loader.cc

sutil/CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/CLionProjects/optix/SDK/sutil/tinyobjloader/tiny_obj_loader.cc > CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.i

sutil/CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/CLionProjects/optix/SDK/sutil/tinyobjloader/tiny_obj_loader.cc -o CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.s

sutil/CMakeFiles/sutil_sdk.dir/glew.c.o: sutil/CMakeFiles/sutil_sdk.dir/flags.make
sutil/CMakeFiles/sutil_sdk.dir/glew.c.o: ../sutil/glew.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object sutil/CMakeFiles/sutil_sdk.dir/glew.c.o"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -Wno-strict-prototypes -Wno-missing-prototypes -o CMakeFiles/sutil_sdk.dir/glew.c.o   -c /home/lidiane/CLionProjects/optix/SDK/sutil/glew.c

sutil/CMakeFiles/sutil_sdk.dir/glew.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sutil_sdk.dir/glew.c.i"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -Wno-strict-prototypes -Wno-missing-prototypes -E /home/lidiane/CLionProjects/optix/SDK/sutil/glew.c > CMakeFiles/sutil_sdk.dir/glew.c.i

sutil/CMakeFiles/sutil_sdk.dir/glew.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sutil_sdk.dir/glew.c.s"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -Wno-strict-prototypes -Wno-missing-prototypes -S /home/lidiane/CLionProjects/optix/SDK/sutil/glew.c -o CMakeFiles/sutil_sdk.dir/glew.c.s

# Object files for target sutil_sdk
sutil_sdk_OBJECTS = \
"CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.o" \
"CMakeFiles/sutil_sdk.dir/Arcball.cpp.o" \
"CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.o" \
"CMakeFiles/sutil_sdk.dir/Mesh.cpp.o" \
"CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.o" \
"CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.o" \
"CMakeFiles/sutil_sdk.dir/sutil.cpp.o" \
"CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.o" \
"CMakeFiles/sutil_sdk.dir/glew.c.o"

# External object files for target sutil_sdk
sutil_sdk_EXTERNAL_OBJECTS =

lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/rply-1.01/rply.c.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/Arcball.cpp.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/HDRLoader.cpp.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/Mesh.cpp.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/OptiXMesh.cpp.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/PPMLoader.cpp.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/sutil.cpp.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/tinyobjloader/tiny_obj_loader.cc.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/glew.c.o
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/build.make
lib/libsutil_sdk.so: /home/lidiane/CLionProjects/optix/lib64/liboptix.so
lib/libsutil_sdk.so: /usr/local/lib/libglut.so
lib/libsutil_sdk.so: /usr/lib/x86_64-linux-gnu/libXmu.so
lib/libsutil_sdk.so: /usr/lib/x86_64-linux-gnu/libXi.so
lib/libsutil_sdk.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
lib/libsutil_sdk.so: /usr/lib/x86_64-linux-gnu/libGLX.so
lib/libsutil_sdk.so: /usr/lib/x86_64-linux-gnu/libGLU.so
lib/libsutil_sdk.so: /usr/local/cuda/lib64/libnvrtc.so
lib/libsutil_sdk.so: sutil/CMakeFiles/sutil_sdk.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library ../lib/libsutil_sdk.so"
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sutil_sdk.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sutil/CMakeFiles/sutil_sdk.dir/build: lib/libsutil_sdk.so

.PHONY : sutil/CMakeFiles/sutil_sdk.dir/build

sutil/CMakeFiles/sutil_sdk.dir/clean:
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil && $(CMAKE_COMMAND) -P CMakeFiles/sutil_sdk.dir/cmake_clean.cmake
.PHONY : sutil/CMakeFiles/sutil_sdk.dir/clean

sutil/CMakeFiles/sutil_sdk.dir/depend:
	cd /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lidiane/CLionProjects/optix/SDK /home/lidiane/CLionProjects/optix/SDK/sutil /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil /home/lidiane/CLionProjects/optix/SDK/cmake-build-debug/sutil/CMakeFiles/sutil_sdk.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sutil/CMakeFiles/sutil_sdk.dir/depend

