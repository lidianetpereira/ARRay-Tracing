# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lidiane/ARRay-TracingGit/artoolkitXSource

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64

# Include any dependencies generated for this target.
include Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/depend.make

# Include the progress variables for this target.
include Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/progress.make

# Include the compile flags for this target's objects.
include Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/flags.make

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o: Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/flags.make
Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o: ../Utilities/dispTexData/dispTexData.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/dispTexData && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/dispTexData/dispTexData.cpp

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/dispTexData && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/dispTexData/dispTexData.cpp > CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.i

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/dispTexData && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/dispTexData/dispTexData.cpp -o CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.s

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.requires:

.PHONY : Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.requires

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.provides: Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.requires
	$(MAKE) -f Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/build.make Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.provides.build
.PHONY : Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.provides

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.provides.build: Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o


# Object files for target artoolkitx_dispTexData
artoolkitx_dispTexData_OBJECTS = \
"CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o"

# External object files for target artoolkitx_dispTexData
artoolkitx_dispTexData_EXTERNAL_OBJECTS =

Utilities/dispTexData/artoolkitx_dispTexData: Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o
Utilities/dispTexData/artoolkitx_dispTexData: Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/build.make
Utilities/dispTexData/artoolkitx_dispTexData: ARX/libARX.so.1.0.6
Utilities/dispTexData/artoolkitx_dispTexData: depends/common/src/Eden/libEden.a
Utilities/dispTexData/artoolkitx_dispTexData: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/dispTexData/artoolkitx_dispTexData: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/dispTexData/artoolkitx_dispTexData: /usr/local/lib/libSDL2main.a
Utilities/dispTexData/artoolkitx_dispTexData: /usr/local/lib/libSDL2.so
Utilities/dispTexData/artoolkitx_dispTexData: /usr/lib/x86_64-linux-gnu/libjpeg.so
Utilities/dispTexData/artoolkitx_dispTexData: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/dispTexData/artoolkitx_dispTexData: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/dispTexData/artoolkitx_dispTexData: /usr/lib/x86_64-linux-gnu/libGLESv2.so
Utilities/dispTexData/artoolkitx_dispTexData: Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable artoolkitx_dispTexData"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/dispTexData && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/artoolkitx_dispTexData.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/build: Utilities/dispTexData/artoolkitx_dispTexData

.PHONY : Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/build

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/requires: Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/dispTexData.cpp.o.requires

.PHONY : Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/requires

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/clean:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/dispTexData && $(CMAKE_COMMAND) -P CMakeFiles/artoolkitx_dispTexData.dir/cmake_clean.cmake
.PHONY : Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/clean

Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/depend:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lidiane/ARRay-TracingGit/artoolkitXSource /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/dispTexData /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/dispTexData /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Utilities/dispTexData/CMakeFiles/artoolkitx_dispTexData.dir/depend

