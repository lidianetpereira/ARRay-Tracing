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
CMAKE_SOURCE_DIR = /home/lidiane/Downloads/artoolkitx-master/Source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64

# Include any dependencies generated for this target.
include Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/depend.make

# Include the progress variables for this target.
include Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/progress.make

# Include the compile flags for this target's objects.
include Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/flags.make

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o: Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/flags.make
Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o: ../Utilities/genTexData/genTexData.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o"
	cd /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o   -c /home/lidiane/Downloads/artoolkitx-master/Source/Utilities/genTexData/genTexData.c

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.i"
	cd /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Downloads/artoolkitx-master/Source/Utilities/genTexData/genTexData.c > CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.i

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.s"
	cd /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Downloads/artoolkitx-master/Source/Utilities/genTexData/genTexData.c -o CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.s

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.requires:

.PHONY : Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.requires

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.provides: Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.requires
	$(MAKE) -f Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/build.make Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.provides.build
.PHONY : Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.provides

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.provides.build: Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o


# Object files for target artoolkitx_genTexData
artoolkitx_genTexData_OBJECTS = \
"CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o"

# External object files for target artoolkitx_genTexData
artoolkitx_genTexData_EXTERNAL_OBJECTS =

Utilities/genTexData/artoolkitx_genTexData: Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o
Utilities/genTexData/artoolkitx_genTexData: Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/build.make
Utilities/genTexData/artoolkitx_genTexData: ARX/AR2/libAR2.a
Utilities/genTexData/artoolkitx_genTexData: ARX/libARX.so.1.0.6
Utilities/genTexData/artoolkitx_genTexData: /usr/lib/x86_64-linux-gnu/libjpeg.so
Utilities/genTexData/artoolkitx_genTexData: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/genTexData/artoolkitx_genTexData: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/genTexData/artoolkitx_genTexData: Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable artoolkitx_genTexData"
	cd /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/artoolkitx_genTexData.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/build: Utilities/genTexData/artoolkitx_genTexData

.PHONY : Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/build

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/requires: Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/genTexData.c.o.requires

.PHONY : Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/requires

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/clean:
	cd /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData && $(CMAKE_COMMAND) -P CMakeFiles/artoolkitx_genTexData.dir/cmake_clean.cmake
.PHONY : Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/clean

Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/depend:
	cd /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lidiane/Downloads/artoolkitx-master/Source /home/lidiane/Downloads/artoolkitx-master/Source/Utilities/genTexData /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64 /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData /home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Utilities/genTexData/CMakeFiles/artoolkitx_genTexData.dir/depend

