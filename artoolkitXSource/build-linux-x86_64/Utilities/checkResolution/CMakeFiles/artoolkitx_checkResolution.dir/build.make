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
include Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/depend.make

# Include the progress variables for this target.
include Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/progress.make

# Include the compile flags for this target's objects.
include Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/flags.make

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o: Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/flags.make
Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o: ../Utilities/checkResolution/checkResolution.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/checkResolution && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/checkResolution/checkResolution.c

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/checkResolution && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/checkResolution/checkResolution.c > CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.i

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/checkResolution && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/checkResolution/checkResolution.c -o CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.s

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.requires:

.PHONY : Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.requires

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.provides: Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.requires
	$(MAKE) -f Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/build.make Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.provides.build
.PHONY : Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.provides

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.provides.build: Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o


# Object files for target artoolkitx_checkResolution
artoolkitx_checkResolution_OBJECTS = \
"CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o"

# External object files for target artoolkitx_checkResolution
artoolkitx_checkResolution_EXTERNAL_OBJECTS =

Utilities/checkResolution/artoolkitx_checkResolution: Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o
Utilities/checkResolution/artoolkitx_checkResolution: Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/build.make
Utilities/checkResolution/artoolkitx_checkResolution: ARX/libARX.so.1.0.6
Utilities/checkResolution/artoolkitx_checkResolution: ARX/AR2/libAR2.a
Utilities/checkResolution/artoolkitx_checkResolution: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/checkResolution/artoolkitx_checkResolution: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/checkResolution/artoolkitx_checkResolution: /usr/lib/x86_64-linux-gnu/libjpeg.so
Utilities/checkResolution/artoolkitx_checkResolution: Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable artoolkitx_checkResolution"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/checkResolution && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/artoolkitx_checkResolution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/build: Utilities/checkResolution/artoolkitx_checkResolution

.PHONY : Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/build

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/requires: Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/checkResolution.c.o.requires

.PHONY : Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/requires

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/clean:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/checkResolution && $(CMAKE_COMMAND) -P CMakeFiles/artoolkitx_checkResolution.dir/cmake_clean.cmake
.PHONY : Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/clean

Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/depend:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lidiane/ARRay-TracingGit/artoolkitXSource /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/checkResolution /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/checkResolution /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Utilities/checkResolution/CMakeFiles/artoolkitx_checkResolution.dir/depend

