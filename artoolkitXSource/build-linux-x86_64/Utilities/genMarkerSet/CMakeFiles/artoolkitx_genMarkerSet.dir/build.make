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
include Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/depend.make

# Include the progress variables for this target.
include Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/progress.make

# Include the compile flags for this target's objects.
include Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/flags.make

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/flags.make
Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o: ../Utilities/genMarkerSet/genMarkerSet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/genMarkerSet/genMarkerSet.cpp

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/genMarkerSet/genMarkerSet.cpp > CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.i

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/genMarkerSet/genMarkerSet.cpp -o CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.s

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.requires:

.PHONY : Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.requires

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.provides: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.requires
	$(MAKE) -f Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/build.make Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.provides.build
.PHONY : Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.provides

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.provides.build: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o


# Object files for target artoolkitx_genMarkerSet
artoolkitx_genMarkerSet_OBJECTS = \
"CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o"

# External object files for target artoolkitx_genMarkerSet
artoolkitx_genMarkerSet_EXTERNAL_OBJECTS =

Utilities/genMarkerSet/artoolkitx_genMarkerSet: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o
Utilities/genMarkerSet/artoolkitx_genMarkerSet: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/build.make
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ARX/AR/libAR.a
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ARX/libARX.so.1.0.6
Utilities/genMarkerSet/artoolkitx_genMarkerSet: depends/common/src/Eden/libEden.a
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/local/lib/libSDL2main.a
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/local/lib/libSDL2.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_calib3d.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_features2d.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_flann.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_highgui.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_videoio.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_imgcodecs.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_video.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_imgproc.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_core.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libjpeg.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGLESv2.so
Utilities/genMarkerSet/artoolkitx_genMarkerSet: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable artoolkitx_genMarkerSet"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/artoolkitx_genMarkerSet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/build: Utilities/genMarkerSet/artoolkitx_genMarkerSet

.PHONY : Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/build

# Object files for target artoolkitx_genMarkerSet
artoolkitx_genMarkerSet_OBJECTS = \
"CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o"

# External object files for target artoolkitx_genMarkerSet
artoolkitx_genMarkerSet_EXTERNAL_OBJECTS =

Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/build.make
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ARX/AR/libAR.a
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ARX/libARX.so.1.0.6
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: depends/common/src/Eden/libEden.a
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/local/lib/libSDL2main.a
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/local/lib/libSDL2.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_calib3d.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_features2d.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_flann.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_highgui.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_videoio.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_imgcodecs.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_video.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_imgproc.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: ../depends/linux/lib/libopencv_core.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libjpeg.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGL.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGLU.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: /usr/lib/x86_64-linux-gnu/libGLESv2.so
Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/relink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/artoolkitx_genMarkerSet.dir/relink.txt --verbose=$(VERBOSE)

# Rule to relink during preinstall.
Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/preinstall: Utilities/genMarkerSet/CMakeFiles/CMakeRelink.dir/artoolkitx_genMarkerSet

.PHONY : Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/preinstall

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/requires: Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/genMarkerSet.cpp.o.requires

.PHONY : Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/requires

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/clean:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet && $(CMAKE_COMMAND) -P CMakeFiles/artoolkitx_genMarkerSet.dir/cmake_clean.cmake
.PHONY : Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/clean

Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/depend:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lidiane/ARRay-TracingGit/artoolkitXSource /home/lidiane/ARRay-TracingGit/artoolkitXSource/Utilities/genMarkerSet /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Utilities/genMarkerSet/CMakeFiles/artoolkitx_genMarkerSet.dir/depend

