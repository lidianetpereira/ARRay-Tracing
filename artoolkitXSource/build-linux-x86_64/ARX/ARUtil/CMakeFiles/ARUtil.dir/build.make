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
include ARX/ARUtil/CMakeFiles/ARUtil.dir/depend.make

# Include the progress variables for this target.
include ARX/ARUtil/CMakeFiles/ARUtil.dir/progress.make

# Include the compile flags for this target's objects.
include ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make

ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o: ../ARX/ARUtil/log.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/log.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/log.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/log.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/log.c > CMakeFiles/ARUtil.dir/log.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/log.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/log.c -o CMakeFiles/ARUtil.dir/log.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o: ../ARX/ARUtil/profile.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/profile.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/profile.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/profile.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/profile.c > CMakeFiles/ARUtil.dir/profile.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/profile.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/profile.c -o CMakeFiles/ARUtil.dir/profile.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o: ../ARX/ARUtil/thread_sub_winrt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/thread_sub_winrt.cpp

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/thread_sub_winrt.cpp > CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/thread_sub_winrt.cpp -o CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o: ../ARX/ARUtil/thread_sub.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/thread_sub.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/thread_sub.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/thread_sub.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/thread_sub.c > CMakeFiles/ARUtil.dir/thread_sub.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/thread_sub.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/thread_sub.c -o CMakeFiles/ARUtil.dir/thread_sub.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o: ../ARX/ARUtil/system.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/system.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/system.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/system.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/system.c > CMakeFiles/ARUtil.dir/system.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/system.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/system.c -o CMakeFiles/ARUtil.dir/system.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o: ../ARX/ARUtil/android_system_property_get.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/android_system_property_get.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/android_system_property_get.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/android_system_property_get.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/android_system_property_get.c > CMakeFiles/ARUtil.dir/android_system_property_get.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/android_system_property_get.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/android_system_property_get.c -o CMakeFiles/ARUtil.dir/android_system_property_get.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o: ../ARX/ARUtil/time.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/time.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/time.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/time.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/time.c > CMakeFiles/ARUtil.dir/time.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/time.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/time.c -o CMakeFiles/ARUtil.dir/time.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o: ../ARX/ARUtil/file_utils.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/file_utils.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/file_utils.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/file_utils.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/file_utils.c > CMakeFiles/ARUtil.dir/file_utils.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/file_utils.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/file_utils.c -o CMakeFiles/ARUtil.dir/file_utils.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o: ../ARX/ARUtil/image_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ARUtil.dir/image_utils.cpp.o -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/image_utils.cpp

ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ARUtil.dir/image_utils.cpp.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/image_utils.cpp > CMakeFiles/ARUtil.dir/image_utils.cpp.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ARUtil.dir/image_utils.cpp.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/image_utils.cpp -o CMakeFiles/ARUtil.dir/image_utils.cpp.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o: ../ARX/ARUtil/crypt.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/crypt.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/crypt.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/crypt.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/crypt.c > CMakeFiles/ARUtil.dir/crypt.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/crypt.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/crypt.c -o CMakeFiles/ARUtil.dir/crypt.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o: ../ARX/ARUtil/ioapi.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/ioapi.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/ioapi.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/ioapi.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/ioapi.c > CMakeFiles/ARUtil.dir/ioapi.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/ioapi.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/ioapi.c -o CMakeFiles/ARUtil.dir/ioapi.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o: ../ARX/ARUtil/unzip.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/unzip.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/unzip.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/unzip.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/unzip.c > CMakeFiles/ARUtil.dir/unzip.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/unzip.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/unzip.c -o CMakeFiles/ARUtil.dir/unzip.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o


ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o: ARX/ARUtil/CMakeFiles/ARUtil.dir/flags.make
ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o: ../ARX/ARUtil/zip.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building C object ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ARUtil.dir/zip.c.o   -c /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/zip.c

ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ARUtil.dir/zip.c.i"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/zip.c > CMakeFiles/ARUtil.dir/zip.c.i

ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ARUtil.dir/zip.c.s"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil/zip.c -o CMakeFiles/ARUtil.dir/zip.c.s

ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.requires:

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.provides: ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.requires
	$(MAKE) -f ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.provides.build
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.provides

ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.provides.build: ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o


# Object files for target ARUtil
ARUtil_OBJECTS = \
"CMakeFiles/ARUtil.dir/log.c.o" \
"CMakeFiles/ARUtil.dir/profile.c.o" \
"CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o" \
"CMakeFiles/ARUtil.dir/thread_sub.c.o" \
"CMakeFiles/ARUtil.dir/system.c.o" \
"CMakeFiles/ARUtil.dir/android_system_property_get.c.o" \
"CMakeFiles/ARUtil.dir/time.c.o" \
"CMakeFiles/ARUtil.dir/file_utils.c.o" \
"CMakeFiles/ARUtil.dir/image_utils.cpp.o" \
"CMakeFiles/ARUtil.dir/crypt.c.o" \
"CMakeFiles/ARUtil.dir/ioapi.c.o" \
"CMakeFiles/ARUtil.dir/unzip.c.o" \
"CMakeFiles/ARUtil.dir/zip.c.o"

# External object files for target ARUtil
ARUtil_EXTERNAL_OBJECTS =

ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/build.make
ARX/ARUtil/libARUtil.a: ARX/ARUtil/CMakeFiles/ARUtil.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX static library libARUtil.a"
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && $(CMAKE_COMMAND) -P CMakeFiles/ARUtil.dir/cmake_clean_target.cmake
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ARUtil.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ARX/ARUtil/CMakeFiles/ARUtil.dir/build: ARX/ARUtil/libARUtil.a

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/build

ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/log.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/profile.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub_winrt.cpp.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/thread_sub.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/system.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/android_system_property_get.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/time.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/file_utils.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/image_utils.cpp.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/crypt.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/ioapi.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/unzip.c.o.requires
ARX/ARUtil/CMakeFiles/ARUtil.dir/requires: ARX/ARUtil/CMakeFiles/ARUtil.dir/zip.c.o.requires

.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/requires

ARX/ARUtil/CMakeFiles/ARUtil.dir/clean:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil && $(CMAKE_COMMAND) -P CMakeFiles/ARUtil.dir/cmake_clean.cmake
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/clean

ARX/ARUtil/CMakeFiles/ARUtil.dir/depend:
	cd /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lidiane/ARRay-TracingGit/artoolkitXSource /home/lidiane/ARRay-TracingGit/artoolkitXSource/ARX/ARUtil /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64 /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil /home/lidiane/ARRay-TracingGit/artoolkitXSource/build-linux-x86_64/ARX/ARUtil/CMakeFiles/ARUtil.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ARX/ARUtil/CMakeFiles/ARUtil.dir/depend

