# Install script for directory: /home/lidiane/Downloads/artoolkitx-master/Source/Utilities

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/lidiane/Downloads/artoolkitx-master/Source/../SDK")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/check_id/cmake_install.cmake")
  include("/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genMarkerSet/cmake_install.cmake")
  include("/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/mk_patt/cmake_install.cmake")
  include("/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/checkResolution/cmake_install.cmake")
  include("/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/genTexData/cmake_install.cmake")
  include("/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/dispTexData/cmake_install.cmake")
  include("/home/lidiane/Downloads/artoolkitx-master/Source/build-linux-x86_64/Utilities/image_database_2d/cmake_install.cmake")

endif()
