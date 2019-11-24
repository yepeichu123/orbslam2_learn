# Install script for directory: /home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libDBoW3.so.0.0.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libDBoW3.so.0.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libDBoW3.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/build/src/libDBoW3.so.0.0.1"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/build/src/libDBoW3.so.0.0"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/build/src/libDBoW3.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libDBoW3.so.0.0.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libDBoW3.so.0.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libDBoW3.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/opt/ros/indigo/lib:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/DBoW3" TYPE FILE FILES
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/BowVector.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/DBoW3.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/Database.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/DescManip.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/FeatureVector.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/QueryResults.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/ScoringObject.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/Vocabulary.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/exports.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/quicklz.h"
    "/home/xiaoc/xiaoC/xiaoC/slam/code/blog/orbslam2_learn/feature_matching/thirdparty/DBoW3/src/timers.h"
    )
endif()

