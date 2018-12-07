cmake_minimum_required( VERSION 2.8 )
project( orb_extractor )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11" )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

find_package( OpenCV 3.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( ${PROJECT_SOURCE_DIR}/src )

#add_library( orb_extractor src/orb_extractor.cpp )

add_executable( orb_extractor src/main.cpp src/orb_extractor.cpp )
target_link_libraries( orb_extractor ${OpenCV_LIBS} )
