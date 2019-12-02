cmake_minimum_required( VERSION 2.8 )
project( KeyframeSelect )

set(CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_FLAGS "-std=c++11" )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


find_package( OpenCV 3.2 REQUIRED )
include_directories( ${PROJECT_SOURCE_DIR}/include 
		     ${OpenCV_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS ${OpenCV_LIBS} )

add_subdirectory( ${PROJECT_SOURCE_DIR}/src )
