cmake_minimum_required( VERSION 2.8 )
project( linear_triangular )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11" )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

find_package( OpenCV 3.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( ${PROJECT_SOURCE_DIR}/src )

add_executable( linear_triangular src/linear_triangular.cpp src/main.cpp ) 
target_link_libraries( linear_triangular ${OpenCV_LIBS} ${PCL_LIBRARIES} ) 

