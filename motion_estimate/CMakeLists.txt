cmake_minimum_required( VERSION 2.8 )
project( motion_estimate )

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# costum modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 3.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

include_directories( ${PROJECT_SOURCE_DIR}/include 
					 ${PROJECT_SOURCE_DIR}/thirdparty/
					 ${PROJECT_SOURCE_DIR}/thirdparty/g2o/
					 ${OpenCV_INCLUDE_DIR} 
					 ${EIGEN3_INCLUDE_DIR}
					 ${Sophus_INCLUDE_DIRS})

set( THIRD_PARTY_LIBS 
	 ${OpenCV_LIBS} 
	 ${Sophus_LIBRARIES} 
	 ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o${CMAKE_SHARED_LIBRARY_SUFFIX} )

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
