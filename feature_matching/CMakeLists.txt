cmake_minimum_required( VERSION 2.8 )
project( feature_matching )

set( CMAKE_BUILD_TYPE "-Release" )
set( CMAKE_CXX_FLAGS "-std=c++11" )

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

# opencv3
find_package( OpenCV 3.0 REQUIRED ) 

include_directories( ${PROJECT_SOURCE_DIR}/include 
					 ${PROJECT_SOURCE_DIR}/thirdparty/
		             ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/
		             ${OpenCV_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/build/src/libDBoW3${CMAKE_SHARED_LIBRARY_SUFFIX} )

add_subdirectory( ${PROJECT_SOURCE_DIR}/src )

add_executable(feature_matching src/main.cpp src/FeatureMatching.cpp)
target_link_libraries(feature_matching ${OpenCV_LIBS} ${THIRD_PARTY_LIBS})