
add_executable(motion_estimate main.cpp MotionEstimate.cpp Frame.cpp G2OType.cpp)
target_link_libraries(motion_estimate ${THIRD_PARTY_LIBS})