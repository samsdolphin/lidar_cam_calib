cmake_minimum_required(VERSION 3.1)
project(marker_detector)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  message_filters
)

find_package(OpenCV REQUIRED)

#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

find_package(aruco REQUIRED)

#catkin_package()

add_executable(aruco src/aruco_detector.cpp)
target_link_libraries(aruco ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(checkerboard src/checkerboard.cpp src/camera_calibration.cpp)
target_link_libraries(checkerboard ${catkin_LIBRARIES} ${OpenCV_LIBS})
