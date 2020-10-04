cmake_minimum_required(VERSION 2.8.3)
project(plane_detector)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_filters
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

#find_package(aruco REQUIRED)

catkin_package()

add_executable(detect_plane src/plane_detector_node.cpp)
target_link_libraries(detect_plane ${catkin_LIBRARIES} ${OpenCV_LIBS} ${aruco_LIBS} ${PCL_LIBRARIES})
