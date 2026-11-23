cmake_minimum_required(VERSION 3.16)
project(orb_matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)  # PoseLib needs it but does not find it itself
find_package(PoseLib REQUIRED)

add_executable(orb_matching src/orb_matching.cpp)
target_link_libraries(orb_matching PRIVATE ${OpenCV_LIBS} Threads::Threads PoseLib::PoseLib)
