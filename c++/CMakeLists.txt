cmake_minimum_required(VERSION 3.16)
project(orb_matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(orb_matching src/orb_matching.cpp)
target_link_libraries(orb_matching PRIVATE ${OpenCV_LIBS} Threads::Threads)
