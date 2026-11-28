cmake_minimum_required(VERSION 3.16)
project(matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)  # PoseLib needs it but does not find it itself
find_package(PoseLib REQUIRED)

add_executable(matching src/matching.cpp)
target_link_libraries(matching PRIVATE ${OpenCV_LIBS} Threads::Threads PoseLib::PoseLib)

# Optional: ONNX Runtime (C++ API) for the learned features ALIKED and XFeat
# (--detector aliked | xfeat). Without it only ORB and SIFT are available.
# The ONNX models are exported once with models/export_models.sh.
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime)
find_library(ONNXRUNTIME_LIBRARY onnxruntime)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  message(STATUS "ONNX Runtime found: ALIKED and XFeat enabled (${ONNXRUNTIME_LIBRARY})")
  target_compile_definitions(matching PRIVATE COLMAP_ODOMETRY_WITH_ONNX
      COLMAP_ODOMETRY_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")
  target_include_directories(matching PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(matching PRIVATE ${ONNXRUNTIME_LIBRARY} ${CMAKE_DL_LIBS})
else()
  message(STATUS "ONNX Runtime not found: only ORB and SIFT (no ALIKED / XFeat)")
endif()
