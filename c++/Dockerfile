# C++ development container with PoseLib and OpenCV
#
# Build (from inside the c++/ folder):
#   docker build -t cpp_poselib .
#
# Run, mounting the current folder as /workspace:
#   docker run -it --rm -v "$(pwd)":/workspace cpp_poselib
#
# In your CMakeLists.txt:
#   find_package(OpenCV REQUIRED)
#   find_package(PoseLib REQUIRED)
#   target_link_libraries(my_app PRIVATE ${OpenCV_LIBS} PoseLib::PoseLib)

FROM ubuntu:24.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# 1. Compiler, build tools, OpenCV and Eigen (PoseLib depends on Eigen)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        libeigen3-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Build and install PoseLib from source (pinned release)
ARG POSELIB_VERSION=v2.0.4
RUN git clone --depth 1 --branch ${POSELIB_VERSION} https://github.com/PoseLib/PoseLib.git /tmp/PoseLib \
    && cmake -S /tmp/PoseLib -B /tmp/PoseLib/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DMARCH_NATIVE=OFF \
    && cmake --build /tmp/PoseLib/build -j"$(nproc)" \
    && cmake --install /tmp/PoseLib/build \
    && rm -rf /tmp/PoseLib

# 3. Work folder for your own code (mount it with -v)
WORKDIR /workspace

CMD ["bash"]
