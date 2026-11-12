# COLMAP Odometry for ROS 2 (Jazzy)
#
# Build (from inside the ros/ folder):
#   docker build -t colmap_odometry .
#
# Run (add --gpus all if you have the NVIDIA Container Toolkit):
#   docker run -it --rm --net=host colmap_odometry
#
# Inside the container, start the node with:
#   ros2 launch colmap_odometry colmap_odometry.launch.py
#
# or with your own parameters:
#   ros2 launch colmap_odometry colmap_odometry.launch.py params_file:=/absolute/path/to/my_stereo.yaml

FROM ros:jazzy

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# 1. System / ROS dependencies (from package.xml)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv \
        python3-pip \
        python3-colcon-common-extensions \
        python3-numpy \
        python3-opencv \
        python3-yaml \
        python3-tqdm \
        python3-psutil \
        ros-jazzy-cv-bridge \
        ros-jazzy-message-filters \
        ros-jazzy-sensor-msgs-py \
        ros-jazzy-tf2-ros \
        ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python virtual environment
#    --system-site-packages keeps ROS Python packages (rclpy, cv_bridge, ...) visible.
#    Putting the venv first in PATH makes it the default "python3".
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Python dependencies (versions from README / rosdep/colmap_odometry-pip.yaml)
RUN pip install --no-cache-dir \
        'pycolmap==3.12.6' \
        'pyceres==2.4' \
        'kornia>=0.7.2' \
        'enlighten>=1.14.1' \
        pyquaternion \
        torch \
        torchvision \
        transformers \
        einops \
        tyro

# 4. Copy the source code into a ROS 2 workspace
WORKDIR /ros2_ws
COPY . src/colmap-odometry

# 5. Build the messages package and the odometry package
#    "python3 -m colcon" runs colcon with the venv Python, so the installed
#    node uses the venv (where pycolmap, torch, ... live) instead of /usr/bin/python3.
RUN . /opt/ros/jazzy/setup.sh && \
    python3 -m colcon build --paths \
        src/colmap-odometry/colmap_odometry_msgs \
        src/colmap-odometry

# 6. Source ROS and the workspace automatically in every shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
