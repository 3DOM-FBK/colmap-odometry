# xhost +local:docker
# docker build -t cpp_poselib .

cd /home/threedom/Desktop/github_lcmrl/pycolmap-odometry/c++
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)":/workspace \
  cpp_poselib


# Inside the docker container, run the following commands to build and run the project:
# cmake -S . -B build
# cmake --build build -j
./build/orb_matching assets/02_set_with_trajectory/images \
    calibration/camera_calibration.yaml \
    calibration/camera_rig.yaml --no-gui --scale 0.5 --single-core

