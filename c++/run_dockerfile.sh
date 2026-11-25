xhost +local:docker
# docker build -t cpp_poselib .

cd /home/threedom/Desktop/github_lcmrl/pycolmap-odometry/c++
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)":/workspace \
  cpp_poselib
