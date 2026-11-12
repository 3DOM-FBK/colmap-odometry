## Build image
# docker build -t colmap_odometry .

## Run container
# docker run -it --rm --net=host --gpus all colmap_odometry

## Check results
# ros2 pkg list | grep colmap_odometry          # package built and visible
# python3 -c "import torch; print(torch.cuda.is_available())"   # True = GPU works
# ros2 launch colmap_odometry colmap_odometry.launch.py
