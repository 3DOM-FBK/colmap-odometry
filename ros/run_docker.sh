## Build image
#docker build -t colmap_odometry .


## Check results
# ros2 pkg list | grep colmap_odometry          # package built and visible
# python3 -c "import torch; print(torch.cuda.is_available())"   # True = GPU works
# ros2 launch colmap_odometry colmap_odometry.launch.py


## Run container

#docker run -it --rm --net=host --gpus all \
#  -v /home/threedom/Desktop/euroc/MH_01_easy/mav0/aaa:/data/my_sequence \
#  colmap_odometry

docker run -it --rm --net=host --gpus all \
  -v /home/threedom/Desktop/VAIPOSA/CARLA_DATASET/02_set_with_trajectory/images/:/data/my_sequence \
  colmap_odometry

# Inside docker:
#ros2 run colmap_odometry colmap_odometry_offline src/colmap-odometry/config/temp/ideal.yaml /data/my_sequence -o /tmp/run
