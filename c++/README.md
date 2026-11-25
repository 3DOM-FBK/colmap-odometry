# Colmap-odometry: C++ version

## In a terminal:

```
xhost +local:docker
docker build -t cpp_poselib .

cd path/to/pycolmap-odometry/c++
./run_dockerfile.sh
```

## Inside the docker container, run the following commands to build and run the project:
```
cmake -S . -B build
cmake --build build -j
./build/orb_matching assets/02_set_with_trajectory/images \
    calibration/camera_calibration.yaml \
    calibration/camera_rig.yaml --scale 0.5 --pnp poselib --features 2000
```

Additional options: `--no-gui --single-core`

Inside `images` folder are expected to be found images in folers `cam0` and `cam1`. Also monocular is supported.
