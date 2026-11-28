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

Local features: `--detector orb` (default) or `--detector sift`.
- **ORB:** binary descriptors, Hamming matching.
- **SIFT:** OpenCV SIFT, float descriptors, L2 matching; slower.

`--features N` sets the number of features per image for either detector.

Camera models in the calibration file (COLMAP names):
- `PINHOLE` with `params: [fx, fy, cx, cy]`;
- `OPENCV` with `params: [fx, fy, cx, cy, k1, k2, p1, p2]`. The radial-tangential distortion is used in triangulation, PnP and the essential matrix.

Inside `images` folder are expected to be found images in folers `cam0` and `cam1`. Also monocular is supported.
