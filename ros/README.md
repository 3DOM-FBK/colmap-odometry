# COLMAP Odometry for ROS 2

`colmap_odometry` is a ROS 2 stereo visual-odometry wrapper around the
COLMAP/PyCOLMAP frontend. It detects and matches stereo features, triangulates
metric points, and estimates inter-frame motion with a minimal
triangulation-and-PnP solver. It can optionally fuse image-rate visual poses
with IMU propagation through an iterated error-state Kalman filter.

The node uses ROS frame conventions at its interface. Camera rig YAML files
are ROS TF-style poses; COLMAP camera-from-world transforms are derived
internally.

## Features

- synchronized raw or `sensor_msgs/CompressedImage` stereo inputs;
- ALIKED, SuperPoint, or ORB features with LightGlue matching;
- metric stereo PnP minimal-solver route, without windowed bundle adjustment;
- optional iEKF visual-inertial output;
- odometry, path, relative pose/covariance, feature-track, diagnostic-image,
  sparse RGB-cloud, and TUM-trajectory outputs;
- per-stage terminal performance diagnostics.

Image synchronization and visual processing are decoupled. The synchronized
callback only queues each pair; a single worker drains the queue in order, so
frames are retained when COLMAP is slower than the camera stream. The
`sync_queue_size` parameter controls only unmatched messages. Set
`processing_queue_size` to a positive value to receive backlog warnings at that
depth; `0` (the default) means no warning threshold and never drops completed
pairs.

For rosbag2 playback, use `image_qos_reliability: reliable` when the bag offers
reliable image publishers. Keep `best_effort` for live cameras that advertise
sensor-data QoS.

## Requirements

- ROS 2 Jazzy (or a compatible ROS 2 distribution);
- COLMAP-compatible `pycolmap` and `pyceres` Python wheels;
- the bundled `colmap_odometry_msgs` ROS interface package, built alongside
  this package and defining the `FeatureTrackArray` message.

System ROS dependencies are in `package.xml`. Python dependencies are listed
in [rosdep/colmap_odometry-pip.yaml](rosdep/colmap_odometry-pip.yaml).

## Build

Create the virtual environment with `uv`. `--system-site-packages` makes ROS
Python packages such as `rclpy` and `cv_bridge` visible inside the environment.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash

uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install 'pycolmap==3.12.6' 'pyceres==2.4' \
  'kornia>=0.7.2' pyquaternion torch torchvision transformers 'enlighten>=1.14.1'
```

PyCOLMAP 3.12 is required for the corrected pose-inverse covariance
propagation API. The node reports the loaded version and module at startup;
older installations are accepted only for legacy runs and emit a warning.

Place this repository under `src/`. Register the local rosdep rules once,
replacing the path with the clone location:

```bash
sudo sh -c 'echo "yaml file:///absolute/path/to/colmap-odometry/rosdep/colmap_odometry-pip.yaml" \
  > /etc/ros/rosdep/sources.list.d/50-colmap-odometry-pip.list'
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

Build and source the overlay:

```bash
colcon build --paths \
  src/colmap-odometry/colmap_odometry_msgs \
  src/colmap-odometry \
  --symlink-install
source install/setup.bash
```

## Quick start

The installed default profile is an example for a calibrated stereo pair. It
uses `/stereo/left/image_raw` and `/stereo/right/image_raw` and writes runtime
data under `/tmp/colmap_odometry`.

```bash
ros2 launch colmap_odometry colmap_odometry.launch.py
```

Pass a configuration filename from the installed `config/` directory, or an
absolute path:

```bash
ros2 launch colmap_odometry colmap_odometry.launch.py \
  params_file:=/absolute/path/to/my_stereo.yaml
```

Open the generic RViz configuration separately:

```bash
rviz2 -d "$(ros2 pkg prefix colmap_odometry)/share/colmap_odometry/rviz/colmap_odometry.rviz"
```

## Configuration

The default profile uses four files in `config/`:

| File | Purpose |
| --- | --- |
| `default.yaml` | ROS topics, frames, outputs, optional IMU, and diagnostics |
| `config_cameras_intrinsics.yaml` | per-camera COLMAP model, image size, and intrinsics |
| `config_camera_rig.yaml` | calibrated ROS `T_rig_camera` stereo transforms |
| `config_odometry.yaml` | features, matching, triangulation, and PnP settings |

Copy all four before configuring a new sensor. The values in the two
calibration examples are only a valid stereo example; they must be replaced by
the actual camera calibration. Keep `resize_width`/`resize_height` consistent
with the configured image dimensions.

For compressed input, set `image_topics` to the two
`sensor_msgs/CompressedImage` topics. The node decodes them for processing and
republishes the synchronized raw images on `~/input/cam0/image_raw` and
`~/input/cam1/image_raw` for visualization.

### Frames and transforms

Camera rig YAML stores the ROS transform `T_rig_camera`:

```text
p_rig = R_rig_camera * p_camera + t_rig_camera
```

Quaternion order is `[x, y, z, w]`. Do not invert or negate the baseline to
imitate COLMAP. The wrapper derives COLMAP's camera-from-rig projection
extrinsic internally:

```text
R_camera_rig = R_rig_camera^T
t_camera_rig = -R_rig_camera^T * t_rig_camera
```

`T_camera_base` is also a ROS-direction transform:

```text
p_camera = T_camera_base * p_base
```

The published trajectory is `world_frame_id -> body_frame_id`. With
`publish_tf: true`, that transform and static body-to-camera transforms are
published as TF.

### Optional IMU fusion

Enable `imu.enabled` only after setting `imu.T_base_imu` and the IMU noise
parameters. The `*_variance` parameters are continuous covariance diagonals,
not standard deviations. `visual_translation_std_floor` and
`visual_rotation_std_floor` are nonzero measurement-noise floors inside the
iEKF; they should not be set to zero. Optional
`visual_model_translation_std` and `visual_model_rotation_std` terms represent
independent model uncertainty and are added in variance to valid solver
covariances. The stereo-temporal covariance uses
`covariance.pixel_std_px` and `covariance.min_points`. The default
`covariance.spatial_clusters` setting (24) estimates a robust spatial-cluster
sandwich covariance from the final Schur-complement residuals, preserving
correlated feature/calibration errors instead of treating every learned
feature as independent.

`imu.T_base_imu` is interpreted as `p_base = T_base_imu * p_imu`. Its
translation is used for rigid-body lever-arm compensation: the measured
specific force is converted from the IMU origin to the propagated base origin
using both tangential and centripetal acceleration terms. The same prepared
measurement is used for iEKF propagation, initialization, and gravity factors.

## Interfaces

Topics are relative to the node namespace (normally `/colmap_odometry_node`):

- `~/odom` (`nav_msgs/Odometry`), `~/pose` (`geometry_msgs/PoseStamped`), and
  `~/path` (`nav_msgs/Path`);
- `~/relative_pose` and `~/relative_pose_covariance`. The latter is retained
  for compatibility and carries only the end timestamp;
- `~/relative_motion` (`colmap_odometry_msgs/RelativePoseWithCovarianceStamped`)
  carries the same relative pose and covariance together with explicit
  `from_stamp` and `to_stamp` endpoints. Its covariance is serialized in ROS
  `[x, y, z, rot_x, rot_y, rot_z]` order and represents the local SE(3)
  relative-motion error;
- `~/feature_tracks` (`colmap_odometry_msgs/FeatureTrackArray`), enabled by
  `create_feature_tracks` and `publish_feature_tracks`;
- `~/feature_matches`, `~/input/cam*/image_raw`, and optionally
  `~/sparse_points` for debugging;
- `~/status` (`std_msgs/String`).

Set `save_trajectory_tum: true` and choose `trajectory_tum_path` to write:

```text
timestamp tx ty tz qx qy qz qw
```

The TUM trajectory uses exactly the same base-frame pose as `~/odom`.

## Performance diagnostics

Diagnostics report decoding, visual odometry, ROS publication, feature
visualization, and core feature/matching/solver times. Configure them with:

```yaml
performance_diagnostics: true
performance_log_every_n_frames: 10
performance_slow_frame_sec: 1.0
performance_stall_warn_sec: 3.0
performance_stall_repeat_sec: 5.0
performance_input_gap_warn_sec: 3.0
```

## License and references

This repository is distributed under the [BSD 3-Clause License](LICENSE). Please cite
the original COLMAP and COLMAP-SLAM work when using it in research.
