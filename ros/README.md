# COLMAP Odometry for ROS 2

`colmap_odometry` is a frame-to-frame multi-camera visual odometry built on
COLMAP geometry (pycolmap), with pluggable local features and optional IMU
priors. For every synchronized rig frame it:

1. extracts features in every camera and matches each *temporal* camera
   against its previous frame and every configured *stereo pair*;
2. filters stereo matches with the calibrated epipolar geometry and
   triangulates them;
3. verifies temporal matches with calibrated essential-matrix MAGSAC;
4. estimates the rig motion with (generalized, multi-camera) PnP from the
   previous frame's stereo points, competing with an IMU-seeded refinement;
5. refines the motion and estimates its covariance with anchored
   inverse-depth landmarks over all temporal matches (per-landmark Schur
   elimination, Huber loss, spatially clustered covariance correction,
   optional gyro-rotation and gravity-direction factors).

There is no windowed bundle adjustment; every interval is solved
independently and published as a relative motion with covariance (consumed
by Stelvio) and as an accumulated trajectory.

## Layout

| Module | Content |
| --- | --- |
| `geometry/` | Lie groups and covariance transport, camera wrapper (any COLMAP model), N-camera rig |
| `features/` | Backend interfaces, plugin registry, resize/extract/match frontend, `plugins/` |
| `vision/` | Stereo filtering and triangulation, temporal verification, feature tracks |
| `estimation/` | PnP, anchored inverse-depth refinement, inertial factors |
| `imu/` | IMU filter (preintegration, optional velocity/bias update) and prior construction |
| `odometry.py` | Per-frame orchestration (`Odometry.process`) |
| `frames.py` | Rig to published body frame conversion |
| `offline.py`, `io/`, `check.py` | Offline runner without ROS (rosbag2 or image folder), accuracy regression check |
| `ros/` | The ROS 2 node (the only module importing ROS) |

## Dependencies

Mandatory: ROS 2 (Jazzy) for the node, `numpy`, `opencv` (>= 4.5),
`pyyaml`, `pycolmap` (>= 3.12).

Feature libraries are git submodules in `colmap_odometry/thirdparty/`, pinned
to tested commits; clone with `--recurse-submodules` or run
`git submodule update --init` once:

| Submodule | Provides | Needs |
| --- | --- | --- |
| `thirdparty/LightGlue` (cvg/LightGlue) | ALIKED, SuperPoint, DISK, LightGlue | torch, torchvision, kornia |
| `thirdparty/LoMa` (davnords/LoMa) | LoMa joint model | torch, torchvision, einops, Pillow |

The vendored copies take precedence over installed packages. Weights are
downloaded on first use into `colmap_odometry/thirdparty/weights/`
(git-ignored). Without any deep-learning backend the OpenCV SIFT +
nearest-neighbour fallback is used; `rosbags` is needed for offline runs and
the regression check. Register the pycolmap rosdep rule once, then build:

```bash
sudo sh -c 'echo "yaml file://$PWD/src/colmap-odometry/rosdep/colmap_odometry-pip.yaml" \
  > /etc/ros/rosdep/sources.list.d/50-colmap-odometry-pip.list'
rosdep update && rosdep install --from-paths src --ignore-src -r -y
colcon build --paths src/colmap-odometry/colmap_odometry_msgs src/colmap-odometry --symlink-install
```

## Running

```bash
ros2 launch colmap_odometry colmap_odometry.launch.py params_file:=tumvi.yaml
```

The same parameter file drives an offline run directly from a bag (no
playback, deterministic):

```bash
ros2 run colmap_odometry colmap_odometry_offline config/tumvi.yaml /data/tum_vi_room1 -o /tmp/run --report
```

### Offline from an image folder

The offline runner also accepts a folder of images instead of a bag, so a
trajectory can be computed from a recorded image sequence and the usual
parameter file alone (no ROS, no bag conversion):

```bash
ros2 run colmap_odometry colmap_odometry_offline config/tumvi.yaml /data/my_sequence -o /tmp/run
# or, without the ROS environment, from the package directory:
python3 -m colmap_odometry.offline config/tumvi.yaml /data/my_sequence -o /tmp/run
```

The result is `/tmp/run/trajectory.tum`: one line
`timestamp tx ty tz qx qy qz qw` per frame (timestamp in seconds), the pose
of the body frame (`body_frame_id`, via `T_camera_base`) in a world frame
equal to the body pose at the first frame.  `/tmp/run/relative_motion.jsonl`
holds the per-frame relative motions with covariance, and `--report` adds
the per-frame diagnostics `frames.jsonl`.

**Folder layout.** One directory per camera, named like the entries of
`camera_names` in the parameter file (`cam0`, `cam1`, ...), optionally below
a `mav0/` directory.  Two layouts are recognized for each camera directory:

```
my_sequence/                         my_sequence/
  cam0/                                cam0/
    data.csv                             1520530308199447626.png
    data/1520530308199447626.png         1520530308249448626.png
    data/...                             ...
  cam1/                                cam1/
    data.csv                             1520530308199447626.png
    data/...                             ...
       EuRoC/ASL layout                        plain layout
```

* *EuRoC/ASL* (EuRoC MAV, TUM-VI exports, Kalibr): `data.csv` lists
  `timestamp [ns],filename` (an optional header line or `#` comments are
  skipped), and the files are read from `data/`.  Datasets in this format
  can be used as downloaded, e.g. `dataset-room1_1024_16/` of TUM-VI.
* *plain*: images named by their timestamp, directly in the camera
  directory or in `data/`.  An integer name is nanoseconds
  (`1520530308199447626.png`), a name with a decimal point is seconds
  (`1520530308.199447626.png`).

Supported image files are PNG, JPEG, BMP, TIFF, PGM and PPM.  8-bit images
are used as stored, 16-bit images are scaled to 8 bit exactly like the ROS
node converts `mono16` images, colour images are used as colour (the feature
backend converts as it needs).  Images must already have the resolution
described by the camera calibration file.

**Synchronization.** The cameras' images are grouped into rig frames by
timestamp, as for bags: a frame is formed when every camera has an image
within `sync_slop` seconds of the others (from the parameter file,
`--sync-slop` overrides it); an image without partners is skipped.  The
frame takes the timestamp of the first camera.  Hardware-triggered rigs
should use identical timestamps in all cameras.

**Parameter file.** Any node parameter file works unchanged: `camera_names`,
`camera_file`, `rig_file`, `config_file`, `T_camera_base` and `sync_slop`
are used; ROS-only entries (topics, QoS, publishing) are ignored.  Folder
input is visual only: the `imu` section is ignored even when enabled.

**Options.**

| Option | Meaning |
| --- | --- |
| `--image-dirs DIR ...` | Camera directories in `camera_names` order, when they are not named like the cameras (e.g. `--image-dirs left right`) |
| `--format auto\|bag\|folder` | Input type; `auto` (default) treats a directory with `metadata.yaml`, `*.db3` or `*.mcap` (or such a file) as a bag, and a directory with the camera directories as a folder |
| `--start-sec S`, `--max-frames N` | Skip the first `S` seconds; stop after `N` frames |
| `--sync-slop S` | Override `sync_slop` |
| `--report` | Also write `frames.jsonl` |
| `--feature-cache DIR` | Store features and matches per frame and replay them in later runs with the same feature configuration (fast parameter studies) |

A folder exported from a bag gives the same trajectory as the bag itself
(verified on TUM-VI room1).

## Configuration

A node parameter file (`config/*.yaml`, `ros__parameters`) names three
further files:

| Parameter | File |
| --- | --- |
| `config_file` | odometry configuration (below) |
| `camera_file` | per-camera COLMAP `model`, `width`, `height`, `params` |
| `rig_file` | per-camera ROS `T_rig_camera`: `rotation: [x, y, z, w]`, `translation` |

Unknown keys in any file are errors. The odometry configuration:

```yaml
stereo_pairs: [[cam0, cam1], [cam2, cam3]]  # (host, partner) pairs that overlap
temporal_cameras: null                      # default: pair hosts + unpaired cameras
features:
  resize: {width: 1024, height: 768}        # or {scale: 0.5}; each axis scaled independently
  extractor: {name: aliked, model_name: aliked-n16, max_num_keypoints: 2000}
  matcher: {name: lightglue, filter_threshold: 0.1}
  # or one joint model instead of extractor/matcher:
  # joint: {name: loma, model: loma_B128, num_keypoints: 1000, compile: true}
stereo: {epipolar_max_error_px: 2.0, reprojection_max_error_px: 3.0,
         min_triangulation_angle_deg: 0.2, min_depth: 0.2, max_depth: 20.0}
temporal: {max_error_px: 2.0, confidence: 0.999, max_iterations: 10000}
pnp: {ransac_max_error_px: 3.0, ransac_max_trials: 10000}
refinement: {pixel_std_px: 1.0, robust_delta: 3.0, max_iterations: 4,
             min_points: 6, spatial_clusters: 24, inlier_gate: 1.0}
feature_tracks: {enabled: true, min_length: 5, cameras: null}
```

Each landmark is anchored at its previous-frame keypoint in the host camera;
the anchor pixel is itself a noisy measurement (2-D correction + inverse
depth per landmark), so the anchor noise shared by all views of a landmark is
modelled exactly: the estimate is unbiased and the covariance consistent
(Monte-Carlo NEES about 1 on synthetic rigs).  Only temporal matches
consistent with the PnP motion (within `inlier_gate` x the PnP threshold)
enter the refinement.

Landmarks live in the host camera of a pair; pairs that do not overlap are
simply not listed. Temporal cameras without a pair contribute monocular
(bearing) constraints. With no `features` section the best available backend
is chosen (ALIKED + LightGlue, else OpenCV SIFT + NN).

### Adding a feature backend

Drop a module into `colmap_odometry/features/plugins/` (or expose one via the
`colmap_odometry.features` entry point) and register either an extractor
(paired with any compatible matcher) or one joint model. Import the
framework inside the class so the module can always be discovered:

```python
from colmap_odometry.features import Extractor, Features, register

@register("extractor", "myfeat", requires=("torch", "myfeat"), default_matcher="nn")
class MyFeatures(Extractor):
    descriptor = "myfeat"

    def __init__(self, max_keypoints: int = 2000):
        import myfeat
        self.model = myfeat.load(max_keypoints)

    def extract(self, images):  # uint8 gray/BGR, already resized
        return [Features(keypoints, descriptors, (image.shape[1], image.shape[0]))
                for image, (keypoints, descriptors) in zip(images, self.model(images))]
```

If the backend needs a third-party library, add it as a pinned submodule
under `colmap_odometry/thirdparty/`, list its import path in
`features/thirdparty.py` (`LIBRARIES`), call `use("<module>")` at the top of
the plugin and construct models inside `weights_directory()`.

All other keys of the config entry are passed to the constructor. Matchers
implement `accepts(descriptor)`, optionally `bind(descriptor)`, and
`match(pairs)`; joint models implement both sides (see `plugins/loma.py`).

### Frames

`T_camera_base` (`p_camera = T_camera_base p_base`, row-major, for the first
camera) defines the published body frame. The world frame is the body at the
first frame (or the gravity-aligned frame with `imu.align_world_with_gravity`).

### IMU

With `imu.enabled`, an error-state filter propagates the IMU between frames
and supplies a PnP seed, a gyro rotation factor (floored at
`rotation_factor_std_floor`) and a gravity-direction factor to each
interval's visual solve; the published pose is that visual-inertial
solution.  With `imu.visual_update` (default) the filter is then updated with
the *visual-only* estimate of the interval (recovered from the same
linearization, so the inertial priors are not counted twice), which keeps
velocity, biases and the published covariance meaningful.  The update floors
(`visual_*_std_floor`) should match the actual per-frame visual accuracy.

On TUM-VI room1 this lowers translation drift by 15-20 % and improves 1 s
motion accuracy in every segment, while long-horizon yaw drift grows by about
0.3 deg: frame-to-frame updates cannot separate the gyro bias from the visual
odometry's own systematic rotation drift.  Publishing the filter pose or a
6-DoF (rotation + translation) inertial factor were evaluated and are worse.

`imu.gyro_bias_window_sec` (off by default) instead takes the gyro bias of
the rotation prior from a robust fit of gyro against visual-only rotations
over the last seconds; with `gyro_bias_window_axes: vertical` only the
component about gravity is taken from it and the tilt components (observable
through gravity) from the filter.  The yaw of the result then follows the
visual odometry's yaw over the window length: on TUM-VI room1 this helps
with LoMa and SuperPoint and hurts with ORB, mixed for the other frontends.

## Interfaces

Topics (relative to the node, normally `/colmap_odometry_node`):

- `~/odom`, `~/pose`, `~/path`: body pose in `world_frame_id`;
- `~/relative_motion` (`colmap_odometry_msgs/RelativePoseWithCovarianceStamped`):
  motion between consecutive frames with right-local covariance in ROS
  `[x, y, z, rot_x, rot_y, rot_z]` order and both endpoint stamps;
- `~/feature_tracks` (`colmap_odometry_msgs/FeatureTrackArray`);
- `~/status` (JSON per frame); optional `~/feature_matches`, `~/sparse_points`,
  and `~/input/<camera>/image_raw` for compressed inputs.

`save_trajectory_tum` writes the `~/odom` poses as
`timestamp tx ty tz qx qy qz qw`; `frame_log_path` writes a per-frame JSON
report (timings, matches, PnP and refinement statistics).

## Accuracy regression check

Compare a candidate run with a reference run (e.g. `main` vs `dev`) on the
same bag and ground truth:

```bash
ros2 bag record -o ref_motion /colmap_odometry_node/relative_motion   # during each run
ros2 run colmap_odometry colmap_odometry_check --ground-truth gt.tum \
  --reference ref/trajectory.tum --reference-motion ref_motion \
  --candidate dev/trajectory.tum --candidate-motion dev_motion
```

It reports APE (SE(3)-aligned), RPE, NEES of relative motions composed over
1 s windows and the number of relative motions, and fails (exit 1) when the
candidate is more than 1 % worse in APE/RPE, its NEES deviates by more than
5 %, or it loses more than 1 % of motions. When the covariance estimator
changed on purpose, `--nees-criterion consistency` instead requires the
candidate NEES to be no further from 1 than the reference. `tools/legacy_parity.py` runs the
legacy implementation offline for a per-frame comparison.

## Tests

```bash
python3 -m pytest test
```

`test/legacy_reference/` holds frozen copies of the legacy solver and IMU
filter used as parity oracles.

## License

BSD 3-Clause; please cite COLMAP when using this work in research.
