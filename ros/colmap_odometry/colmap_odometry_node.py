from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import yaml
from pyquaternion import Quaternion

import message_filters
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from colmap_odometry_msgs.msg import (
    FeatureObservation,
    FeatureTrack,
    FeatureTrackArray,
    RelativePoseWithCovarianceStamped,
)
from std_msgs.msg import Header, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .imu_iekf import ImuIekfConfig, ImuIteratedErrorStateFilter
from .rig_transforms import colmap_camera_from_rig_matrix


_PKG_DIR = Path(__file__).resolve().parent
try:
    _DEFAULT_SHARE_DIR = Path(get_package_share_directory("colmap_odometry"))
except PackageNotFoundError:
    _DEFAULT_SHARE_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG_DIR = _DEFAULT_SHARE_DIR / "config"
_DEFAULT_CALIBRATION_DIR = _DEFAULT_SHARE_DIR / "calibration"


def _pycolmap_version_tuple() -> tuple[int, ...]:
    """Return a comparable PyCOLMAP version without requiring packaging."""
    raw = str(getattr(pycolmap, "__version__", "0.0.0"))
    parts = []
    for component in raw.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts or [0])

# T_camera_base for a colocated REP-103 base frame:
# camera optical x-right/y-down/z-forward, base x-forward/y-left/z-up.
_DEFAULT_T_CAMERA_BASE = [
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def _resolve_path(value: str, base_dir: Path, fallback_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    resolved = (base_dir / path).resolve()
    if resolved.exists() or fallback_dir is None:
        return resolved
    return (fallback_dir / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML file did not contain a mapping: {path}")
    return loaded


def _sanitize_frame_name(stamp) -> str:
    return f"{stamp.sec}_{stamp.nanosec:09d}.png"


def _quaternion_to_msg(quaternion: Quaternion):
    # pyquaternion stores [w, x, y, z].
    return (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )


def _flat_transform(value: list[float], name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.size != 16:
        raise ValueError(f"{name} must contain 16 row-major values")
    transform = transform.reshape(4, 4)
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{name} must be a rigid 4x4 transform with last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform


def _pose_to_matrix(translation: np.ndarray, quaternion: Quaternion) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion.rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _matrix_to_pose(transform: np.ndarray) -> tuple[np.ndarray, Quaternion]:
    u, _, vh = np.linalg.svd(transform[:3, :3])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return transform[:3, 3].copy(), Quaternion(matrix=rotation)


def _adjoint_rotation_translation(transform: np.ndarray) -> np.ndarray:
    """SE(3) adjoint for COLMAP tangent ordering [rotation, translation]."""
    rotation = transform[:3, :3]
    tx, ty, tz = transform[:3, 3]
    skew_translation = np.array(
        [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]],
        dtype=np.float64,
    )
    adjoint = np.zeros((6, 6), dtype=np.float64)
    adjoint[:3, :3] = rotation
    adjoint[3:, :3] = skew_translation @ rotation
    adjoint[3:, 3:] = rotation
    return adjoint


def _rig_transform_to_matrix(config: dict[str, Any], camera_name: str) -> np.ndarray:
    """Return COLMAP T_camera_rig converted from ROS rig YAML."""
    return colmap_camera_from_rig_matrix(config, camera_name)


def _compatible_ros_image_encoding(image: np.ndarray, requested: str) -> str:
    """Choose an encoding compatible with the decoded OpenCV array.

    Some compressed grayscale JPEG streams remain ``8UC1`` even when
    ``compressed_imgmsg_to_cv2`` is asked for ``bgr8``.  Publishing that array
    as bgr8 raises in cv_bridge and aborts the processing callback; infer the
    actual mono/color encoding for republishing instead.
    """
    array = np.asarray(image)
    request = str(requested).lower()
    channels = int(array.shape[2]) if array.ndim == 3 else 1
    depth = {
        np.dtype(np.uint8): "8",
        np.dtype(np.uint16): "16",
        np.dtype(np.float32): "32F",
        np.dtype(np.float64): "64F",
    }.get(array.dtype)
    if depth is None:
        raise ValueError(f"unsupported image dtype for ROS publication: {array.dtype}")
    if channels == 1:
        return f"mono{depth}" if depth in {"8", "16"} else f"32FC1"
    if channels == 3 and request in {"rgb8", "rgb16"}:
        return f"rgb{depth}"
    if channels == 3:
        return f"bgr{depth}"
    if channels == 4 and request in {"rgba8", "rgba16"}:
        return f"rgba{depth}"
    if channels == 4:
        return f"bgra{depth}"
    raise ValueError(f"unsupported image channel count for ROS publication: {channels}")


class ColmapOdometryNode(Node):
    """Run colmap-odometry from ROS2 image streams."""

    def __init__(self) -> None:
        super().__init__("colmap_odometry_node")

        self.declare_parameter("config_file", "config_euroc.yaml")
        self.declare_parameter("camera_file", "calibration_euroc.yaml")
        self.declare_parameter("rig_file", "camera_rig_euroc.yaml")
        self.declare_parameter("work_dir", "/tmp/colmap_odometry")
        self.declare_parameter("image_topics", ["/cam0/image_raw", "/cam1/image_raw"])
        self.declare_parameter("camera_names", ["cam0", "cam1"])
        self.declare_parameter("world_frame_id", "map")
        # body_frame_id is the preferred single name for the estimated body.
        # An empty value preserves the legacy base_frame_id/child_frame_id pair.
        self.declare_parameter("body_frame_id", "")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("camera_frame_ids", ["cam0", "cam1"])
        self.declare_parameter("T_camera_base", _DEFAULT_T_CAMERA_BASE)
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("publish_camera_tf", False)
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("image_qos_reliability", "best_effort")
        # The synchronizer queue only holds messages waiting for a matching
        # camera. Once a pair is formed it is moved to an unbounded processing
        # queue, so a slow visual backend cannot discard completed pairs.
        self.declare_parameter("processing_queue_size", 0)
        self.declare_parameter("image_encoding", "bgr8")
        self.declare_parameter("save_input_images", True)
        self.declare_parameter("publish_every_frame", True)
        self.declare_parameter("reinitialize_after_frames", -1)
        self.declare_parameter("publish_feature_matches", False)
        self.declare_parameter("publish_sparse_point_cloud", False)
        self.declare_parameter("create_feature_tracks", True)
        self.declare_parameter("publish_feature_tracks", True)
        self.declare_parameter("feature_track_publish_min_length", 5)
        self.declare_parameter("feature_match_max_features", 1500)
        self.declare_parameter("feature_match_max_matches", 500)
        # Path is an ever-growing snapshot. Republishing it for every image
        # makes serialization/DDS traffic quadratic in an offline sequence and
        # can leave RViz far behind an otherwise current TF stream.
        self.declare_parameter("path_publish_period_sec", 0.5)
        self.declare_parameter("save_trajectory_tum", False)
        self.declare_parameter("trajectory_tum_path", "")
        self.declare_parameter("trajectory_tum_overwrite", True)
        self.declare_parameter("performance_diagnostics", True)
        self.declare_parameter("performance_log_every_n_frames", 10)
        self.declare_parameter("performance_slow_frame_sec", 1.0)
        self.declare_parameter("performance_stall_warn_sec", 3.0)
        self.declare_parameter("performance_stall_repeat_sec", 5.0)
        self.declare_parameter("performance_input_gap_warn_sec", 3.0)
        # Detailed covariance diagnostics are written independently of the
        # compact ROS status stream so the full matrices and provenance can be
        # inspected after a run.
        self.declare_parameter(
            "diagnostics.covariance_log_path",
            "covariance_diagnostics.jsonl",
        )
        self.declare_parameter("diagnostics.covariance_log_enabled", True)
        self.declare_parameter("imu.enabled", False)
        self.declare_parameter("imu.topic", "/imu")
        self.declare_parameter("imu.time_offset_sec", 0.0)
        self.declare_parameter("imu.T_base_imu", [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
        self.declare_parameter("imu.queue_depth", 2000)
        self.declare_parameter("imu.sample_buffer_size", 0)
        self.declare_parameter("imu.wait_timeout_sec", 0.05)
        self.declare_parameter("imu.initialization_duration_sec", 1.0)
        self.declare_parameter("imu.initialization_max_wait_sec", 2.0)
        self.declare_parameter("imu.initialization_max_gyro_norm", 0.03)
        self.declare_parameter("imu.initialization_max_gyro_std", 0.015)
        self.declare_parameter("imu.initialization_max_accel_std", 0.15)
        self.declare_parameter("imu.gravity_magnitude", 9.80665)
        self.declare_parameter("imu.align_world_with_gravity", True)
        self.declare_parameter("imu.gyro_noise_variance", 5.0e-6)
        self.declare_parameter("imu.accel_noise_variance", 2.5e-3)
        self.declare_parameter("imu.gyro_bias_random_walk_variance", 1.0e-8)
        self.declare_parameter("imu.accel_bias_random_walk_variance", 1.0e-6)
        self.declare_parameter("imu.gravity_random_walk_variance", 0.0)
        self.declare_parameter("imu.initial_position_std", 0.01)
        self.declare_parameter("imu.initial_velocity_std", 0.25)
        self.declare_parameter("imu.initial_orientation_std", 0.03)
        self.declare_parameter("imu.initial_gyro_bias_std", 0.01)
        self.declare_parameter("imu.initial_accel_bias_std", 0.1)
        self.declare_parameter("imu.initial_gravity_std", 0.5)
        # These are iEKF measurement-noise floors, not published-covariance
        # floors. A nonzero value prevents a weak PnP covariance estimate from
        # becoming an effectively exact correction.
        self.declare_parameter("imu.visual_translation_std_floor", 0.03)
        self.declare_parameter("imu.visual_rotation_std_floor", 0.01)
        self.declare_parameter("imu.visual_model_translation_std", 0.0)
        self.declare_parameter("imu.visual_model_rotation_std", 0.0)
        self.declare_parameter("imu.visual_nis_soft_threshold", 16.812)
        self.declare_parameter("imu.maximum_iterations", 4)
        self.declare_parameter("imu.convergence_epsilon", 1.0e-5)
        self.declare_parameter("imu.maximum_propagation_step_sec", 0.01)
        self.declare_parameter("imu.maximum_gap_sec", 0.1)
        self.declare_parameter("imu.prior_maximum_reprojection_error_px", 4.0)
        self.declare_parameter("imu.prior_minimum_inliers", 12)
        # These parameters define residuals in the minimal visual solve. IMU
        # no longer owns a separately corrected output trajectory.
        self.declare_parameter("imu.rotation_factor_std_floor", 0.003)
        self.declare_parameter("imu.gravity_factor_enabled", True)
        self.declare_parameter("imu.gravity_factor_std", 0.05)
        self.declare_parameter("imu.gravity_factor_window_sec", 0.05)
        self.declare_parameter("imu.gravity_factor_minimum_samples", 3)
        self.declare_parameter(
            "imu.gravity_factor_acceleration_tolerance_mps2", 1.5)
        self.declare_parameter("imu.gravity_factor_max_gyro_norm", 1.0)

        config_file = _resolve_path(
            self.get_parameter("config_file").value, _DEFAULT_CONFIG_DIR)
        camera_file = _resolve_path(
            self.get_parameter("camera_file").value,
            _DEFAULT_CONFIG_DIR,
            _DEFAULT_CALIBRATION_DIR,
        )
        rig_file = _resolve_path(
            self.get_parameter("rig_file").value,
            _DEFAULT_CONFIG_DIR,
            _DEFAULT_CALIBRATION_DIR,
        )
        self._work_dir = _resolve_path(
            self.get_parameter("work_dir").value, Path.cwd())
        self._image_topics = list(self.get_parameter("image_topics").value)
        self._camera_names = list(self.get_parameter("camera_names").value)
        self._world_frame_id = str(self.get_parameter("world_frame_id").value)
        body_frame_id = str(self.get_parameter("body_frame_id").value).strip()
        self._base_frame_id = body_frame_id or str(
            self.get_parameter("base_frame_id").value)
        self._child_frame_id = body_frame_id or str(
            self.get_parameter("child_frame_id").value)
        self._camera_frame_ids = list(self.get_parameter("camera_frame_ids").value)
        self._T_camera_base = _flat_transform(
            list(self.get_parameter("T_camera_base").value),
            "T_camera_base",
        )
        self._T_base_camera = np.linalg.inv(self._T_camera_base)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._publish_camera_tf = bool(
            self.get_parameter("publish_camera_tf").value)
        self._image_encoding = str(self.get_parameter("image_encoding").value)
        self._save_input_images = bool(self.get_parameter("save_input_images").value)
        self._publish_every_frame = bool(self.get_parameter("publish_every_frame").value)
        self._reinitialize_after_frames = int(
            self.get_parameter("reinitialize_after_frames").value)
        self._publish_feature_matches = bool(
            self.get_parameter("publish_feature_matches").value)
        self._publish_sparse_point_cloud = bool(
            self.get_parameter("publish_sparse_point_cloud").value)
        self._create_feature_tracks = bool(
            self.get_parameter("create_feature_tracks").value)
        self._publish_feature_tracks = bool(
            self.get_parameter("publish_feature_tracks").value)
        self._feature_track_publish_min_length = max(
            2,
            int(self.get_parameter("feature_track_publish_min_length").value),
        )
        self._feature_match_max_features = int(
            self.get_parameter("feature_match_max_features").value)
        self._feature_match_max_matches = int(
            self.get_parameter("feature_match_max_matches").value)
        self._save_trajectory_tum = bool(
            self.get_parameter("save_trajectory_tum").value)
        self._performance_diagnostics = bool(
            self.get_parameter("performance_diagnostics").value)
        self._performance_log_every_n_frames = max(
            1, int(self.get_parameter("performance_log_every_n_frames").value))
        self._performance_slow_frame_sec = max(
            0.0, float(self.get_parameter("performance_slow_frame_sec").value))
        self._performance_stall_warn_sec = max(
            0.1, float(self.get_parameter("performance_stall_warn_sec").value))
        self._performance_stall_repeat_sec = max(
            0.1, float(self.get_parameter("performance_stall_repeat_sec").value))
        self._performance_input_gap_warn_sec = max(
            0.0, float(self.get_parameter("performance_input_gap_warn_sec").value))
        self._imu_enabled = bool(self.get_parameter("imu.enabled").value)
        self._imu_topic = str(self.get_parameter("imu.topic").value)
        self._imu_time_offset_ns = int(round(
            float(self.get_parameter("imu.time_offset_sec").value) * 1.0e9))
        self._imu_queue_depth = max(
            10, int(self.get_parameter("imu.queue_depth").value))
        self._imu_sample_buffer_size = max(
            0, int(self.get_parameter("imu.sample_buffer_size").value))
        self._imu_wait_timeout_sec = max(
            0.0, float(self.get_parameter("imu.wait_timeout_sec").value))
        self._imu_prior_maximum_error_px = max(
            0.1,
            float(self.get_parameter(
                "imu.prior_maximum_reprojection_error_px").value),
        )
        self._imu_prior_minimum_inliers = max(
            6, int(self.get_parameter("imu.prior_minimum_inliers").value))
        self._imu_rotation_factor_std_floor = max(
            1.0e-6,
            float(self.get_parameter("imu.rotation_factor_std_floor").value),
        )
        self._imu_gravity_factor_enabled = bool(
            self.get_parameter("imu.gravity_factor_enabled").value)
        self._imu_gravity_factor_std = max(
            1.0e-5,
            float(self.get_parameter("imu.gravity_factor_std").value),
        )
        self._imu_gravity_factor_window_ns = int(round(
            max(0.0, float(self.get_parameter(
                "imu.gravity_factor_window_sec").value)) * 1.0e9))
        self._imu_gravity_factor_minimum_samples = max(
            1, int(self.get_parameter(
                "imu.gravity_factor_minimum_samples").value))
        self._imu_gravity_factor_acceleration_tolerance = max(
            0.0, float(self.get_parameter(
                "imu.gravity_factor_acceleration_tolerance_mps2").value))
        self._imu_gravity_factor_max_gyro_norm = max(
            0.0, float(self.get_parameter(
                "imu.gravity_factor_max_gyro_norm").value))
        self._T_base_imu = _flat_transform(
            list(self.get_parameter("imu.T_base_imu").value),
            "imu.T_base_imu",
        )
        imu_filter_config = ImuIekfConfig(
            initialization_duration_sec=max(
                0.0,
                float(self.get_parameter(
                    "imu.initialization_duration_sec").value),
            ),
            initialization_max_wait_sec=max(
                0.0,
                float(self.get_parameter(
                    "imu.initialization_max_wait_sec").value),
            ),
            initialization_max_gyro_norm=max(
                0.0,
                float(self.get_parameter(
                    "imu.initialization_max_gyro_norm").value),
            ),
            initialization_max_gyro_std=max(
                0.0,
                float(self.get_parameter(
                    "imu.initialization_max_gyro_std").value),
            ),
            initialization_max_accel_std=max(
                0.0,
                float(self.get_parameter(
                    "imu.initialization_max_accel_std").value),
            ),
            gravity_magnitude=max(
                1.0, float(self.get_parameter("imu.gravity_magnitude").value)),
            align_world_with_gravity=bool(
                self.get_parameter("imu.align_world_with_gravity").value),
            gyro_noise_variance=max(
                0.0, float(self.get_parameter("imu.gyro_noise_variance").value)),
            accel_noise_variance=max(
                0.0, float(self.get_parameter("imu.accel_noise_variance").value)),
            gyro_bias_random_walk_variance=max(
                0.0,
                float(self.get_parameter(
                    "imu.gyro_bias_random_walk_variance").value),
            ),
            accel_bias_random_walk_variance=max(
                0.0,
                float(self.get_parameter(
                    "imu.accel_bias_random_walk_variance").value),
            ),
            gravity_random_walk_variance=max(
                0.0,
                float(self.get_parameter(
                    "imu.gravity_random_walk_variance").value),
            ),
            initial_position_std=max(
                1.0e-6,
                float(self.get_parameter("imu.initial_position_std").value),
            ),
            initial_velocity_std=max(
                1.0e-6,
                float(self.get_parameter("imu.initial_velocity_std").value),
            ),
            initial_orientation_std=max(
                1.0e-6,
                float(self.get_parameter("imu.initial_orientation_std").value),
            ),
            initial_gyro_bias_std=max(
                1.0e-8,
                float(self.get_parameter("imu.initial_gyro_bias_std").value),
            ),
            initial_accel_bias_std=max(
                1.0e-8,
                float(self.get_parameter("imu.initial_accel_bias_std").value),
            ),
            initial_gravity_std=max(
                1.0e-6,
                float(self.get_parameter("imu.initial_gravity_std").value),
            ),
            visual_translation_std_floor=max(
                1.0e-5,
                float(self.get_parameter(
                    "imu.visual_translation_std_floor").value),
            ),
            visual_rotation_std_floor=max(
                1.0e-5,
                float(self.get_parameter("imu.visual_rotation_std_floor").value),
            ),
            visual_model_translation_std=max(
                0.0,
                float(self.get_parameter(
                    "imu.visual_model_translation_std").value),
            ),
            visual_model_rotation_std=max(
                0.0,
                float(self.get_parameter(
                    "imu.visual_model_rotation_std").value),
            ),
            visual_nis_soft_threshold=max(
                1.0,
                float(self.get_parameter(
                    "imu.visual_nis_soft_threshold").value),
            ),
            maximum_iterations=max(
                1, int(self.get_parameter("imu.maximum_iterations").value)),
            convergence_epsilon=max(
                1.0e-9,
                float(self.get_parameter("imu.convergence_epsilon").value),
            ),
            maximum_propagation_step_sec=max(
                1.0e-4,
                float(self.get_parameter(
                    "imu.maximum_propagation_step_sec").value),
            ),
            maximum_imu_gap_sec=max(
                1.0e-3,
                float(self.get_parameter("imu.maximum_gap_sec").value),
            ),
            sample_buffer_size=self._imu_sample_buffer_size,
        )
        self._trajectory_tum_path = _resolve_path(
            str(self.get_parameter("trajectory_tum_path").value),
            self._work_dir,
        ) if str(self.get_parameter("trajectory_tum_path").value) else (
            self._work_dir / "trajectory_tum.txt"
        )
        self._trajectory_tum_file = None
        self._covariance_diagnostics_file = None
        self._covariance_diagnostics_lock = threading.Lock()
        self._work_dir_lock_file = None
        self._imu_filter = (
            ImuIteratedErrorStateFilter(
                imu_filter_config,
                self._T_base_imu[:3, :3],
                self._T_base_imu[:3, 3],
            )
            if self._imu_enabled else None
        )
        self._imu_condition = threading.Condition()
        self._imu_factor_samples: deque[
            tuple[int, np.ndarray, np.ndarray]
        ] = deque(maxlen=max(100, self._imu_queue_depth))
        self._last_fused_output_pose: np.ndarray | None = None
        self._last_fused_output_covariance: np.ndarray | None = None
        self._last_fused_output_stamp_ns: int | None = None
        self._imu_clone_stamp_ns: int | None = None
        # The processing worker handles synchronized image pairs in order.
        # Retain the previous image epoch so the explicit relative-motion
        # message can identify the exact interval represented by each pose.
        self._last_processed_stamp_ns: int | None = None
        self._imu_initialization_logged = False
        self._last_pose_prior: dict[str, Any] | None = None

        if len(self._image_topics) != len(self._camera_names):
            raise ValueError("image_topics and camera_names must have the same length")
        if len(self._camera_frame_ids) != len(self._camera_names):
            raise ValueError("camera_frame_ids and camera_names must have the same length")
        if not self._image_topics:
            raise ValueError("At least one image topic is required")
        if self._base_frame_id != self._child_frame_id:
            raise ValueError(
                "base_frame_id and child_frame_id must match: T_camera_base "
                "defines the frame published as Odometry.child_frame_id"
            )
        if self._world_frame_id == self._child_frame_id:
            raise ValueError("world_frame_id and child_frame_id must be different")
        if len(set(self._camera_frame_ids)) != len(self._camera_frame_ids):
            raise ValueError("camera_frame_ids must be unique")
        if self._child_frame_id in self._camera_frame_ids:
            raise ValueError("child_frame_id must not also be a camera_frame_id")
        if self._imu_enabled and not self._imu_topic:
            raise ValueError("imu.topic must not be empty when imu.enabled is true")

        try:
            from .odometry.odometry import VisualOdometry
        except Exception as exc:  # pragma: no cover - import-time dependency guard.
            raise RuntimeError(
                "Could not import Colmap-Odometry core dependencies. "
                "Check that rosdep or the Python environment installed pycolmap, "
                "pyquaternion, torch, kornia, and the selected feature dependencies."
            ) from exc

        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_work_dir_lock()
        self._open_covariance_diagnostics_log()
        self._images_dir = self._work_dir / "images"
        for camera_name in self._camera_names:
            (self._images_dir / camera_name).mkdir(parents=True, exist_ok=True)
        if self._save_trajectory_tum:
            self._open_tum_trajectory_log()

        config = _load_yaml(config_file)
        camera_config = _load_yaml(camera_file)
        rig_config = _load_yaml(rig_file)
        self._apply_camera_names(config)
        config.setdefault("general", {})[
            "feature_track_publish_min_length"
        ] = self._feature_track_publish_min_length
        config["general"]["create_feature_tracks"] = self._create_feature_tracks
        self._feature_type = str(
            config.get("local_features", {}).get("features_name", "unknown"))

        self._vo = VisualOdometry(
            working_dir=self._work_dir,
            config=config,
            camera_config=camera_config,
            rig_config=rig_config,
        )
        self.get_logger().info(
            "Rig YAML convention: ROS T_rig_camera (camera pose in rig); "
            "internally using COLMAP T_camera_rig = inverse(T_rig_camera)")
        for camera_name in self._camera_names:
            T_camera_rig = _rig_transform_to_matrix(rig_config, camera_name)
            camera_center = np.linalg.inv(T_camera_rig)[:3, 3]
            q_camera_rig = Quaternion(matrix=T_camera_rig[:3, :3]).normalised
            self.get_logger().info(
                f"Rig {camera_name}: ROS center_in_rig="
                f"[{camera_center[0]:.6f}, {camera_center[1]:.6f}, "
                f"{camera_center[2]:.6f}]; COLMAP camera_from_rig "
                f"q_xyzw=[{q_camera_rig.x:.7f}, {q_camera_rig.y:.7f}, "
                f"{q_camera_rig.z:.7f}, {q_camera_rig.w:.7f}], "
                f"t=[{T_camera_rig[0, 3]:.6f}, "
                f"{T_camera_rig[1, 3]:.6f}, {T_camera_rig[2, 3]:.6f}]")

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._processing_queue_warn_size = max(
            0, int(self.get_parameter("processing_queue_size").value))
        self._processing_condition = threading.Condition()
        self._processing_queue: deque[tuple[Image | CompressedImage, ...]] = deque()
        self._processing_stop = False
        self._processing_thread = threading.Thread(
            target=self._processing_worker,
            name="colmap-odometry-processing",
            daemon=True,
        )
        self._processing_thread.start()
        self._frame_count = 0
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_stop = threading.Event()
        self._diagnostic_state: dict[str, Any] = {
            "active": False,
            "ever_received": False,
            "frame": 0,
            "image": "",
            "frame_started": 0.0,
            "stage": "",
            "stage_started": 0.0,
            "last_callback": time.monotonic(),
            "last_input_received": time.monotonic(),
            "last_input_gap": 0.0,
            "last_warning": 0.0,
            "completed": {},
        }
        self._diagnostic_thread = None
        self._last_base_translation = np.zeros(3, dtype=np.float64)
        self._last_base_quaternion = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
        self._origin_pose_published = False
        self._path_msg = PathMsg()
        self._path_msg.header.frame_id = self._world_frame_id
        self._path_publish_period_sec = float(
            self.get_parameter("path_publish_period_sec").value)
        if self._path_publish_period_sec < 0.0:
            raise ValueError("path_publish_period_sec must be non-negative")
        self._last_path_publish_monotonic = float("-inf")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        estimator_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL,
            durability=DurabilityPolicy.VOLATILE,
            depth=0,
        )
        self._odom_pub = self.create_publisher(Odometry, "~/odom", estimator_qos)
        self._pose_pub = self.create_publisher(PoseStamped, "~/pose", estimator_qos)
        # Keep only the newest complete snapshot. Path is visualization data;
        # historical snapshots must never queue behind one another.
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._path_pub = self.create_publisher(PathMsg, "~/path", path_qos)
        self._relative_pose_pub = self.create_publisher(
            PoseStamped, "~/relative_pose", estimator_qos)
        self._relative_pose_covariance_pub = self.create_publisher(
            PoseWithCovarianceStamped, "~/relative_pose_covariance", estimator_qos)
        self._relative_motion_pub = self.create_publisher(
            RelativePoseWithCovarianceStamped, "~/relative_motion", estimator_qos)
        self._status_pub = self.create_publisher(String, "~/status", qos)
        # This event stream is emitted only for frames without a visual PnP
        # update. It prevents a stale pose from looking like a fresh estimate.
        self._pnp_omission_pub = self.create_publisher(
            String, "~/pnp_omission", qos)
        self._feature_matches_pub = (
            self.create_publisher(Image, "~/feature_matches", qos)
            if self._publish_feature_matches else None
        )
        self._feature_tracks_pub = (
            self.create_publisher(FeatureTrackArray, "~/feature_tracks", qos)
            if self._create_feature_tracks and self._publish_feature_tracks else None
        )
        self._image_message_types = [
            CompressedImage if topic.rstrip("/").endswith("/compressed") else Image
            for topic in self._image_topics
        ]
        image_qos_reliability = str(
            self.get_parameter("image_qos_reliability").value).strip().lower()
        if image_qos_reliability == "reliable":
            reliability = ReliabilityPolicy.RELIABLE
        elif image_qos_reliability in ("best_effort", "besteffort"):
            reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            raise ValueError(
                "image_qos_reliability must be 'reliable' or 'best_effort'"
            )
        image_qos = QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(10, int(self.get_parameter("sync_queue_size").value)),
        )
        self._decompressed_image_pubs = [
            self.create_publisher(
                Image, f"~/input/{camera_name}/image_raw", image_qos)
            if message_type is CompressedImage else None
            for camera_name, message_type in zip(
                self._camera_names, self._image_message_types)
        ]
        self._sparse_point_cloud_pub = (
            self.create_publisher(PointCloud2, "~/sparse_points", qos)
            if self._publish_sparse_point_cloud else None
        )
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None
        self._static_tf_broadcaster = (
            StaticTransformBroadcaster(self) if self._publish_camera_tf else None
        )
        if self._publish_camera_tf:
            self._publish_static_camera_transforms(rig_config)

        subscribers = [
            message_filters.Subscriber(self, message_type, topic, image_qos)
            for topic, message_type in zip(
                self._image_topics, self._image_message_types)
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self._sync.registerCallback(self._on_images)

        self._imu_subscription = None
        self._imu_callback_group = None
        if self._imu_filter is not None:
            self._imu_callback_group = MutuallyExclusiveCallbackGroup()
            imu_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=self._imu_queue_depth,
            )
            self._imu_subscription = self.create_subscription(
                Imu,
                self._imu_topic,
                self._on_imu,
                imu_qos,
                callback_group=self._imu_callback_group,
            )
            lever_arm = self._T_base_imu[:3, 3]
            if np.linalg.norm(lever_arm) > 1.0e-6:
                self.get_logger().info(
                    "Using IMU lever-arm compensation from imu.T_base_imu: "
                    f"r_base_imu=[{lever_arm[0]:.4f}, {lever_arm[1]:.4f}, "
                    f"{lever_arm[2]:.4f}]m"
                )

        if self._performance_diagnostics:
            self._diagnostic_thread = threading.Thread(
                target=self._performance_watchdog,
                name="colmap-odometry-performance-watchdog",
                daemon=True,
            )
            self._diagnostic_thread.start()

        self.get_logger().info(
            "Colmap odometry ready: "
            f"topics={self._image_topics}, work_dir={self._work_dir}, "
            f"decompressed_topics={[
                f'~/input/{name}/image_raw'
                for name, publisher in zip(
                    self._camera_names, self._decompressed_image_pubs)
                if publisher is not None
            ]}, "
            f"config={config_file}, output_base_frame={self._child_frame_id}, "
            f"publish_tf={self._publish_tf}, "
            f"publish_camera_tf={self._publish_camera_tf}, "
            f"publish_feature_matches={self._publish_feature_matches}, "
            f"publish_sparse_point_cloud={self._publish_sparse_point_cloud}, "
            f"create_feature_tracks={self._create_feature_tracks}, "
            f"publish_feature_tracks={self._publish_feature_tracks}, "
            f"feature_track_min_length={self._feature_track_publish_min_length}, "
            f"save_trajectory_tum={self._save_trajectory_tum}, "
            f"imu_factors={'on' if self._imu_enabled else 'off'}, "
            f"performance_diagnostics={self._performance_diagnostics}, "
            f"pnp_covariance=spatial_cluster_sandwich(pixel_std_px="
            f"{self._vo.stereo_covariance_pixel_std_px:.3g}, min_points="
            f"{self._vo.stereo_covariance_min_points})"
            f", spatial_clusters={self._vo.stereo_covariance_spatial_clusters}, "
            "estimator=spatial_cluster_sandwich"
        )
        pycolmap_version = str(getattr(pycolmap, "__version__", "unknown"))
        if _pycolmap_version_tuple() < (3, 12):
            self.get_logger().warning(
                f"PyCOLMAP {pycolmap_version} is older than 3.12; "
                "pose-inverse covariance propagation is known to be "
                "incorrect. Install the pinned workspace version with uv "
                "before relying on covariance",
            )
        else:
            self.get_logger().info(
                f"COLMAP covariance backend: pycolmap={pycolmap_version} "
                f"({getattr(pycolmap, '__file__', 'unknown')}), tangent="
                "right-local [rotation, translation]",
            )
        if self._imu_filter is not None:
            self.get_logger().info(
                "COLMAP minimal-solver IMU factors: "
                f"topic={self._imu_topic}, output_rate=image, "
                "residuals=[gyro_rotation gravity_direction], "
                "output=joint_visual_inertial_pose, "
                "post_filter=off, "
                f"sample_buffer={'lossless' if self._imu_sample_buffer_size == 0 else self._imu_sample_buffer_size}"
            )
        self.get_logger().info(self._vo.compute_backend_summary())
        self.get_logger().info(self._vo.stereo_filter_summary())
        self.get_logger().info(
            "Colmap odometry node initialized; the processing worker will "
            "report readiness after LoMa warm-up."
        )

    def destroy_node(self) -> bool:
        # Drain all synchronized pairs before closing the database and track
        # publisher. This is important for offline bag playback and also makes
        # Ctrl-C deterministic when a processing backlog exists.
        with self._processing_condition:
            self._processing_stop = True
            self._processing_condition.notify_all()
        self._processing_thread.join()
        if self._feature_tracks_pub is not None and hasattr(self, "_vo"):
            with self._lock:
                self._vo.finish_feature_tracks()
            self._publish_feature_track_updates(self.get_clock().now().to_msg())
        self._diagnostic_stop.set()
        if self._diagnostic_thread is not None:
            self._diagnostic_thread.join(timeout=1.0)
            self._diagnostic_thread = None
        if self._trajectory_tum_file is not None:
            self._trajectory_tum_file.close()
            self._trajectory_tum_file = None
        if self._covariance_diagnostics_file is not None:
            with self._covariance_diagnostics_lock:
                self._covariance_diagnostics_file.flush()
                self._covariance_diagnostics_file.close()
                self._covariance_diagnostics_file = None
        if self._work_dir_lock_file is not None:
            fcntl.flock(self._work_dir_lock_file.fileno(), fcntl.LOCK_UN)
            self._work_dir_lock_file.close()
            self._work_dir_lock_file = None
        return super().destroy_node()

    def _open_covariance_diagnostics_log(self) -> None:
        """Open the detailed per-frame covariance/correspondence log."""
        if not bool(self.get_parameter(
                "diagnostics.covariance_log_enabled").value):
            return
        configured = str(self.get_parameter(
            "diagnostics.covariance_log_path").value).strip()
        if not configured:
            configured = "covariance_diagnostics.jsonl"
        path = _resolve_path(configured, self._work_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._covariance_diagnostics_file = path.open(
            "w", encoding="utf-8", buffering=1)
        self.get_logger().info(
            f"Detailed covariance diagnostics: {path}")

    @staticmethod
    def _diagnostics_json_value(value: Any) -> Any:
        """Convert NumPy/Python containers into JSON-safe diagnostics."""
        if isinstance(value, np.ndarray):
            return ColmapOdometryNode._diagnostics_json_value(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {
                str(key): ColmapOdometryNode._diagnostics_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ColmapOdometryNode._diagnostics_json_value(item)
                    for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _covariance_summary(value: Any) -> dict[str, Any] | None:
        """Return diagonal/eigenvalue summaries without losing the matrix."""
        if value is None:
            return None
        try:
            matrix = np.asarray(value, dtype=np.float64).reshape(6, 6)
            matrix = 0.5 * (matrix + matrix.T)
            if not np.all(np.isfinite(matrix)):
                return None
            eigenvalues = np.linalg.eigvalsh(matrix)
            return {
                "diagonal": np.diag(matrix),
                "eigenvalues": eigenvalues,
                "condition_number": float(
                    eigenvalues[-1] / max(eigenvalues[0], 1.0e-30)),
            }
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None

    def _write_covariance_diagnostics(
        self,
        stamp_ns: int,
        image_name: str,
        timings: dict[str, float],
    ) -> None:
        """Persist one detailed visual covariance record as JSONL."""
        stream = self._covariance_diagnostics_file
        if stream is None or not hasattr(self, "_vo"):
            return
        current = getattr(self._vo, "log_data", {}).get("current_frame", {})
        if current.get("frame_name") not in (None, "", image_name):
            return
        prior = self._last_pose_prior or {}
        covariance = current.get("covariance_matrix_angles_translations")
        relative_covariance = current.get(
            "relative_pose_covariance_rotation_translation")
        record = {
            "record_type": "colmap_covariance",
            "stamp_ns": int(stamp_ns),
            "image": image_name,
            "frame": int(getattr(self._vo, "log_data", {}).get(
                "frame_count", 0) or 0),
            "status": current.get("status"),
            "pnp": {
                "success": current.get("pnp_success", False),
                "pose_source": current.get("pnp_pose_source", "none"),
                "temporal_match_source": current.get(
                    "pnp_temporal_match_source", "none"),
                "raw_temporal_matches": current.get(
                    "temporal_raw_matches", 0),
                "magsac_temporal_matches": current.get(
                    "magsac_temporal_matches", 0),
                "magsac_inliers": current.get("magsac_inliers", 0),
                "correspondences": current.get("pnp_correspondences", 0),
                "ransac_inliers": current.get("pnp_ransac_inliers", 0),
                "prior_inliers": current.get("pnp_prior_inliers", 0),
                "ransac_median_error_px": current.get(
                    "pnp_ransac_median_error_px"),
                "prior_median_error_px": current.get(
                    "pnp_prior_median_error_px"),
                "magsac_degenerate": current.get("magsac_degenerate", False),
                "magsac_failure_reason": current.get(
                    "magsac_failure_reason", ""),
            },
            "structureless": {
                "source": current.get("pnp_covariance_source", "none"),
                "success": current.get("pnp_covariance_source") == (
                    "anchored_inverse_depth_stereo_temporal"),
                "used_points": current.get("structureless_used_points",
                                             current.get("pnp_covariance_points", 0)),
                "right_observations": current.get(
                    "structureless_right_observations", 0),
                "left_only_observations": max(
                    0,
                    int(current.get("structureless_used_points", 0) or 0)
                    - int(current.get("structureless_right_observations", 0) or 0),
                ),
                "iterations": current.get("structureless_iterations", 0),
                "initial_cost": current.get("structureless_initial_cost"),
                "final_cost": current.get("structureless_final_cost"),
                "reprojection_rms_px": current.get(
                    "structureless_reprojection_rms_px"),
                "information_eigenvalues": current.get(
                    "structureless_information_eigenvalues"),
                "visual_information_eigenvalues": current.get(
                    "structureless_visual_information_eigenvalues"),
                "inertial_information_eigenvalues": current.get(
                    "structureless_inertial_information_eigenvalues"),
                "view_residual_rms_px": current.get(
                    "structureless_view_residual_rms_px"),
                "robust_weight_stats": current.get(
                    "structureless_robust_weight_stats"),
                "covariance_estimator": current.get(
                    "structureless_covariance_estimator", "unknown"),
                "view_residual_scales": current.get(
                    "structureless_view_residual_scales"),
                "failure_reason": current.get(
                    "structureless_failure_reason"),
            },
            "covariance": {
                "pixel_std_px": float(self._vo.stereo_covariance_pixel_std_px),
                "robust_delta_px": float(self._vo.structureless_robust_delta_px),
                "min_points": int(self._vo.stereo_covariance_min_points),
                "spatial_clusters": int(
                    self._vo.stereo_covariance_spatial_clusters),
                "geometric_verification_method": getattr(
                    self._vo, "geometric_verification_method", "unknown"),
                "magsac_confidence": getattr(
                    self._vo, "magsac_confidence", None),
                "magsac_max_iterations": getattr(
                    self._vo, "magsac_max_iterations", None),
                "camera_pose": covariance,
                "camera_pose_summary": self._covariance_summary(covariance),
                "relative_pose_rotation_translation": relative_covariance,
                "relative_pose_summary": self._covariance_summary(
                    relative_covariance),
            },
            "inertial": {
                "rotation_factor_used": current.get(
                    "inertial_rotation_factor_used", False),
                "gravity_factor_used": current.get(
                    "inertial_gravity_factor_used", False),
                "rotation_prior": prior.get("rotation_prior"),
                "rotation_covariance": prior.get("rotation_covariance"),
                "gravity_sigma_rad": prior.get("gravity_sigma_rad"),
            },
            "timings": timings,
        }
        encoded = json.dumps(
            self._diagnostics_json_value(record),
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._covariance_diagnostics_lock:
            stream.write(encoded + "\n")
            stream.flush()

    def _acquire_work_dir_lock(self) -> None:
        """Prevent two odometry nodes from mutating one COLMAP database."""
        lock_path = self._work_dir / ".colmap_odometry.lock"
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise RuntimeError(
                "COLMAP work_dir is already in use by another odometry node: "
                f"{self._work_dir}. Stop the duplicate launch or configure a "
                "different work_dir; concurrent access corrupts image IDs."
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} node={self.get_fully_qualified_name()}\n")
        lock_file.flush()
        self._work_dir_lock_file = lock_file

    def _begin_performance_frame(
        self,
        image_name: str,
        frame_started: float,
        timings: dict[str, float],
    ) -> None:
        if not self._performance_diagnostics:
            return
        with self._diagnostic_lock:
            if self._diagnostic_state["ever_received"]:
                # This is the gap between synchronized-pair arrivals, not the
                # time spent waiting behind an already queued visual frame.
                timings["input_gap"] = float(
                    self._diagnostic_state["last_input_gap"])
            self._diagnostic_state.update(
                {
                    "active": True,
                    "ever_received": True,
                    "frame": self._frame_count + 1,
                    "image": image_name,
                    "frame_started": frame_started,
                    "stage": "callback_start",
                    "stage_started": frame_started,
                    "last_warning": 0.0,
                    "completed": {},
                }
            )

    @contextmanager
    def _performance_stage(self, name: str, timings: dict[str, float]):
        started = time.monotonic()
        if self._performance_diagnostics:
            with self._diagnostic_lock:
                self._diagnostic_state["stage"] = name
                self._diagnostic_state["stage_started"] = started
        try:
            yield
        finally:
            duration = time.monotonic() - started
            timings[name] = timings.get(name, 0.0) + duration
            if self._performance_diagnostics:
                with self._diagnostic_lock:
                    completed = dict(self._diagnostic_state["completed"])
                    completed[name] = completed.get(name, 0.0) + duration
                    self._diagnostic_state["completed"] = completed
                    self._diagnostic_state["stage"] = "between_stages"
                    self._diagnostic_state["stage_started"] = time.monotonic()

    def _finish_performance_frame(
        self,
        frame_started: float,
        timings: dict[str, float],
    ) -> None:
        if not self._performance_diagnostics:
            return

        finished = time.monotonic()
        total = finished - frame_started
        timings["total"] = total
        with self._diagnostic_lock:
            frame = int(self._diagnostic_state["frame"])
            image_name = str(self._diagnostic_state["image"])
            self._diagnostic_state["active"] = False
            self._diagnostic_state["stage"] = ""
            self._diagnostic_state["last_callback"] = finished
            self._diagnostic_state["last_warning"] = 0.0

        periodic = frame % self._performance_log_every_n_frames == 0
        slow = total >= self._performance_slow_frame_sec
        if not periodic and not slow:
            return

        wrapper_order = (
            "input_gap",
            "sync_span",
            "image_decode",
            "image_republish",
            "image_write",
            "imu_prediction",
            "visual_odometry",
            "imu_correction",
            "status_publish",
            "feature_debug",
            "point_cloud_publish",
            "pose_publish",
            "total",
        )
        core_order = (
            "core_features",
            "core_feature_model",
            "core_feature_transfer",
            "core_matching",
            "core_matching_model",
            "core_matching_geometry",
            "core_matching_database",
            "core_database",
            "core_keyframe_selection",
            "core_stereo_observation",
            "core_reconstruction",
            "core_minimal_solver",
        )
        wrapper_text = " ".join(
            f"{name}={timings[name] * 1000.0:.1f}ms"
            for name in wrapper_order if name in timings
        )
        core_text = " ".join(
            f"{name}={timings[name] * 1000.0:.1f}ms"
            for name in core_order if name in timings
        )
        message = f"[perf] frame={frame} image={image_name} {wrapper_text}"
        if core_text:
            message += f" ({core_text})"
        current_log = getattr(self._vo, "log_data", {}).get("current_frame", {})
        stereo_raw = int(current_log.get("stereo_matches_raw", 0) or 0)
        stereo_candidates = int(
            current_log.get("stereo_points_candidates", 0) or 0)
        if stereo_raw or stereo_candidates:
            message += (
                " stereo["
                f"epipolar={int(current_log.get('stereo_matches_epipolar', 0) or 0)}/"
                f"{stereo_raw} "
                f"valid={int(current_log.get('stereo_points_valid', 0) or 0)}/"
                f"{stereo_candidates} "
                "reject="
                f"index:{int(current_log.get('stereo_rejected_indices', 0) or 0)},"
                f"tri:{int(current_log.get('stereo_rejected_triangulation', 0) or 0)},"
                f"depth:{int(current_log.get('stereo_rejected_depth', 0) or 0)},"
                f"cheirality:{int(current_log.get('stereo_rejected_cheirality', 0) or 0)},"
                f"reproj:{int(current_log.get('stereo_rejected_reprojection', 0) or 0)},"
                f"angle:{int(current_log.get('stereo_rejected_angle', 0) or 0)}]"
            )
        if self._imu_filter is not None:
            message += (
                " imu["
                f"initialized={bool(current_log.get('imu_pose_initialized', False))} "
                f"rotation_factor={bool(current_log.get('inertial_rotation_factor_used', False))} "
                f"gravity_factor={bool(current_log.get('inertial_gravity_factor_used', False))} "
                f"pnp={current_log.get('pnp_pose_source', 'none')} "
                "post_filter=off"
            )
            buffered_samples = current_log.get("imu_buffered_samples")
            buffer_span = current_log.get("imu_buffer_span_sec")
            dropped_samples = current_log.get("imu_dropped_buffered_samples")
            if isinstance(buffered_samples, int):
                message += f" buffer={buffered_samples}"
            if isinstance(buffer_span, (int, float)):
                message += f"/{buffer_span:.1f}s"
            if isinstance(dropped_samples, int) and dropped_samples > 0:
                message += f" dropped={dropped_samples}"
            message += "]"
        if slow:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def _performance_watchdog(self) -> None:
        """Report stalls from a thread independent of the ROS executor."""
        poll_period = min(0.5, self._performance_stall_warn_sec / 2.0)
        while not self._diagnostic_stop.wait(poll_period):
            now = time.monotonic()
            with self._diagnostic_lock:
                state = dict(self._diagnostic_state)

            if state["active"]:
                stage_elapsed = now - float(state["stage_started"])
                if stage_elapsed < self._performance_stall_warn_sec:
                    continue
                if now - float(state["last_warning"]) < self._performance_stall_repeat_sec:
                    continue
                completed = " ".join(
                    f"{name}={duration * 1000.0:.1f}ms"
                    for name, duration in state["completed"].items()
                ) or "none"
                stage = str(state["stage"])
                if stage == "visual_odometry":
                    core_stage = getattr(self._vo, "performance_stage", "unknown")
                    stage += f"/{core_stage}"
                self.get_logger().warning(
                    f"[perf] STALL frame={state['frame']} image={state['image']} "
                    f"stage={stage} stage_elapsed={stage_elapsed:.1f}s "
                    f"frame_elapsed={now - float(state['frame_started']):.1f}s "
                    f"completed={completed}"
                )
                with self._diagnostic_lock:
                    self._diagnostic_state["last_warning"] = now
                continue

            input_gap = now - float(state["last_input_received"])
            with self._processing_condition:
                backlog = len(self._processing_queue)
            if (
                state["ever_received"]
                and backlog == 0
                and self._performance_input_gap_warn_sec > 0.0
                and input_gap >= self._performance_input_gap_warn_sec
                and now - float(state["last_warning"]) >= self._performance_stall_repeat_sec
            ):
                self.get_logger().warning(
                    f"[perf] INPUT STALL no synchronized image callback for "
                    f"{input_gap:.1f}s after frame={state['frame']}; check image rates, "
                    "timestamps, QoS, and synchronization slop"
                )
                with self._diagnostic_lock:
                    self._diagnostic_state["last_warning"] = now

    def _apply_camera_names(self, config: dict[str, Any]) -> None:
        mapping = config.setdefault("mapping", {})
        mapping["cameras"] = self._camera_names
        if len(self._camera_names) == 1:
            mapping["rig_match_rule"] = []
            mapping["cameras_for_baseline_estim"] = [self._camera_names[0]]

    @staticmethod
    def _stamp_to_nanoseconds(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _on_imu(self, message: Imu) -> None:
        if self._imu_filter is None:
            return
        timestamp_ns = (
            self._stamp_to_nanoseconds(message.header.stamp)
            + self._imu_time_offset_ns
        )
        acceleration_imu = np.array(
            [
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        angular_velocity_imu = np.array(
            [
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        accepted = self._imu_filter.add_sample(
            timestamp_ns, acceleration_imu, angular_velocity_imu)
        if not accepted:
            return
        prepared_sample = self._imu_filter.latest_sample
        if prepared_sample is None:
            return
        with self._imu_condition:
            self._imu_factor_samples.append((
                timestamp_ns,
                prepared_sample.acceleration.copy(),
                prepared_sample.angular_velocity.copy(),
            ))
            self._imu_condition.notify_all()
        if (
            self._imu_filter.imu_initialized
            and not self._imu_initialization_logged
        ):
            self._imu_initialization_logged = True
            diagnostics = self._imu_filter.diagnostics()
            mean_acceleration = diagnostics["initial_mean_acceleration"]
            gyro_bias = diagnostics["gyro_bias"]
            self.get_logger().info(
                "IMU initialization complete: base-frame mean_acc="
                f"[{mean_acceleration[0]:.4f}, {mean_acceleration[1]:.4f}, "
                f"{mean_acceleration[2]:.4f}]m/s^2 "
                f"norm={diagnostics['initial_mean_acceleration_norm']:.4f} "
                f"scale={diagnostics['acceleration_scale']:.7f}, "
                f"stationary={diagnostics['initialization_stationary']}, "
                f"gyro_bias=[{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, "
                f"{gyro_bias[2]:.6f}]rad/s; waiting for a camera pose to "
                "initialize the minimal-solver IMU preintegrator"
            )

    def _wait_for_imu(self, timestamp_ns: int) -> None:
        if self._imu_filter is None or self._imu_wait_timeout_sec <= 0.0:
            return
        deadline = time.monotonic() + self._imu_wait_timeout_sec
        with self._imu_condition:
            while True:
                latest = self._imu_filter.latest_sample_timestamp_ns
                if latest is not None and latest >= timestamp_ns:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return
                self._imu_condition.wait(timeout=remaining)

    def _prepare_imu_prediction(
        self,
        timestamp_ns: int,
        previous_timestamp_ns: int | None,
    ) -> dict[str, Any] | None:
        if self._imu_filter is None:
            return None
        if (
            previous_timestamp_ns is None
            or self._imu_clone_stamp_ns != int(previous_timestamp_ns)
        ):
            # The IMU clone and the visual solver must describe the same
            # consecutive-frame interval. After a genuine visual omission,
            # wait for the next accepted visual pose to re-anchor rather than
            # attaching an older inertial interval to a newer image edge.
            return None
        self._wait_for_imu(timestamp_ns)
        if not self._imu_filter.pose_initialized:
            return None
        if not self._imu_filter.predict_to(timestamp_ns):
            self.get_logger().warning(
                "IMU prediction did not reach the image timestamp; this frame "
                "will use visual odometry without an inertial PnP prior",
                throttle_duration_sec=2.0,
            )
            return None

        prediction = self._imu_filter.relative_pose_prediction()
        if prediction is None:
            return None
        previous_base_from_current_base, relative_covariance = prediction
        previous_camera_from_current_camera = (
            self._T_camera_base
            @ previous_base_from_current_base
            @ self._T_base_camera
        )
        # The structureless solver estimates T_currentCamera_previousCamera.
        # Transport the right-local clone-to-current covariance through the
        # base/camera conjugation and pose inversion, then retain its rotation
        # block as a genuine residual rather than merely an initial guess.
        ros_order = np.array([3, 4, 5, 0, 1, 2])
        covariance_base_rt = np.asarray(
            relative_covariance, dtype=np.float64).reshape(6, 6)
        covariance_base_rt = covariance_base_rt[np.ix_(ros_order, ros_order)]
        camera_adjoint = _adjoint_rotation_translation(self._T_camera_base)
        covariance_previous_from_current_camera = (
            camera_adjoint @ covariance_base_rt @ camera_adjoint.T)
        inverse_adjoint = _adjoint_rotation_translation(
            previous_camera_from_current_camera)
        covariance_current_from_previous_camera = (
            inverse_adjoint
            @ covariance_previous_from_current_camera
            @ inverse_adjoint.T
        )
        current_from_previous_camera = np.linalg.inv(
            previous_camera_from_current_camera)
        # The structureless optimizer applies a left-local perturbation,
        # whereas the cloned IMU covariance is right-local. Transport it into
        # the solver tangent before extracting the rotational factor.
        solver_adjoint = _adjoint_rotation_translation(
            current_from_previous_camera)
        covariance_current_from_previous_camera = (
            solver_adjoint
            @ covariance_current_from_previous_camera
            @ solver_adjoint.T
        )
        rotation_covariance = 0.5 * (
            covariance_current_from_previous_camera[:3, :3]
            + covariance_current_from_previous_camera[:3, :3].T)
        try:
            values, vectors = np.linalg.eigh(rotation_covariance)
            floor = self._imu_rotation_factor_std_floor ** 2
            rotation_covariance = vectors @ np.diag(
                np.maximum(values, floor)) @ vectors.T
        except np.linalg.LinAlgError:
            rotation_covariance = np.eye(3) * (
                self._imu_rotation_factor_std_floor ** 2)

        gravity_previous_camera = None
        gravity_current_camera = None
        gravity_sigma = None
        if self._imu_gravity_factor_enabled:
            gravity_previous_base = (
                self._imu_filter.clone_gravity_up_direction())
            gravity_measurement = self._gravity_factor_measurement(
                timestamp_ns, previous_timestamp_ns)
            if gravity_previous_base is not None and gravity_measurement is not None:
                gravity_current_base, gravity_sigma = gravity_measurement
                gravity_previous_camera = (
                    self._T_camera_base[:3, :3] @ gravity_previous_base)
                gravity_current_camera = (
                    self._T_camera_base[:3, :3] @ gravity_current_base)
        return {
            "world_from_camera": previous_camera_from_current_camera,
            "relative_covariance_base": relative_covariance,
            "rotation_prior": current_from_previous_camera[:3, :3],
            "rotation_covariance": rotation_covariance,
            "gravity_previous": gravity_previous_camera,
            "gravity_current": gravity_current_camera,
            "gravity_sigma_rad": gravity_sigma,
            "maximum_error_px": self._imu_prior_maximum_error_px,
            "minimum_inliers": self._imu_prior_minimum_inliers,
        }

    def _gravity_factor_measurement(
        self,
        timestamp_ns: int,
        previous_timestamp_ns: int | None,
    ) -> tuple[np.ndarray, float] | None:
        """Return a motion-gated current-frame specific-force direction."""
        if previous_timestamp_ns is None:
            return None
        start_ns = max(
            int(previous_timestamp_ns),
            int(timestamp_ns) - self._imu_gravity_factor_window_ns,
        )
        with self._imu_condition:
            samples = [
                item for item in self._imu_factor_samples
                if start_ns < item[0] <= timestamp_ns
            ]
        if len(samples) < self._imu_gravity_factor_minimum_samples:
            return None
        accelerations = np.stack([item[1] for item in samples])
        angular_velocities = np.stack([item[2] for item in samples])
        mean_acceleration = np.mean(accelerations, axis=0)
        magnitude = float(np.linalg.norm(mean_acceleration))
        if (
            not np.isfinite(magnitude)
            or magnitude <= 1.0e-6
            or abs(magnitude - self._imu_filter.config.gravity_magnitude)
            > self._imu_gravity_factor_acceleration_tolerance
            or float(np.max(np.linalg.norm(angular_velocities, axis=1)))
            > self._imu_gravity_factor_max_gyro_norm
        ):
            return None
        directions = accelerations / np.maximum(
            np.linalg.norm(accelerations, axis=1)[:, None], 1.0e-9)
        direction = mean_acceleration / magnitude
        angular_errors = np.arccos(np.clip(
            directions @ direction, -1.0, 1.0))
        # Measurement dispersion weakens the direction factor automatically;
        # it can never become more confident than the configured floor.
        sigma = max(
            self._imu_gravity_factor_std,
            float(np.std(angular_errors))
            / math.sqrt(max(1, len(angular_errors))),
        )
        return direction, sigma

    def _on_images(self, *image_msgs: Image | CompressedImage) -> None:
        # message_filters invokes this callback from the ROS executor. Keep it
        # non-blocking: the visual backend may take longer than the camera
        # period, so doing work here would fill DDS/message_filters queues and
        # silently lose later frames.
        received = time.monotonic()
        with self._diagnostic_lock:
            previous = float(self._diagnostic_state["last_input_received"])
            self._diagnostic_state["ever_received"] = True
            self._diagnostic_state["last_input_gap"] = max(
                0.0, received - previous)
            self._diagnostic_state["last_input_received"] = received
        with self._processing_condition:
            if self._processing_stop:
                return
            self._processing_queue.append(tuple(image_msgs))
            if (
                self._processing_queue_warn_size > 0
                and len(self._processing_queue) >= self._processing_queue_warn_size
            ):
                self.get_logger().warning(
                    "Visual processing backlog has %d synchronized pairs; "
                    "frames are retained and will be processed in order",
                    len(self._processing_queue),
                    throttle_duration_sec=2.0,
                )
            self._processing_condition.notify()

    def _processing_worker(self) -> None:
        try:
            warmup_seconds = self._vo.warmup_loma()
            if warmup_seconds > 0.0:
                self.get_logger().info(
                    "LoMa compile and production-shape warm-up complete on "
                    "the processing worker; ready for image processing "
                    f"({warmup_seconds:.1f}s)"
                )
            else:
                self.get_logger().info(
                    "Colmap odometry processing worker ready for image processing"
                )
        except Exception as exc:
            self.get_logger().error(
                f"LoMa worker warm-up failed; image processing will continue: {exc!r}"
            )
        while True:
            with self._processing_condition:
                self._processing_condition.wait_for(
                    lambda: self._processing_stop or self._processing_queue)
                if self._processing_stop and not self._processing_queue:
                    return
                image_msgs = self._processing_queue.popleft()
            self._process_image_batch(image_msgs)

    def _process_image_batch(
        self,
        image_msgs: tuple[Image | CompressedImage, ...],
    ) -> None:
        frame_started = time.monotonic()
        timings: dict[str, float] = {}
        if image_msgs:
            stamp_ns = [
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
                for msg in image_msgs
            ]
            timings["sync_span"] = (max(stamp_ns) - min(stamp_ns)) * 1e-9
        image_name = _sanitize_frame_name(image_msgs[0].header.stamp) if image_msgs else "unknown"
        self._begin_performance_frame(image_name, frame_started, timings)
        try:
            self._process_images(list(image_msgs), timings)
        except Exception as exc:
            self.get_logger().error(
                "Colmap odometry callback failed: "
                f"{type(exc).__name__}: {exc!r}\n{traceback.format_exc()}",
                throttle_duration_sec=2.0,
            )
        finally:
            if image_msgs:
                self._write_covariance_diagnostics(
                    self._stamp_to_nanoseconds(image_msgs[0].header.stamp),
                    image_name,
                    timings,
                )
            self._finish_performance_frame(frame_started, timings)

    def _process_images(
        self,
        image_msgs: list[Image | CompressedImage],
        timings: dict[str, float],
    ) -> None:
        if len(image_msgs) != len(self._camera_names):
            raise RuntimeError("Synchronized callback received the wrong number of images")

        stamp = image_msgs[0].header.stamp
        stamp_ns = self._stamp_to_nanoseconds(stamp)
        previous_processed_stamp_ns = self._last_processed_stamp_ns
        # Update this before doing expensive work. If a later stage reports a
        # frame failure, the visual frontend has nevertheless consumed this
        # image and the next successful interval starts at this epoch.
        self._last_processed_stamp_ns = stamp_ns
        image_name = _sanitize_frame_name(stamp)
        with self._performance_stage("image_decode", timings):
            images = [
                self._bridge.compressed_imgmsg_to_cv2(
                    # Compressed grayscale JPEGs are already single-channel.
                    # Asking cv_bridge for mono8 can incorrectly force a
                    # BGR/RGB-to-mono conversion and fail on that 1-channel
                    # input. Preserve the encoded channel layout here; the
                    # feature frontend normalizes mono/color inputs itself.
                    msg, desired_encoding="passthrough")
                if isinstance(msg, CompressedImage)
                else self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding=self._image_encoding)
                for msg in image_msgs
            ]

        with self._performance_stage("image_republish", timings):
            for msg, image, publisher in zip(
                    image_msgs, images, self._decompressed_image_pubs):
                if publisher is None:
                    continue
                image_msg = self._bridge.cv2_to_imgmsg(
                    image,
                    encoding=_compatible_ros_image_encoding(
                        image, self._image_encoding),
                )
                image_msg.header = msg.header
                publisher.publish(image_msg)

        if self._save_input_images:
            with self._performance_stage("image_write", timings):
                self._write_input_images(image_name, images)

        self._frame_count += 1
        reinitialize = (
            self._reinitialize_after_frames > 0 and
            self._frame_count % self._reinitialize_after_frames == 0
        )

        pose_prior = None
        if self._imu_filter is not None:
            with self._performance_stage("imu_prediction", timings):
                pose_prior = self._prepare_imu_prediction(
                    stamp_ns, previous_processed_stamp_ns)
        self._last_pose_prior = pose_prior

        with self._performance_stage("visual_odometry", timings):
            with self._lock:
                pose_changes, log = self._vo.run(
                    image_name,
                    images,
                    reinitialize=reinitialize,
                    pose_prior=pose_prior,
                )

        current_log = log.get("current_frame", {})
        for output_name, log_name in (
            ("core_features", "feature_extraction_time"),
            ("core_feature_model", "feature_model_time"),
            ("core_feature_transfer", "feature_tensor_transfer_time"),
            ("core_matching", "matching_time"),
            ("core_matching_model", "matching_model_time"),
            ("core_matching_geometry", "matching_geometry_time"),
            ("core_matching_database", "matching_database_time"),
            ("core_database", "database_time"),
            ("core_keyframe_selection", "keyframe_selection_time"),
            ("core_stereo_observation", "stereo_observation_time"),
            ("core_reconstruction", "reconstruction_time"),
            ("core_minimal_solver", "minimal_solver_time"),
        ):
            value = current_log.get(log_name)
            if isinstance(value, (int, float)):
                timings[output_name] = float(value)

        self._publish_feature_track_updates(stamp, image_msgs)

        if self._feature_matches_pub is not None:
            with self._performance_stage("feature_debug", timings):
                self._publish_feature_match_image(stamp, image_name, images)

        pose_change = self._select_pose_change(pose_changes)
        visual_pose_available = bool(
            pose_change is not None
            and len(pose_change) >= 6
            and pose_change[1] is not None
            and pose_change[4] is not None
            and pose_change[5] is not None
        )
        if not visual_pose_available:
            self._publish_pnp_omission(stamp, image_name, current_log)
        if self._imu_filter is not None:
            with self._performance_stage("imu_correction", timings):
                self._process_imu_fused_frame(
                    stamp,
                    stamp_ns,
                    image_name,
                    images,
                    pose_change,
                    current_log,
                    previous_processed_stamp_ns,
                )
            with self._performance_stage("status_publish", timings):
                status_msg = String()
                status_msg.data = json.dumps(self._compact_log(log), default=str)
                self._status_pub.publish(status_msg)
            return
        with self._performance_stage("status_publish", timings):
            status_msg = String()
            status_msg.data = json.dumps(self._compact_log(log), default=str)
            self._status_pub.publish(status_msg)
        if not visual_pose_available:
            # The bootstrap image defines the requested identity origin. Any
            # later failure is explicitly reported above and deliberately does
            # not append a duplicate odometry/path/TUM pose.
            if (
                not self._origin_pose_published
                and current_log.get("pnp_failure_reason")
                == "bootstrap_no_previous_stereo_frame"
            ):
                self._publish_absolute(
                    stamp,
                    self._last_base_translation,
                    self._last_base_quaternion,
                )
                self._origin_pose_published = True
            return

        _, _, delta_t, delta_q, cumulative_t, cumulative_q = pose_change

        with self._performance_stage("pose_publish", timings):
            cumulative_t = np.asarray(cumulative_t, dtype=np.float64).reshape(3)
            base_t, base_q = self._camera_pose_to_base_pose(cumulative_t, cumulative_q)
            self._last_base_translation = base_t
            self._last_base_quaternion = base_q
            self._origin_pose_published = True

            # Publish the interval information before the matching absolute
            # sample. ROS 2 provides no delivery ordering across topics, but
            # this gives consumers of the two messages the intended order.
            if delta_t is not None and delta_q is not None:
                relative_t, relative_q = self._camera_pose_to_base_pose(
                    np.asarray(delta_t, dtype=np.float64).reshape(3),
                    delta_q,
                )
                relative_covariance = self._camera_covariance_to_base_ros(
                    current_log.get(
                        "relative_pose_covariance_rotation_translation")
                    if current_log.get(
                        "relative_pose_covariance_rotation_translation")
                    is not None else current_log.get(
                        "covariance_matrix_angles_translations"),
                    relative_transform=_pose_to_matrix(relative_t, relative_q),
                )
                self._publish_relative(
                    stamp,
                    relative_t,
                    relative_q,
                    relative_covariance,
                    from_timestamp_ns=previous_processed_stamp_ns,
                    metadata=current_log,
                )
            self._publish_absolute(stamp, base_t, base_q)
            if self._sparse_point_cloud_pub is not None:
                with self._performance_stage("point_cloud_publish", timings):
                    self._publish_sparse_point_cloud_message(
                        stamp,
                        image_name,
                        images[0],
                        cumulative_t,
                        cumulative_q,
                    )

    def _process_imu_fused_frame(
        self,
        stamp,
        stamp_ns: int,
        image_name: str,
        images: list[np.ndarray],
        pose_change,
        current_log: dict[str, Any],
        previous_processed_stamp_ns: int | None,
    ) -> None:
        """Publish the joint minimal-solver pose and re-anchor IMU propagation.

        IMU information has already entered ``pose_change`` as gyro/gravity
        residuals in the visual solve. The former post-solve iEKF correction
        is intentionally absent: it applied inertial information a second
        time and allowed an IMU-only state to replace a valid visual pose.
        """
        if self._imu_filter is None:
            return

        raw_world_from_base = _pose_to_matrix(
            self._last_base_translation, self._last_base_quaternion)
        raw_world_from_camera = None
        visual_absolute_available = False
        visual_relative_base = None
        visual_covariance = None
        if pose_change is not None:
            _, _, delta_t, delta_q, cumulative_t, cumulative_q = pose_change
            if cumulative_t is not None and cumulative_q is not None:
                cumulative_t = np.asarray(
                    cumulative_t, dtype=np.float64).reshape(3)
                base_t, base_q = self._camera_pose_to_base_pose(
                    cumulative_t, cumulative_q)
                self._last_base_translation = base_t
                self._last_base_quaternion = base_q
                raw_world_from_base = _pose_to_matrix(base_t, base_q)
                raw_world_from_camera = self._T_base_camera @ _pose_to_matrix(
                    cumulative_t, cumulative_q)
                visual_absolute_available = True
            if delta_t is not None and delta_q is not None:
                relative_t, relative_q = self._camera_pose_to_base_pose(
                    np.asarray(delta_t, dtype=np.float64).reshape(3),
                    delta_q,
                )
                visual_relative_base = _pose_to_matrix(relative_t, relative_q)
                visual_covariance = self._camera_covariance_to_base_ros(
                    current_log.get(
                        "relative_pose_covariance_rotation_translation")
                    if current_log.get(
                        "relative_pose_covariance_rotation_translation")
                    is not None else current_log.get(
                        "covariance_matrix_angles_translations"),
                    relative_transform=visual_relative_base,
                )

        pose_initialized_this_frame = False
        if (
            not self._imu_filter.pose_initialized
            and self._imu_filter.imu_initialized
            and visual_absolute_available
        ):
            if self._imu_filter.initialize_pose(stamp_ns, raw_world_from_base):
                pose_initialized_this_frame = True
                self._imu_clone_stamp_ns = stamp_ns
                self.get_logger().info(
                    "IMU preintegrator anchored at the camera timestamp; the "
                    + (
                        "published visual world is gravity aligned"
                        if self._imu_filter.config.align_world_with_gravity
                        else "original visual reference frame is preserved"
                    )
                )
        current_log["imu_initialized"] = self._imu_filter.imu_initialized
        current_log["imu_pose_initialized"] = self._imu_filter.pose_initialized
        current_log["imu_prediction_aligned"] = bool(
            current_log.get("inertial_rotation_factor_used", False))
        imu_diagnostics = self._imu_filter.diagnostics()
        current_log["imu_initial_mean_acceleration_norm"] = (
            imu_diagnostics["initial_mean_acceleration_norm"])
        current_log["imu_acceleration_scale"] = (
            imu_diagnostics["acceleration_scale"])
        current_log["imu_initialization_stationary"] = (
            imu_diagnostics["initialization_stationary"])
        current_log["imu_gyro_bias"] = imu_diagnostics["gyro_bias"]
        current_log["imu_accel_bias"] = imu_diagnostics["accel_bias"]
        current_log["imu_gravity"] = imu_diagnostics["gravity"]
        current_log["imu_velocity"] = imu_diagnostics["velocity"]
        current_log["imu_buffered_samples"] = (
            imu_diagnostics["buffered_samples"])
        current_log["imu_buffer_span_sec"] = (
            imu_diagnostics["buffer_span_sec"])
        current_log["imu_dropped_buffered_samples"] = (
            imu_diagnostics["dropped_buffered_samples"])
        lever_arm_diagnostics = self._imu_filter.lever_arm_diagnostics()
        current_log["imu_lever_arm_tangential_acceleration"] = (
            np.asarray(
                lever_arm_diagnostics["tangential_acceleration"],
                dtype=np.float64,
            ).tolist())
        current_log["imu_lever_arm_centripetal_acceleration"] = (
            np.asarray(
                lever_arm_diagnostics["centripetal_acceleration"],
                dtype=np.float64,
            ).tolist())
        current_log["imu_lever_arm_angular_acceleration"] = (
            np.asarray(
                lever_arm_diagnostics["angular_acceleration"],
                dtype=np.float64,
            ).tolist())
        current_log["imu_lever_arm_angular_acceleration_valid"] = bool(
            lever_arm_diagnostics["angular_acceleration_valid"])

        # IMU initialization/gaps never suppress a valid visual result. When
        # initialized, make that joint result the exact start of the next
        # preintegration interval; do not generate an IMU-only output pose.
        if not visual_absolute_available:
            if (
                not self._origin_pose_published
                and current_log.get("pnp_failure_reason")
                == "bootstrap_no_previous_stereo_frame"
            ):
                self._publish_absolute(
                    stamp,
                    self._last_base_translation,
                    self._last_base_quaternion,
                )
                self._origin_pose_published = True
            return
        if self._imu_filter.pose_initialized and not pose_initialized_this_frame:
            if self._imu_filter.anchor_visual_pose(stamp_ns, raw_world_from_base):
                self._imu_clone_stamp_ns = stamp_ns
        output_pose = (
            self._imu_filter.pose_matrix()
            if self._imu_filter.pose_initialized
            else raw_world_from_base
        )
        output_covariance = (
            self._imu_filter.pose_covariance_ros()
            if self._imu_filter.pose_initialized
            else None
        )
        output_translation, output_quaternion = _matrix_to_pose(output_pose)
        if visual_relative_base is not None:
            relative_translation, relative_quaternion = _matrix_to_pose(
                visual_relative_base)
            self._publish_relative(
                stamp,
                relative_translation,
                relative_quaternion,
                visual_covariance,
                from_timestamp_ns=previous_processed_stamp_ns,
                metadata=current_log,
            )
        self._publish_absolute(
            stamp,
            output_translation,
            output_quaternion,
            covariance=output_covariance,
        )
        self._origin_pose_published = True
        self._last_fused_output_pose = output_pose.copy()
        self._last_fused_output_covariance = (
            output_covariance.copy() if output_covariance is not None else None)
        self._last_fused_output_stamp_ns = stamp_ns
        if (
            self._sparse_point_cloud_pub is not None
            and raw_world_from_camera is not None
        ):
            self._publish_sparse_point_cloud_message(
                stamp,
                image_name,
                images[0],
                self._last_base_translation,
                self._last_base_quaternion,
                world_from_camera=output_pose @ self._T_base_camera,
            )

    @staticmethod
    def _fused_relative_covariance(
        previous_pose: np.ndarray,
        previous_covariance: np.ndarray,
        current_covariance: np.ndarray,
    ) -> np.ndarray:
        """Fallback relative covariance for two ROS absolute marginals.

        Absolute odometry covariance is serialized as [translation,
        rotation], while the adjoint acts on the canonical [rotation,
        translation] tangent.  Keep this fallback conservative and correctly
        ordered; the normal path uses the iEKF clone cross-covariance instead.
        """
        rotation = previous_pose[:3, :3].T
        basis = np.zeros((6, 6), dtype=np.float64)
        basis[:3, :3] = rotation
        basis[3:, 3:] = rotation
        ros_order = np.array([3, 4, 5, 0, 1, 2])
        previous_rt = np.asarray(previous_covariance, dtype=np.float64)
        current_rt = np.asarray(current_covariance, dtype=np.float64)
        previous_rt = previous_rt[np.ix_(ros_order, ros_order)]
        current_rt = current_rt[np.ix_(ros_order, ros_order)]
        covariance_rt = basis @ (previous_rt + current_rt) @ basis.T
        covariance_ros = covariance_rt[np.ix_(ros_order, ros_order)]
        return 0.5 * (covariance_ros + covariance_ros.T)

    @staticmethod
    def _feature_stamp(image_name: str) -> tuple[int, int]:
        stem = Path(image_name).stem
        seconds, nanoseconds = stem.split("_", 1)
        return int(seconds), int(nanoseconds)

    def _publish_feature_track_updates(
        self,
        fallback_stamp,
        image_msgs: list[Image | CompressedImage] | None = None,
    ) -> None:
        if self._feature_tracks_pub is None:
            return
        updates = list(self._vo.last_feature_track_updates)
        if not updates:
            return

        array_message = FeatureTrackArray()
        array_message.header.stamp = fallback_stamp
        array_message.header.frame_id = self._camera_frame_ids[0]
        array_message.feature_type = self._feature_type
        # For compressed stereo inputs this is a zero-recompression copy of
        # the exact synchronized pair used by COLMAP. Stelvio can therefore
        # render native tracks without joining two independently delivered
        # decompressed streams whose samples may be dropped out of lockstep.
        if (
            image_msgs is not None
            and len(image_msgs) >= 2
            and isinstance(image_msgs[0], CompressedImage)
            and isinstance(image_msgs[1], CompressedImage)
        ):
            array_message.has_stereo_images = True
            array_message.left_image = image_msgs[0]
            array_message.right_image = image_msgs[1]
        for update in updates:
            track_message = FeatureTrack()
            track_message.track_id = int(update["track_id"])
            track_message.camera_name = str(update["camera_name"])
            track_message.is_complete = bool(update["is_complete"])
            for observation in update["observations"]:
                try:
                    seconds, nanoseconds = self._feature_stamp(
                        observation["image_name"])
                except (KeyError, TypeError, ValueError):
                    continue
                observation_message = FeatureObservation()
                observation_message.stamp.sec = seconds
                observation_message.stamp.nanosec = nanoseconds
                observation_message.frame_index = int(observation["frame_index"])
                observation_message.keypoint_index = int(
                    observation["keypoint_index"])
                observation_message.x = float(observation["x"])
                observation_message.y = float(observation["y"])
                observation_message.has_stereo_point = bool(
                    observation.get("has_stereo_point", False))
                observation_message.stereo_x = float(
                    observation.get("stereo_x", 0.0))
                observation_message.stereo_y = float(
                    observation.get("stereo_y", 0.0))
                stereo_point = observation.get(
                    "stereo_point_camera", [0.0, 0.0, 0.0])
                observation_message.stereo_point_camera = [
                    float(value) for value in stereo_point]
                observation_message.triangulation_angle_rad = float(
                    observation.get("triangulation_angle_rad", 0.0))
                track_message.observations.append(observation_message)
            # A completion marker can legitimately contain no observations
            # when its final observation was emitted in the preceding epoch.
            if track_message.observations or track_message.is_complete:
                array_message.tracks.append(track_message)
        if array_message.tracks:
            self._feature_tracks_pub.publish(array_message)

    def _camera_pose_to_base_pose(
        self,
        translation: np.ndarray,
        quaternion: Quaternion,
    ) -> tuple[np.ndarray, Quaternion]:
        # The core supplies T_initial_camera_current_camera in COLMAP optical
        # axes.  Choosing the initial base pose as the ROS world origin changes
        # both the parent and child basis, hence the conjugation (not a
        # one-sided axis swap).
        camera_pose = _pose_to_matrix(translation, quaternion)
        base_pose = self._T_base_camera @ camera_pose @ self._T_camera_base
        return _matrix_to_pose(base_pose)

    def _camera_covariance_to_base_ros(
        self,
        covariance: Any,
        relative_transform: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Convert inverse-PnP covariance into the published ROS body frame.

        PyCOLMAP reports covariance in its left perturbation convention.  The
        graph and the iEKF use the right-local perturbation of the relative
        pose, so the second adjoint transport below is required after the
        camera-to-base conjugation.
        """
        if covariance is None:
            return None
        covariance_camera_rt = np.asarray(covariance, dtype=np.float64)
        if covariance_camera_rt.shape != (6, 6):
            self.get_logger().warning(
                "Ignoring relative pose covariance with shape "
                f"{covariance_camera_rt.shape}; expected (6, 6)"
            )
            return None
        if not np.all(np.isfinite(covariance_camera_rt)):
            self.get_logger().warning("Ignoring non-finite relative pose covariance")
            return None

        # T_base_camera changes the tangent-vector basis for the conjugated
        # pose T_base_camera * T_refCam_curCam * T_camera_base.
        adjoint = _adjoint_rotation_translation(self._T_base_camera)
        covariance_base_rt = (
            adjoint @ covariance_camera_rt @ adjoint.T)
        if relative_transform is not None:
            relative_transform = np.asarray(
                relative_transform, dtype=np.float64).reshape(4, 4)
            # delta_right = Ad_(Z^-1) delta_left.
            local_adjoint = _adjoint_rotation_translation(
                np.linalg.inv(relative_transform))
            covariance_base_rt = (
                local_adjoint @ covariance_base_rt @ local_adjoint.T)
        # geometry_msgs covariance order is [x, y, z, rot_x, rot_y, rot_z].
        ros_order = np.array([3, 4, 5, 0, 1, 2])
        covariance_ros = covariance_base_rt[np.ix_(ros_order, ros_order)]
        return 0.5 * (covariance_ros + covariance_ros.T)

    def _publish_static_camera_transforms(self, rig_config: dict[str, Any]) -> None:
        if self._static_tf_broadcaster is None:
            return

        stamp = self.get_clock().now().to_msg()
        transforms = []
        for camera_name, camera_frame_id in zip(self._camera_names, self._camera_frame_ids):
            # The rig YAML directly stores T_rig_camera and the GrandTour
            # (and all shipped) rig frames are the configured body/rig frame.
            # _rig_transform_to_matrix returns its COLMAP inverse,
            # T_camera_rig, so its inverse is the ROS body->camera TF.  Do not
            # compose T_camera_base again: doing that effectively applied the
            # reference-camera transform twice.
            T_camera_rig = _rig_transform_to_matrix(rig_config, camera_name)
            T_base_camera = np.linalg.inv(T_camera_rig)
            transforms.append(
                self._make_transform(
                    stamp,
                    self._child_frame_id,
                    camera_frame_id,
                    T_base_camera,
                )
            )

        self._static_tf_broadcaster.sendTransform(transforms)

    def _write_input_images(self, image_name: str, images: list[np.ndarray]) -> None:
        for camera_name, image in zip(self._camera_names, images):
            output_path = self._images_dir / camera_name / image_name
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Failed to write input image {output_path}")

    def _open_tum_trajectory_log(self) -> None:
        try:
            self._trajectory_tum_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if bool(self.get_parameter("trajectory_tum_overwrite").value) else "a"
            self._trajectory_tum_file = self._trajectory_tum_path.open(
                mode,
                encoding="utf-8",
                buffering=1,
            )
            self.get_logger().info(
                f"Writing TUM trajectory to {self._trajectory_tum_path}"
            )
        except OSError as exc:
            self._trajectory_tum_file = None
            self._save_trajectory_tum = False
            self.get_logger().error(
                f"Could not open TUM trajectory file {self._trajectory_tum_path}: {exc}. "
                "Continuing without trajectory file logging."
            )

    def _write_tum_pose(
        self,
        stamp,
        translation: np.ndarray,
        quaternion: Quaternion,
    ) -> None:
        if self._trajectory_tum_file is None:
            return
        qx, qy, qz, qw = _quaternion_to_msg(quaternion)
        timestamp = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        self._trajectory_tum_file.write(
            f"{timestamp:.9f} "
            f"{float(translation[0]):.9f} "
            f"{float(translation[1]):.9f} "
            f"{float(translation[2]):.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
        )

    def _publish_feature_match_image(
        self,
        stamp,
        image_name: str,
        images: list[np.ndarray],
    ) -> None:
        debug_items = getattr(self._vo, "last_feature_match_debug", [])
        panels = []
        image_by_camera = {
            camera_name: image
            for camera_name, image in zip(self._camera_names, images)
        }

        for item in debug_items:
            left_key, right_key = item.get("pair", (None, None))
            if left_key is None or right_key is None:
                continue
            left_camera, left_image = self._split_debug_image_key(left_key)
            right_camera, right_image = self._split_debug_image_key(right_key)
            if left_image != image_name or right_image != image_name:
                continue
            if left_camera == right_camera:
                continue
            if left_camera not in image_by_camera or right_camera not in image_by_camera:
                continue

            panels.append(
                self._draw_feature_match_panel(
                    image_by_camera[left_camera],
                    image_by_camera[right_camera],
                    np.asarray(item["keypoints1"], dtype=np.float32),
                    np.asarray(item["keypoints2"], dtype=np.float32),
                    np.asarray(item["matches"], dtype=np.int64),
                    f"{left_camera} <-> {right_camera}",
                )
            )

        if not panels:
            return

        output = panels[0] if len(panels) == 1 else self._stack_panels_vertical(panels)
        msg = self._bridge.cv2_to_imgmsg(output, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self._camera_frame_ids[0] if self._camera_frame_ids else ""
        self._feature_matches_pub.publish(msg)

    @staticmethod
    def _split_debug_image_key(key: str) -> tuple[str, str]:
        camera, _, image = key.partition("/")
        return camera, image

    def _draw_feature_match_panel(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_keypoints: np.ndarray,
        right_keypoints: np.ndarray,
        matches: np.ndarray,
        label: str,
    ) -> np.ndarray:
        left = self._as_bgr(left_image)
        right = self._as_bgr(right_image)
        if left.shape[0] != right.shape[0]:
            scale = left.shape[0] / max(right.shape[0], 1)
            right = cv2.resize(
                right,
                (max(1, int(round(right.shape[1] * scale))), left.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            right_keypoints = right_keypoints.copy()
            right_keypoints *= scale

        panel = np.hstack([left.copy(), right.copy()])
        right_x_offset = left.shape[1]

        left_keypoints = np.asarray(left_keypoints, dtype=np.float32).reshape(-1, 2)
        right_keypoints = np.asarray(right_keypoints, dtype=np.float32).reshape(-1, 2)
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)

        self._draw_keypoints(panel, left_keypoints, (80, 220, 255), 0)
        self._draw_keypoints(panel, right_keypoints, (255, 180, 80), right_x_offset)
        self._draw_matches(panel, left_keypoints, right_keypoints, matches, right_x_offset)

        cv2.putText(
            panel,
            f"{label}: {len(left_keypoints)} / {len(right_keypoints)} features, {len(matches)} matches",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{label}: {len(left_keypoints)} / {len(right_keypoints)} features, {len(matches)} matches",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    @staticmethod
    def _stack_panels_vertical(panels: list[np.ndarray]) -> np.ndarray:
        max_width = max(panel.shape[1] for panel in panels)
        padded = []
        for panel in panels:
            if panel.shape[1] == max_width:
                padded.append(panel)
                continue
            pad_width = max_width - panel.shape[1]
            padded.append(
                cv2.copyMakeBorder(
                    panel,
                    0,
                    0,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            )
        return np.vstack(padded)

    @staticmethod
    def _as_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        color: tuple[int, int, int],
        x_offset: int,
    ) -> None:
        keypoints = self._subsample_rows(keypoints, self._feature_match_max_features)
        height, width = image.shape[:2]
        for x, y in keypoints:
            px = int(round(float(x))) + x_offset
            py = int(round(float(y)))
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(image, (px, py), 2, color, -1, lineType=cv2.LINE_AA)

    def _draw_matches(
        self,
        image: np.ndarray,
        left_keypoints: np.ndarray,
        right_keypoints: np.ndarray,
        matches: np.ndarray,
        right_x_offset: int,
    ) -> None:
        matches = self._subsample_rows(matches, self._feature_match_max_matches)
        height, width = image.shape[:2]
        for match_index, (left_idx, right_idx) in enumerate(matches):
            if left_idx < 0 or right_idx < 0:
                continue
            if left_idx >= len(left_keypoints) or right_idx >= len(right_keypoints):
                continue
            left_pt = left_keypoints[left_idx]
            right_pt = right_keypoints[right_idx]
            p1 = (int(round(float(left_pt[0]))), int(round(float(left_pt[1]))))
            p2 = (
                int(round(float(right_pt[0]))) + right_x_offset,
                int(round(float(right_pt[1]))),
            )
            if not (0 <= p1[0] < width and 0 <= p1[1] < height):
                continue
            if not (0 <= p2[0] < width and 0 <= p2[1] < height):
                continue
            color = (
                int(80 + (37 * match_index) % 176),
                int(80 + (71 * match_index) % 176),
                int(80 + (113 * match_index) % 176),
            )
            cv2.line(image, p1, p2, color, 1, lineType=cv2.LINE_AA)
            cv2.circle(image, p1, 3, color, 1, lineType=cv2.LINE_AA)
            cv2.circle(image, p2, 3, color, 1, lineType=cv2.LINE_AA)

    @staticmethod
    def _subsample_rows(values: np.ndarray, max_rows: int) -> np.ndarray:
        values = np.asarray(values)
        if max_rows <= 0 or len(values) <= max_rows:
            return values
        indices = np.linspace(0, len(values) - 1, max_rows, dtype=np.int64)
        return values[indices]

    @staticmethod
    def _select_pose_change(pose_changes):
        if not pose_changes:
            return None
        for pose_change in reversed(pose_changes):
            if len(pose_change) >= 6 and pose_change[1] is not None:
                return pose_change
        return pose_changes[-1]

    @staticmethod
    def _compact_log(log: dict[str, Any]) -> dict[str, Any]:
        current = log.get("current_frame", {})
        return {
            "frame_count": log.get("frame_count"),
            "keyframe_count": log.get("keyframe_count"),
            "reinitializations": log.get("reinitializations"),
            "status": current.get("status"),
            "frame_name": current.get("frame_name"),
            "is_keyframe": current.get("is_keyframe"),
            "num_features": current.get("num_features"),
            "inlier_matches": current.get("inlier_matches"),
            "inlier_ratio": current.get("inlier_ratio"),
            "processing_time": current.get("processing_time"),
            "feature_extraction_time": current.get("feature_extraction_time"),
            "matching_time": current.get("matching_time"),
            "reconstruction_time": current.get("reconstruction_time"),
            "stereo_matches_raw": current.get("stereo_matches_raw"),
            "stereo_matches_epipolar": current.get("stereo_matches_epipolar"),
            "stereo_points_candidates": current.get("stereo_points_candidates"),
            "stereo_points_triangulated": current.get("stereo_points_triangulated"),
            "stereo_points_valid": current.get("stereo_points_valid"),
            "stereo_rejected_indices": current.get("stereo_rejected_indices"),
            "stereo_rejected_triangulation": current.get(
                "stereo_rejected_triangulation"),
            "stereo_rejected_cheirality": current.get(
                "stereo_rejected_cheirality"),
            "stereo_rejected_depth": current.get("stereo_rejected_depth"),
            "stereo_rejected_reprojection": current.get(
                "stereo_rejected_reprojection"),
            "stereo_rejected_angle": current.get("stereo_rejected_angle"),
            "pnp_pose_source": current.get("pnp_pose_source"),
            "pnp_ransac_inliers": current.get("pnp_ransac_inliers"),
            "pnp_prior_inliers": current.get("pnp_prior_inliers"),
            "pnp_ransac_median_error_px": current.get(
                "pnp_ransac_median_error_px"),
            "pnp_prior_median_error_px": current.get(
                "pnp_prior_median_error_px"),
            "pnp_covariance_reconstructed": current.get(
                "pnp_covariance_reconstructed", False),
            "pnp_covariance_source": current.get(
                "pnp_covariance_source", "none"),
            "pnp_covariance_points": current.get(
                "pnp_covariance_points", 0),
            "pnp_covariance_rank": current.get(
                "pnp_covariance_rank", 0),
            "pnp_covariance_condition_number": current.get(
                "pnp_covariance_condition_number"),
            "pnp_covariance_reprojection_rms_px": current.get(
                "pnp_covariance_reprojection_rms_px"),
            "pnp_success": current.get("pnp_success", False),
            "pnp_failure_reason": current.get("pnp_failure_reason", ""),
            "pnp_correspondences": current.get("pnp_correspondences", 0),
            "pnp_temporal_match_source": current.get(
                "pnp_temporal_match_source", "none"),
            "imu_initialized": current.get("imu_initialized"),
            "imu_pose_initialized": current.get("imu_pose_initialized"),
            "imu_prediction_aligned": current.get("imu_prediction_aligned"),
            "inertial_rotation_factor_used": current.get(
                "inertial_rotation_factor_used", False),
            "inertial_gravity_factor_used": current.get(
                "inertial_gravity_factor_used", False),
            "imu_buffered_samples": current.get("imu_buffered_samples"),
            "imu_buffer_span_sec": current.get("imu_buffer_span_sec"),
            "imu_dropped_buffered_samples": current.get(
                "imu_dropped_buffered_samples"),
            "imu_initialization_stationary": current.get(
                "imu_initialization_stationary"),
            "imu_accepted_visual_updates": current.get(
                "imu_accepted_visual_updates"),
            "imu_rejected_visual_updates": current.get(
                "imu_rejected_visual_updates"),
        }

    def _publish_pnp_omission(
        self,
        stamp,
        image_name: str,
        current_log: dict[str, Any],
    ) -> None:
        """Report a genuine missing visual update; never fabricate a pose."""
        reason = str(current_log.get("pnp_failure_reason", "") or "")
        if not reason:
            reason = "missing_visual_pnp_result"
        event = {
            "stamp_ns": self._stamp_to_nanoseconds(stamp),
            "image": image_name,
            "reason": reason,
            "temporal_matches": current_log.get("inlier_matches", 0),
            "stereo_points_valid": current_log.get("stereo_points_valid", 0),
            "pnp_correspondences": current_log.get("pnp_correspondences", 0),
            "pnp_temporal_match_source": current_log.get(
                "pnp_temporal_match_source", "none"),
        }
        message = String()
        message.data = json.dumps(event, default=str)
        self._pnp_omission_pub.publish(message)
        if reason == "bootstrap_no_previous_stereo_frame":
            self.get_logger().info(
                f"Visual PnP bootstrap at {image_name}: first image defines "
                "the origin",
            )
        else:
            self.get_logger().warning(
                f"Visual PnP omitted for {image_name}: {reason} "
                f"(temporal={event['temporal_matches']} "
                f"stereo={event['stereo_points_valid']} "
                f"pnp_2d3d={event['pnp_correspondences']} "
                f"source={event['pnp_temporal_match_source']})",
                throttle_duration_sec=1.0,
            )

    def _publish_absolute(
        self,
        stamp,
        translation: np.ndarray,
        quaternion: Quaternion,
        covariance: np.ndarray | None = None,
    ) -> None:
        pose_msg = self._make_pose(stamp, self._world_frame_id, translation, quaternion)
        self._pose_pub.publish(pose_msg)

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = self._child_frame_id
        odom_msg.pose.pose = pose_msg.pose
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64).reshape(6, 6)
            odom_msg.pose.covariance = covariance.reshape(-1).tolist()
        odom_msg.twist.covariance[0] = -1.0
        self._odom_pub.publish(odom_msg)

        self._path_msg.header.stamp = stamp
        self._path_msg.poses.append(pose_msg)
        now = time.monotonic()
        if (
            self._path_publish_period_sec == 0.0
            or now - self._last_path_publish_monotonic
            >= self._path_publish_period_sec
        ):
            self._path_pub.publish(self._path_msg)
            self._last_path_publish_monotonic = now
        self._write_tum_pose(stamp, translation, quaternion)

        if self._tf_broadcaster is not None:
            self._tf_broadcaster.sendTransform(
                self._make_transform(
                    stamp,
                    self._world_frame_id,
                    self._child_frame_id,
                    _pose_to_matrix(translation, quaternion),
                )
            )

    def _publish_sparse_point_cloud_message(
        self,
        stamp,
        image_name: str,
        left_image: np.ndarray,
        camera_translation: np.ndarray,
        camera_quaternion: Quaternion,
        world_from_camera: np.ndarray | None = None,
    ) -> None:
        observations = self._vo.stereo_points_for_frame(image_name)
        if not observations:
            return

        # The core pose maps current cam0 coordinates into the initial cam0
        # frame. T_base_camera changes that initial optical basis to the ROS
        # world basis, whose origin is the initial base pose.
        if world_from_camera is None:
            world_from_camera = self._T_base_camera @ _pose_to_matrix(
                camera_translation, camera_quaternion)
        height, width = left_image.shape[:2]
        points = []
        for observation in observations:
            point_camera = np.append(observation["point_camera"], 1.0)
            point_world = (world_from_camera @ point_camera)[:3]
            if not np.all(np.isfinite(point_world)):
                continue

            pixel = np.rint(observation["image_point"][:2]).astype(np.int64)
            u = int(np.clip(pixel[0], 0, width - 1))
            v = int(np.clip(pixel[1], 0, height - 1))
            color = left_image[v, u]
            if np.ndim(color) == 0:
                red = green = blue = int(color)
            elif self._image_encoding.lower().startswith("rgb"):
                red, green, blue = (int(channel) for channel in color[:3])
            else:
                blue, green, red = (int(channel) for channel in color[:3])
            rgb = (red << 16) | (green << 8) | blue
            points.append((
                float(point_world[0]),
                float(point_world[1]),
                float(point_world[2]),
                rgb,
            ))

        if not points:
            return
        header = Header()
        header.stamp = stamp
        header.frame_id = self._world_frame_id
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        self._sparse_point_cloud_pub.publish(
            point_cloud2.create_cloud(header, fields, points))

    def _publish_relative(
        self,
        stamp,
        translation: np.ndarray,
        quaternion: Quaternion,
        covariance: np.ndarray | None,
        *,
        from_timestamp_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        relative_msg = self._make_pose(stamp, self._base_frame_id, translation, quaternion)
        if covariance is None:
            # The legacy reconstruction route may not expose a Ceres/PnP
            # covariance.  Keep the pose sample (and its explicit endpoint)
            # usable with a deliberately weak model covariance instead of
            # aborting the callback or silently dropping a frame.  Minimal
            # stereo/PnP normally never reaches this branch because it either
            # receives the stereo-Schur covariance or reconstructs one from
            # the final reprojection normal matrix.
            self.get_logger().warning(
                "Relative pose has no solver covariance; publishing it with "
                "a conservative fallback model covariance",
                throttle_duration_sec=5.0,
            )
            covariance = np.eye(6, dtype=np.float64) * 1.0e3
        self._relative_pose_pub.publish(relative_msg)
        covariance_msg = PoseWithCovarianceStamped()
        covariance_msg.header = relative_msg.header
        covariance_msg.pose.pose = relative_msg.pose
        covariance_msg.pose.covariance = np.asarray(
            covariance, dtype=np.float64).reshape(6, 6).reshape(-1).tolist()
        self._relative_pose_covariance_pub.publish(covariance_msg)

        # Publish the same pose/covariance atomically with both endpoints. The
        # legacy message above remains available for existing consumers, but
        # cannot express which previous image epoch this interval starts at.
        motion_msg = RelativePoseWithCovarianceStamped()
        motion_msg.header = relative_msg.header
        motion_msg.to_stamp = relative_msg.header.stamp
        if from_timestamp_ns is not None and int(from_timestamp_ns) >= 0:
            from_timestamp_ns = int(from_timestamp_ns)
            motion_msg.from_stamp.sec = from_timestamp_ns // 1_000_000_000
            motion_msg.from_stamp.nanosec = from_timestamp_ns % 1_000_000_000
        motion_msg.relative_pose.pose = relative_msg.pose
        motion_msg.relative_pose.covariance = covariance_msg.pose.covariance
        metadata = metadata or {}
        pose_source = str(metadata.get("pnp_pose_source", ""))
        if pose_source == "imu_seeded_visual":
            inlier_value = metadata.get("pnp_prior_inliers", 0)
            reprojection_value = metadata.get(
                "pnp_prior_median_error_px", 0.0)
        else:
            inlier_value = metadata.get("pnp_ransac_inliers", 0)
            reprojection_value = metadata.get(
                "pnp_ransac_median_error_px", 0.0)
        try:
            motion_msg.inlier_count = max(0, int(inlier_value or 0))
        except (TypeError, ValueError, OverflowError):
            motion_msg.inlier_count = 0
        try:
            reprojection_value = float(reprojection_value)
            motion_msg.reprojection_rms_px = (
                reprojection_value if np.isfinite(reprojection_value) else 0.0)
        except (TypeError, ValueError, OverflowError):
            motion_msg.reprojection_rms_px = 0.0
        try:
            condition_value = float(metadata.get(
                "pnp_covariance_condition_number", 0.0) or 0.0)
            motion_msg.information_condition_number = (
                condition_value if np.isfinite(condition_value) else 0.0)
        except (TypeError, ValueError, OverflowError):
            motion_msg.information_condition_number = 0.0
        covariance_source = metadata.get("pnp_covariance_source")
        if not covariance_source or str(covariance_source).lower() == "none":
            covariance_source = (
                "reconstructed_reprojection"
                if metadata.get("pnp_covariance_reconstructed", False)
                else "solver_or_model_fallback")
        motion_msg.covariance_source = str(covariance_source)
        self._relative_motion_pub.publish(motion_msg)

    @staticmethod
    def _make_pose(stamp, frame_id: str, translation: np.ndarray, quaternion: Quaternion) -> PoseStamped:
        qx, qy, qz, qw = _quaternion_to_msg(quaternion)
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(translation[0])
        msg.pose.position.y = float(translation[1])
        msg.pose.position.z = float(translation[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    @staticmethod
    def _make_transform(
        stamp,
        parent_frame_id: str,
        child_frame_id: str,
        transform: np.ndarray,
    ) -> TransformStamped:
        translation, quaternion = _matrix_to_pose(transform)
        qx, qy, qz, qw = _quaternion_to_msg(quaternion)
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent_frame_id
        msg.child_frame_id = child_frame_id
        msg.transform.translation.x = float(translation[0])
        msg.transform.translation.y = float(translation[1])
        msg.transform.translation.z = float(translation[2])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        return msg


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ColmapOdometryNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
