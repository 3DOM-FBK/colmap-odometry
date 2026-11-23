"""Typed ROS parameters of the node (everything except ``imu.*``)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from ..frames import OPTICAL_FROM_REP103


@dataclass(frozen=True)
class NodeSettings:
    # Algorithm and calibration files (relative paths: share/config).
    config_file: str = ""
    camera_file: str = ""
    rig_file: str = ""
    #: Base directory for relative output paths.
    work_dir: str = "/tmp/colmap_odometry"

    image_topics: tuple[str, ...] = ("/cam0/image_raw", "/cam1/image_raw")
    camera_names: tuple[str, ...] = ("cam0", "cam1")
    #: Topics ending in /compressed are subscribed as CompressedImage.
    image_encoding: str = "bgr8"
    image_qos_reliability: str = "best_effort"
    sync_queue_size: int = 20
    sync_slop: float = 0.05
    #: Warn when this many synchronized frames wait for processing (0: off).
    processing_queue_size: int = 0

    world_frame_id: str = "map"
    body_frame_id: str = "base_link"
    camera_frame_ids: tuple[str, ...] = ()
    #: ``p_camera = T_camera_base p_base`` for the first camera, row-major.
    T_camera_base: tuple[float, ...] = tuple(OPTICAL_FROM_REP103.reshape(-1))
    publish_tf: bool = False
    publish_camera_tf: bool = False

    publish_feature_tracks: bool = True
    publish_feature_matches: bool = False
    publish_sparse_point_cloud: bool = False
    feature_match_max_features: int = 1500
    feature_match_max_matches: int = 500
    path_publish_period_sec: float = 0.5

    save_trajectory_tum: bool = False
    trajectory_tum_path: str = ""
    trajectory_tum_overwrite: bool = True
    #: Per-frame JSON-lines report (empty: disabled).
    frame_log_path: str = "frames.jsonl"

    performance_log_every_n_frames: int = 10
    performance_slow_frame_sec: float = 1.0
    performance_stall_warn_sec: float = 3.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NodeSettings":
        known = {item.name: item for item in fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown node parameters {unknown}")
        converted = {}
        for name, value in values.items():
            default = known[name].default
            if isinstance(default, tuple):
                converted[name] = tuple(value)
            elif isinstance(default, bool):
                converted[name] = bool(value)
            elif isinstance(default, int):
                converted[name] = int(value)
            elif isinstance(default, float):
                converted[name] = float(value)
            else:
                converted[name] = str(value)
        settings = cls(**converted)
        for name in ("config_file", "camera_file", "rig_file"):
            if not getattr(settings, name):
                raise ValueError(f"parameter '{name}' is required")
        if len(settings.image_topics) != len(settings.camera_names):
            raise ValueError("image_topics and camera_names must have the same length")
        if settings.camera_frame_ids and len(settings.camera_frame_ids) != len(
                settings.camera_names):
            raise ValueError("camera_frame_ids must name one frame per camera")
        if settings.image_qos_reliability not in ("reliable", "best_effort"):
            raise ValueError("image_qos_reliability must be 'reliable' or 'best_effort'")
        return settings

    @property
    def camera_frames(self) -> tuple[str, ...]:
        return self.camera_frame_ids or self.camera_names
