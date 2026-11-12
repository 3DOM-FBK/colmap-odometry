"""Conversion of ROS rig poses to COLMAP extrinsics.

Rig YAML files use ROS TF pose semantics throughout this package: a camera
entry stores ``T_rig_camera``, i.e. the pose of the camera child frame in the
rig parent frame.  COLMAP uses the opposite direction, ``T_camera_rig``:

    p_rig = R_ros p_camera + t_ros
    p_camera = R_ros.T p_rig - R_ros.T t_ros

Keeping this conversion in one module prevents feature filtering,
triangulation, reconstruction, and published camera TFs from silently using
different transform directions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyquaternion import Quaternion


def ros_rig_from_camera_matrix(
    config: dict[str, Any], camera_name: str
) -> np.ndarray:
    """Read a ROS ``T_rig_camera`` pose from a rig YAML mapping."""
    if camera_name not in config:
        raise ValueError(f"Rig config does not define camera '{camera_name}'")
    camera_config = config[camera_name]
    if not isinstance(camera_config, dict):
        raise ValueError(f"Rig entry '{camera_name}' must be a mapping")

    rotation_xyzw = camera_config.get("rotation", [0.0, 0.0, 0.0, 1.0])
    translation = camera_config.get("translation", [0.0, 0.0, 0.0])
    if len(rotation_xyzw) != 4:
        raise ValueError(
            f"Rig rotation for camera '{camera_name}' must have 4 values")
    if len(translation) != 3:
        raise ValueError(
            f"Rig translation for camera '{camera_name}' must have 3 values")

    rotation_xyzw = np.asarray(rotation_xyzw, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if not np.all(np.isfinite(rotation_xyzw)):
        raise ValueError(f"Rig rotation for camera '{camera_name}' is not finite")
    if not np.all(np.isfinite(translation)):
        raise ValueError(
            f"Rig translation for camera '{camera_name}' is not finite")
    if np.linalg.norm(rotation_xyzw) <= np.finfo(np.float64).eps:
        raise ValueError(f"Rig rotation for camera '{camera_name}' is zero")

    x, y, z, w = rotation_xyzw
    quaternion = Quaternion(w=w, x=x, y=y, z=z).normalised
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion.rotation_matrix
    transform[:3, 3] = translation
    return transform


def colmap_camera_from_rig_matrix(
    config: dict[str, Any], camera_name: str
) -> np.ndarray:
    """Convert ROS ``T_rig_camera`` to COLMAP ``T_camera_rig``."""
    ros_pose = ros_rig_from_camera_matrix(config, camera_name)
    rotation_rig_camera = ros_pose[:3, :3]
    translation_rig_camera = ros_pose[:3, 3]

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_rig_camera.T
    transform[:3, 3] = -rotation_rig_camera.T @ translation_rig_camera
    return transform
