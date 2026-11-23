"""Conversion of rig-frame odometry into the published body (base) frame.

The odometry estimates the rig frame; outputs are expressed for a body
frame ``base`` rigidly attached to it (``T_rig_base``).  The world frame is
the base at the first frame, so absolute poses and relative motions are
conjugated: ``T_base_rig  X  T_rig_base``.
"""

from __future__ import annotations

import numpy as np

from .geometry.lie import adjoint, invert, rt_to_ros, transport

# Optical camera (x right, y down, z forward) to REP-103 body (x forward,
# y left, z up): p_camera = T p_base.
OPTICAL_FROM_REP103 = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def rigid_from_row_major(values, name: str) -> np.ndarray:
    transform = np.asarray(values, dtype=np.float64)
    if transform.size != 16:
        raise ValueError(f"{name} must contain 16 row-major values")
    transform = transform.reshape(4, 4)
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError(f"{name} must be a rigid transform")
    return transform


class BodyFrame:
    def __init__(self, rig_from_base: np.ndarray):
        self.rig_from_base = np.asarray(rig_from_base, dtype=np.float64).reshape(4, 4)
        self.base_from_rig = invert(self.rig_from_base)
        self._adjoint = adjoint(self.base_from_rig)

    @classmethod
    def from_camera(cls, rig_from_camera: np.ndarray, camera_from_base: np.ndarray):
        """From the reference camera's ``T_camera_base``."""
        return cls(rig_from_camera @ camera_from_base)

    def pose(self, world_from_rig: np.ndarray) -> np.ndarray:
        """``T_world_base`` with the world at the initial base."""
        return self.base_from_rig @ world_from_rig @ self.rig_from_base

    def motion(self, previous_rig_from_current: np.ndarray) -> np.ndarray:
        return self.base_from_rig @ previous_rig_from_current @ self.rig_from_base

    def motion_covariance_ros(self, covariance_rt: np.ndarray) -> np.ndarray:
        """Right-local rt covariance of the rig motion -> ROS-ordered base.

        For ``Z = B X B^-1`` a right perturbation of ``X`` becomes the right
        perturbation ``Ad(B) e`` of ``Z`` with ``B = T_base_rig``.
        """
        return rt_to_ros(transport(covariance_rt, self._adjoint))
