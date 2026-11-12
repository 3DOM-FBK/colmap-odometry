"""IMU configuration (the ``imu.*`` ROS parameters)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ImuConfig:
    enabled: bool = False
    topic: str = "/imu"
    time_offset_sec: float = 0.0
    #: Row-major ``T_base_imu`` (``p_base = T_base_imu p_imu``).  The
    #: translation is the lever arm used for rigid-body compensation.
    T_base_imu: tuple[float, ...] = _IDENTITY
    queue_depth: int = 2000
    #: Buffered samples; 0 keeps all (lossless offline playback).
    sample_buffer_size: int = 0
    #: How long a frame waits for IMU samples up to its timestamp.
    wait_timeout_sec: float = 0.05

    initialization_duration_sec: float = 1.0
    initialization_max_wait_sec: float = 2.0
    initialization_max_gyro_norm: float = 0.03
    initialization_max_gyro_std: float = 0.015
    initialization_max_accel_std: float = 0.15
    gravity_magnitude: float = 9.80665
    align_world_with_gravity: bool = True

    # Continuous noise densities squared (not standard deviations).
    gyro_noise_variance: float = 5.0e-6
    accel_noise_variance: float = 2.5e-3
    gyro_bias_random_walk_variance: float = 1.0e-8
    accel_bias_random_walk_variance: float = 1.0e-6
    gravity_random_walk_variance: float = 0.0
    initial_position_std: float = 0.01
    initial_velocity_std: float = 0.25
    initial_orientation_std: float = 0.03
    initial_gyro_bias_std: float = 0.01
    initial_accel_bias_std: float = 0.1
    initial_gravity_std: float = 0.5
    maximum_propagation_step_sec: float = 0.01
    maximum_gap_sec: float = 0.1

    #: Correct velocity, biases and gravity with each visual relative pose
    #: (the visual-only estimate, so inertial priors are not counted twice).
    #: The published trajectory still comes from the visual-inertial solver.
    visual_update: bool = True
    #: Floors added to the visual covariance in the filter update; they
    #: should reflect the actual per-frame visual accuracy, otherwise the
    #: update is too weak to estimate the biases (NIS far below its dof).
    visual_translation_std_floor: float = 1.0e-3
    visual_rotation_std_floor: float = 3.0e-4
    visual_model_translation_std: float = 0.0
    visual_model_rotation_std: float = 0.0
    visual_nis_soft_threshold: float = 16.812
    maximum_iterations: int = 4
    convergence_epsilon: float = 1.0e-5

    # Priors handed to the visual solver.
    prior_maximum_reprojection_error_px: float = 4.0
    prior_minimum_inliers: int = 12
    #: Floor on the gyro rotation prior per frame.  A strong prior lowers
    #: translation drift but imports gyro/vision rate errors into yaw.
    rotation_factor_std_floor: float = 3.0e-4
    #: Estimate the gyro bias of the rotation prior from gyro and visual
    #: rotations of the last ``gyro_bias_window_sec`` seconds instead of the
    #: filter state (0 disables).
    gyro_bias_window_sec: float = 0.0
    gyro_bias_window_huber: float = 3.0
    gyro_bias_window_minimum_sec: float = 2.0
    #: ``all`` or ``vertical``: take only the component about gravity from
    #: the window and the tilt components (observable through gravity) from
    #: the filter.
    gyro_bias_window_axes: str = "all"
    gravity_factor_enabled: bool = True
    gravity_factor_std: float = 0.05
    gravity_factor_window_sec: float = 0.05
    gravity_factor_minimum_samples: int = 3
    gravity_factor_acceleration_tolerance_mps2: float = 1.5
    gravity_factor_max_gyro_norm: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ImuConfig":
        values = dict(values or {})
        known = {item.name: item for item in fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown imu parameters {unknown}")
        converted = {}
        for name, value in values.items():
            default = known[name].default
            if name == "T_base_imu":
                converted[name] = tuple(float(item) for item in value)
            elif isinstance(default, bool):
                converted[name] = bool(value)
            elif isinstance(default, int):
                converted[name] = int(value)
            elif isinstance(default, float):
                converted[name] = float(value)
            else:
                converted[name] = value
        config = cls(**converted)
        config.base_from_imu()  # validate

        if config.gyro_bias_window_axes not in ("all", "vertical"):
            raise ValueError("imu.gyro_bias_window_axes must be 'all' or 'vertical'")
        if config.maximum_gap_sec <= 0.0 or config.maximum_propagation_step_sec <= 0.0:
            raise ValueError("imu step and gap limits must be positive")
        return config

    def base_from_imu(self) -> np.ndarray:
        transform = np.asarray(self.T_base_imu, dtype=np.float64)
        if transform.size != 16:
            raise ValueError("imu.T_base_imu must contain 16 row-major values")
        transform = transform.reshape(4, 4)
        rotation = transform[:3, :3]
        if (not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
            raise ValueError("imu.T_base_imu must be a rigid transform")
        return transform
