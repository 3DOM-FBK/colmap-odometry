"""IMU handling around the visual odometry, independent of ROS.

Per camera frame:

1. :meth:`InertialPipeline.prior` propagates the filter to the frame and
   returns a :class:`MotionPrior` in the rig frame (PnP seed, gyro rotation
   factor, gravity-direction factor);
2. the odometry runs;
3. :meth:`InertialPipeline.update` initializes or re-anchors the filter at
   the accepted visual pose (optionally correcting velocity and biases).

Poses passed to :meth:`update` are in the published body (base) frame;
priors are converted to the rig frame with ``T_rig_base``.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..estimation.inertial import InertialFactors, MotionPrior
from ..geometry.lie import invert
from .bias_window import GyroBiasWindow, GyroInterval, preintegrate
from .config import ImuConfig
from .filter import ImuFilter, VisualCorrectionResult


@dataclass
class ImuFrameReport:
    initialized: bool = False
    pose_initialized: bool = False
    prior_available: bool = False
    rotation_factor: bool = False
    gravity_factor: bool = False
    visual_update_accepted: bool | None = None
    visual_update_nis: float | None = None
    gyro_bias: list | None = None
    accel_bias: list | None = None
    velocity: list | None = None
    buffered_samples: int = 0


class InertialPipeline:
    def __init__(self, config: ImuConfig, rig_from_base: np.ndarray):
        self.config = config
        self.filter = ImuFilter(config)
        self._rig_from_base = np.asarray(rig_from_base, dtype=np.float64).reshape(4, 4)
        self._base_from_rig = invert(self._rig_from_base)
        self._clone_stamp_ns: int | None = None
        # Prepared (base-frame, lever-arm compensated) samples for gravity factors.
        self._factor_samples: deque = deque(maxlen=max(10, int(config.queue_depth)))
        self._bias_window = None
        if config.gyro_bias_window_sec > 0.0:
            self._bias_window = GyroBiasWindow(
                config.gyro_bias_window_sec, config.initial_gyro_bias_std,
                config.gyro_bias_window_huber, config.gyro_bias_window_minimum_sec)
        self._interval: GyroInterval | None = None
        self._condition = threading.Condition()

    # ------------------------------------------------------------------ input
    def add_sample(self, stamp_ns: int, acceleration: np.ndarray,
                   angular_velocity: np.ndarray) -> bool:
        stamp_ns = int(stamp_ns + round(self.config.time_offset_sec * 1e9))
        if not self.filter.add_sample(stamp_ns, acceleration, angular_velocity):
            return False
        sample = self.filter.latest_sample
        with self._condition:
            if sample is not None:
                self._factor_samples.append(
                    (stamp_ns, sample.acceleration.copy(), sample.angular_velocity.copy()))
            self._condition.notify_all()
        return True

    def wait_for(self, stamp_ns: int, timeout_sec: float | None = None) -> bool:
        """Block until a sample at or after ``stamp_ns`` arrived (or timeout)."""
        timeout = self.config.wait_timeout_sec if timeout_sec is None else timeout_sec
        with self._condition:
            return self._condition.wait_for(
                lambda: (self.filter.latest_sample_timestamp_ns or -1) >= stamp_ns,
                timeout=max(0.0, timeout))

    # ------------------------------------------------------------------ prior
    def prior(self, stamp_ns: int, previous_stamp_ns: int | None,
              report: ImuFrameReport | None = None) -> MotionPrior | None:
        report = report if report is not None else ImuFrameReport()
        report.initialized = self.filter.imu_initialized
        report.pose_initialized = self.filter.pose_initialized
        # The clone and the visual interval must start at the same frame.
        if previous_stamp_ns is None or self._clone_stamp_ns != int(previous_stamp_ns):
            return None
        if not self.filter.pose_initialized or not self.filter.predict_to(stamp_ns):
            return None
        prediction = self.filter.relative_pose_prediction()
        if prediction is None:
            return None
        previous_base_from_current, covariance_ros = prediction
        previous_rig_from_current = (self._rig_from_base @ previous_base_from_current
                                     @ self._base_from_rig)
        seed = invert(previous_rig_from_current)
        # The clone covariance is right-local on the base motion.  Conjugating
        # by T_rig_base gives the right-local covariance of the rig motion,
        # which equals the left-local covariance of its inverse (the solver
        # variable).  Only the rotation block is used:
        #   Sigma_R = R_rig_base Sigma_rr R_rig_base^T.
        covariance_rt_rotation = np.asarray(covariance_ros, dtype=np.float64)[3:, 3:]
        rotation = self._rig_from_base[:3, :3]
        rotation_prior = seed[:3, :3]
        window = self._window_rotation(stamp_ns, previous_stamp_ns)
        if window is not None:
            # previous_base_from_current -> rig current_from_previous.
            rotation_prior = (rotation @ window[0] @ rotation.T).T
            covariance_rt_rotation = window[1]
        rotation_covariance = rotation @ covariance_rt_rotation @ rotation.T
        rotation_covariance = 0.5 * (rotation_covariance + rotation_covariance.T)
        values, vectors = np.linalg.eigh(rotation_covariance)
        floor = self.config.rotation_factor_std_floor ** 2
        rotation_covariance = vectors @ np.diag(np.maximum(values, floor)) @ vectors.T

        gravity_previous = gravity_current = gravity_sigma = None
        if self.config.gravity_factor_enabled:
            previous_up = self.filter.clone_gravity_up_direction()
            measurement = self._gravity_measurement(stamp_ns, previous_stamp_ns)
            if previous_up is not None and measurement is not None:
                gravity_previous = rotation @ previous_up
                gravity_current = rotation @ measurement[0]
                gravity_sigma = measurement[1]
        factors = InertialFactors(
            rotation_prior=rotation_prior,
            rotation_covariance=rotation_covariance,
            gravity_previous=gravity_previous,
            gravity_current=gravity_current,
            gravity_sigma_rad=gravity_sigma,
        )
        report.prior_available = True
        report.rotation_factor = factors.uses_rotation
        report.gravity_factor = factors.uses_gravity
        return MotionPrior(
            seed=seed,
            seed_max_error_px=self.config.prior_maximum_reprojection_error_px,
            seed_min_inliers=self.config.prior_minimum_inliers,
            factors=factors,
        )

    def _window_rotation(self, stamp_ns: int, previous_stamp_ns: int):
        """Gyro rotation over the interval with the window bias, and its covariance."""
        self._interval = None
        if self._bias_window is None:
            return None
        start = int(previous_stamp_ns) - int(self.config.maximum_gap_sec * 1e9)
        with self._condition:
            samples = [item for item in self._factor_samples if start <= item[0]]
        if not samples:
            return None
        stamps = np.array([item[0] for item in samples], dtype=np.int64)
        rates = np.stack([item[2] for item in samples])
        # Hold the last rate up to the frame (samples may end just before it).
        if stamps[-1] < stamp_ns <= stamps[-1] + int(self.config.maximum_gap_sec * 1e9):
            stamps = np.append(stamps, stamp_ns)
            rates = np.vstack((rates, rates[-1]))
        self._interval = preintegrate(stamps, rates, int(previous_stamp_ns), int(stamp_ns))
        window = self._bias_window
        if self._interval is None or window.bias is None:
            return None
        jacobian = self._interval.bias_jacobian
        covariance = (self.config.gyro_noise_variance * self._interval.duration_sec * np.eye(3)
                      + jacobian @ window.covariance @ jacobian.T)
        bias = window.bias
        if self.config.gyro_bias_window_axes == "vertical":
            up = self.filter.clone_gravity_up_direction()
            if up is None:
                return None
            filter_bias = np.array(self.filter.gyro_bias, dtype=np.float64)
            bias = filter_bias + up * float(up @ (window.bias - filter_bias))
        return self._interval.rotation_with_bias(bias), covariance

    def _gravity_measurement(self, stamp_ns: int, previous_stamp_ns: int):
        """Motion-gated specific-force direction over the end of the interval."""
        start = max(int(previous_stamp_ns),
                    int(stamp_ns) - int(self.config.gravity_factor_window_sec * 1e9))
        with self._condition:
            samples = [item for item in self._factor_samples if start < item[0] <= stamp_ns]
        if len(samples) < self.config.gravity_factor_minimum_samples:
            return None
        accelerations = np.stack([item[1] for item in samples])
        angular_velocities = np.stack([item[2] for item in samples])
        mean = np.mean(accelerations, axis=0)
        magnitude = float(np.linalg.norm(mean))
        if (not np.isfinite(magnitude) or magnitude <= 1.0e-6
                or abs(magnitude - self.config.gravity_magnitude)
                > self.config.gravity_factor_acceleration_tolerance_mps2
                or float(np.max(np.linalg.norm(angular_velocities, axis=1)))
                > self.config.gravity_factor_max_gyro_norm):
            return None
        directions = accelerations / np.maximum(
            np.linalg.norm(accelerations, axis=1)[:, None], 1.0e-9)
        direction = mean / magnitude
        errors = np.arccos(np.clip(directions @ direction, -1.0, 1.0))
        # Dispersion weakens the factor; it never beats the configured floor.
        sigma = max(self.config.gravity_factor_std,
                    float(np.std(errors)) / math.sqrt(max(1, len(errors))))
        return direction, sigma

    # ----------------------------------------------------------------- update
    def update(
        self,
        stamp_ns: int,
        world_from_base: np.ndarray | None,
        previous_base_from_current: np.ndarray | None,
        relative_covariance_ros: np.ndarray | None,
        report: ImuFrameReport | None = None,
    ) -> ImuFrameReport:
        """Feed the outcome of a frame (``world_from_base`` None on failure)."""
        report = report if report is not None else ImuFrameReport()
        if world_from_base is not None:
            if not self.filter.pose_initialized:
                if self.filter.imu_initialized and self.filter.initialize_pose(
                        stamp_ns, world_from_base):
                    self._clone_stamp_ns = int(stamp_ns)
            else:
                result: VisualCorrectionResult | None = self.filter.fuse_visual(
                    stamp_ns, world_from_base,
                    previous_base_from_current if self._clone_aligned_with_prediction(stamp_ns)
                    else None,
                    relative_covariance_ros)
                self._add_window_interval(stamp_ns, previous_base_from_current,
                                          relative_covariance_ros)
                self._clone_stamp_ns = int(stamp_ns)
                if result is not None:
                    report.visual_update_accepted = bool(result.accepted)
                    report.visual_update_nis = float(result.mahalanobis_squared)
        diagnostics = self.filter.diagnostics()
        report.initialized = self.filter.imu_initialized
        report.pose_initialized = self.filter.pose_initialized
        report.gyro_bias = diagnostics["gyro_bias"]
        report.accel_bias = diagnostics["accel_bias"]
        report.velocity = diagnostics["velocity"]
        report.buffered_samples = int(diagnostics["buffered_samples"])
        return report

    def _add_window_interval(self, stamp_ns: int, previous_base_from_current,
                             covariance_ros) -> None:
        interval, self._interval = self._interval, None
        if (self._bias_window is None or interval is None or previous_base_from_current is None
                or interval.end_ns != int(stamp_ns) or interval.start_ns != self._clone_stamp_ns):
            return
        covariance = np.zeros((3, 3))
        if covariance_ros is not None and np.all(np.isfinite(covariance_ros)):
            covariance = np.asarray(covariance_ros, dtype=np.float64).reshape(6, 6)[3:, 3:]
        # Additive floor as in the filter update, plus the gyro noise.
        covariance = 0.5 * (covariance + covariance.T) + np.eye(3) * (
            self.config.visual_rotation_std_floor ** 2
            + self.config.gyro_noise_variance * interval.duration_sec)
        self._bias_window.add(interval, np.asarray(previous_base_from_current)[:3, :3],
                              covariance)

    def _clone_aligned_with_prediction(self, stamp_ns: int) -> bool:
        return self.filter.timestamp_ns == int(stamp_ns)

    def output_pose(self, world_from_base: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Visual-inertial pose in the filter world with the filter covariance.

        The world is gravity aligned with ``align_world_with_gravity``.  None
        before initialization.
        """
        if not self.filter.pose_initialized:
            return None
        return (self.filter.aligned_visual_pose(world_from_base),
                self.filter.pose_covariance_ros())
