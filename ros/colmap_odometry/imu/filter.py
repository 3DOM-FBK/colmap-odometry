"""IMU error-state filter used as an interval preintegrator.

The filter propagates IMU samples between camera frames.  At each camera
frame it provides the clone-to-current relative pose prediction (used to
seed and constrain the visual solver) and is then re-anchored to the
accepted visual pose: the visual solver owns the trajectory.  With
``visual_update`` enabled, the visual relative pose is first used as an
iterated-EKF measurement so velocity, biases and gravity are estimated
(and the covariance stays bounded) instead of remaining at their
initialization values.

State (18): position, velocity, attitude (right-local), gyro bias, accel
bias, gravity vector; plus a cloned 6-DoF pose of the previous frame.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..geometry.lie import (
    align_vector_rotation,
    project_rotation,
    se3_log,
    skew,
    so3_exp,
    so3_log,
)
from .config import ImuConfig


@dataclass(frozen=True)
class ImuSample:
    timestamp_ns: int
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    # Jacobian of the base-origin specific force with respect to the
    # measured (base-frame) angular velocity.  This carries the dominant
    # gyro-noise contribution introduced by lever-arm compensation into the
    # propagation covariance.  It is None when no angular-acceleration
    # interval was available (for example, immediately after a gap).
    acceleration_gyro_jacobian: np.ndarray | None = None


@dataclass(frozen=True)
class VisualCorrectionResult:
    accepted: bool
    mahalanobis_squared: float
    iterations: int
    process_noise_inflation: float = 1.0
    raw_mahalanobis_squared: float = math.nan
    visual_trusted: bool = False
    failure_reason: str = ""


class ImuFilter:
    """18-state error-state filter with an iterated relative-pose update."""

    _STATE_SIZE = 18
    _MAX_COVARIANCE_VARIANCE = 1.0e12
    _MAX_PROCESS_NOISE_INFLATION = 1.0e12

    def __init__(self, config: ImuConfig):
        self.config = config
        base_from_imu = config.base_from_imu()
        self._rotation_base_imu = project_rotation(base_from_imu[:3, :3])
        self._translation_base_imu = base_from_imu[:3, 3].copy()
        # A zero size means lossless buffering. This matters for offline bag
        # playback: visual processing can lag behind ROS reception by many
        # seconds, while a bounded deque would silently evict precisely the
        # IMU samples needed by the queued image frames.
        sample_maxlen = (
            int(config.sample_buffer_size)
            if config.sample_buffer_size > 0 else None
        )
        self._samples: deque[ImuSample] = deque(maxlen=sample_maxlen)
        self.dropped_buffered_samples = 0
        self._initialization_samples: list[ImuSample] = []
        self._initialization_start_ns: int | None = None
        self._latest_sample_ns: int | None = None
        self._imu_initialized = False
        self._pose_initialized = False
        self._initial_rotation = np.eye(3)
        self._visual_world_alignment = np.eye(3)
        self._initial_accel_bias = np.zeros(3, dtype=np.float64)
        self._initial_mean_acceleration = np.zeros(3, dtype=np.float64)
        self._initial_mean_angular_velocity = np.zeros(3, dtype=np.float64)
        self._acceleration_scale = 1.0
        self._initialization_stationary = False
        self._lock = threading.RLock()

        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.gravity = np.array(
            [0.0, 0.0, -config.gravity_magnitude], dtype=np.float64)
        self.covariance = np.eye(self._STATE_SIZE, dtype=np.float64)
        self._initial_covariance = self.covariance.copy()
        self.timestamp_ns: int | None = None
        self._last_acceleration: np.ndarray | None = None
        self._last_angular_velocity: np.ndarray | None = None
        self._last_acceleration_gyro_jacobian: np.ndarray | None = None
        # These are measurement-domain values used to estimate angular
        # acceleration.  They are deliberately independent from the
        # propagated endpoint sample, which may be reset by visual anchoring.
        self._last_measurement_angular_velocity: np.ndarray | None = None
        self._last_measurement_timestamp_ns: int | None = None
        self._latest_prepared_sample: ImuSample | None = None
        self._last_lever_arm_diagnostics = {
            "tangential_acceleration": np.zeros(3, dtype=np.float64),
            "centripetal_acceleration": np.zeros(3, dtype=np.float64),
            "angular_acceleration": np.zeros(3, dtype=np.float64),
            "angular_acceleration_valid": False,
        }
        self._clone_position: np.ndarray | None = None
        self._clone_rotation: np.ndarray | None = None
        self._clone_covariance: np.ndarray | None = None
        self._current_clone_cross_covariance: np.ndarray | None = None
        # Process covariance accumulated after the last camera pose clone.
        # It is kept separately so innovation adaptation can acknowledge an
        # under-modelled IMU interval without weakening vision or rewriting
        # uncertainty inherited from earlier frames.
        self._interval_process_covariance = np.zeros(
            (self._STATE_SIZE, self._STATE_SIZE), dtype=np.float64)
        self.propagated_samples = 0
        self.accepted_visual_updates = 0
        self.rejected_visual_updates = 0

    @property
    def imu_initialized(self) -> bool:
        with self._lock:
            return self._imu_initialized

    @property
    def pose_initialized(self) -> bool:
        with self._lock:
            return self._pose_initialized

    @property
    def latest_sample_timestamp_ns(self) -> int | None:
        with self._lock:
            return self._latest_sample_ns

    @property
    def latest_sample(self) -> ImuSample | None:
        """Return the newest prepared, base-origin IMU sample."""
        with self._lock:
            return self._latest_prepared_sample

    def lever_arm_diagnostics(self) -> dict[str, object]:
        """Return the latest lever-arm correction without exposing state."""
        with self._lock:
            result = dict(self._last_lever_arm_diagnostics)
            for key in (
                "tangential_acceleration",
                "centripetal_acceleration",
                "angular_acceleration",
            ):
                result[key] = np.asarray(result[key], dtype=np.float64).copy()
            return result

    def add_sample(
        self,
        timestamp_ns: int,
        acceleration_imu: np.ndarray,
        angular_velocity_imu: np.ndarray,
    ) -> bool:
        raw_acceleration = self._rotation_base_imu @ np.asarray(
            acceleration_imu, dtype=np.float64).reshape(3)
        angular_velocity = self._rotation_base_imu @ np.asarray(
            angular_velocity_imu, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(raw_acceleration)) or not np.all(
            np.isfinite(angular_velocity)
        ):
            return False
        with self._lock:
            timestamp_ns = int(timestamp_ns)
            if (
                self._latest_sample_ns is not None
                and timestamp_ns <= self._latest_sample_ns
            ):
                return False

            # T_base_imu is defined by p_base = R_base_imu p_imu + r, so r
            # is the IMU origin expressed in the propagated base frame.  A
            # rigid-body accelerometer measures
            #   f_imu = f_base + alpha x r + omega x (omega x r).
            # Convert it once here, before initialization, propagation, and
            # the image-rate gravity factors can consume the sample.
            angular_acceleration = None
            derivative_dt = None
            if self._last_measurement_timestamp_ns is not None:
                derivative_dt = (
                    timestamp_ns - self._last_measurement_timestamp_ns
                ) * 1.0e-9
                if (
                    derivative_dt <= 0.0
                    or derivative_dt > self.config.maximum_gap_sec
                ):
                    derivative_dt = None
            if (
                derivative_dt is not None
                and self._last_measurement_angular_velocity is not None
            ):
                angular_acceleration = (
                    angular_velocity - self._last_measurement_angular_velocity
                ) / derivative_dt

            # The stored sample remains raw so the propagation can apply its
            # current gyro-bias estimate.  Use that same estimate for the
            # nonlinear centripetal term; a constant bias cancels from the
            # finite-difference angular acceleration.
            kinematic_angular_velocity = angular_velocity - self.gyro_bias
            centripetal = np.cross(
                kinematic_angular_velocity,
                np.cross(
                    kinematic_angular_velocity, self._translation_base_imu),
            )
            tangential = (
                np.cross(angular_acceleration, self._translation_base_imu)
                if angular_acceleration is not None else np.zeros(3)
            )
            acceleration = raw_acceleration - tangential - centripetal

            # Jacobian of the compensated specific force with respect to the
            # current measured angular velocity.  The finite-difference
            # angular acceleration contributes skew(r)/dt; the second term
            # is the exact derivative of omega x (omega x r).  The previous
            # gyro sample is correlated with this estimate; this first-order
            # model intentionally captures the current-sample contribution
            # without pretending adjacent samples are independent.
            acceleration_gyro_jacobian = None
            if derivative_dt is not None:
                r = self._translation_base_imu
                omega_dot_r = float(np.dot(kinematic_angular_velocity, r))
                centripetal_jacobian = (
                    np.outer(kinematic_angular_velocity, r)
                    + omega_dot_r * np.eye(3)
                    - 2.0 * np.outer(r, kinematic_angular_velocity)
                )
                acceleration_gyro_jacobian = (
                    skew(r) / derivative_dt - centripetal_jacobian
                )

            sample = ImuSample(
                timestamp_ns,
                acceleration,
                angular_velocity,
                acceleration_gyro_jacobian,
            )
            self._latest_sample_ns = sample.timestamp_ns
            if (
                self._samples.maxlen is not None
                and len(self._samples) == self._samples.maxlen
            ):
                self.dropped_buffered_samples += 1
            self._samples.append(sample)
            self._latest_prepared_sample = sample
            self._accumulate_initialization(sample)
            self._last_measurement_angular_velocity = angular_velocity.copy()
            self._last_measurement_timestamp_ns = timestamp_ns
            self._last_lever_arm_diagnostics = {
                "tangential_acceleration": tangential.copy(),
                "centripetal_acceleration": centripetal.copy(),
                "angular_acceleration": (
                    np.asarray(angular_acceleration, dtype=np.float64).copy()
                    if angular_acceleration is not None else np.zeros(3)
                ),
                "angular_acceleration_valid": angular_acceleration is not None,
            }
        return True

    def _accumulate_initialization(self, sample: ImuSample) -> None:
        if self._imu_initialized:
            return
        if self._initialization_start_ns is None:
            self._initialization_start_ns = sample.timestamp_ns
        self._initialization_samples.append(sample)
        elapsed = (sample.timestamp_ns - self._initialization_start_ns) * 1.0e-9
        if elapsed < self.config.initialization_duration_sec:
            return
        window_start_ns = sample.timestamp_ns - int(
            self.config.initialization_duration_sec * 1.0e9)
        window_samples = [
            item for item in self._initialization_samples
            if item.timestamp_ns >= window_start_ns
        ]
        accelerations = np.vstack(
            [item.acceleration for item in window_samples])
        angular_velocities = np.vstack(
            [item.angular_velocity for item in window_samples])
        mean_acceleration = np.mean(accelerations, axis=0)
        mean_acceleration_norm = float(np.linalg.norm(mean_acceleration))
        if mean_acceleration_norm < 1.0e-3:
            return
        mean_angular_velocity = np.mean(angular_velocities, axis=0)
        gyro_std = np.std(angular_velocities, axis=0)
        accel_std = np.std(accelerations, axis=0)
        stationary = (
            np.linalg.norm(mean_angular_velocity)
            <= self.config.initialization_max_gyro_norm
            and np.max(gyro_std) <= self.config.initialization_max_gyro_std
            and np.max(accel_std) <= self.config.initialization_max_accel_std
        )
        if (
            not stationary
            and elapsed < self.config.initialization_max_wait_sec
        ):
            return
        self._initial_mean_acceleration = mean_acceleration.copy()
        self._initial_mean_angular_velocity = mean_angular_velocity.copy()
        self._initialization_stationary = bool(stationary)
        # A moving initialization interval contains real body rotation. Never
        # reinterpret that motion as gyro bias. The calibrated IMU scale is
        # likewise retained; bias and gravity are estimated by later visual
        # innovations.
        self.gyro_bias = (
            mean_angular_velocity.copy() if stationary
            else np.zeros(3, dtype=np.float64)
        )
        self._acceleration_scale = 1.0
        self._initial_rotation = align_vector_rotation(
            mean_acceleration, np.array([0.0, 0.0, 1.0]))
        self._initial_accel_bias.fill(0.0)
        self._imu_initialized = True
        self._initialization_samples.clear()

    def initialize_pose(self, timestamp_ns: int, world_from_base: np.ndarray) -> bool:
        with self._lock:
            if not self._imu_initialized:
                return False
            pose = np.asarray(world_from_base, dtype=np.float64).reshape(4, 4)
            if self.config.align_world_with_gravity:
                # The visual world starts at the initial base. Rotate the
                # complete visual pose into the gravity-aligned world; do not
                # discard motion accumulated before IMU initialization.
                self.position = self._initial_rotation @ pose[:3, 3]
                self.rotation = project_rotation(
                    self._initial_rotation @ pose[:3, :3])
                self._visual_world_alignment = self._initial_rotation.copy()
            else:
                self.position = pose[:3, 3].copy()
                self.rotation = project_rotation(pose[:3, :3])
                self._visual_world_alignment = np.eye(3)
            self.velocity.fill(0.0)
            self.accel_bias = self._initial_accel_bias.copy()
            if self.config.align_world_with_gravity:
                self.gravity = np.array(
                    [0.0, 0.0, -self.config.gravity_magnitude],
                    dtype=np.float64,
                )
            else:
                gravity_direction = -(
                    self.rotation @ self._initial_mean_acceleration)
                gravity_norm = float(np.linalg.norm(gravity_direction))
                if gravity_norm <= 1.0e-6:
                    gravity_direction = np.array([0.0, 0.0, -1.0])
                    gravity_norm = 1.0
                self.gravity = (
                    gravity_direction / gravity_norm
                    * self.config.gravity_magnitude
                )
            standard_deviations = np.concatenate(
                (
                    np.full(3, self.config.initial_position_std),
                    np.full(3, self.config.initial_velocity_std),
                    np.full(3, self.config.initial_orientation_std),
                    np.full(3, self.config.initial_gyro_bias_std),
                    np.full(3, self.config.initial_accel_bias_std),
                    np.full(3, self.config.initial_gravity_std),
                )
            )
            self.covariance = np.diag(standard_deviations * standard_deviations)
            self._initial_covariance = self.covariance.copy()
            self.timestamp_ns = int(timestamp_ns)
            self._last_acceleration = None
            self._last_angular_velocity = None
            while self._samples and self._samples[0].timestamp_ns <= timestamp_ns:
                sample = self._samples.popleft()
                self._last_acceleration = sample.acceleration
                self._last_angular_velocity = sample.angular_velocity
                self._last_acceleration_gyro_jacobian = (
                    sample.acceleration_gyro_jacobian)
            self._pose_initialized = True
            self.commit_pose_clone()
            return True

    def fuse_visual(
        self,
        timestamp_ns: int,
        visual_world_from_base: np.ndarray,
        clone_from_current: np.ndarray | None,
        covariance_ros: np.ndarray | None,
    ) -> VisualCorrectionResult | None:
        """Use an accepted visual frame and start the next interval.

        With ``visual_update`` (and the filter propagated to exactly this
        frame) the clone-to-current visual motion is an iterated-EKF
        measurement of the whole state; the filter keeps its corrected state
        and clones it.  Only relative predictions leave the filter, so its
        absolute pose need not equal the visual one.  Overwriting position
        and attitude after the update (as anchoring does) would invalidate
        the attitude/bias correlations and corrupt the bias estimates.

        Otherwise (or when the update is rejected) the filter is anchored at
        the visual pose without correction, as an interval preintegrator.
        """
        result = None
        with self._lock:
            if (self.config.visual_update and clone_from_current is not None
                    and self.timestamp_ns == int(timestamp_ns)):
                result = self.correct_relative_pose(clone_from_current, covariance_ros)
                if result.accepted:
                    self.commit_pose_clone()
                    return result
            self.anchor_visual_pose(timestamp_ns, visual_world_from_base)
        return result

    def aligned_visual_pose(self, visual_world_from_base: np.ndarray) -> np.ndarray:
        """Express a visual pose in the filter world (gravity aligned if set)."""
        with self._lock:
            pose = np.asarray(visual_world_from_base, dtype=np.float64).reshape(4, 4).copy()
            alignment = self._visual_world_alignment
            pose[:3, :3] = alignment @ pose[:3, :3]
            pose[:3, 3] = alignment @ pose[:3, 3]
            return pose

    def anchor_visual_pose(
        self,
        timestamp_ns: int,
        visual_world_from_base: np.ndarray,
    ) -> bool:
        """Re-anchor propagation at an accepted visual/inertial solver pose.

        Position and attitude are replaced by the visual pose without a
        Kalman correction; common pose uncertainty cancels in the next cloned
        relative prediction, while IMU process/bias uncertainty continues to
        propagate normally.
        """
        with self._lock:
            if not self._pose_initialized:
                return False
            pose = np.asarray(
                visual_world_from_base, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                return False
            alignment = self._visual_world_alignment
            self.position = alignment @ pose[:3, 3]
            self.rotation = project_rotation(alignment @ pose[:3, :3])
            self.timestamp_ns = int(timestamp_ns)
            self.commit_pose_clone()
            return True

    def clone_gravity_up_direction(self) -> np.ndarray | None:
        """Return measured-specific-force direction expected at the clone."""
        with self._lock:
            if self._clone_rotation is None:
                return None
            norm = float(np.linalg.norm(self.gravity))
            if not np.isfinite(norm) or norm <= 1.0e-9:
                return None
            direction = -self._clone_rotation.T @ (self.gravity / norm)
            direction_norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction_norm) or direction_norm <= 1.0e-9:
                return None
            return direction / direction_norm

    def predict_to(self, timestamp_ns: int) -> bool:
        with self._lock:
            if not self._pose_initialized or self.timestamp_ns is None:
                return False
            timestamp_ns = int(timestamp_ns)
            if timestamp_ns < self.timestamp_ns:
                return False
            if timestamp_ns == self.timestamp_ns:
                return True

            while self._samples and self._samples[0].timestamp_ns <= timestamp_ns:
                sample = self._samples[0]
                sample_gap = (sample.timestamp_ns - self.timestamp_ns) * 1.0e-9
                if sample_gap > self.config.maximum_gap_sec:
                    # Never integrate a long interval using two distant endpoint
                    # measurements. This normally means a bounded buffer evicted
                    # data while the image worker was behind.
                    return False
                sample = self._samples.popleft()
                if sample.timestamp_ns <= self.timestamp_ns:
                    self._last_acceleration = sample.acceleration
                    self._last_angular_velocity = sample.angular_velocity
                    self._last_acceleration_gyro_jacobian = (
                        sample.acceleration_gyro_jacobian)
                    continue
                if self._last_acceleration is None:
                    self._last_acceleration = sample.acceleration
                    self._last_angular_velocity = sample.angular_velocity
                    self._last_acceleration_gyro_jacobian = (
                        sample.acceleration_gyro_jacobian)
                acceleration = 0.5 * (
                    self._last_acceleration + sample.acceleration)
                angular_velocity = 0.5 * (
                    self._last_angular_velocity + sample.angular_velocity)
                if (
                    self._last_acceleration_gyro_jacobian is not None
                    and sample.acceleration_gyro_jacobian is not None
                ):
                    acceleration_gyro_jacobian = 0.5 * (
                        self._last_acceleration_gyro_jacobian
                        + sample.acceleration_gyro_jacobian)
                elif self._last_acceleration_gyro_jacobian is not None:
                    acceleration_gyro_jacobian = (
                        self._last_acceleration_gyro_jacobian)
                else:
                    acceleration_gyro_jacobian = (
                        sample.acceleration_gyro_jacobian)
                self._propagate_interval(
                    (sample.timestamp_ns - self.timestamp_ns) * 1.0e-9,
                    acceleration,
                    angular_velocity,
                    acceleration_gyro_jacobian,
                )
                self.timestamp_ns = sample.timestamp_ns
                self._last_acceleration = sample.acceleration
                self._last_angular_velocity = sample.angular_velocity
                self._last_acceleration_gyro_jacobian = (
                    sample.acceleration_gyro_jacobian)
                self.propagated_samples += 1

            remaining = (timestamp_ns - self.timestamp_ns) * 1.0e-9
            if remaining <= 0.0:
                return True
            if (
                self._last_acceleration is None
                or remaining > self.config.maximum_gap_sec
            ):
                return False
            self._propagate_interval(
                remaining,
                self._last_acceleration,
                self._last_angular_velocity,
                self._last_acceleration_gyro_jacobian,
            )
            self.timestamp_ns = timestamp_ns
            return True

    def _propagate_interval(
        self,
        duration: float,
        measured_acceleration: np.ndarray,
        measured_angular_velocity: np.ndarray,
        acceleration_gyro_jacobian: np.ndarray | None = None,
    ) -> None:
        if duration <= 0.0:
            return
        steps = max(
            1,
            int(math.ceil(
                duration / max(self.config.maximum_propagation_step_sec, 1.0e-4)
            )),
        )
        dt = duration / steps
        for _ in range(steps):
            omega = measured_angular_velocity - self.gyro_bias
            specific_force = (
                self._acceleration_scale * measured_acceleration
                - self.accel_bias
            )
            acceleration_world = self.rotation @ specific_force + self.gravity
            self.position += self.velocity * dt + 0.5 * acceleration_world * dt * dt
            self.velocity += acceleration_world * dt

            transition_rate = np.zeros(
                (self._STATE_SIZE, self._STATE_SIZE), dtype=np.float64)
            transition_rate[0:3, 3:6] = np.eye(3)
            transition_rate[3:6, 6:9] = -self.rotation @ skew(specific_force)
            transition_rate[3:6, 12:15] = -self.rotation
            transition_rate[3:6, 15:18] = np.eye(3)
            transition_rate[6:9, 6:9] = -skew(omega)
            transition_rate[6:9, 9:12] = -np.eye(3)
            # Second-order state transition. The previous Euler covariance
            # propagation omitted the acceleration-to-position uncertainty
            # generated within an IMU interval and became overconfident in
            # precisely the vertical motion that later disagreed with VO.
            transition_rate_squared = transition_rate @ transition_rate
            transition = (
                np.eye(self._STATE_SIZE)
                + transition_rate * dt
                + 0.5 * transition_rate_squared * dt * dt
            )

            noise_mapping = np.zeros((self._STATE_SIZE, 15), dtype=np.float64)
            noise_mapping[6:9, 0:3] = -np.eye(3)
            noise_mapping[3:6, 3:6] = -self.rotation
            if acceleration_gyro_jacobian is not None:
                jacobian = np.asarray(
                    acceleration_gyro_jacobian, dtype=np.float64).reshape(3, 3)
                if np.all(np.isfinite(jacobian)):
                    # The first three process-noise components are gyro
                    # measurement noise.  Lever-arm compensation makes that
                    # noise enter specific force as well as attitude.
                    noise_mapping[3:6, 0:3] = (
                        self._acceleration_scale
                        * self.rotation @ jacobian
                    )
            noise_mapping[9:12, 6:9] = np.eye(3)
            noise_mapping[12:15, 9:12] = np.eye(3)
            noise_mapping[15:18, 12:15] = np.eye(3)
            continuous_noise = np.diag(
                np.concatenate(
                    (
                        np.full(3, self.config.gyro_noise_variance),
                        np.full(3, self.config.accel_noise_variance),
                        np.full(
                            3, self.config.gyro_bias_random_walk_variance),
                        np.full(
                            3, self.config.accel_bias_random_walk_variance),
                        np.full(
                            3, self.config.gravity_random_walk_variance),
                    )
                )
            )
            state_noise_density = (
                noise_mapping @ continuous_noise @ noise_mapping.T)
            # Second-order covariance integration of
            # exp(A t) L Q L^T exp(A^T t). This preserves the important
            # position/velocity and attitude/velocity correlations without a
            # costly matrix exponential at every 200 Hz sample.
            process_covariance = (
                state_noise_density * dt
                + 0.5
                * (
                    transition_rate @ state_noise_density
                    + state_noise_density @ transition_rate.T
                )
                * dt * dt
                + (
                    transition_rate @ state_noise_density
                    @ transition_rate.T
                )
                * (dt ** 3 / 3.0)
                + (
                    transition_rate_squared @ state_noise_density
                    + state_noise_density @ transition_rate_squared.T
                )
                * (dt ** 3 / 6.0)
            )
            process_covariance = self._project_psd(process_covariance)
            self.covariance = (
                transition @ self.covariance @ transition.T
                + process_covariance
            )
            self._interval_process_covariance = (
                transition @ self._interval_process_covariance @ transition.T
                + process_covariance
            )
            self._interval_process_covariance = self._project_psd(
                self._interval_process_covariance)
            if self._current_clone_cross_covariance is not None:
                self._current_clone_cross_covariance = (
                    transition @ self._current_clone_cross_covariance)
            self._repair_covariance_state()
            self.rotation = project_rotation(self.rotation @ so3_exp(omega * dt))

    def commit_pose_clone(self) -> None:
        """Clone the current camera-epoch pose and its cross-covariance."""
        with self._lock:
            pose_selector = np.zeros((6, self._STATE_SIZE), dtype=np.float64)
            pose_selector[0:3, 0:3] = np.eye(3)
            pose_selector[3:6, 6:9] = np.eye(3)
            self._clone_position = self.position.copy()
            self._clone_rotation = self.rotation.copy()
            self._clone_covariance = (
                pose_selector @ self.covariance @ pose_selector.T)
            self._current_clone_cross_covariance = (
                self.covariance @ pose_selector.T)
            self._interval_process_covariance.fill(0.0)

    def relative_pose_prediction(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return clone-to-current motion and covariance in ROS ordering.

        The covariance is projected in the canonical GTSAM right-local
        [rotation, translation] tangent, then serialized as ROS
        [translation, rotation].  Keeping this conversion at the boundary
        makes the iEKF and the graph use the same SE(3) error definition.
        """
        with self._lock:
            if (
                self._clone_position is None
                or self._clone_rotation is None
                or self._clone_covariance is None
                or self._current_clone_cross_covariance is None
            ):
                return None
            translation, rotation = self._relative_pose_components(
                self.position, self.rotation,
                self._clone_position, self._clone_rotation)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            self._repair_covariance_state()
            augmented_covariance = self._augmented_covariance()
            jacobian = self._relative_pose_jacobian(
                self.position, self.rotation,
                self._clone_position, self._clone_rotation)
            covariance_rt = jacobian @ augmented_covariance @ jacobian.T
            covariance_rt = self._regularize_covariance(
                covariance_rt, np.eye(6, dtype=np.float64))
            # geometry_msgs uses [translation, rotation], while the
            # canonical SE(3) tangent used by GTSAM is [rotation, translation].
            ros_order = np.array([3, 4, 5, 0, 1, 2])
            covariance_ros = covariance_rt[np.ix_(ros_order, ros_order)]
            return transform, 0.5 * (covariance_ros + covariance_ros.T)

    @staticmethod
    def _relative_pose_components(
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        clone_position: np.ndarray,
        clone_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            clone_rotation.T @ (current_position - clone_position),
            clone_rotation.T @ current_rotation,
        )

    def _augmented_covariance(self) -> np.ndarray:
        assert self._clone_covariance is not None
        assert self._current_clone_cross_covariance is not None
        # ``np.block`` requires nested lists on NumPy versions used by the
        # ROS 2 environment; tuples are not implicitly converted here.
        return np.block([
            [self.covariance, self._current_clone_cross_covariance],
            [self._current_clone_cross_covariance.T, self._clone_covariance],
        ])

    def _regularize_covariance(
        self,
        covariance: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Return a finite, symmetric PSD covariance with bounded variance."""
        matrix = np.asarray(covariance, dtype=np.float64)
        fallback = np.asarray(fallback, dtype=np.float64)
        if matrix.shape != fallback.shape or not np.all(np.isfinite(matrix)):
            return fallback.copy()
        matrix = 0.5 * (matrix + matrix.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            return fallback.copy()
        if not np.all(np.isfinite(eigenvalues)):
            return fallback.copy()
        eigenvalues = np.clip(eigenvalues, 1.0e-12, self._MAX_COVARIANCE_VARIANCE)
        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return 0.5 * (matrix + matrix.T)

    def _project_psd(self, covariance: np.ndarray) -> np.ndarray:
        """Symmetrize process covariance without inventing noise-free modes."""
        matrix = np.asarray(covariance, dtype=np.float64)
        matrix = 0.5 * (matrix + matrix.T)
        if not np.all(np.isfinite(matrix)):
            return np.zeros_like(matrix)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            return np.zeros_like(matrix)
        eigenvalues = np.clip(
            eigenvalues, 0.0, self._MAX_COVARIANCE_VARIANCE)
        result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return 0.5 * (result + result.T)

    def _repair_covariance_state(self) -> None:
        """Keep numerical failures local instead of poisoning future frames."""
        self.covariance = self._regularize_covariance(
            self.covariance, self._initial_covariance)
        if (
            self._clone_covariance is None
            or self._current_clone_cross_covariance is None
        ):
            return
        pose_selector = np.zeros((6, self._STATE_SIZE), dtype=np.float64)
        pose_selector[0:3, 0:3] = np.eye(3)
        pose_selector[3:6, 6:9] = np.eye(3)
        fallback_clone = pose_selector @ self._initial_covariance @ pose_selector.T
        self._clone_covariance = self._regularize_covariance(
            self._clone_covariance, fallback_clone)
        cross = np.asarray(
            self._current_clone_cross_covariance, dtype=np.float64)
        if cross.shape != (self._STATE_SIZE, 6) or not np.all(np.isfinite(cross)):
            self._current_clone_cross_covariance = np.zeros(
                (self._STATE_SIZE, 6), dtype=np.float64)
        else:
            self._current_clone_cross_covariance = cross

    def _relative_pose_jacobian(
        self,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        clone_position: np.ndarray,
        clone_rotation: np.ndarray,
    ) -> np.ndarray:
        """Linearize the clone-to-current pose in the GTSAM tangent.

        The output is the derivative of the right-local [rotation,
        translation] perturbation of the relative transform.  This is the
        same tangent used by a Pose3 BetweenFactor, unlike a separate
        translation subtraction and SO(3) logarithm.
        """
        epsilon = 1.0e-6
        base_translation, base_rotation = self._relative_pose_components(
            current_position, current_rotation,
            clone_position, clone_rotation)
        base_transform = np.eye(4, dtype=np.float64)
        base_transform[:3, :3] = base_rotation
        base_transform[:3, 3] = base_translation
        jacobian = np.zeros((6, self._STATE_SIZE + 6), dtype=np.float64)
        active_columns = list(range(0, 3)) + list(range(6, 9))
        active_columns += list(range(self._STATE_SIZE, self._STATE_SIZE + 6))
        for column in active_columns:
            position = current_position.copy()
            rotation = current_rotation.copy()
            cloned_position = clone_position.copy()
            cloned_rotation = clone_rotation.copy()
            if column < 3:
                position[column] += epsilon
            elif 6 <= column < 9:
                axis = np.zeros(3, dtype=np.float64)
                axis[column - 6] = epsilon
                rotation = rotation @ so3_exp(axis)
            elif self._STATE_SIZE <= column < self._STATE_SIZE + 3:
                cloned_position[column - self._STATE_SIZE] += epsilon
            else:
                axis = np.zeros(3, dtype=np.float64)
                axis[column - self._STATE_SIZE - 3] = epsilon
                cloned_rotation = cloned_rotation @ so3_exp(axis)
            perturbed_translation, perturbed_rotation = (
                self._relative_pose_components(
                    position, rotation, cloned_position, cloned_rotation))
            perturbed_transform = np.eye(4, dtype=np.float64)
            perturbed_transform[:3, :3] = perturbed_rotation
            perturbed_transform[:3, 3] = perturbed_translation
            local_delta = se3_log(
                np.linalg.inv(base_transform) @ perturbed_transform)
            jacobian[:, column] = local_delta / epsilon
        return jacobian

    def _augmented_state_copy(self) -> tuple[np.ndarray, ...]:
        assert self._clone_position is not None
        assert self._clone_rotation is not None
        return self._state_copy() + (
            self._clone_position.copy(), self._clone_rotation.copy())

    def _augmented_difference_from(
        self, reference: tuple[np.ndarray, ...]) -> np.ndarray:
        assert self._clone_position is not None
        assert self._clone_rotation is not None
        return np.concatenate((
            self._difference_from(reference[:6]),
            self._clone_position - reference[6],
            so3_log(reference[7].T @ self._clone_rotation),
        ))

    def _apply_augmented_error(self, error: np.ndarray) -> None:
        assert self._clone_position is not None
        assert self._clone_rotation is not None
        error = np.asarray(error, dtype=np.float64).reshape(self._STATE_SIZE + 6)
        self._apply_error(error[:self._STATE_SIZE])
        self._clone_position += error[self._STATE_SIZE:self._STATE_SIZE + 3]
        self._clone_rotation = project_rotation(
            self._clone_rotation
            @ so3_exp(error[self._STATE_SIZE + 3:self._STATE_SIZE + 6]))

    def _restore_augmented_state(
        self, reference: tuple[np.ndarray, ...]
    ) -> None:
        """Restore the nominal state if an iterated update cannot finish."""
        (
            self.position,
            self.velocity,
            self.rotation,
            self.gyro_bias,
            self.accel_bias,
            self.gravity,
            self._clone_position,
            self._clone_rotation,
        ) = tuple(value.copy() for value in reference)

    @staticmethod
    def _innovation_mahalanobis_squared(
        residual: np.ndarray,
        jacobian: np.ndarray,
        covariance: np.ndarray,
        noise: np.ndarray,
    ) -> float:
        innovation_covariance = jacobian @ covariance @ jacobian.T + noise
        if not np.all(np.isfinite(innovation_covariance)):
            return math.nan
        try:
            result = float(
                residual @ np.linalg.solve(innovation_covariance, residual))
        except np.linalg.LinAlgError:
            return math.inf
        return result

    def _adapt_interval_process_covariance(
        self,
        prior_covariance: np.ndarray,
        residual: np.ndarray,
        jacobian: np.ndarray,
        noise: np.ndarray,
        visual_trusted: bool,
    ) -> tuple[np.ndarray, float, float, float]:
        """Match innovation by scaling only this image interval's IMU noise.

        Visual covariance is a property of the successful geometric solve and
        is never changed here. Process noise accumulated after the pose clone
        is independent of that clone, so adding it to the current-state block
        is the covariance-consistent way to admit an under-modelled inertial
        interval while retaining all earlier information.
        """
        raw_nis = self._innovation_mahalanobis_squared(
            residual, jacobian, prior_covariance, noise)
        threshold = max(self.config.visual_nis_soft_threshold, 1.0)
        maximum_scale = self._MAX_PROCESS_NOISE_INFLATION
        if (
            not visual_trusted
            or not np.isfinite(raw_nis)
            or raw_nis <= threshold
        ):
            return prior_covariance, 1.0, raw_nis, raw_nis

        interval_process_augmented = np.zeros_like(prior_covariance)
        interval_process_augmented[
            :self._STATE_SIZE, :self._STATE_SIZE
        ] = self._interval_process_covariance
        projected_process = (
            jacobian @ interval_process_augmented @ jacobian.T)
        if (
            not np.all(np.isfinite(projected_process))
            or float(np.trace(projected_process)) <= 1.0e-18
        ):
            return prior_covariance, 1.0, raw_nis, raw_nis

        def covariance_for(scale: float) -> np.ndarray:
            return (
                prior_covariance
                + (scale - 1.0) * interval_process_augmented
            )

        maximum_nis = self._innovation_mahalanobis_squared(
            residual,
            jacobian,
            covariance_for(maximum_scale),
            noise,
        )
        if not np.isfinite(maximum_nis) or maximum_nis > threshold:
            scale = maximum_scale
            adapted_nis = maximum_nis
        else:
            # NIS decreases monotonically as PSD process covariance is added.
            # Find the smallest scale which makes the innovation statistically
            # consistent, preserving as much inertial information as possible.
            lower = 1.0
            upper = maximum_scale
            for _ in range(32):
                middle = math.sqrt(lower * upper)
                middle_nis = self._innovation_mahalanobis_squared(
                    residual,
                    jacobian,
                    covariance_for(middle),
                    noise,
                )
                if np.isfinite(middle_nis) and middle_nis <= threshold:
                    upper = middle
                else:
                    lower = middle
            scale = upper
            adapted_nis = self._innovation_mahalanobis_squared(
                residual, jacobian, covariance_for(scale), noise)

        adapted_covariance = self._regularize_covariance(
            covariance_for(scale), prior_covariance)
        return adapted_covariance, scale, raw_nis, adapted_nis

    def correct_relative_pose(
        self,
        clone_from_current_measurement: np.ndarray,
        covariance_ros: np.ndarray | None,
        visual_trusted: bool = True,
    ) -> VisualCorrectionResult:
        """Fuse a visual clone-to-current pose with the inertial prediction."""
        with self._lock:
            if (
                not self._pose_initialized
                or self._clone_position is None
                or self._clone_rotation is None
            ):
                return VisualCorrectionResult(
                    False, math.inf, 0,
                    visual_trusted=visual_trusted,
                    failure_reason="clone_not_initialized")
            measurement = np.asarray(
                clone_from_current_measurement, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(measurement)):
                self.rejected_visual_updates += 1
                return VisualCorrectionResult(
                    False, math.nan, 0,
                    visual_trusted=visual_trusted,
                    failure_reason="nonfinite_measurement")
            if not all(np.all(np.isfinite(value)) for value in (
                self.position,
                self.velocity,
                self.rotation,
                self.gyro_bias,
                self.accel_bias,
                self.gravity,
            )):
                self.rejected_visual_updates += 1
                return VisualCorrectionResult(
                    False, math.nan, 0,
                    visual_trusted=visual_trusted,
                    failure_reason="nonfinite_state")
            self._repair_covariance_state()
            measurement_rotation = project_rotation(measurement[:3, :3])
            measurement_translation = measurement[:3, 3]
            noise = self._visual_noise(covariance_ros)
            prior = self._augmented_state_copy()
            prior_covariance = self._regularize_covariance(
                self._augmented_covariance(),
                np.eye(self._STATE_SIZE + 6, dtype=np.float64),
            )

            predicted_translation, predicted_rotation = (
                self._relative_pose_components(
                    self.position, self.rotation,
                    self._clone_position, self._clone_rotation))
            predicted_transform = np.eye(4, dtype=np.float64)
            predicted_transform[:3, :3] = predicted_rotation
            predicted_transform[:3, 3] = predicted_translation
            measurement_transform = np.eye(4, dtype=np.float64)
            measurement_transform[:3, :3] = measurement_rotation
            measurement_transform[:3, 3] = measurement_translation
            # Express the innovation as measurement minus prediction in the
            # canonical SE(3) tangent.  The negative sign is important:
            # se3_log(measurement^-1 * prediction) is the GTSAM factor error
            # (prediction minus measurement), whereas the Kalman update needs
            # the opposite innovation.
            # With h(x) = Log(Z^-1 P(x)) and target 0 the innovation is
            # -h(x) and the measurement Jacobian is +dh/dx.  (The legacy
            # filter negated the Jacobian as well, so its never-exercised
            # update moved the state away from the measurement.)
            residual = -se3_log(
                np.linalg.inv(measurement_transform) @ predicted_transform)
            if not np.all(np.isfinite(residual)):
                self.rejected_visual_updates += 1
                return VisualCorrectionResult(
                    False, math.nan, 0,
                    visual_trusted=visual_trusted,
                    failure_reason="nonfinite_residual")
            jacobian = self._relative_pose_jacobian(
                self.position, self.rotation,
                self._clone_position, self._clone_rotation)
            (
                prior_covariance,
                process_noise_inflation,
                raw_mahalanobis_squared,
                mahalanobis_squared,
            ) = self._adapt_interval_process_covariance(
                prior_covariance,
                residual,
                jacobian,
                noise,
                visual_trusted,
            )
            if not np.isfinite(mahalanobis_squared):
                self.rejected_visual_updates += 1
                return VisualCorrectionResult(
                    False, mahalanobis_squared, 0,
                    process_noise_inflation=process_noise_inflation,
                    raw_mahalanobis_squared=raw_mahalanobis_squared,
                    visual_trusted=visual_trusted,
                    failure_reason="nonfinite_mahalanobis")

            iterations = 0
            gain = None
            for iteration in range(max(1, self.config.maximum_iterations)):
                iterations = iteration + 1
                predicted_translation, predicted_rotation = (
                    self._relative_pose_components(
                        self.position, self.rotation,
                        self._clone_position, self._clone_rotation))
                predicted_transform = np.eye(4, dtype=np.float64)
                predicted_transform[:3, :3] = predicted_rotation
                predicted_transform[:3, 3] = predicted_translation
                measurement_transform = np.eye(4, dtype=np.float64)
                measurement_transform[:3, :3] = measurement_rotation
                measurement_transform[:3, 3] = measurement_translation
                residual = -se3_log(
                    np.linalg.inv(measurement_transform)
                    @ predicted_transform)
                jacobian = self._relative_pose_jacobian(
                    self.position, self.rotation,
                    self._clone_position, self._clone_rotation)
                innovation_covariance = (
                    jacobian @ prior_covariance @ jacobian.T + noise)
                if not np.all(np.isfinite(innovation_covariance)):
                    self._restore_augmented_state(prior)
                    self.rejected_visual_updates += 1
                    return VisualCorrectionResult(
                        False, mahalanobis_squared, iterations,
                        process_noise_inflation=process_noise_inflation,
                        raw_mahalanobis_squared=raw_mahalanobis_squared,
                        visual_trusted=visual_trusted,
                        failure_reason="nonfinite_iterated_innovation")
                try:
                    gain = np.linalg.solve(
                        innovation_covariance,
                        jacobian @ prior_covariance,
                    ).T
                    # Schmidt update: retain the previously published clone as
                    # a fixed nominal pose while still using its covariance
                    # and cross-correlation to update the active IMU state.
                    gain[self._STATE_SIZE:, :] = 0.0
                except np.linalg.LinAlgError:
                    self._restore_augmented_state(prior)
                    self.rejected_visual_updates += 1
                    return VisualCorrectionResult(
                        False, mahalanobis_squared, iterations,
                        process_noise_inflation=process_noise_inflation,
                        raw_mahalanobis_squared=raw_mahalanobis_squared,
                        visual_trusted=visual_trusted,
                        failure_reason="singular_iterated_innovation")
                iterate_error = self._augmented_difference_from(prior)
                correction = (
                    gain @ (residual + jacobian @ iterate_error)
                    - iterate_error)
                if not np.all(np.isfinite(correction)):
                    self._restore_augmented_state(prior)
                    self.rejected_visual_updates += 1
                    return VisualCorrectionResult(
                        False, mahalanobis_squared, iterations,
                        process_noise_inflation=process_noise_inflation,
                        raw_mahalanobis_squared=raw_mahalanobis_squared,
                        visual_trusted=visual_trusted,
                        failure_reason="nonfinite_state_correction")
                self._apply_augmented_error(correction)
                if np.linalg.norm(correction[:9]) < self.config.convergence_epsilon:
                    break

            assert gain is not None
            identity_minus_gain = (
                np.eye(self._STATE_SIZE + 6) - gain @ jacobian)
            augmented_covariance = (
                identity_minus_gain @ prior_covariance
                @ identity_minus_gain.T
                + gain @ noise @ gain.T)
            # The covariance above is expressed around the pre-update error
            # state. Transport it to the reset error coordinates after the
            # nominal rotation/gravity injection. Omitting this reset causes
            # inconsistency to accumulate during sustained rotations.
            injected_error = self._augmented_difference_from(prior)
            reset_jacobian = np.eye(
                self._STATE_SIZE + 6, dtype=np.float64)
            reset_jacobian[6:9, 6:9] = (
                np.eye(3) - 0.5 * skew(injected_error[6:9]))
            gravity_norm = float(np.linalg.norm(self.gravity))
            if gravity_norm > 1.0e-9:
                gravity_direction = self.gravity / gravity_norm
                reset_jacobian[15:18, 15:18] = (
                    np.eye(3)
                    - np.outer(gravity_direction, gravity_direction))
            augmented_covariance = (
                reset_jacobian @ augmented_covariance
                @ reset_jacobian.T)
            augmented_covariance = 0.5 * (
                augmented_covariance + augmented_covariance.T)
            fallback_augmented = np.zeros(
                (self._STATE_SIZE + 6, self._STATE_SIZE + 6),
                dtype=np.float64,
            )
            fallback_augmented[:self._STATE_SIZE, :self._STATE_SIZE] = (
                self._initial_covariance)
            pose_selector = np.zeros((6, self._STATE_SIZE), dtype=np.float64)
            pose_selector[0:3, 0:3] = np.eye(3)
            pose_selector[3:6, 6:9] = np.eye(3)
            fallback_augmented[self._STATE_SIZE:, self._STATE_SIZE:] = (
                pose_selector @ self._initial_covariance @ pose_selector.T)
            augmented_covariance = self._regularize_covariance(
                augmented_covariance,
                fallback_augmented,
            )
            self.covariance = augmented_covariance[
                :self._STATE_SIZE, :self._STATE_SIZE]
            self._current_clone_cross_covariance = augmented_covariance[
                :self._STATE_SIZE, self._STATE_SIZE:]
            self._clone_covariance = augmented_covariance[
                self._STATE_SIZE:, self._STATE_SIZE:]
            self.accepted_visual_updates += 1
            return VisualCorrectionResult(
                True,
                mahalanobis_squared,
                iterations,
                process_noise_inflation=process_noise_inflation,
                raw_mahalanobis_squared=raw_mahalanobis_squared,
                visual_trusted=visual_trusted,
            )

    def _visual_noise(self, covariance_ros: np.ndarray | None) -> np.ndarray:
        if covariance_ros is None:
            noise = np.zeros((6, 6), dtype=np.float64)
        else:
            noise = np.asarray(covariance_ros, dtype=np.float64).reshape(6, 6)
            if not np.all(np.isfinite(noise)):
                noise = np.zeros((6, 6), dtype=np.float64)
        noise = 0.5 * (noise + noise.T)
        # Convert the ROS [translation, rotation] matrix to the canonical
        # GTSAM [rotation, translation] tangent before using it in the filter.
        ros_order = np.array([3, 4, 5, 0, 1, 2])
        noise = noise[np.ix_(ros_order, ros_order)]
        # The configured floors prevent a weak covariance from becoming an
        # effectively exact measurement.  Model terms represent independent
        # uncertainty (e.g. calibration/time-sync) and are therefore added in
        # variance, rather than replacing diagonal entries or scaling the
        # solver covariance.
        rotation_variance = (
            self.config.visual_rotation_std_floor ** 2
            + self.config.visual_model_rotation_std ** 2)
        translation_variance = (
            self.config.visual_translation_std_floor ** 2
            + self.config.visual_model_translation_std ** 2)
        floors = np.concatenate(
            (
                np.full(3, rotation_variance),
                np.full(3, translation_variance),
            )
        )
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(noise)
        except np.linalg.LinAlgError:
            noise = np.zeros((6, 6), dtype=np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        noise = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Floors represent independent unmodelled visual uncertainty and are
        # therefore additive; modifying only the diagonal after an eigenvalue
        # clamp can otherwise produce an inconsistent anisotropic covariance.
        noise += np.diag(floors)
        return 0.5 * (noise + noise.T)

    def _state_copy(self) -> tuple[np.ndarray, ...]:
        return (
            self.position.copy(),
            self.velocity.copy(),
            self.rotation.copy(),
            self.gyro_bias.copy(),
            self.accel_bias.copy(),
            self.gravity.copy(),
        )

    def _difference_from(self, reference: tuple[np.ndarray, ...]) -> np.ndarray:
        return np.concatenate(
            (
                self.position - reference[0],
                self.velocity - reference[1],
                so3_log(reference[2].T @ self.rotation),
                self.gyro_bias - reference[3],
                self.accel_bias - reference[4],
                self.gravity - reference[5],
            )
        )

    def _apply_error(self, error: np.ndarray) -> None:
        error = np.asarray(error, dtype=np.float64).reshape(self._STATE_SIZE)
        self.position += error[0:3]
        self.velocity += error[3:6]
        self.rotation = project_rotation(self.rotation @ so3_exp(error[6:9]))
        self.gyro_bias += error[9:12]
        self.accel_bias += error[12:15]
        self.gravity += error[15:18]
        gravity_norm = float(np.linalg.norm(self.gravity))
        if gravity_norm > 1.0e-9:
            self.gravity *= self.config.gravity_magnitude / gravity_norm

    def pose_matrix(self) -> np.ndarray:
        with self._lock:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = self.rotation
            pose[:3, 3] = self.position
            return pose

    def pose_covariance_ros(self) -> np.ndarray:
        with self._lock:
            indices = np.array([0, 1, 2, 6, 7, 8])
            result = self.covariance[np.ix_(indices, indices)].copy()
            return 0.5 * (result + result.T)

    def diagnostics(self) -> dict[str, object]:
        """Return a thread-safe snapshot of initialization and nominal state."""
        with self._lock:
            buffer_span_sec = 0.0
            if len(self._samples) >= 2:
                buffer_span_sec = (
                    self._samples[-1].timestamp_ns
                    - self._samples[0].timestamp_ns
                ) * 1.0e-9
            return {
                "initial_mean_acceleration":
                    self._initial_mean_acceleration.tolist(),
                "initial_mean_acceleration_norm": float(
                    np.linalg.norm(self._initial_mean_acceleration)),
                "acceleration_scale": float(self._acceleration_scale),
                "initialization_stationary": self._initialization_stationary,
                "gyro_bias": self.gyro_bias.tolist(),
                "accel_bias": self.accel_bias.tolist(),
                "gravity": self.gravity.tolist(),
                "velocity": self.velocity.tolist(),
                "buffered_samples": len(self._samples),
                "buffer_span_sec": float(buffer_span_sec),
                "dropped_buffered_samples": self.dropped_buffered_samples,
            }
