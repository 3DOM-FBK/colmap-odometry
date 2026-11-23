"""IMU filter: legacy parity of propagation and the new visual update."""

from __future__ import annotations

import dataclasses

import numpy as np

from colmap_odometry.geometry.lie import invert, make_transform, so3_exp
from colmap_odometry.imu.config import ImuConfig
from colmap_odometry.imu.filter import ImuFilter
from colmap_odometry.imu.pipeline import InertialPipeline
from legacy_reference.imu_iekf import ImuIekfConfig, ImuIteratedErrorStateFilter

RATE = 200.0
GRAVITY = np.array([0.0, 0.0, -9.80665])


def _samples(duration, gyro_bias, accel_bias, rng=None, noise=0.0):
    rotation = np.eye(3)
    samples = []
    previous_omega = np.zeros(3)
    position = np.zeros(3)
    velocity = np.zeros(3)
    poses = {}
    for k in range(int(duration * RATE) + 1):
        t = k / RATE
        s = t - 1.0
        if s < 0:
            omega = np.zeros(3)
            acceleration = np.zeros(3)
        else:
            omega = np.array([0.3 * np.sin(0.7 * s), 0.2 * np.cos(0.5 * s), 0.4])
            acceleration = np.array([-0.64 * np.sin(0.8 * s), -0.18 * np.sin(0.6 * s),
                                     0.05 * np.cos(s)])
        specific_force = rotation.T @ (acceleration - GRAVITY)
        measured_acc = specific_force + accel_bias
        measured_gyro = omega + gyro_bias
        if rng is not None:
            measured_acc = measured_acc + rng.normal(scale=noise, size=3)
            measured_gyro = measured_gyro + rng.normal(scale=noise * 0.1, size=3)
        stamp = int(round(1e9 * (10.0 + t)))
        samples.append((stamp, measured_acc, measured_gyro))
        poses[stamp] = (make_transform(rotation, position), velocity.copy())
        # Ground-truth integration (midpoint).
        position = position + velocity / RATE + 0.5 * acceleration / RATE ** 2
        velocity = velocity + acceleration / RATE
        rotation = rotation @ so3_exp(0.5 * (omega + previous_omega) / RATE)
        previous_omega = omega
    return samples, poses


def _legacy_config(config: ImuConfig) -> ImuIekfConfig:
    fields = {item.name for item in dataclasses.fields(ImuIekfConfig)}
    values = {name: getattr(config, name) for name in fields if hasattr(config, name)}
    values["maximum_imu_gap_sec"] = config.maximum_gap_sec
    return ImuIekfConfig(**values)


def test_propagation_matches_legacy_filter():
    config = ImuConfig(enabled=True, T_base_imu=tuple(make_transform(
        so3_exp([0.1, -0.2, 0.05]), [0.05, -0.02, 0.1]).reshape(-1)))
    legacy = ImuIteratedErrorStateFilter(_legacy_config(config),
                                         config.base_from_imu()[:3, :3],
                                         config.base_from_imu()[:3, 3])
    new = ImuFilter(config)
    samples, _ = _samples(4.0, np.array([0.01, -0.02, 0.005]), np.array([0.05, 0.0, -0.1]))
    frame_stamps = [stamp for stamp, _, _ in samples[::10]]
    initialized = False
    for frame_stamp in frame_stamps:
        for stamp, acc, gyro in samples:
            if frame_stamp - 50_000_000 < stamp <= frame_stamp:
                assert legacy.add_sample(stamp, acc, gyro) == new.add_sample(stamp, acc, gyro)
        if not initialized:
            if legacy.imu_initialized:
                assert new.imu_initialized
                assert legacy.initialize_pose(frame_stamp, np.eye(4))
                assert new.initialize_pose(frame_stamp, np.eye(4))
                initialized = True
            continue
        assert legacy.predict_to(frame_stamp) and new.predict_to(frame_stamp)
        legacy_prediction = legacy.relative_pose_prediction()
        new_prediction = new.relative_pose_prediction()
        np.testing.assert_allclose(new_prediction[0], legacy_prediction[0], atol=1e-12)
        np.testing.assert_allclose(new_prediction[1], legacy_prediction[1], rtol=1e-9,
                                   atol=1e-15)
        pose = legacy.pose_matrix()
        legacy.anchor_visual_pose(frame_stamp, pose)
        new.anchor_visual_pose(frame_stamp, pose)
    assert initialized


def test_visual_update_estimates_biases_and_velocity():
    gyro_bias = np.array([0.004, -0.003, 0.002])
    accel_bias = np.array([0.08, -0.05, 0.06])
    samples, truth = _samples(40.0, gyro_bias, accel_bias, np.random.default_rng(0), 0.01)
    results = {}
    for visual_update in (False, True):
        config = ImuConfig(enabled=True, visual_update=visual_update,
                           initialization_duration_sec=0.5, align_world_with_gravity=False,
                           gyro_noise_variance=1e-6, accel_noise_variance=1e-4,
                           visual_translation_std_floor=0.002,
                           visual_rotation_std_floor=0.001)
        pipeline = InertialPipeline(config, np.eye(4))
        frame_stamps = [stamp for stamp, _, _ in samples[::10]]  # 20 Hz camera
        index = 0
        previous = None
        for frame_stamp in frame_stamps:
            while index < len(samples) and samples[index][0] <= frame_stamp:
                pipeline.add_sample(*samples[index])
                index += 1
            prior = pipeline.prior(frame_stamp, previous)
            pose = truth[frame_stamp][0]
            relative = None
            covariance = np.diag([1e-6] * 3 + [1e-6] * 3)
            if previous is not None:
                relative = invert(truth[previous][0]) @ pose
            pipeline.update(frame_stamp, pose, relative, covariance)
            previous = frame_stamp
        state = pipeline.filter
        velocity_error = np.linalg.norm(state.velocity - truth[previous][1])
        results[visual_update] = (np.linalg.norm(state.gyro_bias - gyro_bias),
                                  np.linalg.norm(state.accel_bias - accel_bias),
                                  velocity_error, prior)
    # Without updates the biases never move and the velocity drifts.
    assert results[False][1] > 0.05
    assert results[False][2] > 1.0
    gyro_error, accel_error, velocity_error, prior = results[True]
    assert gyro_error < 1e-3, results[True]
    assert accel_error < 0.03, results[True]
    assert velocity_error < 0.05, results[True]
    assert prior is not None and prior.factors.uses_rotation


def test_gyro_bias_window_recovers_constant_bias():
    from colmap_odometry.imu.bias_window import GyroBiasWindow, preintegrate

    rng = np.random.default_rng(0)
    bias = np.array([4e-4, -7e-4, 2e-4])
    window = GyroBiasWindow(window_sec=30.0, prior_std=0.01)
    dt = 0.005
    times = np.arange(0.0, 40.0, dt)
    stamps = np.round(times * 1e9).astype(np.int64)
    rates = np.column_stack([0.8 * np.sin(1.3 * times), 0.5 * np.cos(0.7 * times + 1.0),
                             0.3 * np.sin(2.1 * times)])
    measured = rates + bias + rng.normal(scale=1.6e-4 / np.sqrt(dt), size=rates.shape)
    frames = np.arange(0, len(times), 10)
    for start, end in zip(frames[:-1], frames[1:]):
        interval = preintegrate(stamps[start:end + 1], measured[start:end + 1],
                                stamps[start], stamps[end])
        truth = np.eye(3)
        for index in range(start, end):
            truth = truth @ so3_exp(0.5 * (rates[index] + rates[index + 1]) * dt)
        visual = truth @ so3_exp(rng.normal(scale=3e-4, size=3))
        window.add(interval, visual, np.eye(3) * 3e-4 ** 2)
    error = window.bias - bias
    assert error @ np.linalg.solve(window.covariance, error) < 16.3  # chi2(3) 99.9 %
    # The window keeps only the last 30 s of 0.05 s intervals.
    assert np.sqrt(window.covariance[0, 0]) < 3.5e-4
