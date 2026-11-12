"""End-to-end odometry on synthetic multi-camera rigs (no images needed).

A fake joint feature model "detects" the projections of known world points
(descriptor = point id) and matches by id, so the whole orchestration
(stereo, temporal verification, generalized PnP, anchored refinement,
covariance) runs on exact geometry plus pixel noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from colmap_odometry.config import OdometryConfig
from colmap_odometry.features import FeatureFrontend, Features, JointModel
from colmap_odometry.frames import BodyFrame
from colmap_odometry.geometry.lie import invert, make_transform, se3_log, so3_exp
from colmap_odometry.odometry import Odometry, Status
from synthetic import make_rig

WIDE = {"model": "PINHOLE", "width": 800, "height": 600,
        "params": [300.0, 300.0, 400.0, 300.0]}


class SyntheticFeatures(JointModel):
    """Projects ``world_points`` for the rig pose set before each frame."""

    descriptor = "synthetic"

    def __init__(self, rig, world_points, noise_px, rng, outliers=0.0):
        self.rig = rig
        self.world_points = world_points
        self.noise_px = noise_px
        self.rng = rng
        self.outliers = outliers
        self.world_from_rig = np.eye(4)

    def extract(self, images):
        features = []
        rig_from_world = invert(self.world_from_rig)
        for name in self.rig.camera_names:
            camera_from_world = self.rig.camera_from_rig(name) @ rig_from_world
            points = self.world_points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
            pixels = self.rig.cameras[name].project(points)
            camera = self.rig.cameras[name]
            visible = (np.isfinite(pixels).all(axis=1) & (points[:, 2] > 0.3)
                       & (pixels[:, 0] > 0) & (pixels[:, 0] < camera.width)
                       & (pixels[:, 1] > 0) & (pixels[:, 1] < camera.height))
            ids = np.flatnonzero(visible)
            keypoints = pixels[ids] + self.rng.normal(scale=self.noise_px, size=(len(ids), 2))
            features.append(Features(keypoints, ids, (camera.width, camera.height)))
        return features

    def match(self, pairs):
        results = []
        for first, second in pairs:
            _, a, b = np.intersect1d(first.descriptors, second.descriptors, return_indices=True)
            matches = np.column_stack((a, b))
            if self.outliers and len(matches):
                corrupt = self.rng.random(len(matches)) < self.outliers
                matches[corrupt, 1] = self.rng.integers(0, len(second), int(corrupt.sum()))
            results.append(matches)
        return results


def _four_camera_rig(pairs):
    # cam0 is deliberately not at the rig origin.
    front = make_transform(so3_exp([0.0, 0.05, 0.0]), [0.3, 0.1, 0.05])
    rear = make_transform(so3_exp([0.0, np.pi, 0.0]), [-0.4, 0.0, 0.0])
    baseline = make_transform(np.eye(3), [0.12, 0.0, 0.0])
    extrinsics = {"cam0": front, "cam1": front @ baseline,
                  "cam2": rear, "cam3": rear @ baseline}
    return make_rig(extrinsics, {name: WIDE for name in extrinsics}, pairs)


def _world_points(rng, count=3000):
    # A room-like shell of points around the trajectory.
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * rng.uniform(3.0, 12.0, size=(count, 1))


def _trajectory(steps):
    poses = []
    for index in range(steps):
        t = 0.1 * index
        poses.append(make_transform(so3_exp([0.02 * np.sin(t), 0.15 * t, 0.01]),
                                    [0.5 * np.sin(t), 0.05 * t, 0.4 * t]))
    return poses


def _run(pairs, temporal=None, noise_px=0.5, steps=12, seed=0, outliers=0.0):
    rng = np.random.default_rng(seed)
    rig = _four_camera_rig(pairs)
    model = SyntheticFeatures(rig, _world_points(rng), noise_px, rng, outliers)
    config = OdometryConfig.from_mapping({
        "stereo_pairs": [list(pair) for pair in pairs],
        "temporal_cameras": temporal,
        "stereo": {"max_depth": 50.0},
        "feature_tracks": {"min_length": 3},
    })
    odometry = Odometry(rig, FeatureFrontend(model, model), config)
    truth = _trajectory(steps)
    images = [np.zeros((1, 1), np.uint8)] * len(rig.camera_names)
    results = []
    for index, pose in enumerate(truth):
        model.world_from_rig = invert(truth[0]) @ pose
        results.append(odometry.process(index * 50_000_000, images))
    return rig, odometry, results, truth


@pytest.mark.parametrize("pairs, temporal", [
    ((("cam0", "cam1"),), None),                     # stereo + monocular rear cameras
    ((("cam0", "cam1"), ("cam2", "cam3")), None),    # front and rear stereo pairs
    ((("cam0", "cam1"), ("cam2", "cam3")), ["cam2"]),  # rear pair only, temporally
])
def test_multicamera_trajectory(pairs, temporal):
    rig, odometry, results, truth = _run(pairs, temporal)
    assert results[0].status is Status.BOOTSTRAP
    assert all(result.tracked for result in results[1:]), [r.failure_reason for r in results]
    for previous, current, result in zip(truth, truth[1:], results[1:]):
        expected = invert(previous) @ current
        error = se3_log(invert(result.motion) @ expected)
        assert np.linalg.norm(error[:3]) < 2e-3
        assert np.linalg.norm(error[3:]) < 5e-3
        assert result.covariance_source == "anchored_inverse_depth"
    final = invert(truth[0]) @ truth[-1]
    assert np.linalg.norm(results[-1].world_from_rig[:3, 3] - final[:3, 3]) < 0.03
    if temporal is None:
        assert set(odometry.temporal_cameras) == {"cam0", "cam2"} | (
            {"cam3"} if len(pairs) == 1 else set())


def test_tracks_and_robustness_to_outliers():
    _, _, results, _ = _run((("cam0", "cam1"),), outliers=0.1, seed=3)
    assert all(result.tracked for result in results[1:])
    updates = [update for result in results for update in result.track_updates]
    assert updates and {update.camera_name for update in updates} <= {"cam0", "cam2", "cam3"}
    first = next(update for update in updates if update.camera_name == "cam0")
    assert len(first.observations) >= 3
    assert any(item.has_stereo_point for item in first.observations)


def _monte_carlo_nees(rig, pairs, temporal, trials=20, frame=np.eye(4), refinement=None):
    """Mean NEES/6 of the second-frame motion; ``frame`` re-expresses the rig."""
    points = _world_points(np.random.default_rng(7), 2500)
    config = OdometryConfig.from_mapping({"stereo_pairs": [list(p) for p in pairs],
                                          "temporal_cameras": temporal,
                                          "stereo": {"max_depth": 50.0},
                                          "refinement": refinement or {}})
    truth = _trajectory(2)
    motion = invert(truth[0]) @ truth[1]
    expected = invert(frame) @ motion @ frame
    values, motions = [], []
    images = [np.zeros((1, 1), np.uint8)] * len(rig.camera_names)
    for trial in range(trials):
        model = SyntheticFeatures(rig, points, 1.0, np.random.default_rng(100 + trial))
        odometry = Odometry(rig, FeatureFrontend(model, model), config)
        model.world_from_rig = frame
        odometry.process(0, images)
        model.world_from_rig = motion @ frame
        result = odometry.process(50_000_000, images)
        assert result.tracked
        error = se3_log(invert(result.motion) @ expected)
        values.append(error @ np.linalg.solve(result.motion_covariance, error) / 6.0)
        motions.append(result.motion)
    return float(np.mean(values)), motions


def test_estimate_is_invariant_to_rig_frame():
    """Moving the rig origin to a host camera changes nothing physical."""
    pairs = (("cam2", "cam3"),)
    rig = _four_camera_rig(pairs)
    rig_from_cam2 = rig.rig_from_camera("cam2")
    moved = make_rig({name: invert(rig_from_cam2) @ rig.rig_from_camera(name)
                      for name in rig.camera_names},
                     {name: WIDE for name in rig.camera_names}, pairs)
    # The PnP-consistency gate depends on RANSAC sampling, which differs per
    # frame; disable it to compare the converged estimates exactly.
    refinement = {"inlier_gate": 1.0e6, "max_iterations": 15}
    nees, motions = _monte_carlo_nees(rig, pairs, ["cam2"], trials=5, refinement=refinement)
    moved_nees, moved_motions = _monte_carlo_nees(moved, pairs, ["cam2"], trials=5,
                                                  frame=rig_from_cam2, refinement=refinement)
    assert moved_nees == pytest.approx(nees, rel=1e-2)
    for original, other in zip(motions, moved_motions):
        np.testing.assert_allclose(invert(rig_from_cam2) @ original @ rig_from_cam2, other,
                                   atol=1e-7)


@pytest.mark.parametrize("pairs, temporal", [
    ((("cam0", "cam1"),), ["cam0"]),                     # front stereo pair
    ((("cam2", "cam3"),), ["cam2"]),                     # rear pair, motion along its axis
    ((("cam0", "cam1"), ("cam2", "cam3")), None),        # both pairs
])
def test_covariance_is_consistent(pairs, temporal):
    """Monte-Carlo NEES with modelled noise (sigma = 1 px) is about 1.

    The anchor pixel is a measurement, so the noise it shares among all views
    of a landmark is modelled exactly.  The PnP-consistency gate truncates
    the noise slightly, hence values a little above one.
    """
    nees, _ = _monte_carlo_nees(_four_camera_rig(pairs), pairs, temporal)
    assert 0.6 < nees < 1.4, nees


def test_body_frame_conversion_matches_rig_motion():
    rig, _, results, truth = _run((("cam0", "cam1"),), steps=3)
    camera_from_base = make_transform(so3_exp([0.3, -0.2, 1.0]), [0.1, 0.2, -0.3])
    body = BodyFrame.from_camera(rig.rig_from_camera("cam0"), camera_from_base)
    motion = body.motion(results[1].motion)
    # Conjugation keeps the motion magnitude and maps the rig axis-angle.
    assert np.isclose(np.linalg.norm(se3_log(motion)[:3]),
                      np.linalg.norm(se3_log(results[1].motion)[:3]))
    covariance = body.motion_covariance_ros(results[1].motion_covariance)
    assert np.allclose(covariance, covariance.T)
    assert np.isclose(np.trace(covariance[3:, 3:]),
                      np.trace(results[1].motion_covariance[:3, :3]))


def test_robust_to_gross_temporal_outliers():
    """Wrong temporal matches are excluded by the PnP-consistency gate."""
    rng = np.random.default_rng(11)
    rig = _four_camera_rig((("cam0", "cam1"),))
    model = SyntheticFeatures(rig, _world_points(rng), 0.5, rng, outliers=0.15)
    config = OdometryConfig.from_mapping({
        "stereo_pairs": [["cam0", "cam1"]], "temporal_cameras": ["cam0"],
        "stereo": {"max_depth": 50.0}})
    odometry = Odometry(rig, FeatureFrontend(model, model), config)
    truth = _trajectory(4)
    images = [np.zeros((1, 1), np.uint8)] * 4
    for index, pose in enumerate(truth):
        model.world_from_rig = invert(truth[0]) @ pose
        result = odometry.process(index, images)
        if index:
            report = result.report["refinement"]
            assert report["candidate_landmarks"] < report["temporal_landmarks"]
            expected = invert(truth[index - 1]) @ pose
            assert np.linalg.norm(se3_log(invert(result.motion) @ expected)[3:]) < 5e-3
