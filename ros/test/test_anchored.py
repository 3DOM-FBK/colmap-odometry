"""Anchored refinement on synthetic stereo data."""

from __future__ import annotations

import numpy as np
import pytest

from colmap_odometry.estimation.anchored import RefinementOptions, View, build_problem, refine
from colmap_odometry.estimation.inertial import InertialFactors
from colmap_odometry.geometry.lie import left_update, se3_log, invert, so3_exp
from synthetic import observe, random_motion, scene_points, stereo_rig


def _scenario(seed: int, outliers: float = 0.05, noise_px: float = 0.7):
    rng = np.random.default_rng(seed)
    rig = stereo_rig()
    motion = random_motion(rng)
    points = scene_points(rng, 400, far=40.0)
    clean = observe(rig, "cam0", np.eye(4), points)
    previous_left = observe(rig, "cam0", np.eye(4), points, rng, noise_px)
    previous_right = observe(rig, "cam1", np.eye(4), points, rng, noise_px)
    current_left = observe(rig, "cam0", motion, points, rng, noise_px)
    current_right = observe(rig, "cam1", motion, points, rng, noise_px)
    keep = np.isfinite(previous_left).all(axis=1) & np.isfinite(current_left).all(axis=1)
    data = dict(previous_left=previous_left[keep], previous_right=previous_right[keep],
                current_left=current_left[keep], current_right=current_right[keep],
                clean_anchor=clean[keep])
    points = points[keep]
    # Gross outliers in the temporal match and missing right views.
    corrupt = rng.random(len(points)) < outliers
    data["current_left"][corrupt] += rng.normal(scale=40.0, size=(int(corrupt.sum()), 2))
    data["current_right"][rng.random(len(points)) < 0.2] = np.nan
    # Initial stereo points: noisy, some unknown (start at infinity).
    initial_points = points * (1.0 + rng.normal(scale=0.02, size=(len(points), 1)))
    initial_points[rng.random(len(points)) < 0.1] = np.nan
    initial_points[np.isnan(data["previous_right"]).any(axis=1)] = np.nan
    data["points"] = initial_points
    data["initial_pose"] = left_update(motion, rng.normal(scale=[0.003] * 3 + [0.01] * 3))
    return rig, motion, data


def _problem(data):
    count = len(data["previous_left"])
    landmarks = np.arange(count)
    views = [View("cam0", "cam0", True, landmarks, data["current_left"])]
    for current, key in ((False, "previous_right"), (True, "current_right")):
        rows = np.flatnonzero(np.isfinite(data[key]).all(axis=1))
        views.append(View("cam1", "cam0", current, rows, data[key][rows]))
    return build_problem(["cam0"], [data["previous_left"]], [data["points"]], views)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_motion_with_outliers(seed):
    rig, motion, data = _scenario(seed)
    solution = refine(rig, _problem(data), data["initial_pose"], RefinementOptions())
    assert solution is not None
    error = se3_log(invert(solution.pose) @ motion)
    assert np.linalg.norm(error[3:]) < 1e-2
    assert np.linalg.norm(error[:3]) < np.radians(0.1)
    assert solution.partner_observations > 0
    values = np.linalg.eigvalsh(solution.covariance)
    assert values[0] > 0.0


def test_inertial_factors_are_used():
    rig, motion, data = _scenario(3)
    rng = np.random.default_rng(3)
    prior = so3_exp(rng.normal(scale=2e-3, size=3)) @ motion[:3, :3]
    gravity_previous = np.array([0.1, -0.98, 0.05])
    gravity_current = motion[:3, :3] @ (gravity_previous / np.linalg.norm(gravity_previous))
    factors = InertialFactors(rotation_prior=prior,
                              rotation_covariance=np.diag([2e-5, 3e-5, 1e-5]),
                              gravity_previous=gravity_previous,
                              gravity_current=gravity_current, gravity_sigma_rad=0.05)
    visual = refine(rig, _problem(data), data["initial_pose"], RefinementOptions())
    fused = refine(rig, _problem(data), data["initial_pose"], RefinementOptions(), factors)
    assert len(fused.inertial_information_eigenvalues) == 6
    assert max(fused.inertial_information_eigenvalues) > 0.0
    # Extra information can only shrink the rotation uncertainty.
    assert np.trace(fused.covariance[:3, :3]) < np.trace(visual.covariance[:3, :3])


def test_anchor_corrections_denoise_the_anchors():
    """Estimated anchors move towards the noise-free projections."""
    from colmap_odometry.estimation import anchored

    rig, motion, data = _scenario(4, outliers=0.0, noise_px=1.0)
    problem = _problem(data)
    captured = {}
    original = anchored._Linearizer.evaluate

    def evaluate(self, pose, state, *args, **kwargs):
        captured["state"] = state
        return original(self, pose, state, *args, **kwargs)

    anchored._Linearizer.evaluate = evaluate
    try:
        solution = refine(rig, problem, data["initial_pose"], RefinementOptions())
    finally:
        anchored._Linearizer.evaluate = original
    assert solution is not None
    corrected = problem.anchor_pixels + captured["state"][:, :2]
    measured_error = np.linalg.norm(problem.anchor_pixels - data["clean_anchor"], axis=1)
    corrected_error = np.linalg.norm(corrected - data["clean_anchor"], axis=1)
    assert np.mean(corrected_error) < 0.9 * np.mean(measured_error)


def test_unprojectable_anchor_drops_only_its_landmark():
    """The legacy solver returned no solution when any anchor failed."""
    rig, motion, data = _scenario(5)
    data["previous_left"][0] = [-5000.0, -5000.0]  # outside the fisheye domain
    problem = _problem(data)
    assert not np.isfinite(rig.cameras["cam0"].bearings(problem.anchor_pixels[:1])).all()
    solution = refine(rig, problem, data["initial_pose"], RefinementOptions())
    assert solution is not None
    assert solution.landmarks <= len(problem) - 1
    assert np.linalg.norm(solution.pose[:3, 3] - motion[:3, 3]) < 0.01
