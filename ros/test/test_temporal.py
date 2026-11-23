"""Calibrated two-view verification of consecutive frames."""

from __future__ import annotations

import numpy as np

from colmap_odometry.vision.temporal import EssentialOptions, essential_rotation, verify_essential
from colmap_odometry.geometry.lie import so3_log
from synthetic import observe, random_motion, scene_points, stereo_rig


def test_unprojectable_keypoint_is_rejected_alone():
    """The legacy check discarded every match when one ray was not finite."""
    rng = np.random.default_rng(0)
    rig = stereo_rig()
    camera = rig.cameras["cam0"]
    motion = random_motion(rng)
    points = scene_points(rng, 300)
    previous = observe(rig, "cam0", np.eye(4), points, rng, 0.3)
    current = observe(rig, "cam0", motion, points, rng, 0.3)
    keep = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
    previous, current = previous[keep], current[keep]
    previous[0] = [-5000.0, -5000.0]  # outside the fisheye domain
    assert not np.isfinite(camera.normalize(previous[:1])).all()
    matches = np.repeat(np.arange(len(previous))[:, None], 2, axis=1)

    check = verify_essential(camera, camera, previous, current, matches, EssentialOptions())
    assert check.failure_reason == ""
    assert 0 not in check.inliers[:, 0]
    assert len(check.inliers) > 0.9 * (len(matches) - 1)

    rotation, inliers = essential_rotation(camera, previous, current, matches, EssentialOptions())
    assert inliers >= 8
    assert np.linalg.norm(so3_log(rotation @ motion[:3, :3].T)) < np.radians(0.1)
