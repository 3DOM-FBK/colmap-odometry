"""Synthetic multi-camera scenes for tests."""

from __future__ import annotations

import numpy as np

from colmap_odometry.geometry import Rig
from colmap_odometry.geometry.lie import make_transform, so3_exp

FISHEYE = {
    "model": "OPENCV_FISHEYE", "width": 1024, "height": 1024,
    "params": [380.8, 380.8, 510.3, 514.3, 0.0102, -0.0108, 0.0059, -0.0017],
}
PINHOLE = {
    "model": "PINHOLE", "width": 800, "height": 600,
    "params": [500.0, 500.0, 400.0, 300.0],
}


def make_rig(
    extrinsics: dict[str, np.ndarray],
    intrinsics: dict[str, dict] | None = None,
    stereo_pairs=(("cam0", "cam1"),),
) -> Rig:
    from colmap_odometry.geometry import Camera

    intrinsics = intrinsics or {}
    cameras = [Camera.from_config(name, intrinsics.get(name, FISHEYE)) for name in extrinsics]
    return Rig(cameras, extrinsics, stereo_pairs)


def stereo_rig(baseline: float = 0.1, cam0_offset: np.ndarray | None = None) -> Rig:
    cam0 = np.eye(4) if cam0_offset is None else cam0_offset
    cam1 = cam0 @ make_transform(so3_exp([0.0, -0.047, 0.0]), [baseline, 0.002, 0.001])
    return make_rig({"cam0": cam0, "cam1": cam1})


def random_motion(rng: np.random.Generator, rotation: float = 0.03, translation: float = 0.08):
    """Return ``T_rig(current)_rig(previous)``."""
    return make_transform(so3_exp(rng.normal(scale=rotation, size=3)),
                          rng.normal(scale=translation, size=3))


def scene_points(rng: np.random.Generator, count: int, near: float = 1.0, far: float = 12.0):
    """Points in the previous rig frame, roughly in front of +z."""
    depth = rng.uniform(near, far, count)
    lateral = rng.uniform(-0.8, 0.8, (count, 2)) * depth[:, None]
    return np.column_stack((lateral, depth))


def observe(rig: Rig, camera: str, rig_from_world: np.ndarray, points: np.ndarray,
            rng: np.random.Generator | None = None, noise_px: float = 0.0) -> np.ndarray:
    """Pixels of world points (in the previous rig frame) in a camera."""
    camera_points = (points @ (rig.camera_from_rig(camera) @ rig_from_world)[:3, :3].T
                     + (rig.camera_from_rig(camera) @ rig_from_world)[:3, 3])
    pixels = rig.cameras[camera].project(camera_points)
    width, height = rig.cameras[camera].width, rig.cameras[camera].height
    inside = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height) & (camera_points[:, 2] > 0.1))
    if rng is not None and noise_px > 0.0:
        pixels = pixels + rng.normal(scale=noise_px, size=pixels.shape)
    pixels[~inside] = np.nan
    return pixels
