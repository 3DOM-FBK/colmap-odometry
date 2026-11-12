"""Batched calibrated camera operations over any COLMAP camera model.

Projection and unprojection are delegated to ``pycolmap.Camera``, so every
COLMAP model (including OpenCV fisheye) is supported.  pycolmap exposes no
projection Jacobian, hence the Jacobians below are central finite
differences evaluated on whole arrays.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pycolmap

# Points closer than this to the image plane are treated as invalid.
MIN_DEPTH = 1.0e-9


class Camera:
    """A named calibrated camera."""

    def __init__(self, name: str, model: pycolmap.Camera):
        self.name = name
        self.model = model

    @classmethod
    def from_config(cls, name: str, config: Mapping[str, Any]) -> "Camera":
        required = {"model", "width", "height", "params"}
        missing = required - set(config)
        if missing:
            raise ValueError(f"camera '{name}' is missing {sorted(missing)}")
        model = pycolmap.Camera(
            model=str(config["model"]),
            width=int(config["width"]),
            height=int(config["height"]),
            params=[float(value) for value in config["params"]],
        )
        return cls(name, model)

    @property
    def width(self) -> int:
        return int(self.model.width)

    @property
    def height(self) -> int:
        return int(self.model.height)

    @property
    def mean_focal_length(self) -> float:
        return float(self.model.mean_focal_length())

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        """Project camera-frame points; invalid rows become NaN."""
        points_camera = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        if len(points_camera) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pixels = np.asarray(self.model.img_from_cam(points_camera), dtype=np.float64)
        pixels = pixels.reshape(-1, 2)
        valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > MIN_DEPTH)
        valid &= np.isfinite(pixels).all(axis=1)
        if not np.all(valid):
            pixels = pixels.copy()
            pixels[~valid] = np.nan
        return pixels

    def project_with_jacobian(
        self,
        points_camera: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pixels ``(N,2)`` and d(pixel)/d(point) ``(N,2,3)``.

        Rows whose projection or Jacobian is not finite are NaN.
        """
        points_camera = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        base = self.project(points_camera)
        jacobian = np.empty((len(points_camera), 2, 3), dtype=np.float64)
        for axis in range(3):
            plus = points_camera.copy()
            minus = points_camera.copy()
            plus[:, axis] += step
            minus[:, axis] -= step
            jacobian[:, :, axis] = (self.project(plus) - self.project(minus)) / (2.0 * step)
        finite = np.isfinite(base).all(axis=1) & np.isfinite(jacobian).all(axis=(1, 2))
        if not np.all(finite):
            jacobian[~finite] = np.nan
        return base, jacobian

    def normalize(self, pixels: np.ndarray) -> np.ndarray:
        """Unproject pixels to normalized image coordinates ``(x/z, y/z)``."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if len(pixels) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(self.model.cam_from_img(pixels), dtype=np.float64).reshape(-1, 2)

    def bearings(self, pixels: np.ndarray, tolerance_px: float = 0.5) -> np.ndarray:
        """Unproject pixels to unit bearing vectors; invalid rows are NaN.

        COLMAP unprojects to normalized coordinates ``(x/z, y/z)``, which
        cannot represent rays at or beyond 90 degrees (wide fisheye corners)
        and may be inaccurate near the model's validity limit.  Rays that do
        not reproject to their pixel within ``tolerance_px`` are invalid.
        """
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        normalized = self.normalize(pixels)
        rays = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1)
        norms = np.linalg.norm(rays, axis=1)
        valid = np.isfinite(rays).all(axis=1) & (norms > 1.0e-12)
        rays = rays / np.maximum(norms[:, None], 1.0e-12)
        if np.any(valid):
            error = np.linalg.norm(self.project(rays[valid]) - pixels[valid], axis=1)
            valid[valid] = error <= tolerance_px
        if not np.all(valid):
            rays[~valid] = np.nan
        return rays

    def normalized_threshold(self, pixel_threshold: float) -> float:
        """Convert a pixel error threshold to normalized image units."""
        return float(self.model.cam_from_img_threshold(float(pixel_threshold)))
