"""Calibrated two-view verification of consecutive frames of one camera."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import Camera

if not hasattr(cv2, "USAC_MAGSAC"):
    raise ImportError("OpenCV >= 4.5 with USAC_MAGSAC is required")


@dataclass(frozen=True)
class EssentialOptions:
    max_error_px: float = 2.0
    confidence: float = 0.999
    max_iterations: int = 10000

    def opencv_arguments(self, threshold: float) -> dict:
        return {
            "cameraMatrix": np.eye(3),
            "method": cv2.USAC_MAGSAC,
            # OpenCV needs a probability strictly inside (0, 1).
            "prob": min(max(self.confidence, 1.0e-6), 1.0 - 1.0e-9),
            "threshold": threshold,
            "maxIters": max(int(self.max_iterations), 100),
        }


@dataclass
class TemporalVerification:
    inliers: np.ndarray
    degenerate: bool = False
    failure_reason: str = ""


def verify_essential(
    camera_previous: Camera,
    camera_current: Camera,
    keypoints_previous: np.ndarray,
    keypoints_current: np.ndarray,
    matches: np.ndarray,
    options: EssentialOptions,
) -> TemporalVerification:
    """Return the matches surviving calibrated essential-matrix MAGSAC.

    Pixels are undistorted to normalized coordinates first, so fisheye
    models are handled correctly; matches without a finite normalized
    coordinate (rays at or beyond 90 degrees) are rejected individually.
    Fewer than eight matches are returned unchanged.  An essential matrix is undefined for a stationary or purely
    rotating camera; OpenCV failures then yield an empty, ``degenerate``
    result and the caller relies on PnP with the raw candidates.
    """
    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    if len(matches) < 8:
        return TemporalVerification(matches)
    matches, normalized_previous, normalized_current = _normalized_matches(
        camera_previous, camera_current, keypoints_previous, keypoints_current, matches)
    if len(matches) < 8:
        return TemporalVerification(np.empty((0, 2), np.int64), failure_reason="nonfinite_bearing")
    threshold = max(camera_previous.normalized_threshold(options.max_error_px),
                    camera_current.normalized_threshold(options.max_error_px), 1.0e-6)
    try:
        _, mask = cv2.findEssentialMat(normalized_previous, normalized_current,
                                       **options.opencv_arguments(threshold))
    except cv2.error as error:
        return TemporalVerification(np.empty((0, 2), np.int64), degenerate=True,
                                    failure_reason=str(error).strip().splitlines()[-1])
    if mask is None:
        return TemporalVerification(np.empty((0, 2), np.int64), failure_reason="no_model")
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if len(mask) != len(matches):
        return TemporalVerification(np.empty((0, 2), np.int64), failure_reason="mask_size")
    return TemporalVerification(matches[mask])


def essential_rotation(
    camera: Camera,
    keypoints_previous: np.ndarray,
    keypoints_current: np.ndarray,
    matches: np.ndarray,
    options: EssentialOptions,
) -> tuple[np.ndarray, int] | None:
    """Estimate ``R_current_previous`` of one camera from calibrated bearings.

    Essential translation is scale-free and unused; the rotation initializes
    the metric refinement.  Returns ``(rotation, inlier_count)`` or None.
    """
    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    if len(matches) < 8:
        return None
    try:
        matches, previous, current = _normalized_matches(
            camera, camera, keypoints_previous, keypoints_current, matches)
        if len(matches) < 8:
            return None
        threshold = max(camera.normalized_threshold(options.max_error_px), 1.0e-6)
        essential, mask = cv2.findEssentialMat(previous, current,
                                               **options.opencv_arguments(threshold))
        if essential is None or mask is None:
            return None
        essential = np.asarray(essential, dtype=np.float64)
        if essential.ndim == 2 and essential.shape[1] == 3 and essential.shape[0] >= 3:
            essential = essential[:3, :3]
        if essential.shape != (3, 3):
            return None
        count, rotation, _, _ = cv2.recoverPose(
            essential, previous, current, cameraMatrix=np.eye(3),
            mask=np.asarray(mask, dtype=np.uint8))
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        if int(count) < 8 or not np.all(np.isfinite(rotation)) or np.linalg.det(rotation) <= 0.0:
            return None
        return rotation, int(count)
    except (cv2.error, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _normalized_matches(camera_previous: Camera, camera_current: Camera,
                        keypoints_previous: np.ndarray, keypoints_current: np.ndarray,
                        matches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized coordinates of both ends; matches without finite ones are dropped."""
    previous = camera_previous.normalize(np.asarray(keypoints_previous)[matches[:, 0], :2])
    current = camera_current.normalize(np.asarray(keypoints_current)[matches[:, 1], :2])
    finite = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
    return matches[finite], previous[finite], current[finite]
