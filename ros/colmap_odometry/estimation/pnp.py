"""Robust absolute pose of the rig from previous-frame stereo points.

With correspondences in one camera the COLMAP P3P LO-RANSAC solver is used;
with several cameras, COLMAP's generalized (multi-camera) solver.  Both are
followed by non-linear refinement.  Poses are ``T_rig(current)_rig(previous)``
with left-local ``[rotation, translation]`` covariances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap

from ..geometry import Rig
from ..geometry.lie import adjoint, positive_definite_or_none, transport


@dataclass(frozen=True)
class PnPOptions:
    ransac_max_error_px: float = 3.0
    ransac_max_trials: int = 10000
    min_correspondences: int = 6


@dataclass
class Correspondences:
    """2-D current-frame observations of 3-D previous-rig-frame points."""

    cameras: list[str]        # rig camera of each row
    points2d: np.ndarray      # (M, 2) pixels in the current frame
    points3d: np.ndarray      # (M, 3) points in the previous rig frame

    def __len__(self) -> int:
        return int(len(self.points2d))

    @property
    def camera_set(self) -> list[str]:
        return sorted(set(self.cameras), key=self.cameras.index)


@dataclass
class PoseHypothesis:
    pose: np.ndarray
    covariance: np.ndarray | None
    inliers: np.ndarray
    inlier_count: int
    median_error_px: float


def _rigid(transform: np.ndarray) -> pycolmap.Rigid3d:
    return pycolmap.Rigid3d(np.asarray(transform, dtype=np.float64)[:3, :])


def _matrix(rigid: pycolmap.Rigid3d) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :] = np.asarray(rigid.matrix(), dtype=np.float64)
    return transform


def reprojection_errors(rig: Rig, data: Correspondences, pose: np.ndarray) -> np.ndarray:
    """Per-row reprojection error in pixels (inf where not projectable)."""
    errors = np.full(len(data), np.inf)
    cameras = np.asarray(data.cameras, dtype=object)
    for name in data.camera_set:
        rows = cameras == name
        camera_from_previous = rig.camera_from_rig(name) @ pose
        points = data.points3d[rows] @ camera_from_previous[:3, :3].T + camera_from_previous[:3, 3]
        projected = rig.cameras[name].project(points)
        error = np.linalg.norm(projected - data.points2d[rows], axis=1)
        errors[rows] = np.where(np.isfinite(error), error, np.inf)
    return errors


def evaluate(rig: Rig, data: Correspondences, pose: np.ndarray, max_error_px: float):
    """Return ``(inlier_count, median_inlier_error, inlier_mask)``."""
    errors = reprojection_errors(rig, data, pose)
    inliers = errors <= max_error_px
    median = float(np.median(errors[inliers])) if np.any(inliers) else float("inf")
    return int(np.count_nonzero(inliers)), median, inliers


def _hypothesis(rig, data, pose, covariance, max_error_px) -> PoseHypothesis:
    count, median, inliers = evaluate(rig, data, pose, max_error_px)
    if covariance is not None:
        covariance = positive_definite_or_none(covariance)
    return PoseHypothesis(pose, covariance, inliers, count, median)


def _camera_problem(rig: Rig, data: Correspondences):
    names = data.camera_set
    indices = [names.index(name) for name in data.cameras]
    return (names, indices, [_rigid(rig.camera_from_rig(name)) for name in names],
            [rig.cameras[name].model for name in names])


def estimate(rig: Rig, data: Correspondences, options: PnPOptions) -> PoseHypothesis | None:
    """LO-RANSAC absolute pose followed by non-linear refinement.

    One camera: COLMAP P3P LO-RANSAC.  Several cameras: P3P LO-RANSAC in
    the best-covered cameras seeds a generalized refinement over all
    correspondences (re-selecting inliers once); COLMAP's generalized
    RANSAC is two to three orders of magnitude slower and is only used when
    no single camera has enough correspondences.
    """
    if len(data) < options.min_correspondences:
        return None
    names = data.camera_set
    if len(names) == 1:
        return _estimate_single(rig, data, options)
    cameras = np.asarray(data.cameras, dtype=object)
    counts = sorted(((int(np.count_nonzero(cameras == name)), name) for name in names),
                    reverse=True)
    best = None
    for count, name in counts[:2]:
        if count < options.min_correspondences:
            break
        rows = cameras == name
        seed = _estimate_single(rig, Correspondences([name] * count, data.points2d[rows],
                                                     data.points3d[rows]), options)
        if seed is None:
            continue
        hypothesis = seed.pose
        refined = None
        for _ in range(2):
            candidate = refine_seed(rig, data, hypothesis, options.ransac_max_error_px,
                                    options.min_correspondences, options.ransac_max_error_px)
            if candidate is None:
                break
            refined, hypothesis = candidate, candidate.pose
        if refined is not None and (best is None or refined.inlier_count > best.inlier_count):
            best = refined
    if best is not None:
        return best
    _, indices, cams_from_rig, cameras_list = _camera_problem(rig, data)
    ransac = pycolmap.RANSACOptions()
    ransac.max_error = float(options.ransac_max_error_px)
    ransac.max_num_trials = int(options.ransac_max_trials)
    ransac.min_num_trials = min(100, int(options.ransac_max_trials))
    result = pycolmap.estimate_and_refine_generalized_absolute_pose(
        data.points2d, data.points3d, indices, cams_from_rig, cameras_list,
        estimation_options=ransac, return_covariance=True)
    if result is None:
        return None
    return _hypothesis(rig, data, _matrix(result["rig_from_world"]), result.get("covariance"),
                       options.ransac_max_error_px)


def _estimate_single(rig: Rig, data: Correspondences, options: PnPOptions
                     ) -> PoseHypothesis | None:
    name = data.cameras[0]
    estimation = pycolmap.AbsolutePoseEstimationOptions()
    estimation.ransac.max_error = float(options.ransac_max_error_px)
    estimation.ransac.max_num_trials = int(options.ransac_max_trials)
    estimation.ransac.min_num_trials = min(int(estimation.ransac.min_num_trials),
                                           int(options.ransac_max_trials))
    result = pycolmap.estimate_and_refine_absolute_pose(
        points2D=data.points2d, points3D=data.points3d, camera=rig.cameras[name].model,
        estimation_options=estimation, return_covariance=True)
    if result is None:
        return None
    rig_from_camera = rig.rig_from_camera(name)
    pose = rig_from_camera @ _matrix(result["cam_from_world"])
    covariance = result.get("covariance")
    if covariance is not None:
        # Left perturbation of T_cam_world maps to T_rig_world via Ad(T_rig_cam).
        covariance = transport(covariance, adjoint(rig_from_camera))
    return _hypothesis(rig, data, pose, covariance, options.ransac_max_error_px)


def refine_seed(
    rig: Rig,
    data: Correspondences,
    seed: np.ndarray,
    max_error_px: float,
    min_inliers: int,
    selection_error_px: float,
) -> PoseHypothesis | None:
    """Refine a predicted pose (e.g. from the IMU) on its inliers.

    The seed is only the initial iterate; the refinement is purely visual.
    The returned hypothesis is scored with ``selection_error_px`` so it is
    comparable with :func:`estimate`.
    """
    min_inliers = max(6, int(min_inliers))
    count, _, inliers = evaluate(rig, data, seed, max_error_px)
    if count < min_inliers:
        return None
    try:
        names, indices, cams_from_rig, cameras = _camera_problem(rig, data)
        mask = inliers.reshape(-1, 1)
        if len(names) == 1:
            name = names[0]
            result = pycolmap.refine_absolute_pose(
                cam_from_world=_rigid(rig.camera_from_rig(name) @ seed),
                points2D=data.points2d, points3D=data.points3d, inlier_mask=mask,
                camera=rig.cameras[name].model, return_covariance=True)
            if result is None:
                return None
            rig_from_camera = rig.rig_from_camera(name)
            pose = rig_from_camera @ _matrix(result["cam_from_world"])
            covariance = result.get("covariance")
            if covariance is not None:
                covariance = transport(covariance, adjoint(rig_from_camera))
        else:
            result = pycolmap.refine_generalized_absolute_pose(
                _rigid(seed), data.points2d, data.points3d, mask, indices,
                cams_from_rig, cameras, return_covariance=True)
            if result is None:
                return None
            pose = _matrix(result["rig_from_world"])
            covariance = result.get("covariance")
    except (RuntimeError, ValueError, TypeError):
        return None
    refined = _hypothesis(rig, data, pose, covariance, max_error_px)
    if refined.inlier_count < min_inliers:
        return None
    return _hypothesis(rig, data, pose, covariance, selection_error_px)
