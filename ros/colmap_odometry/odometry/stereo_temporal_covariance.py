"""Covariance of a stereo-supported temporal PnP measurement.

The usual COLMAP PnP covariance treats the 3-D points supplied to PnP as
exact.  In the minimal stereo odometer those points are triangulated from the
previous stereo pair, so this assumption is particularly optimistic for
translation.  This module keeps one small landmark block per correspondence,
adds the previous stereo reprojection equations as a landmark prior, and
eliminates the landmark blocks with a Schur complement.

The returned covariance has the same local coordinates as COLMAP's
``cam_from_world`` pose covariance: ``[rotation, translation]``.  The caller
is responsible for the subsequent inverse/base-frame transport used by the
published relative pose.

All Jacobians are numerical.  That is intentional here: ``pycolmap.Camera``
supports several camera models (including OpenCV fisheye), while exposing no
projection Jacobian.  The projection calls are batched over all points, so the
extra work is modest compared with feature extraction and matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoTemporalCovariance:
    """Result and diagnostics of :func:`estimate_stereo_temporal_covariance`."""

    covariance: np.ndarray
    used_points: int
    rank: int
    condition_number: float
    reprojection_rms_px: float
    source: str = "stereo_schur"


@dataclass(frozen=True)
class StereoTemporalSolve:
    """Per-frame structureless stereo pose refinement result.

    The pose is the current cam0-from-previous-rig transform.  Landmark
    coordinates are temporary nuisance variables: they are updated inside
    each Gauss--Newton step and eliminated from the pose normal equations via
    a Schur complement.  No landmark or pose survives beyond this frame pair.
    """

    pose: np.ndarray
    covariance: np.ndarray
    used_points: int
    right_observations: int
    iterations: int
    initial_cost: float
    final_cost: float
    rank: int
    condition_number: float
    reprojection_rms_px: float
    source: str = "stereo_structureless_schur"
    # Detailed diagnostics; these do not participate in the solve.
    information_eigenvalues: tuple[float, ...] = ()
    visual_information_eigenvalues: tuple[float, ...] = ()
    inertial_information_eigenvalues: tuple[float, ...] = ()
    view_residual_rms_px: tuple[float, ...] = ()
    robust_weight_stats: tuple[float, ...] = ()
    # The covariance starts at the model-based Schur-complement result.  A
    # positive-semidefinite spatial-cluster excess is added when supported by
    # the final residual scores, so clean frames never report zero variance.
    covariance_estimator: str = "model_hessian"
    view_residual_scales: tuple[float, ...] = ()


def _transform_matrix(transform: Any) -> np.ndarray:
    """Convert a pycolmap ``Rigid3d`` or a 4x4 matrix to a matrix."""
    try:
        matrix = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError):
        matrix = None
    if matrix is not None and matrix.shape == (4, 4):
        return matrix.copy()
    if hasattr(transform, "rotation") and hasattr(transform, "translation"):
        quaternion = np.asarray(transform.rotation.quat, dtype=np.float64)
        if quaternion.shape != (4,):
            raise ValueError("Rigid3d rotation must contain four quaternion values")
        x, y, z, w = quaternion
        # Quaternion to rotation matrix, avoiding a dependency on pyquaternion.
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            raise ValueError("Rigid3d rotation quaternion is invalid")
        x, y, z, w = quaternion / norm
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
                 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
                 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
                 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        matrix[:3, 3] = np.asarray(transform.translation, dtype=np.float64)
        return matrix
    raise TypeError("transform must be a 4x4 matrix or pycolmap Rigid3d")


def _project(
    camera: Any,
    cam_from_world: np.ndarray,
    points_world: np.ndarray,
) -> np.ndarray:
    """Project world points using one camera pose."""
    points_camera = points_world @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
    pixels = np.asarray(camera.img_from_cam(points_camera), dtype=np.float64)
    pixels = pixels.reshape(-1, 2)
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1.0e-9)
    if not np.all(valid):
        pixels = pixels.copy()
        pixels[~valid] = np.nan
    return pixels


def _point_jacobian(
    camera: Any,
    cam_from_world: np.ndarray,
    points_world: np.ndarray,
    step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference projection Jacobian wrt each world point.

    Returns ``(projection, jacobian)`` with shapes ``(N,2)`` and ``(N,2,3)``.
    Invalid perturbed projections are marked in the Jacobian by NaNs and are
    removed by the caller.
    """
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    base = _project(camera, cam_from_world, points_world)
    jacobian = np.empty((len(points_world), 2, 3), dtype=np.float64)
    for axis in range(3):
        plus_points = points_world.copy()
        minus_points = points_world.copy()
        plus_points[:, axis] += step
        minus_points[:, axis] -= step
        plus = _project(camera, cam_from_world, plus_points)
        minus = _project(camera, cam_from_world, minus_points)
        jacobian[:, :, axis] = (plus - minus) / (2.0 * step)
    finite = np.isfinite(base).all(axis=1)
    finite &= np.isfinite(jacobian).all(axis=(1, 2))
    return base, jacobian if np.all(finite) else np.where(
        finite[:, None, None], jacobian, np.nan)


def _pose_jacobian(
    camera: Any,
    cam_from_world: np.ndarray,
    points_world: np.ndarray,
    rotation_step: float,
    translation_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference projection Jacobian in COLMAP pose coordinates.

    COLMAP's pose covariance is local to ``T_camera_world``.  Rotation is
    perturbed by left multiplication and translation is perturbed additively,
    matching the historical fallback covariance in ``odometry.py``.
    """
    base = _project(camera, cam_from_world, points_world)
    jacobian = np.empty((len(points_world), 2, 6), dtype=np.float64)
    rotation = cam_from_world[:3, :3]
    translation = cam_from_world[:3, 3]
    for column in range(6):
        if column < 3:
            axis = np.zeros(3, dtype=np.float64)
            axis[column] = rotation_step
            rotation_plus = cv2.Rodrigues(axis)[0] @ rotation
            rotation_minus = cv2.Rodrigues(-axis)[0] @ rotation
            plus_transform = cam_from_world.copy()
            minus_transform = cam_from_world.copy()
            plus_transform[:3, :3] = rotation_plus
            minus_transform[:3, :3] = rotation_minus
            denominator = 2.0 * rotation_step
        else:
            axis = np.zeros(3, dtype=np.float64)
            axis[column - 3] = translation_step
            plus_transform = cam_from_world.copy()
            minus_transform = cam_from_world.copy()
            plus_transform[:3, 3] = translation + axis
            minus_transform[:3, 3] = translation - axis
            denominator = 2.0 * translation_step
        plus = _project(camera, plus_transform, points_world)
        minus = _project(camera, minus_transform, points_world)
        jacobian[:, :, column] = (plus - minus) / denominator
    finite = np.isfinite(base).all(axis=1)
    finite &= np.isfinite(jacobian).all(axis=(1, 2))
    return base, jacobian if np.all(finite) else np.where(
        finite[:, None, None], jacobian, np.nan)


def _composed_pose_jacobian(
    camera: Any,
    extrinsic: np.ndarray,
    pose: np.ndarray,
    points_world: np.ndarray,
    rotation_step: float,
    translation_step: float,
) -> np.ndarray:
    """Pose Jacobian when a right-camera transform is ``E @ T``.

    The optimized pose ``T`` is expressed in cam0 coordinates.  Perturbing
    the already-composed cam1 transform directly would use the wrong tangent
    basis and introduces a fixed-baseline rotation/translation bias.  Apply
    every perturbation to ``T`` first, then compose the rig extrinsic ``E``.
    """
    base_transform = extrinsic @ pose
    base = _project(camera, base_transform, points_world)
    jacobian = np.empty((len(points_world), 2, 6), dtype=np.float64)
    for column in range(6):
        axis = np.zeros(6, dtype=np.float64)
        step = rotation_step if column < 3 else translation_step
        axis[column] = step
        plus = _project(
            camera, extrinsic @ _left_se3_update(pose, axis), points_world)
        minus = _project(
            camera, extrinsic @ _left_se3_update(pose, -axis), points_world)
        jacobian[:, :, column] = (plus - minus) / (2.0 * step)
    finite = np.isfinite(base).all(axis=1)
    finite &= np.isfinite(jacobian).all(axis=(1, 2))
    return jacobian if np.all(finite) else np.where(
        finite[:, None, None], jacobian, np.nan)


def _left_se3_update(transform: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a small COLMAP-style left-local SE(3) perturbation."""
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    update = np.eye(4, dtype=np.float64)
    update[:3, :3] = cv2.Rodrigues(delta[:3])[0]
    update[:3, 3] = delta[3:]
    return update @ transform


def _se3_pose_jacobian(
    camera: Any,
    transform: np.ndarray,
    points_world: np.ndarray,
    rotation_step: float,
    translation_step: float,
) -> np.ndarray:
    """Finite-difference Jacobian matching :func:`_left_se3_update`."""
    base = _project(camera, transform, points_world)
    jacobian = np.empty((len(points_world), 2, 6), dtype=np.float64)
    for column in range(6):
        axis = np.zeros(6, dtype=np.float64)
        step = rotation_step if column < 3 else translation_step
        axis[column] = step
        plus = _project(camera, _left_se3_update(transform, axis), points_world)
        minus = _project(camera, _left_se3_update(transform, -axis), points_world)
        jacobian[:, :, column] = (plus - minus) / (2.0 * step)
    finite = np.isfinite(base).all(axis=1)
    finite &= np.isfinite(jacobian).all(axis=(1, 2))
    return jacobian if np.all(finite) else np.where(
        finite[:, None, None], jacobian, np.nan)


def _huber_weight(norm: float, delta: float) -> float:
    """Return an IRLS weight for one two-dimensional reprojection residual."""
    if not np.isfinite(norm) or norm <= delta or norm <= np.finfo(float).eps:
        return 1.0
    return float(delta / norm)


def _structureless_views(
    *,
    camera0: Any,
    camera1: Any,
    previous0: np.ndarray,
    previous1: np.ndarray,
    current0: np.ndarray,
    previous_left: np.ndarray,
    previous_right: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
    points: np.ndarray,
    point_step: float,
    rotation_step: float,
    translation_step: float,
    robust_delta_px: float,
    with_jacobians: bool,
):
    """Evaluate the temporary four-view landmarks for one pose iterate.

    The first two views are fixed in the previous rig frame.  The last one or
    two views depend on the current cam0-from-world pose.  ``current_right``
    uses NaNs to mark a feature for which the current stereo match was not
    accepted; such a point still contributes its previous stereo and current
    left observations.
    """
    # ``current0`` is cam0-from-current-world.  Express the current right
    # camera through the calibrated rig relation, which is valid even when a
    # caller supplies a rig file where cam0 is not exactly identity:
    #   T_cam1_world = T_cam1_rig T_rig_cam0 T_cam0_world.
    current1_extrinsic = previous1 @ np.linalg.inv(previous0)
    current1_transform = current1_extrinsic @ current0
    specs = [
        (camera0, previous0, previous_left, False, None),
        (camera1, previous1, previous_right, False, None),
        (camera0, current0, current_left, True, None),
        (camera1, current1_transform, current_right, True, current1_extrinsic),
    ]
    evaluated = []
    for camera, transform, observations, depends_on_pose, extrinsic in specs:
        if with_jacobians:
            projected, point_jacobian = _point_jacobian(
                camera, transform, points, point_step)
            if depends_on_pose:
                # ``transform`` is current0 for cam0 and E@current0 for
                # cam1.  The latter needs the composed tangent Jacobian.
                if extrinsic is None:
                    pose_jacobian = _se3_pose_jacobian(
                        camera,
                        transform,
                        points,
                        rotation_step,
                        translation_step,
                    )
                else:
                    pose_jacobian = _composed_pose_jacobian(
                        camera,
                        extrinsic,
                        current0,
                        points,
                        rotation_step,
                        translation_step,
                    )
            else:
                pose_jacobian = np.zeros(
                    (len(points), 2, 6), dtype=np.float64)
        else:
            projected = _project(camera, transform, points)
            point_jacobian = pose_jacobian = None
        evaluated.append((projected, observations, point_jacobian,
                          pose_jacobian, depends_on_pose))

    entries = []
    cost = 0.0
    residual_dimensions = 0
    for index in range(len(points)):
        point_entries = []
        point_cost = 0.0
        point_dimensions = 0
        point_observations = 0
        point_valid = True
        for projected, observations, point_jacobian, pose_jacobian, _ in evaluated:
            observation = observations[index]
            if not np.all(np.isfinite(observation)):
                # Only the current-right view is optional.  Previous stereo
                # and current-left are required to keep a metric point.
                continue
            prediction = projected[index]
            if (
                not np.all(np.isfinite(prediction))
                or (with_jacobians and (
                    not np.all(np.isfinite(point_jacobian[index]))
                    or not np.all(np.isfinite(pose_jacobian[index]))))
            ):
                point_valid = False
                break
            residual = prediction - observation
            norm = float(np.linalg.norm(residual))
            point_cost += 0.5 * (norm * norm if norm <= robust_delta_px else
                                 robust_delta_px * (norm - 0.5 * robust_delta_px))
            point_dimensions += 2
            point_observations += 1
            if with_jacobians:
                point_entries.append((residual, point_jacobian[index],
                                      pose_jacobian[index]))
        if not point_valid or point_observations < 3:
            continue
        cost += point_cost
        residual_dimensions += point_dimensions
        # Keep the source point index alongside the entries.  Some views may
        # become invalid after a trial update; preserving this index prevents
        # nuisance-point increments from being applied to a different feature.
        entries.append((index, point_entries))
    return cost, residual_dimensions, entries


def _optimize_stereo_temporal_pose_legacy(
    *,
    initial_pose: Any,
    camera0: Any,
    camera1: Any,
    cam0_from_previous_world: Any,
    cam1_from_previous_world: Any,
    previous_left_pixels: np.ndarray,
    previous_right_pixels: np.ndarray,
    current_left_pixels: np.ndarray,
    current_right_pixels: np.ndarray,
    points_world: np.ndarray,
    pixel_std_px: float = 1.0,
    min_points: int = 6,
    max_iterations: int = 4,
    robust_delta_px: float = 3.0,
    rotation_step: float = 1.0e-6,
    translation_step: float = 1.0e-5,
    point_step: float = 1.0e-5,
) -> Optional[StereoTemporalSolve]:
    """Refine one consecutive stereo pose with temporary structureless points.

    This is deliberately not a windowed bundle adjustment.  It solves one
    current relative pose and one temporary 3-D point block per match, then
    discards the points.  Previous-left/right and current-left observations
    are mandatory; current-right observations are included where available.
    The point blocks are Schur-eliminated, both for the update and for the
    returned metric covariance.
    """
    try:
        transform = _transform_matrix(initial_pose)
        previous0 = _transform_matrix(cam0_from_previous_world)
        previous1 = _transform_matrix(cam1_from_previous_world)
        previous_left = np.asarray(previous_left_pixels, dtype=np.float64).reshape(-1, 2)
        previous_right = np.asarray(previous_right_pixels, dtype=np.float64).reshape(-1, 2)
        current_left = np.asarray(current_left_pixels, dtype=np.float64).reshape(-1, 2)
        current_right = np.asarray(current_right_pixels, dtype=np.float64).reshape(-1, 2)
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3).copy()
    except (TypeError, ValueError, AttributeError):
        return None

    count = len(points)
    if (
        not np.isfinite(pixel_std_px) or pixel_std_px <= 0.0
        or count < max(6, int(min_points))
        or len(previous_left) != count
        or len(previous_right) != count
        or len(current_left) != count
        or len(current_right) != count
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(previous_left))
        or not np.all(np.isfinite(previous_right))
        or not np.all(np.isfinite(current_left))
    ):
        return None

    # The normalized residuals use the same pixel noise as the covariance
    # implementation.  A mild Huber IRLS pass protects the four-view solve
    # from a bad right-image match without overriding MAGSAC's inlier set.
    try:
        initial_cost, _, _ = _structureless_views(
            camera0=camera0, camera1=camera1, previous0=previous0,
            previous1=previous1, current0=transform,
            previous_left=previous_left, previous_right=previous_right,
            current_left=current_left, current_right=current_right,
            points=points, point_step=point_step,
            rotation_step=rotation_step, translation_step=translation_step,
            robust_delta_px=robust_delta_px,
            with_jacobians=False)
    except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(initial_cost):
        return None

    final_cost = initial_cost
    accepted_iterations = 0
    for _ in range(max(1, int(max_iterations))):
        try:
            _, _, entries = _structureless_views(
                camera0=camera0, camera1=camera1, previous0=previous0,
                previous1=previous1, current0=transform,
                previous_left=previous_left, previous_right=previous_right,
                current_left=current_left, current_right=current_right,
                points=points, point_step=point_step,
                rotation_step=rotation_step, translation_step=translation_step,
                robust_delta_px=robust_delta_px,
                with_jacobians=True)
        except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
            return None
        if len(entries) < max(6, int(min_points)):
            return None

        h_pose = np.zeros((6, 6), dtype=np.float64)
        g_pose = np.zeros(6, dtype=np.float64)
        reduced_blocks = []
        for _, point_entries in entries:
            h_xx = np.zeros((3, 3), dtype=np.float64)
            h_px = np.zeros((6, 3), dtype=np.float64)
            h_pp = np.zeros((6, 6), dtype=np.float64)
            g_x = np.zeros(3, dtype=np.float64)
            g_p = np.zeros(6, dtype=np.float64)
            for residual, jacobian_x, jacobian_p in point_entries:
                weight = _huber_weight(
                    float(np.linalg.norm(residual)),
                    max(float(robust_delta_px), 1.0e-6),
                ) / float(pixel_std_px * pixel_std_px)
                h_xx += weight * (jacobian_x.T @ jacobian_x)
                h_px += weight * (jacobian_p.T @ jacobian_x)
                h_pp += weight * (jacobian_p.T @ jacobian_p)
                g_x += weight * (jacobian_x.T @ residual)
                g_p += weight * (jacobian_p.T @ residual)
            inverse_x = _safe_inverse(h_xx)
            if inverse_x is None:
                # A point with an unobservable 3-D block cannot contribute a
                # consistent pose update.  Reject the whole frame solve so
                # the caller can retain the robust COLMAP PnP result.
                return None
            reduced_h = h_pp - h_px @ inverse_x @ h_px.T
            reduced_g = g_p - h_px @ inverse_x @ g_x
            h_pose += reduced_h
            g_pose += reduced_g
            reduced_blocks.append((inverse_x, h_px, g_x, h_xx, g_p))

        if len(reduced_blocks) < max(6, int(min_points)):
            return None
        h_pose = 0.5 * (h_pose + h_pose.T)
        try:
            eigvals = np.linalg.eigvalsh(h_pose)
        except np.linalg.LinAlgError:
            return None
        if (
            not np.all(np.isfinite(eigvals))
            or eigvals[-1] <= 1.0e-12
            or eigvals[0] < -1.0e-8 * eigvals[-1]
        ):
            return None
        damping = max(1.0e-10, eigvals[-1] * 1.0e-8)
        try:
            delta_pose = np.linalg.solve(
                h_pose + damping * np.eye(6), -g_pose)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta_pose)):
            return None

        # Recover the nuisance-point steps associated with the pose step.
        delta_points = []
        for inverse_x, h_px, g_x, _, _ in reduced_blocks:
            delta_points.append(-inverse_x @ (g_x + h_px.T @ delta_pose))
        if len(delta_points) != len(entries):
            # This should not happen because the Schur blocks and entries are
            # built together; retain the conservative frame-level fallback if
            # an unexpected numerical path ever breaks that invariant.
            return None

        improved = False
        best_pose = transform
        best_points = points
        best_cost = final_cost
        for alpha in (1.0, 0.5, 0.25, 0.125):
            candidate_pose = _left_se3_update(transform, alpha * delta_pose)
            candidate_points = points.copy()
            for block_index, delta_point in enumerate(delta_points):
                point_index = entries[block_index][0]
                candidate_points[point_index] += alpha * delta_point
            try:
                candidate_cost, _, _ = _structureless_views(
                    camera0=camera0, camera1=camera1, previous0=previous0,
                    previous1=previous1, current0=candidate_pose,
                    previous_left=previous_left, previous_right=previous_right,
                    current_left=current_left, current_right=current_right,
                    points=candidate_points, point_step=point_step,
                    rotation_step=rotation_step,
                    translation_step=translation_step,
                    robust_delta_px=robust_delta_px,
                    with_jacobians=False)
            except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(candidate_cost) and candidate_cost < best_cost:
                best_pose = candidate_pose
                best_points = candidate_points
                best_cost = float(candidate_cost)
                improved = True
                break
        if not improved:
            break
        transform = best_pose
        points = best_points
        final_cost = best_cost
        accepted_iterations += 1
        if np.linalg.norm(delta_pose) < 1.0e-7:
            break

    # Rebuild the final marginalized information matrix for the covariance.
    try:
        _, residual_dimensions, entries = _structureless_views(
            camera0=camera0, camera1=camera1, previous0=previous0,
            previous1=previous1, current0=transform,
            previous_left=previous_left, previous_right=previous_right,
            current_left=current_left, current_right=current_right,
            points=points, point_step=point_step,
            rotation_step=rotation_step, translation_step=translation_step,
            robust_delta_px=robust_delta_px,
            with_jacobians=True)
    except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    h_pose = np.zeros((6, 6), dtype=np.float64)
    valid_points = 0
    right_observations = 0
    for _, point_entries in entries:
        h_xx = np.zeros((3, 3), dtype=np.float64)
        h_px = np.zeros((6, 3), dtype=np.float64)
        h_pp = np.zeros((6, 6), dtype=np.float64)
        for residual, jacobian_x, jacobian_p in point_entries:
            if np.linalg.norm(residual) <= robust_delta_px:
                weight = 1.0
            else:
                weight = _huber_weight(
                    float(np.linalg.norm(residual)), robust_delta_px)
            weight /= float(pixel_std_px * pixel_std_px)
            h_xx += weight * (jacobian_x.T @ jacobian_x)
            h_px += weight * (jacobian_p.T @ jacobian_x)
            h_pp += weight * (jacobian_p.T @ jacobian_p)
        inverse_x = _safe_inverse(h_xx)
        if inverse_x is None:
            return None
        h_pose += h_pp - h_px @ inverse_x @ h_px.T
        valid_points += 1
        if len(point_entries) == 4:
            right_observations += 1
    if valid_points < max(6, int(min_points)):
        return None
    h_pose = 0.5 * (h_pose + h_pose.T)
    try:
        information_values, information_vectors = np.linalg.eigh(h_pose)
    except np.linalg.LinAlgError:
        return None
    maximum = float(information_values[-1])
    if (
        not np.isfinite(maximum) or maximum <= 1.0e-12
        or information_values[0] < -1.0e-8 * maximum
    ):
        return None
    information_floor = max(1.0e-12, maximum * 1.0e-12)
    covariance = information_vectors @ np.diag(
        1.0 / np.maximum(information_values, information_floor)
    ) @ information_vectors.T
    degrees_of_freedom = max(1, residual_dimensions - 3 * valid_points - 6)
    residual_scale = max(
        float(pixel_std_px * pixel_std_px),
        2.0 * float(final_cost) / float(degrees_of_freedom),
    )
    covariance = 0.5 * (covariance * residual_scale
                        + (covariance * residual_scale).T)
    covariance = _safe_covariance(covariance)
    if covariance is None:
        return None
    positive = information_values[information_values > information_floor]
    condition = float("inf") if len(positive) == 0 else float(
        maximum / max(float(positive[0]), information_floor))
    rms = float(np.sqrt(max(0.0, 2.0 * final_cost)
                       / max(1, residual_dimensions)))
    return StereoTemporalSolve(
        pose=transform,
        covariance=covariance,
        used_points=int(valid_points),
        right_observations=int(right_observations),
        iterations=int(accepted_iterations),
        initial_cost=float(initial_cost),
        final_cost=float(final_cost),
        rank=int(np.count_nonzero(information_values > maximum * 1.0e-8)),
        condition_number=condition,
        reprojection_rms_px=rms,
    )


def _safe_inverse(matrix: np.ndarray, relative_floor: float = 1.0e-12) -> Optional[np.ndarray]:
    """Invert a symmetric 3x3 block while retaining weak landmark axes."""
    matrix = 0.5 * (matrix + matrix.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
        return None
    if eigenvalues[0] < -1.0e-8 * eigenvalues[-1]:
        return None
    floor = max(1.0e-12, eigenvalues[-1] * relative_floor)
    return eigenvectors @ np.diag(1.0 / np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _safe_covariance(matrix: np.ndarray) -> Optional[np.ndarray]:
    """Symmetrize and validate a covariance produced by Schur elimination."""
    matrix = 0.5 * (np.asarray(matrix, dtype=np.float64)
                    + np.asarray(matrix, dtype=np.float64).T)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        return None
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return None
    if (
        not np.all(np.isfinite(eigenvalues))
        or eigenvalues[-1] <= 0.0
        or eigenvalues[0] < -1.0e-8 * eigenvalues[-1]
    ):
        return None
    floor = max(1.0e-15, eigenvalues[-1] * 1.0e-12)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def estimate_stereo_temporal_covariance(
    *,
    pose: Any,
    camera0: Any,
    camera1: Any,
    cam0_from_previous_world: Any,
    cam1_from_previous_world: Any,
    previous_left_pixels: np.ndarray,
    previous_right_pixels: np.ndarray,
    current_pixels: np.ndarray,
    points_world: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    pixel_std_px: float = 1.0,
    min_points: int = 6,
    rotation_step: float = 1.0e-6,
    translation_step: float = 1.0e-5,
    point_step: float = 1.0e-5,
) -> Optional[StereoTemporalCovariance]:
    """Estimate pose covariance while marginalizing stereo landmark depth.

    The previous stereo pair supplies a 4-pixel landmark prior and the current
    left image supplies the 2-pixel temporal PnP observation.  For each point,
    the landmark block is eliminated independently; summing the resulting
    Schur complements produces the 6x6 pose information matrix.

    ``points_world`` must be expressed in the same world frame as the two
    previous camera transforms and ``pose``.  The pose may be a pycolmap
    ``Rigid3d`` or a 4x4 ``cam_from_world`` matrix.
    """
    try:
        if not np.isfinite(pixel_std_px) or pixel_std_px <= 0.0:
            return None
        current_transform = _transform_matrix(pose)
        previous0 = _transform_matrix(cam0_from_previous_world)
        previous1 = _transform_matrix(cam1_from_previous_world)
        previous_left = np.asarray(previous_left_pixels, dtype=np.float64).reshape(-1, 2)
        previous_right = np.asarray(previous_right_pixels, dtype=np.float64).reshape(-1, 2)
        current = np.asarray(current_pixels, dtype=np.float64).reshape(-1, 2)
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError, AttributeError):
        return None

    count = len(points)
    if (
        count < min_points
        or len(previous_left) != count
        or len(previous_right) != count
        or len(current) != count
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(previous_left))
        or not np.all(np.isfinite(previous_right))
        or not np.all(np.isfinite(current))
    ):
        return None
    if inlier_mask is not None:
        mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if len(mask) != count:
            return None
        if np.count_nonzero(mask) < min_points:
            return None
        previous_left = previous_left[mask]
        previous_right = previous_right[mask]
        current = current[mask]
        points = points[mask]
        count = len(points)

    try:
        projected_previous_left, jacobian_previous_left = _point_jacobian(
            camera0, previous0, points, point_step)
        projected_previous_right, jacobian_previous_right = _point_jacobian(
            camera1, previous1, points, point_step)
        projected_current, jacobian_current_point = _point_jacobian(
            camera0, current_transform, points, point_step)
        _, jacobian_current_pose = _pose_jacobian(
            camera0,
            current_transform,
            points,
            rotation_step,
            translation_step,
        )
    except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None

    residual_stereo = np.concatenate(
        (projected_previous_left - previous_left,
         projected_previous_right - previous_right),
        axis=1,
    )
    residual_current = projected_current - current
    finite = (
        np.isfinite(residual_stereo).all(axis=1)
        & np.isfinite(residual_current).all(axis=1)
        & np.isfinite(jacobian_previous_left).all(axis=(1, 2))
        & np.isfinite(jacobian_previous_right).all(axis=(1, 2))
        & np.isfinite(jacobian_current_point).all(axis=(1, 2))
        & np.isfinite(jacobian_current_pose).all(axis=(1, 2))
    )
    if np.count_nonzero(finite) < min_points:
        return None

    residual_stereo = residual_stereo[finite]
    residual_current = residual_current[finite]
    jacobian_stereo = np.concatenate(
        (jacobian_previous_left[finite], jacobian_previous_right[finite]),
        axis=1,
    )
    jacobian_current_point = jacobian_current_point[finite]
    jacobian_current_pose = jacobian_current_pose[finite]

    # Each pixel has the same configured independent noise.  The final
    # residual scale is estimated below and multiplies the information inverse.
    pixel_information = 1.0 / float(pixel_std_px * pixel_std_px)
    h_pose = np.einsum(
        "npi,npj->nij", jacobian_current_pose, jacobian_current_pose)
    h_pose_point = np.einsum(
        "npi,npj->nij", jacobian_current_pose, jacobian_current_point)
    h_point = np.einsum(
        "npi,npj->nij", jacobian_current_point, jacobian_current_point)
    h_point += np.einsum(
        "npi,npj->nij", jacobian_stereo, jacobian_stereo)
    h_pose *= pixel_information
    h_pose_point *= pixel_information
    h_point *= pixel_information

    reduced_information = np.zeros((6, 6), dtype=np.float64)
    valid_blocks = 0
    for index in range(len(h_point)):
        inverse_point = _safe_inverse(h_point[index])
        if inverse_point is None or not np.all(np.isfinite(inverse_point)):
            continue
        reduced_information += (
            h_pose[index]
            - h_pose_point[index] @ inverse_point @ h_pose_point[index].T
        )
        valid_blocks += 1
    if valid_blocks < min_points:
        return None
    reduced_information = 0.5 * (reduced_information + reduced_information.T)
    if not np.all(np.isfinite(reduced_information)):
        return None
    try:
        information_values, information_vectors = np.linalg.eigh(
            reduced_information)
    except np.linalg.LinAlgError:
        return None
    maximum = float(information_values[-1])
    if not np.isfinite(maximum) or maximum <= 1.0e-12:
        return None
    if information_values[0] < -1.0e-8 * maximum:
        return None
    information_floor = max(1.0e-12, maximum * 1.0e-12)
    covariance = information_vectors @ np.diag(
        1.0 / np.maximum(information_values, information_floor)
    ) @ information_vectors.T

    residual_count = residual_stereo.size + residual_current.size
    # Three residual degrees of freedom remain per point after eliminating its
    # 3-D landmark; six pose parameters are estimated globally.
    degrees_of_freedom = max(1, 3 * valid_blocks - 6)
    residual_scale = float(
        np.sum(residual_stereo * residual_stereo)
        + np.sum(residual_current * residual_current)
    ) / float(degrees_of_freedom)
    covariance *= max(pixel_std_px * pixel_std_px, residual_scale)
    covariance = 0.5 * (covariance + covariance.T)
    covariance_values = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(covariance_values)) or covariance_values[-1] <= 0.0:
        return None
    positive_information = information_values[information_values > information_floor]
    rank = int(np.count_nonzero(information_values > maximum * 1.0e-8))
    condition = float("inf") if len(positive_information) == 0 else float(
        maximum / max(float(positive_information[0]), information_floor))
    rms = float(np.sqrt(
        (np.sum(residual_stereo * residual_stereo)
         + np.sum(residual_current * residual_current))
        / max(1, residual_count)
    ))
    return StereoTemporalCovariance(
        covariance=covariance,
        used_points=int(valid_blocks),
        rank=rank,
        condition_number=condition,
        reprojection_rms_px=rms,
    )


# ---------------------------------------------------------------------------
# Consecutive-frame anchored inverse-depth solver
# ---------------------------------------------------------------------------

def _camera_ray(camera: Any, pixel: np.ndarray) -> np.ndarray | None:
    """Return a unit bearing from a calibrated pixel for any PyCOLMAP model."""
    try:
        normalized = np.asarray(
            camera.cam_from_img(np.asarray(pixel, dtype=np.float64).reshape(1, 2)),
            dtype=np.float64,
        ).reshape(2)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None
    if not np.all(np.isfinite(normalized)):
        return None
    ray = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(ray))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None
    return ray / norm


def _camera_rays(camera: Any, pixels: np.ndarray) -> np.ndarray | None:
    """Unproject a whole pixel array into unit camera-frame bearings."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    try:
        normalized = np.asarray(camera.cam_from_img(pixels), dtype=np.float64)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None
    if normalized.shape != (len(pixels), 2):
        try:
            normalized = normalized.reshape(len(pixels), 2)
        except ValueError:
            return None
    rays = np.concatenate(
        (normalized, np.ones((len(pixels), 1), dtype=np.float64)), axis=1)
    norms = np.linalg.norm(rays, axis=1)
    valid = np.isfinite(rays).all(axis=1) & (norms > 1.0e-12)
    rays = rays / np.maximum(norms[:, None], 1.0e-12)
    if not np.all(valid):
        rays = rays.copy()
        rays[~valid] = np.nan
    return rays


def _project_vectors(camera: Any, vectors: np.ndarray) -> np.ndarray:
    """Project a batch of camera-frame vectors with calibrated PyCOLMAP."""
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    try:
        pixels = np.asarray(camera.img_from_cam(vectors), dtype=np.float64)
        pixels = pixels.reshape(-1, 2)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return np.full((len(vectors), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(vectors).all(axis=1) & (vectors[:, 2] > 1.0e-9)
    valid &= np.isfinite(pixels).all(axis=1)
    if not np.all(valid):
        pixels = pixels.copy()
        pixels[~valid] = np.nan
    return pixels


def _projection_vector_jacobian(
    camera: Any,
    vectors: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return batched projection and d(pixel)/d(camera-vector Jacobians."""
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    base = _project_vectors(camera, vectors)
    jacobian = np.empty((len(vectors), 2, 3), dtype=np.float64)
    for axis in range(3):
        plus = vectors.copy()
        minus = vectors.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        jacobian[:, :, axis] = (
            _project_vectors(camera, plus)
            - _project_vectors(camera, minus)
        ) / (2.0 * step)
    finite = np.isfinite(base).all(axis=1)
    finite &= np.isfinite(jacobian).all(axis=(1, 2))
    if not np.all(finite):
        jacobian = jacobian.copy()
        jacobian[~finite] = np.nan
    return base, jacobian


def _camera_model_anchor_sigmas(
    camera0: Any,
    output_camera: Any,
    pixels: np.ndarray,
    rays: np.ndarray,
    pixel_std_px: float,
) -> np.ndarray:
    """Propagate anchor-pixel noise through calibrated unprojection.

    The result is a scalar equivalent pixel sigma per observation.  The
    derivative is evaluated for the complete batch, preserving the camera
    model while avoiding one PyCOLMAP call per feature and perturbation.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    step = max(float(pixel_std_px), 1.0e-3)
    ray_jacobian = np.empty((len(pixels), 3, 2), dtype=np.float64)
    for axis in range(2):
        plus_pixels = pixels.copy()
        minus_pixels = pixels.copy()
        plus_pixels[:, axis] += step
        minus_pixels[:, axis] -= step
        plus_rays = _camera_rays(camera0, plus_pixels)
        minus_rays = _camera_rays(camera0, minus_pixels)
        if plus_rays is None or minus_rays is None:
            ray_jacobian[:, :, axis] = np.nan
        else:
            ray_jacobian[:, :, axis] = (
                plus_rays - minus_rays) / (2.0 * step)
    try:
        focal = getattr(output_camera, "mean_focal_length", None)
        focal = focal() if callable(focal) else focal
        if focal is None:
            focal = getattr(output_camera, "focal_length", 1.0)
        focal = max(float(focal), 1.0)
    except (TypeError, ValueError, AttributeError):
        focal = 1.0
    angular_variance = 0.5 * np.sum(ray_jacobian * ray_jacobian, axis=(1, 2))
    angular_variance[~np.isfinite(angular_variance)] = 0.0
    sigma = np.sqrt(np.maximum(
        float(pixel_std_px) ** 2,
        float(pixel_std_px) ** 2
        + focal * focal * angular_variance * float(pixel_std_px) ** 2,
    ))
    sigma[~np.isfinite(sigma) | (sigma <= 0.0)] = float(pixel_std_px)
    return sigma


def _skew_batch(vectors: np.ndarray) -> np.ndarray:
    """Return [v]x for a batch of 3-vectors."""
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    skew = np.zeros((len(vectors), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def _anchored_project(
    camera0: Any,
    camera1: Any,
    relative: np.ndarray,
    rig_right_from_left: np.ndarray,
    ray: np.ndarray,
    inverse_depth: float,
    view: str,
) -> np.ndarray | None:
    """Project an anchored ray in the requested previous/current view.

    ``relative`` maps the previous cam0 frame to the current cam0 frame.
    A point is represented homogeneously as ``ray + rho * translation``;
    therefore rho=0 is a valid point-at-infinity limit and does not make the
    projection singular.
    """
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    rig_rotation = rig_right_from_left[:3, :3]
    rig_translation = rig_right_from_left[:3, 3]
    rho = float(inverse_depth)
    if view == "previous_right":
        vector = rig_rotation @ ray + rho * rig_translation
        camera = camera1
    elif view == "current_left":
        vector = rotation @ ray + rho * translation
        camera = camera0
    elif view == "current_right":
        current_left = rotation @ ray + rho * translation
        vector = rig_rotation @ current_left + rho * rig_translation
        camera = camera1
    else:
        raise ValueError(f"unknown anchored view {view}")
    if not np.all(np.isfinite(vector)) or vector[2] <= 1.0e-9:
        return None
    try:
        pixel = np.asarray(
            camera.img_from_cam(vector.reshape(1, 3)), dtype=np.float64
        ).reshape(2)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None
    return pixel if np.all(np.isfinite(pixel)) else None


def _anchored_view_sigma(
    camera0: Any,
    camera1: Any,
    relative: np.ndarray,
    rig_right_from_left: np.ndarray,
    ray: np.ndarray,
    inverse_depth: float,
    view: str,
    pixel_std_px: float,
) -> float:
    """Propagate anchor-pixel uncertainty through the camera model.

    The previous-left pixel is the inverse-depth anchor.  Perturbing it through
    ``cam_from_img`` and the complete projection model captures fisheye/model
    unprojection sensitivity.  The observed pixel contributes the independent
    ``pixel_std_px`` term.
    """
    base = _anchored_project(
        camera0, camera1, relative, rig_right_from_left, ray,
        inverse_depth, view)
    if base is None:
        return float(pixel_std_px)
    # The local focal scale is already contained in this numerical derivative;
    # using a unit pixel perturbation avoids assuming a pinhole model.
    # ``ray`` is perturbed by finite differences in normalized image space.
    # The caller supplies the anchor ray, so estimate its local pixel
    # sensitivity by a small angular perturbation in each tangent direction.
    tangent_u = np.array([1.0, 0.0, -ray[0] / max(ray[2], 1.0e-9)])
    tangent_v = np.array([0.0, 1.0, -ray[1] / max(ray[2], 1.0e-9)])
    tangents = (tangent_u, tangent_v)
    jacobian = np.zeros((2, 2), dtype=np.float64)
    angular_step = 1.0e-6
    for column, tangent in enumerate(tangents):
        plus_ray = ray + angular_step * tangent
        minus_ray = ray - angular_step * tangent
        plus_ray /= max(float(np.linalg.norm(plus_ray)), 1.0e-12)
        minus_ray /= max(float(np.linalg.norm(minus_ray)), 1.0e-12)
        plus = _anchored_project(
            camera0, camera1, relative, rig_right_from_left,
            plus_ray, inverse_depth, view)
        minus = _anchored_project(
            camera0, camera1, relative, rig_right_from_left,
            minus_ray, inverse_depth, view)
        if plus is None or minus is None:
            continue
        jacobian[:, column] = (plus - minus) / (2.0 * angular_step)
    # Convert the camera-model angular sensitivity to an equivalent pixel
    # variance.  This is deliberately scalar so the existing IRLS/Schur code
    # remains inexpensive while retaining model-dependent uncertainty.
    try:
        focal = getattr(camera0 if view == "current_left" else camera1,
                         "mean_focal_length", None)
        focal = focal() if callable(focal) else focal
        if focal is None:
            focal = getattr(camera0 if view == "current_left" else camera1,
                             "focal_length", 1.0)
        focal = max(float(focal), 1.0)
    except (TypeError, ValueError, AttributeError):
        focal = 1.0
    # ``jacobian`` is pixels/radian, while the anchor noise is in pixels.
    # Convert the latter to an angular perturbation before propagating it;
    # omitting this conversion would inflate every sigma by roughly f.
    anchor_angular_variance = (float(pixel_std_px) / focal) ** 2
    model_variance = (
        0.5 * float(np.sum(jacobian * jacobian))
        * anchor_angular_variance
    )
    if not np.isfinite(model_variance):
        model_variance = 0.0
    return float(np.sqrt(max(
        pixel_std_px * pixel_std_px,
        pixel_std_px * pixel_std_px * (1.0 + model_variance),
    )))


def _unprojection_pixel_sigma(
    camera0: Any,
    output_camera: Any,
    pixel: np.ndarray,
    pixel_std_px: float,
) -> float:
    """Estimate output-pixel noise after calibrated anchor unprojection."""
    pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
    base = _camera_ray(camera0, pixel)
    if base is None:
        return float(pixel_std_px)
    step = max(float(pixel_std_px), 1.0e-3)
    jacobian = np.zeros((3, 2), dtype=np.float64)
    for column in range(2):
        plus_pixel = pixel.copy()
        minus_pixel = pixel.copy()
        plus_pixel[column] += step
        minus_pixel[column] -= step
        plus = _camera_ray(camera0, plus_pixel)
        minus = _camera_ray(camera0, minus_pixel)
        if plus is None or minus is None:
            continue
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    try:
        focal = getattr(output_camera, "mean_focal_length", None)
        focal = focal() if callable(focal) else focal
        if focal is None:
            focal = getattr(output_camera, "focal_length", 1.0)
        focal = max(float(focal), 1.0)
    except (TypeError, ValueError, AttributeError):
        focal = 1.0
    angular_variance = 0.5 * float(np.sum(jacobian * jacobian))
    if not np.isfinite(angular_variance):
        angular_variance = 0.0
    return float(np.sqrt(max(
        pixel_std_px * pixel_std_px,
        pixel_std_px * pixel_std_px
        + focal * focal * angular_variance * pixel_std_px * pixel_std_px,
    )))


def _anchored_linearization(
    *,
    camera0: Any,
    camera1: Any,
    previous_inverse: np.ndarray,
    rig_right_from_left: np.ndarray,
    current0: np.ndarray,
    rays: np.ndarray,
    previous_right: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
    inverse_depths: np.ndarray,
    pixel_std_px: float,
    robust_delta_px: float,
    projection_step: float,
    view_sigmas: dict[str, np.ndarray],
    spatial_clusters: int,
    view_variance_scales: dict[str, float] | None = None,
    with_jacobians: bool,
    compute_sandwich: bool = False,
):
    """Evaluate all anchored residuals in batches.

    The previous-left pixels have already been unprojected into ``rays``.
    Camera projection remains numerically model-agnostic, but all numerical
    differences are performed on whole arrays.  Pose and inverse-depth
    derivatives are analytic through the rigid-body geometry and are chained
    with the batched camera projection Jacobians.
    """
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    inverse_depths = np.asarray(inverse_depths, dtype=np.float64).reshape(-1)
    previous_right = np.asarray(previous_right, dtype=np.float64).reshape(-1, 2)
    current_left = np.asarray(current_left, dtype=np.float64).reshape(-1, 2)
    current_right = np.asarray(current_right, dtype=np.float64).reshape(-1, 2)
    count = len(rays)
    relative = current0 @ previous_inverse
    relative_rotation = relative[:3, :3]
    relative_translation = relative[:3, 3]
    rig_rotation = rig_right_from_left[:3, :3]
    rig_translation = rig_right_from_left[:3, 3]

    current_left_vectors = (
        rays @ relative_rotation.T
        + inverse_depths[:, None] * relative_translation
    )
    previous_right_vectors = (
        rays @ rig_rotation.T
        + inverse_depths[:, None] * rig_translation
    )
    current_right_vectors = (
        current_left_vectors @ rig_rotation.T
        + inverse_depths[:, None] * rig_translation
    )
    # Process current-left first so its validity mask is established before
    # optional right-camera residuals contribute cost or covariance terms.
    vector_data = {
        "current_left": (camera0, current_left_vectors, current_left),
        "previous_right": (camera1, previous_right_vectors, previous_right),
        "current_right": (camera1, current_right_vectors, current_right),
    }
    predictions: dict[str, np.ndarray] = {}
    projection_jacobians: dict[str, np.ndarray] = {}
    if with_jacobians:
        for view, (camera, vectors, _) in vector_data.items():
            prediction, projection_jacobian = _projection_vector_jacobian(
                camera, vectors, float(projection_step))
            predictions[view] = prediction
            projection_jacobians[view] = projection_jacobian
    else:
        for view, (camera, vectors, _) in vector_data.items():
            predictions[view] = _project_vectors(camera, vectors)

    # A feature remains usable only if its anchor ray and current-left
    # reprojection are valid.  Right-camera observations are optional.
    valid_feature = np.isfinite(rays).all(axis=1)
    valid_feature &= np.isfinite(predictions["current_left"]).all(axis=1)
    if count == 0:
        valid_feature[:] = False

    h_rr = np.zeros(count, dtype=np.float64)
    h_pr = np.zeros((count, 6), dtype=np.float64)
    h_pp = np.zeros((count, 6, 6), dtype=np.float64)
    g_r = np.zeros(count, dtype=np.float64)
    g_p = np.zeros((count, 6), dtype=np.float64)
    residual_square_sum = 0.0
    residual_count = 0
    dimensions = 0
    right_observations = 0
    cost = np.zeros(count, dtype=np.float64)
    view_residual_square_sum = {
        view: 0.0 for view in ("previous_right", "current_left", "current_right")
    }
    view_residual_count = {
        view: 0 for view in ("previous_right", "current_left", "current_right")
    }
    robust_weights = []
    # Per-view score contributions are retained until the inverse-depth
    # blocks have been Schur-eliminated below.  They let the covariance use a
    # clustered sandwich estimator instead of assuming every learned feature
    # is an independent draw.  The cluster is an image tile, which captures
    # local calibration/texture bias without treating an entire frame as one
    # observation.
    # Score contributions are needed only for the final empirical covariance.
    # Keeping them out of Gauss--Newton iterations avoids allocating and
    # populating several O(number_of_features) arrays for every trial.
    view_pose_scores = ({
        view: np.zeros((count, 6), dtype=np.float64)
        for view in vector_data
    } if with_jacobians and compute_sandwich else {})
    view_cross_hessians = ({
        view: np.zeros((count, 6), dtype=np.float64)
        for view in vector_data
    } if with_jacobians and compute_sandwich else {})
    view_dispersion = (
        {view: [] for view in vector_data}
        if with_jacobians and compute_sandwich else {})

    if with_jacobians:
        # Left-local perturbation of a homogeneous anchored vector y is
        # d y / d(delta) = [-y]x, rho*I.  The right camera simply rotates
        # this derivative by the calibrated rig baseline rotation.
        current_left_pose_vector_jacobian = np.zeros(
            (count, 3, 6), dtype=np.float64)
        current_left_pose_vector_jacobian[:, :, :3] = -_skew_batch(
            current_left_vectors)
        current_left_pose_vector_jacobian[:, :, 3:] = (
            inverse_depths[:, None, None]
            * np.eye(3, dtype=np.float64)[None, :, :]
        )
        current_right_pose_vector_jacobian = np.einsum(
            "ij,njk->nik", rig_rotation,
            current_left_pose_vector_jacobian)
        rho_vector_jacobians = {
            "previous_right": np.broadcast_to(
                rig_translation, (count, 3)).copy(),
            "current_left": np.broadcast_to(
                relative_translation, (count, 3)).copy(),
            "current_right": np.broadcast_to(
                rig_rotation @ relative_translation + rig_translation,
                (count, 3)).copy(),
        }
        pose_vector_jacobians = {
            "previous_right": np.zeros((count, 3, 6), dtype=np.float64),
            "current_left": current_left_pose_vector_jacobian,
            "current_right": current_right_pose_vector_jacobian,
        }
    for view, (camera, vectors, observations) in vector_data.items():
        observed = np.asarray(observations, dtype=np.float64)
        valid = valid_feature & np.isfinite(observed).all(axis=1)
        valid &= np.isfinite(predictions[view]).all(axis=1)
        projection_jacobian = None
        if with_jacobians:
            projection_jacobian = projection_jacobians[view]
            finite_jacobian = np.isfinite(
                projection_jacobian).all(axis=(1, 2))
            valid &= finite_jacobian
            if view == "current_left":
                valid_feature &= finite_jacobian
        if view == "current_left":
            # This mask is the anchor for the whole feature.  Optional right
            # views may disappear without invalidating the temporal bearing.
            valid_feature &= valid
        if not np.any(valid):
            continue
        sigma = np.asarray(view_sigmas[view], dtype=np.float64).reshape(-1)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma,
                         float(pixel_std_px))
        # Keep robust weights tied to the configured/model sigma.  The
        # empirical scale below is a variance component, applied to the
        # normal equations only during the final covariance relinearization.
        # This preserves the optimizer's IRLS weights and the calibrated
        # anchor/unprojection uncertainty already represented by ``sigma``.
        variance_scale = 1.0 if view_variance_scales is None else float(
            view_variance_scales.get(view, 1.0))
        if not np.isfinite(variance_scale) or variance_scale < 1.0:
            variance_scale = 1.0
        variance_weight = 1.0 / (variance_scale * variance_scale)
        raw_residual = predictions[view] - observed
        normalized_residual = raw_residual / sigma[:, None]
        norms = np.linalg.norm(normalized_residual, axis=1)
        delta = max(float(robust_delta_px), 1.0e-6)
        weights = np.ones(count, dtype=np.float64)
        outlier = norms > delta
        weights[outlier] = delta / np.maximum(norms[outlier], 1.0e-12)
        safe_norms = np.where(valid, norms, 0.0)
        cost += np.where(
            safe_norms <= delta,
            0.5 * safe_norms * safe_norms,
            delta * (safe_norms - 0.5 * delta),
        )
        dimensions += 2 * int(np.count_nonzero(valid))
        residual_square_sum += float(np.sum(
            np.sum(raw_residual[valid] * raw_residual[valid], axis=1)))
        residual_count += 2 * int(np.count_nonzero(valid))
        view_residual_square_sum[view] += float(np.sum(
            np.sum(raw_residual[valid] * raw_residual[valid], axis=1)))
        view_residual_count[view] += 2 * int(np.count_nonzero(valid))
        if np.any(valid):
            robust_weights.extend(weights[valid].tolist())
            if compute_sandwich:
                normalized_valid = normalized_residual[valid]
                weighted_squared = np.sum(
                    weights[valid, None] * normalized_valid * normalized_valid)
                view_dispersion[view].append((
                    float(weighted_squared),
                    int(2 * np.count_nonzero(valid)),
                    float(np.sum(weights[valid])),
                ))
        if view == "current_right":
            right_observations += int(np.count_nonzero(valid))
        if not with_jacobians:
            continue
        pose_jacobian = np.einsum(
            "nij,njk->nik", projection_jacobian,
            pose_vector_jacobians[view])
        rho_jacobian = np.einsum(
            "nij,nj->ni", projection_jacobian,
            rho_vector_jacobians[view])
        inv_sigma = 1.0 / sigma
        residual = normalized_residual
        pose_jacobian *= inv_sigma[:, None, None]
        rho_jacobian *= inv_sigma[:, None]
        # Mask before multiplying: IEEE ``0 * NaN`` is still NaN, and an
        # invalid optional right projection must not contaminate the Schur
        # normal equations for the remaining views.
        safe_weight = np.where(valid, weights, 0.0) * variance_weight
        safe_pose_jacobian = np.where(
            valid[:, None, None], pose_jacobian, 0.0)
        safe_rho_jacobian = np.where(
            valid[:, None], rho_jacobian, 0.0)
        safe_residual = np.where(valid[:, None], residual, 0.0)
        h_rr += safe_weight * np.sum(
            safe_rho_jacobian * safe_rho_jacobian, axis=1)
        h_pr += safe_weight[:, None] * np.einsum(
            "ni,nij->nj", safe_rho_jacobian, safe_pose_jacobian)
        h_pp += safe_weight[:, None, None] * np.einsum(
            "nij,nik->njk", safe_pose_jacobian, safe_pose_jacobian)
        g_r += safe_weight * np.sum(
            safe_rho_jacobian * safe_residual, axis=1)
        g_p += safe_weight[:, None] * np.einsum(
            "ni,nij->nj", safe_residual, safe_pose_jacobian)
        if compute_sandwich:
            view_pose_scores[view] += safe_weight[:, None] * np.einsum(
                "ni,nij->nj", safe_residual, safe_pose_jacobian)
            view_cross_hessians[view] += safe_weight[:, None] * np.einsum(
                "ni,nij->nj", safe_rho_jacobian, safe_pose_jacobian)

    valid_feature &= np.isfinite(current_left).all(axis=1)
    valid_count = int(np.count_nonzero(valid_feature))
    point_sandwich_covariance = None
    point_meat = None
    sandwich_covariance = None
    sandwich_meat = None
    covariance_estimator = "hessian_fallback"
    if with_jacobians:
        inverse_rr = np.zeros(count, dtype=np.float64)
        observable = h_rr > 1.0e-12
        inverse_rr[observable] = 1.0 / h_rr[observable]
        h_pose = np.sum(h_pp, axis=0)
        g_pose = np.sum(g_p, axis=0)
        h_pose -= np.einsum("ni,nj->ij", h_pr * inverse_rr[:, None], h_pr)
        g_pose -= np.sum(h_pr * (inverse_rr * g_r)[:, None], axis=0)
        if compute_sandwich:
            # Score after eliminating each landmark's nuisance inverse depth.
            # ``h_pose`` is the robust weighted bread; the outer products below
            # form the empirical meat.  Grouping by image tile preserves
            # common-mode residuals from texture/calibration regions.
            point_scores = g_p - h_pr * (inverse_rr * g_r)[:, None]
            point_meat = point_scores.T @ point_scores
            cluster_scores: dict[tuple[str, int], np.ndarray] = {}
            cluster_count = max(1, int(spatial_clusters))
            # Six columns and four rows are a useful default for the 24-cluster
            # setting.  For another count, choose a near-square grid while
            # keeping the same deterministic tile assignment.
            columns = max(1, int(np.ceil(np.sqrt(cluster_count * 1.5))))
            rows = max(1, int(np.ceil(cluster_count / columns)))
            for view, (_, _, observations) in vector_data.items():
                observed = np.asarray(observations, dtype=np.float64)
                finite_observed = np.isfinite(observed).all(axis=1)
                finite_observed &= np.isfinite(
                    view_pose_scores[view]).all(axis=1)
                if not np.any(finite_observed):
                    continue
                minimum = np.nanmin(observed[finite_observed], axis=0)
                maximum_observed = np.nanmax(observed[finite_observed], axis=0)
                extent = np.maximum(maximum_observed - minimum, 1.0)
                normalized = (observed - minimum) / extent
                tile_x = np.clip(
                    (normalized[:, 0] * columns).astype(np.int64), 0, columns - 1)
                tile_y = np.clip(
                    (normalized[:, 1] * rows).astype(np.int64), 0, rows - 1)
                score = (
                    view_pose_scores[view]
                    - view_cross_hessians[view]
                    * (inverse_rr * g_r)[:, None]
                )
                for tile in np.unique(tile_y[finite_observed] * columns
                                      + tile_x[finite_observed]):
                    selected = finite_observed & (
                        tile_y * columns + tile_x == tile)
                    cluster_scores[(view, int(tile))] = np.sum(
                        score[selected], axis=0)
            if len(cluster_scores) >= 4:
                cluster_meat = np.zeros((6, 6), dtype=np.float64)
                for score in cluster_scores.values():
                    cluster_meat += np.outer(score, score)
                # A small finite-cluster correction is standard for a clustered
                # covariance estimate.  It is deliberately bounded so a sparse
                # frame cannot manufacture an arbitrarily large covariance.
                clusters = float(len(cluster_scores))
                correction = max(1.0, clusters / max(1.0, clusters - 1.0))
                meat = correction * cluster_meat
                covariance_estimator = "spatial_cluster_sandwich"
            else:
                meat = point_meat
                covariance_estimator = "point_cluster_sandwich"
            try:
                bread = np.linalg.pinv(h_pose, hermitian=True)
                sandwich_meat = 0.5 * (meat + meat.T)
                sandwich_covariance = bread @ sandwich_meat @ bread.T
                point_sandwich_covariance = bread @ (
                    0.5 * (point_meat + point_meat.T)) @ bread.T
                if (
                    not np.all(np.isfinite(sandwich_covariance))
                    or not np.all(np.isfinite(point_sandwich_covariance))
                ):
                    sandwich_covariance = None
                    point_sandwich_covariance = None
                    sandwich_meat = None
            except np.linalg.LinAlgError:
                sandwich_covariance = None
                point_sandwich_covariance = None
                sandwich_meat = None
                covariance_estimator = "hessian_fallback"
    else:
        inverse_rr = None
        h_pose = None
        g_pose = None
        covariance_estimator = "hessian_fallback"
    return {
        "cost": float(np.sum(cost)),
        "dimensions": int(dimensions),
        "valid_count": valid_count,
        "h_pose": h_pose,
        "g_pose": g_pose,
        "inverse_rr": inverse_rr,
        "h_pr": h_pr if with_jacobians else None,
        "g_r": g_r if with_jacobians else None,
        "residual_square_sum": residual_square_sum,
        "residual_count": residual_count,
        "right_observations": right_observations,
        "view_residual_square_sum": view_residual_square_sum,
        "view_residual_count": view_residual_count,
        "robust_weight_stats": (
            float(np.min(robust_weights)) if robust_weights else 1.0,
            float(np.median(robust_weights)) if robust_weights else 1.0,
            float(np.max(robust_weights)) if robust_weights else 1.0,
        ),
        "sandwich_covariance": sandwich_covariance,
        "point_sandwich_covariance": point_sandwich_covariance,
        "point_meat": point_meat,
        "sandwich_meat": sandwich_meat,
        "covariance_estimator": covariance_estimator,
        "view_residual_scales": tuple(
            float(np.sqrt(max(
                1.0,
                sum(item[0] for item in values)
                / max(1.0, 2.0 * sum(item[2] for item in values)),
            )))
            if sum(item[1] for item in values) >= 4 else 1.0
            for view, values in view_dispersion.items()
        ),
    }


def _augment_with_inertial_factors(
    linearization: dict,
    pose: np.ndarray,
    rotation_prior: np.ndarray | None,
    rotation_covariance: np.ndarray | None,
    gravity_previous: np.ndarray | None,
    gravity_current: np.ndarray | None,
    gravity_sigma_rad: float | None,
    with_jacobians: bool,
) -> dict:
    """Add gyro rotation and gravity direction to one pose linearization."""
    result = dict(linearization)
    result["h_pose"] = (
        None if result["h_pose"] is None else result["h_pose"].copy())
    result["g_pose"] = (
        None if result["g_pose"] is None else result["g_pose"].copy())
    if result["h_pose"] is not None:
        result["visual_h_pose"] = result["h_pose"].copy()
    else:
        result["visual_h_pose"] = None
    rotation = pose[:3, :3]
    if rotation_prior is not None and rotation_covariance is not None:
        prior = np.asarray(rotation_prior, dtype=np.float64).reshape(3, 3)
        covariance = np.asarray(
            rotation_covariance, dtype=np.float64).reshape(3, 3)
        try:
            covariance = 0.5 * (covariance + covariance.T)
            values, vectors = np.linalg.eigh(covariance)
            if not np.all(np.isfinite(values)):
                raise np.linalg.LinAlgError
            information = vectors @ np.diag(
                1.0 / np.maximum(values, 1.0e-12)) @ vectors.T
            residual = cv2.Rodrigues(prior.T @ rotation)[0].reshape(3)
            if not np.all(np.isfinite(residual)):
                raise np.linalg.LinAlgError
            result["cost"] += float(0.5 * residual @ information @ residual)
            result["dimensions"] += 3
            if with_jacobians:
                jacobian = np.empty((3, 6), dtype=np.float64)
                jacobian[:, 3:] = 0.0
                step = 1.0e-6
                for axis in range(3):
                    perturbation = np.zeros(3, dtype=np.float64)
                    perturbation[axis] = step
                    plus = cv2.Rodrigues(
                        prior.T @ (cv2.Rodrigues(perturbation)[0] @ rotation)
                    )[0].reshape(3)
                    minus = cv2.Rodrigues(
                        prior.T @ (cv2.Rodrigues(-perturbation)[0] @ rotation)
                    )[0].reshape(3)
                    jacobian[:, axis] = (plus - minus) / (2.0 * step)
                result["h_pose"] += jacobian.T @ information @ jacobian
                result["g_pose"] += jacobian.T @ information @ residual
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            pass

    if (
        gravity_previous is not None
        and gravity_current is not None
        and gravity_sigma_rad is not None
        and np.isfinite(gravity_sigma_rad)
        and gravity_sigma_rad > 0.0
    ):
        previous = np.asarray(gravity_previous, dtype=np.float64).reshape(3)
        current = np.asarray(gravity_current, dtype=np.float64).reshape(3)
        previous_norm = float(np.linalg.norm(previous))
        current_norm = float(np.linalg.norm(current))
        if previous_norm > 1.0e-9 and current_norm > 1.0e-9:
            previous /= previous_norm
            current /= current_norm
            predicted = rotation @ previous
            # Cross product is a two-degree-of-freedom direction residual and
            # therefore cannot accidentally constrain yaw about gravity.
            residual = np.cross(predicted, current)
            information = 1.0 / float(gravity_sigma_rad * gravity_sigma_rad)
            result["cost"] += float(
                0.5 * information * residual @ residual)
            result["dimensions"] += 2
            if with_jacobians:
                jacobian = np.zeros((3, 6), dtype=np.float64)
                step = 1.0e-6
                for axis in range(3):
                    perturbation = np.zeros(3, dtype=np.float64)
                    perturbation[axis] = step
                    plus = np.cross(
                        cv2.Rodrigues(perturbation)[0] @ predicted, current)
                    minus = np.cross(
                        cv2.Rodrigues(-perturbation)[0] @ predicted, current)
                    jacobian[:, axis] = (plus - minus) / (2.0 * step)
                result["h_pose"] += information * (jacobian.T @ jacobian)
                result["g_pose"] += information * (jacobian.T @ residual)
    if result["h_pose"] is not None and result["visual_h_pose"] is not None:
        result["inertial_h_pose"] = (
            result["h_pose"] - result["visual_h_pose"])
    else:
        result["inertial_h_pose"] = None
    return result


def optimize_stereo_temporal_pose(
    *,
    initial_pose: Any,
    camera0: Any,
    camera1: Any,
    cam0_from_previous_world: Any,
    cam1_from_previous_world: Any,
    previous_left_pixels: np.ndarray,
    previous_right_pixels: np.ndarray,
    current_left_pixels: np.ndarray,
    current_right_pixels: np.ndarray,
    points_world: np.ndarray,
    pixel_std_px: float = 1.0,
    min_points: int = 6,
    max_iterations: int = 4,
    robust_delta_px: float = 3.0,
    covariance_spatial_clusters: int = 24,
    rotation_step: float = 1.0e-6,
    translation_step: float = 1.0e-5,
    point_step: float = 1.0e-5,
    inertial_rotation_prior: np.ndarray | None = None,
    inertial_rotation_covariance: np.ndarray | None = None,
    gravity_direction_previous: np.ndarray | None = None,
    gravity_direction_current: np.ndarray | None = None,
    gravity_sigma_rad: float | None = None,
) -> Optional[StereoTemporalSolve]:
    """Refine one frame with anchored inverse-depth temporal landmarks.

    Every temporal match contributes a calibrated current-left bearing.  A
    previous/current right pixel contributes an additional stereo reprojection
    whenever available.  Each point has one nuisance inverse depth, which is
    Schur-eliminated immediately; rho=0 naturally represents a point at
    infinity.  The returned covariance is formed from the same final residuals
    and robust weights used by the optimizer.
    """
    try:
        current0 = _transform_matrix(initial_pose)
        previous0 = _transform_matrix(cam0_from_previous_world)
        previous1 = _transform_matrix(cam1_from_previous_world)
        previous_left = np.asarray(previous_left_pixels, dtype=np.float64).reshape(-1, 2)
        previous_right = np.asarray(previous_right_pixels, dtype=np.float64).reshape(-1, 2)
        current_left = np.asarray(current_left_pixels, dtype=np.float64).reshape(-1, 2)
        current_right = np.asarray(current_right_pixels, dtype=np.float64).reshape(-1, 2)
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError, AttributeError):
        return None
    count = len(previous_left)
    if (
        not np.isfinite(pixel_std_px) or pixel_std_px <= 0.0
        or count < max(6, int(min_points))
        or len(previous_right) != count
        or len(current_left) != count
        or len(current_right) != count
        or len(points) != count
        or not np.all(np.isfinite(previous_left))
        or not np.all(np.isfinite(current_left))
    ):
        return None

    # Initialize each inverse depth from the previous stereo point where it is
    # finite; NaN rows are explicitly initialized at infinity.
    inverse_depths = np.zeros(count, dtype=np.float64)
    for index, point in enumerate(points):
        if not np.all(np.isfinite(point)):
            continue
        previous_point = (
            previous0[:3, :3] @ point + previous0[:3, 3])
        distance = float(np.linalg.norm(previous_point))
        if np.isfinite(distance) and distance > 1.0e-6:
            inverse_depths[index] = min(1.0e3, 1.0 / distance)

    previous_inverse = np.linalg.inv(previous0)
    rig_right_from_left = previous1 @ previous_inverse
    rays = _camera_rays(camera0, previous_left)
    if rays is None or not np.all(np.isfinite(rays)):
        return None

    # This uncertainty depends only on the anchor camera/model and the output
    # camera model, so compute it once per frame rather than inside every
    # Gauss--Newton trial.  The right-camera sigma is shared by both right
    # views; the previous implementation recomputed this identical array.
    right_sigma = _camera_model_anchor_sigmas(
        camera0, camera1, previous_left, rays, float(pixel_std_px))
    left_sigma = _camera_model_anchor_sigmas(
        camera0, camera0, previous_left, rays, float(pixel_std_px))
    view_sigmas = {
        "previous_right": right_sigma,
        "current_left": left_sigma,
        "current_right": right_sigma,
    }

    def evaluate(
        pose, depths, with_jacobians, compute_sandwich=False,
        view_variance_scales=None,
    ):
        visual = _anchored_linearization(
            camera0=camera0,
            camera1=camera1,
            previous_inverse=previous_inverse,
            rig_right_from_left=rig_right_from_left,
            current0=pose,
            rays=rays,
            previous_right=previous_right,
            current_left=current_left,
            current_right=current_right,
            inverse_depths=depths,
            pixel_std_px=float(pixel_std_px),
            robust_delta_px=float(robust_delta_px),
            projection_step=max(1.0e-6, float(point_step)),
            view_sigmas=view_sigmas,
            spatial_clusters=max(1, int(covariance_spatial_clusters)),
            view_variance_scales=view_variance_scales,
            with_jacobians=with_jacobians,
            compute_sandwich=compute_sandwich,
        )
        return _augment_with_inertial_factors(
            visual,
            pose,
            inertial_rotation_prior,
            inertial_rotation_covariance,
            gravity_direction_previous,
            gravity_direction_current,
            gravity_sigma_rad,
            with_jacobians,
        )

    initial = evaluate(current0, inverse_depths, False)
    if initial is None or not np.isfinite(initial["cost"]):
        return None
    initial_cost = float(initial["cost"])
    final_cost = initial_cost
    accepted_iterations = 0
    for _ in range(max(1, int(max_iterations))):
        # The empirical sandwich is a final-covariance diagnostic and is not
        # needed to solve a Gauss--Newton step.  Avoid its feature-score and
        # spatial-clustering allocations during every iteration.
        linearized = evaluate(current0, inverse_depths, True, False)
        if linearized is None:
            return None
        if linearized["valid_count"] < max(6, int(min_points)):
            return None
        h_pose = linearized["h_pose"]
        g_pose = linearized["g_pose"]
        inverse_rr = linearized["inverse_rr"]
        h_pr = linearized["h_pr"]
        g_r = linearized["g_r"]
        try:
            eigenvalues = np.linalg.eigvalsh(h_pose)
        except np.linalg.LinAlgError:
            return None
        maximum = float(eigenvalues[-1])
        if (
            not np.isfinite(maximum) or maximum <= 1.0e-12
            or eigenvalues[0] < -1.0e-8 * maximum
        ):
            return None
        damping = max(1.0e-10, maximum * 1.0e-8)
        try:
            delta_pose = np.linalg.solve(
                h_pose + damping * np.eye(6), -g_pose)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta_pose)):
            return None
        delta_depths = -inverse_rr * (
            g_r + np.einsum("ni,i->n", h_pr, delta_pose))

        improved = False
        best_pose = current0
        best_depths = inverse_depths
        best_cost = final_cost
        for alpha in (1.0, 0.5, 0.25, 0.125):
            candidate_pose = _left_se3_update(current0, alpha * delta_pose)
            candidate_depths = inverse_depths.copy()
            candidate_depths = np.clip(
                candidate_depths + alpha * delta_depths, 0.0, 1.0e3)
            candidate = evaluate(candidate_pose, candidate_depths, False)
            if (
                candidate is not None
                and np.isfinite(candidate["cost"])
                and candidate["valid_count"] >= max(6, int(min_points))
                and candidate["cost"] < best_cost
            ):
                best_pose = candidate_pose
                best_depths = candidate_depths
                best_cost = float(candidate["cost"])
                improved = True
                break
        if not improved:
            break
        current0 = best_pose
        inverse_depths = best_depths
        final_cost = best_cost
        accepted_iterations += 1
        if np.linalg.norm(delta_pose) < 1.0e-7:
            break

    final = evaluate(current0, inverse_depths, True, True)
    if final is None:
        return None
    # Estimate unexplained per-view dispersion from the final robust residuals
    # and relinearize once with those variance components.  The estimate is a
    # robust-weighted RMS of normalized residuals, floored at the configured
    # pixel/model variance; at least two observations (four scalar residuals)
    # are required, otherwise that view retains its model variance.  Applying
    # the factor inside this second Schur linearization keeps inverse-depth
    # elimination and camera-model anchor uncertainty consistent, rather than
    # multiplying a finished 6x6 covariance blindly.
    provisional_scales = tuple(float(value) for value in final.get(
        "view_residual_scales", ()))
    scale_names = ("previous_right", "current_left", "current_right")
    view_variance_scales = {
        name: max(1.0, provisional_scales[index])
        for index, name in enumerate(scale_names)
        if index < len(provisional_scales)
    }
    if view_variance_scales:
        final_relinearized = evaluate(
            current0, inverse_depths, True, True, view_variance_scales)
        if final_relinearized is not None:
            final = final_relinearized
    residual_dimensions = final["dimensions"]
    if final["valid_count"] < max(6, int(min_points)):
        return None
    h_pose = final["h_pose"]
    inverse_rr = final["inverse_rr"]
    observable_depths = int(np.count_nonzero(inverse_rr > 0.0))
    residual_square_sum = float(final["residual_square_sum"])
    residual_count = int(final["residual_count"])
    right_observations = int(final["right_observations"])
    view_residual_rms = tuple(
        float(np.sqrt(
            max(0.0, final["view_residual_square_sum"].get(view, 0.0))
            / max(1, int(final["view_residual_count"].get(view, 0)))))
        for view in ("previous_right", "current_left", "current_right")
    )
    robust_weight_stats = tuple(
        float(value) for value in final.get(
            "robust_weight_stats", (1.0, 1.0, 1.0)))
    try:
        information_values, information_vectors = np.linalg.eigh(h_pose)
    except np.linalg.LinAlgError:
        return None
    maximum = float(information_values[-1])
    if (
        not np.isfinite(maximum) or maximum <= 1.0e-12
        or information_values[0] < -1.0e-8 * maximum
    ):
        return None
    information_floor = max(1.0e-12, maximum * 1.0e-12)
    covariance = information_vectors @ np.diag(
        1.0 / np.maximum(information_values, information_floor)
    ) @ information_vectors.T
    # Model-based covariance from the final robust normal equations.  It
    # already includes calibrated pixel uncertainty, Schur-eliminated depth,
    # and any inertial/gravity factors.
    model_covariance = 0.5 * (covariance + covariance.T)
    degrees_of_freedom = max(
        1,
        int(residual_dimensions) - int(observable_depths) - 6,
    )
    # Use the same final residuals and robust weights as the optimizer to
    # estimate only *excess* uncertainty from spatially correlated errors.
    # The empirical meat can be arbitrarily small on a clean frame (or cancel
    # across features), so replacing the model covariance with a sandwich
    # would incorrectly report near-zero uncertainty.  Keep the model
    # covariance as the floor and add the positive-semidefinite difference
    # between clustered and independent empirical estimates.
    sandwich_meat = final.get("sandwich_meat")
    point_meat = final.get("point_meat")
    covariance_estimator = "model_hessian"
    if (
        sandwich_meat is not None
        and point_meat is not None
    ):
        # Recompute both empirical covariances with the full pose bread.  The
        # linearization helper sees only visual factors; inertial/gravity
        # factors are added immediately afterwards and must also attenuate
        # the correlated visual excess.
        try:
            bread = np.linalg.pinv(h_pose, hermitian=True)
            clustered = bread @ np.asarray(sandwich_meat, dtype=np.float64) @ bread.T
            independent = bread @ np.asarray(point_meat, dtype=np.float64) @ bread.T
        except np.linalg.LinAlgError:
            clustered = None
            independent = None
        if clustered is None or independent is None:
            covariance = model_covariance
            covariance_estimator = "model_hessian"
        else:
            excess = 0.5 * (clustered - independent +
                            (clustered - independent).T)
            try:
                excess_values, excess_vectors = np.linalg.eigh(excess)
                excess_values = np.maximum(excess_values, 0.0)
                excess = excess_vectors @ np.diag(excess_values) @ excess_vectors.T
                if np.all(np.isfinite(excess)):
                    covariance = model_covariance + excess
                    covariance_estimator = "model_plus_spatial_cluster_excess"
                else:
                    covariance = model_covariance
            except np.linalg.LinAlgError:
                covariance = model_covariance
    else:
        residual_scale = max(
            1.0,
            2.0 * float(final_cost) / float(degrees_of_freedom),
        )
        covariance = model_covariance * residual_scale
        covariance_estimator = "model_hessian_residual_scaled"
    covariance = _safe_covariance(covariance)
    if covariance is None:
        return None
    positive = information_values[information_values > information_floor]
    condition = float("inf") if len(positive) == 0 else float(
        maximum / max(float(positive[0]), information_floor))
    rms = float(np.sqrt(
        residual_square_sum / max(1, residual_count)))
    try:
        visual_information_eigenvalues = tuple(
            float(value) for value in np.linalg.eigvalsh(
                final["visual_h_pose"])) if final.get(
                    "visual_h_pose") is not None else ()
    except np.linalg.LinAlgError:
        visual_information_eigenvalues = ()
    try:
        inertial_information_eigenvalues = tuple(
            float(value) for value in np.linalg.eigvalsh(
                final["inertial_h_pose"])) if final.get(
                    "inertial_h_pose") is not None else ()
    except np.linalg.LinAlgError:
        inertial_information_eigenvalues = ()
    return StereoTemporalSolve(
        pose=current0,
        covariance=covariance,
        used_points=int(final["valid_count"]),
        right_observations=int(right_observations),
        iterations=int(accepted_iterations),
        initial_cost=float(initial_cost),
        final_cost=float(final_cost),
        rank=int(np.count_nonzero(information_values > maximum * 1.0e-8)),
        condition_number=condition,
        reprojection_rms_px=rms,
        source="anchored_inverse_depth_stereo_temporal",
        information_eigenvalues=tuple(float(value)
                                      for value in information_values),
        visual_information_eigenvalues=visual_information_eigenvalues,
        inertial_information_eigenvalues=inertial_information_eigenvalues,
        view_residual_rms_px=view_residual_rms,
        robust_weight_stats=robust_weight_stats,
        covariance_estimator=covariance_estimator,
        view_residual_scales=tuple(float(value) for value in final.get(
            "view_residual_scales", ())),
    )
