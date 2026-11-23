"""Consecutive-frame rig pose refinement with anchored inverse-depth landmarks.

Every temporal feature match defines a temporary landmark anchored at its
keypoint in the *host* camera at the previous frame.  The landmark has three
unknowns: a 2-D correction of the anchor pixel and the inverse depth
``rho`` along the bearing ``b`` of the corrected anchor, i.e. the point is
``b / rho``.  In any other view it projects from the homogeneous vector
``y = R b + rho t``, where ``(R, t)`` maps the host camera at the previous
frame into that view, so ``rho = 0`` (a point at infinity) is regular.

All observations are measurements with the a-priori pixel noise:

* the anchor itself (residual: the anchor correction),
* the host camera at the current frame (the temporal match, required),
* stereo partners at the previous frame (independent of the pose),
* stereo partners at the current frame.

Treating the anchor as a measurement models the noise it shares among all
views of a landmark; this is the maximum-likelihood formulation (Triggs et
al., "Bundle Adjustment - A Modern Synthesis", 2000; Civera et al., "Inverse
Depth Parametrization for Monocular SLAM", T-RO 2008).  Fixing the anchor
instead (the legacy model) makes the estimate biased (errors in variables)
and the covariance optimistic.

The unknown is ``T = T_rig(current)_rig(previous)``.  It is updated with the
left retraction :func:`~colmap_odometry.geometry.lie.left_update`, so the
returned covariance is a left-local ``[rotation, translation]`` covariance of
``T``.  Landmark unknowns are eliminated per landmark by a 3x3 Schur
complement; nothing survives beyond the frame pair (this is not a windowed
bundle adjustment).

Projection Jacobians are central finite differences on whole arrays because
pycolmap exposes no projection derivatives; the rigid-body derivatives are
analytic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..geometry import Rig
from ..geometry.lie import (
    inverse_with_floor,
    left_update,
    positive_definite_or_none,
    skew_batch,
)
from .inertial import InertialFactors

# Finite-difference step for projection Jacobians (camera-frame units).
_PROJECTION_STEP = 1.0e-5
# Inverse depths are kept within [0, MAX_INVERSE_DEPTH] (1 mm).
_MAX_INVERSE_DEPTH = 1.0e3
# Line-search step fractions.
_LINE_SEARCH = (1.0, 0.5, 0.25, 0.125)
# Finite-difference step (pixels) of the anchor unprojection Jacobian.
_ANCHOR_STEP = 1.0e-2


@dataclass(frozen=True)
class RefinementOptions:
    pixel_std_px: float = 1.0
    #: Huber threshold on the sigma-normalized 2-D residual norm.
    robust_delta: float = 3.0
    max_iterations: int = 4
    min_points: int = 6
    #: Approximate number of image tiles per image for the clustered
    #: (spatially correlated) covariance correction.
    spatial_clusters: int = 24
    #: Only temporal matches consistent with the PnP motion within
    #: ``inlier_gate`` x the PnP threshold are used (stereo landmarks by
    #: reprojection, bearing-only landmarks by epipolar error).
    inlier_gate: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.pixel_std_px) or self.pixel_std_px <= 0.0:
            raise ValueError("pixel_std_px must be positive")
        if self.robust_delta <= 0.0:
            raise ValueError("robust_delta must be positive")
        if self.inlier_gate <= 0.0:
            raise ValueError("refinement.inlier_gate must be positive")


    @property
    def minimum_landmarks(self) -> int:
        return max(6, int(self.min_points))


@dataclass
class View:
    """Observations of landmarks of one host in one (camera, frame)."""

    camera: str
    host: str
    current: bool
    landmarks: np.ndarray  # unique landmark indices
    pixels: np.ndarray     # (M, 2)

    @property
    def required(self) -> bool:
        return self.current and self.camera == self.host

    @property
    def image(self) -> tuple[str, bool]:
        """Image this view belongs to; residual statistics group by image."""
        return self.camera, self.current


@dataclass
class AnchoredProblem:
    """Temporary landmarks of one interval.

    ``hosts[i]`` names the camera of landmark ``i``; ``anchor_pixels`` are
    its previous-frame host pixels; ``initial_points_rig`` are previous-frame
    rig points (NaN rows start at infinity).  Views must contain exactly one
    required (host, current) view per host, listing every landmark of it.
    """

    hosts: np.ndarray
    anchor_pixels: np.ndarray
    initial_points_rig: np.ndarray
    views: list[View]

    def __len__(self) -> int:
        return int(len(self.anchor_pixels))

    def subset(self, keep: np.ndarray) -> "AnchoredProblem":
        """Problem restricted to landmarks where ``keep`` is true."""
        keep = np.asarray(keep, dtype=bool)
        new_index = np.full(len(keep), -1, dtype=np.int64)
        new_index[keep] = np.arange(int(keep.sum()))
        views = []
        for view in self.views:
            rows = keep[view.landmarks]
            if np.any(rows):
                views.append(View(view.camera, view.host, view.current,
                                  new_index[view.landmarks[rows]], view.pixels[rows]))
        return AnchoredProblem(self.hosts[keep], self.anchor_pixels[keep],
                               self.initial_points_rig[keep], views)


@dataclass(frozen=True)
class AnchoredSolution:
    pose: np.ndarray
    covariance: np.ndarray
    landmarks: int
    partner_observations: int
    iterations: int
    initial_cost: float
    final_cost: float
    rank: int
    condition_number: float
    reprojection_rms_px: float
    covariance_estimator: str
    information_eigenvalues: tuple[float, ...] = ()
    visual_information_eigenvalues: tuple[float, ...] = ()
    inertial_information_eigenvalues: tuple[float, ...] = ()
    image_residual_rms_px: dict[str, float] = field(default_factory=dict)
    image_variance_scales: dict[str, float] = field(default_factory=dict)
    robust_weight_stats: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: With inertial factors: the visual-only estimate and its covariance,
    #: recovered from the same final linearization (one Gauss-Newton step
    #: on the visual terms alone).
    visual_pose: np.ndarray | None = None
    visual_covariance: np.ndarray | None = None


def _image_label(image: tuple[str, bool]) -> str:
    return f"{image[0]}@{'current' if image[1] else 'previous'}"


class _Linearizer:
    """Evaluates residuals and Schur-reduced normal equations of a problem.

    The landmark state is an ``(N, 3)`` array ``[du, dv, rho]``: the anchor
    pixel correction and the inverse depth.
    """

    def __init__(self, rig: Rig, problem: AnchoredProblem, options: RefinementOptions,
                 inertial: InertialFactors | None):
        self.rig = rig
        self.problem = problem
        self.options = options
        self.inertial = inertial
        count = len(problem)
        self.count = count
        self.host_rows = {host: np.flatnonzero(problem.hosts == host)
                          for host in np.unique(problem.hosts)}
        # Anchor bearings at the measured anchor pixels.
        self.rays, _ = self.anchor_rays(np.zeros((count, 2)), with_jacobian=False)
        # Required views first: they establish which landmarks are usable.
        self.views = sorted(problem.views, key=lambda view: not view.required)
        self.has_required = np.zeros(count, dtype=bool)
        for view in self.views:
            if view.required:
                self.has_required[view.landmarks] = True
        for view in self.views:
            if np.any(problem.hosts[view.landmarks] != view.host):
                raise ValueError("view lists landmarks of another host")
        images = {view.image for view in self.views}
        images |= {(host, False) for host in self.host_rows}
        self.images = sorted(images, key=lambda image: (not image[1], image[0]))
        # Fixed extrinsics: view camera from host camera, and rig from host.
        self.camera_from_rig = {name: rig.camera_from_rig(name) for name in rig.camera_names}
        self.rig_from_camera = {name: rig.rig_from_camera(name) for name in rig.camera_names}

    # -------------------------------------------------------------- state
    def initial_state(self, inverse_depths: np.ndarray) -> np.ndarray:
        depths = np.asarray(inverse_depths, dtype=np.float64).reshape(-1, 1)
        return np.hstack((np.zeros((len(depths), 2)), depths))

    @staticmethod
    def updated(state: np.ndarray, step: np.ndarray) -> np.ndarray:
        result = state + step
        result[:, -1] = np.clip(result[:, -1], 0.0, _MAX_INVERSE_DEPTH)
        return result

    def anchor_rays(self, corrections: np.ndarray, with_jacobian: bool):
        """Bearings of the (corrected) anchor pixels and d(bearing)/d(pixel)."""
        pixels = self.problem.anchor_pixels + corrections
        rays = np.full((self.count, 3), np.nan)
        jacobian = np.full((self.count, 3, 2), np.nan) if with_jacobian else None
        for host, rows in self.host_rows.items():
            camera = self.rig.cameras[host]
            rays[rows] = camera.bearings(pixels[rows])
            if with_jacobian:
                for axis in range(2):
                    step = np.zeros(2)
                    step[axis] = _ANCHOR_STEP
                    jacobian[rows, :, axis] = (camera.bearings(pixels[rows] + step)
                                               - camera.bearings(pixels[rows] - step)
                                               ) / (2.0 * _ANCHOR_STEP)
        return rays, jacobian

    def view_transform(self, view: View, pose: np.ndarray) -> np.ndarray:
        """Return ``T_view_host``."""
        if view.current:
            return self.camera_from_rig[view.camera] @ pose @ self.rig_from_camera[view.host]
        return self.camera_from_rig[view.camera] @ self.rig_from_camera[view.host]

    @staticmethod
    def _variance_weight(variance_scales, image) -> float:
        scale = 1.0 if variance_scales is None else float(variance_scales.get(image, 1.0))
        if not np.isfinite(scale) or scale < 1.0:
            scale = 1.0
        return 1.0 / (scale * scale)

    # --------------------------------------------------------- evaluation
    def evaluate(self, pose: np.ndarray, state: np.ndarray, with_jacobians: bool,
                 compute_sandwich: bool = False,
                 variance_scales: dict[tuple[str, bool], float] | None = None) -> dict:
        options = self.options
        count = self.count
        sigma = float(options.pixel_std_px)
        delta = max(float(options.robust_delta), 1.0e-6)
        rays, ray_jacobian = self.anchor_rays(state[:, :2], with_jacobians)
        rho = state[:, -1]
        valid_feature = np.isfinite(rays).all(axis=1) & self.has_required
        if ray_jacobian is not None:
            valid_feature &= np.isfinite(ray_jacobian).all(axis=(1, 2))

        h_ll = np.zeros((count, 3, 3))
        h_lp = np.zeros((count, 3, 6))
        h_pp = np.zeros((count, 6, 6))
        g_l = np.zeros((count, 3))
        g_p = np.zeros((count, 6))
        cost = np.zeros(count)
        stats = {"dimensions": 0, "partner_observations": 0, "square_sum": 0.0, "count": 0,
                 "image_square_sum": {image: 0.0 for image in self.images},
                 "image_count": {image: 0 for image in self.images},
                 "image_dispersion": {image: [0.0, 0, 0.0] for image in self.images},
                 "weights": []}
        view_scores = []

        def robust(normalized, valid, image):
            """Huber weights; accumulate cost and statistics of valid rows."""
            norms = np.linalg.norm(normalized, axis=1)
            weights = np.ones(len(normalized))
            outlier = norms > delta
            weights[outlier] = delta / np.maximum(norms[outlier], 1.0e-12)
            safe = np.where(valid, norms, 0.0)
            valid_count = int(np.count_nonzero(valid))
            stats["dimensions"] += 2 * valid_count
            stats["weights"].append(weights[valid])
            if compute_sandwich:
                dispersion = stats["image_dispersion"][image]
                dispersion[0] += float(np.sum(weights[valid, None] * normalized[valid] ** 2))
                dispersion[1] += 2 * valid_count
                dispersion[2] += float(np.sum(weights[valid]))
            return weights, np.where(safe <= delta, 0.5 * safe * safe,
                                     delta * (safe - 0.5 * delta))

        for view in self.views:
            rows = view.landmarks
            transform = self.view_transform(view, pose)
            rotation, translation = transform[:3, :3], transform[:3, 3]
            vectors = rays[rows] @ rotation.T + rho[rows, None] * translation
            camera = self.rig.cameras[view.camera]
            if with_jacobians:
                prediction, projection_jacobian = camera.project_with_jacobian(
                    vectors, _PROJECTION_STEP)
            else:
                prediction = camera.project(vectors)
            observed = view.pixels
            valid = valid_feature[rows] & np.isfinite(observed).all(axis=1)
            valid &= np.isfinite(prediction).all(axis=1)
            if with_jacobians:
                valid &= np.isfinite(projection_jacobian).all(axis=(1, 2))
            if view.required:
                # The temporal observation anchors the whole landmark; optional
                # partner views may be missing without invalidating it.
                valid_feature[rows] &= valid
            if not np.any(valid):
                view_scores.append(None)
                continue

            raw_residual = prediction - observed
            normalized = raw_residual / sigma
            weights, view_cost = robust(normalized, valid, view.image)
            cost[rows] += view_cost
            square_sum = float(np.sum(raw_residual[valid] ** 2))
            valid_count = int(np.count_nonzero(valid))
            stats["square_sum"] += square_sum
            stats["count"] += 2 * valid_count
            stats["image_square_sum"][view.image] += square_sum
            stats["image_count"][view.image] += 2 * valid_count
            if view.current and not view.required:
                stats["partner_observations"] += valid_count
            if not with_jacobians:
                view_scores.append(None)
                continue

            # d(view vector)/d(rho) is the translation of T_view_host.
            rho_jacobian = np.einsum("nij,j->ni", projection_jacobian, translation)
            anchor_jacobian = np.einsum("nij,jk,nkl->nil", projection_jacobian, rotation,
                                        ray_jacobian[rows])
            landmark_jacobian = np.concatenate((anchor_jacobian, rho_jacobian[:, :, None]),
                                               axis=2)
            if view.current:
                # Left perturbation of T acts on z = T T_rig_host (b, rho):
                # dz/d(delta) = [-[z]x, rho I], then rotated into the view.
                host_in_rig = pose @ self.rig_from_camera[view.host]
                z = rays[rows] @ host_in_rig[:3, :3].T + rho[rows, None] * host_in_rig[:3, 3]
                z_jacobian = np.zeros((len(rows), 3, 6))
                z_jacobian[:, :, :3] = -skew_batch(z)
                z_jacobian[:, :, 3:] = rho[rows, None, None] * np.eye(3)[None]
                camera_rotation = self.camera_from_rig[view.camera][:3, :3]
                pose_jacobian = np.einsum("nij,jk,nkl->nil", projection_jacobian,
                                          camera_rotation, z_jacobian)
            else:
                pose_jacobian = np.zeros((len(rows), 2, 6))
            # Mask before multiplying: 0 * NaN is NaN.
            safe_weight = (np.where(valid, weights, 0.0)
                           * self._variance_weight(variance_scales, view.image))
            safe_pose = np.where(valid[:, None, None], pose_jacobian / sigma, 0.0)
            safe_landmark = np.where(valid[:, None, None], landmark_jacobian / sigma, 0.0)
            safe_residual = np.where(valid[:, None], normalized, 0.0)
            h_ll[rows] += safe_weight[:, None, None] * np.einsum(
                "nid,nie->nde", safe_landmark, safe_landmark)
            cross = safe_weight[:, None, None] * np.einsum(
                "nid,nik->ndk", safe_landmark, safe_pose)
            h_lp[rows] += cross
            h_pp[rows] += safe_weight[:, None, None] * np.einsum(
                "nij,nik->njk", safe_pose, safe_pose)
            g_l[rows] += safe_weight[:, None] * np.einsum(
                "nid,ni->nd", safe_landmark, safe_residual)
            pose_score = safe_weight[:, None] * np.einsum(
                "ni,nij->nj", safe_residual, safe_pose)
            g_p[rows] += pose_score
            view_scores.append((pose_score, cross) if compute_sandwich else None)

        # The anchor pixel is a measurement: residual (corrected - measured).
        for host, rows in self.host_rows.items():
            image = (host, False)
            valid = valid_feature[rows]
            if not np.any(valid):
                continue
            correction = state[rows, :2]
            normalized = correction / sigma
            weights, anchor_cost = robust(normalized, valid, image)
            cost[rows] += anchor_cost
            square_sum = float(np.sum(correction[valid] ** 2))
            stats["square_sum"] += square_sum
            stats["count"] += 2 * int(np.count_nonzero(valid))
            stats["image_square_sum"][image] += square_sum
            stats["image_count"][image] += 2 * int(np.count_nonzero(valid))
            if with_jacobians:
                weight = (np.where(valid, weights, 0.0)
                          * self._variance_weight(variance_scales, image))
                h_ll[rows, 0, 0] += weight / sigma ** 2
                h_ll[rows, 1, 1] += weight / sigma ** 2
                g_l[rows, :2] += weight[:, None] * np.where(
                    valid[:, None], normalized, 0.0) / sigma

        result = {
            "cost": float(np.sum(cost)),
            "dimensions": int(stats["dimensions"]),
            "valid_count": int(np.count_nonzero(valid_feature)),
            "residual_square_sum": stats["square_sum"],
            "residual_count": stats["count"],
            "partner_observations": stats["partner_observations"],
            "image_residual_rms": {
                image: float(np.sqrt(max(0.0, stats["image_square_sum"][image])
                                     / max(1, stats["image_count"][image])))
                for image in self.images},
            "robust_weight_stats": _weight_stats(stats["weights"]),
            "variance_scales": {
                image: (float(np.sqrt(max(1.0, values[0] / max(1.0, 2.0 * values[2]))))
                        if values[1] >= 4 else 1.0)
                for image, values in stats["image_dispersion"].items()},
            "h_pose": None,
            "g_pose": None,
            "point_meat": None,
            "cluster_meat": None,
        }
        if not with_jacobians:
            if self.inertial is not None:
                self.inertial.augment(result, pose, with_jacobians=False)
            return result

        inverse_ll, landmark_rank = _pseudo_inverse(h_ll)
        landmark_step = np.einsum("nde,ne->nd", inverse_ll, g_l)
        h_pose = np.sum(h_pp, axis=0) - np.einsum("ndk,nde,nel->kl", h_lp, inverse_ll, h_lp)
        g_pose = np.sum(g_p, axis=0) - np.einsum("ndk,nd->k", h_lp, landmark_step)
        result.update(h_pose=h_pose, g_pose=g_pose, inverse_ll=inverse_ll, h_lp=h_lp,
                      g_l=g_l, landmark_rank=landmark_rank)
        if compute_sandwich:
            result.update(self._meats(h_pose, g_p, h_lp, landmark_step, view_scores))
        result["visual_h_pose"] = h_pose.copy()
        result["visual_g_pose"] = g_pose.copy()
        if self.inertial is not None:
            self.inertial.augment(result, pose, with_jacobians=True)
        return result

    def _meats(self, h_pose, g_p, h_lp, landmark_step, view_scores) -> dict:
        """Empirical score outer products for the sandwich covariance.

        Scores are taken after eliminating each landmark's unknowns.
        ``point_meat`` treats landmarks as independent; ``cluster_meat`` sums
        scores within image tiles first, preserving correlated residuals
        caused by texture or calibration errors in one image region.
        """
        point_scores = g_p - np.einsum("ndk,nd->nk", h_lp, landmark_step)
        point_meat = point_scores.T @ point_scores
        clusters = max(1, int(self.options.spatial_clusters))
        columns = max(1, int(np.ceil(np.sqrt(clusters * 1.5))))
        rows_count = max(1, int(np.ceil(clusters / columns)))
        cluster_scores: dict[tuple, np.ndarray] = {}
        for image in self.images:
            members = [(view, scores) for view, scores in zip(self.views, view_scores)
                       if view.image == image and scores is not None]
            if not members:
                continue
            pixels = np.concatenate([view.pixels for view, _ in members])
            scores = np.concatenate([
                pose_scores - np.einsum("mdk,md->mk", cross, landmark_step[view.landmarks])
                for view, (pose_scores, cross) in members])
            finite = np.isfinite(pixels).all(axis=1) & np.isfinite(scores).all(axis=1)
            if not np.any(finite):
                continue
            minimum = np.min(pixels[finite], axis=0)
            extent = np.maximum(np.max(pixels[finite], axis=0) - minimum, 1.0)
            normalized = (pixels - minimum) / extent
            with np.errstate(invalid="ignore"):
                tile_x = np.clip((normalized[:, 0] * columns).astype(np.int64), 0, columns - 1)
                tile_y = np.clip((normalized[:, 1] * rows_count).astype(np.int64),
                                 0, rows_count - 1)
            tiles = tile_y * columns + tile_x
            for tile in np.unique(tiles[finite]):
                selected = finite & (tiles == tile)
                cluster_scores[(image, int(tile))] = np.sum(scores[selected], axis=0)
        if len(cluster_scores) >= 4:
            cluster_meat = sum(np.outer(score, score) for score in cluster_scores.values())
            count = float(len(cluster_scores))
            cluster_meat = max(1.0, count / max(1.0, count - 1.0)) * cluster_meat
            estimator = "spatial_cluster"
        else:
            cluster_meat = point_meat
            estimator = "point_cluster"
        cluster_meat = 0.5 * (cluster_meat + cluster_meat.T)
        point_meat = 0.5 * (point_meat + point_meat.T)
        try:
            bread = np.linalg.pinv(h_pose, hermitian=True)
            if not (np.all(np.isfinite(bread @ cluster_meat @ bread.T))
                    and np.all(np.isfinite(bread @ point_meat @ bread.T))):
                return {}
        except np.linalg.LinAlgError:
            return {}
        return {"point_meat": point_meat, "cluster_meat": cluster_meat,
                "meat_estimator": estimator}


def _pseudo_inverse(blocks: np.ndarray) -> tuple[np.ndarray, int]:
    """Batched symmetric pseudo-inverse of landmark blocks.

    Unobservable directions (e.g. inverse depth without translation or a
    stereo partner) are left out of the elimination.  Returns the inverses
    and the total number of observable landmark directions.
    """
    values, vectors = np.linalg.eigh(0.5 * (blocks + np.swapaxes(blocks, 1, 2)))
    floor = np.maximum(1.0e-12, 1.0e-12 * values[:, -1:])
    observable = values > floor
    inverse_values = np.where(observable, 1.0 / np.where(observable, values, 1.0), 0.0)
    inverse = np.einsum("nij,nj,nkj->nik", vectors, inverse_values, vectors)
    return inverse, int(np.count_nonzero(observable))


def _weight_stats(weights: list[np.ndarray]) -> tuple[float, float, float]:
    if not weights:
        return 1.0, 1.0, 1.0
    values = np.concatenate(weights)
    if len(values) == 0:
        return 1.0, 1.0, 1.0
    return float(np.min(values)), float(np.median(values)), float(np.max(values))


def initial_inverse_depths(rig: Rig, problem: AnchoredProblem) -> np.ndarray:
    """Inverse depth (1/distance along the anchor bearing) of each landmark."""
    depths = np.zeros(len(problem))
    for host in np.unique(problem.hosts):
        rows = np.flatnonzero(problem.hosts == host)
        points = problem.initial_points_rig[rows]
        finite = np.isfinite(points).all(axis=1)
        host_from_rig = rig.camera_from_rig(host)
        distances = np.linalg.norm(points[finite] @ host_from_rig[:3, :3].T
                                   + host_from_rig[:3, 3], axis=1)
        usable = np.isfinite(distances) & (distances > 1.0e-6)
        values = np.zeros(len(distances))
        values[usable] = np.minimum(_MAX_INVERSE_DEPTH, 1.0 / distances[usable])
        depths[rows[finite]] = values
    return depths


def refine(
    rig: Rig,
    problem: AnchoredProblem,
    initial_pose: np.ndarray,
    options: RefinementOptions,
    inertial: InertialFactors | None = None,
) -> AnchoredSolution | None:
    """Refine ``T_rig(current)_rig(previous)`` and estimate its covariance.

    Gauss-Newton with per-landmark Schur elimination, Huber IRLS and a
    backtracking line search.  The covariance is the model covariance of the
    final robust normal equations (including inertial factors), plus the
    positive-semidefinite excess of a spatially clustered sandwich estimate
    over an independent-landmark one.  Returns None when the problem is too
    small or numerically degenerate.
    """
    minimum = options.minimum_landmarks
    if len(problem) < minimum or not np.all(np.isfinite(problem.anchor_pixels)):
        return None
    linearizer = _Linearizer(rig, problem, options, inertial)
    # Anchors outside the camera model's invertible domain (e.g. fisheye
    # corners) only drop their own landmark.  The legacy solver rejected the
    # whole frame instead, which disabled refinement on TUM-VI entirely.
    if np.count_nonzero(np.isfinite(linearizer.rays).all(axis=1)) < minimum:
        return None
    pose = np.asarray(initial_pose, dtype=np.float64).reshape(4, 4).copy()
    state = linearizer.initial_state(initial_inverse_depths(rig, problem))

    initial = linearizer.evaluate(pose, state, with_jacobians=False)
    if not np.isfinite(initial["cost"]):
        return None
    initial_cost = final_cost = float(initial["cost"])
    iterations = 0
    for _ in range(max(1, int(options.max_iterations))):
        linearized = linearizer.evaluate(pose, state, with_jacobians=True)
        if linearized["valid_count"] < minimum:
            return None
        h_pose = linearized["h_pose"]
        try:
            eigenvalues = np.linalg.eigvalsh(h_pose)
        except np.linalg.LinAlgError:
            return None
        largest = float(eigenvalues[-1])
        if not np.isfinite(largest) or largest <= 1.0e-12 or eigenvalues[0] < -1.0e-8 * largest:
            return None
        damping = max(1.0e-10, largest * 1.0e-8)
        try:
            delta_pose = np.linalg.solve(h_pose + damping * np.eye(6), -linearized["g_pose"])
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta_pose)):
            return None
        delta_state = -np.einsum(
            "nde,ne->nd", linearized["inverse_ll"],
            linearized["g_l"] + np.einsum("ndk,k->nd", linearized["h_lp"], delta_pose))

        improved = False
        for alpha in _LINE_SEARCH:
            candidate_pose = left_update(pose, alpha * delta_pose)
            candidate_state = linearizer.updated(state, alpha * delta_state)
            candidate = linearizer.evaluate(candidate_pose, candidate_state,
                                            with_jacobians=False)
            if (np.isfinite(candidate["cost"]) and candidate["valid_count"] >= minimum
                    and candidate["cost"] < final_cost):
                pose, state = candidate_pose, candidate_state
                final_cost = float(candidate["cost"])
                improved = True
                break
        if not improved:
            break
        iterations += 1
        if np.linalg.norm(delta_pose) < 1.0e-7:
            break

    final = linearizer.evaluate(pose, state, with_jacobians=True, compute_sandwich=True)
    # Per-image variance components from the final robust residuals; one
    # relinearization with them keeps landmark elimination consistent
    # instead of scaling a finished covariance.
    variance_scales = final["variance_scales"]
    final = linearizer.evaluate(pose, state, with_jacobians=True, compute_sandwich=True,
                                variance_scales=variance_scales)
    if final["valid_count"] < minimum:
        return None
    h_pose = final["h_pose"]
    estimate = _covariance(h_pose, final, final_cost)
    if estimate is None:
        return None
    covariance, estimator, information_values = estimate
    largest = float(information_values[-1])
    floor = max(1.0e-12, largest * 1.0e-12)
    visual_pose = visual_covariance = None
    if inertial is not None and final.get("visual_h_pose") is not None:
        try:
            step = np.linalg.solve(final["visual_h_pose"], -final["visual_g_pose"])
            visual = _covariance(final["visual_h_pose"], final, final_cost)
            if visual is not None and np.all(np.isfinite(step)):
                visual_pose, visual_covariance = left_update(pose, step), visual[0]
        except np.linalg.LinAlgError:
            pass

    positive = information_values[information_values > floor]
    condition = float("inf") if len(positive) == 0 else largest / max(float(positive[0]), floor)

    def eigenvalues_of(matrix) -> tuple[float, ...]:
        if matrix is None:
            return ()
        try:
            return tuple(float(value) for value in np.linalg.eigvalsh(matrix))
        except np.linalg.LinAlgError:
            return ()

    return AnchoredSolution(
        pose=pose,
        covariance=covariance,
        landmarks=int(final["valid_count"]),
        partner_observations=int(final["partner_observations"]),
        iterations=iterations,
        initial_cost=initial_cost,
        final_cost=final_cost,
        rank=int(np.count_nonzero(information_values > largest * 1.0e-8)),
        condition_number=condition,
        reprojection_rms_px=float(np.sqrt(final["residual_square_sum"]
                                          / max(1, final["residual_count"]))),
        covariance_estimator=estimator,
        information_eigenvalues=tuple(float(value) for value in information_values),
        visual_information_eigenvalues=eigenvalues_of(final.get("visual_h_pose")),
        inertial_information_eigenvalues=eigenvalues_of(final.get("inertial_h_pose")),
        image_residual_rms_px={_image_label(image): value
                               for image, value in final["image_residual_rms"].items()},
        image_variance_scales={_image_label(image): float(value)
                               for image, value in variance_scales.items()},
        robust_weight_stats=final["robust_weight_stats"],
        visual_pose=visual_pose,
        visual_covariance=visual_covariance,
    )


def _covariance(h_pose: np.ndarray, final: dict, final_cost: float):
    """Model covariance plus the PSD clustered-sandwich excess.

    Returns ``(covariance, estimator, information_eigenvalues)`` or None.
    """
    inverted = inverse_with_floor(h_pose)
    if inverted is None:
        return None
    model_covariance, information_values = inverted
    estimator = "model_hessian"
    if final.get("cluster_meat") is not None and final.get("point_meat") is not None:
        covariance = model_covariance
        try:
            bread = np.linalg.pinv(h_pose, hermitian=True)
            clustered = bread @ final["cluster_meat"] @ bread.T
            independent = bread @ final["point_meat"] @ bread.T
            excess = clustered - independent
            excess = 0.5 * (excess + excess.T)
            values, vectors = np.linalg.eigh(excess)
            excess = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
            if np.all(np.isfinite(excess)):
                covariance = model_covariance + excess
                estimator = f"model_plus_{final['meat_estimator']}_excess"
        except np.linalg.LinAlgError:
            covariance = model_covariance
    else:
        degrees_of_freedom = max(1, int(final["dimensions"]) - final["landmark_rank"] - 6)
        covariance = model_covariance * max(1.0, 2.0 * final_cost / degrees_of_freedom)
        estimator = "model_hessian_residual_scaled"
    covariance = positive_definite_or_none(covariance)
    if covariance is None:
        return None
    return covariance, estimator, information_values


def build_problem(
    hosts: Sequence[str],
    anchor_pixels: Sequence[np.ndarray],
    initial_points_rig: Sequence[np.ndarray],
    views: Sequence[View],
) -> AnchoredProblem:
    """Concatenate per-host landmark blocks into one problem.

    ``views[*].landmarks`` must already index the concatenated landmarks.
    """
    host_array = np.concatenate([np.full(len(pixels), host, dtype=object)
                                 for host, pixels in zip(hosts, anchor_pixels)]) \
        if hosts else np.empty(0, dtype=object)
    return AnchoredProblem(
        hosts=host_array,
        anchor_pixels=(np.concatenate(anchor_pixels) if anchor_pixels
                       else np.empty((0, 2))),
        initial_points_rig=(np.concatenate(initial_points_rig) if initial_points_rig
                            else np.empty((0, 3))),
        views=list(views),
    )
