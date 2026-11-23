"""Frame-to-frame multi-camera visual odometry.

For each synchronized rig frame:

1. extract features in every camera and match, in one batch, each temporal
   camera against its previous frame and every stereo pair;
2. filter stereo matches with the calibrated epipolar geometry and
   triangulate them;
3. verify temporal matches with calibrated essential-matrix MAGSAC;
4. estimate the rig motion with (generalized) PnP from previous-frame stereo
   points, optionally competing with an IMU-seeded refinement;
5. refine the motion and estimate its covariance with anchored
   inverse-depth landmarks over all temporal matches
   (:mod:`colmap_odometry.estimation.anchored`).

Motions are ``T_rig(previous)_rig(current)`` with a right-local
``[rotation, translation]`` covariance; the world frame is the rig at the
first frame.
"""

from __future__ import annotations

import enum
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pycolmap

from .config import OdometryConfig
from .diagnostics import Profiler
from .estimation import pnp
from .estimation.anchored import AnchoredSolution, View, build_problem, refine
from .estimation.inertial import MotionPrior
from .features import FeatureFrontend, Features
from .geometry import Rig
from .geometry.lie import invert, skew, so3_log
from .vision.stereo import StereoObservations, stereo_observations
from .vision.temporal import essential_rotation, verify_essential
from .vision.tracks import TrackManager, TrackObservation, TrackUpdate


class Status(enum.Enum):
    BOOTSTRAP = "bootstrap"
    TRACKED = "tracked"
    FAILED = "failed"


@dataclass
class StereoPoints:
    """Metric stereo points of one host camera at the current frame."""

    camera: str
    pixels: np.ndarray
    points_rig: np.ndarray


@dataclass
class MatchDebug:
    first: str
    second: str
    keypoints_first: np.ndarray
    keypoints_second: np.ndarray
    matches: np.ndarray


@dataclass
class FrameResult:
    stamp_ns: int
    previous_stamp_ns: int | None
    status: Status
    failure_reason: str = ""
    #: ``T_rig(previous)_rig(current)``.
    motion: np.ndarray | None = None
    #: Right-local rt covariance of ``motion``.
    motion_covariance: np.ndarray | None = None
    #: With inertial factors: the visual-only motion and covariance.
    visual_motion: np.ndarray | None = None
    visual_motion_covariance: np.ndarray | None = None
    #: ``T_world_rig`` after this frame (unchanged on failure).
    world_from_rig: np.ndarray = field(default_factory=lambda: np.eye(4))
    covariance_source: str = "none"
    inlier_count: int = 0
    reprojection_error_px: float = 0.0
    condition_number: float = 0.0
    track_updates: list[TrackUpdate] = field(default_factory=list)
    stereo_points: list[StereoPoints] = field(default_factory=list)
    match_debug: list[MatchDebug] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def tracked(self) -> bool:
        return self.status is Status.TRACKED


@dataclass
class _Frame:
    stamp_ns: int
    index: int
    features: dict[str, Features]
    # Stereo observations per host camera, in pair order.
    stereo: dict[str, list[StereoObservations]]
    _lookups: dict = field(default_factory=dict)

    def stereo_rows(self, host: str) -> list[tuple[StereoObservations, np.ndarray]]:
        """``(observations, row_by_host_keypoint)`` for each partner of host."""
        if host not in self._lookups:
            count = len(self.features[host])
            self._lookups[host] = [(item, item.rows_by_host_index(count))
                                   for item in self.stereo.get(host, [])]
        return self._lookups[host]


class Odometry:
    def __init__(self, rig: Rig, frontend: FeatureFrontend, config: OdometryConfig):
        self.rig = rig
        self.frontend = frontend
        self.config = config
        temporal = (list(config.temporal_cameras) if config.temporal_cameras is not None
                    else rig.default_temporal_cameras())
        for name in temporal:
            if name not in rig.cameras:
                raise ValueError(f"temporal camera '{name}' is not in the rig")
        if not temporal:
            raise ValueError("no temporal cameras: configure stereo_pairs or temporal_cameras")
        self.temporal_cameras = temporal
        tracks = config.feature_tracks
        track_cameras = list(tracks.cameras) if tracks.cameras is not None else temporal
        for name in track_cameras:
            if name not in temporal:
                raise ValueError(f"feature-track camera '{name}' is not a temporal camera")
        self._track_ids = [0]
        self._trackers = ({name: TrackManager(name, tracks.min_length, self._track_ids)
                           for name in track_cameras} if tracks.enabled else {})
        self.profiler = Profiler()
        self.world_from_rig = np.eye(4)
        self._previous: _Frame | None = None
        self._frame_index = 0
        # Deterministic COLMAP RANSAC.
        pycolmap.set_random_seed(0)

    # ------------------------------------------------------------- lifecycle
    def warmup(self) -> None:
        self.frontend.warmup([(camera.width, camera.height)
                              for camera in self.rig.cameras.values()])

    def reset(self) -> list[TrackUpdate]:
        """Drop the previous frame (the trajectory continues); close tracks."""
        updates = self.finish_tracks()
        self._previous = None
        return updates

    def finish_tracks(self) -> list[TrackUpdate]:
        updates: list[TrackUpdate] = []
        for tracker in self._trackers.values():
            tracker.finish_all(updates)
        return updates

    # ------------------------------------------------------------- per frame
    def process(self, stamp_ns: int, images: Sequence[np.ndarray],
                prior: MotionPrior | None = None) -> FrameResult:
        if len(images) != len(self.rig.cameras):
            raise ValueError(f"expected {len(self.rig.cameras)} images, got {len(images)}")
        self.profiler.reset()
        started = time.perf_counter()
        report: dict[str, Any] = {"frame": self._frame_index}
        names = self.rig.camera_names
        previous = self._previous

        with self.profiler.measure("features"):
            extracted = self.frontend.extract(images)
        features = dict(zip(names, extracted))
        report["features"] = {name: len(item) for name, item in features.items()}
        frame = _Frame(int(stamp_ns), self._frame_index, features, {})
        self._frame_index += 1

        pairs: list[tuple[str, str, Features, Features]] = []
        if previous is not None:
            pairs += [(name, name, previous.features[name], features[name])
                      for name in self.temporal_cameras]
        pairs += [(pair.host, pair.partner, features[pair.host], features[pair.partner])
                  for pair in self.rig.stereo_pairs]
        with self.profiler.measure("matching"):
            raw = self.frontend.match([(a, b) for _, _, a, b in pairs])
        temporal_count = len(self.temporal_cameras) if previous is not None else 0
        raw_temporal = dict(zip(self.temporal_cameras, raw[:temporal_count]))
        stereo_raw = raw[temporal_count:]

        result = FrameResult(int(stamp_ns), previous.stamp_ns if previous else None,
                             Status.FAILED, world_from_rig=self.world_from_rig.copy())
        with self.profiler.measure("stereo"):
            stereo_report = {}
            for pair, matches in zip(self.rig.stereo_pairs, stereo_raw):
                observations, filtered = stereo_observations(
                    self.rig, pair.host, pair.partner, features[pair.host].keypoints,
                    features[pair.partner].keypoints, matches, self.config.stereo)
                frame.stereo.setdefault(pair.host, []).append(observations)
                stereo_report[f"{pair.host}-{pair.partner}"] = asdict(observations.stats)
                result.match_debug.append(MatchDebug(
                    pair.host, pair.partner, features[pair.host].keypoints,
                    features[pair.partner].keypoints, filtered))
            report["stereo"] = stereo_report
        result.stereo_points = [
            StereoPoints(host, np.concatenate([o.host_pixels[o.metric] for o in items]),
                         np.concatenate([o.points_rig[o.metric] for o in items]))
            for host, items in frame.stereo.items()]

        self._previous = frame
        if previous is None:
            result.status = Status.BOOTSTRAP
            result.failure_reason = "bootstrap"
        else:
            try:
                self._track(previous, frame, raw_temporal, prior, result, report)
            except _FrameFailure as failure:
                result.status = Status.FAILED
                result.failure_reason = str(failure)
        if result.tracked:
            self.world_from_rig = self.world_from_rig @ result.motion
        result.world_from_rig = self.world_from_rig.copy()
        report["status"] = result.status.value
        report["failure_reason"] = result.failure_reason
        report["timings"] = dict(self.profiler.timings,
                                 total=time.perf_counter() - started)
        result.report = report
        return result

    def _track(self, previous: _Frame, frame: _Frame, raw_temporal: dict[str, np.ndarray],
               prior: MotionPrior | None, result: FrameResult, report: dict) -> None:
        config = self.config
        temporal_report = {}
        verified: dict[str, np.ndarray] = {}
        with self.profiler.measure("temporal_verification"):
            for name in self.temporal_cameras:
                camera = self.rig.cameras[name]
                check = verify_essential(
                    camera, camera, previous.features[name].keypoints,
                    frame.features[name].keypoints, raw_temporal[name], config.temporal)
                verified[name] = check.inliers
                temporal_report[name] = {
                    "raw": int(len(raw_temporal[name])), "verified": int(len(check.inliers)),
                    "degenerate": check.degenerate, "failure_reason": check.failure_reason}
        report["temporal"] = temporal_report

        with self.profiler.measure("tracks"):
            for name, tracker in self._trackers.items():
                tracker.update(verified[name],
                               lambda index, n=name: self._track_observation(previous, n, index),
                               lambda index, n=name: self._track_observation(frame, n, index),
                               result.track_updates)

        # --- PnP from previous-frame metric stereo points.
        pnp_report: dict[str, Any] = {}
        report["pnp"] = pnp_report
        temporal_matches = verified
        correspondences = self._correspondences(previous, frame, verified)
        source = "verified"
        if len(correspondences) < config.pnp.min_correspondences:
            # Essential estimation degenerates for stationary or purely
            # rotating motion, where PnP with fixed stereo depth is well posed.
            fallback = self._correspondences(previous, frame, raw_temporal)
            if len(fallback) >= config.pnp.min_correspondences:
                correspondences, temporal_matches = fallback, raw_temporal
                source = "raw_degeneracy_fallback"
        pnp_report.update(correspondences=len(correspondences), source=source)
        if len(correspondences) < config.pnp.min_correspondences:
            raise _FrameFailure(f"insufficient_stereo_supported_temporal_matches:"
                                f"{len(correspondences)}")

        with self.profiler.measure("pnp"):
            ransac = pnp.estimate(self.rig, correspondences, config.pnp)
            if ransac is None and source == "verified":
                fallback = self._correspondences(previous, frame, raw_temporal)
                if len(fallback) >= config.pnp.min_correspondences:
                    recovered = pnp.estimate(self.rig, fallback, config.pnp)
                    if recovered is not None:
                        ransac, correspondences = recovered, fallback
                        temporal_matches, source = raw_temporal, "raw_pnp_recovery"
                        pnp_report.update(correspondences=len(fallback), source=source)
            seeded = None
            if prior is not None and prior.seed is not None:
                seeded = pnp.refine_seed(
                    self.rig, correspondences, prior.seed, prior.seed_max_error_px,
                    prior.seed_min_inliers, config.pnp.ransac_max_error_px)
        hypothesis, pose_source = ransac, "ransac"
        if seeded is not None and (
                ransac is None or seeded.inlier_count > ransac.inlier_count
                or (seeded.inlier_count == ransac.inlier_count
                    and seeded.median_error_px < ransac.median_error_px)):
            hypothesis, pose_source = seeded, "prior_seeded"
        pnp_report.update(
            pose_source=pose_source,
            ransac_inliers=ransac.inlier_count if ransac else 0,
            ransac_median_error_px=ransac.median_error_px if ransac else None,
            seeded_inliers=seeded.inlier_count if seeded else 0,
            seeded_median_error_px=seeded.median_error_px if seeded else None)
        if hypothesis is None:
            raise _FrameFailure("pnp_failed")
        result.inlier_count = hypothesis.inlier_count
        result.reprojection_error_px = hypothesis.median_error_px

        # --- Anchored inverse-depth refinement over all temporal matches.
        pose = hypothesis.pose
        covariance = None
        refinement_report: dict[str, Any] = {}
        report["refinement"] = refinement_report
        with self.profiler.measure("refinement"):
            problem = self._problem(previous, frame, temporal_matches)
            refinement_report["temporal_landmarks"] = len(problem)
            problem = self._select_landmarks(problem, pose)
            refinement_report["candidate_landmarks"] = len(problem)
            if len(problem) >= config.refinement.minimum_landmarks:
                initial = self._initial_pose(previous, frame, temporal_matches, pose,
                                             refinement_report)
                solution = refine(self.rig, problem, initial, config.refinement,
                                  prior.factors if prior is not None else None)
                if solution is not None:
                    pose, covariance = solution.pose, solution.covariance
                    if solution.visual_pose is not None:
                        result.visual_motion = invert(solution.visual_pose)
                        result.visual_motion_covariance = solution.visual_covariance
                    result.covariance_source = "anchored_inverse_depth"
                    result.condition_number = solution.condition_number
                    refinement_report.update(self._solution_report(solution))
                else:
                    refinement_report["failure_reason"] = "solver_returned_no_solution"
            else:
                refinement_report["failure_reason"] = f"too_few_landmarks:{len(problem)}"
        if covariance is None:
            covariance = hypothesis.covariance
            result.covariance_source = "pnp_fixed_points"
        if covariance is None:
            raise _FrameFailure("no_covariance")

        result.status = Status.TRACKED
        result.motion = invert(pose)
        # Left-local covariance of T equals the right-local covariance of T^-1.
        result.motion_covariance = covariance

    # ---------------------------------------------------------------- helpers
    def _correspondences(self, previous: _Frame, frame: _Frame,
                         temporal: dict[str, np.ndarray]) -> pnp.Correspondences:
        """Current pixels of temporal matches whose previous keypoint has a
        metric stereo point (first partner in pair order)."""
        cameras, points2d, points3d = [], [], []
        for name in self.temporal_cameras:
            matches = temporal[name]
            if len(matches) == 0:
                continue
            point = np.full((len(matches), 3), np.nan)
            for observations, rows in previous.stereo_rows(name):
                row = rows[matches[:, 0]]
                usable = (row >= 0) & np.isnan(point[:, 0])
                usable[usable] = observations.metric[row[usable]]
                point[usable] = observations.points_rig[row[usable]]
            keep = np.isfinite(point).all(axis=1)
            if not np.any(keep):
                continue
            cameras += [name] * int(keep.sum())
            points2d.append(frame.features[name].keypoints[matches[keep, 1]])
            points3d.append(point[keep])
        return pnp.Correspondences(
            cameras,
            np.concatenate(points2d) if points2d else np.empty((0, 2)),
            np.concatenate(points3d) if points3d else np.empty((0, 3)))

    def _problem(self, previous: _Frame, frame: _Frame, temporal: dict[str, np.ndarray]):
        """Landmarks from all temporal matches with their stereo views."""
        hosts, anchors, initial_points, views = [], [], [], []
        offset = 0
        for name in self.temporal_cameras:
            matches = temporal[name]
            count = len(matches)
            if count == 0:
                continue
            landmarks = offset + np.arange(count)
            hosts.append(name)
            anchors.append(previous.features[name].keypoints[matches[:, 0]])
            points = np.full((count, 3), np.nan)
            views.append(View(name, name, True, landmarks,
                              frame.features[name].keypoints[matches[:, 1]]))
            current_rows = dict((item.partner, rows) for item, rows in frame.stereo_rows(name))
            current_items = dict((item.partner, item) for item, _ in frame.stereo_rows(name))
            for observations, rows in previous.stereo_rows(name):
                row = rows[matches[:, 0]]
                present = row >= 0
                fill = present & np.isnan(points[:, 0])
                points[fill] = observations.points_rig[row[fill]]
                if np.any(present):
                    views.append(View(observations.partner, name, False, landmarks[present],
                                      observations.partner_pixels[row[present]]))
                partner = observations.partner
                if partner in current_rows:
                    row = current_rows[partner][matches[:, 1]]
                    present = row >= 0
                    if np.any(present):
                        views.append(View(partner, name, True, landmarks[present],
                                          current_items[partner].partner_pixels[row[present]]))
            # Partners only observed at the current frame.
            seen = {item.partner for item, _ in previous.stereo_rows(name)}
            for partner, rows in current_rows.items():
                if partner in seen:
                    continue
                row = rows[matches[:, 1]]
                present = row >= 0
                if np.any(present):
                    views.append(View(partner, name, True, landmarks[present],
                                      current_items[partner].partner_pixels[row[present]]))
            initial_points.append(points)
            offset += count
        return build_problem(hosts, anchors, initial_points, views)

    def _select_landmarks(self, problem, pose: np.ndarray):
        """Keep temporal matches consistent with the PnP pose.

        Consistency uses ``inlier_gate`` times the PnP
        RANSAC threshold: stereo landmarks by reprojection of their previous
        stereo point into the current host view, bearing-only landmarks by
        their Sampson epipolar error under the PnP motion of the host camera.
        """
        options = self.config.refinement
        if len(problem) == 0:
            return problem
        stereo = np.isfinite(problem.initial_points_rig).all(axis=1)
        threshold = options.inlier_gate * self.config.pnp.ransac_max_error_px
        keep = np.zeros(len(problem), dtype=bool)
        for view in problem.views:
            if not view.required:
                continue
            rows = view.landmarks
            camera = self.rig.cameras[view.camera]
            camera_from_previous = self.rig.camera_from_rig(view.camera) @ pose
            points = problem.initial_points_rig[rows]
            projected = camera.project(
                points @ camera_from_previous[:3, :3].T + camera_from_previous[:3, 3])
            consistent = np.linalg.norm(projected - view.pixels, axis=1) <= threshold
            motion = camera_from_previous @ self.rig.rig_from_camera(view.host)
            essential = skew(motion[:3, 3]) @ motion[:3, :3]
            previous = np.column_stack((camera.normalize(problem.anchor_pixels[rows]),
                                        np.ones(len(rows))))
            current = np.column_stack((camera.normalize(view.pixels), np.ones(len(rows))))
            ex1, etx2 = previous @ essential.T, current @ essential
            denominator = np.sqrt(ex1[:, 0] ** 2 + ex1[:, 1] ** 2
                                  + etx2[:, 0] ** 2 + etx2[:, 1] ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                sampson = np.abs(np.sum(current * ex1, axis=1)) / denominator
            # Without translation there is no epipolar constraint.
            sampson = np.where(denominator > 1.0e-12, sampson, 0.0)
            epipolar = sampson <= camera.normalized_threshold(threshold)
            consistent = np.where(stereo[rows], consistent, epipolar)
            keep[rows] = consistent
        return problem.subset(keep)

    def _initial_pose(self, previous: _Frame, frame: _Frame, temporal: dict[str, np.ndarray],
                      pose: np.ndarray, report: dict) -> np.ndarray:
        """PnP translation with the essential-matrix rotation when they agree."""
        name = max(self.temporal_cameras, key=lambda item: len(temporal[item]))
        estimate = essential_rotation(self.rig.cameras[name], previous.features[name].keypoints,
                                      frame.features[name].keypoints, temporal[name],
                                      self.config.temporal)
        initial = pose.copy()
        if estimate is None:
            report["initialization"] = "pnp"
            return initial
        rotation_camera, inliers = estimate
        rig_from_camera = self.rig.rig_from_camera(name)[:3, :3]
        rotation = rig_from_camera @ rotation_camera @ rig_from_camera.T
        difference = float(np.linalg.norm(so3_log(rotation @ pose[:3, :3].T)))
        report["essential_rotation_inliers"] = inliers
        report["essential_pnp_rotation_difference_deg"] = float(np.rad2deg(difference))
        if difference <= np.deg2rad(self.config.initial_rotation_gate_deg):
            initial[:3, :3] = rotation
            report["initialization"] = "essential_rotation_pnp_translation"
        else:
            report["initialization"] = "pnp_essential_rejected"
        return initial

    @staticmethod
    def _solution_report(solution: AnchoredSolution) -> dict[str, Any]:
        return {
            "landmarks": solution.landmarks,
            "partner_observations": solution.partner_observations,
            "iterations": solution.iterations,
            "initial_cost": solution.initial_cost,
            "final_cost": solution.final_cost,
            "rank": solution.rank,
            "condition_number": solution.condition_number,
            "reprojection_rms_px": solution.reprojection_rms_px,
            "covariance_estimator": solution.covariance_estimator,
            "information_eigenvalues": list(solution.information_eigenvalues),
            "inertial_information_eigenvalues": list(solution.inertial_information_eigenvalues),
            "image_residual_rms_px": solution.image_residual_rms_px,
            "image_variance_scales": solution.image_variance_scales,
            "robust_weight_stats": list(solution.robust_weight_stats),
        }

    def _track_observation(self, frame: _Frame, camera: str, index: int) -> TrackObservation:
        x, y = frame.features[camera].keypoints[index, :2]
        for observations, rows in frame.stereo_rows(camera):
            row = rows[index]
            if row >= 0 and observations.metric[row]:
                partner = observations.partner_pixels[row]
                return TrackObservation(
                    frame.stamp_ns, frame.index, int(index), float(x), float(y), True,
                    float(partner[0]), float(partner[1]),
                    tuple(float(v) for v in observations.points_host[row]),
                    float(observations.angles[row]))
        return TrackObservation(frame.stamp_ns, frame.index, int(index), float(x), float(y))


class _FrameFailure(RuntimeError):
    pass
