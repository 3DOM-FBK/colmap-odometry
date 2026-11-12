"""Calibrated stereo filtering and triangulation for a fixed camera pair."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Rig
from ..geometry.lie import transform_points


@dataclass(frozen=True)
class StereoGates:
    epipolar_max_error_px: float = 2.0
    reprojection_max_error_px: float = 3.0
    min_triangulation_angle_deg: float = 0.2
    min_depth: float = 0.2
    max_depth: float = 20.0

    def __post_init__(self):
        if self.epipolar_max_error_px <= 0.0 or self.reprojection_max_error_px <= 0.0:
            raise ValueError("stereo error thresholds must be positive")
        if self.min_triangulation_angle_deg < 0.0:
            raise ValueError("stereo min_triangulation_angle_deg cannot be negative")
        if not 0.0 <= self.min_depth < self.max_depth:
            raise ValueError("stereo depth limits must satisfy 0 <= min < max")


@dataclass
class StereoStats:
    raw_matches: int = 0
    epipolar_matches: int = 0
    triangulated: int = 0
    rejected_triangulation: int = 0
    rejected_cheirality: int = 0
    rejected_depth: int = 0
    rejected_reprojection: int = 0
    rejected_angle: int = 0
    bearing_valid: int = 0
    metric_valid: int = 0


@dataclass
class StereoObservations:
    """Triangulated stereo matches of one (host, partner) pair at one frame.

    Every row passed cheirality and reprojection checks, so its partner
    pixel is a valid stereo observation of the host keypoint.  ``metric``
    marks rows that also pass the depth and triangulation-angle gates and can
    serve as metric 3-D points (PnP, published stereo points).
    """

    host: str
    partner: str
    host_index: np.ndarray
    partner_index: np.ndarray
    host_pixels: np.ndarray
    partner_pixels: np.ndarray
    points_rig: np.ndarray
    points_host: np.ndarray
    angles: np.ndarray
    metric: np.ndarray
    stats: StereoStats = field(default_factory=StereoStats)

    @classmethod
    def empty(cls, host: str, partner: str, stats: StereoStats | None = None):
        return cls(host, partner,
                   np.empty(0, np.int64), np.empty(0, np.int64),
                   np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 3)),
                   np.empty(0), np.empty(0, bool), stats or StereoStats())

    def __len__(self) -> int:
        return int(len(self.host_index))

    def rows_by_host_index(self, keypoint_count: int) -> np.ndarray:
        """Dense lookup from host keypoint index to row (-1 if absent)."""
        lookup = np.full(int(keypoint_count), -1, dtype=np.int64)
        lookup[self.host_index] = np.arange(len(self.host_index))
        return lookup


def epipolar_filter(
    rig: Rig,
    first: str,
    second: str,
    keypoints_first: np.ndarray,
    keypoints_second: np.ndarray,
    matches: np.ndarray,
    max_error_px: float,
) -> np.ndarray:
    """Keep matches consistent with the calibrated essential matrix.

    Uses the Sampson distance in normalized image coordinates, with the
    pixel threshold converted by each camera model (the larger is used).
    """
    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    if len(matches) == 0:
        return matches
    camera1, camera2 = rig.cameras[first], rig.cameras[second]
    essential = rig.essential(first, second)
    normalized1 = camera1.normalize(np.asarray(keypoints_first)[matches[:, 0], :2])
    normalized2 = camera2.normalize(np.asarray(keypoints_second)[matches[:, 1], :2])
    homogeneous1 = np.column_stack((normalized1, np.ones(len(normalized1))))
    homogeneous2 = np.column_stack((normalized2, np.ones(len(normalized2))))
    ex1 = homogeneous1 @ essential.T
    etx2 = homogeneous2 @ essential
    numerator = np.sum(homogeneous2 * ex1, axis=1)
    denominator = np.sqrt(ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2)
    error = np.full(len(matches), np.inf)
    valid = denominator > np.finfo(np.float64).eps
    error[valid] = np.abs(numerator[valid]) / denominator[valid]
    threshold = max(camera1.normalized_threshold(max_error_px),
                    camera2.normalized_threshold(max_error_px))
    return matches[np.isfinite(error) & (error <= threshold)]


def triangulate(
    rig: Rig,
    host: str,
    partner: str,
    keypoints_host: np.ndarray,
    keypoints_partner: np.ndarray,
    matches: np.ndarray,
    gates: StereoGates,
    stats: StereoStats | None = None,
) -> StereoObservations:
    """Triangulate epipolar-filtered matches with linear (DLT) triangulation.

    Points are expressed in the rig frame.  Rows failing cheirality or
    reprojection are dropped; depth and angle only clear ``metric``.
    """
    stats = stats or StereoStats()
    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    keypoints_host = np.asarray(keypoints_host, dtype=np.float64)
    keypoints_partner = np.asarray(keypoints_partner, dtype=np.float64)
    valid_index = ((matches[:, 0] >= 0) & (matches[:, 0] < len(keypoints_host))
                   & (matches[:, 1] >= 0) & (matches[:, 1] < len(keypoints_partner)))
    matches = matches[valid_index]
    if len(matches) == 0:
        return StereoObservations.empty(host, partner, stats)

    camera_host, camera_partner = rig.cameras[host], rig.cameras[partner]
    host_from_rig = rig.camera_from_rig(host)
    partner_from_rig = rig.camera_from_rig(partner)
    center_host = rig.rig_from_camera(host)[:3, 3]
    center_partner = rig.rig_from_camera(partner)[:3, 3]
    index_host = matches[:, 0]
    index_partner = matches[:, 1]
    pixels_host = keypoints_host[index_host, :2]
    pixels_partner = keypoints_partner[index_partner, :2]
    normalized_host = camera_host.normalize(pixels_host)
    normalized_partner = camera_partner.normalize(pixels_partner)
    projection_host = host_from_rig[:3, :]
    projection_partner = partner_from_rig[:3, :]
    systems = np.stack((
        normalized_host[:, 0, None] * projection_host[2] - projection_host[0],
        normalized_host[:, 1, None] * projection_host[2] - projection_host[1],
        normalized_partner[:, 0, None] * projection_partner[2] - projection_partner[0],
        normalized_partner[:, 1, None] * projection_partner[2] - projection_partner[1],
    ), axis=1)
    finite_systems = np.isfinite(systems).all(axis=(1, 2))
    homogeneous = np.full((len(systems), 4), np.nan)
    if np.any(finite_systems):
        _, _, vh = np.linalg.svd(systems[finite_systems])
        homogeneous[finite_systems] = vh[:, -1, :]
    triangulated = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1.0e-12)
    stats.rejected_triangulation += int((~triangulated).sum())
    stats.triangulated += int(triangulated.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        points_rig = homogeneous[:, :3] / homogeneous[:, 3, None]
    points_host = transform_points(host_from_rig, points_rig)
    points_partner = transform_points(partner_from_rig, points_rig)
    cheirality = (triangulated
                  & np.isfinite(points_rig).all(axis=1)
                  & np.isfinite(points_host).all(axis=1)
                  & np.isfinite(points_partner).all(axis=1)
                  & (points_host[:, 2] > 0.0) & (points_partner[:, 2] > 0.0))
    stats.rejected_cheirality += int((triangulated & ~cheirality).sum())
    depth = (cheirality
             & (points_host[:, 2] >= gates.min_depth) & (points_partner[:, 2] >= gates.min_depth)
             & (points_host[:, 2] <= gates.max_depth) & (points_partner[:, 2] <= gates.max_depth))
    stats.rejected_depth += int((cheirality & ~depth).sum())
    # Reprojection validity is independent of the metric depth interval so
    # far, low-parallax points remain usable as bearings.
    reprojection = np.zeros(len(matches), dtype=bool)
    if np.any(cheirality):
        projected_host = camera_host.project(points_host[cheirality])
        projected_partner = camera_partner.project(points_partner[cheirality])
        reprojection[cheirality] = (
            np.isfinite(projected_host).all(axis=1)
            & np.isfinite(projected_partner).all(axis=1)
            & (np.maximum(
                np.linalg.norm(projected_host - pixels_host[cheirality], axis=1),
                np.linalg.norm(projected_partner - pixels_partner[cheirality], axis=1),
            ) <= gates.reprojection_max_error_px))
    stats.rejected_reprojection += int((depth & ~reprojection).sum())
    ray_host = points_rig - center_host
    ray_partner = points_rig - center_partner
    denominator = np.linalg.norm(ray_host, axis=1) * np.linalg.norm(ray_partner, axis=1)
    angles = np.zeros(len(matches))
    valid_angle = reprojection & (denominator > 1.0e-12)
    angles[valid_angle] = np.arccos(np.clip(
        np.einsum("ij,ij->i", ray_host[valid_angle], ray_partner[valid_angle])
        / denominator[valid_angle], -1.0, 1.0))
    metric = depth & valid_angle & (angles >= np.deg2rad(gates.min_triangulation_angle_deg))
    stats.rejected_angle += int((reprojection & ~metric).sum())

    rows = cheirality & reprojection
    stats.bearing_valid += int(rows.sum())
    stats.metric_valid += int(metric.sum())
    return StereoObservations(
        host=host,
        partner=partner,
        host_index=index_host[rows],
        partner_index=index_partner[rows],
        host_pixels=pixels_host[rows].copy(),
        partner_pixels=pixels_partner[rows].copy(),
        points_rig=points_rig[rows].copy(),
        points_host=points_host[rows].copy(),
        angles=angles[rows],
        metric=metric[rows],
        stats=stats,
    )


def stereo_observations(
    rig: Rig,
    host: str,
    partner: str,
    keypoints_host: np.ndarray,
    keypoints_partner: np.ndarray,
    raw_matches: np.ndarray,
    gates: StereoGates,
) -> tuple[StereoObservations, np.ndarray]:
    """Epipolar-filter and triangulate one stereo pair.

    Returns the observations and the epipolar-consistent matches.
    """
    stats = StereoStats(raw_matches=int(len(raw_matches)))
    filtered = epipolar_filter(rig, host, partner, keypoints_host, keypoints_partner,
                               raw_matches, gates.epipolar_max_error_px)
    stats.epipolar_matches = int(len(filtered))
    observations = triangulate(rig, host, partner, keypoints_host, keypoints_partner,
                               filtered, gates, stats)
    return observations, filtered
