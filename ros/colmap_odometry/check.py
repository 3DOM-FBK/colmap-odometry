"""Accuracy regression check of a candidate run against a reference run.

Both runs are evaluated against the same ground truth; the candidate passes
when it is not worse than the reference beyond the tolerances::

    colmap_odometry_check --ground-truth gt.tum \\
        --reference ref/trajectory.tum --reference-motion ref/relative_motion \\
        --candidate dev/trajectory.tum --candidate-motion dev/relative_motion

Trajectories are TUM files (``timestamp tx ty tz qx qy qz qw``, as written
by the node with ``save_trajectory_tum``) or rosbag2 recordings of the
``nav_msgs/Odometry`` topic.  Relative motions are optional and are either
rosbag2 recordings of ``~/relative_motion`` (``ros2 bag record
/colmap_odometry_node/relative_motion``) or the ``relative_motion.jsonl`` of
``colmap_odometry_offline``.

Metrics (ground truth interpolated at estimate timestamps):

* APE: translation RMSE after SE(3) alignment (stereo odometry is metric);
* RPE: translation and rotation RMSE over a fixed time delta;
* NEES: Mahalanobis error of relative motions composed over windows of
  ``--nees-window`` seconds, normalized by 6 (1.0 = consistent);
* the number of published relative motions (tracking failures).

The ground truth must describe the same body frame as the odometry output.
Exit status 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry.lie import (
    adjoint,
    invert,
    make_transform,
    quaternion_to_rotation,
    ros_to_rt,
    se3_log,
    so3_exp,
    so3_log,
    transport,
)

RELATIVE_MOTION_TYPE = "colmap_odometry_msgs/msg/RelativePoseWithCovarianceStamped"


# ------------------------------------------------------------------- inputs
@dataclass
class Trajectory:
    stamps: np.ndarray  # seconds
    poses: np.ndarray   # (N, 4, 4) world_from_body

    def __len__(self) -> int:
        return len(self.stamps)


@dataclass
class Motion:
    from_stamp: float
    to_stamp: float
    transform: np.ndarray
    covariance_rt: np.ndarray  # right-local [rotation, translation]
    source: str


def _pose(values) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = (float(v) for v in values)
    return make_transform(quaternion_to_rotation([qx, qy, qz, qw]), [x, y, z])


def read_trajectory(path: Path, topic: str | None = None) -> Trajectory:
    if path.is_dir() or path.suffix in (".mcap", ".db3"):
        return _trajectory_from_bag(path, topic)
    stamps, poses = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.replace(",", " ").split()
        stamps.append(float(values[0]))
        poses.append(_pose(values[1:8]))
    order = np.argsort(stamps)
    return Trajectory(np.asarray(stamps)[order], np.asarray(poses)[order])


def _stamp(time) -> float:
    return int(time.sec) + 1e-9 * int(time.nanosec)


def _bag_messages(path: Path, topic_filter, typestore=None):
    from .io.rosbag import iter_messages

    if path.is_dir() and not (path / "metadata.yaml").exists():
        raise FileNotFoundError(
            f"{path} has no metadata.yaml (recording not finalized); "
            f"repair it with: ros2 bag reindex {path}")
    for _, _, message in iter_messages(path, topic_filter, typestore):
        yield message


def _trajectory_from_bag(path: Path, topic: str | None) -> Trajectory:
    stamps, poses = [], []
    for message in _bag_messages(
            path, lambda c: c.msgtype == "nav_msgs/msg/Odometry"
            and (topic is None or c.topic == topic)):
        pose = message.pose.pose
        stamps.append(_stamp(message.header.stamp))
        poses.append(_pose([pose.position.x, pose.position.y, pose.position.z,
                            pose.orientation.x, pose.orientation.y,
                            pose.orientation.z, pose.orientation.w]))
    order = np.argsort(stamps)
    return Trajectory(np.asarray(stamps)[order], np.asarray(poses)[order])


def _relative_motion_typestore():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    msg_dir = None
    for root in Path(__file__).resolve().parents:
        candidate = root / "colmap_odometry_msgs" / "msg"
        if candidate.is_dir():
            msg_dir = candidate
            break
    typestore = get_typestore(Stores.ROS2_JAZZY)
    if msg_dir is not None:
        types = {}
        for name in ("FeatureObservation", "FeatureTrack", "FeatureTrackArray",
                     "RelativePoseWithCovarianceStamped"):
            types.update(get_types_from_msg((msg_dir / f"{name}.msg").read_text(),
                                            f"colmap_odometry_msgs/msg/{name}"))
        typestore.register(types)
    return typestore


def read_motions(path: Path) -> list[Motion]:
    motions = []
    if path.is_file() and path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item.get("from_stamp_ns") is None:
                continue
            motions.append(Motion(
                item["from_stamp_ns"] * 1e-9, item["to_stamp_ns"] * 1e-9,
                make_transform(quaternion_to_rotation(item["quaternion_xyzw"]),
                               item["translation"]),
                ros_to_rt(np.reshape(item["covariance_ros"], (6, 6))),
                str(item.get("covariance_source", ""))))
        return motions
    for message in _bag_messages(path, lambda c: c.msgtype == RELATIVE_MOTION_TYPE,
                                 _relative_motion_typestore()):
        if message.from_stamp.sec == 0 and message.from_stamp.nanosec == 0:
            continue
        pose = message.relative_pose.pose
        motions.append(Motion(
            _stamp(message.from_stamp), _stamp(message.to_stamp),
            _pose([pose.position.x, pose.position.y, pose.position.z,
                   pose.orientation.x, pose.orientation.y, pose.orientation.z,
                   pose.orientation.w]),
            ros_to_rt(np.reshape(message.relative_pose.covariance, (6, 6))),
            str(message.covariance_source)))
    return motions


# ------------------------------------------------------------------ metrics
def interpolate(truth: Trajectory, stamps: np.ndarray, max_gap: float) -> np.ndarray:
    """Ground-truth poses at ``stamps`` (NaN where not bracketed closely)."""
    result = np.full((len(stamps), 4, 4), np.nan)
    index = np.searchsorted(truth.stamps, stamps)
    for row, (stamp, upper) in enumerate(zip(stamps, index)):
        if upper == 0 or upper >= len(truth):
            if upper < len(truth) and abs(truth.stamps[upper] - stamp) < 1e-6:
                result[row] = truth.poses[upper]
            continue
        lower = upper - 1
        t0, t1 = truth.stamps[lower], truth.stamps[upper]
        if t1 - t0 > max_gap:
            continue
        alpha = (stamp - t0) / (t1 - t0)
        p0, p1 = truth.poses[lower], truth.poses[upper]
        rotation = p0[:3, :3] @ so3_exp(alpha * so3_log(p0[:3, :3].T @ p1[:3, :3]))
        result[row] = make_transform(rotation, (1 - alpha) * p0[:3, 3] + alpha * p1[:3, 3])
    return result


def associate(estimate: Trajectory, truth: Trajectory, max_gap: float):
    reference = interpolate(truth, estimate.stamps, max_gap)
    valid = np.isfinite(reference).all(axis=(1, 2))
    return estimate.stamps[valid], estimate.poses[valid], reference[valid]


def align_se3(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Umeyama rigid alignment of points (no scale): ``target ~ T source``."""
    mean_source, mean_target = source.mean(axis=0), target.mean(axis=0)
    covariance = (target - mean_target).T @ (source - mean_source) / len(source)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    return make_transform(rotation, mean_target - rotation @ mean_source)


def ape(poses: np.ndarray, truth: np.ndarray) -> dict:
    alignment = align_se3(poses[:, :3, 3], truth[:, :3, 3])
    aligned = np.einsum("ij,njk->nik", alignment, poses)
    translation = np.linalg.norm(aligned[:, :3, 3] - truth[:, :3, 3], axis=1)
    rotation = np.array([np.linalg.norm(so3_log(a[:3, :3].T @ b[:3, :3]))
                         for a, b in zip(aligned, truth)])
    return {"ape_translation_rmse_m": float(np.sqrt(np.mean(translation ** 2))),
            "ape_rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(rotation ** 2))))}


def rpe(stamps: np.ndarray, poses: np.ndarray, truth: np.ndarray, delta: float) -> dict:
    translation, rotation = [], []
    for start in range(len(stamps)):
        end = int(np.searchsorted(stamps, stamps[start] + delta))
        if end >= len(stamps) or abs(stamps[end] - stamps[start] - delta) > 0.5 * delta:
            continue
        estimated = invert(poses[start]) @ poses[end]
        reference = invert(truth[start]) @ truth[end]
        error = invert(reference) @ estimated
        translation.append(np.linalg.norm(error[:3, 3]))
        rotation.append(np.linalg.norm(so3_log(error[:3, :3])))
    if not translation:
        return {"rpe_translation_rmse_m": float("nan"), "rpe_rotation_rmse_deg": float("nan"),
                "rpe_pairs": 0}
    return {"rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(translation)))),
            "rpe_rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(np.square(rotation))))),
            "rpe_pairs": len(translation)}


def _windows(motions: list[Motion], truth: Trajectory, window: float, max_gap: float):
    """Compose motion chains over ``window`` seconds; yield (error, covariance).

    Covariances are composed assuming independent intervals:
    ``Z = Z1 Z2`` gives ``e = Ad(Z2^-1) e1 + e2`` for right-local errors.
    """
    motions = sorted(motions, key=lambda item: item.to_stamp)
    chain = None
    for motion in motions:
        if chain is not None and abs(motion.from_stamp - chain[1]) > 1e-6:
            chain = None  # gap: tracking failure or dropped message
        if chain is None:
            chain = [motion.from_stamp, motion.to_stamp, motion.transform.copy(),
                     motion.covariance_rt.copy()]
        else:
            chain[3] = transport(chain[3], adjoint(invert(motion.transform))) \
                + motion.covariance_rt
            chain[2] = chain[2] @ motion.transform
            chain[1] = motion.to_stamp
        if chain[1] - chain[0] >= window:
            endpoints = interpolate(truth, np.array([chain[0], chain[1]]), max_gap)
            if np.isfinite(endpoints).all():
                reference = invert(endpoints[0]) @ endpoints[1]
                yield se3_log(invert(chain[2]) @ reference), chain[3]
            chain = None


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3:
        return float("nan")
    ranks = [np.argsort(np.argsort(values)) for values in (first, second)]
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


def nees(motions: list[Motion], truth: Trajectory, window: float, max_gap: float) -> dict:
    """NEES/6 of motion chains composed over ``window`` seconds, plus how well
    the predicted standard deviations rank the actual errors (Spearman)."""
    values, errors, sigmas = [], [], []
    for error, covariance in _windows(motions, truth, window, max_gap):
        try:
            values.append(float(error @ np.linalg.solve(covariance, error)) / 6.0)
        except np.linalg.LinAlgError:
            continue
        errors.append((np.linalg.norm(error[3:]), np.linalg.norm(error[:3])))
        sigmas.append((np.sqrt(np.trace(covariance[3:, 3:])),
                       np.sqrt(np.trace(covariance[:3, :3]))))
    if not values:
        return {"nees_mean": float("nan"), "nees_median": float("nan"), "nees_windows": 0}
    errors, sigmas = np.asarray(errors), np.asarray(sigmas)
    return {"nees_mean": float(np.mean(values)), "nees_median": float(np.median(values)),
            "nees_windows": len(values),
            "sigma_rank_correlation_translation": _rank_correlation(sigmas[:, 0], errors[:, 0]),
            "sigma_rank_correlation_rotation": _rank_correlation(sigmas[:, 1], errors[:, 1])}


def evaluate(trajectory: Trajectory, motions: list[Motion] | None, truth: Trajectory,
             rpe_delta: float, nees_window: float, max_gap: float) -> dict:
    stamps, poses, reference = associate(trajectory, truth, max_gap)
    if len(stamps) < 3:
        raise ValueError("fewer than three trajectory poses overlap the ground truth")
    result = {"poses": int(len(trajectory)), "poses_with_ground_truth": int(len(stamps))}
    result.update(ape(poses, reference))
    result.update(rpe(stamps, poses, reference, rpe_delta))
    if motions is not None:
        result["relative_motions"] = len(motions)
        sources: dict[str, int] = {}
        for motion in motions:
            sources[motion.source] = sources.get(motion.source, 0) + 1
        result["covariance_sources"] = sources
        result.update(nees(motions, truth, nees_window, max_gap))
    return result


# --------------------------------------------------------------- comparison
@dataclass
class Tolerances:
    error_increase: float = 0.01   # APE/RPE may grow by at most 1 %
    nees_change: float = 0.05      # NEES within +-5 % (or consistency margin)
    motion_loss: float = 0.01      # at most 1 % fewer relative motions
    #: ``ratio``: candidate NEES within +-nees_change of the reference.
    #: ``consistency``: candidate NEES no further from 1 (in log scale) than
    #: the reference, plus the margin; use when the covariance estimator
    #: intentionally changed.
    nees_criterion: str = "ratio"


def compare(reference: dict, candidate: dict, tolerances: Tolerances) -> list[tuple]:
    rows = []
    for key in ("ape_translation_rmse_m", "ape_rotation_rmse_deg",
                "rpe_translation_rmse_m", "rpe_rotation_rmse_deg"):
        limit = reference[key] * (1.0 + tolerances.error_increase)
        rows.append((key, reference[key], candidate[key], f"<= {limit:.6g}",
                     bool(candidate[key] <= limit)))
    if "nees_mean" in reference and "nees_mean" in candidate:
        ratio = candidate["nees_mean"] / reference["nees_mean"]
        if tolerances.nees_criterion == "consistency":
            limit = abs(np.log(reference["nees_mean"])) + np.log1p(tolerances.nees_change)
            rows.append(("nees_mean", reference["nees_mean"], candidate["nees_mean"],
                         f"|log NEES| {abs(np.log(candidate['nees_mean'])):.3f} <= {limit:.3f}",
                         bool(abs(np.log(candidate["nees_mean"])) <= limit)))
        else:
            rows.append(("nees_mean", reference["nees_mean"], candidate["nees_mean"],
                         f"ratio {ratio:.3f} in 1+-{tolerances.nees_change}",
                         bool(abs(ratio - 1.0) <= tolerances.nees_change)))
        limit = reference["relative_motions"] * (1.0 - tolerances.motion_loss)
        rows.append(("relative_motions", reference["relative_motions"],
                     candidate["relative_motions"], f">= {limit:.0f}",
                     bool(candidate["relative_motions"] >= limit)))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="reference trajectory (TUM file or bag)")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="candidate trajectory (TUM file or bag)")
    parser.add_argument("--reference-motion", type=Path,
                        help="reference relative motions (bag or .jsonl)")
    parser.add_argument("--candidate-motion", type=Path,
                        help="candidate relative motions (bag or .jsonl)")
    parser.add_argument("--odom-topic", help="Odometry topic when trajectories are bags")
    parser.add_argument("--rpe-delta", type=float, default=1.0, help="RPE interval [s]")
    parser.add_argument("--nees-window", type=float, default=1.0, help="NEES window [s]")
    parser.add_argument("--max-gt-gap", type=float, default=0.1,
                        help="maximum ground-truth sample spacing to interpolate [s]")
    parser.add_argument("--error-tolerance", type=float, default=0.01)
    parser.add_argument("--nees-tolerance", type=float, default=0.05)
    parser.add_argument("--motion-tolerance", type=float, default=0.01)
    parser.add_argument("--nees-criterion", choices=("ratio", "consistency"), default="ratio",
                        help="ratio: NEES within the tolerance of the reference; "
                             "consistency: NEES not further from 1 than the reference")
    parser.add_argument("--json", type=Path, help="write all metrics to this file")
    args = parser.parse_args(argv)
    if (args.reference_motion is None) != (args.candidate_motion is None):
        parser.error("give relative motions for both runs or for neither")

    truth = read_trajectory(args.ground_truth)
    results = {}
    for name, trajectory_path, motion_path in (
            ("reference", args.reference, args.reference_motion),
            ("candidate", args.candidate, args.candidate_motion)):
        motions = read_motions(motion_path) if motion_path is not None else None
        results[name] = evaluate(read_trajectory(trajectory_path, args.odom_topic), motions,
                                 truth, args.rpe_delta, args.nees_window, args.max_gt_gap)
    rows = compare(results["reference"], results["candidate"],
                   Tolerances(args.error_tolerance, args.nees_tolerance, args.motion_tolerance,
                              args.nees_criterion))

    width = max(len(row[0]) for row in rows)
    print(f"{'metric':<{width}}  {'reference':>12}  {'candidate':>12}  criterion")
    for key, reference, candidate, criterion, passed in rows:
        print(f"{key:<{width}}  {reference:>12.6g}  {candidate:>12.6g}  {criterion}"
              f"  {'PASS' if passed else 'FAIL'}")
    for name in ("reference", "candidate"):
        result = results[name]
        print(f"{name}: {result['poses_with_ground_truth']}/{result['poses']} poses with "
              f"ground truth, RPE pairs {result.get('rpe_pairs')}, NEES windows "
              f"{result.get('nees_windows', '-')}, covariance sources "
              f"{result.get('covariance_sources', '-')}")
        if "sigma_rank_correlation_translation" in result:
            print(f"  covariance ranks actual errors (Spearman, 1 = perfect): translation "
                  f"{result['sigma_rank_correlation_translation']:.2f}, rotation "
                  f"{result['sigma_rank_correlation_rotation']:.2f}")
    if args.json is not None:
        args.json.write_text(json.dumps({"results": results, "checks": [
            {"metric": r[0], "reference": r[1], "candidate": r[2], "criterion": r[3],
             "passed": r[4]} for r in rows]}, indent=2))
    passed = all(row[4] for row in rows)
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
