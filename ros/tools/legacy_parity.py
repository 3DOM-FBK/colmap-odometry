#!/usr/bin/env python3
"""Compare this package with the legacy implementation on a rosbag, offline.

The legacy code (the ``main`` branch checkout) is run in a subprocess on the
same synchronized frames; both relative motions are expressed in the
published base frame (ROS covariance order) and compared per frame::

    tools/legacy_parity.py --legacy-root ~/ws/src/colmap-odometry \\
        --legacy-params config/tumvi.yaml --params config/tumvi.yaml \\
        --bag ~/data/tum_vi_room1 --max-frames 300 -o /tmp/parity

The new run uses ``colmap_odometry.offline``; the legacy run reads the same
bag with the same synchronizer.  Feature backends must be equivalent (e.g.
LoMa without compilation) for a meaningful comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_LEGACY_RUNNER = r'''
import json, sys
from pathlib import Path
import numpy as np, yaml
legacy_root, params_file, bag, out, max_frames, start_sec = sys.argv[1:7]
sys.path.insert(0, legacy_root)
new_root = sys.argv[7]
from colmap_odometry.odometry.odometry import VisualOdometry

def _adjoint_rotation_translation(T):  # legacy node helper, inlined (no ROS)
    R = T[:3, :3]; t = T[:3, 3]
    S = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    A = np.zeros((6, 6)); A[:3, :3] = R; A[3:, :3] = S @ R; A[3:, 3:] = R
    return A

def _pose_to_matrix(t, q):
    T = np.eye(4); T[:3, :3] = q.rotation_matrix; T[:3, 3] = t
    return T

def _flat_transform(values, name):
    return np.asarray(values, dtype=float).reshape(4, 4)
spec = __import__("importlib.util").util.spec_from_file_location(
    "bagreader", Path(new_root) / "colmap_odometry" / "io" / "rosbag.py")
bagreader = __import__("importlib.util").util.module_from_spec(spec)
sys.modules["bagreader"] = bagreader
spec.loader.exec_module(bagreader)

def load(path):
    return yaml.safe_load(open(path))
params = next(v for v in load(params_file).values())["ros__parameters"]
config_dir = Path(legacy_root) / "config"
def find(name):
    for d in (config_dir, Path(legacy_root) / "calibration"):
        if (d / name).exists(): return d / name
    raise FileNotFoundError(name)
config = load(find(params["config_file"]))
config.setdefault("local_features", {}).setdefault("loma", {})["compile"] = False
config["general"]["create_feature_tracks"] = False
config["mapping"]["cameras"] = params["camera_names"]
cameras = load(find(params["camera_file"])); rig = load(find(params["rig_file"]))
work = Path(out) / "legacy_work"; work.mkdir(parents=True, exist_ok=True)
vo = VisualOdometry(work, config, cameras, rig)
T_cb = _flat_transform(params["T_camera_base"], "T_camera_base"); T_bc = np.linalg.inv(T_cb)
reader = bagreader.BagReader(bag)
records = []
previous = None
(Path(out) / "legacy").mkdir(parents=True, exist_ok=True)
trajectory = open(Path(out) / "legacy" / "trajectory.tum", "w")
motions_file = open(Path(out) / "legacy" / "relative_motion.jsonl", "w")

def tum(stamp, T):
    from pyquaternion import Quaternion
    q = Quaternion(matrix=T[:3, :3], atol=1e-6)
    values = [*T[:3, 3], q.x, q.y, q.z, q.w]
    trajectory.write(f"{stamp // 10**9}.{stamp % 10**9:09d} " + " ".join(f"{v:.9f}" for v in values) + "\n")

for frame in reader.frames(params["image_topics"], float(params.get("sync_slop", 0.005)),
                           float(start_sec), int(max_frames)):
    name = f"{frame.stamp_ns // 10**9}_{frame.stamp_ns % 10**9:09d}.png"
    changes, log = vo.run(name, frame.images, reinitialize=False)
    change = changes[-1]
    current = log["current_frame"]
    if previous is None:
        tum(frame.stamp_ns, np.eye(4))
    if change[1] is not None:
        tum(frame.stamp_ns, T_bc @ _pose_to_matrix(np.asarray(change[4]), change[5]) @ T_cb)
        delta = _pose_to_matrix(np.asarray(change[2]), change[3])
        Z = T_bc @ delta @ T_cb
        cov = current.get("relative_pose_covariance_rotation_translation")
        A = _adjoint_rotation_translation(T_bc)
        C = A @ cov @ A.T
        L = _adjoint_rotation_translation(np.linalg.inv(Z))
        C = L @ C @ L.T
        order = np.array([3, 4, 5, 0, 1, 2]); C = C[np.ix_(order, order)]
        from pyquaternion import Quaternion
        qz = Quaternion(matrix=Z[:3, :3], atol=1e-6)
        motions_file.write(json.dumps({"from_stamp_ns": previous, "to_stamp_ns": frame.stamp_ns,
            "translation": Z[:3, 3].tolist(), "quaternion_xyzw": [qz.x, qz.y, qz.z, qz.w],
            "covariance_ros": C.reshape(-1).tolist(),
            "covariance_source": current.get("pnp_covariance_source") or "none",
            "inlier_count": current.get("pnp_ransac_inliers")}) + "\n")
        records.append({"from_stamp_ns": previous, "to_stamp_ns": frame.stamp_ns,
                        "motion": Z.tolist(), "covariance_ros": C.tolist(),
                        "inliers": current.get("pnp_ransac_inliers")})
    previous = frame.stamp_ns
trajectory.close(); motions_file.close()
json.dump(records, open(Path(out) / "legacy.json", "w"))
print(f"legacy: {len(records)} motions", file=sys.stderr)
'''


def _transform(translation, quaternion):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from colmap_odometry.geometry.lie import transform_from_pose
    return transform_from_pose(translation, quaternion)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--legacy-params", type=Path, required=True,
                        help="legacy node parameter file (in the legacy tree)")
    parser.add_argument("--params", type=Path, required=True, help="new node parameter file")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--skip-legacy", action="store_true", help="reuse legacy.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    if not args.skip_legacy:
        subprocess.run([sys.executable, "-c", _LEGACY_RUNNER, str(args.legacy_root),
                        str(args.legacy_params.resolve()), str(args.bag), str(args.output),
                        str(args.max_frames), str(args.start_sec), str(root)],
                       check=True, cwd=args.output)
    subprocess.run([sys.executable, "-m", "colmap_odometry.offline", str(args.params),
                    str(args.bag), "-o", str(args.output / "new"), "--max-frames",
                    str(args.max_frames), "--start-sec", str(args.start_sec), "--quiet"],
                   check=True, cwd=root)

    legacy = {item["to_stamp_ns"]: item for item in json.load(open(args.output / "legacy.json"))}
    new = {}
    for line in open(args.output / "new" / "relative_motion.jsonl"):
        item = json.loads(line)
        new[item["to_stamp_ns"]] = item
    common = sorted(set(legacy) & set(new))
    print(f"motions: legacy {len(legacy)}, new {len(new)}, common {len(common)}")
    translation_errors, rotation_errors, covariance_ratios = [], [], []
    for stamp in common:
        reference = np.asarray(legacy[stamp]["motion"])
        candidate = _transform(new[stamp]["translation"], new[stamp]["quaternion_xyzw"])
        translation_errors.append(np.linalg.norm(candidate[:3, 3] - reference[:3, 3]))
        cosine = (np.trace(candidate[:3, :3].T @ reference[:3, :3]) - 1) / 2
        rotation_errors.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        reference_cov = np.asarray(legacy[stamp]["covariance_ros"])
        candidate_cov = np.asarray(new[stamp]["covariance_ros"]).reshape(6, 6)
        covariance_ratios.append(np.diag(candidate_cov) / np.diag(reference_cov))
    translation_errors = np.asarray(translation_errors)
    rotation_errors = np.asarray(rotation_errors)
    ratios = np.asarray(covariance_ratios)

    def summary(values):
        return (f"median {np.median(values):.3g}, p95 {np.percentile(values, 95):.3g}, "
                f"max {np.max(values):.3g}")

    print(f"translation difference [m]: {summary(translation_errors)}")
    print(f"rotation difference [deg]: {summary(rotation_errors)}")
    print(f"covariance diagonal ratio new/legacy: median {np.median(ratios, axis=0).round(4)}"
          f", p5 {np.percentile(ratios, 5, axis=0).round(3)}"
          f", p95 {np.percentile(ratios, 95, axis=0).round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
