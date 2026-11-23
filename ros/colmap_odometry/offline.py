"""Run the odometry offline (deterministic, no ROS or playback).

The input is a rosbag2 (sqlite3 or MCAP) or a folder of images with one
directory per camera (EuRoC/ASL layout, e.g. ``mav0/cam0/data.csv``, or
timestamp-named images; see ``colmap_odometry.io.folder``).  Folder runs
are visual only: the IMU configuration is ignored.  Outputs mirror
the ROS node: a TUM trajectory of the body frame (identical to ``~/odom``)
and a JSON-lines file of relative motions with the ``~/relative_motion``
content (ROS-ordered covariance)::

    colmap_odometry_offline config/tumvi.yaml /data/tum_vi_room1 -o out/
    colmap_odometry_offline config/tumvi.yaml /data/dataset-room1_1024_16 -o out/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .diagnostics import json_default
from .features.cache import CachedFrontend
from .geometry.lie import pose_from_transform
from .imu.pipeline import ImuFrameReport
from .io.folder import FolderReader, is_dataset_folder
from .io.rosbag import BagReader
from .odometry import Status
from .setup_pipeline import build, node_parameters


def _tum_line(stamp_ns: int, transform: np.ndarray) -> str:
    translation, quaternion = pose_from_transform(transform)
    values = " ".join(f"{value:.9f}" for value in (*translation, *quaternion))
    return f"{stamp_ns // 1_000_000_000}.{stamp_ns % 1_000_000_000:09d} {values}\n"


def is_bag(path: Path) -> bool:
    if path.is_file():
        return path.suffix in (".db3", ".mcap")
    return ((path / "metadata.yaml").is_file() or any(path.glob("*.db3"))
            or any(path.glob("*.mcap")))


def run(params_file: Path, source: Path, output: Path, max_frames: int | None = None,
        start_sec: float = 0.0, sync_slop_sec: float | None = None,
        report: bool = False, quiet: bool = False,
        feature_cache: Path | None = None, input_format: str = "auto",
        image_dirs: list[str] | None = None) -> None:
    parameters = node_parameters(params_file)
    search = [params_file.parent, Path(__file__).resolve().parents[1] / "config"]
    pipeline = build(parameters, search)
    odometry, body, inertial = pipeline.odometry, pipeline.body, pipeline.inertial
    cache = None
    if feature_cache is not None:
        cache = CachedFrontend(odometry.frontend, feature_cache,
                               odometry.config.features)
        odometry.frontend = cache
    slop = float(parameters.get("sync_slop", 0.005) if sync_slop_sec is None else sync_slop_sec)
    # Folder cameras are directories named like the cameras unless given;
    # they are matched to camera_names in order.
    streams = list(image_dirs or [str(name) for name in parameters["camera_names"]])
    if input_format == "auto":
        if is_bag(source):
            input_format = "bag"
        elif is_dataset_folder(source, streams):
            input_format = "folder"
        else:
            raise SystemExit(f"{source} is neither a rosbag2 nor a folder with camera "
                             f"directories {streams} (see --format, --image-dirs)")
    if input_format == "bag":
        reader = BagReader(source)
        streams = [str(topic) for topic in parameters["image_topics"]]
    else:
        reader = FolderReader(source)
        if inertial is not None:
            inertial = None
            if not quiet:
                print("folder input: imu.enabled is ignored (visual only)", file=sys.stderr)
    if len(streams) != len(pipeline.rig.camera_names):
        raise SystemExit(f"{len(streams)} image inputs for cameras {pipeline.rig.camera_names}")

    imu_samples = list(reader.imu(pipeline.imu.topic)) if inertial is not None else []
    imu_index = 0

    output.mkdir(parents=True, exist_ok=True)
    trajectory = (output / "trajectory.tum").open("w", encoding="utf-8")
    motions = (output / "relative_motion.jsonl").open("w", encoding="utf-8")
    reports = (output / "frames.jsonl").open("w", encoding="utf-8") if report else None
    if not quiet:
        print(f"features: {odometry.frontend.description}; cameras: {pipeline.rig.camera_names};"
              f" pairs: {[(p.host, p.partner) for p in pipeline.rig.stereo_pairs]};"
              f" temporal: {odometry.temporal_cameras}", file=sys.stderr)
    odometry.warmup()

    previous_stamp = None
    counts = {status: 0 for status in Status}
    started = time.monotonic()
    for frame in reader.frames(streams, slop, start_sec, max_frames):
        prior = None
        imu_report = ImuFrameReport()
        if inertial is not None:
            offset_ns = int(round(pipeline.imu.time_offset_sec * 1e9))
            while (imu_index < len(imu_samples)
                   and imu_samples[imu_index].stamp_ns + offset_ns <= frame.stamp_ns):
                sample = imu_samples[imu_index]
                inertial.add_sample(sample.stamp_ns, sample.acceleration,
                                    sample.angular_velocity)
                imu_index += 1
            prior = inertial.prior(frame.stamp_ns, previous_stamp, imu_report)

        if cache is not None:
            cache.stamp_ns = frame.stamp_ns
        result = odometry.process(frame.stamp_ns, frame.images, prior)
        counts[result.status] += 1
        base_pose = body.pose(result.world_from_rig)
        if inertial is not None:
            relative = covariance = None
            if result.tracked:
                motion, motion_covariance = result.motion, result.motion_covariance
                # Update with the visual-only estimate: the solver result
                # already contains the inertial priors.
                if result.visual_motion is not None:
                    motion = result.visual_motion
                    motion_covariance = result.visual_motion_covariance
                relative = body.motion(motion)
                covariance = body.motion_covariance_ros(motion_covariance)
            inertial.update(frame.stamp_ns, base_pose if result.status is not Status.FAILED
                            else None, relative, covariance, imu_report)
            filtered = inertial.output_pose(base_pose)
            if filtered is not None:
                base_pose = filtered[0]
            result.report["imu"] = imu_report.__dict__

        if result.status is Status.BOOTSTRAP and previous_stamp is None:
            trajectory.write(_tum_line(frame.stamp_ns, base_pose))
        if result.tracked:
            trajectory.write(_tum_line(frame.stamp_ns, base_pose))
            translation, quaternion = pose_from_transform(body.motion(result.motion))
            motions.write(json.dumps({
                "from_stamp_ns": result.previous_stamp_ns,
                "to_stamp_ns": result.stamp_ns,
                "translation": translation.tolist(),
                "quaternion_xyzw": quaternion.tolist(),
                "covariance_ros": body.motion_covariance_ros(
                    result.motion_covariance).reshape(-1).tolist(),
                "covariance_source": result.covariance_source,
                "inlier_count": result.inlier_count,
                "reprojection_error_px": result.reprojection_error_px,
                "condition_number": result.condition_number,
            }) + "\n")
        if reports is not None:
            reports.write(json.dumps({"stamp_ns": frame.stamp_ns, **result.report},
                                     default=json_default) + "\n")
        previous_stamp = frame.stamp_ns
        processed = sum(counts.values())
        if not quiet and processed % 50 == 0:
            rate = processed / max(time.monotonic() - started, 1e-9)
            print(f"{processed} frames ({rate:.1f} Hz), failed {counts[Status.FAILED]}",
                  file=sys.stderr)
    trajectory.close()
    motions.close()
    if reports is not None:
        reports.close()
    if not quiet:
        print(f"done: {counts[Status.TRACKED]} tracked, {counts[Status.FAILED]} failed, "
              f"{time.monotonic() - started:.1f}s -> {output}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("params", type=Path, help="node parameter YAML (ros__parameters)")
    parser.add_argument("source", type=Path,
                        help="rosbag2 (directory, .db3 or .mcap) or dataset folder")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--sync-slop", type=float, help="override sync_slop (s)")
    parser.add_argument("--report", action="store_true", help="write per-frame frames.jsonl")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--feature-cache", type=Path,
                        help="directory to store/replay features and matches (one per "
                             "feature configuration); makes repeated studies fast")
    parser.add_argument("--format", choices=("auto", "bag", "folder"), default="auto",
                        help="input type (default: detect)")
    parser.add_argument("--image-dirs", nargs="+", metavar="DIR",
                        help="folder input: camera directories in camera_names order "
                             "(default: the camera names)")
    args = parser.parse_args(argv)
    run(args.params, args.source, args.output, args.max_frames, args.start_sec, args.sync_slop,
        args.report, args.quiet, args.feature_cache, args.format, args.image_dirs)


if __name__ == "__main__":
    main()
