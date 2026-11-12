"""ROS 2 node: synchronized images (+ IMU) in, odometry out.

Image callbacks only enqueue synchronized frames; one worker thread drains
the queue in order, so frames are never dropped when processing is slower
than the camera.  IMU samples arrive on their own callback group.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import message_filters
import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from colmap_odometry_msgs.msg import FeatureTrackArray, RelativePoseWithCovarianceStamped
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMessage
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .. import setup_pipeline
from ..diagnostics import json_default
from ..imu.pipeline import ImuFrameReport
from ..geometry.lie import pose_from_transform
from ..odometry import FrameResult, Status
from . import messages, visualization
from .settings import NodeSettings

_IGNORED_PARAMETERS = ("use_sim_time", "qos_overrides.", "start_type_description_service")


class ColmapOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("colmap_odometry_node",
                         automatically_declare_parameters_from_overrides=True,
                         allow_undeclared_parameters=True)
        parameters = self._nested_parameters()
        imu_parameters = parameters.pop("imu", {})
        self.settings = settings = NodeSettings.from_mapping(parameters)
        search = [Path(_share_directory()) / "config"] if _share_directory() else []
        self.pipeline = setup_pipeline.build(
            {"config_file": settings.config_file, "camera_file": settings.camera_file,
             "rig_file": settings.rig_file, "camera_names": list(settings.camera_names),
             "T_camera_base": list(settings.T_camera_base), "imu": imu_parameters},
            search + [Path.cwd()])
        self.odometry = self.pipeline.odometry
        self.body = self.pipeline.body
        self.inertial = self.pipeline.inertial
        self.work_dir = Path(settings.work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()

        self._create_publishers()
        self._create_outputs()
        self._subscribe()
        if settings.publish_camera_tf:
            self._publish_camera_transforms()

        self._queue: deque = deque()
        self._condition = threading.Condition()
        self._stopping = False
        self._previous_stamp_ns: int | None = None
        self._origin_published = False
        self._frames = 0
        self._frame_started: float | None = None
        self._worker = threading.Thread(target=self._work, name="colmap_odometry_worker",
                                        daemon=True)
        self._worker.start()
        self.create_timer(1.0, self._watchdog)
        rig = self.pipeline.rig
        self.get_logger().info(
            f"features {self.odometry.frontend.description}; cameras {rig.camera_names}; "
            f"stereo pairs {[(p.host, p.partner) for p in rig.stereo_pairs]}; temporal "
            f"{self.odometry.temporal_cameras}; IMU {'on' if self.inertial else 'off'}")

    # ------------------------------------------------------------------ setup
    def _nested_parameters(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, parameter in self.get_parameters_by_prefix("").items():
            if any(name == item or name.startswith(item) for item in _IGNORED_PARAMETERS):
                continue
            node = result
            *parents, leaf = name.split(".")
            for parent in parents:
                node = node.setdefault(parent, {})
            node[leaf] = parameter.value
        return result

    def _create_publishers(self) -> None:
        settings = self.settings
        self.odom_pub = self.create_publisher(Odometry, "~/odom", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "~/pose", 10)
        self.path_pub = self.create_publisher(PathMessage, "~/path", 1)
        self.motion_pub = self.create_publisher(
            RelativePoseWithCovarianceStamped, "~/relative_motion", 50)
        self.status_pub = self.create_publisher(String, "~/status", 10)
        self.tracks_pub = (self.create_publisher(FeatureTrackArray, "~/feature_tracks", 50)
                           if settings.publish_feature_tracks else None)
        self.matches_pub = (self.create_publisher(Image, "~/feature_matches", 1)
                            if settings.publish_feature_matches else None)
        self.cloud_pub = (self.create_publisher(PointCloud2, "~/sparse_points", 1)
                          if settings.publish_sparse_point_cloud else None)
        self.republishers = [
            self.create_publisher(Image, f"~/input/{name}/image_raw", 1)
            if topic.endswith("/compressed") else None
            for name, topic in zip(settings.camera_names, settings.image_topics)]
        self.tf_broadcaster = TransformBroadcaster(self) if settings.publish_tf else None
        self.path = PathMessage()
        self.path.header.frame_id = settings.world_frame_id
        self._last_path_publish = 0.0

    def _create_outputs(self) -> None:
        settings = self.settings
        self.tum_file = None
        if settings.save_trajectory_tum:
            path = Path(settings.trajectory_tum_path or "trajectory.tum")
            path = path if path.is_absolute() else self.work_dir / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.tum_file = path.open("w" if settings.trajectory_tum_overwrite else "a",
                                      encoding="utf-8", buffering=1)
            self.get_logger().info(f"writing TUM trajectory to {path}")
        self.frame_log = None
        if settings.frame_log_path:
            path = Path(settings.frame_log_path)
            path = path if path.is_absolute() else self.work_dir / path
            self.frame_log = path.open("w", encoding="utf-8", buffering=1)

    def _subscribe(self) -> None:
        settings = self.settings
        reliability = (ReliabilityPolicy.RELIABLE if settings.image_qos_reliability == "reliable"
                       else ReliabilityPolicy.BEST_EFFORT)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=max(1, settings.sync_queue_size),
                         reliability=reliability)
        subscribers = [
            message_filters.Subscriber(
                self, CompressedImage if topic.endswith("/compressed") else Image, topic,
                qos_profile=qos)
            for topic in settings.image_topics]
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=max(1, settings.sync_queue_size), slop=settings.sync_slop)
        self.synchronizer.registerCallback(self._on_images)
        self._subscribers = subscribers
        if self.inertial is not None:
            self.create_subscription(
                Imu, self.pipeline.imu.topic, self._on_imu,
                QoSProfile(depth=max(10, self.pipeline.imu.queue_depth),
                           reliability=qos_profile_sensor_data.reliability),
                callback_group=MutuallyExclusiveCallbackGroup())

    def _publish_camera_transforms(self) -> None:
        stamp = self.get_clock().now().to_msg()
        rig = self.pipeline.rig
        transforms = [
            messages.transform_stamped(stamp, self.settings.body_frame_id, frame,
                                       self.body.base_from_rig @ rig.rig_from_camera(name))
            for name, frame in zip(self.settings.camera_names, self.settings.camera_frames)]
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.static_broadcaster.sendTransform(transforms)

    # -------------------------------------------------------------- callbacks
    def _on_imu(self, message: Imu) -> None:
        self.inertial.add_sample(
            messages.stamp_to_ns(message.header.stamp),
            np.array([message.linear_acceleration.x, message.linear_acceleration.y,
                      message.linear_acceleration.z]),
            np.array([message.angular_velocity.x, message.angular_velocity.y,
                      message.angular_velocity.z]))

    def _on_images(self, *image_messages) -> None:
        with self._condition:
            if self._stopping:
                return
            self._queue.append(image_messages)
            limit = self.settings.processing_queue_size
            if limit > 0 and len(self._queue) >= limit:
                self.get_logger().warning(
                    f"{len(self._queue)} frames waiting for processing (none are dropped)",
                    throttle_duration_sec=2.0)
            self._condition.notify()

    def _work(self) -> None:
        try:
            # Warm up (and compile) on the thread that processes frames.
            started = time.monotonic()
            self.odometry.warmup()
            self.get_logger().info(f"ready ({time.monotonic() - started:.1f}s warm-up)")
        except Exception as error:  # noqa: BLE001 - warm-up is optional
            self.get_logger().error(f"feature warm-up failed: {error!r}")
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or self._queue)
                if self._stopping and not self._queue:
                    return
                image_messages = self._queue.popleft()
            try:
                self._process(image_messages)
            except Exception as error:  # noqa: BLE001 - keep the node alive
                self.get_logger().error(
                    f"frame processing failed: {error!r}\n{traceback.format_exc()}",
                    throttle_duration_sec=2.0)
            finally:
                self._frame_started = None

    # ------------------------------------------------------------- processing
    def _process(self, image_messages) -> None:
        self._frame_started = time.monotonic()
        settings = self.settings
        stamp = image_messages[0].header.stamp
        stamp_ns = messages.stamp_to_ns(stamp)
        images = [
            self.bridge.compressed_imgmsg_to_cv2(message, desired_encoding="passthrough")
            if isinstance(message, CompressedImage)
            else self.bridge.imgmsg_to_cv2(message, desired_encoding=settings.image_encoding)
            for message in image_messages]
        for message, image, publisher in zip(image_messages, images, self.republishers):
            if publisher is not None:
                output = self.bridge.cv2_to_imgmsg(
                    image, encoding="mono8" if image.ndim == 2 else "bgr8")
                output.header = message.header
                publisher.publish(output)

        previous = self._previous_stamp_ns
        self._previous_stamp_ns = stamp_ns
        imu_report = ImuFrameReport()
        prior = None
        if self.inertial is not None:
            self.inertial.wait_for(stamp_ns)
            prior = self.inertial.prior(stamp_ns, previous, imu_report)

        result = self.odometry.process(stamp_ns, images, prior)
        output_pose = self.body.pose(result.world_from_rig)
        output_covariance = None
        relative = covariance = None
        if result.tracked:
            relative = self.body.motion(result.motion)
            covariance = self.body.motion_covariance_ros(result.motion_covariance)
        if self.inertial is not None:
            update_relative, update_covariance = relative, covariance
            # Update with the visual-only estimate: the solver result already
            # contains the inertial priors.
            if result.visual_motion is not None:
                update_relative = self.body.motion(result.visual_motion)
                update_covariance = self.body.motion_covariance_ros(
                    result.visual_motion_covariance)
            self.inertial.update(stamp_ns, output_pose if result.status is not Status.FAILED
                                 else None, update_relative, update_covariance, imu_report)
            filtered = self.inertial.output_pose(output_pose)
            if filtered is not None:
                output_pose, output_covariance = filtered
            result.report["imu"] = imu_report.__dict__

        if result.tracked:
            # Publish the interval before the matching absolute sample.
            self.motion_pub.publish(messages.relative_motion(
                stamp, settings.body_frame_id, result.previous_stamp_ns, relative, covariance,
                result.inlier_count, result.reprojection_error_px, result.condition_number,
                result.covariance_source))
        if result.tracked or (result.status is Status.BOOTSTRAP and not self._origin_published):
            self._publish_absolute(stamp, output_pose, output_covariance)
            self._origin_published = True
        if result.status is Status.FAILED:
            self.get_logger().warning(f"no motion for frame {stamp_ns}: "
                                      f"{result.failure_reason}", throttle_duration_sec=1.0)
        self._publish_tracks(stamp, result, image_messages)
        self._publish_debug(stamp, result, images, output_pose)
        self._publish_status(stamp_ns, result)

    def _publish_absolute(self, stamp, pose: np.ndarray, covariance: np.ndarray | None) -> None:
        settings = self.settings
        pose_message = messages.pose_stamped(stamp, settings.world_frame_id, pose)
        self.pose_pub.publish(pose_message)
        self.odom_pub.publish(messages.odometry(stamp, settings.world_frame_id,
                                                settings.body_frame_id, pose, covariance))
        self.path.header.stamp = stamp
        self.path.poses.append(pose_message)
        now = time.monotonic()
        if now - self._last_path_publish >= settings.path_publish_period_sec:
            self.path_pub.publish(self.path)
            self._last_path_publish = now
        if self.tf_broadcaster is not None:
            self.tf_broadcaster.sendTransform(messages.transform_stamped(
                stamp, settings.world_frame_id, settings.body_frame_id, pose))
        if self.tum_file is not None:
            translation, quaternion = pose_from_transform(pose)
            stamp_ns = messages.stamp_to_ns(stamp)
            self.tum_file.write(
                f"{stamp_ns // 1_000_000_000}.{stamp_ns % 1_000_000_000:09d} "
                + " ".join(f"{value:.9f}" for value in (*translation, *quaternion)) + "\n")

    def _publish_tracks(self, stamp, result: FrameResult, image_messages) -> None:
        if self.tracks_pub is None or not result.track_updates:
            return
        stereo_images = None
        pairs = self.pipeline.rig.stereo_pairs
        names = list(self.settings.camera_names)
        if pairs:
            first = image_messages[names.index(pairs[0].host)]
            second = image_messages[names.index(pairs[0].partner)]
            if isinstance(first, CompressedImage) and isinstance(second, CompressedImage):
                stereo_images = (first, second)
        self.tracks_pub.publish(messages.feature_tracks(
            stamp, self.settings.camera_frames[0], self.odometry.frontend.description,
            result.track_updates, stereo_images))

    def _publish_debug(self, stamp, result: FrameResult, images, output_pose) -> None:
        settings = self.settings
        names = list(settings.camera_names)
        if self.matches_pub is not None and result.match_debug:
            panels = [visualization.match_panel(
                images[names.index(item.first)], images[names.index(item.second)], item,
                settings.feature_match_max_features, settings.feature_match_max_matches)
                for item in result.match_debug]
            message = self.bridge.cv2_to_imgmsg(visualization.stack_vertical(panels), "bgr8")
            message.header.stamp = stamp
            message.header.frame_id = settings.camera_frames[0]
            self.matches_pub.publish(message)
        if self.cloud_pub is not None and result.stereo_points:
            world_from_rig = output_pose @ self.body.base_from_rig
            points, colors = [], []
            for item in result.stereo_points:
                if len(item.points_rig) == 0:
                    continue
                points.append(item.points_rig @ world_from_rig[:3, :3].T + world_from_rig[:3, 3])
                colors.append(visualization.pixel_colors(
                    images[names.index(item.camera)], item.pixels,
                    settings.image_encoding.lower().startswith("rgb")))
            if points:
                self.cloud_pub.publish(messages.point_cloud(
                    stamp, settings.world_frame_id, np.concatenate(points),
                    np.concatenate(colors)))

    def _publish_status(self, stamp_ns: int, result: FrameResult) -> None:
        self._frames += 1
        report = result.report
        status = {
            "stamp_ns": stamp_ns,
            "status": result.status.value,
            "failure_reason": result.failure_reason,
            "features": report.get("features"),
            "pnp": report.get("pnp"),
            "covariance_source": result.covariance_source,
            "inliers": result.inlier_count,
            "timings": report.get("timings"),
        }
        self.status_pub.publish(String(data=json.dumps(status, default=json_default)))
        if self.frame_log is not None:
            self.frame_log.write(json.dumps({"stamp_ns": stamp_ns, **report},
                                            default=json_default) + "\n")
        settings = self.settings
        total = report.get("timings", {}).get("total", 0.0)
        if (settings.performance_log_every_n_frames > 0
                and self._frames % settings.performance_log_every_n_frames == 0) \
                or total > settings.performance_slow_frame_sec:
            timings = ", ".join(f"{key}={value * 1e3:.0f}ms"
                                for key, value in report.get("timings", {}).items())
            with self._condition:
                backlog = len(self._queue)
            self.get_logger().info(f"frame {self._frames} {result.status.value}: {timings}; "
                                   f"backlog {backlog}")

    def _watchdog(self) -> None:
        started = self._frame_started
        if started is None:
            return
        elapsed = time.monotonic() - started
        if elapsed > self.settings.performance_stall_warn_sec:
            self.get_logger().warning(
                f"frame processing stalled for {elapsed:.1f}s in stage "
                f"'{self.odometry.profiler.stage}'", throttle_duration_sec=5.0)

    # ---------------------------------------------------------------- shutdown
    def destroy_node(self) -> bool:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=30.0)
        if self.tracks_pub is not None:
            updates = self.odometry.finish_tracks()
            if updates:
                self.tracks_pub.publish(messages.feature_tracks(
                    self.get_clock().now().to_msg(), self.settings.camera_frames[0],
                    self.odometry.frontend.description, updates))
        for stream in (self.tum_file, self.frame_log):
            if stream is not None:
                stream.close()
        return super().destroy_node()


def _share_directory() -> str | None:
    try:
        return get_package_share_directory("colmap_odometry")
    except PackageNotFoundError:
        return None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ColmapOdometryNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
