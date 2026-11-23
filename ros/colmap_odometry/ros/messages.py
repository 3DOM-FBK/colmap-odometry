"""Conversions between numpy results and ROS messages."""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time
from colmap_odometry_msgs.msg import (
    FeatureObservation,
    FeatureTrack,
    FeatureTrackArray,
    RelativePoseWithCovarianceStamped,
)
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from ..geometry.lie import pose_from_transform
from ..vision.tracks import TrackUpdate


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_stamp(stamp_ns: int) -> Time:
    return Time(sec=int(stamp_ns // 1_000_000_000), nanosec=int(stamp_ns % 1_000_000_000))


def pose_message(transform: np.ndarray) -> Pose:
    translation, quaternion = pose_from_transform(transform)
    message = Pose()
    message.position.x, message.position.y, message.position.z = (float(v) for v in translation)
    (message.orientation.x, message.orientation.y, message.orientation.z,
     message.orientation.w) = (float(v) for v in quaternion)
    return message


def pose_stamped(stamp, frame_id: str, transform: np.ndarray) -> PoseStamped:
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.pose = pose_message(transform)
    return message


def odometry(stamp, frame_id: str, child_frame_id: str, transform: np.ndarray,
             covariance_ros: np.ndarray | None) -> Odometry:
    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.child_frame_id = child_frame_id
    message.pose.pose = pose_message(transform)
    if covariance_ros is not None:
        message.pose.covariance = np.asarray(covariance_ros, dtype=np.float64).reshape(-1).tolist()
    message.twist.covariance[0] = -1.0  # twist is not estimated
    return message


def transform_stamped(stamp, parent: str, child: str, transform: np.ndarray) -> TransformStamped:
    translation, quaternion = pose_from_transform(transform)
    message = TransformStamped()
    message.header.stamp = stamp
    message.header.frame_id = parent
    message.child_frame_id = child
    (message.transform.translation.x, message.transform.translation.y,
     message.transform.translation.z) = (float(v) for v in translation)
    (message.transform.rotation.x, message.transform.rotation.y,
     message.transform.rotation.z, message.transform.rotation.w) = (float(v) for v in quaternion)
    return message


def relative_motion(stamp, frame_id: str, from_stamp_ns: int, transform: np.ndarray,
                    covariance_ros: np.ndarray, inlier_count: int, reprojection_px: float,
                    condition_number: float, covariance_source: str
                    ) -> RelativePoseWithCovarianceStamped:
    message = RelativePoseWithCovarianceStamped()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.from_stamp = ns_to_stamp(from_stamp_ns)
    message.to_stamp = stamp
    message.relative_pose.pose = pose_message(transform)
    message.relative_pose.covariance = np.asarray(
        covariance_ros, dtype=np.float64).reshape(-1).tolist()
    message.inlier_count = max(0, int(inlier_count))
    message.reprojection_rms_px = float(reprojection_px) if np.isfinite(reprojection_px) else 0.0
    message.information_condition_number = (
        float(condition_number) if np.isfinite(condition_number) else 0.0)
    message.covariance_source = covariance_source
    return message


def feature_tracks(stamp, frame_id: str, feature_type: str, updates: list[TrackUpdate],
                   stereo_images=None) -> FeatureTrackArray:
    message = FeatureTrackArray()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.feature_type = feature_type
    if stereo_images is not None:
        message.has_stereo_images = True
        message.left_image, message.right_image = stereo_images
    for update in updates:
        track = FeatureTrack()
        track.track_id = int(update.track_id)
        track.camera_name = update.camera_name
        track.is_complete = bool(update.is_complete)
        for item in update.observations:
            observation = FeatureObservation()
            observation.stamp = ns_to_stamp(item.stamp_ns)
            observation.frame_index = int(item.frame_index)
            observation.keypoint_index = int(item.keypoint_index)
            observation.x, observation.y = float(item.x), float(item.y)
            observation.has_stereo_point = bool(item.has_stereo_point)
            observation.stereo_x, observation.stereo_y = float(item.stereo_x), float(item.stereo_y)
            observation.stereo_point_camera = [float(v) for v in item.point_camera]
            observation.triangulation_angle_rad = float(item.triangulation_angle_rad)
            track.observations.append(observation)
        # A completion marker may legitimately carry no new observation.
        if track.observations or track.is_complete:
            message.tracks.append(track)
    return message


def point_cloud(stamp, frame_id: str, points: np.ndarray, colors_rgb: np.ndarray) -> PointCloud2:
    header = Header(stamp=stamp, frame_id=frame_id)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    packed = ((colors_rgb[:, 0].astype(np.uint32) << 16)
              | (colors_rgb[:, 1].astype(np.uint32) << 8) | colors_rgb[:, 2].astype(np.uint32))
    rows = [(float(x), float(y), float(z), int(c)) for (x, y, z), c in zip(points, packed)]
    return point_cloud2.create_cloud(header, fields, rows)
