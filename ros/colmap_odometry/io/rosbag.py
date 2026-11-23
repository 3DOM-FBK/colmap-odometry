"""Read synchronized camera frames and IMU samples from a rosbag2 directly.

Uses the pure-Python ``rosbags`` package (optional dependency), so offline
runs need no ROS installation or bag playback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .common import ImuMessage, SyncedFrame, synchronize


def _stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def decode_image(message, msgtype: str) -> np.ndarray:
    """Decode sensor_msgs Image/CompressedImage to uint8 gray or BGR."""
    if msgtype.endswith("CompressedImage"):
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("could not decode compressed image")
        return image
    encoding = message.encoding.lower()
    height, width = int(message.height), int(message.width)
    data = np.frombuffer(message.data, dtype=np.uint8)
    if encoding in ("mono8", "8uc1"):
        return data.reshape(height, int(message.step))[:, :width].copy()
    if encoding in ("mono16", "16uc1"):
        array = data.view("<u2" if not message.is_bigendian else ">u2")
        array = array.reshape(height, int(message.step) // 2)[:, :width]
        # Same rounding as cv_bridge's mono16 -> mono8 conversion, so offline
        # runs see exactly the images the ROS node sees.
        return cv2.convertScaleAbs(array, alpha=255.0 / 65535.0)
    if encoding in ("bgr8", "rgb8", "8uc3"):
        array = data.reshape(height, int(message.step))[:, :width * 3].reshape(height, width, 3)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else array.copy()
    if encoding in ("bgra8", "rgba8"):
        array = data.reshape(height, int(message.step))[:, :width * 4].reshape(height, width, 4)
        code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(array, code)
    raise ValueError(f"unsupported image encoding '{message.encoding}'")


def iter_messages(path: str | Path, topics, typestore=None):
    """Yield ``(topic, msgtype, message)`` for the requested topics only.

    ``topics`` is a list of names or a predicate on a rosbags connection.
    Only the selected messages are deserialized.  sqlite3 bags are read
    directly, because rosbags validates every stored type definition on
    open and fails on bags containing (unrelated) unparsable custom types.
    """
    from rosbags.typesys import Stores, get_typestore

    typestore = typestore or get_typestore(Stores.ROS2_JAZZY)
    path = Path(path)
    databases = sorted(path.glob("*.db3")) if path.is_dir() else (
        [path] if path.suffix == ".db3" else [])
    if databases:
        yield from _sqlite_messages(databases, topics, typestore)
        return
    from rosbags.rosbag2 import Reader

    wanted = topics if callable(topics) else (lambda connection: connection.topic in topics)
    with Reader(path) as reader:
        connections = [connection for connection in reader.connections if wanted(connection)]
        if not connections:
            raise ValueError(f"{path} contains none of the requested topics")
        for connection, _, raw in reader.messages(connections=connections):
            yield (connection.topic, connection.msgtype,
                   typestore.deserialize_cdr(raw, connection.msgtype))


def _sqlite_messages(databases, topics, typestore):
    import sqlite3
    from types import SimpleNamespace

    for database in databases:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT id, name, type FROM topics").fetchall()
            selected = {}
            for identifier, name, msgtype in rows:
                candidate = SimpleNamespace(topic=name, msgtype=msgtype)
                if (topics(candidate) if callable(topics) else name in topics):
                    selected[identifier] = (name, msgtype)
            if not selected:
                raise ValueError(f"{database} contains none of the requested topics")
            marks = ",".join("?" * len(selected))
            cursor = connection.execute(
                f"SELECT topic_id, data FROM messages WHERE topic_id IN ({marks}) "
                "ORDER BY timestamp", tuple(selected))
            for topic_id, data in cursor:
                name, msgtype = selected[topic_id]
                yield name, msgtype, typestore.deserialize_cdr(data, msgtype)
        finally:
            connection.close()


class BagReader:
    """Iterate time-synchronized image sets (and IMU samples) from a bag."""

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def frames(
        self,
        image_topics: Sequence[str],
        slop_sec: float = 0.005,
        start_sec: float = 0.0,
        max_frames: int | None = None,
    ) -> Iterator[SyncedFrame]:
        """Yield frames whose images were stamped within ``slop_sec``."""
        messages = ((topic, _stamp_ns(message), (message, msgtype))
                    for topic, msgtype, message in iter_messages(self._path, list(image_topics)))
        yield from synchronize(messages, list(image_topics),
                               lambda item: decode_image(*item), slop_sec, start_sec, max_frames)

    def imu(self, topic: str) -> Iterator[ImuMessage]:
        for _, _, message in iter_messages(self._path, [topic]):
            yield ImuMessage(
                stamp_ns=_stamp_ns(message),
                acceleration=np.array([message.linear_acceleration.x,
                                       message.linear_acceleration.y,
                                       message.linear_acceleration.z]),
                angular_velocity=np.array([message.angular_velocity.x,
                                           message.angular_velocity.y,
                                           message.angular_velocity.z]),
            )
