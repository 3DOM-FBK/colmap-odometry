"""Calibrated multi-camera rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .camera import Camera
from .lie import invert, skew, transform_from_pose


@dataclass(frozen=True)
class StereoPair:
    """Two overlapping cameras; landmarks are hosted in ``host``."""

    host: str
    partner: str


class Rig:
    """Named cameras with fixed extrinsics in a common rig frame.

    ``T_rig_camera`` maps camera points into the rig frame (ROS TF
    direction, ``p_rig = T_rig_camera p_camera``).  The rig frame is the
    frame whose motion the odometry estimates.
    """

    def __init__(
        self,
        cameras: Sequence[Camera],
        rig_from_camera: Mapping[str, np.ndarray],
        stereo_pairs: Sequence[tuple[str, str]] = (),
    ):
        if not cameras:
            raise ValueError("a rig needs at least one camera")
        names = [camera.name for camera in cameras]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate camera names in {names}")
        self.cameras: dict[str, Camera] = {camera.name: camera for camera in cameras}
        self._rig_from_camera = {}
        self._camera_from_rig = {}
        for name in names:
            if name not in rig_from_camera:
                raise ValueError(f"missing rig extrinsics for camera '{name}'")
            transform = np.asarray(rig_from_camera[name], dtype=np.float64).reshape(4, 4)
            self._rig_from_camera[name] = transform
            self._camera_from_rig[name] = invert(transform)
        pairs = []
        for host, partner in stereo_pairs:
            for name in (host, partner):
                if name not in self.cameras:
                    raise ValueError(f"stereo pair references unknown camera '{name}'")
            if host == partner:
                raise ValueError(f"stereo pair ({host}, {partner}) uses one camera twice")
            pairs.append(StereoPair(str(host), str(partner)))
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate stereo pairs")
        self.stereo_pairs: tuple[StereoPair, ...] = tuple(pairs)

    @classmethod
    def from_config(
        cls,
        camera_names: Sequence[str],
        intrinsics: Mapping[str, Any],
        extrinsics: Mapping[str, Any],
        stereo_pairs: Sequence[Sequence[str]] = (),
    ) -> "Rig":
        """Build a rig from the intrinsics and ``T_rig_camera`` YAML mappings.

        Each extrinsics entry has ``rotation: [x, y, z, w]`` and
        ``translation: [x, y, z]`` (ROS TF direction).
        """
        cameras = []
        rig_from_camera = {}
        for name in camera_names:
            if name not in intrinsics:
                raise ValueError(f"intrinsics do not define camera '{name}'")
            cameras.append(Camera.from_config(name, intrinsics[name]))
            entry = extrinsics.get(name)
            if not isinstance(entry, Mapping):
                raise ValueError(f"rig extrinsics do not define camera '{name}'")
            rotation = np.asarray(entry.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
            translation = np.asarray(entry.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
            if rotation.shape != (4,) or translation.shape != (3,):
                raise ValueError(f"camera '{name}' extrinsics need 4 rotation and 3 translation values")
            if not (np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation))):
                raise ValueError(f"camera '{name}' extrinsics are not finite")
            rig_from_camera[name] = transform_from_pose(translation, rotation)
        pairs = [tuple(pair) for pair in stereo_pairs]
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"stereo pair {list(pair)} must name exactly two cameras")
        return cls(cameras, rig_from_camera, pairs)

    @property
    def camera_names(self) -> list[str]:
        return list(self.cameras)

    def index(self, name: str) -> int:
        return self.camera_names.index(name)

    def rig_from_camera(self, name: str) -> np.ndarray:
        return self._rig_from_camera[name].copy()

    def camera_from_rig(self, name: str) -> np.ndarray:
        return self._camera_from_rig[name].copy()

    def relative(self, target: str, source: str) -> np.ndarray:
        """Return ``T_target_source`` between two rig cameras."""
        return self._camera_from_rig[target] @ self._rig_from_camera[source]

    def essential(self, first: str, second: str) -> np.ndarray:
        """Essential matrix with ``x_second^T E x_first = 0``."""
        transform = self.relative(second, first)
        return skew(transform[:3, 3]) @ transform[:3, :3]

    def partners(self, host: str) -> list[str]:
        return [pair.partner for pair in self.stereo_pairs if pair.host == host]

    def default_temporal_cameras(self) -> list[str]:
        """Pair hosts plus cameras that belong to no pair (monocular)."""
        in_pairs = {name for pair in self.stereo_pairs for name in (pair.host, pair.partner)}
        hosts = {pair.host for pair in self.stereo_pairs}
        return [name for name in self.camera_names if name in hosts or name not in in_pairs]
