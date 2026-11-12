"""Build the odometry pipeline from the node's parameter set.

Shared by the ROS node and the offline runner so both interpret a
``ros__parameters`` file identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import OdometryConfig, load_yaml
from .features import build_frontend
from .frames import OPTICAL_FROM_REP103, BodyFrame, rigid_from_row_major
from .geometry import Rig
from .imu.config import ImuConfig
from .imu.pipeline import InertialPipeline
from .odometry import Odometry


@dataclass
class Pipeline:
    rig: Rig
    odometry: Odometry
    body: BodyFrame
    inertial: InertialPipeline | None
    imu: ImuConfig


def resolve(path: str, search: Sequence[Path]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    for directory in search:
        if (directory / candidate).exists():
            return (directory / candidate).resolve()
    raise FileNotFoundError(f"{path} not found in {[str(d) for d in search]}")


def build(parameters: Mapping[str, Any], search: Sequence[Path]) -> Pipeline:
    """Build from parameters with keys ``config_file``, ``camera_file``,
    ``rig_file``, ``camera_names``, ``T_camera_base`` and ``imu``."""
    camera_names = [str(name) for name in parameters["camera_names"]]
    config = OdometryConfig.load(resolve(str(parameters["config_file"]), search))
    intrinsics = load_yaml(resolve(str(parameters["camera_file"]), search))
    extrinsics = load_yaml(resolve(str(parameters["rig_file"]), search))
    rig = Rig.from_config(camera_names, intrinsics, extrinsics, config.stereo_pairs)
    camera_from_base = rigid_from_row_major(
        parameters.get("T_camera_base", OPTICAL_FROM_REP103.reshape(-1)), "T_camera_base")
    body = BodyFrame.from_camera(rig.rig_from_camera(camera_names[0]), camera_from_base)
    imu = ImuConfig.from_mapping(parameters.get("imu"))
    inertial = InertialPipeline(imu, body.rig_from_base) if imu.enabled else None
    odometry = Odometry(rig, build_frontend(config.features), config)
    return Pipeline(rig, odometry, body, inertial, imu)


def node_parameters(path: str | Path) -> dict[str, Any]:
    """Return the ``ros__parameters`` mapping of a node parameter file."""
    loaded = load_yaml(path)
    for value in loaded.values():
        if isinstance(value, Mapping) and "ros__parameters" in value:
            return dict(value["ros__parameters"])
    if "ros__parameters" in loaded:
        return dict(loaded["ros__parameters"])
    raise ValueError(f"{path} has no ros__parameters section")
