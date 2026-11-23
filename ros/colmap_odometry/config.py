"""Odometry configuration (the YAML file referenced by ``config_file``).

Example::

    stereo_pairs: [[cam0, cam1]]   # (host, partner); landmarks live in host
    temporal_cameras: null         # default: pair hosts + unpaired cameras
    features:                      # see colmap_odometry.features.build_frontend
      resize: {width: 1024, height: 1024}
      joint: {name: loma, model: loma_B128, num_keypoints: 1000}
    stereo: {epipolar_max_error_px: 2.0, min_depth: 0.2, max_depth: 20.0}
    temporal: {max_error_px: 2.0}
    pnp: {ransac_max_error_px: 3.0}
    refinement: {pixel_std_px: 1.0, robust_delta: 3.0, max_iterations: 4}
    feature_tracks: {enabled: true, min_length: 5}

Every section is optional; unknown keys are errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .estimation.anchored import RefinementOptions
from .estimation.pnp import PnPOptions
from .vision.stereo import StereoGates
from .vision.temporal import EssentialOptions


@dataclass(frozen=True)
class TrackOptions:
    enabled: bool = True
    min_length: int = 5
    #: Cameras whose temporal matches form tracks (default: all temporal).
    cameras: tuple[str, ...] | None = None


@dataclass(frozen=True)
class OdometryConfig:
    stereo_pairs: tuple[tuple[str, str], ...] = ()
    temporal_cameras: tuple[str, ...] | None = None
    features: Mapping[str, Any] = field(default_factory=dict)
    stereo: StereoGates = StereoGates()
    temporal: EssentialOptions = EssentialOptions()
    pnp: PnPOptions = PnPOptions()
    refinement: RefinementOptions = RefinementOptions()
    feature_tracks: TrackOptions = TrackOptions()
    #: Reject the essential-matrix rotation as a refinement initializer when
    #: it differs from the PnP rotation by more than this angle.
    initial_rotation_gate_deg: float = 15.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "OdometryConfig":
        values = dict(values or {})
        sections = {
            "stereo": StereoGates,
            "temporal": EssentialOptions,
            "pnp": PnPOptions,
            "refinement": RefinementOptions,
            "feature_tracks": TrackOptions,
        }
        _reject_unknown(values, {item.name for item in fields(cls)}, "odometry config")
        kwargs: dict[str, Any] = {}
        for name, section in sections.items():
            if name in values:
                kwargs[name] = _section(section, values[name], name)
        if "stereo_pairs" in values:
            kwargs["stereo_pairs"] = tuple(tuple(str(name) for name in pair)
                                           for pair in values["stereo_pairs"] or ())
        if values.get("temporal_cameras") is not None:
            kwargs["temporal_cameras"] = tuple(str(name) for name in values["temporal_cameras"])
        if "features" in values:
            kwargs["features"] = dict(values["features"] or {})
        if "initial_rotation_gate_deg" in values:
            kwargs["initial_rotation_gate_deg"] = float(values["initial_rotation_gate_deg"])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "OdometryConfig":
        return cls.from_mapping(load_yaml(path))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML file does not contain a mapping: {path}")
    return loaded


def _reject_unknown(values: Mapping[str, Any], known: set[str], section: str) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{section} has unknown keys {unknown}; valid: {sorted(known)}")


def _section(cls, values: Any, name: str):
    if values is None:
        return cls()
    if not isinstance(values, Mapping):
        raise ValueError(f"odometry config section '{name}' must be a mapping")
    known = {item.name: item for item in fields(cls)}
    _reject_unknown(values, set(known), name)
    converted = {}
    for key, value in values.items():
        default = known[key].default
        if value is None or default is None:
            converted[key] = tuple(value) if isinstance(value, list) else value
        elif isinstance(default, bool):
            converted[key] = bool(value)
        elif isinstance(default, int):
            converted[key] = int(value)
        elif isinstance(default, float):
            converted[key] = float(value)
        else:
            converted[key] = value
    return cls(**converted)
