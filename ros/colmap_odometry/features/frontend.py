"""The feature frontend used by the odometry: resize, extract, match."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from . import registry
from .base import Extractor, Features, JointModel, Matcher


@dataclass(frozen=True)
class ResizeConfig:
    """Image size handed to the extractor.

    Either an absolute ``width``/``height`` or a ``scale`` factor; ``None``
    everywhere keeps native resolution.  Each axis is scaled independently.
    """

    width: int | None = None
    height: int | None = None
    scale: float | None = None

    @classmethod
    def from_config(cls, value: Any) -> "ResizeConfig":
        if value is None:
            return cls()
        if isinstance(value, (int, float)):
            return cls(scale=float(value))
        if not isinstance(value, Mapping):
            raise ValueError("features.resize must be null, a scale or {width, height}")
        unknown = set(value) - {"width", "height", "scale"}
        if unknown:
            raise ValueError(f"features.resize has unknown keys {sorted(unknown)}")
        config = cls(
            width=None if value.get("width") is None else int(value["width"]),
            height=None if value.get("height") is None else int(value["height"]),
            scale=None if value.get("scale") is None else float(value["scale"]),
        )
        if config.scale is not None and (config.width or config.height):
            raise ValueError("features.resize takes either scale or width/height")
        if (config.width is None) != (config.height is None):
            raise ValueError("features.resize needs both width and height")
        return config

    def target_size(self, native: tuple[int, int]) -> tuple[int, int]:
        if self.width is not None and self.height is not None:
            return self.width, self.height
        if self.scale is not None:
            return (max(1, int(round(native[0] * self.scale))),
                    max(1, int(round(native[1] * self.scale))))
        return native


class FeatureFrontend:
    """Extract features from rig images and match image pairs.

    Keypoints in returned :class:`Features` are always in native camera
    pixels, whatever resolution the backend processed.
    """

    def __init__(self, extractor: Extractor, matcher: Matcher,
                 resize: ResizeConfig = ResizeConfig(), description: str = ""):
        if not matcher.accepts(extractor.descriptor):
            raise ValueError(
                f"matcher {type(matcher).__name__} cannot match "
                f"'{extractor.descriptor}' descriptors")
        matcher.bind(extractor.descriptor)
        self.extractor = extractor
        self.matcher = matcher
        self.resize = resize
        self.description = description or (
            f"{type(extractor).__name__}+{type(matcher).__name__}")

    def warmup(self, native_sizes: Sequence[tuple[int, int]]) -> None:
        for size in sorted(set(self.resize.target_size(s) for s in native_sizes)):
            self.extractor.warmup(size)
            if self.matcher is not self.extractor:
                self.matcher.warmup(size)

    def extract(self, images: Sequence[np.ndarray]) -> list[Features]:
        resized = []
        scales = []
        for image in images:
            native = (int(image.shape[1]), int(image.shape[0]))
            target = self.resize.target_size(native)
            if target != native:
                image = cv2.resize(image, target)
            resized.append(image)
            scales.append(np.array([native[0] / target[0], native[1] / target[1]]))
        features = self.extractor.extract(resized)
        if len(features) != len(images):
            raise RuntimeError("feature extractor returned the wrong number of results")
        output = []
        for item, image, scale in zip(features, images, scales):
            keypoints = np.asarray(item.keypoints, dtype=np.float64).reshape(-1, 2)
            if not np.all(scale == 1.0):
                keypoints = keypoints * scale
            output.append(replace(
                item,
                keypoints=keypoints,
                image_size=(int(image.shape[1]), int(image.shape[0])),
            ))
        return output

    def match(self, pairs: Sequence[tuple[Features, Features]]) -> list[np.ndarray]:
        if not pairs:
            return []
        results = self.matcher.match(pairs)
        if len(results) != len(pairs):
            raise RuntimeError("feature matcher returned the wrong number of results")
        output = []
        for (first, second), matches in zip(pairs, results):
            matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
            valid = ((matches[:, 0] >= 0) & (matches[:, 0] < len(first))
                     & (matches[:, 1] >= 0) & (matches[:, 1] < len(second)))
            output.append(matches[valid])
        return output


def build_frontend(config: Mapping[str, Any] | None) -> FeatureFrontend:
    """Build the frontend from the ``features`` configuration section.

    ``joint: {name: ...}`` selects a joint model; otherwise
    ``extractor: {name: ...}`` and an optional ``matcher: {name: ...}`` (the
    extractor's default matcher if omitted).  With neither, the best
    available extractor is chosen automatically (ALIKED + LightGlue when
    installed, otherwise OpenCV SIFT + nearest neighbour).  All other keys of
    each entry are passed to the backend's constructor.
    """
    config = dict(config or {})
    resize = ResizeConfig.from_config(config.pop("resize", None))
    joint = config.pop("joint", None)
    extractor_config = config.pop("extractor", None)
    matcher_config = config.pop("matcher", None)
    if config:
        raise ValueError(f"features has unknown keys {sorted(config)}")

    if joint is not None:
        if extractor_config is not None or matcher_config is not None:
            raise ValueError("features: use either joint or extractor/matcher, not both")
        options = _options(joint, "features.joint")
        name = options.pop("name")
        model = registry.instantiate(registry.get("joint", name), options)
        if not isinstance(model, JointModel):
            raise TypeError(f"joint backend '{name}' is not a JointModel")
        return FeatureFrontend(model, model, resize, description=name)

    if extractor_config is None:
        backend = registry.best_auto("extractor")
        if backend is None:
            raise RuntimeError("no feature extractor is available")
        extractor_config = {"name": backend.name}
    extractor_options = _options(extractor_config, "features.extractor")
    extractor_name = extractor_options.pop("name")
    extractor_backend = registry.get("extractor", extractor_name)
    extractor = registry.instantiate(extractor_backend, extractor_options)

    if matcher_config is None:
        if extractor_backend.default_matcher is None:
            raise ValueError(f"extractor '{extractor_name}' needs an explicit matcher")
        matcher_config = {"name": extractor_backend.default_matcher}
    matcher_options = _options(matcher_config, "features.matcher")
    matcher_name = matcher_options.pop("name")
    matcher = registry.instantiate(registry.get("matcher", matcher_name), matcher_options)
    return FeatureFrontend(extractor, matcher, resize,
                           description=f"{extractor_name}+{matcher_name}")


def _options(value: Any, section: str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"name": value}
    if not isinstance(value, Mapping) or "name" not in value:
        raise ValueError(f"{section} must be a name or a mapping with a 'name' key")
    return dict(value)
