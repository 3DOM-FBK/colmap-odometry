"""Replay stored features and matches (offline studies).

Feature extraction and matching dominate the runtime and are unaffected by
geometry or IMU settings.  Caching them per frame makes repeated offline
runs fast and their comparisons exact.  A cache directory belongs to one
feature configuration, which is recorded and checked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base import Features
from .frontend import FeatureFrontend


class CachedFrontend:
    """Wraps a :class:`FeatureFrontend`; set ``stamp_ns`` before each frame."""

    def __init__(self, frontend: FeatureFrontend, directory: str | Path,
                 features_config: Mapping[str, Any]):
        self.frontend = frontend
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.description = f"{frontend.description} (cached)"
        self.stamp_ns: int | None = None
        meta = {"description": frontend.description,
                "features": json.loads(json.dumps(dict(features_config), default=str))}
        meta_path = self.directory / "meta.json"
        if meta_path.exists():
            stored = json.loads(meta_path.read_text())
            if stored != meta:
                raise ValueError(f"feature cache {self.directory} was made with {stored}, "
                                 f"not {meta}")
        else:
            meta_path.write_text(json.dumps(meta, indent=2))
        self._sizes: list[tuple[int, int]] = []
        self._warm = False
        self._features: list[Features] | None = None
        self._matches: list[np.ndarray] | None = None

    def _path(self) -> Path:
        if self.stamp_ns is None:
            raise RuntimeError("CachedFrontend.stamp_ns must be set for every frame")
        return self.directory / f"{self.stamp_ns}.npz"

    def warmup(self, native_sizes: Sequence[tuple[int, int]]) -> None:
        # Deferred until a frame is actually missing from the cache.
        self._sizes = list(native_sizes)

    def extract(self, images: Sequence[np.ndarray]) -> list[Features]:
        path = self._path()
        if path.exists():
            data = np.load(path)
            self._matches = [data[f"matches_{index}"] for index in range(int(data["pairs"]))]
            return [Features(data[f"keypoints_{index}"], None,
                             tuple(int(v) for v in data[f"size_{index}"]))
                    for index in range(int(data["cameras"]))]
        if not self._warm:
            self.frontend.warmup(self._sizes)
            self._warm = True
        self._matches = None
        self._features = self.frontend.extract(images)
        return self._features

    def match(self, pairs: Sequence[tuple[Features, Features]]) -> list[np.ndarray]:
        if self._matches is not None:
            if len(self._matches) != len(pairs):
                raise RuntimeError(f"cached frame {self.stamp_ns} has {len(self._matches)} "
                                   f"matched pairs, {len(pairs)} requested")
            return self._matches
        matches = self.frontend.match(pairs)
        arrays = {"cameras": len(self._features), "pairs": len(matches)}
        for index, item in enumerate(self._features):
            arrays[f"keypoints_{index}"] = item.keypoints
            arrays[f"size_{index}"] = np.asarray(item.image_size)
        for index, item in enumerate(matches):
            arrays[f"matches_{index}"] = np.asarray(item, dtype=np.int64)
        np.savez(self._path(), **arrays)
        return matches
