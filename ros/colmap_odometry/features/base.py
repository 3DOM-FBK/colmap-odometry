"""Interfaces implemented by feature backends.

There are exactly two ways to provide features:

* an :class:`Extractor` paired with a :class:`Matcher` (e.g. ALIKED +
  LightGlue, SIFT + nearest neighbour), or
* a :class:`JointModel` that detects, describes and matches with one network
  (e.g. LoMa).

Backends receive images already resized by the frontend and return
keypoints in the coordinates of the image they were given; the frontend maps
them back to native camera pixels.  Descriptors stay in the backend's native
container (e.g. a CUDA tensor) and are only ever handed back to the same
backend's matcher.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class Features:
    """Features of one image.

    ``keypoints`` are ``(N, 2)`` float64 pixel coordinates.  ``image_size``
    is ``(width, height)`` of the image the keypoints refer to.  ``extra``
    holds backend-specific per-keypoint data (scales, orientations, ...)
    that its matcher may need.
    """

    keypoints: np.ndarray
    descriptors: Any
    image_size: tuple[int, int]
    scores: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.keypoints))


class Extractor(ABC):
    """Detects and describes keypoints."""

    #: Descriptor family produced, used to check matcher compatibility.
    descriptor: str = ""

    @abstractmethod
    def extract(self, images: Sequence[np.ndarray]) -> list[Features]:
        """Extract features from a batch of ``uint8`` images (gray or BGR)."""

    def warmup(self, image_size: tuple[int, int]) -> None:  # noqa: B027
        """Optional: run once on dummy input before real frames arrive."""


class Matcher(ABC):
    """Matches descriptors of two images."""

    @abstractmethod
    def accepts(self, descriptor: str) -> bool:
        """Return whether descriptors of this family can be matched."""

    def bind(self, descriptor: str) -> None:  # noqa: B027
        """Called once with the paired extractor's descriptor family.

        Matchers whose weights depend on the descriptor (e.g. LightGlue)
        load them here.
        """

    @abstractmethod
    def match(self, pairs: Sequence[tuple[Features, Features]]) -> list[np.ndarray]:
        """Return one ``(K, 2)`` int64 array of keypoint indices per pair."""

    def warmup(self, image_size: tuple[int, int]) -> None:  # noqa: B027
        """Optional: run once on dummy input before real frames arrive."""


class JointModel(Extractor, Matcher):
    """A single model providing extraction and its own matching."""

    def accepts(self, descriptor: str) -> bool:
        return descriptor == self.descriptor
