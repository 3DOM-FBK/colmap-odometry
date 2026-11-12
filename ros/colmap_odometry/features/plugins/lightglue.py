"""Extractors (ALIKED, SuperPoint, DISK) and the LightGlue matcher.

Vendored as the ``thirdparty/LightGlue`` submodule (cvg/LightGlue); needs
torch, torchvision and kornia.  Weights are downloaded on first use into
``thirdparty/weights``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..base import Extractor, Features, Matcher
from ..image import default_device, to_rgb, to_tensor
from ..registry import register
from ..thirdparty import use, weights_directory

use("lightglue")
_REQUIRES = ("torch", "torchvision", "kornia", "lightglue")


class _LightGlueExtractor(Extractor):
    """Common wrapper for the ``lightglue`` package extractors."""

    _class_name = ""

    def __init__(self, device: str | None = None, **options):
        import lightglue
        import torch

        self._torch = torch
        self.device = default_device(device)
        model_class = getattr(lightglue, self._class_name)
        with weights_directory():
            self.model = model_class(**options).eval().to(self.device)

    def extract(self, images: Sequence[np.ndarray]) -> list[Features]:
        results = []
        with self._torch.inference_mode():
            for image in images:
                tensor = to_tensor(to_rgb(image), self.device)
                # The frontend already resized the image; disable the
                # package's own resizing.
                output = self.model.extract(tensor, resize=None)
                keypoints = output["keypoints"][0]
                scores = output.get("keypoint_scores")
                results.append(Features(
                    keypoints=keypoints.detach().cpu().numpy().astype(np.float64),
                    descriptors=output["descriptors"][0],
                    image_size=(int(image.shape[1]), int(image.shape[0])),
                    scores=None if scores is None else scores[0].detach().cpu().numpy(),
                ))
        return results

    def warmup(self, image_size: tuple[int, int]) -> None:
        dummy = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        self.extract([dummy])


@register("extractor", "aliked", requires=_REQUIRES, auto_priority=100,
          default_matcher="lightglue")
class AlikedExtractor(_LightGlueExtractor):
    """ALIKED; options e.g. ``model_name``, ``max_num_keypoints``,
    ``detection_threshold``, ``nms_radius``."""

    descriptor = "aliked"
    _class_name = "ALIKED"


@register("extractor", "superpoint", requires=_REQUIRES, default_matcher="lightglue")
class SuperPointExtractor(_LightGlueExtractor):
    descriptor = "superpoint"
    _class_name = "SuperPoint"


@register("extractor", "disk", requires=_REQUIRES, default_matcher="lightglue")
class DiskExtractor(_LightGlueExtractor):
    descriptor = "disk"
    _class_name = "DISK"


@register("matcher", "lightglue", requires=_REQUIRES)
class LightGlueMatcher(Matcher):
    """LightGlue; options e.g. ``filter_threshold``, ``depth_confidence``,
    ``width_confidence``, ``mp``, ``flash``."""

    _SUPPORTED = ("aliked", "superpoint", "disk")

    def __init__(self, device: str | None = None, **options):
        import torch

        self._torch = torch
        self.device = default_device(device)
        self._options = options
        self.model = None

    def accepts(self, descriptor: str) -> bool:
        return descriptor in self._SUPPORTED

    def bind(self, descriptor: str) -> None:
        from lightglue import LightGlue

        with weights_directory():
            self.model = LightGlue(features=descriptor, **self._options).eval().to(self.device)

    def match(self, pairs: Sequence[tuple[Features, Features]]) -> list[np.ndarray]:
        if self.model is None:
            raise RuntimeError("LightGlueMatcher.bind() was not called")
        torch = self._torch
        results = []
        with torch.inference_mode():
            for first, second in pairs:
                if len(first) == 0 or len(second) == 0:
                    results.append(np.empty((0, 2), dtype=np.int64))
                    continue
                data = {"image0": self._inputs(first), "image1": self._inputs(second)}
                output = self.model(data)
                results.append(output["matches"][0].detach().cpu().numpy().astype(np.int64))
        return results

    def _inputs(self, features: Features) -> dict:
        torch = self._torch
        return {
            "keypoints": torch.as_tensor(
                features.keypoints, dtype=torch.float32, device=self.device)[None],
            "descriptors": torch.as_tensor(features.descriptors, device=self.device)[None],
            "image_size": torch.tensor(
                [features.image_size], dtype=torch.float32, device=self.device),
        }
