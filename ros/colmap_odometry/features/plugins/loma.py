"""LoMa (DaD detector + DeDoDe descriptor + LoMa matcher) as a joint model.

Vendored as the ``thirdparty/LoMa`` submodule (davnords/LoMa); needs torch,
torchvision, einops and Pillow.  Weights are downloaded on first use into
``thirdparty/weights``.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from typing import Sequence

import numpy as np

from ..base import Features, JointModel
from ..image import to_rgb
from ..registry import register
from ..thirdparty import use, weights_directory

use("loma")

# ViT patch size of the DeDoDe backbone; inputs are padded to a multiple.
_PATCH = 14


@register("joint", "loma", requires=("torch", "torchvision", "einops", "PIL", "loma"))
class LoMaModel(JointModel):
    descriptor = "loma"

    def __init__(
        self,
        model: str = "loma_B128",
        num_keypoints: int | None = None,
        filter_threshold: float | None = None,
        compile: bool = False,
    ):
        import loma
        import torch

        self._torch = torch
        configs = {
            "loma_B128": loma.LoMaB128,
            "loma_B": loma.LoMaB,
            "loma_L": loma.LoMaL,
            "loma_G": loma.LoMaG,
            "loma_R": loma.LoMaR,
        }
        if model not in configs:
            raise ValueError(f"unknown LoMa model '{model}'; choose from {sorted(configs)}")
        # TensorFloat32 for the large float32 matmuls, as recommended upstream.
        torch.set_float32_matmul_precision("high")
        config = configs[model]()
        overrides = {"compile": False}
        if num_keypoints is not None:
            overrides["num_keypoints"] = int(num_keypoints)
        config = dataclasses.replace(config, **overrides)
        self.num_keypoints = int(config.num_keypoints)
        self.filter_threshold = float(
            config.filter_threshold if filter_threshold is None else filter_threshold)
        with weights_directory():
            self.model = loma.LoMa(config)
        self.device = next(self.model.parameters()).device
        if compile:
            # Frames are processed on a worker thread; inductor CUDA graphs
            # keep thread-local state, so capture must stay disabled.
            options = {"triton.cudagraphs": False}
            self.model._detector.compile(options=options)
            self.model._descriptor.compile(options=options)
            if sys.platform == "linux":
                self.model.compile(options=options)

    def extract(self, images: Sequence[np.ndarray]) -> list[Features]:
        torch = self._torch
        results = []
        with torch.inference_mode():
            for image in images:
                height, width = image.shape[:2]
                rgb = to_rgb(image)
                pad_height = -height % _PATCH
                pad_width = -width % _PATCH
                if pad_height or pad_width:
                    rgb = np.pad(rgb, ((0, pad_height), (0, pad_width), (0, 0)))
                tensor = torch.from_numpy(rgb / 255.0).permute(2, 0, 1).float()[None]
                keypoints, descriptors, padded_height, padded_width = \
                    self.model.detect_and_describe(tensor, num_keypoints=self.num_keypoints)
                # Normalized [-1, 1] coordinates of the padded image; padding
                # is bottom/right, so pixel coordinates are unaffected.
                pixels = self.model._detector.to_pixel_coords(
                    keypoints[0], padded_height, padded_width)
                results.append(Features(
                    keypoints=pixels.detach().cpu().numpy().astype(np.float64),
                    descriptors=descriptors[0].clone(),
                    image_size=(int(width), int(height)),
                ))
        return results

    def match(self, pairs: Sequence[tuple[Features, Features]]) -> list[np.ndarray]:
        if not pairs:
            return []
        shapes = {(len(a), len(b), tuple(a.descriptors.shape), tuple(b.descriptors.shape))
                  for a, b in pairs}
        if len(shapes) > 1:
            return [self.match([pair])[0] for pair in pairs]
        from loma.loma import filter_matches

        torch = self._torch
        with torch.inference_mode():
            k1 = torch.stack([self._normalized(a) for a, _ in pairs]).to(self.device)
            k2 = torch.stack([self._normalized(b) for _, b in pairs]).to(self.device)
            d1 = torch.stack([a.descriptors for a, _ in pairs]).to(self.device)
            d2 = torch.stack([b.descriptors for _, b in pairs]).to(self.device)
            scores = self.model(k1, k2, d1, d2)["scores"].clone()
            matches0, _, _, _ = filter_matches(scores, self.filter_threshold)
            results = []
            for row in matches0:
                valid = row > -1
                first = torch.where(valid)[0]
                results.append(torch.stack((first, row[valid]), dim=-1).cpu().numpy())
        return results

    def _normalized(self, features: Features):
        # LoMa's positional encoding expects coordinates in [-1, 1] of the
        # image the keypoints refer to (native pixels after the frontend).
        width, height = features.image_size
        keypoints = self._torch.as_tensor(features.keypoints, dtype=self._torch.float32)
        return self._torch.stack(
            (2 * keypoints[:, 0] / width - 1, 2 * keypoints[:, 1] / height - 1), dim=-1)

    def warmup(self, image_size: tuple[int, int]) -> None:
        """Run detector, descriptor and matcher once at production shape."""
        dummy = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Online softmax is disabled on the fly.*")
            features = self.extract([dummy, dummy])
            self.match([(features[0], features[1])])
