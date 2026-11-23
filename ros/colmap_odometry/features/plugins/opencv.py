"""OpenCV SIFT/ORB extractors and a brute-force nearest-neighbour matcher.

These need nothing beyond OpenCV and are the fallback when no deep-learning
backend is installed.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..base import Extractor, Features, Matcher
from ..image import to_gray
from ..registry import register


def _as_numpy(descriptors) -> np.ndarray:
    if hasattr(descriptors, "detach"):
        descriptors = descriptors.detach().cpu().numpy()
    return np.asarray(descriptors)


@register("extractor", "sift", requires=("cv2",), auto_priority=0, default_matcher="nn")
class SiftExtractor(Extractor):
    """OpenCV SIFT, optionally with RootSIFT descriptor normalization."""

    descriptor = "sift"

    def __init__(
        self,
        max_keypoints: int = 4096,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
        octave_layers: int = 3,
        root_sift: bool = True,
    ):
        self._sift = cv2.SIFT_create(
            nfeatures=int(max_keypoints),
            nOctaveLayers=int(octave_layers),
            contrastThreshold=float(contrast_threshold),
            edgeThreshold=float(edge_threshold),
            sigma=float(sigma),
        )
        self._root_sift = bool(root_sift)

    def extract(self, images: Sequence[np.ndarray]) -> list[Features]:
        results = []
        for image in images:
            keypoints, descriptors = self._sift.detectAndCompute(to_gray(image), None)
            if descriptors is None or not keypoints:
                points = np.empty((0, 2))
                descriptors = np.empty((0, 128), dtype=np.float32)
                scores = np.empty(0)
                scales = orientations = np.empty(0)
            else:
                points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float64)
                scores = np.array([keypoint.response for keypoint in keypoints])
                scales = np.array([keypoint.size for keypoint in keypoints])
                orientations = np.deg2rad([keypoint.angle for keypoint in keypoints])
                descriptors = descriptors.astype(np.float32)
                if self._root_sift:
                    descriptors /= np.maximum(np.abs(descriptors).sum(axis=1, keepdims=True), 1e-12)
                    descriptors = np.sqrt(descriptors)
            results.append(Features(
                keypoints=points,
                descriptors=descriptors,
                image_size=(int(image.shape[1]), int(image.shape[0])),
                scores=scores,
                extra={"scales": scales, "orientations": orientations},
            ))
        return results


@register("extractor", "orb", requires=("cv2",), default_matcher="nn")
class OrbExtractor(Extractor):
    """OpenCV ORB with binary descriptors (matched with Hamming distance)."""

    descriptor = "orb"

    def __init__(self, max_keypoints: int = 2000, scale_factor: float = 1.2,
                 levels: int = 8, fast_threshold: int = 20):
        self._orb = cv2.ORB_create(
            nfeatures=int(max_keypoints),
            scaleFactor=float(scale_factor),
            nlevels=int(levels),
            fastThreshold=int(fast_threshold),
        )

    def extract(self, images: Sequence[np.ndarray]) -> list[Features]:
        results = []
        for image in images:
            keypoints, descriptors = self._orb.detectAndCompute(to_gray(image), None)
            if descriptors is None or not keypoints:
                points = np.empty((0, 2))
                descriptors = np.empty((0, 32), dtype=np.uint8)
            else:
                points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float64)
            results.append(Features(
                keypoints=points,
                descriptors=descriptors,
                image_size=(int(image.shape[1]), int(image.shape[0])),
            ))
        return results


@register("matcher", "nn", requires=("cv2",))
class NearestNeighborMatcher(Matcher):
    """Brute-force nearest neighbour with Lowe's ratio test and mutual check.

    Float descriptors use the L2 norm and ``uint8`` descriptors are treated
    as binary strings (Hamming norm), so any extractor producing such arrays
    (NumPy or torch) can use this matcher.
    """

    def __init__(self, ratio_threshold: float = 0.8, mutual: bool = True):
        if not 0.0 < ratio_threshold <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")
        self._ratio = float(ratio_threshold)
        self._mutual = bool(mutual)

    def accepts(self, descriptor: str) -> bool:
        return True

    def match(self, pairs: Sequence[tuple[Features, Features]]) -> list[np.ndarray]:
        return [self._match_pair(first, second) for first, second in pairs]

    def _match_pair(self, first: Features, second: Features) -> np.ndarray:
        descriptors1 = _as_numpy(first.descriptors)
        descriptors2 = _as_numpy(second.descriptors)
        if len(descriptors1) < 2 or len(descriptors2) < 2:
            return np.empty((0, 2), dtype=np.int64)
        if descriptors1.dtype == np.uint8:
            norm = cv2.NORM_HAMMING
        else:
            norm = cv2.NORM_L2
            descriptors1 = descriptors1.astype(np.float32)
            descriptors2 = descriptors2.astype(np.float32)
        matcher = cv2.BFMatcher(norm)
        forward = matcher.knnMatch(descriptors1, descriptors2, k=2)
        matches = []
        for candidates in forward:
            if len(candidates) < 2:
                continue
            best, second_best = candidates
            if best.distance <= self._ratio * second_best.distance:
                matches.append((best.queryIdx, best.trainIdx))
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
        if self._mutual and len(matches):
            backward = matcher.match(descriptors2, descriptors1)
            reverse = np.full(len(descriptors2), -1, dtype=np.int64)
            for item in backward:
                reverse[item.queryIdx] = item.trainIdx
            matches = matches[reverse[matches[:, 1]] == matches[:, 0]]
        return matches
