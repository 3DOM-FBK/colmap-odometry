"""Debug images (ROS independent)."""

from __future__ import annotations

import cv2
import numpy as np

from ..odometry import MatchDebug


def as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return cv2.cvtColor(image.reshape(image.shape[:2]), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _subsample(values: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(values) <= limit:
        return values
    return values[np.linspace(0, len(values) - 1, limit, dtype=np.int64)]


def match_panel(first_image: np.ndarray, second_image: np.ndarray, item: MatchDebug,
                max_features: int, max_matches: int) -> np.ndarray:
    left, right = as_bgr(first_image), as_bgr(second_image)
    keypoints_left = np.asarray(item.keypoints_first, dtype=np.float32)
    keypoints_right = np.asarray(item.keypoints_second, dtype=np.float32)
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / max(right.shape[0], 1)
        right = cv2.resize(right, (max(1, round(right.shape[1] * scale)), left.shape[0]))
        keypoints_right = keypoints_right * scale
    panel = np.hstack([left, right])
    offset = left.shape[1]
    for points, color, shift in ((keypoints_left, (80, 220, 255), 0),
                                 (keypoints_right, (255, 180, 80), offset)):
        for x, y in _subsample(points, max_features):
            cv2.circle(panel, (int(round(x)) + shift, int(round(y))), 2, color, -1, cv2.LINE_AA)
    for index, (a, b) in enumerate(_subsample(np.asarray(item.matches), max_matches)):
        color = (80 + (37 * index) % 176, 80 + (71 * index) % 176, 80 + (113 * index) % 176)
        p1 = tuple(int(round(v)) for v in keypoints_left[a])
        p2 = (int(round(keypoints_right[b][0])) + offset, int(round(keypoints_right[b][1])))
        cv2.line(panel, p1, p2, color, 1, cv2.LINE_AA)
    label = (f"{item.first} <-> {item.second}: {len(keypoints_left)} / "
             f"{len(keypoints_right)} features, {len(item.matches)} matches")
    for thickness, color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
        cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness,
                    cv2.LINE_AA)
    return panel


def stack_vertical(panels: list[np.ndarray]) -> np.ndarray:
    width = max(panel.shape[1] for panel in panels)
    return np.vstack([cv2.copyMakeBorder(panel, 0, 0, 0, width - panel.shape[1],
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
                      for panel in panels])


def pixel_colors(image: np.ndarray, pixels: np.ndarray, rgb_input: bool) -> np.ndarray:
    """``(N, 3)`` RGB colors sampled at pixel positions."""
    height, width = image.shape[:2]
    u = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, width - 1)
    v = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, height - 1)
    samples = image[v, u]
    if samples.ndim == 1:
        return np.repeat(samples[:, None], 3, axis=1)
    samples = samples[:, :3]
    return samples if rgb_input else samples[:, ::-1]
