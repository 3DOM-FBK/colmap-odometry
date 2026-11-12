"""Image conversions shared by feature backends.

Frames enter the pipeline as ``uint8`` arrays that are either single-channel
grayscale or 3-channel BGR (OpenCV / cv_bridge convention).
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_tensor(image: np.ndarray, device):
    """Return a float ``(1, C, H, W)`` tensor in ``[0, 1]``."""
    import torch

    array = image[:, :, None] if image.ndim == 2 else image
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return (tensor.float() / 255.0)[None].to(device)


def default_device(requested: str | None = None):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
