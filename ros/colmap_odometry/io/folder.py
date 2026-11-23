"""Read synchronized camera frames from a folder of images.

Two layouts are recognized per camera directory (named like the camera,
e.g. ``cam0``), optionally below a ``mav0`` directory:

* EuRoC/ASL (EuRoC MAV, TUM-VI, Kalibr exports)::

      cam0/data.csv          # "#timestamp [ns],filename"
      cam0/data/<file>.png

* plain: images named by their timestamp, either directly in ``cam0/`` or
  in ``cam0/data/``.  An integer name is nanoseconds
  (``1520530308199447626.png``), a name with a decimal point is seconds
  (``1520530308.199447626.png``).

Images are decoded like the bag reader decodes ``sensor_msgs/Image``:
8-bit images are used as stored, 16-bit images are scaled to 8 bit as
cv_bridge does, colour images become BGR.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .common import SyncedFrame, synchronize

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm"}


def dataset_root(path: str | Path) -> Path:
    """The directory holding the sensor directories (``path`` or ``path/mav0``)."""
    path = Path(path)
    return path / "mav0" if (path / "mav0").is_dir() else path


def is_dataset_folder(path: str | Path, cameras: Sequence[str]) -> bool:
    root = dataset_root(path)
    return bool(cameras) and all((root / name).is_dir() for name in cameras)


def parse_stamp(text: str) -> int:
    """Nanoseconds from an integer (ns) or decimal (s) timestamp string."""
    text = text.strip()
    try:
        if "." in text or "e" in text.lower():
            return int(Decimal(text) * 1_000_000_000)
        return int(text)
    except (InvalidOperation, ValueError):
        raise ValueError(f"'{text}' is not a timestamp") from None


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read image {path}")
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)
    elif image.dtype != np.uint8:
        raise ValueError(f"unsupported image type {image.dtype} in {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _rows(path: Path) -> Iterator[list[str]]:
    """Data rows of a CSV file; comments and a header line are skipped."""
    with path.open(newline="", encoding="utf-8") as handle:
        for number, row in enumerate(csv.reader(handle)):
            row = [cell.strip() for cell in row]
            if not any(row) or row[0].startswith("#") or (number == 0 and not _is_number(row[0])):
                continue
            yield row


def camera_images(directory: Path) -> list[tuple[int, Path]]:
    """``(stamp_ns, path)`` of one camera, sorted by time."""
    images = directory / "data" if (directory / "data").is_dir() else directory
    listing = directory / "data.csv"
    if listing.is_file():
        entries = [(parse_stamp(row[0]), images / row[1]) for row in _rows(listing)]
    else:
        entries = []
        for path in images.iterdir():
            if path.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    entries.append((parse_stamp(path.stem), path))
                except ValueError:
                    raise ValueError(f"{path.name}: images without data.csv must be named "
                                     "by their timestamp") from None
    if not entries:
        raise ValueError(f"no images in {directory}")
    return sorted(entries, key=lambda entry: entry[0])


class FolderReader:
    """Iterate time-synchronized image sets from a folder."""

    def __init__(self, path: str | Path):
        self._root = dataset_root(path)

    def frames(
        self,
        cameras: Sequence[str],
        slop_sec: float = 0.005,
        start_sec: float = 0.0,
        max_frames: int | None = None,
    ) -> Iterator[SyncedFrame]:
        """Yield frames whose images were stamped within ``slop_sec``."""
        cameras = list(cameras)
        missing = [name for name in cameras if not (self._root / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"{self._root} has no camera directories {missing}")
        events = sorted(
            ((stamp, index, path) for index, name in enumerate(cameras)
             for stamp, path in camera_images(self._root / name)),
            key=lambda event: (event[0], event[1]))
        items = ((cameras[index], stamp, path) for stamp, index, path in events)
        yield from synchronize(items, cameras, read_image, slop_sec, start_sec, max_frames)
