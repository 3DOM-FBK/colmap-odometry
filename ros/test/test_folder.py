"""Image-folder input of the offline runner."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from colmap_odometry.io.folder import FolderReader, is_dataset_folder, parse_stamp
from colmap_odometry.offline import is_bag

STAMPS = [1_520_530_308_199_447_626, 1_520_530_308_249_448_626, 1_520_530_308_299_450_626]


def _image(value: int, dtype=np.uint8) -> np.ndarray:
    return np.full((4, 6), value, dtype=dtype)


def test_euroc_layout(tmp_path):
    for camera, offset in (("cam0", 0), ("cam1", 100)):
        data = tmp_path / "mav0" / camera / "data"
        data.mkdir(parents=True)
        rows = ["#timestamp [ns],filename"]
        for index, stamp in enumerate(STAMPS):
            cv2.imwrite(str(data / f"{stamp}.png"), _image(index + offset))
            rows.append(f"{stamp},{stamp}.png")
        (data.parent / "data.csv").write_text("\n".join(rows) + "\n")
    assert is_dataset_folder(tmp_path, ["cam0", "cam1"]) and not is_bag(tmp_path)
    frames = list(FolderReader(tmp_path).frames(["cam0", "cam1"]))
    assert [frame.stamp_ns for frame in frames] == STAMPS
    assert [(int(a[0, 0]), int(b[0, 0])) for a, b in (f.images for f in frames)] == \
        [(0, 100), (1, 101), (2, 102)]
    assert len(list(FolderReader(tmp_path).frames(["cam0", "cam1"], start_sec=0.06))) == 1
    assert len(list(FolderReader(tmp_path).frames(["cam0", "cam1"], max_frames=2))) == 2


def test_plain_layout_seconds_names_and_unmatched_images(tmp_path):
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    for stamp in STAMPS:
        name = f"{stamp // 10**9}.{stamp % 10**9:09d}.png"
        cv2.imwrite(str(tmp_path / "left" / name), _image(1))
    # The right camera misses the first frame and is 1 ms late.
    for stamp in STAMPS[1:]:
        cv2.imwrite(str(tmp_path / "right" / f"{stamp + 1_000_000}.png"),
                    _image(65535, np.uint16))
    frames = list(FolderReader(tmp_path).frames(["left", "right"], slop_sec=0.005))
    assert [frame.stamp_ns for frame in frames] == STAMPS[1:]
    # 16-bit images are scaled to 8 bit like cv_bridge's mono16 -> mono8.
    assert frames[0].images[1].dtype == np.uint8 and frames[0].images[1][0, 0] == 255
    assert not list(FolderReader(tmp_path).frames(["left", "right"], slop_sec=0.0005))


def test_parse_stamp():
    assert parse_stamp("1520530308199447626") == 1520530308199447626
    assert parse_stamp("1520530308.199447626") == 1520530308199447626
    assert parse_stamp("12.5") == 12_500_000_000
    with pytest.raises(ValueError):
        parse_stamp("frame_0001")
