"""Types and frame synchronization shared by the offline readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class SyncedFrame:
    stamp_ns: int
    images: list[np.ndarray]


@dataclass(frozen=True)
class ImuMessage:
    stamp_ns: int
    acceleration: np.ndarray
    angular_velocity: np.ndarray


def synchronize(
    images: Iterable[tuple[str, int, Any]],
    streams: Sequence[str],
    decode: Callable[[Any], np.ndarray],
    slop_sec: float = 0.005,
    start_sec: float = 0.0,
    max_frames: int | None = None,
) -> Iterator[SyncedFrame]:
    """Group ``(stream, stamp_ns, payload)`` items into frames of all streams.

    Items are buffered per stream and paired when every stream has one
    stamped within ``slop_sec`` of the others; an older unmatched item is
    dropped.  This reproduces exact-stamp synchronization of
    hardware-triggered rigs.  Payloads are decoded only for emitted frames;
    the frame stamp is that of the first stream.
    """
    slop_ns = int(slop_sec * 1e9)
    pending: dict[str, list[tuple[int, Any]]] = {name: [] for name in streams}
    first_stamp = None
    emitted = 0
    for name, stamp, payload in images:
        if first_stamp is None:
            first_stamp = stamp
        if (stamp - first_stamp) * 1e-9 < start_sec:
            continue
        pending[name].append((stamp, payload))
        while all(pending[stream] for stream in streams):
            heads = [pending[stream][0][0] for stream in streams]
            if max(heads) - min(heads) <= slop_ns:
                items = [pending[stream].pop(0) for stream in streams]
                yield SyncedFrame(stamp_ns=items[0][0],
                                  images=[decode(item) for _, item in items])
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    return
            else:
                pending[streams[int(np.argmin(heads))]].pop(0)
