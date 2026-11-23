"""Chaining of verified temporal matches into feature tracks.

Publication semantics: a track is first emitted with its complete history
once it reaches ``min_length`` observations; afterwards each update carries
only observations appended since the previous emission, and completion is
emitted exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    stamp_ns: int
    frame_index: int
    keypoint_index: int
    x: float
    y: float
    has_stereo_point: bool = False
    stereo_x: float = 0.0
    stereo_y: float = 0.0
    #: Stereo point in the observing camera's optical frame.
    point_camera: tuple[float, float, float] = (0.0, 0.0, 0.0)
    triangulation_angle_rad: float = 0.0


@dataclass
class TrackUpdate:
    track_id: int
    camera_name: str
    is_complete: bool
    observations: list[TrackObservation]


@dataclass
class _Track:
    camera_name: str
    observations: list[TrackObservation] = field(default_factory=list)
    published: int = 0


class TrackManager:
    """Tracks of one camera; track ids are shared through ``id_source``."""

    def __init__(self, camera_name: str, min_length: int, id_source: list[int]):
        self.camera_name = camera_name
        self.min_length = max(2, int(min_length))
        self._ids = id_source
        self._tracks: dict[int, _Track] = {}
        # Track id of each keypoint of the most recent frame.
        self._active: dict[int, int] = {}

    def update(
        self,
        matches: np.ndarray,
        previous_observation,
        current_observation,
        updates: list[TrackUpdate],
    ) -> None:
        """Extend tracks with ``(previous_index, current_index)`` matches.

        ``*_observation(index)`` return the :class:`TrackObservation` of a
        keypoint in the previous / current frame.  Tracks that are not
        extended are completed.
        """
        extended: set[int] = set()
        current_active: dict[int, int] = {}
        used_current: set[int] = set()
        previous_ids = set(self._active.values())
        for previous_index, current_index in np.asarray(matches, dtype=np.int64).reshape(-1, 2):
            previous_index, current_index = int(previous_index), int(current_index)
            if current_index in used_current:
                continue
            track_id = self._active.get(previous_index)
            if track_id is not None and track_id in extended:
                continue
            if track_id is None:
                track_id = self._ids[0]
                self._ids[0] += 1
                self._tracks[track_id] = _Track(
                    self.camera_name, [previous_observation(previous_index)])
            self._tracks[track_id].observations.append(current_observation(current_index))
            extended.add(track_id)
            used_current.add(current_index)
            current_active[current_index] = track_id
            self._snapshot(track_id, False, updates)
        for track_id in previous_ids - extended:
            self._finish(track_id, updates)
        self._active = current_active

    def finish_all(self, updates: list[TrackUpdate]) -> None:
        for track_id in list(self._tracks):
            self._finish(track_id, updates)
        self._active.clear()

    def _finish(self, track_id: int, updates: list[TrackUpdate]) -> None:
        if track_id in self._tracks:
            self._snapshot(track_id, True, updates)
            del self._tracks[track_id]

    def _snapshot(self, track_id: int, complete: bool, updates: list[TrackUpdate]) -> None:
        track = self._tracks[track_id]
        if track.published == 0 and len(track.observations) < self.min_length:
            return
        appended = track.observations[track.published:]
        if not appended and not complete:
            return
        updates.append(TrackUpdate(track_id, track.camera_name, complete, list(appended)))
        track.published = len(track.observations)
