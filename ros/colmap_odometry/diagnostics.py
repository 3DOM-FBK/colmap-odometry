"""Lightweight per-frame stage timing."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np


class Profiler:
    """Accumulates stage durations of the current frame.

    ``stage`` is a plain string readable from other threads (e.g. a stall
    watchdog) without locking.
    """

    def __init__(self):
        self.timings: dict[str, float] = {}
        self.stage = "idle"

    def reset(self) -> dict[str, float]:
        timings, self.timings = self.timings, {}
        return timings

    @contextmanager
    def measure(self, name: str):
        previous = self.stage
        self.stage = name
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - start
            self.stage = previous


def json_default(value):
    """``json.dumps`` fallback for numpy values in frame reports."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)
