"""Sliding-window gyro-bias estimate from gyro and visual relative rotations.

For every frame interval the gyro rotation ``dR(b) = prod Exp((w_i - b) dt_i)``
is preintegrated with ``b = 0`` together with its bias Jacobian ``J``
(``dR(b) ~= dR(0) Exp(J b)``), and paired with the visual-only relative
rotation ``R_v`` and its covariance.  The bias is the robust (Huber)
least-squares solution of

    r_k(b) = Log(R_v,k^T dR_k(0)) + J_k b

over the intervals of the last ``window`` seconds, assuming the bias is
constant over the window, with a weak zero-mean prior.  Only rotations
enter: unlike the filter, translation residuals cannot move the gyro bias
through the attitude/velocity/gravity coupling, and the finite window
re-weights outliers jointly.  Linearizing at ``b = 0`` is exact to second
order in ``|J b|`` (~1e-4 rad per frame interval).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..geometry.lie import so3_exp, so3_log


@dataclass(frozen=True)
class GyroInterval:
    """Preintegrated gyro rotation ``previous_base_from_current`` at zero bias."""

    start_ns: int
    end_ns: int
    rotation: np.ndarray
    bias_jacobian: np.ndarray

    @property
    def duration_sec(self) -> float:
        return (self.end_ns - self.start_ns) * 1e-9

    def rotation_with_bias(self, bias: np.ndarray) -> np.ndarray:
        return self.rotation @ so3_exp(self.bias_jacobian @ bias)


def preintegrate(stamps_ns: np.ndarray, angular_velocities: np.ndarray,
                 start_ns: int, end_ns: int) -> GyroInterval | None:
    """Integrate linearly interpolated angular velocity over ``[start, end]``.

    Needs samples at or before ``start`` and at or after ``end``.
    """
    stamps_ns = np.asarray(stamps_ns, dtype=np.int64)
    if (len(stamps_ns) < 2 or end_ns <= start_ns
            or stamps_ns[0] > start_ns or stamps_ns[-1] < end_ns):
        return None
    inside = stamps_ns[(stamps_ns > start_ns) & (stamps_ns < end_ns)]
    grid = np.concatenate(([start_ns], inside, [end_ns]))
    times = (stamps_ns - start_ns) * 1e-9
    grid_times = (grid - start_ns) * 1e-9
    rates = np.column_stack([np.interp(grid_times, times, angular_velocities[:, axis])
                             for axis in range(3)])
    rotation = np.eye(3)
    jacobian = np.zeros((3, 3))
    for index in range(len(grid) - 1):
        step = grid_times[index + 1] - grid_times[index]
        increment = so3_exp(0.5 * (rates[index] + rates[index + 1]) * step)
        # d/db of R Exp((w - b) dt): carry the previous Jacobian through the
        # increment; the right Jacobian of a ~1e-2 rad step is identity.
        jacobian = increment.T @ jacobian - step * np.eye(3)
        rotation = rotation @ increment
    return GyroInterval(int(start_ns), int(end_ns), rotation, jacobian)


class GyroBiasWindow:
    def __init__(self, window_sec: float, prior_std: float, huber_threshold: float = 3.0,
                 minimum_duration_sec: float = 2.0, iterations: int = 5):
        self.window_ns = int(window_sec * 1e9)
        self.prior_information = np.eye(3) / prior_std ** 2
        self.huber_threshold = float(huber_threshold)
        self.minimum_duration_ns = int(minimum_duration_sec * 1e9)
        self.iterations = int(iterations)
        self._residuals: deque = deque()
        self._jacobians: deque = deque()
        self._information: deque = deque()
        self._stamps: deque = deque()
        self._first_ns: int | None = None
        self.bias: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    def add(self, interval: GyroInterval, visual_rotation: np.ndarray,
            visual_covariance: np.ndarray) -> None:
        """Add an interval with the visual ``previous_base_from_current`` rotation."""
        information = np.linalg.inv(visual_covariance)
        self._residuals.append(so3_log(np.asarray(visual_rotation).T @ interval.rotation))
        self._jacobians.append(interval.bias_jacobian)
        self._information.append(0.5 * (information + information.T))
        self._stamps.append((interval.start_ns, interval.end_ns))
        if self._first_ns is None:
            self._first_ns = interval.start_ns
        while self._stamps and self._stamps[0][0] < interval.end_ns - self.window_ns:
            for items in (self._residuals, self._jacobians, self._information, self._stamps):
                items.popleft()
        if interval.end_ns - self._first_ns >= self.minimum_duration_ns:
            self._solve()

    def _solve(self) -> None:
        residuals = np.array(self._residuals)
        jacobians = np.array(self._jacobians)
        information = np.array(self._information)
        bias = np.zeros(3) if self.bias is None else self.bias
        for _ in range(self.iterations):
            error = residuals + jacobians @ bias
            norm = np.sqrt(np.einsum("ni,nij,nj->n", error, information, error))
            weights = np.minimum(1.0, self.huber_threshold / np.maximum(norm, 1e-12))
            weighted = weights[:, None, None] * information
            jt_w = np.einsum("nji,njk->nik", jacobians, weighted)
            hessian = self.prior_information + (jt_w @ jacobians).sum(axis=0)
            gradient = np.einsum("nij,nj->i", jt_w, residuals)
            bias = np.linalg.solve(hessian, -gradient)
        self.bias = bias
        self.covariance = np.linalg.inv(hessian)
