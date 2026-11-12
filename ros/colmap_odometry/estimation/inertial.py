"""Inertial factors on the rotation of ``T_rig(current)_rig(previous)``.

* A gyro rotation prior: residual ``Log(R_prior^T R)`` with the
  preintegrated rotation covariance.
* A gravity-direction factor: the previous-frame gravity direction rotated
  into the current frame must align with the current measured direction.
  The cross-product residual has two degrees of freedom and cannot
  constrain yaw about gravity.

Both use the solver's left rotation perturbation ``Exp(e) R``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.lie import skew, so3_left_jacobian_inverse, so3_log


@dataclass(frozen=True)
class MotionPrior:
    """Predicted interval motion supplied to the odometry (e.g. by an IMU).

    ``seed`` is a predicted ``T_rig(current)_rig(previous)``; it only
    initializes a purely visual PnP refinement, which competes with the
    RANSAC estimate.  ``factors`` enter the refinement as residuals.
    """

    seed: np.ndarray | None = None
    seed_max_error_px: float = 4.0
    seed_min_inliers: int = 12
    factors: "InertialFactors | None" = None


@dataclass(frozen=True)
class InertialFactors:
    rotation_prior: np.ndarray | None = None
    rotation_covariance: np.ndarray | None = None
    gravity_previous: np.ndarray | None = None
    gravity_current: np.ndarray | None = None
    gravity_sigma_rad: float | None = None

    @property
    def uses_rotation(self) -> bool:
        return self.rotation_prior is not None and self.rotation_covariance is not None

    @property
    def uses_gravity(self) -> bool:
        return (self.gravity_previous is not None and self.gravity_current is not None
                and self.gravity_sigma_rad is not None
                and np.isfinite(self.gravity_sigma_rad) and self.gravity_sigma_rad > 0.0)

    def augment(self, result: dict, pose: np.ndarray, with_jacobians: bool) -> None:
        """Add factor costs (and normal-equation terms) to a linearization."""
        rotation = pose[:3, :3]
        if with_jacobians and result.get("h_pose") is not None:
            result["h_pose"] = result["h_pose"].copy()
            result["g_pose"] = result["g_pose"].copy()
        else:
            with_jacobians = False

        if self.uses_rotation:
            prior = np.asarray(self.rotation_prior, dtype=np.float64).reshape(3, 3)
            covariance = np.asarray(self.rotation_covariance, dtype=np.float64).reshape(3, 3)
            covariance = 0.5 * (covariance + covariance.T)
            values, vectors = np.linalg.eigh(covariance)
            if np.all(np.isfinite(values)):
                information = vectors @ np.diag(1.0 / np.maximum(values, 1.0e-12)) @ vectors.T
                residual = so3_log(prior.T @ rotation)
                if np.all(np.isfinite(residual)):
                    result["cost"] += float(0.5 * residual @ information @ residual)
                    result["dimensions"] += 3
                    if with_jacobians:
                        # Log(P^T Exp(e) R) = Log(Exp(r) Exp(R^T e))
                        #                   ~ r + Jr^-1(r) R^T e
                        jacobian = np.zeros((3, 6))
                        jacobian[:, :3] = so3_left_jacobian_inverse(-residual) @ rotation.T
                        result["h_pose"] += jacobian.T @ information @ jacobian
                        result["g_pose"] += jacobian.T @ information @ residual

        if self.uses_gravity:
            previous = np.asarray(self.gravity_previous, dtype=np.float64).reshape(3)
            current = np.asarray(self.gravity_current, dtype=np.float64).reshape(3)
            previous_norm = float(np.linalg.norm(previous))
            current_norm = float(np.linalg.norm(current))
            if previous_norm > 1.0e-9 and current_norm > 1.0e-9:
                predicted = rotation @ (previous / previous_norm)
                current = current / current_norm
                residual = np.cross(predicted, current)
                information = 1.0 / float(self.gravity_sigma_rad) ** 2
                result["cost"] += float(0.5 * information * residual @ residual)
                result["dimensions"] += 2
                if with_jacobians:
                    # (Exp(e) p) x c ~ p x c + [c]x [p]x e
                    jacobian = np.zeros((3, 6))
                    jacobian[:, :3] = skew(current) @ skew(predicted)
                    result["h_pose"] += information * (jacobian.T @ jacobian)
                    result["g_pose"] += information * (jacobian.T @ residual)

        if with_jacobians and result.get("visual_h_pose") is not None:
            result["inertial_h_pose"] = result["h_pose"] - result["visual_h_pose"]
