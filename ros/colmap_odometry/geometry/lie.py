"""Rotation, rigid-transform and covariance-transport utilities.

Conventions used throughout the package:

* A transform ``T_a_b`` is a 4x4 matrix mapping points from frame ``b`` to
  frame ``a``: ``p_a = T_a_b @ p_b``.  COLMAP's ``cam_from_world`` is
  ``T_cam_world`` in this notation.
* Tangent vectors are ordered ``[rotation, translation]`` ("rt").  ROS
  messages use ``[translation, rotation]``; convert only at the ROS boundary
  with :func:`rt_to_ros` / :func:`ros_to_rt`.
* A *left* perturbation is ``Exp(delta) @ T``, a *right* perturbation is
  ``T @ Exp(delta)``.  Covariances move between the two with
  :func:`adjoint`: ``delta_left = Ad(T) delta_right``.
* Quaternions are stored ``[x, y, z, w]`` (ROS and COLMAP order).
"""

from __future__ import annotations

import math

import numpy as np

# Index permutation between [rotation, translation] and ROS [translation, rotation].
_RT_ROS_ORDER = np.array([3, 4, 5, 0, 1, 2])


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix ``[v]x``."""
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def skew_batch(vectors: np.ndarray) -> np.ndarray:
    """Return ``[v]x`` for each row of an ``(N, 3)`` array."""
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    result = np.zeros((len(vectors), 3, 3), dtype=np.float64)
    result[:, 0, 1] = -vectors[:, 2]
    result[:, 0, 2] = vectors[:, 1]
    result[:, 1, 0] = vectors[:, 2]
    result[:, 1, 2] = -vectors[:, 0]
    result[:, 2, 0] = -vectors[:, 1]
    result[:, 2, 1] = vectors[:, 0]
    return result


def so3_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    tangent = skew(vector)
    if angle < 1.0e-8:
        return np.eye(3) + tangent + 0.5 * tangent @ tangent
    a = math.sin(angle) / angle
    b = (1.0 - math.cos(angle)) / (angle * angle)
    return np.eye(3) + a * tangent + b * tangent @ tangent


def so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = project_rotation(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return 0.5 * vee
    if math.pi - angle < 1.0e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        # The eigenvector sign is arbitrary; pick the one consistent with vee.
        if float(np.dot(axis, vee)) < 0.0:
            axis = -axis
        return angle * axis
    return angle * vee / (2.0 * math.sin(angle))


def so3_left_jacobian(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    omega = skew(vector)
    if theta < 1.0e-7:
        return np.eye(3) + 0.5 * omega + (1.0 / 6.0) * omega @ omega
    theta2 = theta * theta
    return (
        np.eye(3)
        + ((1.0 - math.cos(theta)) / theta2) * omega
        + ((theta - math.sin(theta)) / (theta2 * theta)) * omega @ omega
    )


def so3_left_jacobian_inverse(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    omega = skew(vector)
    if theta < 1.0e-7:
        return np.eye(3) - 0.5 * omega + (1.0 / 12.0) * omega @ omega
    coefficient = (
        1.0 / (theta * theta)
        - (1.0 + math.cos(theta)) / (2.0 * theta * max(math.sin(theta), 1.0e-12))
    )
    return np.eye(3) - 0.5 * omega + coefficient * omega @ omega


def se3_exp(vector: np.ndarray) -> np.ndarray:
    """Map an rt tangent vector to an SE(3) matrix."""
    vector = np.asarray(vector, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = so3_exp(vector[:3])
    transform[:3, 3] = so3_left_jacobian(vector[:3]) @ vector[3:]
    return transform


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Map an SE(3) matrix to an rt tangent vector."""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation_vector = so3_log(transform[:3, :3])
    translation = so3_left_jacobian_inverse(rotation_vector) @ transform[:3, 3]
    return np.concatenate((rotation_vector, translation))


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation (SVD projection)."""
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    result = u @ vh
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def align_vector_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimum rotation mapping ``source`` onto ``target``."""
    source = np.asarray(source, dtype=np.float64).reshape(3).copy()
    target = np.asarray(target, dtype=np.float64).reshape(3).copy()
    source /= max(np.linalg.norm(source), 1.0e-12)
    target /= max(np.linalg.norm(target), 1.0e-12)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1.0e-10:
        if cosine > 0.0:
            return np.eye(3)
        basis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, basis)
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        return so3_exp(math.pi * axis)
    return so3_exp(math.atan2(sine, cosine) * cross / sine)


# --------------------------------------------------------------------------
# Rigid transforms
# --------------------------------------------------------------------------

def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation_t = transform[:3, :3].T
    return make_transform(rotation_t, -rotation_t @ transform[:3, 3])


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def left_update(transform: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply the solver's left retraction ``[Exp(w) | v] @ T``.

    This is SO(3) x R^3 on the left, which agrees with the SE(3) left
    perturbation ``Exp(delta) @ T`` to first order; covariances from the
    solver are therefore left-SE(3) covariances.
    """
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    return make_transform(so3_exp(delta[:3]), delta[3:]) @ transform


def quaternion_to_rotation(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a unit quaternion ``[x, y, z, w]`` with ``w >= 0``."""
    m = project_rotation(rotation)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                      0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def transform_from_pose(translation: np.ndarray, xyzw: np.ndarray) -> np.ndarray:
    return make_transform(quaternion_to_rotation(xyzw), translation)


def pose_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(translation, quaternion_xyzw)``."""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return transform[:3, 3].copy(), rotation_to_quaternion(transform[:3, :3])


# --------------------------------------------------------------------------
# Covariance transport
# --------------------------------------------------------------------------

def adjoint(transform: np.ndarray) -> np.ndarray:
    """SE(3) adjoint in rt order: ``Exp(Ad(T) x) = T Exp(x) T^-1``."""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[3:, :3] = skew(transform[:3, 3]) @ rotation
    result[3:, 3:] = rotation
    return result


def transport(covariance: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Return the symmetrized ``J C J^T``."""
    result = jacobian @ np.asarray(covariance, dtype=np.float64) @ jacobian.T
    return 0.5 * (result + result.T)


def rt_to_ros(covariance_rt: np.ndarray) -> np.ndarray:
    covariance_rt = np.asarray(covariance_rt, dtype=np.float64).reshape(6, 6)
    return covariance_rt[np.ix_(_RT_ROS_ORDER, _RT_ROS_ORDER)].copy()


def ros_to_rt(covariance_ros: np.ndarray) -> np.ndarray:
    # The permutation is its own inverse.
    return rt_to_ros(covariance_ros)


def positive_definite_or_none(
    covariance: np.ndarray,
    relative_floor: float = 1.0e-12,
    absolute_floor: float = 1.0e-15,
) -> np.ndarray | None:
    """Symmetrize a covariance, repair round-off and reject indefinite input.

    Small negative eigenvalues from covariance transport are clamped to a
    floor relative to the largest eigenvalue; a genuinely indefinite matrix
    (negative eigenvalue beyond 1e-8 of the largest) is rejected.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    matrix = 0.5 * (matrix + matrix.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
        return None
    if eigenvalues[0] < -1.0e-8 * eigenvalues[-1]:
        return None
    floor = max(absolute_floor, eigenvalues[-1] * relative_floor)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def inverse_with_floor(
    information: np.ndarray,
    relative_floor: float = 1.0e-12,
    absolute_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Invert a PSD information matrix, capping its condition number.

    Unobservable directions keep an appropriately large variance instead of
    making the inverse fail.  Returns ``(covariance, eigenvalues)`` or None
    for a non-finite, zero or clearly indefinite input.
    """
    information = 0.5 * (information + information.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(information)
    except np.linalg.LinAlgError:
        return None
    maximum = float(eigenvalues[-1])
    if not np.isfinite(maximum) or maximum <= 1.0e-12:
        return None
    if eigenvalues[0] < -1.0e-8 * maximum:
        return None
    floor = max(absolute_floor, maximum * relative_floor)
    covariance = eigenvectors @ np.diag(1.0 / np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T), eigenvalues
