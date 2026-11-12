"""The regression check must report known errors and consistent covariances."""

import json

import numpy as np

from colmap_odometry import check
from colmap_odometry.geometry.lie import (
    invert, make_transform, pose_from_transform, rt_to_ros, se3_exp, so3_exp)


def _ground_truth(count=400, rate=20.0):
    stamps = 1000.0 + np.arange(count) / rate
    poses = []
    for stamp in stamps:
        t = stamp - stamps[0]
        poses.append(make_transform(so3_exp([0.1 * np.sin(t), 0.2 * t, 0.05]),
                                    [np.cos(t), np.sin(0.7 * t), 0.1 * t]))
    return stamps, np.asarray(poses)


def _write_tum(path, stamps, poses):
    with open(path, "w") as stream:
        for stamp, pose in zip(stamps, poses):
            translation, quaternion = pose_from_transform(pose)
            stream.write(f"{stamp:.9f} " + " ".join(f"{v:.12f}" for v in
                                                     (*translation, *quaternion)) + "\n")


def _noisy_run(tmp_path, name, stamps, poses, sigma, rng, stated_sigma=None):
    covariance = np.diag(np.square(sigma if stated_sigma is None else stated_sigma))
    chol = np.linalg.cholesky(np.diag(np.square(sigma)))
    estimate = [poses[0]]
    motions = []
    for index in range(1, len(stamps)):
        truth = invert(poses[index - 1]) @ poses[index]
        # Right-local error: Z_true = Z Exp(e)  =>  Z = Z_true Exp(-e).
        measured = truth @ se3_exp(-(chol @ rng.normal(size=6)))
        estimate.append(estimate[-1] @ measured)
        translation, quaternion = pose_from_transform(measured)
        motions.append({"from_stamp_ns": int(round(stamps[index - 1] * 1e9)),
                        "to_stamp_ns": int(round(stamps[index] * 1e9)),
                        "translation": translation.tolist(),
                        "quaternion_xyzw": quaternion.tolist(),
                        "covariance_ros": rt_to_ros(covariance).reshape(-1).tolist(),
                        "covariance_source": "test"})
    _write_tum(tmp_path / f"{name}.tum", stamps, estimate)
    (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(m) for m in motions))
    return tmp_path / f"{name}.tum", tmp_path / f"{name}.jsonl"


def test_metrics_and_nees(tmp_path):
    stamps, poses = _ground_truth(2000)
    # Ground truth at a different rate exercises interpolation.
    truth_stamps = np.arange(stamps[0] - 0.05, stamps[-1] + 0.05, 1 / 120.0)
    _, dense = _ground_truth(len(truth_stamps), 120.0)
    dense_stamps = 1000.0 + np.arange(len(truth_stamps)) / 120.0
    _write_tum(tmp_path / "gt.tum", dense_stamps, dense)
    truth = check.read_trajectory(tmp_path / "gt.tum")

    exact = check.Trajectory(stamps[stamps <= dense_stamps[-1]],
                             poses[stamps <= dense_stamps[-1]])
    result = check.evaluate(exact, None, truth, 1.0, 1.0, 0.1)
    assert result["ape_translation_rmse_m"] < 1e-3
    assert result["rpe_rotation_rmse_deg"] < 0.05

    rng = np.random.default_rng(0)
    sigma = [2e-3, 1e-3, 1.5e-3, 4e-3, 3e-3, 5e-3]
    trajectory, motion = _noisy_run(tmp_path, "run", stamps, poses, sigma, rng)
    motions = check.read_motions(motion)
    result = check.evaluate(check.read_trajectory(trajectory), motions, truth, 1.0, 1.0, 0.1)
    assert result["ape_translation_rmse_m"] > 1e-3
    assert 0.8 < result["nees_mean"] < 1.25, result

    # A candidate claiming 4x smaller standard deviations is inconsistent.
    for item in motions:
        item.covariance_rt = item.covariance_rt / 16
    overconfident = check.nees(motions, truth, 1.0, 0.1)
    assert overconfident["nees_mean"] > 10


def test_cli_pass_and_fail(tmp_path):
    stamps, poses = _ground_truth()
    _write_tum(tmp_path / "gt.tum", stamps, poses)
    rng = np.random.default_rng(1)
    sigma = [1e-3] * 3 + [2e-3] * 3
    ref_t, ref_m = _noisy_run(tmp_path, "ref", stamps, poses, sigma, rng)
    good_t, good_m = _noisy_run(tmp_path, "good", stamps, poses,
                                [s * 0.5 for s in sigma], np.random.default_rng(1),
                                stated_sigma=sigma)
    arguments = ["--ground-truth", str(tmp_path / "gt.tum"), "--reference", str(ref_t),
                 "--reference-motion", str(ref_m)]
    # Half the noise but the same stated covariance: errors drop, NEES drops
    # far beyond +-5 %, so the consistency check fails.
    assert check.main(arguments + ["--candidate", str(good_t),
                                   "--candidate-motion", str(good_m)]) == 1
    assert check.main(arguments + ["--candidate", str(ref_t),
                                   "--candidate-motion", str(ref_m)]) == 0
