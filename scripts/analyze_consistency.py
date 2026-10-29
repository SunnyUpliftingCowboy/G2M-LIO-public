#!/usr/bin/env python3
"""Compute and plot NIS/NEES consistency metrics for G2M-LIO."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import chi2


POSE_COV_COLUMNS = [f"P{row}{col}" for row in range(6) for col in range(6)]


def load_consistency(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    data = np.atleast_1d(data)
    required = {
        "timestamp", "x", "y", "z", "qx", "qy", "qz", "qw",
        "nis", "nis_dof", *POSE_COV_COLUMNS,
    }
    missing = sorted(required.difference(data.dtype.names or ()))
    if missing:
        raise ValueError(f"consistency log is missing columns: {', '.join(missing)}")
    if data.size == 0:
        raise ValueError("consistency log is empty")
    return data


def load_ground_truth(path, fmt):
    if fmt == "auto":
        fmt = "ntuviral" if Path(path).suffix.lower() == ".csv" else "tum"
    if fmt == "ntuviral":
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        timestamps = raw[:, 2] / 1e9
        positions = raw[:, 3:6]
        quaternions = raw[:, 6:10]
    else:
        raw = np.loadtxt(path, comments="#", ndmin=2)
        if raw.shape[1] < 8:
            raise ValueError("TUM ground truth must contain: t x y z qx qy qz qw")
        timestamps = raw[:, 0]
        positions = raw[:, 1:4]
        quaternions = raw[:, 4:8]
    order = np.argsort(timestamps)
    return timestamps[order], positions[order], quaternions[order]


def associate_nearest(est_timestamps, gt_timestamps, max_diff):
    right = np.searchsorted(gt_timestamps, est_timestamps)
    right = np.clip(right, 0, len(gt_timestamps) - 1)
    left = np.clip(right - 1, 0, len(gt_timestamps) - 1)
    use_left = (
        np.abs(est_timestamps - gt_timestamps[left])
        <= np.abs(est_timestamps - gt_timestamps[right])
    )
    gt_indices = np.where(use_left, left, right)
    differences = np.abs(est_timestamps - gt_timestamps[gt_indices])
    est_indices = np.flatnonzero(differences <= max_diff)
    return est_indices, gt_indices[est_indices], differences[est_indices]


def interpolate_ground_truth(est_timestamps, gt_timestamps, gt_positions,
                             gt_quaternions, max_gap):
    right = np.searchsorted(gt_timestamps, est_timestamps, side="right")
    valid = (right > 0) & (right < len(gt_timestamps))
    candidate_indices = np.flatnonzero(valid)
    left = right[candidate_indices] - 1
    right_valid = right[candidate_indices]
    gaps = gt_timestamps[right_valid] - gt_timestamps[left]
    finite_gap = np.isfinite(gaps) & (gaps > 0) & (gaps <= max_gap)
    est_indices = candidate_indices[finite_gap]
    left = left[finite_gap]
    right_valid = right_valid[finite_gap]
    gaps = gaps[finite_gap]
    if len(est_indices) == 0:
        return est_indices, np.empty((0, 3)), np.empty((0, 3, 3)), gaps

    query_times = est_timestamps[est_indices]
    alpha = (query_times - gt_timestamps[left]) / gaps
    positions = ((1.0 - alpha[:, None]) * gt_positions[left] +
                 alpha[:, None] * gt_positions[right_valid])
    rotations = Slerp(
        gt_timestamps, Rotation.from_quat(gt_quaternions))(query_times).as_matrix()
    return est_indices, positions, rotations, gaps


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rigid_alignment(from_points, to_points):
    from_mean = np.mean(from_points, axis=0)
    to_mean = np.mean(to_points, axis=0)
    covariance = (from_points - from_mean).T @ (to_points - to_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = to_mean - rotation @ from_mean
    return rotation, translation


def align_estimates(est_positions, est_rotations, gt_positions, gt_rotations,
                    method, alignment_samples):
    if method == "none":
        rotation = np.eye(3)
        translation = np.zeros(3)
    elif method == "first":
        rotation = gt_rotations[0] @ est_rotations[0].T
        translation = gt_positions[0] - rotation @ est_positions[0]
    else:
        count = min(alignment_samples, len(est_positions))
        if count < 3:
            raise ValueError("SE(3) alignment needs at least three associated poses")
        rotation, translation = rigid_alignment(
            est_positions[:count], gt_positions[:count])
    positions = (rotation @ est_positions.T).T + translation
    rotations = rotation[None, :, :] @ est_rotations
    return positions, rotations, rotation, translation


def quadratic_form(errors, covariances):
    values = np.full(len(errors), np.nan)
    for index, (error, covariance) in enumerate(zip(errors, covariances)):
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if eigenvalues[0] < -1e-8 * scale or not np.all(np.isfinite(eigenvalues)):
            continue
        eigenvalues = np.maximum(eigenvalues, 1e-12 * scale)
        projected = eigenvectors.T @ error
        values[index] = np.sum(projected * projected / eigenvalues)
    return values


def confidence_bounds(dof, confidence):
    tail = (1.0 - confidence) / 2.0
    return chi2.ppf(tail, dof), chi2.ppf(1.0 - tail, dof)


def metric_summary(name, values, dof, confidence):
    values = np.asarray(values)
    if np.ndim(dof) == 0:
        dof_values = np.full(values.shape, float(dof))
    else:
        dof_values = np.asarray(dof)
    valid = np.isfinite(values) & np.isfinite(dof_values) & (dof_values > 0)
    dofs = dof_values[valid]
    selected = values[valid]
    if selected.size == 0:
        return f"{name:14s}: no valid samples"
    lower, upper = confidence_bounds(dofs, confidence)
    coverage = np.mean((selected >= lower) & (selected <= upper)) * 100.0
    total_dof = np.sum(dofs)
    ratio = np.sum(selected) / total_dof
    aggregate_lower, aggregate_upper = confidence_bounds(total_dof, confidence)
    return (
        f"{name:14s}: N={selected.size:5d}, normalized mean={ratio:.4f}, "
        f"pointwise coverage={coverage:5.1f}%, aggregate {confidence:.0%} interval="
        f"[{aggregate_lower / total_dof:.4f}, {aggregate_upper / total_dof:.4f}]"
    )


def save_metrics(path, timestamps, est_positions, gt_positions, nis, nis_dof,
                 position_nees, rotation_nees, pose_nees):
    columns = np.column_stack([
        timestamps, est_positions, gt_positions, nis, nis_dof,
        np.divide(nis, nis_dof, out=np.full_like(nis, np.nan), where=nis_dof > 0),
        position_nees, rotation_nees, pose_nees,
    ])
    header = (
        "timestamp,est_x_aligned,est_y_aligned,est_z_aligned,"
        "gt_x,gt_y,gt_z,nis,nis_dof,nis_per_dof,"
        "position_nees,rotation_nees,pose_nees"
    )
    np.savetxt(path, columns, delimiter=",", header=header, comments="", fmt="%.12g")


def plot_metrics(path, timestamps, est_positions, gt_positions, nis, nis_dof,
                 position_nees, rotation_nees, pose_nees, confidence, show):
    elapsed = timestamps - timestamps[0]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(gt_positions[:, 0], gt_positions[:, 1], color="black", linewidth=1.2,
              label="Ground truth")
    axis.plot(est_positions[:, 0], est_positions[:, 1], color="#2471a3", linewidth=1.0,
              label="Estimate")
    color_values = np.clip(pose_nees / 6.0, 0.0, 3.0)
    scatter = axis.scatter(est_positions[:, 0], est_positions[:, 1], c=color_values,
                           cmap="viridis", s=7, alpha=0.75, zorder=3)
    figure.colorbar(scatter, ax=axis, label="Pose NEES / 6 (clipped at 3)")
    axis.set(xlabel="X [m]", ylabel="Y [m]", title="Aligned XY trajectory")
    axis.axis("equal")
    axis.legend()
    axis.grid(True, alpha=0.25)

    axis = axes[0, 1]
    valid_nis = np.isfinite(nis) & (nis_dof > 0)
    normalized_nis = nis[valid_nis] / nis_dof[valid_nis]
    lower, upper = confidence_bounds(nis_dof[valid_nis], confidence)
    axis.plot(elapsed[valid_nis], normalized_nis, linewidth=0.8, color="#2471a3")
    axis.plot(elapsed[valid_nis], lower / nis_dof[valid_nis], "--", linewidth=0.8,
              color="#1e8449", label=f"{confidence:.0%} bounds")
    axis.plot(elapsed[valid_nis], upper / nis_dof[valid_nis], "--", linewidth=0.8,
              color="#c0392b")
    axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set(xlabel="Time [s]", ylabel="NIS / dof", title="Normalized innovation squared")
    axis.grid(True, alpha=0.25)
    axis.legend()

    axis = axes[1, 0]
    lower3, upper3 = confidence_bounds(3, confidence)
    axis.plot(elapsed, position_nees / 3.0, linewidth=0.9, label="Position")
    axis.plot(elapsed, rotation_nees / 3.0, linewidth=0.9, label="Rotation")
    axis.axhspan(lower3 / 3.0, upper3 / 3.0, color="#d5f5e3", alpha=0.6,
                 label=f"{confidence:.0%} bounds")
    axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set(xlabel="Time [s]", ylabel="NEES / dof", title="Position and rotation NEES")
    axis.grid(True, alpha=0.25)
    axis.legend()

    axis = axes[1, 1]
    lower6, upper6 = confidence_bounds(6, confidence)
    axis.plot(elapsed, pose_nees / 6.0, linewidth=0.9, color="#8e44ad")
    axis.axhspan(lower6 / 6.0, upper6 / 6.0, color="#d5f5e3", alpha=0.6)
    axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set(xlabel="Time [s]", ylabel="NEES / dof", title="6-DoF pose NEES")
    axis.grid(True, alpha=0.25)

    figure.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Associate G2M-LIO consistency.csv with ground truth and compute NIS/NEES.")
    parser.add_argument("--log", required=True, type=Path, help="G2M-LIO Log/consistency.csv")
    parser.add_argument("--gt", required=True, type=Path, help="TUM or NTU VIRAL ground truth")
    parser.add_argument("--gt-format", choices=("auto", "tum", "ntuviral"), default="auto")
    parser.add_argument("--max-time-diff", type=float, default=0.05,
                        help="maximum nearest-timestamp difference in seconds")
    parser.add_argument("--association", choices=("interpolate", "nearest"),
                        default="interpolate",
                        help="ground-truth time association method")
    parser.add_argument("--max-interpolation-gap", type=float, default=0.2,
                        help="largest GT bracketing interval accepted in seconds")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="seconds added to estimator timestamps before association")
    parser.add_argument("--start-elapsed-time", type=float, default=0.0,
                        help="only analyze poses at least this many seconds after the first associated pose")
    parser.add_argument("--align", choices=("first", "se3", "none"), default="first",
                        help="map-frame to ground-truth-frame alignment")
    parser.add_argument("--alignment-samples", type=int, default=100,
                        help="initial associated samples used by --align se3")
    parser.add_argument("--lever-arm", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"), help="body-to-GT-sensor translation in body frame")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-prefix", type=Path, default=None,
                        help="output path without extension (default: beside the log)")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")

    log = load_consistency(args.log)
    gt_timestamps, gt_positions_all, gt_quaternions_all = load_ground_truth(
        args.gt, args.gt_format)
    estimator_timestamps = log["timestamp"] + args.time_offset
    if args.association == "nearest":
        est_indices, gt_indices, association_intervals = associate_nearest(
            estimator_timestamps, gt_timestamps, args.max_time_diff)
        gt_positions = gt_positions_all[gt_indices]
        gt_rotations = Rotation.from_quat(
            gt_quaternions_all[gt_indices]).as_matrix()
    else:
        est_indices, gt_positions, gt_rotations, association_intervals = \
            interpolate_ground_truth(
                estimator_timestamps, gt_timestamps, gt_positions_all,
                gt_quaternions_all, args.max_interpolation_gap)
    if len(est_indices) < 3:
        raise RuntimeError(
            f"only {len(est_indices)} poses associated; check timestamp units, offset, and association limits")

    if args.start_elapsed_time < 0.0:
        raise ValueError("--start-elapsed-time must be non-negative")
    if args.start_elapsed_time > 0.0:
        associated_times = estimator_timestamps[est_indices]
        keep = associated_times >= associated_times[0] + args.start_elapsed_time
        est_indices = est_indices[keep]
        if args.association == "nearest":
            gt_positions = gt_positions[keep]
            gt_rotations = gt_rotations[keep]
            association_intervals = association_intervals[keep]
        else:
            gt_positions = gt_positions[keep]
            gt_rotations = gt_rotations[keep]
            association_intervals = association_intervals[keep]
        if len(est_indices) < 3:
            raise RuntimeError(
                f"only {len(est_indices)} poses remain after --start-elapsed-time={args.start_elapsed_time:g}s")

    timestamps = estimator_timestamps[est_indices]
    est_positions = np.column_stack([log[name][est_indices] for name in ("x", "y", "z")])
    est_quaternions = np.column_stack([
        log[name][est_indices] for name in ("qx", "qy", "qz", "qw")])
    est_rotations = Rotation.from_quat(est_quaternions).as_matrix()
    pose_covariances = np.column_stack(
        [log[name][est_indices] for name in POSE_COV_COLUMNS]).reshape(-1, 6, 6)

    lever_arm = np.asarray(args.lever_arm, dtype=float)
    if np.linalg.norm(lever_arm) > 0:
        est_positions = est_positions + np.einsum("nij,j->ni", est_rotations, lever_arm)
        for index, rotation in enumerate(est_rotations):
            jacobian = np.eye(6)
            jacobian[:3, 3:] = -rotation @ skew(lever_arm)
            pose_covariances[index] = jacobian @ pose_covariances[index] @ jacobian.T

    est_positions, est_rotations, alignment_rotation, _ = align_estimates(
        est_positions, est_rotations, gt_positions, gt_rotations,
        args.align, args.alignment_samples)
    covariance_alignment = np.eye(6)
    covariance_alignment[:3, :3] = alignment_rotation
    # Both IKFoM and StatesGroup use right-multiplicative SO(3) errors.
    # A left map-frame alignment rotates world-frame position errors, while
    # the local/body-frame rotation perturbation remains unchanged.
    pose_covariances = np.einsum(
        "ij,njk,lk->nil", covariance_alignment, pose_covariances, covariance_alignment)

    position_errors = gt_positions - est_positions
    rotation_errors = Rotation.from_matrix(
        np.transpose(est_rotations, (0, 2, 1)) @ gt_rotations).as_rotvec()
    pose_errors = np.column_stack([position_errors, rotation_errors])
    position_nees = quadratic_form(position_errors, pose_covariances[:, :3, :3])
    rotation_nees = quadratic_form(rotation_errors, pose_covariances[:, 3:, 3:])
    pose_nees = quadratic_form(pose_errors, pose_covariances)
    nis = log["nis"][est_indices]
    nis_dof = log["nis_dof"][est_indices]

    output_prefix = args.output_prefix or args.log.with_name("consistency_analysis").with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    plot_path = output_prefix.with_suffix(".png")
    save_metrics(csv_path, timestamps, est_positions, gt_positions, nis, nis_dof,
                 position_nees, rotation_nees, pose_nees)
    plot_metrics(plot_path, timestamps, est_positions, gt_positions, nis, nis_dof,
                 position_nees, rotation_nees, pose_nees, args.confidence, args.show)

    interval_label = "nearest dt" if args.association == "nearest" else "GT gap"
    print(f"Associated poses: {len(est_indices)} ({args.association}, max {interval_label}="
          f"{np.max(association_intervals):.6f} s)")
    print(metric_summary("NIS", nis, nis_dof, args.confidence))
    print(metric_summary("Position NEES", position_nees, 3, args.confidence))
    print(metric_summary("Rotation NEES", rotation_nees, 3, args.confidence))
    print(metric_summary("Pose NEES", pose_nees, 6, args.confidence))
    print(f"Saved metrics: {csv_path}")
    print(f"Saved plot:    {plot_path}")
    if args.start_elapsed_time > 0.0:
        print(f"Analysis start: elapsed time >= {args.start_elapsed_time:.3f} s")
    if args.align == "se3":
        print("Note: post-hoc SE(3) alignment makes NEES diagnostic rather than a strict consistency test.")


if __name__ == "__main__":
    main()
