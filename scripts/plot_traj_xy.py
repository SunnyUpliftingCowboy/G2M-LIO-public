#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Modified for the public preview on 2026-10-29: portable paths and output handling.
"""Plot aligned XY trajectory comparisons with optional zooms and NEES overlay."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


DEFAULT_TRAJECTORIES = [
    ("gt.txt", "Ground Truth", "black", "-", 1.0),
    ("ours.txt", "Ours", "#c0392b", "-", 1.0),
    ("AKF_LIO_traj.txt", "AKF-LIO", "#2471a3", "--", 0.85),
    ("FAST_LIO_traj.txt", "FAST-LIO", "#229954", "-.", 0.85),
    ("Point_LIO_traj.txt", "Point-LIO", "#7d3c98", ":", 0.85),
]
DEFAULT_ZOOMS = [(40, 65, -30, -5), (80, 110, 10, 35)]


def load_tum(path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(f"{path} must contain: t x y z qx qy qz qw")
    return data[:, 0], data[:, 1:4]


def matching_time_indices(reference, estimate, max_diff):
    right = np.searchsorted(estimate, reference)
    right = np.clip(right, 0, len(estimate) - 1)
    left = np.clip(right - 1, 0, len(estimate) - 1)
    use_left = np.abs(reference - estimate[left]) <= np.abs(reference - estimate[right])
    estimate_indices = np.where(use_left, left, right)
    valid = np.abs(reference - estimate[estimate_indices]) <= max_diff
    return np.flatnonzero(valid), estimate_indices[valid]


def umeyama_se3(from_points, to_points):
    from_mean = np.mean(from_points, axis=0)
    to_mean = np.mean(to_points, axis=0)
    covariance = (from_points - from_mean).T @ (to_points - to_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = to_mean - rotation @ from_mean
    return rotation, translation


def align_to_ground_truth(timestamps, positions, gt_timestamps, gt_positions, max_diff):
    gt_indices, est_indices = matching_time_indices(gt_timestamps, timestamps, max_diff)
    if len(gt_indices) < 10:
        raise RuntimeError(f"only {len(gt_indices)} timestamp pairs found")
    rotation, translation = umeyama_se3(positions[est_indices], gt_positions[gt_indices])
    return (rotation @ positions.T).T + translation


def draw_trajectories(axis, trajectories, show_legend=False, line_width=1.5,
                      xlim=None, ylim=None):
    for trajectory in trajectories:
        xy = trajectory["positions"][:, :2]
        axis.plot(xy[:, 0], xy[:, 1], color=trajectory["color"],
                  linestyle=trajectory["linestyle"], linewidth=line_width,
                  alpha=trajectory["alpha"],
                  label=trajectory["label"] if show_legend else None)
        axis.plot(xy[0, 0], xy[0, 1], "o", color=trajectory["color"],
                  markersize=6, zorder=5)
        axis.plot(xy[-1, 0], xy[-1, 1], "s", color=trajectory["color"],
                  markersize=5, zorder=5)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim is not None:
        axis.set_ylim(ylim)


def true_runs(mask):
    """Return inclusive index ranges for contiguous True samples."""
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return list(zip(starts, ends))


def draw_z_comparison(axis, trajectories, regions, region_colors):
    time_origin = trajectories[0]["timestamps"][0]
    for trajectory in trajectories:
        elapsed = trajectory["timestamps"] - time_origin
        axis.plot(elapsed, trajectory["positions"][:, 2],
                  color=trajectory["color"], linestyle=trajectory["linestyle"],
                  linewidth=1.25, alpha=trajectory["alpha"])

    # Use ground truth to map each XY zoom rectangle to one or more time spans.
    reference = trajectories[0]
    elapsed = reference["timestamps"] - time_origin
    xy = reference["positions"][:, :2]
    zoom_handles = []
    for index, (xmin, xmax, ymin, ymax) in enumerate(regions):
        color = region_colors[index % len(region_colors)]
        inside = ((xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) &
                  (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax))
        for start, end in true_runs(inside):
            if end == start:
                continue
            axis.axvspan(elapsed[start], elapsed[end], color=color,
                        alpha=0.14, linewidth=0, zorder=0)
            axis.axvline(elapsed[start], color=color, alpha=0.45, linewidth=0.7)
            axis.axvline(elapsed[end], color=color, alpha=0.45, linewidth=0.7)
        zoom_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.22,
                                  label=f"Zoom {index + 1}"))

    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
    axis.set_xlabel("Elapsed time [s]")
    axis.set_ylabel("Z [m]")
    axis.set_title("Z trajectory comparison (shading maps the XY zoom regions)")
    axis.grid(True, alpha=0.25)
    if zoom_handles:
        axis.legend(handles=zoom_handles, loc="best", ncol=min(3, len(zoom_handles)))


def add_consistency_overlay(figure, axis, trajectories, metrics_path, target_label,
                            metric_name, max_time_diff):
    metrics = np.genfromtxt(metrics_path, delimiter=",", names=True, dtype=float)
    metrics = np.atleast_1d(metrics)
    required = {"timestamp", metric_name}
    if not required.issubset(metrics.dtype.names or ()):
        raise ValueError(f"{metrics_path} must contain timestamp and {metric_name}")
    target = next((item for item in trajectories if item["label"] == target_label), None)
    if target is None:
        raise ValueError(f"NEES target trajectory label not found: {target_label}")
    trajectory_indices, metric_indices = matching_time_indices(
        target["timestamps"], metrics["timestamp"], max_time_diff)
    metric_dof = {
        "position_nees": 3.0,
        "rotation_nees": 3.0,
        "pose_nees": 6.0,
        "nis_per_dof": 1.0,
    }[metric_name]
    values = metrics[metric_name][metric_indices] / metric_dof
    finite = np.isfinite(values)
    trajectory_indices = trajectory_indices[finite]
    values = values[finite]
    if len(values) == 0:
        raise RuntimeError("no timestamp-associated finite pose NEES values")
    positions = target["positions"][trajectory_indices]
    scatter = axis.scatter(
        positions[:, 0], positions[:, 1], c=np.clip(values, 0.0, 3.0),
        cmap="viridis", s=8, alpha=0.8, zorder=6)
    labels = {
        "position_nees": "Position NEES / 3",
        "rotation_nees": "Rotation NEES / 3",
        "pose_nees": "Pose NEES / 6",
        "nis_per_dof": "NIS / dof",
    }
    figure.colorbar(scatter, ax=axis, label=f"{labels[metric_name]} (clipped at 3)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_zoom", nargs="*", type=float,
                        help="backward-compatible groups of: xmin xmax ymin ymax")
    parser.add_argument("--traj-dir", type=Path, default=Path("."),
                        help="directory containing the trajectory files (default: current directory)")
    parser.add_argument("--trajectory", nargs=2, action="append", metavar=("FILE", "LABEL"),
                        help="trajectory file and legend label; first entry is ground truth")
    parser.add_argument("--zoom", nargs=4, type=float, action="append",
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--no-zoom", action="store_true")
    parser.add_argument("--auto-zoom", action="store_true",
                        help="derive two local zoom regions from the ground-truth path")
    parser.add_argument("--z-comparison", action="store_true",
                        help="add an overall Z-versus-time panel below the main XY plot")
    parser.add_argument("--max-time-diff", type=float, default=0.05)
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--consistency-csv", type=Path,
                        help="output CSV from analyze_consistency.py")
    parser.add_argument("--nees-target", default="Ours",
                        help="trajectory label receiving the consistency color overlay")
    parser.add_argument(
        "--consistency-metric",
        choices=("pose_nees", "position_nees", "rotation_nees", "nis_per_dof"),
        default="pose_nees", help="metric used to color the selected trajectory")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.legacy_zoom and len(args.legacy_zoom) % 4:
        raise ValueError("legacy zoom arguments must be groups of four values")
    if args.no_zoom:
        regions = []
    elif args.auto_zoom:
        regions = None
    elif args.zoom:
        regions = [tuple(region) for region in args.zoom]
    elif args.legacy_zoom:
        regions = [tuple(args.legacy_zoom[index:index + 4])
                   for index in range(0, len(args.legacy_zoom), 4)]
    else:
        regions = DEFAULT_ZOOMS

    if args.trajectory:
        palette = ["black", "#c0392b", "#2471a3", "#229954", "#7d3c98"]
        styles = ["-", "-", "--", "-.", ":"]
        definitions = [
            (filename, label, palette[index % len(palette)],
             styles[index % len(styles)], 1.0 if index < 2 else 0.85)
            for index, (filename, label) in enumerate(args.trajectory)
        ]
    else:
        definitions = DEFAULT_TRAJECTORIES

    gt_path = args.traj_dir / definitions[0][0]
    gt_timestamps, gt_positions = load_tum(gt_path)
    if regions is None:
        xy = gt_positions[:, :2]
        span = np.maximum(np.ptp(xy, axis=0), 1.0)
        half_window = np.maximum(0.12 * span, 1.0)
        regions = []
        for fraction in (1.0 / 3.0, 2.0 / 3.0):
            center = xy[min(int(fraction * (len(xy) - 1)), len(xy) - 1)]
            regions.append((center[0] - half_window[0],
                            center[0] + half_window[0],
                            center[1] - half_window[1],
                            center[1] + half_window[1]))
    trajectories = []
    for index, (filename, label, color, linestyle, alpha) in enumerate(definitions):
        path = args.traj_dir / filename
        if not path.exists():
            print(f"[WARN] {path} not found, skipping")
            continue
        timestamps, positions = load_tum(path)
        if index > 0:
            positions = align_to_ground_truth(
                timestamps, positions, gt_timestamps, gt_positions, args.max_time_diff)
        trajectories.append({
            "label": label, "color": color, "linestyle": linestyle, "alpha": alpha,
            "timestamps": timestamps, "positions": positions,
        })
    if not trajectories:
        raise RuntimeError("no trajectories were loaded")

    zoom_count = len(regions)
    row_count = max(zoom_count, 1)
    zoom_axes = []
    z_axis = None
    if args.z_comparison:
        grid_rows = 2 * row_count
        xy_rows = max(1, int(round(0.68 * grid_rows)))
        if xy_rows >= grid_rows:
            xy_rows = grid_rows - 1
        figure = plt.figure(figsize=(14, max(8.0, 3.0 * row_count)),
                            constrained_layout=True)
        grid = figure.add_gridspec(
            grid_rows, 2, width_ratios=(1.55, 1.0), hspace=0.06, wspace=0.10)
        figure.set_constrained_layout_pads(
            w_pad=0.025, h_pad=0.025, wspace=0.035, hspace=0.025)
        main_axis = figure.add_subplot(grid[:xy_rows, 0])
        z_axis = figure.add_subplot(grid[xy_rows:, 0])
        for index in range(zoom_count):
            start = 2 * index
            zoom_axes.append(figure.add_subplot(grid[start:start + 2, 1]))
    else:
        figure = plt.figure(figsize=(16, 6.0 * row_count))
        main_axis_width = 0.88 if zoom_count == 0 else 0.50
        main_axis = figure.add_axes([0.06, 0.08, main_axis_width, 0.84])
    draw_trajectories(main_axis, trajectories, show_legend=True)
    if args.z_comparison:
        # Equal-aspect XY axes otherwise sit in the center of a wider grid
        # cell. Anchoring both sides closes the apparent main/zoom gap.
        main_axis.set_anchor("E")
    main_axis.set_xlabel("X [m]")
    main_axis.set_ylabel("Y [m]")
    main_axis.set_title(args.title or f"{args.traj_dir.name} - XY trajectory (SE(3) aligned)")
    main_axis.legend(loc="best")

    rectangle_colors = ["#8e44ad", "#2471a3", "#229954", "#d35400"]
    panel_height = (0.84 - (row_count - 1) * 0.08) / row_count
    panel_top = 0.92
    for index, (xmin, xmax, ymin, ymax) in enumerate(regions):
        if args.z_comparison:
            zoom_axis = zoom_axes[index]
        else:
            bottom = panel_top - (index + 1) * panel_height - index * 0.08
            zoom_axis = figure.add_axes([0.65, bottom, 0.31, panel_height])
        draw_trajectories(zoom_axis, trajectories, line_width=1.2,
                          xlim=(xmin, xmax), ylim=(ymin, ymax))
        if args.z_comparison:
            zoom_axis.set_anchor("W")
        zoom_axis.set_title(f"Zoom {index + 1}: X [{xmin:g}, {xmax:g}], Y [{ymin:g}, {ymax:g}]")
        zoom_axis.set_xlabel("X [m]")
        zoom_axis.set_ylabel("Y [m]")
        color = rectangle_colors[index % len(rectangle_colors)]
        main_axis.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2.0,
            edgecolor=color, facecolor="none", zorder=4))
        main_axis.text(xmin, ymax, f"Zoom {index + 1}", color=color,
                       fontweight="bold", verticalalignment="bottom")

    if z_axis is not None:
        draw_z_comparison(z_axis, trajectories, regions, rectangle_colors)

    if args.consistency_csv:
        add_consistency_overlay(
            figure, main_axis, trajectories, args.consistency_csv,
            args.nees_target, args.consistency_metric, args.max_time_diff)

    if z_axis is not None:
        # Constrained layout first resolves the equal-aspect XY size. Match
        # the Z panel to that exact horizontal extent. Move zoom axes slightly
        # left while retaining room for their tick labels, then freeze layout.
        figure.canvas.draw()
        xy_position = main_axis.get_position()
        z_position = z_axis.get_position()
        z_axis.set_position([
            xy_position.x0, z_position.y0, xy_position.width, z_position.height])
        zoom_shift = 0.035
        for zoom_axis in zoom_axes:
            position = zoom_axis.get_position()
            zoom_axis.set_position([
                position.x0 - zoom_shift, position.y0,
                position.width, position.height])
        figure.set_layout_engine(None)

    output = args.output or args.traj_dir / "xy_traj_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    print(f"Saved to {output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
