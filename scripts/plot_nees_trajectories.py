#!/usr/bin/env python3
"""Plot aligned trajectories colored by normalized pose NEES with local zooms."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import chi2


def load_analysis(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    data = np.atleast_1d(data)
    required = {
        "est_x_aligned", "est_y_aligned", "gt_x", "gt_y", "pose_nees",
    }
    missing = sorted(required.difference(data.dtype.names or ()))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return data


def colored_segments(xy, values, norm, cmap, linewidth=2.0):
    points = xy[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_values = 0.5 * (values[:-1] + values[1:])
    finite = np.isfinite(segment_values) & np.all(np.isfinite(segments), axis=(1, 2))
    collection = LineCollection(
        segments[finite], array=segment_values[finite], cmap=cmap, norm=norm,
        linewidth=linewidth, capstyle="round", zorder=3,
    )
    return collection


def draw_panel(axis, data, norm, cmap, xlim=None, ylim=None, main=False):
    gt = np.column_stack([data["gt_x"], data["gt_y"]])
    estimate = np.column_stack([data["est_x_aligned"], data["est_y_aligned"]])
    normalized_nees = data["pose_nees"] / 6.0
    axis.plot(gt[:, 0], gt[:, 1], color="black", linewidth=1.4,
              alpha=0.85, zorder=2)
    axis.add_collection(colored_segments(
        estimate, normalized_nees, norm, cmap, 2.2 if main else 1.8))
    axis.plot(estimate[0, 0], estimate[0, 1], marker="o", color="#34495e",
              markersize=5, zorder=4)
    axis.plot(estimate[-1, 0], estimate[-1, 1], marker="s", color="#34495e",
              markersize=5, zorder=4)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim is not None:
        axis.set_ylim(ylim)


def automatic_zoom_regions(data):
    xy = np.column_stack([data["gt_x"], data["gt_y"]])
    span = np.maximum(np.ptp(xy, axis=0), 1.0)
    half_window = np.maximum(0.12 * span, 1.0)
    regions = []
    for fraction in (1.0 / 3.0, 2.0 / 3.0):
        center = xy[min(int(fraction * (len(xy) - 1)), len(xy) - 1)]
        regions.append((center[0] - half_window[0], center[0] + half_window[0],
                        center[1] - half_window[1], center[1] + half_window[1]))
    return regions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", nargs=2, action="append", required=True,
                        metavar=("CSV", "LABEL"), help="analysis CSV and algorithm label")
    parser.add_argument("--zoom", nargs=4, type=float, action="append",
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Trajectory consistency from normalized pose NEES")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--color-scale", choices=("observed", "absolute"),
                        default="observed",
                        help="observed expands differences in this experiment; absolute keeps NEES=1 visible")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    series = [(load_analysis(Path(path)), label) for path, label in args.analysis]
    regions = ([tuple(region) for region in args.zoom]
               if args.zoom else automatic_zoom_regions(series[0][0]))
    if not regions:
        raise ValueError("at least one zoom region is required")

    all_values = np.concatenate([data["pose_nees"] / 6.0 for data, _ in series])
    finite_positive = all_values[np.isfinite(all_values) & (all_values > 0.0)]
    if finite_positive.size == 0:
        raise RuntimeError("no finite positive pose NEES values")
    lower95 = chi2.ppf(0.025, 6) / 6.0
    upper95 = chi2.ppf(0.975, 6) / 6.0
    if args.color_scale == "observed":
        # A robust shared scale expands differences between algorithms and
        # locations without letting a few extreme samples dominate the map.
        color_min = max(float(np.quantile(finite_positive, 0.005)), 1e-12)
        color_max = float(np.quantile(finite_positive, 0.995))
        color_max = max(color_max, color_min * 1.01)
    else:
        color_min = 0.1
        color_max = max(10.0, float(np.quantile(finite_positive, 0.995)))
    norm = LogNorm(vmin=color_min, vmax=color_max, clip=True)
    cmap = plt.get_cmap("RdYlGn_r")

    figure, axes = plt.subplots(
        len(series), len(regions) + 1,
        figsize=(6.0 + 4.0 * len(regions), 4.7 * len(series)),
        gridspec_kw={"width_ratios": (1.55, *([1.0] * len(regions)))},
        constrained_layout=True, squeeze=False,
    )
    rectangle_colors = ("#8e44ad", "#2471a3", "#229954", "#d35400")
    for row, (data, label) in enumerate(series):
        normalized_nees = data["pose_nees"] / 6.0
        draw_panel(axes[row, 0], data, norm, cmap, main=True)
        axes[row, 0].set_title(
            f"{label}: full trajectory (mean NEES/6 = {np.nanmean(normalized_nees):.1f})")
        axes[row, 0].set_ylabel("Y [m]")
        for index, (xmin, xmax, ymin, ymax) in enumerate(regions):
            color = rectangle_colors[index % len(rectangle_colors)]
            axes[row, 0].add_patch(Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1.8,
                edgecolor=color, facecolor="none", zorder=5))
            axes[row, 0].text(xmin, ymax, f"Zoom {index + 1}", color=color,
                              fontweight="bold", va="bottom")
            draw_panel(axes[row, index + 1], data, norm, cmap,
                       xlim=(xmin, xmax), ylim=(ymin, ymax))
            axes[row, index + 1].set_title(f"Zoom {index + 1}")
        for axis in axes[row]:
            axis.set_xlabel("X [m]")

    legend = [
        Line2D([0], [0], color="black", linewidth=1.5, label="Ground truth"),
        Line2D([0], [0], color="#7f8c8d", linewidth=2.2,
               label="Estimate (colored by pose NEES / 6)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#34495e",
               label="Start", markersize=6),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#34495e",
               label="End", markersize=6),
    ]
    figure.legend(handles=legend, loc="outside lower center", ncol=4)
    title = args.title
    if np.min(finite_positive) > upper95:
        title += (f"\nAll samples exceed the pointwise 95% upper bound ({upper95:.2f}); "
                  "green means relatively lower NEES, not consistency.")
    figure.suptitle(title, fontsize=15)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=axes, location="right", shrink=0.92, pad=0.015)
    if args.color_scale == "observed":
        colorbar.set_label(
            "Observed pose NEES / 6 (shared logarithmic scale; green = relatively lower)")
    else:
        colorbar.set_label(
            "Normalized pose NEES (NEES / 6, logarithmic; green is more consistent)")
        for value in (lower95, 1.0, upper95):
            colorbar.ax.axhline(value, color="black", linewidth=0.9, linestyle="--")
        colorbar.ax.text(1.04, upper95, f" 95% upper={upper95:.2f}",
                         transform=colorbar.ax.get_yaxis_transform(), va="center", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    print(f"Saved to {args.output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
