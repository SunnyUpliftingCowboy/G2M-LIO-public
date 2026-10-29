# Analysis utilities

Run commands from the repository root. Python 3.8.10, NumPy 1.24.4, SciPy 1.10.1, and Matplotlib 3.7.5 are the local validation environment; requirements-analysis.txt lists the runtime dependencies.

## Included tools

| Script | Input | Output |
| --- | --- | --- |
| plot_traj_xy.py | Timestamped TUM-format reference and estimate files | SE(3)-aligned XY trajectory figure, optional height and zoom panels |
| analyze_consistency.py | G2M-LIO consistency log and matching reference poses | NIS/NEES diagnostics CSV and figure |
| plot_nees_trajectories.py | Metrics CSV from analyze_consistency.py | Trajectories colored by normalized pose NEES |

## Trajectory comparison

Each trajectory row must contain t x y z qx qy qz qw, with timestamps in seconds and positions in meters. The first --trajectory argument is the reference. Provide increasing timestamps, compatible frames and overlapping time intervals.

```bash
python3 scripts/plot_traj_xy.py \
  --traj-dir /path/to/trajectories \
  --trajectory gt.txt 'Ground truth' \
  --trajectory ours.txt 'G2M-LIO' \
  --auto-zoom --z-comparison \
  --output results/trajectory_comparison.png
```

Use --no-zoom for a single XY panel. Without --trajectory, the script searches for its documented default filenames; without --traj-dir, it searches the current directory. SE(3) alignment uses translation and rotation, without a scale fit.

## Consistency analysis

The log must have a CSV header with timestamp, x, y, z, qx, qy, qz, qw, nis, nis_dof and all 36 covariance fields P00 through P55 in row-major order. The expected pose covariance state order is position then local/body-frame rotation error.

```bash
python3 scripts/analyze_consistency.py \
  --log /path/to/consistency.csv \
  --gt /path/to/ground_truth.txt --gt-format tum \
  --align first --output-prefix results/consistency

python3 scripts/plot_nees_trajectories.py \
  --analysis results/consistency.csv 'G2M-LIO' \
  --color-scale absolute --output results/trajectory_nees.png
```

For NTU VIRAL CSV input, --gt-format ntuviral skips one header row and reads column 3 as nanosecond timestamps, columns 4-6 as positions and columns 7-10 as xyzw quaternions (column numbers are one-based). Verify that your reference file uses this layout.

Default association interpolates reference poses with a maximum bracketing interval of 0.2 seconds. For nearest association use --association nearest --max-time-diff VALUE. Account for sensor frame, lever arm, clock offsets, covariance convention, and units before interpreting NEES. Post-hoc alignment and temporal correlation limit strict statistical interpretation.

Dataset logs, ground truth and complete odometry code are not bundled in this preview.
