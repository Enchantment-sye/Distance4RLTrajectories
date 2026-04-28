# Migration Notes

## Source Mapping

| Original file | New module |
| --- | --- |
| `src/core/isolation_kernel.py` | `reachability_metrics/state_metrics/isolation_kernel.py` |
| `src/core/kme_module.py` | `reachability_metrics/trajectory_metrics/kme.py` |
| `src/analysis/similarity_metrics.py` | `reachability_metrics/state_metrics/*`, `reachability_metrics/evaluation/*` |
| `src/analysis/fitted_baselines.py` | `reachability_metrics/state_metrics/mahalanobis.py`, `adaptive_gaussian.py`, `one_step_dynamics.py` |
| `src/analysis/successor_distance.py` | `reachability_metrics/experiments/successor_distance.py` |
| `src/analysis/knn_planning.py` | `reachability_metrics/experiments/knn_planning.py`, `reachability_metrics/evaluation/planning.py` |
| `src/analysis/knn_relabeling.py` | `reachability_metrics/experiments/knn_relabeling.py`, `reachability_metrics/evaluation/relabeling.py` |
| `src/analysis/reachability_alignment.py` | `reachability_metrics/experiments/reachability_alignment.py` |
| `src/analysis/proxy_ground_truth.py` | `reachability_metrics/experiments/proxy_ground_truth.py` |
| `src/analysis/maze_geodesic.py` | `reachability_metrics/evaluation/planning.py`, `reachability_metrics/visualization/maze.py` |
| `outputs/paper_toy_pointmaze_summary_v1/*` | `reachability_metrics/experiments/paper_reproduction.py` Figure 2 manifest/copy |
| `outputs/knn_relabel_ik_sweep_v1_protocol/*` | `reachability_metrics/experiments/paper_reproduction.py` Table 2 |
| `outputs/knn_planning_paper_summary_20260412/*` | `reachability_metrics/experiments/paper_reproduction.py` Tables 3-5 |
| `outputs/successor_distance_paper_summary_20260412/*` | `reachability_metrics/experiments/paper_reproduction.py` Table 6 |

## Deliberately Not Migrated

- METRA/DADS/SAC/DrQ training algorithms.
- IsaacLab/Galaxea/ROS integration.
- Environment wrappers and video tooling.
- Hard-coded local dataset roots.

## Validation Environment

The requested validation environment is:

```bash
conda run -n metra_idk python -V
conda run -n metra_idk pip install -e ".[dev,torch,t2vec]"
conda run -n metra_idk pytest -q
```

## Paper Reproduction Notes

Run:

```bash
python -m reachability_metrics.cli.reproduce_paper \
  --legacy_outputs_dir /share/shangyy/codes/metra/outputs \
  --output_dir outputs/paper_reproduction \
  --include_figures \
  --verify-paper-values
```

The command emits `paper_source_manifest.json`, `paper_hyperparameters.csv`,
Figure 2 copies, Tables 1-6 CSV files, and `report.md`. Tables 2-6 are backed by
complete legacy CSVs. Table 1 did not have a complete final CSV in the legacy
outputs; the new command reconstructs the traceable cells and records
version-conflict or unresolved cells in `table1_reconstruction_notes.md`.
