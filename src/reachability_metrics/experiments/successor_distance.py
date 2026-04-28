"""H-step successor distribution matching experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reachability_metrics.data import TrajectoryDataset, load_dataset_or_synthetic
from reachability_metrics.data.windows import future_windows
from reachability_metrics.evaluation import auc_from_binary_labels, average_precision_from_binary_labels, recall_at_k
from reachability_metrics.evaluation.reports import save_csv
from reachability_metrics.state_metrics import AdaptiveGaussianDistance, IsolationKernelDistance
from reachability_metrics.trajectory_metrics import (
    AdaptiveGDKTrajectoryDistance,
    GDKTrajectoryDistance,
    IDKTrajectoryDistance,
    TrajectoryWassersteinDistance,
)
from reachability_metrics.utils import dataset_slug, ensure_dir
from reachability_metrics.visualization.plots import plot_successor_auroc


DEFAULT_DATASETS = [
    "D4RL/pointmaze/umaze-v2",
    "D4RL/pointmaze/large-v2",
    "D4RL/antmaze/umaze-diverse-v1",
]
DEFAULT_HORIZONS = [10, 20, 50]


@dataclass
class SuccessorDistanceConfig:
    datasets: list[str]
    output_dir: str
    cache_dir: str | None = None
    seed: int = 0
    horizon_values: list[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    grid_nx: int = 20
    grid_ny: int = 20
    search_num_pairs: int = 20000
    eval_num_pairs: int = 50000
    num_queries: int = 128
    num_candidates: int = 256
    recall_k_values: tuple[int, ...] = (5, 10, 20)
    raw_gamma: float | None = None
    adaptive_gaussian_k: int = 10
    adaptive_gaussian_eps: float = 1e-6
    fit_pool_size: int = 50000
    ik_ensemble_sizes: tuple[int, ...] = (100,)
    ik_subsample_sizes: tuple[int, ...] = (32,)
    ik_temperatures: tuple[float, ...] = (0.01,)
    ik_batch_size: int = 4096
    ik_device: str = "auto"
    selection_metric: str = "auprc"
    minari_datasets_path: str | None = None
    overwrite_cache: bool = False

    @property
    def tables_dir(self) -> str:
        return f"{self.output_dir}/tables"

    @property
    def figures_dir(self) -> str:
        return f"{self.output_dir}/figures"


def _grid_labels(endpoints: np.ndarray, nx: int, ny: int) -> np.ndarray:
    mins = endpoints.min(axis=0)
    maxs = endpoints.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (endpoints - mins) / span
    xb = np.clip((norm[:, 0] * nx).astype(int), 0, nx - 1)
    yb = np.clip((norm[:, 1] * ny).astype(int), 0, ny - 1)
    return xb * ny + yb


def _raw_window_distance(a: np.ndarray, b: np.ndarray, gamma: float | None = None) -> np.ndarray:
    h = a.shape[1]
    if gamma is None:
        weights = np.full(h, 1.0 / h, dtype=np.float32)
    else:
        weights = np.power(float(gamma), np.arange(h, dtype=np.float32))
        weights = weights / weights.sum()
    sq = np.sum((a - b) ** 2, axis=2)
    return np.sqrt(np.sum(sq * weights[None, :], axis=1)).astype(np.float32)


def _fit_method(method: str, windows: np.ndarray, cfg: SuccessorDistanceConfig) -> Any:
    if method == "raw":
        return None
    rng = np.random.default_rng(cfg.seed)
    sample_count = min(windows.shape[0], max(1, int(cfg.fit_pool_size) // max(1, windows.shape[1])))
    sample_idx = rng.choice(windows.shape[0], size=sample_count, replace=False)
    sample = [x for x in windows[sample_idx]]
    if method == "gdk":
        return GDKTrajectoryDistance().fit(sample)
    if method == "adaptive_gdk":
        return AdaptiveGDKTrajectoryDistance(k=cfg.adaptive_gaussian_k, eps=cfg.adaptive_gaussian_eps).fit(sample)
    if method == "wasserstein_w2":
        return TrajectoryWassersteinDistance().fit(sample)
    if method == "idk":
        return IDKTrajectoryDistance(
            ensemble_size=int(cfg.ik_ensemble_sizes[0]),
            subsample_size=int(cfg.ik_subsample_sizes[0]),
            temperature=float(cfg.ik_temperatures[0]),
            device=cfg.ik_device,
            batch_size=cfg.ik_batch_size,
            random_state=cfg.seed,
        ).fit(sample)
    raise ValueError(method)


def _metric_distance(method: str, metric: Any, a: np.ndarray, b: np.ndarray, cfg: SuccessorDistanceConfig) -> np.ndarray:
    traj_a = [x for x in a]
    traj_b = [x for x in b]
    if method == "raw":
        return _raw_window_distance(a, b, gamma=cfg.raw_gamma)
    if method == "idk" and hasattr(metric, "transform"):
        emb_a = metric.transform(traj_a)
        emb_b = metric.transform(traj_b)
        return np.linalg.norm(emb_a - emb_b, axis=1).astype(np.float32)
    out = np.empty(len(traj_a), dtype=np.float32)
    for i, (ta, tb) in enumerate(zip(traj_a, traj_b)):
        out[i] = float(metric.pairwise_distance([ta], [tb])[0, 0])
    return out


def _load_dataset(dataset_id: str, cfg: SuccessorDistanceConfig) -> TrajectoryDataset:
    return load_dataset_or_synthetic(
        dataset_id,
        minari_datasets_path=cfg.minari_datasets_path,
        use_achieved_goal=True,
        synthetic_seed=cfg.seed,
    )


def run_successor_distance(cfg: SuccessorDistanceConfig) -> dict[str, Any]:
    """Run successor-distance evaluation and write csv/figures/report."""
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir or f"{cfg.output_dir}/cache")
    ensure_dir(cfg.tables_dir)
    ensure_dir(cfg.figures_dir)
    rng = np.random.default_rng(cfg.seed)
    summary_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    methods = ["raw", "idk", "gdk", "wasserstein_w2", "adaptive_gdk"]
    for dataset_id in cfg.datasets:
        dataset = _load_dataset(dataset_id, cfg)
        states = dataset.states()
        for horizon in cfg.horizon_values:
            windows, _, _ = future_windows(dataset, horizon)
            if windows.shape[0] < 2:
                continue
            labels_region = _grid_labels(windows[:, -1, :2], cfg.grid_nx, cfg.grid_ny)
            pair_count = min(int(cfg.eval_num_pairs), max(2, windows.shape[0] * 4))
            first = rng.integers(0, windows.shape[0], size=pair_count)
            second = rng.integers(0, windows.shape[0], size=pair_count)
            labels = (labels_region[first] == labels_region[second]).astype(np.int64)
            query_ids = rng.choice(windows.shape[0], size=min(cfg.num_queries, windows.shape[0]), replace=False)
            cand_ids = rng.choice(windows.shape[0], size=min(cfg.num_candidates, windows.shape[0]), replace=False)
            for method in methods:
                metric = _fit_method(method, windows, cfg)
                distances = _metric_distance(method, metric, windows[first], windows[second], cfg)
                scores = -distances
                row: dict[str, Any] = {
                    "dataset": dataset_id,
                    "dataset_slug": dataset_slug(dataset_id),
                    "horizon": int(horizon),
                    "method": method,
                    "num_pairs": int(pair_count),
                    "positive_fraction": float(np.mean(labels)),
                    "auroc": auc_from_binary_labels(labels, scores),
                    "auprc": average_precision_from_binary_labels(labels, scores),
                }
                matrix = np.zeros((len(query_ids), len(cand_ids)), dtype=np.float32)
                for qi, qid in enumerate(query_ids):
                    qwin = np.repeat(windows[qid : qid + 1], len(cand_ids), axis=0)
                    matrix[qi] = _metric_distance(method, metric, qwin, windows[cand_ids], cfg)
                    qlabels = (labels_region[cand_ids] == labels_region[qid]).astype(np.float32)
                    for k in cfg.recall_k_values:
                        rec = recall_at_k(qlabels, -matrix[qi], int(k))
                        recall_rows.append({"dataset": dataset_id, "horizon": horizon, "method": method, "query": qi, f"recall_at_{k}": rec})
                for k in cfg.recall_k_values:
                    vals = [r[f"recall_at_{k}"] for r in recall_rows if r["dataset"] == dataset_id and r["horizon"] == horizon and r["method"] == method and f"recall_at_{k}" in r]
                    row[f"recall_at_{k}"] = float(np.mean(vals)) if vals else 0.0
                summary_rows.append(row)
    overall_rows = []
    for method in sorted({r["method"] for r in summary_rows}):
        group = [r for r in summary_rows if r["method"] == method]
        overall_rows.append({
            "method": method,
            "auroc": float(np.mean([r["auroc"] for r in group])),
            "auprc": float(np.mean([r["auprc"] for r in group])),
        })
    per_dataset_path = f"{cfg.tables_dir}/per_dataset_metrics.csv"
    overall_path = f"{cfg.tables_dir}/overall_summary.csv"
    recall_path = f"{cfg.tables_dir}/recall_rows.csv"
    save_csv(per_dataset_path, summary_rows)
    save_csv(overall_path, overall_rows)
    save_csv(recall_path, recall_rows)
    fig_path = plot_successor_auroc(summary_rows, f"{cfg.figures_dir}/successor_auroc_seed{cfg.seed}.png")
    report_path = f"{cfg.output_dir}/report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Successor Distance Report\n\n")
        handle.write(f"- datasets: {', '.join(cfg.datasets)}\n")
        handle.write(f"- horizons: {cfg.horizon_values}\n")
        handle.write(f"- summary table: `{per_dataset_path}`\n")
        handle.write(f"- figure: `{fig_path}`\n")
    return {
        "summary_rows": summary_rows,
        "overall_rows": overall_rows,
        "per_dataset_path": per_dataset_path,
        "overall_path": overall_path,
        "recall_path": recall_path,
        "report_path": report_path,
        "figure_path": fig_path,
        "search_full_path": per_dataset_path,
        "search_best_path": overall_path,
    }
