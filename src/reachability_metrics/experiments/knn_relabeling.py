"""kNN relabeling benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from reachability_metrics.data import load_dataset_or_synthetic
from reachability_metrics.evaluation import ndcg_at_k, safe_spearman
from reachability_metrics.evaluation.relabeling import diversity_at_k, goal_precision_at_k, mean_gt_score_at_k, unique_goal_ratio_at_k
from reachability_metrics.evaluation.reports import save_csv
from reachability_metrics.state_metrics import (
    AdaptiveGaussianDistance,
    EuclideanDistance,
    GaussianKernelDistance,
    IsolationKernelDistance,
    MahalanobisDistance,
    OneStepDynamicsDistance,
)
from reachability_metrics.utils import dataset_slug, ensure_dir
from reachability_metrics.visualization.maze import plot_topk_goals_on_maze
from reachability_metrics.visualization.plots import plot_relabel_bars


DEFAULT_DATASETS = [
    "D4RL/pointmaze/umaze-v2",
    "D4RL/pointmaze/large-v2",
    "D4RL/antmaze/umaze-diverse-v1",
]


@dataclass
class KNNRelabelConfig:
    datasets: list[str]
    output_dir: str
    cache_dir: str | None = None
    seed: int = 0
    minari_datasets_path: str | None = None
    horizon: int = 20
    num_anchors: int = 200
    num_candidates: int = 1000
    top_k: int = 20
    fit_pool_size: int = 50000
    ik_ensemble_size: int = 100
    ik_subsample_size: int = 32
    ik_temperature: float = 0.01
    ik_batch_size: int = 4096
    ik_device: str = "auto"
    min_time_gap: int = 5
    min_goal_dist: float = 0.0
    overwrite_cache: bool = False

    @property
    def tables_dir(self) -> str:
        return f"{self.output_dir}/tables"

    @property
    def figures_dir(self) -> str:
        return f"{self.output_dir}/figures"


def _reachability_gt(states: np.ndarray, episode_ids: np.ndarray, timesteps: np.ndarray, anchors: np.ndarray, candidates: np.ndarray, horizon: int) -> np.ndarray:
    gt = np.zeros((anchors.shape[0], candidates.shape[0]), dtype=np.float32)
    candidate_pos = states[candidates]
    for i, aidx in enumerate(anchors):
        ep = episode_ids[aidx]
        ts = timesteps[aidx]
        mask = (episode_ids == ep) & (timesteps > ts) & (timesteps <= ts + int(horizon))
        future = states[mask]
        if future.size == 0:
            continue
        d = cdist(candidate_pos, future)
        gt[i] = (np.min(d, axis=1) < 0.25).astype(np.float32)
        gt[i] += 1.0 / (1.0 + np.linalg.norm(states[aidx][None, :] - candidate_pos, axis=1))
        gt[i] *= 0.5
    return gt


def _temporal_scores(episode_ids: np.ndarray, timesteps: np.ndarray, anchors: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    same = episode_ids[anchors][:, None] == episode_ids[candidates][None, :]
    delta = timesteps[candidates][None, :] - timesteps[anchors][:, None]
    valid = same & (delta > 0)
    scores = np.zeros((anchors.shape[0], candidates.shape[0]), dtype=np.float32)
    scores[valid] = 1.0 / (1.0 + delta[valid].astype(np.float32))
    return scores


def _method_scores(method: str, fit: np.ndarray, states: np.ndarray, anchors: np.ndarray, candidates: np.ndarray, cfg: KNNRelabelConfig) -> np.ndarray:
    x = states[anchors]
    y = states[candidates]
    if method == "euclidean":
        return -EuclideanDistance().pairwise_distance(x, y)
    if method == "gaussian":
        return GaussianKernelDistance().fit(fit).pairwise_similarity(x, y)
    if method == "adaptive_gaussian":
        return AdaptiveGaussianDistance().fit(fit).pairwise_similarity(x, y)
    if method == "mahalanobis":
        return -MahalanobisDistance().fit(fit).pairwise_distance(x, y)
    if method == "ik":
        return IsolationKernelDistance(
            cfg.ik_ensemble_size,
            cfg.ik_subsample_size,
            cfg.ik_temperature,
            device=cfg.ik_device,
            batch_size=cfg.ik_batch_size,
            random_state=cfg.seed,
        ).fit(fit).pairwise_similarity(x, y)
    raise ValueError(method)


def run_relabel_benchmark(cfg: KNNRelabelConfig, dataset_id: str | None = None) -> dict[str, Any]:
    """Run relabeling for one or more datasets."""
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir or f"{cfg.output_dir}/cache")
    ensure_dir(cfg.tables_dir)
    ensure_dir(cfg.figures_dir)
    datasets = [dataset_id] if dataset_id is not None else cfg.datasets
    rng = np.random.default_rng(cfg.seed)
    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    for ds in datasets:
        dataset = load_dataset_or_synthetic(ds, minari_datasets_path=cfg.minari_datasets_path, use_achieved_goal=True, synthetic_seed=cfg.seed)
        states = dataset.states()
        episode_ids = dataset.episode_ids()
        timesteps = dataset.timesteps()
        valid = np.flatnonzero(timesteps < dataset.episode_lengths()[episode_ids] - cfg.horizon - 1)
        if valid.size == 0:
            valid = np.arange(states.shape[0])
        anchors = rng.choice(valid, size=min(cfg.num_anchors, valid.size), replace=False)
        candidates = rng.choice(states.shape[0], size=min(cfg.num_candidates, states.shape[0]), replace=False)
        fit = states[rng.choice(states.shape[0], size=min(cfg.fit_pool_size, states.shape[0]), replace=False)]
        gt = _reachability_gt(states, episode_ids, timesteps, anchors, candidates, cfg.horizon)
        geo = np.linalg.norm(states[anchors][:, None, :2] - states[candidates][None, :, :2], axis=-1)
        methods = ["euclidean", "gaussian", "mahalanobis", "adaptive_gaussian", "ik", "temporal_distance", "one_step_dynamics"]
        transition_states, transition_next = dataset.transition_pairs()
        for method in methods:
            if method == "temporal_distance":
                scores = _temporal_scores(episode_ids, timesteps, anchors, candidates)
            elif method == "one_step_dynamics" and transition_states.shape[0] > 0:
                scores = -OneStepDynamicsDistance().fit(transition_states, transition_next).pairwise_distance(states[anchors], states[candidates])
            else:
                scores = _method_scores(method, fit, states, anchors, candidates, cfg)
            if cfg.min_goal_dist > 0:
                scores = scores.copy()
                scores[geo < float(cfg.min_goal_dist)] = -np.inf
            rows = []
            for i in range(anchors.shape[0]):
                labels = (gt[i] > np.percentile(gt[i], 75.0)).astype(np.int64)
                row = {
                    "dataset": ds,
                    "method": method,
                    "anchor_row": i,
                    "spearman": safe_spearman(scores[i], gt[i]),
                    "ndcg_at_k": ndcg_at_k(gt[i], scores[i], cfg.top_k),
                    "goal_precision_at_k": goal_precision_at_k(labels, scores[i], cfg.top_k),
                    "mean_gt_reachability": mean_gt_score_at_k(gt[i], scores[i], cfg.top_k),
                    "mean_geodesic": float(np.mean(geo[i][np.argsort(-scores[i])[: cfg.top_k]])),
                    "diversity": diversity_at_k(states[candidates, :2], scores[i], cfg.top_k),
                    "unique_goal_ratio": unique_goal_ratio_at_k(states[candidates, :2], scores[i], cfg.top_k),
                }
                rows.append(row)
                per_anchor_rows.append(row)
            summary_rows.append({
                "dataset": ds,
                "dataset_slug": dataset_slug(ds),
                "method": method,
                "spearman": float(np.mean([r["spearman"] for r in rows])),
                "ndcg_at_k": float(np.mean([r["ndcg_at_k"] for r in rows])),
                "goal_precision_at_k": float(np.mean([r["goal_precision_at_k"] for r in rows])),
                "mean_gt_reachability": float(np.mean([r["mean_gt_reachability"] for r in rows])),
                "diversity": float(np.mean([r["diversity"] for r in rows])),
            })
        plot_topk_goals_on_maze(states[anchors[0], :2], states[candidates, :2], scores[0], f"{cfg.figures_dir}/{dataset_slug(ds)}_ik_topk_seed{cfg.seed}.png", cfg.top_k)
    summary_path = f"{cfg.tables_dir}/summary.csv"
    per_anchor_path = f"{cfg.tables_dir}/per_anchor.csv"
    save_csv(summary_path, summary_rows)
    save_csv(per_anchor_path, per_anchor_rows)
    fig_path = plot_relabel_bars(summary_rows, f"{cfg.figures_dir}/relabel_bars_seed{cfg.seed}.png")
    report_path = f"{cfg.output_dir}/report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# kNN Relabeling Report\n\n")
        handle.write(f"- summary: `{summary_path}`\n")
        handle.write(f"- figure: `{fig_path}`\n")
    return {"summary_rows": summary_rows, "per_anchor_rows": per_anchor_rows, "summary_path": summary_path, "per_anchor_path": per_anchor_path, "report_path": report_path}

