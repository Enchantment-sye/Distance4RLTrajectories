"""kNN relabeling benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from reachability_metrics.evaluation import ndcg_at_k, safe_spearman
from reachability_metrics.evaluation.relabeling import (
    diversity_at_k,
    goal_precision_at_k,
    mean_gt_score_at_k,
    unique_goal_ratio_at_k,
)
from reachability_metrics.experiments._sampling import (
    load_state_dataset_sample,
    state_scoring_context,
)
from reachability_metrics.experiments.artifacts import ArtifactWriter
from reachability_metrics.experiments.scorers import build_experiment_scorer
from reachability_metrics.utils import dataset_slug
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


def _reachability_gt(
    states: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    anchors: np.ndarray,
    candidates: np.ndarray,
    horizon: int,
) -> np.ndarray:
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


def run_relabel_benchmark(cfg: KNNRelabelConfig, dataset_id: str | None = None) -> dict[str, Any]:
    """Run relabeling for one or more datasets."""
    artifacts = ArtifactWriter(cfg.output_dir, cfg.cache_dir).prepare()
    datasets = [dataset_id] if dataset_id is not None else cfg.datasets
    rng = np.random.default_rng(cfg.seed)
    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    for ds in datasets:
        sample = load_state_dataset_sample(ds, cfg, rng)
        gt = _reachability_gt(
            sample.states,
            sample.episode_ids,
            sample.timesteps,
            sample.anchors,
            sample.candidates,
            cfg.horizon,
        )
        geo = np.linalg.norm(
            sample.states[sample.anchors][:, None, :2]
            - sample.states[sample.candidates][None, :, :2],
            axis=-1,
        )
        methods = [
            "euclidean",
            "gaussian",
            "mahalanobis",
            "adaptive_gaussian",
            "ik",
            "temporal_distance",
            "one_step_dynamics",
        ]
        context = state_scoring_context(sample, cfg)
        plot_scores: np.ndarray | None = None
        for method in methods:
            scores = build_experiment_scorer(method, cfg).score(context)
            if cfg.min_goal_dist > 0:
                scores = scores.copy()
                scores[geo < float(cfg.min_goal_dist)] = -np.inf
            if method == "ik":
                plot_scores = scores
            rows = []
            for i in range(sample.anchors.shape[0]):
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
                    "diversity": diversity_at_k(
                        sample.states[sample.candidates, :2],
                        scores[i],
                        cfg.top_k,
                    ),
                    "unique_goal_ratio": unique_goal_ratio_at_k(
                        sample.states[sample.candidates, :2],
                        scores[i],
                        cfg.top_k,
                    ),
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
        if plot_scores is not None:
            plot_topk_goals_on_maze(
                sample.states[sample.anchors[0], :2],
                sample.states[sample.candidates, :2],
                plot_scores[0],
                artifacts.figure_path(f"{dataset_slug(ds)}_ik_topk_seed{cfg.seed}.png"),
                cfg.top_k,
            )
    summary_path = artifacts.save_csv("summary.csv", summary_rows)
    per_anchor_path = artifacts.save_csv("per_anchor.csv", per_anchor_rows)
    fig_path = plot_relabel_bars(
        summary_rows,
        artifacts.figure_path(f"relabel_bars_seed{cfg.seed}.png"),
    )
    report_path = artifacts.write_report(
        "kNN Relabeling Report",
        [
            f"- summary: `{summary_path}`",
            f"- figure: `{fig_path}`",
        ],
    )
    return {
        "summary_rows": summary_rows,
        "per_anchor_rows": per_anchor_rows,
        "summary_path": summary_path,
        "per_anchor_path": per_anchor_path,
        "report_path": report_path,
    }
