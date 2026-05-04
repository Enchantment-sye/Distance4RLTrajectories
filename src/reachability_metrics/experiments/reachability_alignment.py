"""Reachability ranking-alignment experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reachability_metrics.evaluation import (
    auc_from_binary_labels,
    average_precision_from_binary_labels,
    ndcg_at_k,
    recall_at_k,
    safe_pearson,
    safe_spearman,
)
from reachability_metrics.experiments._sampling import (
    load_state_dataset_sample,
    state_scoring_context,
)
from reachability_metrics.experiments.artifacts import ArtifactWriter
from reachability_metrics.experiments.proxy_ground_truth import empirical_h_reachability_scores
from reachability_metrics.experiments.scorers import build_experiment_scorer
from reachability_metrics.utils import dataset_slug
from reachability_metrics.visualization.plots import plot_alignment_scatter, plot_relabel_bars


DEFAULT_DATASETS = [
    "D4RL/pointmaze/umaze-v2",
    "D4RL/pointmaze/large-v2",
    "D4RL/antmaze/umaze-diverse-v1",
]


@dataclass
class ReachabilityAnalysisConfig:
    """Configuration for reachability ranking alignment."""

    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    output_dir: str = "outputs/reachability_alignment"
    cache_dir: str | None = None
    seed: int = 0
    minari_datasets_path: str | None = None
    horizon: int = 20
    hit_radius: float = 0.25
    num_anchors: int = 200
    num_candidates: int = 1000
    fit_pool_size: int = 50000
    recall_k_values: tuple[int, ...] = (5, 10, 20)
    ik_ensemble_size: int = 100
    ik_subsample_size: int = 32
    ik_temperature: float = 0.01
    ik_batch_size: int = 4096
    ik_device: str = "auto"

    @property
    def tables_dir(self) -> str:
        return f"{self.output_dir}/tables"

    @property
    def figures_dir(self) -> str:
        return f"{self.output_dir}/figures"


def analyze_single_dataset(dataset_id: str, cfg: ReachabilityAnalysisConfig) -> dict[str, Any]:
    """Run alignment analysis for one dataset."""

    rng = np.random.default_rng(cfg.seed)
    sample = load_state_dataset_sample(dataset_id, cfg, rng)
    ground_truth = empirical_h_reachability_scores(
        sample.dataset,
        sample.anchors,
        sample.candidates,
        horizon=cfg.horizon,
        hit_radius=cfg.hit_radius,
    )
    methods = [
        "euclidean",
        "gaussian",
        "adaptive_gaussian",
        "mahalanobis",
        "ik",
        "temporal",
        "dyn_1",
    ]
    rows: list[dict[str, Any]] = []
    per_anchor: list[dict[str, Any]] = []
    first_scatter: str | None = None
    context = state_scoring_context(sample, cfg)
    for method in methods:
        scores = build_experiment_scorer(method, cfg).score(context)
        method_rows: list[dict[str, Any]] = []
        for i in range(sample.anchors.shape[0]):
            labels = (ground_truth[i] >= np.percentile(ground_truth[i], 75.0)).astype(np.int64)
            row = {
                "dataset": dataset_id,
                "dataset_slug": dataset_slug(dataset_id),
                "method": method,
                "anchor_row": int(i),
                "spearman": safe_spearman(scores[i], ground_truth[i]),
                "pearson": safe_pearson(scores[i], ground_truth[i]),
                "auroc": auc_from_binary_labels(labels, scores[i]),
                "average_precision": average_precision_from_binary_labels(labels, scores[i]),
                "ndcg_at_20": ndcg_at_k(ground_truth[i], scores[i], 20),
            }
            for k in cfg.recall_k_values:
                row[f"recall_at_{k}"] = recall_at_k(labels, scores[i], int(k))
            method_rows.append(row)
            per_anchor.append(row)
        rows.append({
            "dataset": dataset_id,
            "dataset_slug": dataset_slug(dataset_id),
            "method": method,
            "spearman": float(np.mean([r["spearman"] for r in method_rows])),
            "pearson": float(np.mean([r["pearson"] for r in method_rows])),
            "auroc": float(np.mean([r["auroc"] for r in method_rows])),
            "average_precision": float(np.mean([r["average_precision"] for r in method_rows])),
            "ndcg_at_20": float(np.mean([r["ndcg_at_20"] for r in method_rows])),
        })
        if method == "ik":
            first_scatter = plot_alignment_scatter(
                ground_truth.reshape(-1),
                scores.reshape(-1),
                f"{cfg.figures_dir}/{dataset_slug(dataset_id)}_ik_alignment_seed{cfg.seed}.png",
                title=f"{dataset_slug(dataset_id)} IK alignment",
            )
    return {"summary_rows": rows, "per_anchor_rows": per_anchor, "scatter_path": first_scatter}


def analyze_datasets(cfg: ReachabilityAnalysisConfig) -> dict[str, Any]:
    """Run reachability alignment for all configured datasets."""

    artifacts = ArtifactWriter(cfg.output_dir, cfg.cache_dir).prepare()
    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    scatter_paths = []
    for dataset_id in cfg.datasets:
        result = analyze_single_dataset(dataset_id, cfg)
        summary_rows.extend(result["summary_rows"])
        per_anchor_rows.extend(result["per_anchor_rows"])
        if result["scatter_path"]:
            scatter_paths.append(result["scatter_path"])
    summary_path = artifacts.save_csv("summary.csv", summary_rows)
    per_anchor_path = artifacts.save_csv("per_anchor.csv", per_anchor_rows)
    bar_path = plot_relabel_bars(
        summary_rows,
        artifacts.figure_path(f"alignment_spearman_seed{cfg.seed}.png"),
    )
    report_path = artifacts.write_report(
        "Reachability Alignment Report",
        [
            f"- summary: `{summary_path}`",
            f"- per-anchor: `{per_anchor_path}`",
            f"- spearman figure: `{bar_path}`",
            *[f"- scatter: `{path}`" for path in scatter_paths],
        ],
    )
    return {
        "summary_rows": summary_rows,
        "per_anchor_rows": per_anchor_rows,
        "summary_path": summary_path,
        "per_anchor_path": per_anchor_path,
        "report_path": report_path,
        "figure_path": bar_path,
        "scatter_paths": scatter_paths,
    }


def run_search(cfg: ReachabilityAnalysisConfig) -> dict[str, Any]:
    """Compatibility wrapper for IK hyperparameter search entry points."""

    return analyze_datasets(cfg)


def run_final_evaluation(cfg: ReachabilityAnalysisConfig) -> dict[str, Any]:
    """Compatibility wrapper for final evaluation entry points."""

    return analyze_datasets(cfg)
