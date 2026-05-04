"""Offline kNN planning experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reachability_metrics.data import load_dataset_or_synthetic
from reachability_metrics.evaluation.planning import multi_source_dijkstra
from reachability_metrics.experiments._sampling import sample_planning_queries
from reachability_metrics.experiments.artifacts import ArtifactWriter
from reachability_metrics.state_metrics import build_state_metric
from reachability_metrics.torch_utils import cpu_numpy
from reachability_metrics.utils import dataset_slug
from reachability_metrics.visualization.maze import plot_query_paths
from reachability_metrics.visualization.plots import (
    plot_planning_suboptimality,
    plot_planning_success_rate,
)


DEFAULT_DATASETS = [
    "D4RL/pointmaze/umaze-v2",
    "D4RL/pointmaze/large-v2",
    "D4RL/antmaze/umaze-diverse-v1",
]


@dataclass
class KNNPlanningEvalConfig:
    datasets: list[str]
    output_dir: str
    cache_dir: str | None = None
    seed: int = 0
    minari_datasets_path: str | None = None
    retrieval_top_k: int = 20
    num_queries: int = 200
    alpha: float = 1.5
    pointmaze_h_bridge: float = 10.0
    antmaze_h_bridge: float = 15.0
    ik_ensemble_size: int = 100
    ik_subsample_size: int = 32
    ik_temperature: float = 0.01
    ik_device: str = "auto"
    task_preset: str = "default"
    overwrite_cache: bool = False

    @property
    def tables_dir(self) -> str:
        return f"{self.output_dir}/tables"

    @property
    def figures_dir(self) -> str:
        return f"{self.output_dir}/figures"


def _topk_scores(metric: Any, states: np.ndarray, k: int) -> np.ndarray:
    scores = cpu_numpy(metric.pairwise_similarity(states, states))
    np.fill_diagonal(scores, -np.inf)
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1) - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def run_knn_planning_eval(cfg: KNNPlanningEvalConfig) -> dict[str, Any]:
    artifacts = ArtifactWriter(cfg.output_dir, cfg.cache_dir).prepare()
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    methods = ["euclidean", "gaussian", "ik"]
    for ds in cfg.datasets:
        dataset = load_dataset_or_synthetic(
            ds,
            minari_datasets_path=cfg.minari_datasets_path,
            use_achieved_goal=True,
            synthetic_seed=cfg.seed,
        )
        states_all = cpu_numpy(dataset.states())
        stride = max(states_all.shape[0] // 600, 1)
        node_xy = states_all[::stride, :2]
        n = node_xy.shape[0]
        temporal_targets = [[] for _ in range(n)]
        temporal_costs = [[] for _ in range(n)]
        for i in range(n - 1):
            temporal_targets[i].append(i + 1)
            temporal_costs[i].append(float(np.linalg.norm(node_xy[i + 1] - node_xy[i])))
        fit = node_xy
        metric_map = {
            "euclidean": build_state_metric("euclidean").fit(fit),
            "gaussian": build_state_metric("gaussian").fit(fit),
            "ik": build_state_metric(
                "ik",
                ensemble_size=cfg.ik_ensemble_size,
                subsample_size=cfg.ik_subsample_size,
                temperature=cfg.ik_temperature,
                device=cfg.ik_device,
                random_state=cfg.seed,
            ).fit(fit),
        }
        queries = sample_planning_queries(rng, n, cfg.num_queries)
        for method in methods:
            topk = _topk_scores(
                metric_map[method],
                node_xy,
                min(cfg.retrieval_top_k, max(n - 1, 1)),
            )
            targets = [list(t) for t in temporal_targets]
            costs = [list(c) for c in temporal_costs]
            bridge_budget = (
                cfg.antmaze_h_bridge if "antmaze" in ds.lower() else cfg.pointmaze_h_bridge
            )
            valid_edges = 0
            total_edges = 0
            for i in range(n):
                for j in topk[i]:
                    total_edges += 1
                    d = float(np.linalg.norm(node_xy[i] - node_xy[j]))
                    if d <= bridge_budget:
                        targets[i].append(int(j))
                        costs[i].append(d)
                        valid_edges += 1
            graph = {
                "edge_targets": tuple(np.asarray(t, dtype=np.int64) for t in targets),
                "edge_costs": tuple(np.asarray(c, dtype=np.float32) for c in costs),
            }
            successes = []
            subopts = []
            paths = []
            for qid, (s, g) in enumerate(queries):
                target_mask = np.zeros(n, dtype=bool)
                target_mask[int(g)] = True
                result = multi_source_dijkstra(graph, n, np.asarray([int(s)]), target_mask)
                geo = float(np.linalg.norm(node_xy[int(s)] - node_xy[int(g)]))
                success = bool(result["found"]) and result["path_cost"] <= cfg.alpha * max(
                    geo,
                    1e-6,
                )
                successes.append(float(success))
                if result["found"]:
                    subopts.append(float(result["path_cost"] / max(geo, 1e-6)))
                    paths.append(result["path_nodes"])
                query_rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "query_id": int(qid),
                        "success": int(success),
                        "path_found": int(result["found"]),
                    }
                )
            rows.append({
                "dataset": ds,
                "dataset_slug": dataset_slug(ds),
                "method": method,
                "success_rate": float(np.mean(successes)),
                "path_suboptimality": float(np.mean(subopts)) if subopts else float("nan"),
                "precision": float(valid_edges / max(total_edges, 1)),
            })
        if paths:
            plot_query_paths(
                node_xy,
                paths[:3],
                artifacts.figure_path(f"{dataset_slug(ds)}_query_paths_seed{cfg.seed}.png"),
            )
    per_dataset_path = artifacts.save_csv("per_dataset_metrics.csv", rows)
    per_query_path = artifacts.save_csv("per_query_metrics.csv", query_rows)
    fig1 = plot_planning_success_rate(
        rows,
        artifacts.figure_path(f"planning_success_seed{cfg.seed}.png"),
    )
    fig2 = plot_planning_suboptimality(
        rows,
        artifacts.figure_path(f"planning_suboptimality_seed{cfg.seed}.png"),
    )
    report_path = artifacts.write_report(
        "kNN Planning Report",
        [
            f"- per dataset: `{per_dataset_path}`",
            f"- figures: `{fig1}`, `{fig2}`",
        ],
    )
    return {
        "per_dataset_table": per_dataset_path,
        "per_query_table": per_query_path,
        "report_path": report_path,
        "summary_rows": rows,
    }
