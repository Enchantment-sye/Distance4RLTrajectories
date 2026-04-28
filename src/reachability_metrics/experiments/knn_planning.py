"""Offline kNN planning experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from reachability_metrics.data import load_dataset_or_synthetic
from reachability_metrics.evaluation.planning import multi_source_dijkstra
from reachability_metrics.evaluation.reports import save_csv
from reachability_metrics.state_metrics import EuclideanDistance, GaussianKernelDistance, IsolationKernelDistance
from reachability_metrics.utils import dataset_slug, ensure_dir
from reachability_metrics.visualization.maze import plot_query_paths
from reachability_metrics.visualization.plots import plot_planning_suboptimality, plot_planning_success_rate


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
    scores = metric.pairwise_similarity(states, states)
    np.fill_diagonal(scores, -np.inf)
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1) - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def run_knn_planning_eval(cfg: KNNPlanningEvalConfig) -> dict[str, Any]:
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir or f"{cfg.output_dir}/cache")
    ensure_dir(cfg.tables_dir)
    ensure_dir(cfg.figures_dir)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    methods = ["euclidean", "gaussian", "ik"]
    for ds in cfg.datasets:
        dataset = load_dataset_or_synthetic(ds, minari_datasets_path=cfg.minari_datasets_path, use_achieved_goal=True, synthetic_seed=cfg.seed)
        states_all = dataset.states()
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
            "euclidean": EuclideanDistance().fit(fit),
            "gaussian": GaussianKernelDistance().fit(fit),
            "ik": IsolationKernelDistance(cfg.ik_ensemble_size, cfg.ik_subsample_size, cfg.ik_temperature, device=cfg.ik_device, random_state=cfg.seed).fit(fit),
        }
        queries = rng.integers(0, n, size=(min(cfg.num_queries, max(n // 2, 1)), 2))
        queries = queries[queries[:, 0] != queries[:, 1]]
        if queries.size == 0:
            queries = np.asarray([[0, n - 1]], dtype=np.int64)
        for method in methods:
            topk = _topk_scores(metric_map[method], node_xy, min(cfg.retrieval_top_k, max(n - 1, 1)))
            targets = [list(t) for t in temporal_targets]
            costs = [list(c) for c in temporal_costs]
            bridge_budget = cfg.antmaze_h_bridge if "antmaze" in ds.lower() else cfg.pointmaze_h_bridge
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
                success = bool(result["found"]) and result["path_cost"] <= cfg.alpha * max(geo, 1e-6)
                successes.append(float(success))
                if result["found"]:
                    subopts.append(float(result["path_cost"] / max(geo, 1e-6)))
                    paths.append(result["path_nodes"])
                query_rows.append({"dataset": ds, "method": method, "query_id": int(qid), "success": int(success), "path_found": int(result["found"])})
            rows.append({
                "dataset": ds,
                "dataset_slug": dataset_slug(ds),
                "method": method,
                "success_rate": float(np.mean(successes)),
                "path_suboptimality": float(np.mean(subopts)) if subopts else float("nan"),
                "precision": float(valid_edges / max(total_edges, 1)),
            })
        if paths:
            plot_query_paths(node_xy, paths[:3], f"{cfg.figures_dir}/{dataset_slug(ds)}_query_paths_seed{cfg.seed}.png")
    per_dataset_path = f"{cfg.tables_dir}/per_dataset_metrics.csv"
    per_query_path = f"{cfg.tables_dir}/per_query_metrics.csv"
    save_csv(per_dataset_path, rows)
    save_csv(per_query_path, query_rows)
    fig1 = plot_planning_success_rate(rows, f"{cfg.figures_dir}/planning_success_seed{cfg.seed}.png")
    fig2 = plot_planning_suboptimality(rows, f"{cfg.figures_dir}/planning_suboptimality_seed{cfg.seed}.png")
    report_path = f"{cfg.output_dir}/report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# kNN Planning Report\n\n")
        handle.write(f"- per dataset: `{per_dataset_path}`\n- figures: `{fig1}`, `{fig2}`\n")
    return {"per_dataset_table": per_dataset_path, "per_query_table": per_query_path, "report_path": report_path, "summary_rows": rows}

