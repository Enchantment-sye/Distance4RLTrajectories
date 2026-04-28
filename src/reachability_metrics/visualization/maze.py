"""Maze-style trajectory visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.utils import ensure_dir


def _setup() -> Any:
    import os
    import tempfile

    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib-{os.getuid()}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_topk_goals_on_maze(
    anchor_xy: np.ndarray,
    candidate_xy: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    top_k: int = 20,
) -> str:
    plt = _setup()
    ensure_dir(__import__("os").path.dirname(output_path))
    anchor = np.asarray(anchor_xy, dtype=np.float32).reshape(-1, 2)[0]
    candidates = np.asarray(candidate_xy, dtype=np.float32)
    order = np.argsort(-np.asarray(scores, dtype=np.float64).reshape(-1))[: int(top_k)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(candidates[:, 0], candidates[:, 1], s=5, color="#C7CED8", alpha=0.4)
    ax.scatter(candidates[order, 0], candidates[order, 1], s=35, color="#E45756")
    ax.scatter(anchor[0], anchor[1], s=120, marker="*", color="#111827")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_query_paths(
    node_xy: np.ndarray,
    paths: list[np.ndarray],
    output_path: str,
    starts: np.ndarray | None = None,
    goals: np.ndarray | None = None,
) -> str:
    plt = _setup()
    ensure_dir(__import__("os").path.dirname(output_path))
    xy = np.asarray(node_xy, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xy[:, 0], xy[:, 1], s=5, color="#C7CED8", alpha=0.3)
    for path in paths:
        ids = np.asarray(path, dtype=np.int64)
        if ids.size:
            ax.plot(xy[ids, 0], xy[ids, 1], linewidth=2)
    if starts is not None:
        ax.scatter(starts[:, 0], starts[:, 1], s=50, marker="o", color="#2CA02C")
    if goals is not None:
        ax.scatter(goals[:, 0], goals[:, 1], s=50, marker="x", color="#D62728")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
