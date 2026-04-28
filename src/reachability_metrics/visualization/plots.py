"""Matplotlib report figures."""

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


def _bar(rows: list[dict[str, Any]], metric: str, path: str, title: str) -> str:
    plt = _setup()
    ensure_dir(__import__("os").path.dirname(path))
    labels = [str(r.get("method", r.get("name", "?"))) for r in rows]
    values = [float(r.get(metric, 0.0)) if np.isfinite(float(r.get(metric, 0.0))) else 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_relabel_bars(rows: list[dict[str, Any]], output_path: str) -> str:
    return _bar(rows, "spearman", output_path, "kNN Relabeling Spearman")


def plot_planning_success_rate(rows: list[dict[str, Any]], output_path: str) -> str:
    return _bar(rows, "success_rate", output_path, "Planning Success Rate")


def plot_planning_suboptimality(rows: list[dict[str, Any]], output_path: str) -> str:
    return _bar(rows, "path_suboptimality", output_path, "Planning Path Suboptimality")


def plot_successor_auroc(rows: list[dict[str, Any]], output_path: str) -> str:
    return _bar(rows, "auroc", output_path, "H-step Successor AUROC")


def plot_ik_sweep_heatmap(matrix: np.ndarray, xlabels: list[str], ylabels: list[str], output_path: str) -> str:
    plt = _setup()
    ensure_dir(__import__("os").path.dirname(output_path))
    fig, ax = plt.subplots(figsize=(max(6, len(xlabels) * 0.6), max(4, len(ylabels) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_alignment_scatter(y_true: np.ndarray, y_score: np.ndarray, output_path: str, title: str = "Alignment") -> str:
    plt = _setup()
    ensure_dir(__import__("os").path.dirname(output_path))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.asarray(y_true).reshape(-1), np.asarray(y_score).reshape(-1), s=5, alpha=0.35)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
