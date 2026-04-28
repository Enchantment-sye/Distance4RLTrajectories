"""kNN relabeling evaluation helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def goal_precision_at_k(y_true_binary: np.ndarray, y_score: np.ndarray, k: int) -> float:
    labels = np.asarray(y_true_binary, dtype=np.int64)
    n = min(int(k), labels.size)
    if n <= 0:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:n]
    return float(np.mean(labels[order] > 0))


def mean_gt_score_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = min(int(k), np.asarray(y_true).size)
    if n <= 0:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:n]
    return float(np.mean(np.asarray(y_true, dtype=np.float64)[order]))


def diversity_at_k(candidate_positions: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = min(int(k), candidate_positions.shape[0])
    if n < 2:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:n]
    d = pdist(np.asarray(candidate_positions[order], dtype=np.float64), metric="euclidean")
    return float(np.mean(d)) if d.size else 0.0


def unique_goal_ratio_at_k(candidate_positions: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = min(int(k), candidate_positions.shape[0])
    if n <= 0:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:n]
    unique = np.unique(np.round(candidate_positions[order], decimals=6), axis=0)
    return float(unique.shape[0] / n)

