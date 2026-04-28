"""Binary-label ranking metrics without sklearn dependency at call sites."""

from __future__ import annotations

import numpy as np
from scipy import stats


def auc_from_binary_labels(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.asarray(y_true_binary, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return 0.5
    ranks = stats.rankdata(scores, method="average")
    pos_ranks = np.sum(ranks[labels == 1])
    auc = (pos_ranks - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(auc) if np.isfinite(auc) else 0.5


def average_precision_from_binary_labels(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.asarray(y_true_binary, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    pos = int(np.sum(labels == 1))
    if pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1, dtype=np.float64)
    fp = np.cumsum(sorted_labels == 0, dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(pos)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

