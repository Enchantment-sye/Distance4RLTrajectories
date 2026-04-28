"""Ranking metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = float(stats.pearsonr(x, y)[0])
    return value if np.isfinite(value) else 0.0


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = float(stats.spearmanr(x, y)[0])
    return value if np.isfinite(value) else 0.0


def topk_overlap(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = min(int(k), int(np.asarray(y_true).size))
    if n <= 0:
        return 0.0
    truth = set(np.argsort(np.asarray(y_true))[-n:].tolist())
    pred = set(np.argsort(np.asarray(y_score))[-n:].tolist())
    return float(len(truth & pred) / n)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    labels = np.asarray(y_true, dtype=np.float64)
    positives = set(np.flatnonzero(labels > 0).tolist())
    if not positives:
        return 0.0
    n = min(int(k), labels.size)
    pred = set(np.argsort(np.asarray(y_score, dtype=np.float64))[-n:].tolist())
    return float(len(positives & pred) / len(positives))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = min(int(k), y_true.size)
    if n <= 0:
        return 0.0
    order = np.argsort(y_score)[::-1][:n]
    gains = np.maximum(y_true[order], 0.0)
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(np.maximum(y_true, 0.0))[::-1][:n]
    ideal_dcg = float(np.sum(ideal * discounts))
    return dcg / ideal_dcg if ideal_dcg > 1e-12 else 0.0

