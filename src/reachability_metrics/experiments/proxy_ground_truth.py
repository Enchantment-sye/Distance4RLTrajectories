"""Proxy ground-truth utilities for offline reachability experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from reachability_metrics.data import TrajectoryDataset


@dataclass
class ProxyGroundTruthConfig:
    """Configuration for empirical H-step reachability labels."""

    horizon: int = 20
    hit_radius: float = 0.25
    distance_weight: float = 0.5


def empirical_h_reachability_scores(
    dataset: TrajectoryDataset,
    anchors: np.ndarray,
    candidates: np.ndarray,
    *,
    horizon: int = 20,
    hit_radius: float = 0.25,
    distance_weight: float = 0.5,
) -> np.ndarray:
    """Score whether candidates are reached within H future steps of anchors.

    The score combines an empirical binary hit indicator with a smooth inverse
    distance to the same future window. This mirrors the proxy labels used by
    the original analysis code while staying independent of environment wrappers.
    """

    states = dataset.states()
    episode_ids = dataset.episode_ids()
    timesteps = dataset.timesteps()
    anchors = np.asarray(anchors, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_states = states[candidates]
    scores = np.zeros((anchors.shape[0], candidates.shape[0]), dtype=np.float32)
    for row, anchor_idx in enumerate(anchors):
        ep = episode_ids[int(anchor_idx)]
        ts = timesteps[int(anchor_idx)]
        mask = (episode_ids == ep) & (timesteps > ts) & (timesteps <= ts + int(horizon))
        future = states[mask]
        if future.size == 0:
            continue
        distances = cdist(candidate_states, future)
        nearest = np.min(distances, axis=1)
        hit = (nearest <= float(hit_radius)).astype(np.float32)
        smooth = 1.0 / (1.0 + nearest.astype(np.float32))
        weight = float(distance_weight)
        scores[row] = (1.0 - weight) * hit + weight * smooth
    return scores


def first_hit_temporal_distances(
    dataset: TrajectoryDataset,
    anchors: np.ndarray,
    candidates: np.ndarray,
    *,
    hit_radius: float = 0.25,
    max_window: int | None = None,
) -> np.ndarray:
    """Approximate first-hit temporal distance from each anchor to candidates."""

    states = dataset.states()
    episode_ids = dataset.episode_ids()
    timesteps = dataset.timesteps()
    anchors = np.asarray(anchors, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_states = states[candidates]
    out = np.full((anchors.shape[0], candidates.shape[0]), np.inf, dtype=np.float32)
    for row, anchor_idx in enumerate(anchors):
        ep = episode_ids[int(anchor_idx)]
        ts = timesteps[int(anchor_idx)]
        mask = (episode_ids == ep) & (timesteps > ts)
        if max_window is not None:
            mask &= timesteps <= ts + int(max_window)
        future_idx = np.flatnonzero(mask)
        if future_idx.size == 0:
            continue
        d = cdist(candidate_states, states[future_idx])
        hits = d <= float(hit_radius)
        for col in range(candidates.shape[0]):
            hit_pos = np.flatnonzero(hits[col])
            if hit_pos.size:
                out[row, col] = float(timesteps[future_idx[int(hit_pos[0])]] - ts)
    return out


def geodesic_proxy_distances(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Local geometric proxy used when an environment-specific maze map is absent."""

    a = np.asarray(states_a, dtype=np.float32)
    b = np.asarray(states_b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return cdist(a[:, :2], b[:, :2]).astype(np.float32)

