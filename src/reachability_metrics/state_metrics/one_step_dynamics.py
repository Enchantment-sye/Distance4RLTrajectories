"""One-step dynamics distance baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from .base import StateMetric


class OneStepDynamicsDistance(StateMetric):
    """Compare empirical one-step next-state distributions or local successor clouds."""

    def __init__(
        self,
        backend: str = "local_knn_nextstate",
        num_bins: int = 64,
        distance_metric: str = "jsd",
        local_knn_m: int = 20,
        alpha: float = 1e-3,
        min_count: int = 5,
        random_state: int = 0,
    ) -> None:
        self.backend = backend
        self.num_bins = num_bins
        self.distance_metric = distance_metric
        self.local_knn_m = local_knn_m
        self.alpha = alpha
        self.min_count = min_count
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "OneStepDynamicsDistance":
        states = np.asarray(X, dtype=np.float64)
        if y is None:
            if states.ndim != 3:
                raise ValueError("fit expects (states, next_states) or a trajectory array when y is None")
            current = states[:, :-1, :].reshape(-1, states.shape[-1])
            nxt = states[:, 1:, :].reshape(-1, states.shape[-1])
        else:
            current = np.asarray(X, dtype=np.float64)
            nxt = np.asarray(y, dtype=np.float64)
        if current.ndim != 2 or nxt.ndim != 2 or current.shape != nxt.shape:
            raise ValueError("train states and next states must both have shape (N, D)")
        self.X_fit_ = current
        self.next_states_ = nxt
        self.n_features_in_ = int(current.shape[1])
        self.tree_ = cKDTree(current) if current.shape[0] else None
        if str(self.backend).lower() != "local_knn_nextstate":
            self._fit_distribution_model(current, nxt)
        return self

    def _fit_distribution_model(self, states: np.ndarray, next_states: np.ndarray) -> None:
        n_bins = min(max(int(self.num_bins), 1), max(int(states.shape[0]), 1))
        if states.shape[0] == 0:
            self.transition_probabilities_ = np.ones((1, 1), dtype=np.float64)
            return
        key = str(self.backend).lower()
        if key == "grid" and states.shape[1] == 2:
            side = max(int(np.ceil(np.sqrt(n_bins))), 1)
            mins = np.min(states, axis=0)
            maxs = np.max(states, axis=0)
            maxs = np.where(np.isclose(maxs, mins), mins + 1.0, maxs)
            self.grid_edges_ = (
                np.linspace(mins[0], maxs[0], side + 1),
                np.linspace(mins[1], maxs[1], side + 1),
            )
            cur = self._assign_grid(states)
            nxt = self._assign_grid(next_states)
            n_bins = side * side
            self.kmeans_ = None
        else:
            self.kmeans_ = KMeans(n_clusters=n_bins, n_init=5, random_state=self.random_state).fit(states)
            cur = self.kmeans_.predict(states)
            nxt = self.kmeans_.predict(next_states)
        counts = np.zeros((n_bins, n_bins), dtype=np.float64)
        np.add.at(counts, (cur, nxt), 1.0)
        alpha = max(float(self.alpha), 1e-12)
        row_counts = counts.sum(axis=1)
        global_dist = (counts.sum(axis=0) + alpha) / (counts.sum() + alpha * n_bins)
        probs = (counts + alpha) / np.maximum(row_counts[:, None] + alpha * n_bins, 1e-12)
        probs[row_counts < int(self.min_count)] = global_dist[None, :]
        self.transition_probabilities_ = probs
        self.row_counts_ = row_counts

    def _assign_grid(self, x: np.ndarray) -> np.ndarray:
        ex, ey = self.grid_edges_
        xb = np.clip(np.digitize(x[:, 0], ex[1:-1]), 0, len(ex) - 2)
        yb = np.clip(np.digitize(x[:, 1], ey[1:-1]), 0, len(ey) - 2)
        return (xb * (len(ey) - 1) + yb).astype(np.int64)

    def _assign_bins(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self, "kmeans_") and self.kmeans_ is not None:
            return self.kmeans_.predict(x)
        if hasattr(self, "grid_edges_"):
            return self._assign_grid(x)
        return np.zeros(x.shape[0], dtype=np.int64)

    def _distribution_distance(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        if str(self.distance_metric).lower() in {"l1", "manhattan"}:
            return np.sum(np.abs(p[:, None, :] - q[None, :, :]), axis=-1)
        eps = 1e-12
        ps = np.clip(p, eps, None)
        qs = np.clip(q, eps, None)
        m = 0.5 * (ps[:, None, :] + qs[None, :, :])
        kl_p = np.sum(ps[:, None, :] * np.log(ps[:, None, :] / m), axis=-1)
        kl_q = np.sum(qs[None, :, :] * np.log(qs[None, :, :] / m), axis=-1)
        return np.sqrt(np.maximum(0.5 * (kl_p + kl_q), 0.0))

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        if not hasattr(self, "X_fit_"):
            raise RuntimeError("OneStepDynamicsDistance must be fitted")
        x, y = self._check_pair_inputs(X, Y)
        if str(self.backend).lower() == "local_knn_nextstate":
            if self.tree_ is None:
                return np.full((x.shape[0], y.shape[0]), np.inf, dtype=np.float32)
            k = min(max(int(self.local_knn_m), 1), self.X_fit_.shape[0])
            _, nn = self.tree_.query(x, k=k)
            if nn.ndim == 1:
                nn = nn[:, None]
            clouds = self.next_states_[nn]
            out = np.empty((x.shape[0], y.shape[0]), dtype=np.float32)
            for i in range(x.shape[0]):
                out[i] = np.min(cdist(clouds[i], y, metric="euclidean"), axis=0)
            return out
        bx = self._assign_bins(x)
        by = self._assign_bins(y)
        ux, ix = np.unique(bx, return_inverse=True)
        uy, iy = np.unique(by, return_inverse=True)
        lookup = self._distribution_distance(self.transition_probabilities_[ux], self.transition_probabilities_[uy])
        return lookup[ix[:, None], iy[None, :]].astype(np.float32)

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        return (-self.pairwise_distance(X, Y)).astype(np.float32)

