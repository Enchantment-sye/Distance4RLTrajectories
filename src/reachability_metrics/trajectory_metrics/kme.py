"""Kernel mean embedding for trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.state_metrics import StateMetric
from reachability_metrics.utils import as_trajectory_list


class KernelMeanEmbedding:
    """Represent a trajectory by the mean feature map of its states."""

    def __init__(self, base_kernel: StateMetric, normalize: bool = True) -> None:
        self.base_kernel = base_kernel
        self.normalize = normalize

    def fit(self, X: Any) -> "KernelMeanEmbedding":
        """Fit the base kernel from all states in trajectories or a state matrix."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            states = X
        else:
            states = np.concatenate(as_trajectory_list(X, dtype=np.float64), axis=0)
        self.base_kernel.fit(states)
        return self

    def _state_features(self, states: np.ndarray) -> np.ndarray:
        if hasattr(self.base_kernel, "transform"):
            try:
                return self.base_kernel.transform(states, normalize=bool(self.normalize))  # type: ignore[misc]
            except TypeError:
                return self.base_kernel.transform(states)  # type: ignore[misc]
        reference = getattr(self.base_kernel, "X_fit_", states)
        return self.base_kernel.pairwise_similarity(states, reference)

    def transform_trajectory(self, traj: Any) -> np.ndarray:
        """Return a mean embedding for one trajectory."""
        states = np.asarray(traj.states if hasattr(traj, "states") else traj, dtype=np.float64)
        return np.mean(self._state_features(states), axis=0).astype(np.float32)

    def transform(self, trajectories: Any) -> np.ndarray:
        """Return mean embeddings for multiple trajectories."""
        return np.stack([self.transform_trajectory(traj) for traj in as_trajectory_list(trajectories)], axis=0)

    def pairwise_kernel(self, trajectories_a: Any, trajectories_b: Any | None = None) -> np.ndarray:
        """Pairwise trajectory kernel from mean embeddings."""
        a = self.transform(trajectories_a)
        b = a if trajectories_b is None else self.transform(trajectories_b)
        return (a @ b.T).astype(np.float32)

    def pairwise_distance(self, trajectories_a: Any, trajectories_b: Any | None = None) -> np.ndarray:
        """Kernel-induced trajectory distribution distance."""
        k_ab = self.pairwise_kernel(trajectories_a, trajectories_b)
        k_aa = np.diag(self.pairwise_kernel(trajectories_a, trajectories_a))
        if trajectories_b is None:
            k_bb = k_aa
        else:
            k_bb = np.diag(self.pairwise_kernel(trajectories_b, trajectories_b))
        return np.sqrt(np.maximum(k_aa[:, None] + k_bb[None, :] - 2.0 * k_ab, 0.0)).astype(np.float32)

