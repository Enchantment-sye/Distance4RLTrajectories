"""Temporal distance baselines."""

from __future__ import annotations

from typing import Any

from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.torch_utils import as_tensor, require_torch
from .base import StateMetric


class TemporalDistance(StateMetric):
    """Temporal distance from trajectory identity and timestep metadata."""

    def __init__(
        self,
        mode: str = "same_trajectory_min_gap",
        max_window: int | None = None,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.mode = mode
        self.max_window = max_window

    def fit(self, X: Any, y: Any = None) -> "TemporalDistance":
        if isinstance(X, TrajectoryDataset):
            self.episode_ids_ = X.episode_ids()
            self.timesteps_ = X.timesteps()
            self.X_fit_ = X.states()
            self.n_features_in_ = int(self.X_fit_.shape[1])
            return self
        return super().fit(X, y)

    def pairwise_distance_indices_tensor(self, indices_a: Any, indices_b: Any):
        """Distance between stacked dataset indices."""
        if not hasattr(self, "episode_ids_"):
            raise RuntimeError("TemporalDistance.fit requires a TrajectoryDataset for index distances")
        torch = require_torch()
        a = as_tensor(indices_a, dtype=torch.long, device=self.episode_ids_.device)
        b = as_tensor(indices_b, dtype=torch.long, device=self.episode_ids_.device)
        same = self.episode_ids_[a][:, None] == self.episode_ids_[b][None, :]
        gap = torch.abs(self.timesteps_[a][:, None] - self.timesteps_[b][None, :]).to(dtype=self._dtype())
        dist = torch.where(same, gap, torch.full_like(gap, float("inf")))
        if self.max_window is not None:
            dist = torch.where(dist > int(self.max_window), torch.full_like(dist, float("inf")), dist)
        return dist

    def pairwise_distance_indices(self, indices_a: Any, indices_b: Any):
        return self._return(self.pairwise_distance_indices_tensor(indices_a, indices_b))

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        x, y = self._check_pair_inputs(X, Y)
        if Y is None and hasattr(self, "X_fit_") and x.shape[0] == self.X_fit_.shape[0] and torch.allclose(x, self.X_fit_):
            idx = torch.arange(x.shape[0], dtype=torch.long, device=x.device)
            return self.pairwise_distance_indices_tensor(idx, idx)
        return torch.full((x.shape[0], y.shape[0]), float("inf"), dtype=self._dtype(), device=x.device)

    def pairwise_similarity_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        d = self.pairwise_distance_tensor(X, Y)
        sim = torch.zeros_like(d)
        finite = torch.isfinite(d)
        sim[finite] = 1.0 / (1.0 + d[finite])
        return sim
