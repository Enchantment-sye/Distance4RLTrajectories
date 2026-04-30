"""Trajectory-to-set distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.aggregation import build_aggregation
from reachability_metrics.base import PairwiseTensorMetricMixin
from reachability_metrics.state_metrics import GaussianKernelDistance
from reachability_metrics.torch_utils import as_tensor, require_torch
from reachability_metrics.trajectory_metrics import GDKTrajectoryDistance
from reachability_metrics.trajectory_metrics.kme import KernelMeanEmbedding


class TrajectoryToSetDistance(PairwiseTensorMetricMixin):
    """Aggregate trajectory-to-trajectory distances over a set."""

    def __init__(
        self,
        trajectory_metric: Any | None = None,
        aggregation: str = "min",
        softmin_tau: float = 1.0,
        k: int = 3,
        method: str = "two_level_kme",
        second_level_kernel: Any | None = None,
        distance_mode: str = "rkhs_norm",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.trajectory_metric = trajectory_metric
        self.aggregation = aggregation
        self.softmin_tau = softmin_tau
        self.k = k
        self.method = method
        self.second_level_kernel = second_level_kernel
        self.distance_mode = distance_mode
        self.aggregation_strategy_ = build_aggregation(aggregation, softmin_tau=softmin_tau, k=k)
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def fit(self, trajectory_sets: list[list[Any]]) -> "TrajectoryToSetDistance":
        flat = [traj for group in trajectory_sets for traj in group]
        metric = self.trajectory_metric or GDKTrajectoryDistance()
        metric.fit(flat)
        self.trajectory_metric_ = metric
        self.trajectory_sets_ = trajectory_sets
        if str(self.method).lower() == "two_level_kme":
            self._fit_two_level(trajectory_sets)
        return self

    def _fit_two_level(self, trajectory_sets: list[list[Any]]) -> None:
        group_points = []
        for group in trajectory_sets:
            group_points.append(self.trajectory_metric_.transform_tensor(group))
        kernel = self.second_level_kernel or GaussianKernelDistance()
        self.second_level_kme_ = KernelMeanEmbedding(
            kernel,
            distance_mode=self.distance_mode,
            return_numpy=False,
        ).fit(group_points)
        self.set_embeddings_ = self.second_level_kme_.transform_tensor(group_points)

    def _aggregate_distance_tensor(self, trajectories: Any, trajectory_sets: list[list[Any]] | None = None):
        torch = require_torch()
        sets = self.trajectory_sets_ if trajectory_sets is None else trajectory_sets
        rows = []
        for group in sets:
            d = as_tensor(self.trajectory_metric_.pairwise_distance(trajectories, group))
            rows.append(self.aggregation_strategy_.reduce(d, dim=1))
        return torch.stack(rows, dim=1)

    def _two_level_distance_tensor(self, trajectories: Any, trajectory_sets: list[list[Any]] | None = None):
        torch = require_torch()
        q_points = self.trajectory_metric_.transform_tensor(trajectories)
        q_emb = self.second_level_kme_.transform_tensor([point.reshape(1, -1) for point in q_points])
        if trajectory_sets is None:
            set_emb = self.set_embeddings_
        else:
            group_points = [self.trajectory_metric_.transform_tensor(group) for group in trajectory_sets]
            set_emb = self.second_level_kme_.transform_tensor(group_points)
        if str(self.distance_mode).lower() == "cosine":
            from reachability_metrics.torch_utils import cosine_distance_matrix

            return cosine_distance_matrix(q_emb, set_emb)
        return torch.sqrt(torch.clamp(torch.sum(q_emb * q_emb, dim=1, keepdim=True) + torch.sum(set_emb * set_emb, dim=1, keepdim=True).T - 2.0 * q_emb @ set_emb.T, min=0.0))

    def pairwise_distance_tensor(self, trajectories: Any, trajectory_sets: list[list[Any]] | None = None):
        if str(self.method).lower() == "two_level_kme":
            return self._two_level_distance_tensor(trajectories, trajectory_sets)
        return self._aggregate_distance_tensor(trajectories, trajectory_sets)
