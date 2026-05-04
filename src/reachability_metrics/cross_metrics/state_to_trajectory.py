"""State-to-trajectory distances."""

from __future__ import annotations

from typing import Any

from reachability_metrics.aggregation import aggregate_groupwise_distances, build_aggregation
from reachability_metrics.base import PairwiseTensorMetricMixin
from reachability_metrics.state_metrics import StateMetric
from reachability_metrics.trajectory_metrics.kme import KernelMeanEmbedding
from reachability_metrics.torch_utils import (
    as_2d_tensor,
    as_trajectory_tensor_list,
    require_torch,
)


class StateToTrajectoryDistance(PairwiseTensorMetricMixin):
    """Aggregate pointwise state distances from a state to a trajectory."""

    def __init__(
        self,
        state_metric: StateMetric,
        aggregation: str = "min",
        softmin_tau: float = 1.0,
        k: int = 3,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.state_metric = state_metric
        self.aggregation = aggregation
        self.softmin_tau = softmin_tau
        self.k = k
        self.aggregation_strategy_ = build_aggregation(aggregation, softmin_tau=softmin_tau, k=k)
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def fit(self, trajectories: Any) -> "StateToTrajectoryDistance":
        torch = require_torch()
        trajs = as_trajectory_tensor_list(trajectories)
        self.trajectories_ = trajs
        self.state_metric.fit(torch.cat(trajs, dim=0))
        return self

    def pairwise_distance_tensor(self, states: Any, trajectories: Any | None = None):
        x = as_2d_tensor(states, dtype=getattr(self.state_metric, "dtype", "float32"), device=getattr(self.state_metric, "device", "auto"), name="states")
        trajs = self.trajectories_ if trajectories is None else as_trajectory_tensor_list(trajectories, dtype=x.dtype, device=x.device)
        return aggregate_groupwise_distances(
            self.state_metric,
            x,
            trajs,
            self.aggregation_strategy_,
            dtype=x.dtype,
            device=x.device,
        )


class StateToTrajectoryKMEDistance(PairwiseTensorMetricMixin):
    """State-to-trajectory distance from a KME base kernel."""

    def __init__(
        self,
        base_kernel: StateMetric,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.base_kernel = base_kernel
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def fit(self, trajectories: Any) -> "StateToTrajectoryKMEDistance":
        torch = require_torch()
        self.trajectories_ = as_trajectory_tensor_list(trajectories)
        self.kme_ = KernelMeanEmbedding(self.base_kernel, normalize=True).fit(self.trajectories_)
        self.traj_embeddings_ = self.kme_.transform_tensor(self.trajectories_)
        self.traj_self_ = torch.sum(self.traj_embeddings_ * self.traj_embeddings_, dim=1)
        return self

    def pairwise_similarity_tensor(self, states: Any, trajectories: Any | None = None):
        x = as_2d_tensor(states, dtype=self.kme_._dtype(), device=self.kme_._device(), name="states")
        fx = self.kme_._state_features_tensor(x)
        emb = self.traj_embeddings_ if trajectories is None else self.kme_.transform_tensor(trajectories)
        return fx @ emb.T

    def pairwise_distance_tensor(self, states: Any, trajectories: Any | None = None):
        torch = require_torch()
        x = as_2d_tensor(states, dtype=self.kme_._dtype(), device=self.kme_._device(), name="states")
        fx = self.kme_._state_features_tensor(x)
        emb = self.traj_embeddings_ if trajectories is None else self.kme_.transform_tensor(trajectories)
        traj_self = self.traj_self_ if trajectories is None else torch.sum(emb * emb, dim=1)
        sx = torch.sum(fx * fx, dim=1)
        sim = fx @ emb.T
        return torch.sqrt(torch.clamp(sx[:, None] + traj_self[None, :] - 2.0 * sim, min=0.0))
