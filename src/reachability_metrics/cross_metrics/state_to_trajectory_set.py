"""State-to-trajectory-set distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.aggregation import aggregate_groupwise_distances, build_aggregation
from reachability_metrics.base import PairwiseTensorMetricMixin


class StateToTrajectorySetDistance(PairwiseTensorMetricMixin):
    """Aggregate state-to-trajectory distances over a set of trajectories."""

    def __init__(
        self,
        state_to_trajectory_metric: Any,
        aggregation: str = "min",
        softmin_tau: float = 1.0,
        k: int = 3,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.state_to_trajectory_metric = state_to_trajectory_metric
        self.aggregation = aggregation
        self.softmin_tau = softmin_tau
        self.k = k
        self.aggregation_strategy_ = build_aggregation(aggregation, softmin_tau=softmin_tau, k=k)
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def fit(self, trajectory_sets: list[list[Any]]) -> "StateToTrajectorySetDistance":
        flat = [traj for group in trajectory_sets for traj in group]
        self.state_to_trajectory_metric.fit(flat)
        self.trajectory_sets_ = trajectory_sets
        return self

    def pairwise_distance_tensor(self, states: Any, trajectory_sets: list[list[Any]] | None = None):
        sets = self.trajectory_sets_ if trajectory_sets is None else trajectory_sets
        return aggregate_groupwise_distances(
            self.state_to_trajectory_metric,
            states,
            sets,
            self.aggregation_strategy_,
        )
