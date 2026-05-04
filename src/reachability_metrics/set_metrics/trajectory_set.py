"""Two-level trajectory-set distance."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

from reachability_metrics.aggregation import (
    mean_group_embeddings,
    pairwise_distance_tensor,
    pairwise_embedding_distance_tensor,
    transform_tensor,
)
from reachability_metrics.base import PairwiseTensorMetricMixin, TransformTensorMixin
from reachability_metrics.registry import MetricRegistry
from reachability_metrics.trajectory_metrics import (
    AdaptiveGDKTrajectoryDistance,
    GDKTrajectoryDistance,
    IDKTrajectoryDistance,
)
from reachability_metrics.trajectory_metrics.kme_strategies import transform_embedding_pair
from reachability_metrics.torch_utils import as_trajectory_tensor_list, require_torch


_SET_KWARGS = {"normalize", "return_numpy", "output_format"}


def _split_set_metric_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_kwargs = dict(kwargs)
    set_kwargs = {key: metric_kwargs.pop(key) for key in list(metric_kwargs) if key in _SET_KWARGS}
    if "set_distance_mode" in metric_kwargs:
        set_kwargs["distance_mode"] = metric_kwargs.pop("set_distance_mode")
    return set_kwargs, metric_kwargs


def _trajectory_set_factory(trajectory_metric_cls: Any, **kwargs: Any) -> "TrajectorySetDistance":
    set_kwargs, metric_kwargs = _split_set_metric_kwargs(kwargs)
    return TrajectorySetDistance(trajectory_metric_cls(**metric_kwargs), **set_kwargs)


class TrajectorySetDistance(TransformTensorMixin, PairwiseTensorMetricMixin, BaseEstimator):
    """A compact two-level KME-style set metric over trajectory embeddings."""

    def __init__(
        self,
        trajectory_metric: Any | None = None,
        normalize: bool = True,
        distance_mode: str = "rkhs_norm",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.trajectory_metric = trajectory_metric
        self.normalize = normalize
        self.distance_mode = distance_mode
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def fit(self, trajectory_sets: list[list[Any]], y: Any = None) -> "TrajectorySetDistance":
        self.trajectory_sets_ = trajectory_sets
        flat = [traj for group in trajectory_sets for traj in group]
        metric = self.trajectory_metric or GDKTrajectoryDistance()
        metric.fit(flat)
        self.trajectory_metric_ = metric
        self._flat_ref_ = flat
        self.reference_embeddings_ = self.transform_trajectories(flat)
        return self

    def _normalize(self, emb: Any):
        torch = require_torch()
        if not self.normalize:
            return emb
        return emb / torch.linalg.norm(emb, dim=-1, keepdim=True).clamp_min(1e-12)

    def transform_trajectory_tensor(self, trajectory: Any):
        if hasattr(self.trajectory_metric_, "transform_tensor") or hasattr(self.trajectory_metric_, "transform"):
            emb = transform_tensor(self.trajectory_metric_, [trajectory])[0]
        else:
            d = pairwise_distance_tensor(self.trajectory_metric_, [trajectory], getattr(self, "_flat_ref_", [trajectory]))
            emb = -d.reshape(-1)
        return self._normalize(emb)

    def transform_trajectory(self, trajectory: Any):
        return self._return(self.transform_trajectory_tensor(trajectory))

    def transform_trajectories_tensor(self, trajectories: Any):
        trajs = as_trajectory_tensor_list(trajectories)
        if hasattr(self.trajectory_metric_, "transform_tensor") or hasattr(self.trajectory_metric_, "transform"):
            emb = transform_tensor(self.trajectory_metric_, trajs)
            return self._normalize(emb)
        ref = getattr(self, "_flat_ref_", trajs)
        d = pairwise_distance_tensor(self.trajectory_metric_, trajs, ref)
        return -d

    def transform_trajectories(self, trajectories: Any):
        return self._return(self.transform_trajectories_tensor(trajectories))

    def transform_set_tensor(self, trajectories: list[Any]):
        emb = self.transform_trajectories_tensor(trajectories)
        return emb.mean(dim=0)

    def transform_set(self, trajectories: list[Any]):
        return self._return(self.transform_set_tensor(trajectories))

    def transform_tensor(self, trajectory_sets: list[list[Any]]):
        group_embeddings = [self.transform_trajectories_tensor(group) for group in trajectory_sets]
        return mean_group_embeddings(group_embeddings)

    def pairwise_distance_tensor(self, sets_a: list[list[Any]], sets_b: list[list[Any]] | None = None):
        a, b = transform_embedding_pair(self.transform_tensor, sets_a, sets_b)
        return pairwise_embedding_distance_tensor(a, b, self.distance_mode)

    def novelty_score_tensor(self, trajectory_or_set: Any):
        """Distance to the fitted reference trajectory-set distribution."""
        ref = self.transform_tensor(self.trajectory_sets_).mean(dim=0, keepdim=True)
        if isinstance(trajectory_or_set, list) and trajectory_or_set and not hasattr(trajectory_or_set, "states"):
            emb = self.transform_set_tensor(trajectory_or_set)[None, :]
        else:
            emb = self.transform_trajectories_tensor([trajectory_or_set])
        return pairwise_embedding_distance_tensor(emb, ref, self.distance_mode).reshape(-1)

    def novelty_score(self, trajectory_or_set: Any):
        return self._return(self.novelty_score_tensor(trajectory_or_set))


_SET_METRIC_REGISTRY = {
    "adaptive_gdk2": lambda **kwargs: _trajectory_set_factory(
        AdaptiveGDKTrajectoryDistance,
        **kwargs,
    ),
    "gdk2": lambda **kwargs: _trajectory_set_factory(GDKTrajectoryDistance, **kwargs),
    "idk2": lambda **kwargs: _trajectory_set_factory(IDKTrajectoryDistance, **kwargs),
}

_SET_METRIC_FACTORY = MetricRegistry("set metric", _SET_METRIC_REGISTRY)


def build_set_metric(method: str = "gdk2", **kwargs: Any) -> TrajectorySetDistance:
    return _SET_METRIC_FACTORY.build(method, **kwargs)
