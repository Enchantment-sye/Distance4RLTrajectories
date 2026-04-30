"""Two-level trajectory-set distance."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

from reachability_metrics.base import PairwiseTensorMetricMixin
from reachability_metrics.trajectory_metrics import (
    AdaptiveGDKTrajectoryDistance,
    GDKTrajectoryDistance,
    IDKTrajectoryDistance,
)
from reachability_metrics.torch_utils import as_trajectory_tensor_list, cosine_distance_matrix, pairwise_euclidean, require_torch


class TrajectorySetDistance(PairwiseTensorMetricMixin, BaseEstimator):
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
        if hasattr(self.trajectory_metric_, "transform"):
            emb = self.trajectory_metric_.transform_tensor([trajectory])[0]
        else:
            d = self.trajectory_metric_.pairwise_distance([trajectory], getattr(self, "_flat_ref_", [trajectory]))
            emb = -d.reshape(-1)
        return self._normalize(emb)

    def transform_trajectory(self, trajectory: Any):
        return self._return(self.transform_trajectory_tensor(trajectory))

    def transform_trajectories_tensor(self, trajectories: Any):
        trajs = as_trajectory_tensor_list(trajectories)
        if hasattr(self.trajectory_metric_, "transform"):
            emb = self.trajectory_metric_.transform_tensor(trajs)
            return self._normalize(emb)
        ref = getattr(self, "_flat_ref_", trajs)
        d = self.trajectory_metric_.pairwise_distance(trajs, ref)
        return -d

    def transform_trajectories(self, trajectories: Any):
        return self._return(self.transform_trajectories_tensor(trajectories))

    def transform_set_tensor(self, trajectories: list[Any]):
        emb = self.transform_trajectories_tensor(trajectories)
        return emb.mean(dim=0)

    def transform_set(self, trajectories: list[Any]):
        return self._return(self.transform_set_tensor(trajectories))

    def transform_tensor(self, trajectory_sets: list[list[Any]]):
        torch = require_torch()
        return torch.stack([self.transform_set_tensor(group) for group in trajectory_sets], dim=0)

    def transform(self, trajectory_sets: list[list[Any]]):
        return self._return(self.transform_tensor(trajectory_sets))

    def pairwise_distance_tensor(self, sets_a: list[list[Any]], sets_b: list[list[Any]] | None = None):
        a = self.transform_tensor(sets_a)
        b = a if sets_b is None else self.transform_tensor(sets_b)
        if str(self.distance_mode).lower() == "cosine":
            return cosine_distance_matrix(a, b)
        return pairwise_euclidean(a, b)

    def novelty_score_tensor(self, trajectory_or_set: Any):
        """Distance to the fitted reference trajectory-set distribution."""
        ref = self.transform_tensor(self.trajectory_sets_).mean(dim=0, keepdim=True)
        if isinstance(trajectory_or_set, list) and trajectory_or_set and not hasattr(trajectory_or_set, "states"):
            emb = self.transform_set_tensor(trajectory_or_set)[None, :]
        else:
            emb = self.transform_trajectories_tensor([trajectory_or_set])
        if str(self.distance_mode).lower() == "cosine":
            return cosine_distance_matrix(emb, ref).reshape(-1)
        return pairwise_euclidean(emb, ref).reshape(-1)

    def novelty_score(self, trajectory_or_set: Any):
        return self._return(self.novelty_score_tensor(trajectory_or_set))


def build_set_metric(method: str = "gdk2", **kwargs: Any) -> TrajectorySetDistance:
    set_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in {"normalize", "return_numpy", "output_format"}}
    if "set_distance_mode" in kwargs:
        set_kwargs["distance_mode"] = kwargs.pop("set_distance_mode")
    key = method.lower()
    if key == "idk2":
        traj_metric = IDKTrajectoryDistance(**kwargs)
    elif key == "adaptive_gdk2":
        traj_metric = AdaptiveGDKTrajectoryDistance(**kwargs)
    elif key == "gdk2":
        traj_metric = GDKTrajectoryDistance(**kwargs)
    else:
        raise ValueError("Unknown set metric '{method}'. Available: adaptive_gdk2, gdk2, idk2".format(method=method))
    return TrajectorySetDistance(traj_metric, **set_kwargs)
