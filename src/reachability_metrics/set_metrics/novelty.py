"""Trajectory novelty scoring."""

from __future__ import annotations

from typing import Any

from .trajectory_set import build_set_metric


class TrajectoryNoveltyScorer:
    """Score trajectories by distance from a reference trajectory distribution."""

    def __init__(self, method: str = "idk2", **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs

    def fit(self, trajectories: Any, y: Any = None) -> "TrajectoryNoveltyScorer":
        self.reference_ = list(trajectories)
        self.metric_ = build_set_metric(self.method, **self.kwargs).fit([self.reference_])
        return self

    def score(self, trajectories: Any) -> Any:
        return self.novelty_score(trajectories)

    def novelty_score(self, trajectories: Any) -> Any:
        return self.metric_.novelty_score(trajectories)

