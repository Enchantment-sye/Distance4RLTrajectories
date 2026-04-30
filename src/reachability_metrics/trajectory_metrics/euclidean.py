"""Flattened/resampled Euclidean trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_euclidean, torch_resample_trajectory
from .base import TrajectoryMetric


class TrajectoryEuclideanDistance(TrajectoryMetric):
    """Euclidean distance after optional linear resampling."""

    def __init__(
        self,
        target_length: int | None = None,
        resample: str = "linear",
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.target_length = target_length
        self.resample = resample

    def _prepare(self, trajs: list[Any]):
        torch = __import__("torch")
        if self.target_length is None:
            lengths = {traj.shape[0] for traj in trajs}
            if len(lengths) != 1:
                target = max(lengths)
                return torch.stack([torch_resample_trajectory(traj, target).reshape(-1) for traj in trajs], dim=0)
            return torch.stack([traj.reshape(-1) for traj in trajs], dim=0)
        return torch.stack([torch_resample_trajectory(traj, int(self.target_length)).reshape(-1) for traj in trajs], dim=0)

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        a, b = self._check_pair_inputs(A, B)
        xa = self._prepare(a)
        xb = xa if B is None else self._prepare(b)
        return pairwise_euclidean(xa, xb)
