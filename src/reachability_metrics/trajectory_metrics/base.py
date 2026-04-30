"""Torch-first base trajectory metric interface."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

from reachability_metrics.base import PairwiseTensorMetricMixin
from reachability_metrics.torch_utils import (
    as_trajectory_tensor_list,
    resolve_torch_device,
    torch_dtype,
)


class TrajectoryMetric(PairwiseTensorMetricMixin, BaseEstimator):
    """Sklearn-style trajectory metric base class with torch tensor outputs by default."""

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "float32",
        batch_size: int = 1024,
        block_size: int = 1024,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.block_size = block_size
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def _device(self):
        return resolve_torch_device(getattr(self, "device", "auto"))

    def _dtype(self):
        return torch_dtype(getattr(self, "dtype", "float32"))

    def fit(self, trajectories: Any, y: Any = None) -> "TrajectoryMetric":
        self.trajectories_ = as_trajectory_tensor_list(trajectories, dtype=self._dtype(), device=self._device())
        return self

    def _check_pair_inputs(self, A: Any, B: Any | None = None):
        a = as_trajectory_tensor_list(A, dtype=self._dtype(), device=self._device())
        b = a if B is None else as_trajectory_tensor_list(B, dtype=self._dtype(), device=self._device())
        return a, b

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        raise NotImplementedError
