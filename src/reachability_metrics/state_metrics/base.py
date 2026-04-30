"""Torch-first base state metric interface."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

from reachability_metrics.base import PairwiseTensorMetricMixin
from reachability_metrics.torch_utils import (
    as_2d_tensor,
    pairwise_sqeuclidean,
    resolve_torch_device,
    torch_dtype,
)


class StateMetric(PairwiseTensorMetricMixin, BaseEstimator):
    """Sklearn-style state metric base class with torch tensor outputs by default."""

    higher_similarity_is_closer = True

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "float32",
        batch_size: int = 4096,
        block_size: int = 4096,
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

    def fit(self, X: Any, y: Any = None) -> "StateMetric":
        """Fit the metric."""
        self.X_fit_ = as_2d_tensor(X, dtype=self._dtype(), device=self._device(), name="X")
        self.n_features_in_ = int(self.X_fit_.shape[1])
        return self

    def _check_pair_inputs(self, X: Any, Y: Any | None = None):
        x = as_2d_tensor(X, dtype=self._dtype(), device=self._device(), name="X")
        y = x if Y is None else as_2d_tensor(Y, dtype=self._dtype(), device=x.device, name="Y")
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"X and Y feature dims must match, got {x.shape[1]} and {y.shape[1]}")
        return x, y

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        """Pairwise distances as a torch tensor."""
        raise NotImplementedError

    def kneighbors(self, X: Any, Y: Any | None = None, k: int = 20):
        """Return nearest-neighbor distances and indices as torch tensors by default."""
        torch = __import__("torch")
        distances = self.pairwise_distance_tensor(X, Y)
        top_k = min(int(k), distances.shape[1])
        if top_k <= 0:
            raise ValueError("k must be positive")
        vals, idx = torch.topk(distances, k=top_k, dim=1, largest=False, sorted=True)
        return self._return((vals, idx.to(torch.long)))

    def _sqeuclidean(self, X: Any, Y: Any | None = None):
        x, y = self._check_pair_inputs(X, Y)
        return pairwise_sqeuclidean(x, y)
