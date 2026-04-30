"""Euclidean state distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_euclidean
from .base import StateMetric


class EuclideanDistance(StateMetric):
    """Plain Euclidean distance implemented with torch."""

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "float32",
        batch_size: int = 4096,
        block_size: int = 4096,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            block_size=block_size,
            return_numpy=return_numpy,
            output_format=output_format,
        )

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        x, y = self._check_pair_inputs(X, Y)
        return pairwise_euclidean(x, y)
