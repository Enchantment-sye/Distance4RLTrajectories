"""Reusable aggregation strategies for cross and set metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reachability_metrics.torch_utils import require_torch, torch_softmin


@dataclass(frozen=True)
class AggregationStrategy:
    """Reduce a distance matrix along one axis."""

    mode: str = "min"
    softmin_tau: float = 1.0
    k: int = 3

    def reduce(self, values: Any, dim: int = 1) -> Any:
        torch = require_torch()
        mode = str(self.mode).lower()
        if mode == "min":
            return torch.min(values, dim=dim).values
        if mode == "mean":
            return torch.mean(values, dim=dim)
        if mode == "softmin":
            return torch_softmin(values, tau=float(self.softmin_tau), dim=dim)
        if mode in {"kmin_mean", "topk_mean"}:
            kk = min(max(int(self.k), 1), values.shape[dim])
            part = torch.topk(values, k=kk, largest=False, dim=dim).values
            return torch.mean(part, dim=dim)
        raise ValueError(f"Unsupported aggregation: {self.mode}")


def build_aggregation(
    aggregation: str | AggregationStrategy = "min",
    *,
    softmin_tau: float = 1.0,
    k: int = 3,
) -> AggregationStrategy:
    """Build or normalize an aggregation strategy."""
    if isinstance(aggregation, AggregationStrategy):
        return aggregation
    return AggregationStrategy(mode=str(aggregation), softmin_tau=softmin_tau, k=k)

