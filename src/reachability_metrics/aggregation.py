"""Reusable aggregation strategies for cross and set metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reachability_metrics.torch_utils import (
    as_tensor,
    cosine_distance_matrix,
    pairwise_euclidean,
    require_torch,
    torch_softmin,
)


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


def ensure_tensor(values: Any, *, dtype: Any | None = None, device: Any | None = None) -> Any:
    """Return a torch tensor while preserving tensor dtype/device when possible."""
    torch = require_torch()
    if isinstance(values, torch.Tensor):
        if dtype is not None or device is not None:
            return values.to(dtype=dtype or values.dtype, device=device or values.device)
        return values
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    return as_tensor(values, **kwargs)


def pairwise_distance_tensor(
    metric: Any,
    A: Any,
    B: Any | None = None,
    *,
    dtype: Any | None = None,
    device: Any | None = None,
) -> Any:
    """Call a metric's tensor distance path when available."""
    if hasattr(metric, "pairwise_distance_tensor"):
        return ensure_tensor(metric.pairwise_distance_tensor(A, B), dtype=dtype, device=device)
    return ensure_tensor(metric.pairwise_distance(A, B), dtype=dtype, device=device)


def transform_tensor(metric: Any, values: Any) -> Any:
    """Call a metric's tensor transform path when available."""
    if hasattr(metric, "transform_tensor"):
        return ensure_tensor(metric.transform_tensor(values))
    return ensure_tensor(metric.transform(values))


def aggregate_groupwise_distances(
    metric: Any,
    queries: Any,
    groups: list[Any],
    aggregation: AggregationStrategy,
    *,
    reduce_dim: int = 1,
    stack_dim: int = 1,
    dtype: Any | None = None,
    device: Any | None = None,
) -> Any:
    """Reduce query-to-member distances for each group and stack by group order."""
    torch = require_torch()
    columns = []
    for group in groups:
        distances = pairwise_distance_tensor(metric, queries, group, dtype=dtype, device=device)
        columns.append(aggregation.reduce(distances, dim=reduce_dim))
    return torch.stack(columns, dim=stack_dim)


def transform_groups_tensor(metric: Any, groups: list[Any]) -> list[Any]:
    """Transform each group independently, preserving input group order."""
    return [transform_tensor(metric, group) for group in groups]


def mean_group_embeddings(embeddings_by_group: list[Any]) -> Any:
    """Mean-pool per-item embeddings into one embedding per group."""
    torch = require_torch()
    return torch.stack([emb.mean(dim=0) for emb in embeddings_by_group], dim=0)


def pairwise_embedding_distance_tensor(a: Any, b: Any, distance_mode: str = "rkhs_norm") -> Any:
    """Pairwise distance between embedding matrices."""
    if str(distance_mode).lower() == "cosine":
        return cosine_distance_matrix(a, b)
    return pairwise_euclidean(a, b)
