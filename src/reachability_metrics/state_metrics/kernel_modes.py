"""Helpers for kernel-induced state distance modes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from reachability_metrics.torch_utils import require_torch


def kernel_distance_from_similarity(
    similarity: Any,
    distance_mode: str,
    *,
    rkhs_modes: Iterable[str] = ("rkhs",),
    one_minus_modes: Iterable[str] = ("one_minus_kernel", "1-k"),
) -> Any:
    """Convert a kernel similarity matrix into a supported distance matrix."""
    torch = require_torch()
    mode = str(distance_mode).lower()
    if mode in set(rkhs_modes):
        return torch.sqrt(torch.clamp(2.0 - 2.0 * similarity, min=0.0))
    if mode in set(one_minus_modes):
        return 1.0 - similarity
    raise ValueError(f"Unsupported distance_mode: {distance_mode}")
