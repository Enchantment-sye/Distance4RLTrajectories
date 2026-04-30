"""Factory registry for trajectory-set metrics."""

from __future__ import annotations

from .trajectory_set import build_set_metric

__all__ = ["build_set_metric"]
