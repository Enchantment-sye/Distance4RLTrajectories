"""Factory registry for state metrics."""

from __future__ import annotations

from typing import Any, Callable

from reachability_metrics.registry import MetricRegistry

from .adaptive_gaussian import AdaptiveGaussianDistance
from .euclidean import EuclideanDistance
from .gaussian import GaussianKernelDistance
from .isolation_kernel import IsolationKernelDistance
from .mahalanobis import MahalanobisDistance
from .one_step_dynamics import OneStepDynamicsDistance
from .temporal import TemporalDistance


STATE_METRIC_REGISTRY: dict[str, Callable[..., Any]] = {
    "euclidean": EuclideanDistance,
    "gaussian": GaussianKernelDistance,
    "gaussian_kernel": GaussianKernelDistance,
    "adaptive_gaussian": AdaptiveGaussianDistance,
    "mahalanobis": MahalanobisDistance,
    "ik": IsolationKernelDistance,
    "isolation_kernel": IsolationKernelDistance,
    "one_step_dynamics": OneStepDynamicsDistance,
    "dyn_1": OneStepDynamicsDistance,
    "temporal": TemporalDistance,
    "temporal_distance": TemporalDistance,
}

_STATE_METRIC_FACTORY = MetricRegistry("state metric", STATE_METRIC_REGISTRY)


def build_state_metric(method: str, **kwargs: Any) -> Any:
    """Construct a state metric from a public method key."""
    return _STATE_METRIC_FACTORY.build(method, **kwargs)
