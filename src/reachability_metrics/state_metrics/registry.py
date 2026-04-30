"""Factory registry for state metrics."""

from __future__ import annotations

from typing import Any, Callable

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


def build_state_metric(method: str, **kwargs: Any) -> Any:
    """Construct a state metric from a public method key."""
    key = str(method).lower()
    try:
        factory = STATE_METRIC_REGISTRY[key]
    except KeyError as exc:
        options = ", ".join(sorted(STATE_METRIC_REGISTRY))
        raise ValueError(f"Unknown state metric '{method}'. Available: {options}") from exc
    return factory(**kwargs)

