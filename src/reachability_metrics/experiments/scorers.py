"""Registry-backed experiment scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from reachability_metrics.state_metrics.registry import build_state_metric
from reachability_metrics.torch_utils import cpu_numpy


@dataclass(frozen=True)
class StateScoringContext:
    """Inputs needed by state-pair experiment scorers."""

    fit: np.ndarray
    states: np.ndarray
    anchors: np.ndarray
    candidates: np.ndarray
    cfg: Any | None = None
    dataset: Any | None = None
    episode_ids: np.ndarray | None = None
    timesteps: np.ndarray | None = None

    @property
    def x(self) -> np.ndarray:
        return self.states[self.anchors]

    @property
    def y(self) -> np.ndarray:
        return self.states[self.candidates]


class ExperimentScorer:
    """Callable scorer strategy."""

    def __init__(self, method: str, score_fn: Callable[[StateScoringContext], np.ndarray]) -> None:
        self.method = method
        self.score_fn = score_fn

    def score(self, context: StateScoringContext) -> np.ndarray:
        return self.score_fn(context)


def _ik_kwargs(cfg: Any | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    return {
        "ensemble_size": getattr(cfg, "ik_ensemble_size", 100),
        "subsample_size": getattr(cfg, "ik_subsample_size", 32),
        "temperature": getattr(cfg, "ik_temperature", 0.01),
        "device": getattr(cfg, "ik_device", "auto"),
        "batch_size": getattr(cfg, "ik_batch_size", 4096),
        "random_state": getattr(cfg, "seed", 0),
    }


def _temporal_scores(context: StateScoringContext) -> np.ndarray:
    if context.episode_ids is not None and context.timesteps is not None:
        episode_ids = context.episode_ids
        timesteps = context.timesteps
    elif context.dataset is not None:
        episode_ids = cpu_numpy(context.dataset.episode_ids())
        timesteps = cpu_numpy(context.dataset.timesteps())
    else:
        raise ValueError("temporal scorer requires episode_ids/timesteps or a dataset")
    same = episode_ids[context.anchors][:, None] == episode_ids[context.candidates][None, :]
    delta = timesteps[context.candidates][None, :] - timesteps[context.anchors][:, None]
    valid = same & (delta > 0)
    scores = np.zeros((context.anchors.shape[0], context.candidates.shape[0]), dtype=np.float32)
    scores[valid] = 1.0 / (1.0 + delta[valid].astype(np.float32))
    return scores


def _one_step_scores(context: StateScoringContext) -> np.ndarray:
    if context.dataset is None:
        raise ValueError("one-step dynamics scorer requires a dataset")
    s0, s1 = context.dataset.transition_pairs()
    if s0.shape[0] == 0:
        metric = build_state_metric("euclidean").fit(context.fit)
        return -cpu_numpy(metric.pairwise_distance(context.x, context.y))
    metric = build_state_metric("one_step_dynamics").fit(cpu_numpy(s0), cpu_numpy(s1))
    return -cpu_numpy(metric.pairwise_distance(context.x, context.y))


def _metric_scores(method: str, context: StateScoringContext) -> np.ndarray:
    key = method.lower()
    if key in {"ik", "isolation_kernel"}:
        metric = build_state_metric("ik", **_ik_kwargs(context.cfg)).fit(context.fit)
        return cpu_numpy(metric.pairwise_similarity(context.x, context.y))
    if key in {"gaussian", "adaptive_gaussian"}:
        metric = build_state_metric(key).fit(context.fit)
        return cpu_numpy(metric.pairwise_similarity(context.x, context.y))
    metric = build_state_metric(key).fit(context.fit)
    return -cpu_numpy(metric.pairwise_distance(context.x, context.y))


def build_experiment_scorer(method: str, cfg: Any | None = None) -> ExperimentScorer:
    """Build a scorer by method key."""
    key = method.lower()
    aliases = {
        "temporal_distance": "temporal",
        "dyn_1": "one_step_dynamics",
    }
    key = aliases.get(key, key)
    if key == "temporal":
        return ExperimentScorer(method, _temporal_scores)
    if key == "one_step_dynamics":
        return ExperimentScorer(method, _one_step_scores)
    return ExperimentScorer(method, lambda context: _metric_scores(key, context))

