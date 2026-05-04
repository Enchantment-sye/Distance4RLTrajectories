"""Sampling and context helpers shared by experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reachability_metrics.data import load_dataset_or_synthetic
from reachability_metrics.experiments.scorers import StateScoringContext
from reachability_metrics.torch_utils import cpu_numpy


@dataclass(frozen=True)
class StateExperimentSample:
    dataset: Any
    states: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    episode_lengths: np.ndarray
    valid: np.ndarray
    anchors: np.ndarray
    candidates: np.ndarray
    fit: np.ndarray


def load_state_dataset_sample(
    dataset_id: str,
    cfg: Any,
    rng: np.random.Generator,
) -> StateExperimentSample:
    dataset = load_dataset_or_synthetic(
        dataset_id,
        minari_datasets_path=cfg.minari_datasets_path,
        use_achieved_goal=True,
        synthetic_seed=cfg.seed,
    )
    states = cpu_numpy(dataset.states())
    timesteps = cpu_numpy(dataset.timesteps())
    episode_ids = cpu_numpy(dataset.episode_ids())
    episode_lengths = cpu_numpy(dataset.episode_lengths())
    valid = valid_anchor_indices(timesteps, episode_ids, episode_lengths, cfg.horizon)
    anchors, candidates, fit = sample_state_pairs(
        rng,
        states,
        valid,
        num_anchors=cfg.num_anchors,
        num_candidates=cfg.num_candidates,
        fit_pool_size=cfg.fit_pool_size,
    )
    return StateExperimentSample(
        dataset=dataset,
        states=states,
        episode_ids=episode_ids,
        timesteps=timesteps,
        episode_lengths=episode_lengths,
        valid=valid,
        anchors=anchors,
        candidates=candidates,
        fit=fit,
    )


def valid_anchor_indices(
    timesteps: np.ndarray,
    episode_ids: np.ndarray,
    episode_lengths: np.ndarray,
    horizon: int,
) -> np.ndarray:
    valid = np.flatnonzero(timesteps < episode_lengths[episode_ids] - horizon - 1)
    if valid.size == 0:
        valid = np.arange(timesteps.shape[0])
    return valid


def sample_state_pairs(
    rng: np.random.Generator,
    states: np.ndarray,
    valid: np.ndarray,
    *,
    num_anchors: int,
    num_candidates: int,
    fit_pool_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = rng.choice(valid, size=min(num_anchors, valid.size), replace=False)
    candidates = rng.choice(
        states.shape[0],
        size=min(num_candidates, states.shape[0]),
        replace=False,
    )
    fit = states[
        rng.choice(states.shape[0], size=min(fit_pool_size, states.shape[0]), replace=False)
    ]
    return anchors, candidates, fit


def state_scoring_context(sample: StateExperimentSample, cfg: Any) -> StateScoringContext:
    return StateScoringContext(
        fit=sample.fit,
        states=sample.states,
        anchors=sample.anchors,
        candidates=sample.candidates,
        cfg=cfg,
        dataset=sample.dataset,
        episode_ids=sample.episode_ids,
        timesteps=sample.timesteps,
    )


def sample_planning_queries(rng: np.random.Generator, n: int, num_queries: int) -> np.ndarray:
    queries = rng.integers(0, n, size=(min(num_queries, max(n // 2, 1)), 2))
    queries = queries[queries[:, 0] != queries[:, 1]]
    if queries.size == 0:
        queries = np.asarray([[0, n - 1]], dtype=np.int64)
    return queries


def sample_successor_fit_windows(windows: np.ndarray, cfg: Any) -> list[np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    sample_count = min(windows.shape[0], max(1, int(cfg.fit_pool_size) // max(1, windows.shape[1])))
    sample_idx = rng.choice(windows.shape[0], size=sample_count, replace=False)
    return [x for x in windows[sample_idx]]


def sample_successor_eval_ids(
    rng: np.random.Generator,
    window_count: int,
    cfg: Any,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_count = min(int(cfg.eval_num_pairs), max(2, window_count * 4))
    first = rng.integers(0, window_count, size=pair_count)
    second = rng.integers(0, window_count, size=pair_count)
    query_ids = rng.choice(window_count, size=min(cfg.num_queries, window_count), replace=False)
    cand_ids = rng.choice(window_count, size=min(cfg.num_candidates, window_count), replace=False)
    return pair_count, first, second, query_ids, cand_ids
