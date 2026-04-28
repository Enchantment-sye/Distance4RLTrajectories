"""D4RL/Minari loading utilities with graceful synthetic fallback."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .trajectory import Trajectory, TrajectoryDataset


def _extract_observations(observations: Any, state_key: str = "observation") -> np.ndarray:
    if isinstance(observations, dict):
        if state_key in observations:
            return np.asarray(observations[state_key], dtype=np.float32)
        if "achieved_goal" in observations:
            return np.asarray(observations["achieved_goal"], dtype=np.float32)
        first_key = next(iter(observations))
        return np.asarray(observations[first_key], dtype=np.float32)
    return np.asarray(observations, dtype=np.float32)


def load_minari_dataset(
    dataset_id: str,
    *,
    minari_datasets_path: str | None = None,
    state_key: str = "observation",
    use_achieved_goal: bool = False,
) -> TrajectoryDataset:
    """Load a Minari dataset into :class:`TrajectoryDataset`."""
    try:
        import minari
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install reachability-metrics[d4rl] to load Minari datasets") from exc
    if minari_datasets_path:
        os.environ.setdefault("MINARI_DATASETS_PATH", minari_datasets_path)
    dataset = minari.load_dataset(dataset_id)
    trajectories: list[Trajectory] = []
    for episode_id in range(int(dataset.total_episodes)):
        episode = dataset[episode_id]
        observations = episode.observations
        key = "achieved_goal" if use_achieved_goal else state_key
        states = _extract_observations(observations, state_key=key)
        actions = np.asarray(episode.actions) if hasattr(episode, "actions") else None
        rewards = np.asarray(episode.rewards) if hasattr(episode, "rewards") else None
        dones = np.asarray(episode.terminations) if hasattr(episode, "terminations") else None
        trajectories.append(
            Trajectory(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                episode_id=episode_id,
                metadata={"dataset_id": dataset_id},
            )
        )
    return TrajectoryDataset(trajectories)


def load_dataset_or_synthetic(
    dataset_id: str,
    *,
    minari_datasets_path: str | None = None,
    state_key: str = "observation",
    use_achieved_goal: bool = False,
    synthetic_seed: int = 0,
    synthetic_num_trajectories: int = 50,
    synthetic_length: int = 32,
) -> TrajectoryDataset:
    """Load Minari/D4RL if available, otherwise return a deterministic synthetic dataset."""
    try:
        return load_minari_dataset(
            dataset_id,
            minari_datasets_path=minari_datasets_path,
            state_key=state_key,
            use_achieved_goal=use_achieved_goal,
        )
    except Exception:
        return TrajectoryDataset.synthetic(
            num_trajectories=synthetic_num_trajectories,
            length=synthetic_length,
            dim=2,
            seed=synthetic_seed,
        )

