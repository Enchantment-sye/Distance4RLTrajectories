"""Trajectory containers for offline MDP data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    """A single offline MDP trajectory."""

    states: np.ndarray
    actions: Optional[np.ndarray] = None
    rewards: Optional[np.ndarray] = None
    dones: Optional[np.ndarray] = None
    timesteps: Optional[np.ndarray] = None
    episode_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        if self.states.ndim != 2:
            raise ValueError(f"Trajectory.states must have shape (T, D), got {self.states.shape}")
        length = self.states.shape[0]
        if self.timesteps is None:
            self.timesteps = np.arange(length, dtype=np.int64)
        else:
            self.timesteps = np.asarray(self.timesteps, dtype=np.int64)
            if self.timesteps.shape[0] != length:
                raise ValueError("timesteps length must match states length")
        for name in ("actions", "rewards", "dones"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value)
                if arr.shape[0] not in {length, max(length - 1, 0)}:
                    raise ValueError(f"{name} length must be T or T-1, got {arr.shape[0]} for T={length}")
                setattr(self, name, arr)


@dataclass
class TrajectoryDataset:
    """A collection of offline MDP trajectories."""

    trajectories: list[Trajectory]

    def __post_init__(self) -> None:
        self.trajectories = [
            traj if isinstance(traj, Trajectory) else Trajectory(np.asarray(traj, dtype=np.float32))
            for traj in self.trajectories
        ]
        if not self.trajectories:
            raise ValueError("TrajectoryDataset requires at least one trajectory")

    @classmethod
    def from_arrays(cls, trajectories: list[np.ndarray] | np.ndarray) -> "TrajectoryDataset":
        """Build a dataset from a list of arrays or a single ndarray."""
        if isinstance(trajectories, np.ndarray) and trajectories.ndim == 3:
            arrays = [trajectories[i] for i in range(trajectories.shape[0])]
        elif isinstance(trajectories, np.ndarray) and trajectories.ndim == 2:
            arrays = [trajectories]
        else:
            arrays = list(trajectories)  # type: ignore[arg-type]
        return cls([Trajectory(np.asarray(arr, dtype=np.float32), episode_id=i) for i, arr in enumerate(arrays)])

    @classmethod
    def synthetic(
        cls,
        *,
        num_trajectories: int = 40,
        length: int = 32,
        dim: int = 2,
        seed: int = 0,
        drift_scale: float = 0.15,
        noise_scale: float = 0.05,
    ) -> "TrajectoryDataset":
        """Generate a deterministic smooth synthetic trajectory dataset."""
        rng = np.random.default_rng(seed)
        trajectories: list[Trajectory] = []
        for episode_id in range(int(num_trajectories)):
            phase = 2.0 * np.pi * (episode_id / max(num_trajectories, 1))
            t = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
            base = np.zeros((int(length), int(dim)), dtype=np.float32)
            base[:, 0] = np.cos(phase) + t * (1.0 + 0.3 * np.sin(phase))
            if dim > 1:
                base[:, 1] = np.sin(phase) + 0.4 * np.sin(2.0 * np.pi * t + phase)
            for d in range(2, dim):
                base[:, d] = (d + 1) * 0.05 * np.cos(2.0 * np.pi * t + phase)
            walk = np.cumsum(rng.normal(scale=drift_scale, size=(length, dim)), axis=0)
            states = base + 0.1 * walk + rng.normal(scale=noise_scale, size=(length, dim))
            trajectories.append(Trajectory(states.astype(np.float32), episode_id=episode_id))
        return cls(trajectories)

    def states(self) -> np.ndarray:
        """Return all states stacked into ``(N, D)``."""
        return np.concatenate([traj.states for traj in self.trajectories], axis=0).astype(np.float32)

    def episode_ids(self) -> np.ndarray:
        """Return the episode id for each stacked state."""
        blocks = []
        for idx, traj in enumerate(self.trajectories):
            episode_id = idx if traj.episode_id is None else int(traj.episode_id)
            blocks.append(np.full(traj.states.shape[0], episode_id, dtype=np.int64))
        return np.concatenate(blocks, axis=0)

    def timesteps(self) -> np.ndarray:
        """Return the timestep for each stacked state."""
        return np.concatenate([np.asarray(traj.timesteps, dtype=np.int64) for traj in self.trajectories], axis=0)

    def episode_offsets(self) -> np.ndarray:
        """Return inclusive start offsets and final end offset."""
        offsets = [0]
        for traj in self.trajectories:
            offsets.append(offsets[-1] + traj.states.shape[0])
        return np.asarray(offsets, dtype=np.int64)

    def episode_lengths(self) -> np.ndarray:
        """Return trajectory lengths."""
        return np.asarray([traj.states.shape[0] for traj in self.trajectories], dtype=np.int64)

    def transition_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stacked one-step transition state pairs."""
        states = []
        next_states = []
        for traj in self.trajectories:
            if traj.states.shape[0] < 2:
                continue
            states.append(traj.states[:-1])
            next_states.append(traj.states[1:])
        if not states:
            dim = self.trajectories[0].states.shape[1]
            empty = np.empty((0, dim), dtype=np.float32)
            return empty, empty.copy()
        return np.concatenate(states, axis=0).astype(np.float32), np.concatenate(next_states, axis=0).astype(np.float32)

    def windows(self, horizon: int, include_current: bool = False) -> list[np.ndarray]:
        """Return valid same-trajectory future windows."""
        h = int(horizon)
        if h <= 0:
            raise ValueError("horizon must be positive")
        windows: list[np.ndarray] = []
        start_offset = 0 if include_current else 1
        window_len = h + 1 if include_current else h
        for traj in self.trajectories:
            max_start = traj.states.shape[0] - start_offset - h + 1
            for t in range(max(0, max_start)):
                windows.append(traj.states[t + start_offset : t + start_offset + window_len])
        return windows

    def split_by_trajectory(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple["TrajectoryDataset", "TrajectoryDataset", "TrajectoryDataset"]:
        """Split by entire trajectories."""
        ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
        if not np.isclose(np.sum(ratios), 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.trajectories))
        counts = np.floor(ratios * len(order)).astype(int)
        while counts.sum() < len(order):
            counts[np.argmax(ratios * len(order) - counts)] += 1
        train_idx = order[: counts[0]]
        val_idx = order[counts[0] : counts[0] + counts[1]]
        test_idx = order[counts[0] + counts[1] :]
        return (
            TrajectoryDataset([self.trajectories[int(i)] for i in train_idx]),
            TrajectoryDataset([self.trajectories[int(i)] for i in val_idx]),
            TrajectoryDataset([self.trajectories[int(i)] for i in test_idx]),
        )

