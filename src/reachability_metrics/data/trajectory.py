"""Torch-first trajectory containers for offline MDP data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from reachability_metrics.torch_utils import require_torch


def _tensor(values, *, dtype=None):
    torch = require_torch()
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype) if dtype is not None else values
    return torch.as_tensor(values, dtype=dtype)


@dataclass
class Trajectory:
    """A single offline MDP trajectory stored as torch tensors."""

    states: object
    actions: Optional[object] = None
    rewards: Optional[object] = None
    dones: Optional[object] = None
    timesteps: Optional[object] = None
    episode_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        torch = require_torch()
        self.states = _tensor(self.states, dtype=torch.float32)
        if self.states.ndim != 2:
            raise ValueError(f"Trajectory.states must have shape (T, D), got {tuple(self.states.shape)}")
        length = int(self.states.shape[0])
        device = self.states.device
        if self.timesteps is None:
            self.timesteps = torch.arange(length, dtype=torch.long, device=device)
        else:
            self.timesteps = _tensor(self.timesteps, dtype=torch.long).to(device=device)
            if int(self.timesteps.shape[0]) != length:
                raise ValueError("timesteps length must match states length")
        if self.actions is not None:
            self.actions = _tensor(self.actions, dtype=torch.float32).to(device=device)
            if int(self.actions.shape[0]) not in {length, max(length - 1, 0)}:
                raise ValueError(f"actions length must be T or T-1, got {self.actions.shape[0]} for T={length}")
        if self.rewards is not None:
            self.rewards = _tensor(self.rewards, dtype=torch.float32).to(device=device)
            if int(self.rewards.shape[0]) not in {length, max(length - 1, 0)}:
                raise ValueError(f"rewards length must be T or T-1, got {self.rewards.shape[0]} for T={length}")
        if self.dones is not None:
            self.dones = _tensor(self.dones, dtype=torch.bool).to(device=device)
            if int(self.dones.shape[0]) not in {length, max(length - 1, 0)}:
                raise ValueError(f"dones length must be T or T-1, got {self.dones.shape[0]} for T={length}")


@dataclass
class TrajectoryDataset:
    """A collection of torch-first offline MDP trajectories."""

    trajectories: list[Trajectory]

    def __post_init__(self) -> None:
        self.trajectories = [
            traj if isinstance(traj, Trajectory) else Trajectory(traj)
            for traj in self.trajectories
        ]
        if not self.trajectories:
            raise ValueError("TrajectoryDataset requires at least one trajectory")

    @classmethod
    def from_arrays(cls, trajectories: list[object] | object) -> "TrajectoryDataset":
        """Build a dataset from a list of arrays/tensors or a single array/tensor."""
        torch = require_torch()
        if isinstance(trajectories, torch.Tensor):
            if trajectories.ndim == 3:
                arrays = [trajectories[i] for i in range(trajectories.shape[0])]
            elif trajectories.ndim == 2:
                arrays = [trajectories]
            else:
                raise ValueError(f"trajectory tensor must be 2D or 3D, got {tuple(trajectories.shape)}")
        elif isinstance(trajectories, np.ndarray) and trajectories.ndim == 3:
            arrays = [trajectories[i] for i in range(trajectories.shape[0])]
        elif isinstance(trajectories, np.ndarray) and trajectories.ndim == 2:
            arrays = [trajectories]
        else:
            arrays = list(trajectories)  # type: ignore[arg-type]
        return cls([Trajectory(arr, episode_id=i) for i, arr in enumerate(arrays)])

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

    def states(self):
        """Return all states stacked into ``(N, D)`` as a torch tensor."""
        return require_torch().cat([traj.states for traj in self.trajectories], dim=0).to(torch_float32())

    def episode_ids(self):
        """Return the episode id for each stacked state as a torch tensor."""
        torch = require_torch()
        blocks = []
        device = self.trajectories[0].states.device
        for idx, traj in enumerate(self.trajectories):
            episode_id = idx if traj.episode_id is None else int(traj.episode_id)
            blocks.append(torch.full((traj.states.shape[0],), episode_id, dtype=torch.long, device=device))
        return torch.cat(blocks, dim=0)

    def timesteps(self):
        """Return the timestep for each stacked state as a torch tensor."""
        return require_torch().cat([traj.timesteps.to(dtype=require_torch().long) for traj in self.trajectories], dim=0)

    def episode_offsets(self):
        """Return inclusive start offsets and final end offset as a torch tensor."""
        torch = require_torch()
        offsets = [0]
        for traj in self.trajectories:
            offsets.append(offsets[-1] + int(traj.states.shape[0]))
        return torch.as_tensor(offsets, dtype=torch.long, device=self.trajectories[0].states.device)

    def episode_lengths(self):
        """Return trajectory lengths as a torch tensor."""
        torch = require_torch()
        return torch.as_tensor(
            [int(traj.states.shape[0]) for traj in self.trajectories],
            dtype=torch.long,
            device=self.trajectories[0].states.device,
        )

    def transition_pairs(self):
        """Return stacked one-step transition state pairs as torch tensors."""
        torch = require_torch()
        states = []
        next_states = []
        for traj in self.trajectories:
            if traj.states.shape[0] < 2:
                continue
            states.append(traj.states[:-1])
            next_states.append(traj.states[1:])
        if not states:
            dim = int(self.trajectories[0].states.shape[1])
            empty = torch.empty((0, dim), dtype=torch.float32, device=self.trajectories[0].states.device)
            return empty, empty.clone()
        return torch.cat(states, dim=0).to(torch.float32), torch.cat(next_states, dim=0).to(torch.float32)

    def windows(self, horizon: int, include_current: bool = False) -> list[object]:
        """Return valid same-trajectory future windows as a list of tensors."""
        h = int(horizon)
        if h <= 0:
            raise ValueError("horizon must be positive")
        windows: list[object] = []
        start_offset = 0 if include_current else 1
        window_len = h + 1 if include_current else h
        for traj in self.trajectories:
            max_start = int(traj.states.shape[0]) - start_offset - h + 1
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


def torch_float32():
    """Small helper to avoid importing torch at module import time."""
    return require_torch().float32

