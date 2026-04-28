"""Future-window helpers."""

from __future__ import annotations

import numpy as np

from .trajectory import TrajectoryDataset


def future_windows(
    dataset: TrajectoryDataset,
    horizon: int,
    *,
    include_current: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build valid future windows with global state indices and episode ids."""
    h = int(horizon)
    windows = []
    global_indices = []
    episode_ids = []
    offset = 0
    for episode_id, traj in enumerate(dataset.trajectories):
        start_offset = 0 if include_current else 1
        window_len = h + 1 if include_current else h
        max_start = traj.states.shape[0] - start_offset - h + 1
        for t in range(max(0, max_start)):
            windows.append(traj.states[t + start_offset : t + start_offset + window_len])
            global_indices.append(offset + t)
            episode_ids.append(episode_id)
        offset += traj.states.shape[0]
    if not windows:
        dim = dataset.trajectories[0].states.shape[1]
        return (
            np.empty((0, h, dim), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.stack(windows, axis=0).astype(np.float32),
        np.asarray(global_indices, dtype=np.int64),
        np.asarray(episode_ids, dtype=np.int64),
    )

