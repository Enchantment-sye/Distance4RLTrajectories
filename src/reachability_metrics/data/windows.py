"""Future-window helpers."""

from __future__ import annotations

from .trajectory import TrajectoryDataset
from reachability_metrics.torch_utils import require_torch


def future_windows(
    dataset: TrajectoryDataset,
    horizon: int,
    *,
    include_current: bool = False,
):
    """Build valid future windows with global state indices and episode ids as tensors."""
    torch = require_torch()
    h = int(horizon)
    windows = []
    global_indices = []
    episode_ids = []
    offset = 0
    device = dataset.trajectories[0].states.device
    for episode_id, traj in enumerate(dataset.trajectories):
        start_offset = 0 if include_current else 1
        window_len = h + 1 if include_current else h
        max_start = int(traj.states.shape[0]) - start_offset - h + 1
        for t in range(max(0, max_start)):
            windows.append(traj.states[t + start_offset : t + start_offset + window_len])
            global_indices.append(offset + t)
            episode_ids.append(episode_id)
        offset += int(traj.states.shape[0])
    if not windows:
        dim = int(dataset.trajectories[0].states.shape[1])
        return (
            torch.empty((0, h, dim), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )
    return (
        torch.stack(windows, dim=0).to(torch.float32),
        torch.as_tensor(global_indices, dtype=torch.long, device=device),
        torch.as_tensor(episode_ids, dtype=torch.long, device=device),
    )

