"""Standalone temporal feature helpers."""

from __future__ import annotations

import numpy as np


def sinusoidal_position_encoding(timesteps: np.ndarray, dim: int = 8) -> np.ndarray:
    """Transformer sinusoidal position encoding."""
    t = np.asarray(timesteps, dtype=np.float32).reshape(-1)
    d = int(dim) + (int(dim) % 2)
    frequencies = np.exp(np.arange(0, d, 2, dtype=np.float32) * (-np.log(10000.0) / d))
    angles = t[:, None] * frequencies[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)


def normalized_time(timesteps: np.ndarray, length: int) -> np.ndarray:
    """Return ``t / (T - 1)`` as a column vector."""
    return (np.asarray(timesteps, dtype=np.float32).reshape(-1, 1) / max(float(length - 1), 1.0)).astype(np.float32)

