"""Soft Isolation Kernel state distance."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from reachability_metrics.utils import as_2d_array, block_slices, resolve_device
from .base import StateMetric


class SoftIsolationKernel:
    """Torch-backed soft-assignment Isolation Kernel.

    This is a small standalone equivalent of the original project's
    ``SoftIsolationKernel`` without any METRA dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        ensemble_size: int = 100,
        subsample_size: int = 32,
        temperature: float = 0.01,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        import torch

        self.input_dim = int(input_dim)
        self.ensemble_size = int(ensemble_size)
        self.subsample_size = int(subsample_size)
        self.temperature = float(temperature)
        self.device = resolve_device(device)
        self.random_state = int(random_state)
        self.anchors = torch.empty(
            self.ensemble_size * self.subsample_size,
            self.input_dim,
            dtype=torch.float32,
            device=self.device,
        )

    def fit(self, data: Any) -> "SoftIsolationKernel":
        """Sample anchors from a training state pool."""
        import torch

        values = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        if values.ndim != 2:
            raise ValueError(f"data must be 2D, got {tuple(values.shape)}")
        if values.shape[0] == 0:
            raise ValueError("SoftIsolationKernel requires at least one training state")
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.random_state)
        total = self.ensemble_size * self.subsample_size
        idx = torch.randint(0, values.shape[0], (total,), generator=gen, device=self.device)
        self.anchors = values[idx].clone()
        return self

    def compute_ik_map(self, x: Any) -> Any:
        """Return raw soft assignments with shape ``(N, E*S)``."""
        import torch
        import torch.nn.functional as F

        values = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if values.ndim != 2:
            raise ValueError(f"x must be 2D, got {tuple(values.shape)}")
        dist = torch.cdist(values, self.anchors, p=2).view(values.shape[0], self.ensemble_size, self.subsample_size)
        assign = F.softmax(-dist / max(self.temperature, 1e-8), dim=-1)
        return assign.reshape(values.shape[0], self.ensemble_size * self.subsample_size)

    def __call__(self, x: Any) -> Any:
        return self.compute_ik_map(x)

    def kernel_mean(self, data: Any) -> Any:
        """Mean raw feature map over a state set."""
        return self.compute_ik_map(data).mean(dim=0)


class IsolationKernelDistance(StateMetric):
    """Isolation Kernel-induced state dissimilarity.

    Similarity is ``<phi(x), phi(y)> / ensemble_size`` where each ensemble's
    soft assignment sums to one.
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        subsample_size: int = 32,
        temperature: float = 0.01,
        device: str = "auto",
        batch_size: int = 4096,
        block_size: int = 4096,
        feature_mode: str = "soft",
        random_state: int = 0,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.subsample_size = subsample_size
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.feature_mode = feature_mode
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "IsolationKernelDistance":
        x = as_2d_array(X, dtype=np.float32, name="X")
        if str(self.feature_mode).lower() != "soft":
            raise ValueError("Only feature_mode='soft' is implemented")
        self.n_features_in_ = int(x.shape[1])
        self.kernel_ = SoftIsolationKernel(
            input_dim=self.n_features_in_,
            ensemble_size=self.ensemble_size,
            subsample_size=self.subsample_size,
            temperature=self.temperature,
            device=self.device,
            random_state=self.random_state,
        ).fit(x)
        self.X_fit_ = x
        return self

    def transform(self, X: Any, *, normalize: bool = False) -> np.ndarray:
        """Encode states into IK features.

        If ``normalize=True`` features are divided by ``sqrt(ensemble_size)`` so
        a dot product equals the similarity directly.
        """
        if not hasattr(self, "kernel_"):
            raise RuntimeError("IsolationKernelDistance must be fitted")
        import torch

        x = as_2d_array(X, dtype=np.float32, name="X")
        chunks = []
        with torch.no_grad():
            for start, end in block_slices(x.shape[0], int(self.batch_size)):
                feat = self.kernel_.compute_ik_map(x[start:end])
                if normalize:
                    feat = feat / math.sqrt(float(self.ensemble_size))
                chunks.append(feat.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        if not hasattr(self, "kernel_"):
            self.fit(X)
        y_source = X if Y is None else Y
        fx = self.transform(X, normalize=False)
        fy = fx if Y is None else self.transform(y_source, normalize=False)
        sim = np.empty((fx.shape[0], fy.shape[0]), dtype=np.float32)
        for start, end in block_slices(fx.shape[0], int(self.block_size)):
            sim[start:end] = (fx[start:end] @ fy.T / float(self.ensemble_size)).astype(np.float32)
        return sim

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        return (1.0 - self.pairwise_similarity(X, Y)).astype(np.float32)

