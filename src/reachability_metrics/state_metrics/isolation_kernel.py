"""Soft Isolation Kernel state distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.base import TransformTensorMixin
from reachability_metrics.torch_utils import as_2d_tensor, block_slices, resolve_torch_device
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
        self.input_dim = int(input_dim)
        self.ensemble_size = int(ensemble_size)
        self.subsample_size = int(subsample_size)
        self.temperature = float(temperature)
        self.device = resolve_torch_device(device)
        self.random_state = int(random_state)
        torch = __import__("torch")
        self.anchors = torch.empty(
            self.ensemble_size * self.subsample_size,
            self.input_dim,
            dtype=torch.float32,
            device=self.device,
        )

    def fit(self, data: Any) -> "SoftIsolationKernel":
        """Sample anchors from a training state pool."""
        torch = __import__("torch")
        values = as_2d_tensor(data, dtype=torch.float32, device=self.device, name="data")
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

        values = as_2d_tensor(x, dtype=torch.float32, device=self.device, name="x")
        dist = torch.cdist(values, self.anchors, p=2).view(values.shape[0], self.ensemble_size, self.subsample_size)
        assign = F.softmax(-dist / max(self.temperature, 1e-8), dim=-1)
        return assign.reshape(values.shape[0], self.ensemble_size * self.subsample_size)

    def __call__(self, x: Any) -> Any:
        return self.compute_ik_map(x)

    def kernel_mean(self, data: Any) -> Any:
        """Mean raw feature map over a state set."""
        return self.compute_ik_map(data).mean(dim=0)


class IsolationKernelDistance(TransformTensorMixin, StateMetric):
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
        dtype: str = "float32",
        batch_size: int = 4096,
        block_size: int = 4096,
        feature_mode: str = "soft",
        random_state: int = 0,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            block_size=block_size,
            return_numpy=return_numpy,
            output_format=output_format,
        )
        self.ensemble_size = ensemble_size
        self.subsample_size = subsample_size
        self.temperature = temperature
        self.feature_mode = feature_mode
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "IsolationKernelDistance":
        x = as_2d_tensor(X, dtype=self._dtype(), device=self._device(), name="X")
        if str(self.feature_mode).lower() != "soft":
            raise ValueError("Only feature_mode='soft' is implemented")
        self.n_features_in_ = int(x.shape[1])
        self.kernel_ = SoftIsolationKernel(
            input_dim=self.n_features_in_,
            ensemble_size=self.ensemble_size,
            subsample_size=self.subsample_size,
            temperature=self.temperature,
            device=x.device,
            random_state=self.random_state,
        ).fit(x)
        self.X_fit_ = x
        return self

    def transform_tensor(self, X: Any, *, normalize: bool = False):
        """Encode states into IK features.

        If ``normalize=True`` features are divided by ``sqrt(ensemble_size)`` so
        a dot product equals the similarity directly.
        """
        if not hasattr(self, "kernel_"):
            raise RuntimeError("IsolationKernelDistance must be fitted")
        torch = __import__("torch")
        x = as_2d_tensor(X, dtype=self._dtype(), device=self._device(), name="X")
        chunks = []
        with torch.no_grad():
            for start, end in block_slices(x.shape[0], int(self.batch_size)):
                feat = self.kernel_.compute_ik_map(x[start:end])
                if normalize:
                    feat = feat / (float(self.ensemble_size) ** 0.5)
                chunks.append(feat.to(dtype=self._dtype()))
        return torch.cat(chunks, dim=0)

    def pairwise_similarity_tensor(self, X: Any, Y: Any | None = None):
        if not hasattr(self, "kernel_"):
            self.fit(X)
        y_source = X if Y is None else Y
        fx = self.transform_tensor(X, normalize=False)
        fy = fx if Y is None else self.transform_tensor(y_source, normalize=False)
        torch = __import__("torch")
        sim = torch.empty((fx.shape[0], fy.shape[0]), dtype=fx.dtype, device=fx.device)
        for start, end in block_slices(fx.shape[0], int(self.block_size)):
            sim[start:end] = fx[start:end] @ fy.T / float(self.ensemble_size)
        return sim

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        return 1.0 - self.pairwise_similarity_tensor(X, Y)
