"""Kernel mean embedding for trajectories."""

from __future__ import annotations

from typing import Any

from reachability_metrics.base import TensorOutputMixin
from reachability_metrics.state_metrics import StateMetric
from reachability_metrics.trajectory_metrics.kme_strategies import (
    EmbeddingDistanceStrategy,
    MeanEmbeddingReducer,
    build_feature_map_strategy,
)
from reachability_metrics.torch_utils import (
    as_2d_tensor,
    as_trajectory_tensor,
    as_trajectory_tensor_list,
    require_torch,
    resolve_torch_device,
    torch_dtype,
)


class KernelMeanEmbedding(TensorOutputMixin):
    """Represent a trajectory by the mean feature map of its states."""

    def __init__(
        self,
        base_kernel: StateMetric,
        normalize: bool = False,
        distance_mode: str = "rkhs_norm",
        feature_approximation: str = "exact",
        num_landmarks: int = 512,
        landmark_strategy: str = "random",
        eps: float = 1e-6,
        random_state: int = 0,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.base_kernel = base_kernel
        self.normalize = normalize
        self.distance_mode = distance_mode
        self.feature_approximation = feature_approximation
        self.num_landmarks = num_landmarks
        self.landmark_strategy = landmark_strategy
        self.eps = eps
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def _device(self):
        return resolve_torch_device(getattr(self, "device", "auto"))

    def _dtype(self):
        return torch_dtype(getattr(self, "dtype", "float32"))

    def fit(self, X: Any) -> "KernelMeanEmbedding":
        """Fit the base kernel from all states in trajectories or a state matrix."""
        torch = require_torch()
        try:
            states = as_2d_tensor(X, dtype=self._dtype(), device=self._device(), name="states")
        except Exception:
            states = torch.cat(as_trajectory_tensor_list(X, dtype=self._dtype(), device=self._device()), dim=0)
        self.base_kernel.fit(states)
        self.X_fit_ = states
        self.feature_map_ = build_feature_map_strategy(
            self.base_kernel,
            distance_mode=self.distance_mode,
            feature_approximation=self.feature_approximation,
            num_landmarks=self.num_landmarks,
            landmark_strategy=self.landmark_strategy,
            eps=self.eps,
            random_state=self.random_state,
            states=states,
        )
        self.reducer_ = MeanEmbeddingReducer()
        self.embedding_distance_ = EmbeddingDistanceStrategy(
            distance_mode=self.distance_mode,
            partition_scale=float(getattr(self.base_kernel, "ensemble_size", 1.0)),
        )
        return self

    def _state_features_tensor(self, states: Any):
        values = as_2d_tensor(states, dtype=self._dtype(), device=self._device(), name="states")
        return self.feature_map_.transform(self.base_kernel, values)

    def _normalize_embeddings(self, emb: Any):
        torch = require_torch()
        if not bool(self.normalize):
            return emb
        return emb / torch.linalg.norm(emb, dim=1, keepdim=True).clamp_min(1e-12)

    def transform_trajectory_tensor(self, traj: Any):
        """Return a mean embedding for one trajectory."""
        states = as_trajectory_tensor(traj, dtype=self._dtype(), device=self._device())
        return self._state_features_tensor(states).mean(dim=0)

    def transform_trajectory(self, traj: Any):
        return self._return(self.transform_trajectory_tensor(traj))

    def transform_tensor(self, trajectories: Any):
        """Return mean embeddings for multiple trajectories."""
        torch = require_torch()
        trajs = as_trajectory_tensor_list(trajectories, dtype=self._dtype(), device=self._device())
        lengths = [int(traj.shape[0]) for traj in trajs]
        if not lengths:
            raise ValueError("at least one trajectory is required")
        states = torch.cat(trajs, dim=0)
        features = self._state_features_tensor(states)
        emb = self.reducer_.transform(features, lengths)
        return self._normalize_embeddings(emb)

    def transform(self, trajectories: Any):
        return self._return(self.transform_tensor(trajectories))

    def pairwise_kernel_tensor(self, trajectories_a: Any, trajectories_b: Any | None = None):
        """Pairwise trajectory kernel from mean embeddings."""
        a = self.transform_tensor(trajectories_a)
        b = a if trajectories_b is None else self.transform_tensor(trajectories_b)
        return self.embedding_distance_.kernel(a, b)

    def pairwise_kernel(self, trajectories_a: Any, trajectories_b: Any | None = None):
        return self._return(self.pairwise_kernel_tensor(trajectories_a, trajectories_b))

    def pairwise_distance_tensor(self, trajectories_a: Any, trajectories_b: Any | None = None):
        """Kernel-induced trajectory distribution distance."""
        a = self.transform_tensor(trajectories_a)
        b = a if trajectories_b is None else self.transform_tensor(trajectories_b)
        return self.embedding_distance_.distance(a, b)

    def pairwise_distance(self, trajectories_a: Any, trajectories_b: Any | None = None):
        return self._return(self.pairwise_distance_tensor(trajectories_a, trajectories_b))

    def pairwise_similarity_tensor(self, trajectories_a: Any, trajectories_b: Any | None = None):
        a = self.transform_tensor(trajectories_a)
        b = a if trajectories_b is None else self.transform_tensor(trajectories_b)
        return self.embedding_distance_.similarity(a, b)

    def pairwise_similarity(self, trajectories_a: Any, trajectories_b: Any | None = None):
        return self._return(self.pairwise_similarity_tensor(trajectories_a, trajectories_b))
