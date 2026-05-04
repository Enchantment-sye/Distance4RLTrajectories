"""Strategies used by kernel mean trajectory embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reachability_metrics.state_metrics import GaussianKernelDistance, IsolationKernelDistance
from reachability_metrics.torch_utils import (
    as_2d_tensor,
    cosine_distance_matrix,
    pairwise_euclidean,
    require_torch,
)


def transform_embedding_pair(
    transform: Any,
    trajectories_a: Any,
    trajectories_b: Any | None = None,
) -> tuple[Any, Any]:
    """Transform one or two trajectory collections into a reusable embedding pair."""
    a = transform(trajectories_a)
    b = a if trajectories_b is None else transform(trajectories_b)
    return a, b


class FeatureMapStrategy:
    """State feature map used before reducing states into trajectory embeddings."""

    def fit(self, base_kernel: Any, states: Any) -> "FeatureMapStrategy":
        return self

    def transform(self, base_kernel: Any, states: Any) -> Any:
        raise NotImplementedError


@dataclass
class NystromFeatureMap(FeatureMapStrategy):
    """Exact or landmarked Nystrom feature map for kernel similarities."""

    feature_approximation: str = "exact"
    num_landmarks: int = 512
    landmark_strategy: str = "random"
    eps: float = 1e-6
    random_state: int = 0

    def fit(self, base_kernel: GaussianKernelDistance, states: Any) -> "NystromFeatureMap":
        torch = require_torch()
        n = states.shape[0]
        mode = str(self.feature_approximation).lower()
        if mode not in {"exact", "nystrom"}:
            raise ValueError("feature_approximation must be 'exact' or 'nystrom'")
        if mode == "nystrom" and n > int(self.num_landmarks):
            if str(self.landmark_strategy).lower() == "first":
                idx = torch.arange(int(self.num_landmarks), device=states.device)
            else:
                gen = torch.Generator(device=states.device)
                gen.manual_seed(int(self.random_state))
                idx = torch.randperm(n, generator=gen, device=states.device)[
                    : int(self.num_landmarks)
                ]
            landmarks = states[idx].clone()
        else:
            landmarks = states.clone()
        w = base_kernel.pairwise_similarity_tensor(landmarks, landmarks)
        eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)
        vals, vecs = torch.linalg.eigh(0.5 * (w + w.T) + float(self.eps) * eye)
        self.landmarks_ = landmarks
        self.inv_sqrt_ = (vecs * torch.rsqrt(vals.clamp_min(float(self.eps)))[None, :]) @ vecs.T
        return self

    def transform(self, base_kernel: GaussianKernelDistance, states: Any) -> Any:
        kxz = base_kernel.pairwise_similarity_tensor(states, self.landmarks_)
        return kxz @ self.inv_sqrt_


@dataclass
class TransformFeatureMap(FeatureMapStrategy):
    """Use a kernel's explicit transform when available."""

    normalize: bool = False

    def transform(self, base_kernel: Any, states: Any) -> Any:
        if isinstance(base_kernel, IsolationKernelDistance):
            return base_kernel.transform_tensor(states, normalize=self.normalize)
        return base_kernel.transform_tensor(states)


@dataclass
class PairwiseKernelFeatureMap(FeatureMapStrategy):
    """Fallback feature map made from similarities to fitted reference states."""

    reference: Any | None = None

    def fit(self, base_kernel: Any, states: Any) -> "PairwiseKernelFeatureMap":
        self.reference = states
        return self

    def transform(self, base_kernel: Any, states: Any) -> Any:
        return base_kernel.pairwise_similarity_tensor(states, self.reference)


class MeanEmbeddingReducer:
    """Reduce per-state features into per-trajectory mean embeddings."""

    def transform(self, features: Any, lengths: list[int]) -> Any:
        torch = require_torch()
        chunks = torch.split(features, lengths, dim=0)
        return torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)


@dataclass(frozen=True)
class EmbeddingDistanceStrategy:
    """Pairwise distance/similarity over trajectory embeddings."""

    distance_mode: str = "rkhs_norm"
    partition_scale: float = 1.0

    def kernel(self, a: Any, b: Any) -> Any:
        dot = a @ b.T
        mode = str(self.distance_mode).lower()
        if mode == "one_minus_partition_similarity":
            return dot / max(float(self.partition_scale), 1e-12)
        if mode == "cosine":
            return 1.0 - cosine_distance_matrix(a, b)
        return dot

    def distance(self, a: Any, b: Any) -> Any:
        mode = str(self.distance_mode).lower()
        if mode == "cosine":
            return cosine_distance_matrix(a, b)
        if mode == "one_minus_partition_similarity":
            return 1.0 - (a @ b.T) / max(float(self.partition_scale), 1e-12)
        if mode not in {"rkhs_norm", "euclidean"}:
            raise ValueError(f"Unsupported distance_mode: {self.distance_mode}")
        return pairwise_euclidean(a, b)

    def similarity(self, a: Any, b: Any) -> Any:
        mode = str(self.distance_mode).lower()
        if mode in {"cosine", "one_minus_partition_similarity"}:
            return 1.0 - self.distance(a, b)
        return -self.distance(a, b)


def build_feature_map_strategy(
    base_kernel: Any,
    *,
    distance_mode: str,
    feature_approximation: str,
    num_landmarks: int,
    landmark_strategy: str,
    eps: float,
    random_state: int,
    states: Any,
) -> FeatureMapStrategy:
    """Choose the appropriate feature-map strategy for a fitted base kernel."""
    values = as_2d_tensor(
        states,
        dtype=getattr(states, "dtype", "float32"),
        device=getattr(states, "device", "auto"),
    )
    if isinstance(base_kernel, GaussianKernelDistance):
        return NystromFeatureMap(
            feature_approximation=feature_approximation,
            num_landmarks=num_landmarks,
            landmark_strategy=landmark_strategy,
            eps=eps,
            random_state=random_state,
        ).fit(base_kernel, values)
    if isinstance(base_kernel, IsolationKernelDistance):
        normalize = str(distance_mode).lower() != "one_minus_partition_similarity"
        return TransformFeatureMap(normalize=normalize).fit(base_kernel, values)
    if hasattr(base_kernel, "transform_tensor"):
        return TransformFeatureMap().fit(base_kernel, values)
    return PairwiseKernelFeatureMap().fit(base_kernel, values)
