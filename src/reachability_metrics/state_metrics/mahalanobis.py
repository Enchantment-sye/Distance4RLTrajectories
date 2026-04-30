"""Mahalanobis state distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_sqeuclidean, require_torch
from .base import StateMetric


class MahalanobisDistance(StateMetric):
    """Global covariance-normalized distance implemented with torch.linalg."""

    def __init__(
        self,
        covariance_estimator: str = "ledoitwolf",
        implementation: str = "whitening",
        eps: float = 1e-6,
        device: str = "auto",
        dtype: str = "float32",
        batch_size: int = 4096,
        block_size: int = 4096,
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
        self.covariance_estimator = covariance_estimator
        self.implementation = implementation
        self.eps = eps

    def fit(self, X: Any, y: Any = None) -> "MahalanobisDistance":
        torch = require_torch()
        super().fit(X, y)
        x = self.X_fit_
        dim = x.shape[1]
        eps = max(float(self.eps), 1e-12)
        if x.shape[0] < 2:
            cov = torch.eye(dim, dtype=x.dtype, device=x.device)
            estimator_name = "identity"
        else:
            centered = x - x.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / max(x.shape[0] - 1, 1)
            key = str(self.covariance_estimator).lower()
            if key in {"ledoitwolf", "lw"}:
                # Lightweight torch Ledoit-Wolf style shrinkage toward scaled identity.
                mu = torch.trace(cov) / dim
                shrinkage = min(1.0, dim / max(float(x.shape[0]), 1.0)) * 0.1
                cov = (1.0 - shrinkage) * cov + shrinkage * mu * torch.eye(dim, dtype=x.dtype, device=x.device)
                estimator_name = "torch_ledoitwolf_shrinkage"
            else:
                estimator_name = "torch_empirical"
        cov = 0.5 * (cov + cov.T) + eps * torch.eye(dim, dtype=x.dtype, device=x.device)
        vals, vecs = torch.linalg.eigh(cov)
        vals = vals.clamp_min(eps)
        self.covariance_matrix_ = cov
        self.precision_matrix_ = (vecs * (1.0 / vals)[None, :]) @ vecs.T
        self.whitening_matrix_ = (vecs * (1.0 / torch.sqrt(vals))[None, :]) @ vecs.T
        self.estimator_name_ = estimator_name
        return self

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        if not hasattr(self, "whitening_matrix_"):
            self.fit(X)
        x, y = self._check_pair_inputs(X, Y)
        if str(self.implementation).lower() == "precision":
            xp = x @ self.precision_matrix_
            yp = y @ self.precision_matrix_
            x_sq = torch.sum(xp * x, dim=1, keepdim=True)
            y_sq = torch.sum(yp * y, dim=1, keepdim=True).T
            return torch.sqrt(torch.clamp(x_sq + y_sq - 2.0 * xp @ y.T, min=0.0))
        xw = x @ self.whitening_matrix_.T
        yw = y @ self.whitening_matrix_.T
        return torch.sqrt(pairwise_sqeuclidean(xw, yw).clamp_min(0.0))
