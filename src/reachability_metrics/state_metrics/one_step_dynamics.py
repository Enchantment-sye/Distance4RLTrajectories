"""One-step dynamics distance baselines."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import as_2d_tensor, as_trajectory_tensor_list, pairwise_sqeuclidean, require_torch
from .base import StateMetric


class OneStepDynamicsDistance(StateMetric):
    """Compare empirical one-step next-state distributions or local successor clouds."""

    def __init__(
        self,
        backend: str = "local_knn_nextstate",
        num_bins: int = 64,
        distance_metric: str = "jsd",
        local_knn_m: int = 20,
        alpha: float = 1e-3,
        min_count: int = 5,
        random_state: int = 0,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.backend = backend
        self.num_bins = num_bins
        self.distance_metric = distance_metric
        self.local_knn_m = local_knn_m
        self.alpha = alpha
        self.min_count = min_count
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "OneStepDynamicsDistance":
        torch = require_torch()
        if y is None:
            trajectories = as_trajectory_tensor_list(X, dtype=self._dtype(), device=self._device())
            current_blocks = [traj[:-1] for traj in trajectories if traj.shape[0] > 1]
            next_blocks = [traj[1:] for traj in trajectories if traj.shape[0] > 1]
            if not current_blocks:
                raise ValueError("at least one transition is required")
            current = torch.cat(current_blocks, dim=0)
            nxt = torch.cat(next_blocks, dim=0)
        else:
            current = as_2d_tensor(X, dtype=self._dtype(), device=self._device(), name="X")
            nxt = as_2d_tensor(y, dtype=self._dtype(), device=current.device, name="y")
        if current.ndim != 2 or nxt.ndim != 2 or current.shape != nxt.shape:
            raise ValueError("train states and next states must both have shape (N, D)")
        self.X_fit_ = current
        self.next_states_ = nxt
        self.n_features_in_ = int(current.shape[1])
        if str(self.backend).lower() != "local_knn_nextstate":
            self._fit_distribution_model(current, nxt)
        return self

    def _fit_distribution_model(self, states: Any, next_states: Any) -> None:
        torch = require_torch()
        n_bins = min(max(int(self.num_bins), 1), max(int(states.shape[0]), 1))
        if states.shape[0] == 0:
            self.transition_probabilities_ = torch.ones((1, 1), dtype=self._dtype(), device=self._device())
            return
        key = str(self.backend).lower()
        if key == "grid" and states.shape[1] == 2:
            side = max(int(n_bins**0.5 + 0.999999), 1)
            mins = torch.min(states, dim=0).values
            maxs = torch.max(states, dim=0).values
            maxs = torch.where(torch.isclose(maxs, mins), mins + 1.0, maxs)
            self.grid_edges_ = (
                torch.linspace(float(mins[0]), float(maxs[0]), side + 1, dtype=states.dtype, device=states.device),
                torch.linspace(float(mins[1]), float(maxs[1]), side + 1, dtype=states.dtype, device=states.device),
            )
            cur = self._assign_grid(states)
            nxt = self._assign_grid(next_states)
            n_bins = side * side
            self.centers_ = None
        else:
            self.centers_ = self._fit_kmeans(states, n_bins)
            cur = self._assign_kmeans(states)
            nxt = self._assign_kmeans(next_states)
        counts = torch.zeros((n_bins, n_bins), dtype=states.dtype, device=states.device)
        counts.index_put_((cur, nxt), torch.ones_like(cur, dtype=states.dtype), accumulate=True)
        alpha = max(float(self.alpha), 1e-12)
        row_counts = counts.sum(dim=1)
        global_dist = (counts.sum(dim=0) + alpha) / (counts.sum() + alpha * n_bins)
        probs = (counts + alpha) / torch.clamp(row_counts[:, None] + alpha * n_bins, min=1e-12)
        probs[row_counts < int(self.min_count)] = global_dist[None, :]
        self.transition_probabilities_ = probs
        self.row_counts_ = row_counts

    def _fit_kmeans(self, x: Any, n_bins: int):
        torch = require_torch()
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(self.random_state))
        idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:n_bins]
        centers = x[idx].clone()
        if centers.shape[0] < n_bins:
            centers = x[torch.randint(0, x.shape[0], (n_bins,), generator=gen, device=x.device)].clone()
        for _ in range(20):
            labels = torch.argmin(pairwise_sqeuclidean(x, centers), dim=1)
            new_centers = centers.clone()
            for cid in range(n_bins):
                mask = labels == cid
                if torch.any(mask):
                    new_centers[cid] = x[mask].mean(dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        return centers

    def _assign_grid(self, x: Any):
        torch = require_torch()
        ex, ey = self.grid_edges_
        xb = torch.bucketize(x[:, 0].contiguous(), ex[1:-1]).clamp(0, len(ex) - 2)
        yb = torch.bucketize(x[:, 1].contiguous(), ey[1:-1]).clamp(0, len(ey) - 2)
        return (xb * (len(ey) - 1) + yb).to(torch.long)

    def _assign_kmeans(self, x: Any):
        torch = require_torch()
        return torch.argmin(pairwise_sqeuclidean(x, self.centers_), dim=1).to(torch.long)

    def _assign_bins(self, x: Any):
        if hasattr(self, "centers_") and self.centers_ is not None:
            return self._assign_kmeans(x)
        if hasattr(self, "grid_edges_"):
            return self._assign_grid(x)
        return require_torch().zeros(x.shape[0], dtype=require_torch().long, device=x.device)

    def _distribution_distance(self, p: Any, q: Any):
        torch = require_torch()
        if str(self.distance_metric).lower() in {"l1", "manhattan"}:
            return torch.sum(torch.abs(p[:, None, :] - q[None, :, :]), dim=-1)
        eps = 1e-12
        ps = torch.clamp(p, min=eps)
        qs = torch.clamp(q, min=eps)
        m = 0.5 * (ps[:, None, :] + qs[None, :, :])
        kl_p = torch.sum(ps[:, None, :] * torch.log(ps[:, None, :] / m), dim=-1)
        kl_q = torch.sum(qs[None, :, :] * torch.log(qs[None, :, :] / m), dim=-1)
        return torch.sqrt(torch.clamp(0.5 * (kl_p + kl_q), min=0.0))

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        if not hasattr(self, "X_fit_"):
            raise RuntimeError("OneStepDynamicsDistance must be fitted")
        x, y = self._check_pair_inputs(X, Y)
        if str(self.backend).lower() == "local_knn_nextstate":
            k = min(max(int(self.local_knn_m), 1), self.X_fit_.shape[0])
            nn = torch.topk(pairwise_sqeuclidean(x, self.X_fit_), k=k, largest=False, dim=1).indices
            clouds = self.next_states_[nn]
            out = torch.empty((x.shape[0], y.shape[0]), dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                out[i] = torch.min(torch.sqrt(pairwise_sqeuclidean(clouds[i], y)), dim=0).values
            return out
        bx = self._assign_bins(x)
        by = self._assign_bins(y)
        ux, ix = torch.unique(bx, return_inverse=True)
        uy, iy = torch.unique(by, return_inverse=True)
        lookup = self._distribution_distance(self.transition_probabilities_[ux], self.transition_probabilities_[uy])
        return lookup[ix[:, None], iy[None, :]]

    def pairwise_similarity_tensor(self, X: Any, Y: Any | None = None):
        return -self.pairwise_distance_tensor(X, Y)
