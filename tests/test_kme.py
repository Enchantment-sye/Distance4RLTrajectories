from __future__ import annotations

import numpy as np
import torch

from reachability_metrics.state_metrics import GaussianKernelDistance
from reachability_metrics.trajectory_metrics import KernelMeanEmbedding


def test_kernel_mean_embedding_pairwise() -> None:
    trajectories = [
        np.array([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [1.1, 1.0], [1.2, 1.0]], dtype=np.float32),
    ]
    kme = KernelMeanEmbedding(GaussianKernelDistance(sigma_value=1.0), normalize=True).fit(trajectories)
    emb = kme.transform(trajectories)
    dist = kme.pairwise_distance(trajectories)
    assert emb.shape[0] == 2
    assert dist.shape == (2, 2)
    torch.testing.assert_close(torch.diag(dist), torch.zeros(2), atol=1e-6, rtol=1e-6)
