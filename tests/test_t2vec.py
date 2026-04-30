from __future__ import annotations

import numpy as np
import pytest
import torch

from reachability_metrics.trajectory_metrics import T2VecDistance


def test_t2vec_training_save_load_and_similarity(tmp_path) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    base_t = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    train = [
        np.stack([base_t, base_t + 0.01 * rng.normal(size=base_t.shape)], axis=1).astype(np.float32)
        for _ in range(8)
    ]
    noise = (rng.normal(size=(10, 2)) + 4.0).astype(np.float32)
    path = tmp_path / "t2vec.pt"
    metric = T2VecDistance(
        model_path=str(path),
        train_if_missing=True,
        normalize=True,
        embedding_dim=8,
        hidden_size=12,
        num_layers=1,
        batch_size=4,
        epochs=1,
        device="cpu",
        random_state=0,
    ).fit(train)
    emb = metric.transform(train[:3])
    assert emb.shape == (3, 8)
    assert path.exists()

    loaded = T2VecDistance(model_path=str(path), device="cpu").load(str(path))
    emb_loaded = loaded.transform(train[:3])
    torch.testing.assert_close(emb, emb_loaded, atol=1e-5, rtol=1e-5)

    d_train = metric.pairwise_distance([train[0]], [train[1]])[0, 0]
    d_noise = metric.pairwise_distance([train[0]], [noise])[0, 0]
    assert bool(d_train < d_noise)
