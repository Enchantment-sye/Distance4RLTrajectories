from __future__ import annotations

import numpy as np

from reachability_metrics.set_metrics import GDK2SetDistance, TrajectoryNoveltyScorer


def test_gdk2_set_distance_and_novelty() -> None:
    rng = np.random.default_rng(0)
    reference = [rng.normal(scale=0.05, size=(8, 2)).astype(np.float32) for _ in range(6)]
    shifted = [(rng.normal(scale=0.05, size=(8, 2)) + 5.0).astype(np.float32) for _ in range(3)]
    metric = GDK2SetDistance().fit([reference])
    d_ref = metric.novelty_score(reference[0])[0]
    d_noise = metric.novelty_score(shifted[0])[0]
    assert d_ref < d_noise

    scorer = TrajectoryNoveltyScorer(method="gdk2").fit(reference)
    assert scorer.score(reference[1])[0] < scorer.score(shifted[1])[0]

