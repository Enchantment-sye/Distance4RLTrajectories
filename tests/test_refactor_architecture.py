from __future__ import annotations

import numpy as np
import pytest
import torch

from reachability_metrics.aggregation import build_aggregation
from reachability_metrics.cross_metrics import StateToTrajectoryDistance
from reachability_metrics.experiments.reachability_alignment import ReachabilityAnalysisConfig, analyze_datasets
from reachability_metrics.experiments.scorers import StateScoringContext, build_experiment_scorer
from reachability_metrics.set_metrics import build_set_metric
from reachability_metrics.state_metrics import EuclideanDistance, build_state_metric
from reachability_metrics.trajectory_metrics import build_trajectory_metric


def test_output_format_adapter_keeps_return_numpy_compatibility() -> None:
    x = np.arange(12, dtype=np.float32).reshape(6, 2)

    torch_result = EuclideanDistance().fit(x).pairwise_distance(x[:2], x[2:4])
    numpy_result = EuclideanDistance(output_format="numpy").fit(x).pairwise_distance(x[:2], x[2:4])
    compat_result = EuclideanDistance(return_numpy=True).fit(x).pairwise_distance(x[:2], x[2:4])

    assert isinstance(torch_result, torch.Tensor)
    assert isinstance(numpy_result, np.ndarray)
    assert isinstance(compat_result, np.ndarray)
    np.testing.assert_allclose(numpy_result, compat_result)
    np.testing.assert_allclose(numpy_result, torch_result.detach().cpu().numpy())


def test_metric_registries_build_public_keys_and_reject_unknowns() -> None:
    assert build_state_metric("euclidean").__class__ is EuclideanDistance
    assert build_state_metric("ik", ensemble_size=4, subsample_size=2).__class__.__name__ == "IsolationKernelDistance"
    assert build_trajectory_metric("gdk").__class__.__name__ == "GDKTrajectoryDistance"
    assert build_set_metric("idk2", ensemble_size=4, subsample_size=2).__class__.__name__ == "TrajectorySetDistance"

    with pytest.raises(ValueError, match="Unknown state metric"):
        build_state_metric("missing")
    with pytest.raises(ValueError, match="Unknown trajectory metric"):
        build_trajectory_metric("missing")
    with pytest.raises(ValueError, match="Unknown set metric"):
        build_set_metric("missing")


def test_shared_aggregation_strategy_matches_cross_metric_results() -> None:
    values = torch.tensor([[1.0, 3.0, 2.0], [4.0, 2.0, 8.0]])
    torch.testing.assert_close(build_aggregation("min").reduce(values), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(build_aggregation("mean").reduce(values), torch.tensor([2.0, 14.0 / 3.0]))
    torch.testing.assert_close(build_aggregation("kmin_mean", k=2).reduce(values), torch.tensor([1.5, 3.0]))

    trajectories = [
        np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        np.array([[10.0, 0.0], [12.0, 0.0], [14.0, 0.0]], dtype=np.float32),
    ]
    states = np.array([[1.0, 0.0]], dtype=np.float32)
    metric = StateToTrajectoryDistance(EuclideanDistance(), aggregation="kmin_mean", k=2).fit(trajectories)
    d = metric.pairwise_distance(states, trajectories)
    torch.testing.assert_close(d[0], torch.tensor([1.0, 10.0]))


def test_experiment_scorer_registry_matches_metric_semantics() -> None:
    states = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    context = StateScoringContext(
        fit=states,
        states=states,
        anchors=np.array([0]),
        candidates=np.array([1, 2]),
    )
    scores = build_experiment_scorer("euclidean").score(context)
    np.testing.assert_allclose(scores, np.array([[-1.0, -3.0]], dtype=np.float32))


def test_alignment_experiment_smoke_uses_artifact_writer(tmp_path) -> None:
    result = analyze_datasets(
        ReachabilityAnalysisConfig(
            datasets=["synthetic/alignment"],
            output_dir=str(tmp_path),
            num_anchors=4,
            num_candidates=16,
            fit_pool_size=64,
            ik_ensemble_size=4,
            ik_subsample_size=3,
            ik_device="cpu",
        )
    )

    assert result["summary_rows"]
    assert (tmp_path / "tables" / "summary.csv").exists()
    assert (tmp_path / "report.md").exists()

