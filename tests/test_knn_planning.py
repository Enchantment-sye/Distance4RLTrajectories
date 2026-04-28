from __future__ import annotations

from reachability_metrics.experiments.knn_planning import KNNPlanningEvalConfig, run_knn_planning_eval


def test_knn_planning_smoke(tmp_path) -> None:
    result = run_knn_planning_eval(
        KNNPlanningEvalConfig(
            datasets=["synthetic/planning"],
            output_dir=str(tmp_path),
            num_queries=4,
            retrieval_top_k=4,
            ik_ensemble_size=4,
            ik_subsample_size=3,
            ik_device="cpu",
        )
    )
    assert result["summary_rows"]
    assert (tmp_path / "tables" / "per_dataset_metrics.csv").exists()

