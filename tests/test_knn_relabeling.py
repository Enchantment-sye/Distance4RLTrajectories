from __future__ import annotations

from reachability_metrics.experiments.knn_relabeling import KNNRelabelConfig, run_relabel_benchmark


def test_knn_relabeling_smoke(tmp_path) -> None:
    result = run_relabel_benchmark(
        KNNRelabelConfig(
            datasets=["synthetic/relabel"],
            output_dir=str(tmp_path),
            num_anchors=4,
            num_candidates=16,
            top_k=4,
            fit_pool_size=64,
            ik_ensemble_size=4,
            ik_subsample_size=3,
            ik_device="cpu",
        )
    )
    assert result["summary_rows"]
    assert (tmp_path / "tables" / "summary.csv").exists()

