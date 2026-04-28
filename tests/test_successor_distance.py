from __future__ import annotations

from reachability_metrics.experiments.successor_distance import SuccessorDistanceConfig, run_successor_distance


def test_successor_distance_smoke(tmp_path) -> None:
    result = run_successor_distance(
        SuccessorDistanceConfig(
            datasets=["synthetic/successor"],
            output_dir=str(tmp_path),
            horizon_values=[3],
            eval_num_pairs=24,
            num_queries=3,
            num_candidates=8,
            ik_ensemble_sizes=(4,),
            ik_subsample_sizes=(3,),
            ik_temperatures=(0.05,),
            ik_device="cpu",
            recall_k_values=(2,),
        )
    )
    assert result["summary_rows"]
    assert (tmp_path / "tables" / "per_dataset_metrics.csv").exists()

