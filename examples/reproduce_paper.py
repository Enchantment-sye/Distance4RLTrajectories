"""Small reproducibility driver for the main experiment CLIs and paper tables."""

from __future__ import annotations

from reachability_metrics.experiments.knn_planning import KNNPlanningEvalConfig, run_knn_planning_eval
from reachability_metrics.experiments.knn_relabeling import KNNRelabelConfig, run_relabel_benchmark
from reachability_metrics.experiments.paper_reproduction import PaperReproductionConfig, run_paper_reproduction
from reachability_metrics.experiments.successor_distance import SuccessorDistanceConfig, run_successor_distance


DATASETS = [
    "D4RL/pointmaze/umaze-v2",
    "D4RL/pointmaze/large-v2",
    "D4RL/antmaze/umaze-diverse-v1",
]


def main() -> None:
    run_successor_distance(
        SuccessorDistanceConfig(
            datasets=DATASETS,
            horizon_values=[10, 20, 50],
            output_dir="outputs/successor_distance",
        )
    )
    run_relabel_benchmark(
        KNNRelabelConfig(
            datasets=DATASETS,
            output_dir="outputs/knn_relabeling",
            num_anchors=200,
            num_candidates=1000,
            top_k=20,
        )
    )
    run_knn_planning_eval(
        KNNPlanningEvalConfig(
            datasets=DATASETS,
            output_dir="outputs/knn_planning",
            retrieval_top_k=20,
            num_queries=200,
        )
    )
    run_paper_reproduction(
        PaperReproductionConfig(
            legacy_outputs_dir="/share/shangyy/codes/metra/outputs",
            output_dir="outputs/paper_reproduction",
            include_figures=True,
            verify_paper_values=True,
        )
    )


if __name__ == "__main__":
    main()
