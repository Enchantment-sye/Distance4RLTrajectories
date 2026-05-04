"""CLI for offline kNN planning experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.cli._helpers import (
    add_dataset_output_args,
    add_seed_minari_args,
    add_state_ik_args,
    config_from_args,
    experiment_parser,
)
from reachability_metrics.experiments.knn_planning import (
    DEFAULT_DATASETS,
    KNNPlanningEvalConfig,
    run_knn_planning_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = experiment_parser(__doc__)
    add_dataset_output_args(
        parser,
        datasets_default=DEFAULT_DATASETS,
        output_dir_default="outputs/knn_planning",
    )
    add_seed_minari_args(parser)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--pointmaze_bridge_budget", type=float, default=10.0)
    parser.add_argument("--antmaze_bridge_budget", type=float, default=15.0)
    add_state_ik_args(parser, include_batch_size=False)
    parser.add_argument("--task_preset", default="default")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(
        KNNPlanningEvalConfig,
        args,
        aliases={
            "pointmaze_h_bridge": "pointmaze_bridge_budget",
            "antmaze_h_bridge": "antmaze_bridge_budget",
        },
        list_fields=("datasets",),
    )
    result = run_knn_planning_eval(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()
