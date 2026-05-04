"""CLI for H-step successor distance experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.cli._helpers import (
    add_dataset_output_args,
    add_seed_minari_args,
    add_successor_ik_args,
    config_from_args,
    experiment_parser,
)
from reachability_metrics.experiments.successor_distance import (
    DEFAULT_DATASETS,
    SuccessorDistanceConfig,
    run_successor_distance,
)


def build_parser() -> argparse.ArgumentParser:
    parser = experiment_parser(__doc__)
    add_dataset_output_args(
        parser,
        datasets_default=DEFAULT_DATASETS,
        output_dir_default="outputs/successor_distance",
    )
    add_seed_minari_args(parser)
    parser.add_argument("--horizon_values", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--eval_num_pairs", type=int, default=50000)
    parser.add_argument("--search_num_pairs", type=int, default=20000)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--num_candidates", type=int, default=256)
    parser.add_argument("--recall_k_values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--grid_nx", type=int, default=20)
    parser.add_argument("--grid_ny", type=int, default=20)
    add_successor_ik_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(
        SuccessorDistanceConfig,
        args,
        list_fields=("datasets", "horizon_values"),
        tuple_fields=(
            "recall_k_values",
            "ik_ensemble_sizes",
            "ik_subsample_sizes",
            "ik_temperatures",
        ),
    )
    result = run_successor_distance(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()
