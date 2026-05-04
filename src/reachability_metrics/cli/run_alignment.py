"""CLI for reachability alignment experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.cli._helpers import (
    add_dataset_output_args,
    add_seed_minari_args,
    add_state_ik_args,
    config_from_args,
    experiment_parser,
)
from reachability_metrics.experiments.reachability_alignment import (
    DEFAULT_DATASETS,
    ReachabilityAnalysisConfig,
    analyze_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = experiment_parser(__doc__)
    add_dataset_output_args(
        parser,
        datasets_default=DEFAULT_DATASETS,
        output_dir_default="outputs/reachability_alignment",
    )
    add_seed_minari_args(parser)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--hit_radius", type=float, default=0.25)
    parser.add_argument("--num_anchors", type=int, default=200)
    parser.add_argument("--num_candidates", type=int, default=1000)
    parser.add_argument("--fit_pool_size", type=int, default=50000)
    add_state_ik_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(
        ReachabilityAnalysisConfig,
        args,
        list_fields=("datasets",),
    )
    result = analyze_datasets(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()
