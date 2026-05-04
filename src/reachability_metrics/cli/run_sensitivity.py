"""CLI for temporal-distance sensitivity experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.cli._helpers import (
    add_dataset_output_args,
    add_seed_minari_args,
    config_from_args,
    experiment_parser,
)
from reachability_metrics.experiments.sensitivity import (
    SensitivityConfig,
    run_sensitivity_experiments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = experiment_parser(__doc__)
    add_dataset_output_args(
        parser,
        datasets_default=None,
        output_dir_default="outputs/sensitivity",
        include_cache_dir=False,
    )
    add_seed_minari_args(parser)
    parser.add_argument("--num_anchors", type=int, default=100)
    parser.add_argument("--candidate_counts", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--top_k", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(
        SensitivityConfig,
        args,
        tuple_fields=("candidate_counts", "horizons"),
        overrides={
            "datasets": (
                args.datasets if args.datasets is not None else SensitivityConfig().datasets
            ),
        },
    )
    result = run_sensitivity_experiments(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()
