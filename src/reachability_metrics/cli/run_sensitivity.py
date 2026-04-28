"""CLI for temporal-distance sensitivity experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.experiments.sensitivity import SensitivityConfig, run_sensitivity_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output_dir", default="outputs/sensitivity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minari_datasets_path", default=None)
    parser.add_argument("--num_anchors", type=int, default=100)
    parser.add_argument("--candidate_counts", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--top_k", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = SensitivityConfig(
        datasets=args.datasets if args.datasets is not None else SensitivityConfig().datasets,
        output_dir=args.output_dir,
        seed=args.seed,
        minari_datasets_path=args.minari_datasets_path,
        num_anchors=args.num_anchors,
        candidate_counts=tuple(args.candidate_counts),
        horizons=tuple(args.horizons),
        top_k=args.top_k,
    )
    result = run_sensitivity_experiments(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()

