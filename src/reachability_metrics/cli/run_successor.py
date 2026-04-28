"""CLI for H-step successor distance experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.experiments.successor_distance import DEFAULT_DATASETS, SuccessorDistanceConfig, run_successor_distance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output_dir", default="outputs/successor_distance")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minari_datasets_path", default=None)
    parser.add_argument("--horizon_values", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--eval_num_pairs", type=int, default=50000)
    parser.add_argument("--search_num_pairs", type=int, default=20000)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--num_candidates", type=int, default=256)
    parser.add_argument("--recall_k_values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--grid_nx", type=int, default=20)
    parser.add_argument("--grid_ny", type=int, default=20)
    parser.add_argument("--ik_ensemble_sizes", nargs="+", type=int, default=[100])
    parser.add_argument("--ik_subsample_sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--ik_temperatures", nargs="+", type=float, default=[0.01])
    parser.add_argument("--ik_batch_size", type=int, default=4096)
    parser.add_argument("--ik_device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = SuccessorDistanceConfig(
        datasets=list(args.datasets),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        minari_datasets_path=args.minari_datasets_path,
        horizon_values=list(args.horizon_values),
        eval_num_pairs=args.eval_num_pairs,
        search_num_pairs=args.search_num_pairs,
        num_queries=args.num_queries,
        num_candidates=args.num_candidates,
        recall_k_values=tuple(args.recall_k_values),
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        ik_ensemble_sizes=tuple(args.ik_ensemble_sizes),
        ik_subsample_sizes=tuple(args.ik_subsample_sizes),
        ik_temperatures=tuple(args.ik_temperatures),
        ik_batch_size=args.ik_batch_size,
        ik_device=args.ik_device,
    )
    result = run_successor_distance(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()

