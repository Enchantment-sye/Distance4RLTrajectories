"""CLI for kNN relabeling experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.experiments.knn_relabeling import DEFAULT_DATASETS, KNNRelabelConfig, run_relabel_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output_dir", default="outputs/knn_relabeling")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minari_datasets_path", default=None)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--num_anchors", type=int, default=200)
    parser.add_argument("--num_candidates", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--fit_pool_size", type=int, default=50000)
    parser.add_argument("--ik_ensemble_size", type=int, default=100)
    parser.add_argument("--ik_subsample_size", type=int, default=32)
    parser.add_argument("--ik_temperature", type=float, default=0.01)
    parser.add_argument("--ik_batch_size", type=int, default=4096)
    parser.add_argument("--ik_device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = KNNRelabelConfig(
        datasets=list(args.datasets),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        minari_datasets_path=args.minari_datasets_path,
        horizon=args.horizon,
        num_anchors=args.num_anchors,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        fit_pool_size=args.fit_pool_size,
        ik_ensemble_size=args.ik_ensemble_size,
        ik_subsample_size=args.ik_subsample_size,
        ik_temperature=args.ik_temperature,
        ik_batch_size=args.ik_batch_size,
        ik_device=args.ik_device,
    )
    result = run_relabel_benchmark(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()

