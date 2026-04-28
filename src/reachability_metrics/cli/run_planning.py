"""CLI for offline kNN planning experiments."""

from __future__ import annotations

import argparse

from reachability_metrics.experiments.knn_planning import DEFAULT_DATASETS, KNNPlanningEvalConfig, run_knn_planning_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output_dir", default="outputs/knn_planning")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minari_datasets_path", default=None)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--pointmaze_bridge_budget", type=float, default=10.0)
    parser.add_argument("--antmaze_bridge_budget", type=float, default=15.0)
    parser.add_argument("--ik_ensemble_size", type=int, default=100)
    parser.add_argument("--ik_subsample_size", type=int, default=32)
    parser.add_argument("--ik_temperature", type=float, default=0.01)
    parser.add_argument("--ik_device", default="auto")
    parser.add_argument("--task_preset", default="default")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = KNNPlanningEvalConfig(
        datasets=list(args.datasets),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        minari_datasets_path=args.minari_datasets_path,
        retrieval_top_k=args.retrieval_top_k,
        num_queries=args.num_queries,
        alpha=args.alpha,
        pointmaze_h_bridge=args.pointmaze_bridge_budget,
        antmaze_h_bridge=args.antmaze_bridge_budget,
        ik_ensemble_size=args.ik_ensemble_size,
        ik_subsample_size=args.ik_subsample_size,
        ik_temperature=args.ik_temperature,
        ik_device=args.ik_device,
        task_preset=args.task_preset,
    )
    result = run_knn_planning_eval(cfg)
    print(result["report_path"])


if __name__ == "__main__":
    main()

