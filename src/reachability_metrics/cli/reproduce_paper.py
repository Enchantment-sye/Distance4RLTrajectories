"""Reproduce paper tables and figures from legacy experiment outputs."""

from __future__ import annotations

import argparse

from reachability_metrics.cli._helpers import config_from_args, experiment_parser
from reachability_metrics.experiments.paper_reproduction import (
    PaperReproductionConfig,
    run_paper_reproduction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = experiment_parser(__doc__)
    parser.add_argument("--legacy_outputs_dir", default="/share/shangyy/codes/metra/outputs")
    parser.add_argument("--output_dir", default="outputs/paper_reproduction")
    parser.add_argument("--include_figures", action="store_true")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--verify-paper-values", action="store_true", dest="verify_paper_values")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    include_figures = True
    if args.no_figures:
        include_figures = False
    if args.include_figures:
        include_figures = True
    cfg = config_from_args(
        PaperReproductionConfig,
        args,
        overrides={
            "include_figures": include_figures,
            "verify_paper_values": bool(args.verify_paper_values),
            "strict": bool(args.strict),
        },
    )
    result = run_paper_reproduction(cfg)
    print(result["report_path"])
    if result["verification_errors"]:
        print("verification completed with non-strict notes:")
        for error in result["verification_errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
