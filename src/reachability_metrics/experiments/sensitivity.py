"""Sensitivity experiments for temporal baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reachability_metrics.experiments.artifacts import ArtifactWriter
from reachability_metrics.experiments.knn_relabeling import DEFAULT_DATASETS, KNNRelabelConfig, run_relabel_benchmark


@dataclass
class SensitivityConfig:
    """Configuration for lightweight sensitivity sweeps."""

    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    output_dir: str = "outputs/sensitivity"
    seed: int = 0
    minari_datasets_path: str | None = None
    num_anchors: int = 100
    candidate_counts: tuple[int, ...] = (64, 128, 256)
    horizons: tuple[int, ...] = (10, 20, 50)
    top_k: int = 20

    @property
    def tables_dir(self) -> str:
        return f"{self.output_dir}/tables"


def run_sensitivity_experiments(cfg: SensitivityConfig) -> dict[str, Any]:
    """Run temporal-sample-starvation and window-mismatch sweeps."""

    artifacts = ArtifactWriter(cfg.output_dir).prepare(figures=False, cache=False)
    rows: list[dict[str, Any]] = []
    for count in cfg.candidate_counts:
        relabel_cfg = KNNRelabelConfig(
            datasets=cfg.datasets,
            output_dir=f"{cfg.output_dir}/temporal_sample_starvation_{count}",
            seed=cfg.seed,
            minari_datasets_path=cfg.minari_datasets_path,
            num_anchors=cfg.num_anchors,
            num_candidates=int(count),
            top_k=cfg.top_k,
            horizon=20,
        )
        result = run_relabel_benchmark(relabel_cfg)
        for row in result["summary_rows"]:
            if row["method"] == "temporal_distance":
                rows.append({"experiment": "temporal_sample_starvation", "num_candidates": count, **row})
    for horizon in cfg.horizons:
        relabel_cfg = KNNRelabelConfig(
            datasets=cfg.datasets,
            output_dir=f"{cfg.output_dir}/window_mismatch_{horizon}",
            seed=cfg.seed,
            minari_datasets_path=cfg.minari_datasets_path,
            num_anchors=cfg.num_anchors,
            num_candidates=max(cfg.candidate_counts),
            top_k=cfg.top_k,
            horizon=int(horizon),
        )
        result = run_relabel_benchmark(relabel_cfg)
        for row in result["summary_rows"]:
            if row["method"] == "temporal_distance":
                rows.append({"experiment": "window_mismatch", "horizon": horizon, **row})
    table_path = artifacts.save_csv("sensitivity_summary.csv", rows)
    report_path = artifacts.write_report("Sensitivity Report", [f"- summary: `{table_path}`"])
    return {"summary_rows": rows, "summary_path": table_path, "report_path": report_path}
