"""Experiment output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reachability_metrics.evaluation.reports import save_csv
from reachability_metrics.utils import ensure_dir
from reachability_metrics.visualization.reports import write_report


@dataclass
class ArtifactWriter:
    """Centralized paths and artifact writing for experiment runners."""

    output_dir: str
    cache_dir: str | None = None

    @property
    def tables_dir(self) -> str:
        return f"{self.output_dir}/tables"

    @property
    def figures_dir(self) -> str:
        return f"{self.output_dir}/figures"

    def prepare(self, *, figures: bool = True, cache: bool = True) -> "ArtifactWriter":
        ensure_dir(self.output_dir)
        if cache:
            ensure_dir(self.cache_dir or f"{self.output_dir}/cache")
        ensure_dir(self.tables_dir)
        if figures:
            ensure_dir(self.figures_dir)
        return self

    def table_path(self, name: str) -> str:
        return f"{self.tables_dir}/{name}"

    def figure_path(self, name: str) -> str:
        return f"{self.figures_dir}/{name}"

    def report_path(self, name: str = "report.md") -> str:
        return f"{self.output_dir}/{name}"

    def save_csv(self, name: str, rows: list[dict[str, Any]]) -> str:
        path = self.table_path(name)
        save_csv(path, rows)
        return path

    def write_report(self, title: str, lines: list[str], name: str = "report.md") -> str:
        return write_report(self.report_path(name), title, lines)


class ExperimentRunner:
    """Small base runner that owns artifact preparation."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.artifacts = ArtifactWriter(
            output_dir=str(cfg.output_dir),
            cache_dir=getattr(cfg, "cache_dir", None),
        )

    def prepare_artifacts(self, *, figures: bool = True, cache: bool = True) -> ArtifactWriter:
        return self.artifacts.prepare(figures=figures, cache=cache)

