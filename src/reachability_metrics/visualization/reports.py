"""Markdown report writing."""

from __future__ import annotations

from reachability_metrics.utils import ensure_dir


def write_report(path: str, title: str, lines: list[str]) -> str:
    ensure_dir(__import__("os").path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# " + title + "\n\n")
        handle.write("\n".join(lines))
        handle.write("\n")
    return path

