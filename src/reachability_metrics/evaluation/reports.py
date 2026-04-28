"""Small report helpers."""

from __future__ import annotations

import csv
from typing import Any

from reachability_metrics.utils import ensure_dir


def save_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Save rows to CSV."""
    ensure_dir(__import__("os").path.dirname(path))
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

