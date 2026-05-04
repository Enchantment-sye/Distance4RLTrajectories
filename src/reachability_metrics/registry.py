"""Internal factory registry helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class MetricRegistry:
    """Small wrapper for public metric factory dictionaries."""

    def __init__(self, kind: str, factories: Mapping[str, Callable[..., Any]]) -> None:
        self.kind = kind
        self.factories = factories

    def build(self, method: str, **kwargs: Any) -> Any:
        key = str(method).lower()
        try:
            factory = self.factories[key]
        except KeyError as exc:
            options = ", ".join(sorted(self.factories))
            raise ValueError(f"Unknown {self.kind} '{method}'. Available: {options}") from exc
        return factory(**kwargs)
