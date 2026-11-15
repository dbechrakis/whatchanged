"""Result objects. Each carries a ``kind``: ``exact``, ``heuristic`` or ``statistical``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Kind = Literal["exact", "heuristic", "statistical"]


def _clean(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Change:
    """The metric on each side and the difference. Arithmetic, so ``exact``."""

    metric: str
    split: str
    baseline: Any
    current: Any
    baseline_value: float
    current_value: float
    kind: Kind = "exact"

    @property
    def difference(self) -> float:
        return self.current_value - self.baseline_value

    @property
    def relative(self) -> float:
        if self.baseline_value == 0:
            return float("nan")
        return self.difference / abs(self.baseline_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "split": self.split,
            "baseline": self.baseline,
            "current": self.current,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "difference": self.difference,
            "relative": self.relative,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


@dataclass(frozen=True)
class Decomposition:
    """An exact split of the change over factors. The components add up to the change.

    ``components`` has one row per factor. ``cells`` has one row per cell and factor."""

    metric: str
    by: tuple[str, ...]
    baseline_value: float
    current_value: float
    components: pd.DataFrame
    cells: pd.DataFrame
    missing: str
    kind: Kind = "exact"

    @property
    def change(self) -> float:
        return self.current_value - self.baseline_value

    def to_frame(self) -> pd.DataFrame:
        return self.components.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "by": list(self.by),
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change": self.change,
            "missing": self.missing,
            "components": _clean(self.components),
            "cells": _clean(self.cells),
        }
