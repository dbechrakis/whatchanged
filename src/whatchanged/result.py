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


@dataclass(frozen=True)
class Segment:
    """A set of dimension values, for example region equal to West and category equal to Tech."""

    conditions: tuple[tuple[str, Any], ...]
    explanatory_power: float
    surprise: float

    def label(self) -> str:
        return " and ".join(f"{name} = {value}" for name, value in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": {name: value for name, value in self.conditions},
            "explanatory_power": self.explanatory_power,
            "surprise": self.surprise,
        }


@dataclass(frozen=True)
class CandidateSet:
    """The elements of one dimension that together explain most of the change."""

    dimension: str
    segments: tuple[Segment, ...]
    explains: float
    surprise: float


@dataclass(frozen=True)
class Explanation:
    """The segments a heuristic search names.

    ``found`` is false when no segment explains the change. The search is Adtributor, a
    heuristic. It can name the wrong segment, and the benchmark says when."""

    metric: str
    method: str
    found: bool
    message: str
    candidates: tuple[CandidateSet, ...]
    segments: tuple[Segment, ...]
    parameters: dict[str, Any]
    kind: Kind = "heuristic"

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "segment": seg.label(),
                **{name: value for name, value in seg.conditions},
                "explanatory_power": seg.explanatory_power,
                "surprise": seg.surprise,
            }
            for seg in self.segments
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "method": self.method,
            "found": self.found,
            "message": self.message,
            "segments": [seg.to_dict() for seg in self.segments],
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class NoiseCheck:
    """Whether each segment's contribution is outside its sampling noise or its usual range.

    ``sampling`` has a bootstrap interval per segment. ``history`` has the rank of the current
    contribution among earlier period pairs. Both are statistical statements."""

    metric: str
    unit: str | None
    sampling: pd.DataFrame
    history: pd.DataFrame | None
    parameters: dict[str, Any]
    kind: Kind = "statistical"

    def to_frame(self) -> pd.DataFrame:
        if self.history is None:
            return self.sampling.copy()
        return self.sampling.merge(self.history, on="segment", how="left")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "unit": self.unit,
            "sampling": _clean(self.sampling),
            "history": None if self.history is None else _clean(self.history),
            "parameters": dict(self.parameters),
        }
