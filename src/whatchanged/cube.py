"""Aggregate rows to cells. A cell is one combination of dimension values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Metric, Sum

SIDES = ("base", "cur")
ROW_COUNT = "__rows__"


@dataclass(frozen=True)
class Cube:
    """The cells of a comparison.

    ``num_base``, ``den_base``, ``num_cur`` and ``den_cur`` are sums over the rows of a cell.

    For a ``Sum`` the denominators are missing. For a ``Mean`` the denominator is the row count."""

    metric: Metric
    dimensions: tuple[str, ...]
    cells: pd.DataFrame
    split: str
    baseline: Any
    current: Any

    def totals(self) -> dict[str, float]:
        out = {c: float(self.cells[c].sum()) for c in ("num_base", "num_cur")}
        for c in ("den_base", "den_cur"):
            out[c] = float(self.cells[c].sum()) if not self.metric.additive else float("nan")
        return out

    def value(self, side: str) -> float:
        """The metric of one side, computed from the cells."""
        totals = self.totals()
        num = totals[f"num_{side}"]
        if self.metric.additive:
            return num
        return num / totals[f"den_{side}"]


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise WhatChangedError(f"unknown column(s) {missing}: the table has {list(df.columns)}")


def aggregate(
    df: pd.DataFrame,
    metric: Metric,
    split: str,
    baseline: Any,
    current: Any,
    dimensions: list[str],
) -> Cube:
    """Group the rows of the two sides into cells over ``dimensions``."""
    if not dimensions:
        raise WhatChangedError("give at least one dimension")
    if len(set(dimensions)) != len(dimensions):
        raise WhatChangedError(f"the dimensions repeat a name: {dimensions}")
    needed = [metric.numerator, *dimensions, split]
    if metric.denominator is not None:
        needed.append(metric.denominator)
    _check_columns(df, needed)
    if split in dimensions:
        raise WhatChangedError(f"the split column {split!r} cannot also be a dimension")
    if baseline == current:
        raise WhatChangedError("the baseline and the current value are the same")
    for column in dimensions:
        if df[column].isna().any():
            raise WhatChangedError(
                f"the dimension {column!r} has missing values: fill or drop them first"
            )
    parts: dict[str, pd.DataFrame] = {}
    for side, value in zip(SIDES, (baseline, current), strict=True):
        rows = df[df[split] == value]
        if rows.empty:
            raise WhatChangedError(f"no rows have {split} == {value!r}")
        work = pd.DataFrame({column: rows[column] for column in dimensions})
        work["num"] = rows[metric.numerator].astype(float)
        if isinstance(metric, Sum):
            work["den"] = float("nan")
        elif metric.denominator is None:
            work["den"] = 1.0
        else:
            work["den"] = rows[metric.denominator].astype(float)
        grouped = work.groupby(dimensions, observed=True, sort=True)[["num", "den"]].sum(
            min_count=1
        )
        parts[side] = grouped.rename(columns={"num": f"num_{side}", "den": f"den_{side}"})
    cells = parts["base"].join(parts["cur"], how="outer").reset_index()
    for column in ("num_base", "num_cur"):
        cells[column] = cells[column].fillna(0.0)
    if not metric.additive:
        for column in ("den_base", "den_cur"):
            cells[column] = cells[column].fillna(0.0)
        for side, value in zip(SIDES, (baseline, current), strict=True):
            if float(cells[f"den_{side}"].sum()) == 0.0:
                raise WhatChangedError(
                    f"the denominator of {metric.name} is zero on the side {split} == {value!r}"
                )
    return Cube(metric, tuple(dimensions), cells, split, baseline, current)
