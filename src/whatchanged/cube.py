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
    if isinstance(metric, Sum) and metric.units is not None:
        needed.append(metric.units)
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
            work["den"] = float("nan") if metric.units is None else rows[metric.units].astype(float)
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
    if isinstance(metric, Sum) and metric.units is not None:
        for column in ("den_base", "den_cur"):
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


def quote(identifier: str) -> str:
    """Quote an SQL identifier. Values are passed as parameters and never formatted in."""
    return '"' + identifier.replace('"', '""') + '"'


def aggregate_sql(
    connection: Any,
    table: str,
    metric: Metric,
    split: str,
    baseline: Any,
    current: Any,
    dimensions: list[str],
) -> Cube:
    """The same cells as ``aggregate``, aggregated by DuckDB. Only the cells come back."""
    if not dimensions:
        raise WhatChangedError("give at least one dimension")
    if len(set(dimensions)) != len(dimensions):
        raise WhatChangedError(f"the dimensions repeat a name: {dimensions}")
    if split in dimensions:
        raise WhatChangedError(f"the split column {split!r} cannot also be a dimension")
    if baseline == current:
        raise WhatChangedError("the baseline and the current value are the same")
    try:
        known = [row[0] for row in connection.execute(f"DESCRIBE {quote(table)}").fetchall()]
    except Exception as error:
        raise WhatChangedError(f"cannot read the table {table!r}: {error}") from error
    needed = [metric.numerator, *dimensions, split]
    if metric.denominator is not None:
        needed.append(metric.denominator)
    if isinstance(metric, Sum) and metric.units is not None:
        needed.append(metric.units)
    missing = [c for c in needed if c not in known]
    if missing:
        raise WhatChangedError(f"unknown column(s) {missing}: the table has {known}")
    for value in (baseline, current):
        found = connection.execute(
            f"SELECT COUNT(*) FROM {quote(table)} WHERE {quote(split)} = ?", [value]
        ).fetchall()[0][0]
        if not found:
            raise WhatChangedError(f"no rows have {split} == {value!r}")
    for column in dimensions:
        nulls = connection.execute(
            f"SELECT COUNT(*) FROM {quote(table)} WHERE {quote(column)} IS NULL "
            f"AND {quote(split)} IN (?, ?)",
            [baseline, current],
        ).fetchall()[0][0]
        if nulls:
            raise WhatChangedError(
                f"the dimension {column!r} has missing values: fill or drop them first"
            )
    select = [quote(d) for d in dimensions]
    params: list[Any] = []
    for side, value in zip(SIDES, (baseline, current), strict=True):
        test = f"{quote(split)} = ?"
        select.append(
            f"SUM(CASE WHEN {test} THEN CAST({quote(metric.numerator)} AS DOUBLE) END) "
            f"AS num_{side}"
        )
        params.append(value)
        if isinstance(metric, Sum) and metric.units is None:
            continue
        if isinstance(metric, Sum):
            select.append(
                f"SUM(CASE WHEN {test} THEN CAST({quote(str(metric.units))} AS DOUBLE) END) "
                f"AS den_{side}"
            )
        elif metric.denominator is None:
            select.append(f"SUM(CASE WHEN {test} THEN 1.0 END) AS den_{side}")
        else:
            select.append(
                f"SUM(CASE WHEN {test} THEN CAST({quote(metric.denominator)} AS DOUBLE) END) "
                f"AS den_{side}"
            )
        params.append(value)
    query = (
        f"SELECT {', '.join(select)} FROM {quote(table)} "
        f"WHERE {quote(split)} IN (?, ?) GROUP BY {', '.join(quote(d) for d in dimensions)}"
    )
    cells = connection.execute(query, [*params, baseline, current]).df()
    cells = cells.sort_values(dimensions).reset_index(drop=True)
    if isinstance(metric, Sum) and metric.units is None:
        cells["den_base"] = float("nan")
        cells["den_cur"] = float("nan")
    for column in ("num_base", "num_cur"):
        cells[column] = cells[column].fillna(0.0).astype(float)
    if isinstance(metric, Sum) and metric.units is not None:
        for column in ("den_base", "den_cur"):
            cells[column] = cells[column].fillna(0.0).astype(float)
    if not metric.additive:
        for column in ("den_base", "den_cur"):
            cells[column] = cells[column].fillna(0.0).astype(float)
        for side, value in zip(SIDES, (baseline, current), strict=True):
            if float(cells[f"den_{side}"].sum()) == 0.0:
                raise WhatChangedError(
                    f"the denominator of {metric.name} is zero on the side {split} == {value!r}"
                )
    cells = cells[[*dimensions, "num_base", "den_base", "num_cur", "den_cur"]]
    return Cube(metric, tuple(dimensions), cells, split, baseline, current)
