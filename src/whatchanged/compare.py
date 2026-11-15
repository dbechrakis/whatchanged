"""The entry point: compare two sides of a table and ask what changed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from whatchanged.attribute import SPREAD_AT, TEEP, TEP, TOP_K, explain
from whatchanged.cube import Cube, aggregate, aggregate_sql, quote
from whatchanged.decompose import Missing, decompose
from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Metric
from whatchanged.noise import noise_check
from whatchanged.result import Change, Decomposition, Explanation, NoiseCheck, Segment

MAX_SUMMARY_LEVELS = 4


def _fmt(value: float) -> str:
    return f"{value:.4g}"


@dataclass(frozen=True)
class Comparison:
    """Two sides of a table, ready to be decomposed, searched and checked for noise."""

    table: pd.DataFrame | None
    metric: Metric
    split: str
    baseline: Any
    current: Any
    dimensions: tuple[str, ...]
    cube: Cube
    connection: Any = None
    table_name: str | None = None

    def _rows(self, unit: str | None) -> pd.DataFrame:
        if self.table is not None:
            return self.table
        columns = [self.split, *self.dimensions, self.metric.numerator]
        if self.metric.denominator is not None:
            columns.append(self.metric.denominator)
        if unit is not None:
            columns.append(unit)
        names = ", ".join(quote(c) for c in dict.fromkeys(columns))
        query = (
            f"SELECT {names} FROM {quote(str(self.table_name))} WHERE {quote(self.split)} IN (?, ?)"
        )
        rows: pd.DataFrame = self.connection.execute(query, [self.baseline, self.current]).df()
        return rows

    @property
    def change(self) -> Change:
        """The metric on each side and the difference. Exact."""
        return Change(
            self.metric.name,
            self.split,
            self.baseline,
            self.current,
            self.cube.value("base"),
            self.cube.value("cur"),
        )

    def decompose(self, by: list[str] | None = None, missing: Missing = "error") -> Decomposition:
        """Exact mix and rate decomposition over nested dimensions, outermost first."""
        return decompose(self.cube, list(by) if by else list(self.dimensions), missing)

    def explain(
        self,
        method: str = "single",
        depth: int = 2,
        tep: float = TEP,
        teep: float = TEEP,
        spread_at: int = SPREAD_AT,
        top_k: int = TOP_K,
    ) -> Explanation:
        """A heuristic search for the segments that account for the change, or none."""
        return explain(self.cube, method, depth, tep, teep, spread_at, top_k)

    def noise(
        self,
        unit: str | None = None,
        segments: list[Segment] | None = None,
        history: list[tuple[Any, Any]] | None = None,
        n_boot: int = 1000,
        level: float = 0.95,
        seed: int = 0,
        method: str = "single",
    ) -> NoiseCheck:
        """Bootstrap intervals over units, and the rank among earlier period pairs when given.

        Without ``segments`` the segments named by ``explain`` are checked."""
        chosen = segments if segments is not None else list(self.explain(method=method).segments)
        return noise_check(
            self._rows(unit),
            self.metric,
            self.split,
            self.baseline,
            self.current,
            chosen,
            unit,
            n_boot,
            level,
            seed,
            history,
        )

    def summary(self) -> str:
        """At most five plain sentences. They describe where the change sits and do not say why."""
        change = self.change
        direction = (
            "rose" if change.difference > 0 else "fell" if change.difference < 0 else "did not move"
        )
        first = (
            f"{change.metric} {direction} from {_fmt(change.baseline_value)} to "
            f"{_fmt(change.current_value)} between {self.baseline} and {self.current}."
        )
        sentences = [first]
        if not self.metric.additive and change.difference != 0:
            levels = list(self.dimensions)[:MAX_SUMMARY_LEVELS]
            try:
                parts = self.decompose(levels, missing="entry_exit").components
            except WhatChangedError:
                parts = None
            if parts is not None:
                mix = float(
                    parts.loc[parts["component"].str.startswith("mix"), "contribution"].sum()
                )
                rate = float(parts.loc[parts["component"] == "rate", "contribution"].sum())
                other = change.difference - mix - rate
                sentences.append(
                    f"In exact arithmetic, {_fmt(mix)} of the change is a shift in the mix across "
                    f"{', '.join(levels)}, {_fmt(rate)} is a change in the rate within cells, "
                    f"and {_fmt(other)} is cells that entered or left."
                )
        explanation = self.explain()
        if explanation.found:
            names = ", ".join(seg.label() for seg in explanation.segments)
            share = sum(seg.explanatory_power for seg in explanation.segments)
            sentences.append(
                f"A heuristic search names {names}, "
                f"which account for about {share:.0%} of the change."
            )
        else:
            sentences.append(explanation.message)
        sentences.append(
            "This says where the change sits. It does not say why it happened, "
            "and sampling noise has not been checked unless you ran noise()."
        )
        return " ".join(sentences)


def compare(
    table: Any,
    metric: Metric,
    split: str,
    baseline: Any,
    current: Any,
    dimensions: list[str],
    name: str | None = None,
) -> Comparison:
    """Compare two values of the ``split`` column of a table over some dimensions.

    ``table`` is a pandas DataFrame, or a DuckDB connection with the table's name in ``name``.
    With a connection the aggregation runs in SQL and only the cells come back."""
    if isinstance(table, pd.DataFrame):
        cube = aggregate(table, metric, split, baseline, current, list(dimensions))
        return Comparison(table, metric, split, baseline, current, tuple(dimensions), cube)
    if name is None:
        raise WhatChangedError("give the table name in 'name' when passing a DuckDB connection")
    cube = aggregate_sql(table, name, metric, split, baseline, current, list(dimensions))
    return Comparison(None, metric, split, baseline, current, tuple(dimensions), cube, table, name)
