"""The metrics a comparison can explain: a sum, a ratio of two sums, and a mean."""

from __future__ import annotations

from dataclasses import dataclass

from whatchanged.errors import WhatChangedError


@dataclass(frozen=True)
class Sum:
    """The sum of a column. Additive: the cells add up to the total."""

    column: str

    @property
    def name(self) -> str:
        return f"sum({self.column})"

    @property
    def additive(self) -> bool:
        return True

    @property
    def numerator(self) -> str:
        return self.column

    @property
    def denominator(self) -> str | None:
        return None


@dataclass(frozen=True)
class Ratio:
    """The ratio of the sum of one column to the sum of another. Not additive."""

    numerator: str
    denominator: str

    @property
    def name(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def additive(self) -> bool:
        return False


@dataclass(frozen=True)
class Mean:
    """The mean of a column: its sum over the number of rows."""

    column: str

    @property
    def name(self) -> str:
        return f"mean({self.column})"

    @property
    def additive(self) -> bool:
        return False

    @property
    def numerator(self) -> str:
        return self.column

    @property
    def denominator(self) -> str | None:
        return None


Metric = Sum | Ratio | Mean


def parse_metric(text: str) -> Metric:
    """Read a metric from text: ``a/b`` is a ratio, ``sum:a`` a sum, ``mean:a`` a mean."""
    text = text.strip()
    if "/" in text:
        numerator, denominator = (part.strip() for part in text.split("/", 1))
        if not numerator or not denominator:
            raise WhatChangedError(
                f"cannot read the metric {text!r}: a ratio needs two column names"
            )
        return Ratio(numerator, denominator)
    kind, _, column = text.partition(":")
    if kind == "sum" and column:
        return Sum(column.strip())
    if kind == "mean" and column:
        return Mean(column.strip())
    raise WhatChangedError(f"cannot read the metric {text!r}: use 'a/b', 'sum:a' or 'mean:a'")
