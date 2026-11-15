"""Exact decomposition of a change in a ratio into mix and rate, by Shapley allocation."""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Literal

import numpy as np
import pandas as pd

from whatchanged.cube import Cube
from whatchanged.errors import WhatChangedError
from whatchanged.result import Decomposition

MAX_FACTORS = 6
TOLERANCE = 1e-9

Missing = Literal["error", "entry_exit"]


class ReconciliationError(WhatChangedError):
    """The components did not add up to the change. This is a bug, not a property of the data."""


def _shapley_weights(n: int) -> dict[int, float]:
    return {s: factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)}


def _leaf_table(cube: Cube, by: list[str]) -> pd.DataFrame:
    unknown = [d for d in by if d not in cube.dimensions]
    if unknown:
        raise WhatChangedError(
            f"unknown dimension(s) {unknown}: the comparison has {list(cube.dimensions)}"
        )
    if len(set(by)) != len(by):
        raise WhatChangedError(f"the 'by' list repeats a name: {by}")
    sums = ["num_base", "den_base", "num_cur", "den_cur"]
    if cube.metric.additive:
        sums = ["num_base", "num_cur"]
    return cube.cells.groupby(by, observed=True, sort=True)[sums].sum().reset_index()


def _decompose_sum(cube: Cube, by: list[str]) -> Decomposition:
    leaves = _leaf_table(cube, by)
    delta = leaves["num_cur"] - leaves["num_base"]
    cells = leaves[by].copy()
    cells["component"] = "difference"
    cells["contribution"] = delta
    total = float(delta.sum())
    components = pd.DataFrame(
        [
            {
                "component": "difference",
                "contribution": total,
                "share_of_change": 1.0 if total else 0.0,
            }
        ]
    )
    base, cur = float(leaves["num_base"].sum()), float(leaves["num_cur"].sum())
    _check(total, cur - base)
    return Decomposition(cube.metric.name, tuple(by), base, cur, components, cells, "error")


def _check(total: float, change: float) -> None:
    scale = max(1.0, abs(change))
    if abs(total - change) > TOLERANCE * scale:
        raise ReconciliationError(
            f"the components add up to {total!r} but the change is {change!r}"
        )


def decompose(cube: Cube, by: list[str], missing: Missing = "error") -> Decomposition:
    """Split the change of a ratio into one mix factor per level of ``by`` and a rate factor.

    ``by`` lists nested dimensions, outermost first. The metric of a side is the sum over leaf
    cells of the share at the first level, times the share within it at the next level, and so
    on, times the leaf's own rate. Each factor's effect is averaged over every order in which
    the factors can be switched from baseline to current, so the split does not depend on the
    order. The components add up to the change exactly."""
    if missing not in ("error", "entry_exit"):
        raise WhatChangedError("missing must be 'error' or 'entry_exit'")
    by = list(by)
    if not by:
        raise WhatChangedError("give at least one dimension in 'by'")
    if cube.metric.additive:
        return _decompose_sum(cube, by)
    n_factors = len(by) + 1
    if n_factors > MAX_FACTORS:
        raise WhatChangedError(
            f"{len(by)} levels make {n_factors} factors. The most allowed is {MAX_FACTORS}, "
            "because the number of orders grows as a factorial. "
            f"Use at most {MAX_FACTORS - 1} levels."
        )
    leaves = _leaf_table(cube, by)
    present_base = leaves["den_base"] > 0
    present_cur = leaves["den_cur"] > 0
    leaves = leaves[present_base | present_cur].reset_index(drop=True)
    present_base = leaves["den_base"] > 0
    present_cur = leaves["den_cur"] > 0
    only_cur = present_cur & ~present_base
    only_base = present_base & ~present_cur
    both = present_base & present_cur
    if (only_cur | only_base).any() and missing == "error":
        names = leaves.loc[only_cur | only_base, by].astype(str).agg(" / ".join, axis=1).tolist()
        raise WhatChangedError(
            f"cells present on one side only: {names}. A rate cannot be invented for them. "
            "Use missing='entry_exit' to report them as 'entered' and 'left'."
        )
    d0_total = float(leaves["den_base"].sum())
    d1_total = float(leaves["den_cur"].sum())
    base_value = float(leaves["num_base"].sum()) / d0_total
    cur_value = float(leaves["num_cur"].sum()) / d1_total

    keep = leaves[both].reset_index(drop=True)
    shares: list[tuple[np.ndarray, np.ndarray]] = []
    for level in range(1, len(by) + 1):
        node = by[:level]
        parent = by[: level - 1]
        pair = []
        for side, total in (("den_base", d0_total), ("den_cur", d1_total)):
            node_sum = leaves.groupby(node, observed=True)[side].transform("sum")
            parent_sum = (
                leaves.groupby(parent, observed=True)[side].transform("sum") if parent else total
            )
            share = (node_sum / parent_sum).to_numpy(dtype=float)
            pair.append(share[both.to_numpy()])
        shares.append((pair[0], pair[1]))
    rate0 = (keep["num_base"] / keep["den_base"]).to_numpy(dtype=float)
    rate1 = (keep["num_cur"] / keep["den_cur"]).to_numpy(dtype=float)
    factors = [*shares, (rate0, rate1)]

    def value(switched: frozenset[int]) -> np.ndarray:
        out = np.ones(len(keep))
        for i, (a, b) in enumerate(factors):
            out = out * (b if i in switched else a)
        return out

    weights = _shapley_weights(n_factors)
    contributions = np.zeros((n_factors, len(keep)))
    everything = list(range(n_factors))
    cache: dict[frozenset[int], np.ndarray] = {}

    def cached(s: frozenset[int]) -> np.ndarray:
        if s not in cache:
            cache[s] = value(s)
        return cache[s]

    for j in everything:
        others = [i for i in everything if i != j]
        for size in range(n_factors):
            for subset in combinations(others, size):
                s = frozenset(subset)
                contributions[j] += weights[size] * (cached(s | {j}) - cached(s))

    names = [f"mix:{d}" for d in by] + ["rate"]
    rows = []
    cell_parts = []
    for j, name in enumerate(names):
        rows.append({"component": name, "contribution": float(contributions[j].sum())})
        part = keep[by].copy()
        part["component"] = name
        part["contribution"] = contributions[j]
        cell_parts.append(part)
    if only_cur.any():
        entered = leaves.loc[only_cur, by].copy()
        entered["component"] = "entered"
        entered["contribution"] = (leaves.loc[only_cur, "num_cur"] / d1_total).to_numpy(dtype=float)
        rows.append({"component": "entered", "contribution": float(entered["contribution"].sum())})
        cell_parts.append(entered)
    if only_base.any():
        left = leaves.loc[only_base, by].copy()
        left["component"] = "left"
        left["contribution"] = (-leaves.loc[only_base, "num_base"] / d0_total).to_numpy(dtype=float)
        rows.append({"component": "left", "contribution": float(left["contribution"].sum())})
        cell_parts.append(left)
    components = pd.DataFrame(rows)
    change = cur_value - base_value
    _check(float(components["contribution"].sum()), change)
    components["share_of_change"] = components["contribution"] / change if change != 0 else 0.0
    cells = pd.concat(cell_parts, ignore_index=True)
    return Decomposition(
        cube.metric.name, tuple(by), base_value, cur_value, components, cells, missing
    )
