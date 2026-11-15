"""Adtributor: a heuristic search for the few segments that account for a change."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from whatchanged.cube import Cube
from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Metric
from whatchanged.result import CandidateSet, Explanation, Segment

TEP = 0.67
TEEP = 0.10
SPREAD_AT = 3
TOP_K = 3
NO_CHANGE = 1e-12


def js_surprise(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """The Jensen-Shannon term of each element (equations 5 to 7 of the paper), base 2."""
    total_f, total_a = float(forecast.sum()), float(actual.sum())
    p = forecast / total_f if total_f != 0 else np.zeros_like(forecast)
    q = actual / total_a if total_a != 0 else np.zeros_like(actual)
    mid = p + q
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p * np.log2(2 * p / mid), 0.0)
        right = np.where(q > 0, q * np.log2(2 * q / mid), 0.0)
    return np.asarray(0.5 * (left + right), dtype=float)


def additive_power(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Equation 4: each element's change over the total change."""
    change = float(actual.sum() - forecast.sum())
    return np.asarray((actual - forecast) / change, dtype=float)


def ratio_power(
    f1: np.ndarray, a1: np.ndarray, f2: np.ndarray, a2: np.ndarray
) -> np.ndarray | None:
    """Equation 8, normalised to sum to one. Returns None when the raw values sum to zero."""
    total_f1, total_f2 = float(f1.sum()), float(f2.sum())
    denominator = total_f2 * (total_f2 + a2 - f2)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = ((a1 - f1) * total_f2 - (a2 - f2) * total_f1) / denominator
    raw = np.where(np.isfinite(raw), raw, 0.0)
    norm = float(raw.sum())
    if abs(norm) < NO_CHANGE:
        return None
    return np.asarray(raw / norm, dtype=float)


def metric_values(cells: pd.DataFrame, metric: Metric) -> tuple[float, float]:
    base_num, cur_num = float(cells["num_base"].sum()), float(cells["num_cur"].sum())
    if metric.additive:
        return base_num, cur_num
    return base_num / float(cells["den_base"].sum()), cur_num / float(cells["den_cur"].sum())


def _dimension_sets(
    cells: pd.DataFrame,
    metric: Metric,
    dimension: str,
    parents: tuple[tuple[str, Any], ...],
    tep: float,
    teep: float,
    spread_at: int,
) -> CandidateSet | None:
    sums = (
        ["num_base", "num_cur"]
        if metric.additive
        else ["num_base", "den_base", "num_cur", "den_cur"]
    )
    table = cells.groupby(dimension, observed=True, sort=True)[sums].sum().reset_index()
    if len(table) < 2:
        return None
    f1, a1 = table["num_base"].to_numpy(float), table["num_cur"].to_numpy(float)
    surprise = js_surprise(f1, a1)
    if metric.additive:
        power = additive_power(f1, a1)
    else:
        f2, a2 = table["den_base"].to_numpy(float), table["den_cur"].to_numpy(float)
        ratio = ratio_power(f1, a1, f2, a2)
        if ratio is None:
            return None
        power = ratio
        surprise = surprise + js_surprise(f2, a2)
    order = np.argsort(-surprise, kind="stable")
    chosen: list[Segment] = []
    explains = 0.0
    total_surprise = 0.0
    for index in order:
        if power[index] > teep:
            value = table[dimension].iloc[index]
            chosen.append(
                Segment((*parents, (dimension, value)), float(power[index]), float(surprise[index]))
            )
            total_surprise += float(surprise[index])
            explains += float(power[index])
        if explains > tep:
            if len(chosen) >= spread_at:
                return None
            return CandidateSet(dimension, tuple(chosen), explains, total_surprise)
    return None


def candidate_sets(
    cells: pd.DataFrame,
    metric: Metric,
    dimensions: list[str],
    parents: tuple[tuple[str, Any], ...] = (),
    tep: float = TEP,
    teep: float = TEEP,
    spread_at: int = SPREAD_AT,
    top_k: int = TOP_K,
) -> list[CandidateSet]:
    """The candidate set of each dimension that has one, most surprising first."""
    found = []
    for dimension in dimensions:
        candidate = _dimension_sets(cells, metric, dimension, parents, tep, teep, spread_at)
        if candidate is not None:
            found.append(candidate)
    found.sort(key=lambda c: -c.surprise)
    return found[:top_k]


def _has_change(cells: pd.DataFrame, metric: Metric) -> bool:
    base, cur = metric_values(cells, metric)
    return abs(cur - base) > NO_CHANGE * max(1.0, abs(base))


def _recurse(
    cells: pd.DataFrame,
    metric: Metric,
    dimensions: list[str],
    segment: Segment,
    depth: int,
    tep: float,
    teep: float,
    spread_at: int,
) -> list[Segment]:
    if depth <= 0 or not dimensions:
        return [segment]
    ((dim, value),) = segment.conditions[-1:]
    inside = cells[cells[dim] == value]
    remaining = [d for d in dimensions if d != dim]
    if not remaining or inside.empty or not _has_change(inside, metric):
        return [segment]
    found = candidate_sets(inside, metric, remaining, segment.conditions, tep, teep, spread_at, 1)
    if not found:
        return [segment]
    out: list[Segment] = []
    for child in found[0].segments:
        out.extend(_recurse(inside, metric, remaining, child, depth - 1, tep, teep, spread_at))
    return out


def explain(
    cube: Cube,
    method: str = "single",
    depth: int = 2,
    tep: float = TEP,
    teep: float = TEEP,
    spread_at: int = SPREAD_AT,
    top_k: int = TOP_K,
) -> Explanation:
    """Name the segments that account for most of the change, or say that none does.

    ``single`` looks at one dimension at a time and returns the elements of the most surprising
    dimension. ``recursive`` then repeats the search inside each element to reach cells of more
    than one dimension, ``depth`` levels deep. The result is a heuristic."""
    if method not in ("single", "recursive"):
        raise WhatChangedError("method must be 'single' or 'recursive'")
    if not 0 < teep < tep <= 1:
        raise WhatChangedError("the thresholds must satisfy 0 < teep < tep <= 1")
    if spread_at < 1 or depth < 1:
        raise WhatChangedError("spread_at and depth must be at least 1")
    params = {"tep": tep, "teep": teep, "spread_at": spread_at, "top_k": top_k, "depth": depth}
    dims = list(cube.dimensions)
    name = cube.metric.name
    if not _has_change(cube.cells, cube.metric):
        return Explanation(name, method, False, "There is no change to explain.", (), (), params)
    candidates = candidate_sets(cube.cells, cube.metric, dims, (), tep, teep, spread_at, top_k)
    if not candidates:
        message = (
            "No segment explains the change. It is spread over the elements of every dimension, "
            f"with no set of fewer than {spread_at} elements passing the thresholds."
        )
        return Explanation(name, method, False, message, (), (), params)
    segments: list[Segment] = list(candidates[0].segments)
    if method == "recursive":
        segments = []
        for segment in candidates[0].segments:
            segments.extend(
                _recurse(cube.cells, cube.metric, dims, segment, depth - 1, tep, teep, spread_at)
            )
    message = f"{len(segments)} segment(s) account for the change. This is a heuristic."
    return Explanation(name, method, True, message, tuple(candidates), tuple(segments), params)
