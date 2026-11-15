"""Statistical checks on the segments that a comparison names: sampling noise and usual range."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Metric, Sum
from whatchanged.result import NoiseCheck, Segment

BOOTSTRAP_CHUNK = 200


def _mask(df: pd.DataFrame, conditions: tuple[tuple[str, Any], ...]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in conditions:
        mask &= df[column] == value
    return mask


def _parts(df: pd.DataFrame, metric: Metric) -> tuple[np.ndarray, np.ndarray]:
    num = df[metric.numerator].to_numpy(dtype=float)
    if isinstance(metric, Sum):
        return num, np.zeros(len(df))
    if metric.denominator is None:
        return num, np.ones(len(df))
    return num, df[metric.denominator].to_numpy(dtype=float)


def contribution(
    metric: Metric,
    n0: Any,
    d0: Any,
    n1: Any,
    d1: Any,
    n0_seg: Any,
    d0_seg: Any,
    n1_seg: Any,
    d1_seg: Any,
) -> Any:
    """The change in the metric when only the segment moves from baseline to current values."""
    if metric.additive:
        return n1_seg - n0_seg
    return (n0 - n0_seg + n1_seg) / (d0 - d0_seg + d1_seg) - n0 / d0


def segment_contribution(
    df: pd.DataFrame,
    metric: Metric,
    split: str,
    baseline: Any,
    current: Any,
    conditions: tuple[tuple[str, Any], ...],
) -> float | None:
    """The contribution of a segment for one pair of sides, or None when it cannot be computed."""
    sides = []
    for value in (baseline, current):
        rows = df[df[split] == value]
        if rows.empty:
            return None
        num, den = _parts(rows, metric)
        mask = _mask(rows, conditions).to_numpy()
        sides.append((num.sum(), den.sum(), num[mask].sum(), den[mask].sum()))
    (n0, d0, n0s, d0s), (n1, d1, n1s, d1s) = sides
    if not metric.additive and (d0 <= 0 or d1 <= 0 or d0 - d0s + d1s <= 0):
        return None
    return float(contribution(metric, n0, d0, n1, d1, n0s, d0s, n1s, d1s))


def _unit_matrix(
    rows: pd.DataFrame, metric: Metric, unit: str | None, segments: list[Segment]
) -> np.ndarray:
    num, den = _parts(rows, metric)
    columns = [num, den]
    for seg in segments:
        mask = _mask(rows, seg.conditions).to_numpy()
        columns.extend([np.where(mask, num, 0.0), np.where(mask, den, 0.0)])
    frame = pd.DataFrame(np.column_stack(columns))
    if unit is None:
        return frame.to_numpy()
    return frame.groupby(rows[unit].to_numpy(), sort=False).sum().to_numpy()


def _bootstrap_sums(matrix: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n_units = matrix.shape[0]
    out = np.empty((n_boot, matrix.shape[1]))
    done = 0
    probabilities = np.full(n_units, 1.0 / n_units)
    while done < n_boot:
        size = min(BOOTSTRAP_CHUNK, n_boot - done)
        counts = rng.multinomial(n_units, probabilities, size=size)
        out[done : done + size] = counts @ matrix
        done += size
    return out


def _history_frame(
    df: pd.DataFrame,
    metric: Metric,
    split: str,
    segments: list[Segment],
    current_values: dict[str, float],
    history: list[tuple[Any, Any]],
) -> pd.DataFrame:
    rows = []
    for seg in segments:
        past = [
            c
            for pair in history
            if (c := segment_contribution(df, metric, split, pair[0], pair[1], seg.conditions))
            is not None
        ]
        current = current_values[seg.label()]
        if past:
            absolute = np.abs(np.array(past))
            rank = float(np.mean(absolute <= abs(current)))
            larger = bool(abs(current) > absolute.max())
        else:
            rank, larger = float("nan"), False
        rows.append(
            {
                "segment": seg.label(),
                "history_pairs": len(past),
                "rank_in_history": rank,
                "larger_than_all_history": larger,
            }
        )
    return pd.DataFrame(rows)


def noise_check(
    df: pd.DataFrame,
    metric: Metric,
    split: str,
    baseline: Any,
    current: Any,
    segments: list[Segment],
    unit: str | None = None,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    history: list[tuple[Any, Any]] | None = None,
) -> NoiseCheck:
    """Resample units within each side and give each segment's contribution an interval.

    A segment whose interval contains zero is flagged as within sampling noise. With ``history``,
    the current contribution is also ranked among the contributions of earlier period pairs."""
    if not 0 < level < 1:
        raise WhatChangedError("level must be between 0 and 1")
    if n_boot < 2:
        raise WhatChangedError("n_boot must be at least 2")
    if unit is not None and unit not in df.columns:
        raise WhatChangedError(f"unknown column {unit!r} for the unit")
    params = {"n_boot": n_boot, "level": level, "seed": seed, "unit": unit}
    if not segments:
        empty = pd.DataFrame(
            columns=["segment", "contribution", "ci_low", "ci_high", "within_noise"]
        )
        return NoiseCheck(metric.name, unit, empty, None, params)
    rng = np.random.default_rng(seed)
    sides = []
    for value in (baseline, current):
        rows = df[df[split] == value]
        if rows.empty:
            raise WhatChangedError(f"no rows have {split} == {value!r}")
        matrix = _unit_matrix(rows, metric, unit, segments)
        sides.append((matrix.sum(axis=0), _bootstrap_sums(matrix, n_boot, rng), matrix.shape[0]))
    (point0, boot0, units0), (point1, boot1, units1) = sides
    tail = (1 - level) / 2
    rows_out = []
    current_values: dict[str, float] = {}
    for i, seg in enumerate(segments):
        k = 2 + 2 * i

        point = float(
            contribution(
                metric,
                point0[0],
                point0[1],
                point1[0],
                point1[1],
                point0[k],
                point0[k + 1],
                point1[k],
                point1[k + 1],
            )
        )
        draws = np.asarray(
            contribution(
                metric,
                boot0[:, 0],
                boot0[:, 1],
                boot1[:, 0],
                boot1[:, 1],
                boot0[:, k],
                boot0[:, k + 1],
                boot1[:, k],
                boot1[:, k + 1],
            )
        )
        draws = draws[np.isfinite(draws)]
        low, high = (float(q) for q in np.quantile(draws, [tail, 1 - tail]))
        current_values[seg.label()] = point
        rows_out.append(
            {
                "segment": seg.label(),
                "contribution": point,
                "ci_low": low,
                "ci_high": high,
                "within_noise": bool(low <= 0.0 <= high),
                "units_baseline": units0,
                "units_current": units1,
            }
        )
    sampling = pd.DataFrame(rows_out)
    past = None
    if history:
        past = _history_frame(df, metric, split, segments, current_values, history)
    return NoiseCheck(metric.name, unit, sampling, past, params)
