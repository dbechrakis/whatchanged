"""Tables with a planted change, for tests and the benchmark. Deterministic under a seed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from whatchanged.metrics import Ratio

SCENARIOS = ("one_element", "two_elements", "one_cell", "mix_shift", "uniform", "null")
NOISE_SD = {"low": 0.1, "high": 0.5}
RATE0 = 0.15
MIN_SHARE = 0.05
MIN_CELL_SHARE = 0.03
UNIT_SIZE = 3
OFFSET_SPAN = 0.08


@dataclass(frozen=True)
class Planted:
    """A table with two sides and the segments where the change was planted.

    ``planted`` lists segments, each a tuple of (dimension, value) pairs. It is empty for
    ``uniform`` and ``null``."""

    table: pd.DataFrame
    metric: Ratio
    split: str
    baseline: Any
    current: Any
    dimensions: list[str]
    planted: tuple[tuple[tuple[str, str], ...], ...]
    scenario: str
    effect: float
    unit: str = "unit"


def _zipf(n: int) -> np.ndarray:
    weights = 1.0 / np.arange(1, n + 1)
    return np.asarray(weights / weights.sum())


def _side(
    rng: np.random.Generator,
    probabilities: list[np.ndarray],
    names: list[list[str]],
    offsets: np.ndarray,
    rows: int,
    noise_sd: float,
    side: str,
) -> pd.DataFrame:
    codes = [rng.choice(len(p), size=rows, p=p) for p in probabilities]
    columns: dict[str, Any] = {
        f"d{i}": np.asarray(names[i])[codes[i]] for i in range(len(probabilities))
    }
    sales = np.exp(rng.normal(np.log(3.0), 0.5, rows))
    units = np.arange(rows) // UNIT_SIZE
    unit_shock = rng.normal(0.0, noise_sd / 2, units.max() + 1)[units]
    rate = RATE0 + offsets[codes[0]] + unit_shock + rng.normal(0.0, noise_sd, rows)
    frame = pd.DataFrame(columns)
    frame["sales"] = sales
    frame["profit"] = sales * rate
    frame["unit"] = [f"{side}{u}" for u in units]
    frame["side"] = side
    return frame


def _mask(table: pd.DataFrame, conditions: tuple[tuple[str, str], ...]) -> np.ndarray:
    mask = np.ones(len(table), dtype=bool)
    for column, value in conditions:
        mask &= (table[column] == value).to_numpy()
    return mask


def planted(
    scenario: str,
    seed: int = 0,
    n_dims: int = 3,
    n_elements: int = 8,
    effect: float = 0.10,
    noise: str = "low",
    rows: int = 5000,
) -> Planted:
    """Build a baseline and a current table over ``n_dims`` dimensions of ``n_elements`` elements.

    Cell sizes follow a Zipf law. The metric is profit over sales. The planted change moves the
    overall ratio by ``effect`` times its baseline value. Scenarios: ``one_element``,
    ``two_elements``, ``one_cell``, ``mix_shift``, ``uniform`` and ``null``."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if noise not in NOISE_SD:
        raise ValueError(f"noise must be one of {tuple(NOISE_SD)}")
    if n_dims < 2 or n_elements < 4:
        raise ValueError("use at least 2 dimensions of at least 4 elements")
    rng = np.random.default_rng(seed)
    base_p = _zipf(n_elements)
    names = [[f"v{j}" for j in range(n_elements)] for _ in range(n_dims)]
    perms = [rng.permutation(n_elements) for _ in range(n_dims)]
    probabilities = [base_p[np.argsort(perm)] for perm in perms]
    order0 = np.argsort(-probabilities[0])
    offsets = np.zeros(n_elements)
    ranked = np.linspace(OFFSET_SPAN, -OFFSET_SPAN, n_elements)
    offsets[order0[0]] = ranked[0]
    offsets[order0[1]] = ranked[-1]
    offsets[order0[2:]] = np.linspace(OFFSET_SPAN / 2, -OFFSET_SPAN / 2, n_elements - 2)
    sd = NOISE_SD[noise]
    base = _side(rng, probabilities, names, offsets, rows, sd, "a")
    current_p = [p.copy() for p in probabilities]
    chosen: list[tuple[tuple[str, str], ...]] = []
    if scenario == "mix_shift":
        high, low = int(order0[0]), int(order0[1])
        gap = offsets[high] - offsets[low]
        r0 = float(base["profit"].sum() / base["sales"].sum())
        mass = min(effect * r0 / gap, 0.95 * current_p[0][high])
        current_p[0][high] -= mass
        current_p[0][low] += mass
        chosen = [(("d0", names[0][high]),), (("d0", names[0][low]),)]
    cur = _side(rng, current_p, names, offsets, rows, sd, "b")
    r0 = float(base["profit"].sum() / base["sales"].sum())
    total_sales = float(cur["sales"].sum())
    if scenario == "uniform":
        cur["profit"] = cur["profit"] * (1 + effect)
    elif scenario in ("one_element", "two_elements"):
        dim = f"d{int(rng.integers(n_dims))}"
        shares = cur.groupby(dim)["sales"].sum() / total_sales
        eligible = [str(v) for v in shares.index[shares >= MIN_SHARE]]
        take = 1 if scenario == "one_element" else 2
        if len(eligible) < take:
            eligible = [str(v) for v in shares.sort_values(ascending=False).index[:take]]
        values = list(rng.choice(eligible, size=take, replace=False))
        chosen = [((dim, str(v)),) for v in values]
    elif scenario == "one_cell":
        picks = rng.choice(n_dims, size=2, replace=False)
        dims = (f"d{int(picks[0])}", f"d{int(picks[1])}")
        cells = cur.groupby(list(dims))["sales"].sum() / total_sales
        eligible_cells = list(cells.index[cells >= MIN_CELL_SHARE])
        if not eligible_cells:
            eligible_cells = [cells.idxmax()]
        pair = eligible_cells[int(rng.integers(len(eligible_cells)))]
        chosen = [((dims[0], str(pair[0])), (dims[1], str(pair[1])))]
    if scenario in ("one_element", "two_elements", "one_cell"):
        union = np.zeros(len(cur), dtype=bool)
        for conditions in chosen:
            union |= _mask(cur, conditions)
        share = float(cur.loc[union, "sales"].sum() / total_sales)
        shift = effect * r0 / share
        cur.loc[union, "profit"] = cur.loc[union, "profit"] + cur.loc[union, "sales"] * shift
    table = pd.concat([base, cur], ignore_index=True)
    return Planted(
        table=table,
        metric=Ratio("profit", "sales"),
        split="side",
        baseline="a",
        current="b",
        dimensions=[f"d{i}" for i in range(n_dims)],
        planted=tuple(chosen),
        scenario=scenario,
        effect=effect,
    )
