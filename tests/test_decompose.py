from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from whatchanged.cube import aggregate
from whatchanged.decompose import MAX_FACTORS, decompose
from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Ratio, Sum

DATA = Path(__file__).parent / "data" / "superstore_cells.csv"


def long_table(cells: pd.DataFrame) -> pd.DataFrame:
    base = cells.assign(side="a", num=cells["num_base"], den=cells["den_base"])
    cur = cells.assign(side="b", num=cells["num_cur"], den=cells["den_cur"])
    both = pd.concat([base, cur], ignore_index=True)
    return both[both["den"] > 0].drop(columns=["num_base", "den_base", "num_cur", "den_cur"])


def build(cells: pd.DataFrame, dims: list[str]):
    return aggregate(long_table(cells), Ratio("num", "den"), "side", "a", "b", dims)


def grid(draw_num, draw_den, levels=(3, 3)) -> pd.DataFrame:
    keys = np.stack(np.meshgrid(*[np.arange(n) for n in levels], indexing="ij"), -1).reshape(
        -1, len(levels)
    )
    cells = pd.DataFrame(keys, columns=[f"d{i}" for i in range(len(levels))]).astype(str)
    cells["num_base"], cells["den_base"] = draw_num[0], draw_den[0]
    cells["num_cur"], cells["den_cur"] = draw_num[1], draw_den[1]
    return cells


positive = st.floats(min_value=0.5, max_value=50, allow_nan=False)


@st.composite
def tables(draw, levels=(3, 3)):
    n = int(np.prod(levels))
    arr = lambda: np.array(draw(st.lists(positive, min_size=n, max_size=n)))  # noqa: E731
    return grid((arr(), arr()), (arr(), arr()), levels)


FAST = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def test_the_superstore_gap_is_reproduced_to_three_decimals():
    raw = pd.read_csv(DATA)
    pair = raw[raw["region"].isin(["West", "Central"])]
    cube = aggregate(
        pair, Ratio("profit", "sales"), "region", "West", "Central", ["category", "subcategory"]
    )
    result = decompose(cube, ["category", "subcategory"])
    pp = {row.component: 100 * row.contribution for row in result.components.itertuples()}
    assert 100 * result.change == pytest.approx(-7.023, abs=5e-4)
    assert pp["mix:category"] == pytest.approx(0.273, abs=5e-4)
    assert pp["mix:subcategory"] == pytest.approx(0.451, abs=5e-4)
    assert pp["rate"] == pytest.approx(-7.747, abs=5e-4)
    assert sum(pp.values()) == pytest.approx(100 * result.change, abs=1e-9)


@FAST
@given(tables())
def test_the_components_sum_to_the_change(cells):
    result = decompose(build(cells, ["d0", "d1"]), ["d0", "d1"])
    assert result.components["contribution"].sum() == pytest.approx(result.change, abs=1e-9)


@FAST
@given(tables())
def test_swapping_baseline_and_current_flips_every_sign(cells):
    forward = decompose(build(cells, ["d0", "d1"]), ["d0", "d1"]).components.set_index("component")[
        "contribution"
    ]
    swapped = cells.rename(
        columns={
            "num_base": "num_cur",
            "num_cur": "num_base",
            "den_base": "den_cur",
            "den_cur": "den_base",
        }
    )
    backward = decompose(build(swapped, ["d0", "d1"]), ["d0", "d1"]).components.set_index(
        "component"
    )["contribution"]
    np.testing.assert_allclose(
        backward.to_numpy(), -forward.reindex(backward.index).to_numpy(), atol=1e-9
    )


@FAST
@given(tables(), st.permutations(range(3)))
def test_relabelling_elements_changes_nothing(cells, order):
    first = decompose(build(cells, ["d0", "d1"]), ["d0", "d1"]).components.set_index("component")[
        "contribution"
    ]
    relabelled = cells.assign(d0=cells["d0"].map({str(i): f"z{order[i]}" for i in range(3)}))
    second = decompose(build(relabelled, ["d0", "d1"]), ["d0", "d1"]).components.set_index(
        "component"
    )["contribution"]
    np.testing.assert_allclose(second.reindex(first.index).to_numpy(), first.to_numpy(), atol=1e-9)


@FAST
@given(tables(), st.floats(min_value=0.01, max_value=1000))
def test_scaling_numerator_and_denominator_by_a_constant_changes_nothing(cells, factor):
    first = (
        decompose(build(cells, ["d0", "d1"]), ["d0", "d1"]).components["contribution"].to_numpy()
    )
    scaled = cells.assign(
        num_base=cells["num_base"] * factor,
        num_cur=cells["num_cur"] * factor,
        den_base=cells["den_base"] * factor,
        den_cur=cells["den_cur"] * factor,
    )
    second = (
        decompose(build(scaled, ["d0", "d1"]), ["d0", "d1"]).components["contribution"].to_numpy()
    )
    np.testing.assert_allclose(second, first, rtol=1e-8, atol=1e-9)


@FAST
@given(tables())
def test_identical_sides_give_zero_contributions(cells):
    same = cells.assign(num_cur=cells["num_base"], den_cur=cells["den_base"])
    result = decompose(build(same, ["d0", "d1"]), ["d0", "d1"])
    assert np.allclose(result.components["contribution"], 0.0, atol=1e-12)
    assert np.allclose(result.cells["contribution"], 0.0, atol=1e-12)


def test_a_pure_mix_shift_has_no_rate_component():
    cells = pd.DataFrame(
        {
            "d0": ["hi", "lo"],
            "num_base": [0.5 * 80, 0.1 * 20],
            "den_base": [80.0, 20.0],
            "num_cur": [0.5 * 20, 0.1 * 80],
            "den_cur": [20.0, 80.0],
        }
    )
    result = decompose(build(cells, ["d0"]), ["d0"])
    parts = result.components.set_index("component")["contribution"]
    assert parts["rate"] == pytest.approx(0.0, abs=1e-12)
    assert parts["mix:d0"] == pytest.approx(result.change)
    assert result.change < 0


def two_sided_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d0": ["a", "b", "c", "d"],
            "num_base": [10.0, 20.0, 5.0, 0.0],
            "den_base": [100.0, 100.0, 50.0, 0.0],
            "num_cur": [12.0, 18.0, 0.0, 9.0],
            "den_cur": [100.0, 90.0, 0.0, 30.0],
        }
    )


def test_cells_on_one_side_only_are_refused_by_default_and_reported_with_entry_exit():
    cube = build(two_sided_missing(), ["d0"])
    with pytest.raises(WhatChangedError, match="on one side only"):
        decompose(cube, ["d0"])
    result = decompose(cube, ["d0"], missing="entry_exit")
    parts = result.components.set_index("component")["contribution"]
    assert {"entered", "left"} <= set(parts.index)
    assert parts.sum() == pytest.approx(result.change, abs=1e-12)
    assert parts["entered"] == pytest.approx(9.0 / 220.0)
    assert parts["left"] == pytest.approx(-5.0 / 250.0)


def test_a_sum_metric_is_the_plain_difference_per_cell():
    df = pd.DataFrame(
        {"m": ["a", "a", "b", "b"], "k": ["x", "y", "x", "y"], "v": [5.0, 3.0, 9.0, 2.0]}
    )
    cube = aggregate(df, Sum("v"), "m", "a", "b", ["k"])
    result = decompose(cube, ["k"])
    assert result.change == pytest.approx(3.0)
    deltas = result.cells.set_index("k")["contribution"]
    assert deltas["x"] == pytest.approx(4.0) and deltas["y"] == pytest.approx(-1.0)


def test_too_many_factors_and_unknown_dimensions_are_refused():
    cells = pd.DataFrame({f"d{i}": ["a", "b"] for i in range(6)})
    for column in ("num_base", "den_base", "num_cur", "den_cur"):
        cells[column] = 1.0
    cube = build(cells, [f"d{i}" for i in range(6)])
    with pytest.raises(WhatChangedError, match="most allowed"):
        decompose(cube, [f"d{i}" for i in range(MAX_FACTORS)])
    with pytest.raises(WhatChangedError, match="unknown dimension"):
        decompose(cube, ["nope"])
    with pytest.raises(WhatChangedError, match="at least one"):
        decompose(cube, [])
    with pytest.raises(WhatChangedError, match="missing must be"):
        decompose(cube, ["d0"], missing="skip")  # type: ignore[arg-type]


def revenue_table():
    return pd.DataFrame(
        {
            "side": ["a", "a", "a", "b", "b", "b"],
            "region": ["n", "s", "e", "n", "s", "w"],
            "revenue": [100.0, 60.0, 20.0, 150.0, 40.0, 30.0],
            "units": [10.0, 5.0, 4.0, 10.0, 5.0, 6.0],
        }
    )


def test_a_sum_with_units_splits_into_volume_and_value_per_unit():
    from whatchanged import Sum

    df = revenue_table().iloc[[0, 1, 3, 4]]
    cube = aggregate(df, Sum("revenue", units="units"), "side", "a", "b", ["region"])
    result = decompose(cube, ["region"])
    parts = result.components.set_index("component")["contribution"]
    assert parts["volume"] == pytest.approx(0.0)
    assert parts["value"] == pytest.approx(30.0)
    assert parts.sum() == pytest.approx(result.change) and result.kind == "exact"
    by_cell = result.cells.set_index(["region", "component"])["contribution"]
    assert by_cell[("n", "value")] == pytest.approx(50.0)
    assert by_cell[("s", "value")] == pytest.approx(-20.0)


def test_volume_and_value_with_cells_that_enter_and_leave():
    from whatchanged import Sum

    cube = aggregate(revenue_table(), Sum("revenue", units="units"), "side", "a", "b", ["region"])
    with pytest.raises(WhatChangedError, match="on one side only"):
        decompose(cube, ["region"])
    result = decompose(cube, ["region"], missing="entry_exit")
    parts = result.components.set_index("component")["contribution"]
    assert parts["entered"] == pytest.approx(30.0) and parts["left"] == pytest.approx(-20.0)
    assert parts.sum() == pytest.approx(result.change)


def test_volume_and_value_agree_through_duckdb():
    import duckdb

    from whatchanged import Sum, compare

    df = revenue_table()
    con = duckdb.connect()
    con.register("t", df)
    sql = compare(con, Sum("revenue", units="units"), "side", "a", "b", ["region"], name="t")
    ref = compare(df, Sum("revenue", units="units"), "side", "a", "b", ["region"])
    a = sql.decompose(missing="entry_exit").components.set_index("component")["contribution"]
    b = ref.decompose(missing="entry_exit").components.set_index("component")["contribution"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
