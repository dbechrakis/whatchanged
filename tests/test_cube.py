import numpy as np
import pandas as pd
import pytest

from whatchanged.cube import aggregate
from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Mean, Ratio, Sum, parse_metric


def table(seed: int = 0, rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "month": rng.choice(["a", "b"], rows),
            "region": rng.choice(["n", "s", "e"], rows),
            "kind": rng.choice(["x", "y"], rows),
            "sales": rng.uniform(1, 10, rows),
            "profit": rng.uniform(-1, 3, rows),
        }
    )


@pytest.mark.parametrize("metric", [Sum("sales"), Ratio("profit", "sales"), Mean("profit")])
def test_a_metric_from_cells_equals_the_metric_from_rows(metric):
    df = table()
    cube = aggregate(df, metric, "month", "a", "b", ["region", "kind"])
    for side, value in (("base", "a"), ("cur", "b")):
        rows = df[df["month"] == value]
        if isinstance(metric, Sum):
            expected = rows["sales"].sum()
        elif isinstance(metric, Ratio):
            expected = rows["profit"].sum() / rows["sales"].sum()
        else:
            expected = rows["profit"].mean()
        assert cube.value(side) == pytest.approx(expected)


def test_cells_missing_on_one_side_are_kept_with_zero_sums():
    df = table()
    df = df[~((df["month"] == "a") & (df["region"] == "e"))]
    cube = aggregate(df, Ratio("profit", "sales"), "month", "a", "b", ["region"])
    row = cube.cells[cube.cells["region"] == "e"].iloc[0]
    assert row["den_base"] == 0.0 and row["den_cur"] > 0


def test_unknown_columns_name_the_problem():
    with pytest.raises(WhatChangedError, match="unknown column"):
        aggregate(table(), Sum("revenue"), "month", "a", "b", ["region"])
    with pytest.raises(WhatChangedError, match="unknown column"):
        aggregate(table(), Sum("sales"), "month", "a", "b", ["country"])


def test_a_zero_denominator_is_refused():
    df = table()
    df.loc[df["month"] == "a", "sales"] = 0.0
    with pytest.raises(WhatChangedError, match="denominator"):
        aggregate(df, Ratio("profit", "sales"), "month", "a", "b", ["region"])


def test_an_empty_side_and_a_missing_split_value_are_refused():
    with pytest.raises(WhatChangedError, match="no rows have month == 'z'"):
        aggregate(table(), Sum("sales"), "month", "a", "z", ["region"])
    with pytest.raises(WhatChangedError, match="same"):
        aggregate(table(), Sum("sales"), "month", "a", "a", ["region"])


def test_dimension_problems_are_refused():
    df = table()
    with pytest.raises(WhatChangedError, match="at least one"):
        aggregate(df, Sum("sales"), "month", "a", "b", [])
    with pytest.raises(WhatChangedError, match="cannot also"):
        aggregate(df, Sum("sales"), "month", "a", "b", ["month"])
    df.loc[0, "region"] = None
    with pytest.raises(WhatChangedError, match="missing values"):
        aggregate(df, Sum("sales"), "month", "a", "b", ["region"])


def test_parse_metric():
    assert parse_metric("profit/sales") == Ratio("profit", "sales")
    assert parse_metric("sum:revenue") == Sum("revenue")
    assert parse_metric("mean:basket") == Mean("basket")
    for bad in ("profit/", "revenue", "avg:x"):
        with pytest.raises(WhatChangedError):
            parse_metric(bad)
