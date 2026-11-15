import json

import duckdb
import pandas as pd
import pytest

from whatchanged import Mean, Ratio, Sum, compare
from whatchanged.cli import main
from whatchanged.errors import WhatChangedError
from whatchanged.synthetic import planted


def table():
    p = planted("one_element", seed=4, rows=3000)
    return p, p.table


@pytest.mark.parametrize("metric", [Ratio("profit", "sales"), Sum("profit"), Mean("profit")])
def test_both_paths_give_the_same_cells(metric):
    p, df = table()
    con = duckdb.connect()
    con.register("orders", df)
    sql = compare(con, metric, "side", "a", "b", ["d0", "d1"], name="orders")
    ref = compare(df, metric, "side", "a", "b", ["d0", "d1"])
    cols = ["d0", "d1", "num_base", "num_cur"]
    a = sql.cube.cells.sort_values(["d0", "d1"]).reset_index(drop=True)[cols]
    b = ref.cube.cells.sort_values(["d0", "d1"]).reset_index(drop=True)[cols]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    assert sql.change.difference == pytest.approx(ref.change.difference)
    assert sql.explain().to_dict() == ref.explain().to_dict()


def test_noise_through_a_connection_loads_only_the_needed_rows():
    p, df = table()
    con = duckdb.connect()
    con.register("orders", df)
    c = compare(con, p.metric, "side", "a", "b", p.dimensions, name="orders")
    check = c.noise(unit="unit", n_boot=50)
    assert check.kind == "statistical"


def test_sql_errors_name_the_problem_and_identifiers_are_quoted():
    p, df = table()
    con = duckdb.connect()
    con.register("orders", df)
    with pytest.raises(WhatChangedError, match="unknown column"):
        compare(con, p.metric, "side", "a", "b", ["nope"], name="orders")
    with pytest.raises(WhatChangedError, match="no rows have"):
        compare(con, p.metric, "side", "a", "zz", ["d0"], name="orders")
    with pytest.raises(WhatChangedError, match="table name"):
        compare(con, p.metric, "side", "a", "b", ["d0"])
    weird = df.rename(columns={"d0": 'd"0; DROP TABLE x'})
    con.register("weird", weird)
    result = compare(con, p.metric, "side", "a", "b", ['d"0; DROP TABLE x'], name="weird")
    assert len(result.cube.cells) > 0
    with pytest.raises(WhatChangedError):
        compare(con, p.metric, "side", "a", "b", ["d0"], name="orders; DROP TABLE orders")
    assert con.execute("SELECT COUNT(*) FROM orders").fetchall()[0][0] == len(df)


def test_the_command_line_prints_a_summary_and_json(tmp_path, capsys):
    p, df = table()
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    args = [
        str(path),
        "--metric",
        "profit/sales",
        "--split",
        "side",
        "--baseline",
        "a",
        "--current",
        "b",
        "--by",
        "d0,d1,d2",
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "profit/sales" in text and "cause" not in text.lower()
    assert main([*args, "--json", "--unit", "unit"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["change"]["kind"] == "exact" and out["explain"]["kind"] == "heuristic"
    assert out["noise"]["kind"] == "statistical"


def test_the_command_line_reports_errors_with_a_status(tmp_path, capsys):
    assert (
        main(
            [
                str(tmp_path / "missing.csv"),
                "--metric",
                "a/b",
                "--split",
                "s",
                "--baseline",
                "x",
                "--current",
                "y",
                "--by",
                "d",
            ]
        )
        == 2
    )
    assert "no such file" in capsys.readouterr().err
    path = tmp_path / "t.txt"
    path.write_text("x")
    assert (
        main(
            [
                str(path),
                "--metric",
                "a/b",
                "--split",
                "s",
                "--baseline",
                "x",
                "--current",
                "y",
                "--by",
                "d",
            ]
        )
        == 2
    )
