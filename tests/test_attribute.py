import numpy as np
import pandas as pd
import pytest

from whatchanged.attribute import (
    additive_power,
    explain,
    js_surprise,
    ratio_power,
)
from whatchanged.compare import compare
from whatchanged.cube import aggregate
from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Ratio, Sum


def paper_example():
    revenue_f = np.array([50.0, 0.0, 40.0, 10.0])
    revenue_a = np.array([10.0, 0.0, 70.0, 10.0])
    clicks_f = np.array([100.0, 200.0, 100.0, 100.0])
    clicks_a = np.array([20.0, 360.0, 100.0, 100.0])
    return revenue_f, revenue_a, clicks_f, clicks_a


def test_the_papers_worked_example_for_cost_per_click():
    f1, a1, f2, a2 = paper_example()
    power = ratio_power(f1, a1, f2, a2)
    assert power is not None
    assert np.round(power * 100).tolist() == [125, 106, -131, 0]
    assert power.sum() == pytest.approx(1.0)


def test_the_papers_additive_example_of_the_data_center():
    power = additive_power(np.array([500_000.0, 500_000.0]), np.array([400_000.0, 400_000.0]))
    assert power.tolist() == [0.5, 0.5]
    single = additive_power(np.array([1_000_000.0, 0.0]), np.array([800_000.0, 0.0]))
    assert single[0] == pytest.approx(1.0)


def test_surprise_is_zero_for_equal_distributions_and_bounded_by_one():
    same = js_surprise(np.array([1.0, 3.0]), np.array([2.0, 6.0]))
    assert same == pytest.approx([0.0, 0.0])
    full = js_surprise(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert full.sum() == pytest.approx(1.0)
    assert (js_surprise(np.array([5.0, 0.0, 2.0]), np.array([0.0, 4.0, 4.0])) >= 0).all()


def frame(rows):
    return pd.DataFrame(rows, columns=["side", "region", "channel", "kind", "v"])


def hand_example():
    rows = []
    for region, channel, kind, base, cur in [
        ("n", "web", "a", 100, 100),
        ("n", "web", "b", 50, 50),
        ("n", "shop", "a", 80, 80),
        ("s", "web", "a", 120, 20),
        ("s", "shop", "a", 60, 60),
        ("s", "shop", "b", 40, 40),
        ("e", "web", "b", 70, 70),
        ("e", "shop", "a", 30, 30),
    ]:
        rows.append(("a", region, channel, kind, base))
        rows.append(("b", region, channel, kind, cur))
    return frame(rows)


def test_a_hand_worked_example_with_three_dimensions():
    cube = aggregate(hand_example(), Sum("v"), "side", "a", "b", ["region", "channel", "kind"])
    result = explain(cube)
    assert result.found and result.kind == "heuristic"
    top = result.candidates[0]
    assert top.dimension == "region"
    assert [seg.conditions for seg in top.segments] == [(("region", "s"),)]
    assert top.explains == pytest.approx(1.0)


def test_recursive_search_reaches_a_cell_of_two_dimensions():
    cube = aggregate(hand_example(), Sum("v"), "side", "a", "b", ["region", "channel", "kind"])
    result = explain(cube, method="recursive", depth=3)
    labels = {seg.label() for seg in result.segments}
    assert labels == {"region = s and channel = web"}


def test_degenerate_cases_one_element_zero_change_and_all_change_in_one_element():
    one = frame([("a", "n", "web", "a", 5.0), ("b", "n", "web", "a", 9.0)])
    cube = aggregate(one, Sum("v"), "side", "a", "b", ["region"])
    assert explain(cube).found is False
    flat = hand_example()
    flat.loc[flat["side"] == "b", "v"] = flat.loc[flat["side"] == "a", "v"].to_numpy()
    cube = aggregate(flat, Sum("v"), "side", "a", "b", ["region", "channel"])
    answer = explain(cube)
    assert answer.found is False and "no change" in answer.message.lower()
    one_mover = hand_example()
    cube = aggregate(one_mover, Sum("v"), "side", "a", "b", ["region"])
    assert explain(cube).segments[0].explanatory_power == pytest.approx(1.0)


def test_a_change_spread_over_every_element_returns_the_answer_that_no_segment_explains_it():
    rows = []
    for region in "abcdefgh":
        for channel in "xyzw":
            rows.append(("a", region, channel, "k", 10.0))
            rows.append(("b", region, channel, "k", 11.0))
    cube = aggregate(frame(rows), Sum("v"), "side", "a", "b", ["region", "channel"])
    result = explain(cube)
    assert result.found is False
    assert result.segments == () and result.candidates == ()
    assert "No segment explains" in result.message


def test_ratio_metric_search_finds_the_moving_rate():
    rng = np.random.default_rng(1)
    rows = []
    for region in "abcd":
        for side in "ab":
            for _ in range(200):
                sales = rng.uniform(1, 2)
                rate = 0.2 - (0.1 if (region == "c" and side == "b") else 0.0)
                rows.append((side, region, "x", "k", sales, sales * rate))
    df = pd.DataFrame(rows, columns=["side", "region", "channel", "kind", "sales", "profit"])
    cube = aggregate(df, Ratio("profit", "sales"), "side", "a", "b", ["region", "channel"])
    result = explain(cube)
    assert result.found and result.segments[0].conditions == (("region", "c"),)


def test_arguments_are_validated():
    cube = aggregate(hand_example(), Sum("v"), "side", "a", "b", ["region"])
    for kwargs in ({"method": "x"}, {"teep": 0.9, "tep": 0.5}, {"spread_at": 0}, {"depth": 0}):
        with pytest.raises(WhatChangedError):
            explain(cube, **kwargs)


def test_summary_is_short_and_never_says_cause_or_because():
    comparison = compare(hand_example(), Sum("v"), "side", "a", "b", ["region", "channel", "kind"])
    text = comparison.summary()
    assert text.count(". ") + 1 <= 5
    lowered = text.lower()
    assert "cause" not in lowered and "because" not in lowered and ";" not in text
