import numpy as np
import pandas as pd
import pytest

from whatchanged import Segment, compare
from whatchanged.noise import segment_contribution
from whatchanged.synthetic import SCENARIOS, planted


def overall(p, side):
    rows = p.table[p.table["side"] == side]
    return rows["profit"].sum() / rows["sales"].sum()


@pytest.mark.parametrize("scenario", ["one_element", "two_elements", "one_cell", "uniform"])
def test_the_planted_change_moves_the_total_by_the_stated_amount(scenario):
    moves = []
    for seed in range(20):
        p = planted(scenario, seed=seed, effect=0.10, rows=20000)
        moves.append(overall(p, "b") / overall(p, "a") - 1)
    assert np.mean(moves) == pytest.approx(0.10, abs=0.02)


def test_null_moves_nothing_and_is_deterministic():
    moves = [
        overall(p, "b") / overall(p, "a") - 1
        for p in (planted("null", seed=s, rows=20000) for s in range(20))
    ]
    assert abs(np.mean(moves)) < 0.02
    first, second = planted("null", seed=3), planted("null", seed=3)
    pd.testing.assert_frame_equal(first.table, second.table)
    assert first.planted == ()


def test_in_mix_shift_every_elements_own_rate_is_unchanged_and_the_total_falls():
    p = planted("mix_shift", seed=1, effect=0.10, rows=400000, noise="low")
    table = p.table
    rates = (
        table.groupby(["side", "d0"])
        .apply(lambda g: g["profit"].sum() / g["sales"].sum(), include_groups=False)
        .unstack(0)
    )
    assert (rates["b"] - rates["a"]).abs().max() < 0.01
    assert overall(p, "b") / overall(p, "a") - 1 == pytest.approx(-0.10, abs=0.02)
    assert len(p.planted) == 2


def test_scenarios_and_arguments():
    assert set(SCENARIOS) == {
        "one_element",
        "two_elements",
        "one_cell",
        "mix_shift",
        "uniform",
        "null",
    }
    with pytest.raises(ValueError):
        planted("nothing")
    with pytest.raises(ValueError):
        planted("null", noise="loud")
    with pytest.raises(ValueError):
        planted("null", n_elements=2)
    assert len(planted("two_elements", seed=2).planted) == 2
    assert len(planted("one_cell", seed=2).planted[0]) == 2


def test_the_bootstrap_intervals_contain_zero_on_null_for_most_elements():
    covered = []
    for seed in range(12):
        p = planted("null", seed=seed, rows=4000)
        c = compare(p.table, p.metric, p.split, p.baseline, p.current, p.dimensions)
        segments = [Segment((("d0", f"v{i}"),), 0.0, 0.0) for i in range(8)]
        check = c.noise(unit="unit", segments=segments, n_boot=300, seed=seed)
        covered.extend(check.sampling["within_noise"].tolist())
    assert np.mean(covered) >= 0.9


def test_on_a_ten_percent_change_the_planted_element_is_flagged():
    flagged = 0
    for seed in range(10):
        p = planted("one_element", seed=seed, effect=0.10, rows=20000)
        c = compare(p.table, p.metric, p.split, p.baseline, p.current, p.dimensions)
        segment = Segment(p.planted[0], 0.0, 0.0)
        check = c.noise(unit="unit", segments=[segment], n_boot=300, seed=seed)
        flagged += int(not check.sampling["within_noise"].iloc[0])
    assert flagged >= 9


def test_the_noise_check_without_segments_uses_the_search_and_handles_none():
    p = planted("one_element", seed=1, effect=0.10, rows=20000)
    c = compare(p.table, p.metric, p.split, p.baseline, p.current, p.dimensions)
    check = c.noise(unit="unit", n_boot=100)
    assert check.kind == "statistical" and len(check.sampling) >= 1
    u = planted("uniform", seed=1, effect=0.10, rows=20000)
    cu = compare(u.table, u.metric, u.split, u.baseline, u.current, u.dimensions)
    assert cu.noise(unit="unit", n_boot=100).sampling.empty


def test_the_history_rank_is_between_zero_and_one_and_uniform_on_exchangeable_history():
    rng = np.random.default_rng(0)
    months = [f"m{i}" for i in range(14)]
    rows = []
    for month in months:
        for _ in range(600):
            region = rng.choice(["n", "s"])
            sales = rng.uniform(1, 3)
            rows.append((month, region, sales, sales * (0.2 + rng.normal(0, 0.3))))
    df = pd.DataFrame(rows, columns=["month", "region", "sales", "profit"])
    from whatchanged import Ratio

    segment = Segment((("region", "n"),), 0.0, 0.0)
    history = [(months[i], months[i + 1]) for i in range(12)]
    ranks = []
    for k in range(40):
        sub = df[df["month"].isin([*months[:12], months[12 + (k % 2)]])]
        c = compare(
            sub, Ratio("profit", "sales"), "month", months[11], months[12 + (k % 2)], ["region"]
        )
        check = c.noise(segments=[segment], history=history[:11], n_boot=20)
        ranks.append(float(check.history["rank_in_history"].iloc[0]))
    assert all(0.0 <= r <= 1.0 for r in ranks)
    assert 0.25 < np.mean(ranks) < 0.75


def test_segment_contribution_matches_a_hand_calculation():
    df = pd.DataFrame(
        {
            "side": ["a", "a", "b", "b"],
            "region": ["n", "s", "n", "s"],
            "sales": [10.0, 10.0, 10.0, 10.0],
            "profit": [2.0, 1.0, 4.0, 1.0],
        }
    )
    from whatchanged import Ratio

    value = segment_contribution(df, Ratio("profit", "sales"), "side", "a", "b", (("region", "n"),))
    assert value == pytest.approx((4.0 + 1.0) / 20.0 - 3.0 / 20.0)
    assert segment_contribution(df, Ratio("profit", "sales"), "side", "a", "zz", ()) is None
