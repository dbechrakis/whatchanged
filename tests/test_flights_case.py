import re
from pathlib import Path

import pandas as pd
import pytest

import whatchanged as wc

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "data" / "flights_cells.csv"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="the flights fixture is not built")


def case():
    cells = pd.read_csv(FIXTURE)
    return cells, wc.compare(
        cells, wc.Ratio("late", "flights"), "month", "2024-05", "2024-06", ["carrier", "origin"]
    )


def test_the_decomposition_adds_up_and_the_page_quotes_it():
    cells, result = case()
    change = result.change
    parts = result.decompose(["carrier", "origin"], missing="entry_exit")
    assert parts.components["contribution"].sum() == pytest.approx(change.difference, abs=1e-12)
    text = (ROOT / "docs" / "flights.md").read_text()
    assert (
        f"{100 * change.baseline_value:.2f}%" in text
        and f"{100 * change.current_value:.2f}%" in text
    )
    for row in parts.components.itertuples():
        assert f"{100 * row.contribution:+.2f}" in text, row.component


def test_the_history_ranks_are_between_zero_and_one_and_quoted():
    cells, result = case()
    explanation = result.explain(method="recursive")
    months = sorted(cells["month"].unique())
    pairs = [(months[i], months[i + 1]) for i in range(len(months) - 2)]
    assert len(pairs) == 12
    check = result.noise(segments=list(explanation.segments), history=pairs, n_boot=20)
    if check.history is not None:
        assert check.history["rank_in_history"].between(0, 1).all()
        text = (ROOT / "docs" / "flights.md").read_text()
        for row in check.history.itertuples():
            assert f"{row.rank_in_history:.2f}" in text
    assert (
        re.search(
            r"cause|because",
            (ROOT / "docs" / "flights.md").read_text().replace("does not say why", ""),
        )
        is None
    )
