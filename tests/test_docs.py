import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def python_blocks(path: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", (ROOT / path).read_text(), flags=re.S)


def test_the_documentation_code_blocks_run(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    for page in ("docs/index.md", "docs/case-study.md", "docs/flights.md"):
        namespace: dict[str, object] = {}
        for block in python_blocks(page):
            exec(compile(block, page, "exec"), namespace)
    out = capsys.readouterr().out
    assert "-7.747" in out and "0.273" in out and "0.451" in out


def test_the_case_study_numbers_are_the_ones_in_the_text():
    from whatchanged import Ratio, compare

    cells = pd.read_csv(ROOT / "tests" / "data" / "superstore_cells.csv")
    pair = cells[cells["region"].isin(["West", "Central"])]
    result = compare(
        pair, Ratio("profit", "sales"), "region", "West", "Central", ["category", "subcategory"]
    )
    parts = result.decompose(["category", "subcategory"]).components.set_index("component")[
        "contribution"
    ]
    text = (ROOT / "docs" / "case-study.md").read_text()
    for value in (
        100 * result.change.difference,
        100 * parts["mix:category"],
        100 * parts["mix:subcategory"],
        100 * parts["rate"],
    ):
        assert f"{value:.3f}" in text


def test_the_benchmark_page_matches_the_results_file():
    results = json.loads((ROOT / "benchmarks" / "results.json").read_text())
    page = (ROOT / "docs" / "benchmark.md").read_text()
    for name, item in results["expectations"].items():
        assert f"| {name} | {'passed' if item['passed'] else 'failed'} |" in page
