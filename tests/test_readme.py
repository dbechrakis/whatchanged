import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
RESULTS = json.loads((ROOT / "benchmarks" / "results.json").read_text())


def blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_the_five_minute_start_runs_unchanged(capsys):
    namespace: dict[str, object] = {}
    start = blocks(README)[0]
    exec(compile(start, "README start", "exec"), namespace)
    out = capsys.readouterr().out
    assert "profit/sales" in out and "cause" not in out.lower()


def test_the_numbers_in_where_it_fails_match_the_results_file():
    e = RESULTS["expectations"]
    table = RESULTS["table"]

    def f1(**kw):
        return next(r for r in table if all(r[k] == v for k, v in kw.items()))["f1"]

    recursive_5x4 = f1(
        scenario="one_element", noise="low", effect=0.1, shape="5x4", method="adtributor_recursive"
    )
    b1, b2, b3 = e["B1"]["values"], e["B2"]["values"], e["B3"]["values"]
    wanted = [
        f"F1 of {b1['adtributor']:.2f}, {b1['adtributor_recursive']:.2f} and {b1['decompose']:.2f}",
        f"drops to {recursive_5x4:.2f}",
        f"`top_delta` is wrong in {b2['top_delta_wrong']:.0%} of runs",
        f"median mix share is {b2['median_mix_share']:.0%}",
        f"Recall of {b3['adtributor']:.2f} and {b3['adtributor_recursive']:.2f}",
        f"names one in {e['B4']['values']['adtributor_named_a_segment']:.1%} of runs",
        f"{e['B5']['values']['kept_a_flagged_segment']:.1%} of runs keep one",
        f"on any scenario is {max(e['B6']['values'].values()):.2f}",
    ]
    for text in wanted:
        assert text in README, text
    for name, item in e.items():
        row = next(line for line in README.splitlines() if line.startswith(f"| {name}."))
        assert ("passed" if item["passed"] else "failed") in row


def test_the_run_follows_the_plan():
    assert RESULTS["seeds"] == 100 and RESULTS["rows"] == 5000 and RESULTS["n_boot"] == 200
    assert RESULTS["shapes"] == ["3x8", "5x4", "3x20"]
    runs = [r for r in RESULTS["table"] if r["shape"] == "all" and r["scenario"] == "one_element"]
    assert {r["runs"] for r in runs} == {300}
