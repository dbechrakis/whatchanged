import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 0.05


def load_runner():
    spec = importlib.util.spec_from_file_location("bench_run", ROOT / "benchmarks" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_reduced_rerun_stays_close_to_the_committed_reduced_results():
    bench = load_runner()
    committed = json.loads((ROOT / "benchmarks" / "reduced.json").read_text())["table"]
    records = bench.collect(10, bench.SHAPES[:1])
    fresh = bench.summarise(records)["table"]
    by_key = {
        (r["scenario"], r["noise"], r["effect"], r["shape"], r["method"]): r for r in committed
    }
    checked = 0
    for row in fresh:
        old = by_key[(row["scenario"], row["noise"], row["effect"], row["shape"], row["method"])]
        for field in ("f1", "named_any"):
            assert abs(row[field] - old[field]) <= TOLERANCE, (row, old)
            checked += 1
    assert checked > 100
