"""The benchmark of docs/benchmark-plan.md. Writes results.json and results.md next to it."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from whatchanged import compare
from whatchanged.synthetic import Planted, planted

SHAPES = ((3, 8), (5, 4), (3, 20))
SEEDS = 100
ROWS = 5000
N_BOOT = 200
EFFECTS = (0.02, 0.10)
NOISES = ("low", "high")
SCENARIOS = ("one_element", "two_elements", "one_cell", "mix_shift", "uniform", "null")
METHODS = ("top_delta", "adtributor", "adtributor_recursive", "decompose")
HERE = Path(__file__).parent


def key(segment: tuple[tuple[str, Any], ...]) -> frozenset[tuple[str, str]]:
    return frozenset((str(a), str(b)) for a, b in segment)


def rank_elements(c: Any, method: str, k: int) -> list[frozenset[tuple[str, str]]]:
    scored: list[tuple[float, frozenset[tuple[str, str]]]] = []
    for dim in c.dimensions:
        if method == "top_delta":
            g = c.cube.cells.groupby(dim)[["num_base", "den_base", "num_cur", "den_cur"]].sum()
            g = g[(g["den_base"] > 0) & (g["den_cur"] > 0)]
            change = g["num_cur"] / g["den_cur"] - g["num_base"] / g["den_base"]
            for element, value in change.items():
                scored.append((abs(float(value)), frozenset({(dim, str(element))})))
        else:
            cells = c.decompose([dim], missing="entry_exit").cells
            totals = cells.groupby(dim)["contribution"].sum()
            for element, value in totals.items():
                scored.append((abs(float(value)), frozenset({(dim, str(element))})))
    scored.sort(key=lambda item: -item[0])
    return [item[1] for item in scored[:k]]


def f1_scores(
    returned: set[frozenset[tuple[str, str]]], truth: set[frozenset[tuple[str, str]]]
) -> tuple[float, float, float]:
    hits = len(returned & truth)
    precision = hits / len(returned) if returned else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def one_run(p: Planted, scenario: str) -> dict[str, Any]:
    c = compare(p.table, p.metric, p.split, p.baseline, p.current, p.dimensions)
    truth = {key(seg) for seg in p.planted}
    k = max(1, len(truth))
    out: dict[str, Any] = {}
    single = c.explain(method="single")
    recursive = c.explain(method="recursive", depth=2)
    returned = {
        "adtributor": {key(s.conditions) for s in single.segments},
        "adtributor_recursive": {key(s.conditions) for s in recursive.segments},
        "top_delta": set(rank_elements(c, "top_delta", k)),
        "decompose": set(rank_elements(c, "decompose", k)),
    }
    for method in METHODS:
        precision, recall, f1 = f1_scores(returned[method], truth)
        out[method] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "named_any": bool(returned[method]),
            "wrong_top": bool(returned[method] and not (returned[method] & truth)),
        }
    if scenario == "mix_shift":
        parts = c.decompose(["d0"]).components.set_index("component")["contribution"]
        change = c.change.difference
        out["mix_share"] = float(parts["mix:d0"] / change) if change else float("nan")
    if scenario == "null" and single.found:
        check = c.noise(unit=p.unit, segments=list(single.segments), n_boot=N_BOOT, seed=0)
        out["kept_flagged"] = bool((~check.sampling["within_noise"]).any())
    elif scenario == "null":
        out["kept_flagged"] = False
    return out


def collect(seeds: int, shapes: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    records = []
    started = time.time()
    for shape_index, (n_dims, n_elements) in enumerate(shapes):
        for scenario in SCENARIOS:
            effects = (0.10,) if scenario == "null" else EFFECTS
            for noise in NOISES:
                for effect in effects:
                    for i in range(seeds):
                        p = planted(
                            scenario,
                            seed=1000 * shape_index + i,
                            n_dims=n_dims,
                            n_elements=n_elements,
                            effect=effect,
                            noise=noise,
                            rows=ROWS,
                        )
                        run = one_run(p, scenario)
                        records.append(
                            {
                                "shape": f"{n_dims}x{n_elements}",
                                "scenario": scenario,
                                "noise": noise,
                                "effect": effect,
                                "seed": i,
                                "result": run,
                            }
                        )
            print(
                f"shape {n_dims}x{n_elements} {scenario} done in {time.time() - started:.0f}s",
                flush=True,
            )
    return records


def mean(records: list[dict[str, Any]], method: str, field: str) -> float:
    values = [r["result"][method][field] for r in records]
    return float(np.mean(values)) if values else float("nan")


def select(records: list[dict[str, Any]], **where: Any) -> list[dict[str, Any]]:
    return [r for r in records if all(r[k] == v for k, v in where.items())]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    table = []
    shapes = sorted({r["shape"] for r in records})
    for scenario in SCENARIOS:
        for noise in NOISES:
            for effect in sorted({r["effect"] for r in select(records, scenario=scenario)}):
                for shape in ("all", *shapes):
                    rows = select(records, scenario=scenario, noise=noise, effect=effect)
                    if shape != "all":
                        rows = [r for r in rows if r["shape"] == shape]
                    for method in METHODS:
                        table.append(
                            {
                                "scenario": scenario,
                                "noise": noise,
                                "effect": effect,
                                "shape": shape,
                                "method": method,
                                "runs": len(rows),
                                "precision": mean(rows, method, "precision"),
                                "recall": mean(rows, method, "recall"),
                                "f1": mean(rows, method, "f1"),
                                "named_any": mean(rows, method, "named_any"),
                                "wrong_top": mean(rows, method, "wrong_top"),
                            }
                        )
    return {"table": table}


def cell(table: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    rows = [r for r in table if all(r[k] == v for k, v in where.items())]
    return rows[0]


def grade(records: list[dict[str, Any]], table: list[dict[str, Any]]) -> dict[str, Any]:
    def t(**kw: Any) -> dict[str, Any]:
        return cell(table, shape="all", **kw)

    graded: dict[str, Any] = {}
    b1 = {
        m: t(scenario="one_element", noise="low", effect=0.10, method=m)["f1"]
        for m in ("adtributor", "adtributor_recursive", "decompose")
    }
    graded["B1"] = {"passed": all(v > 0.9 for v in b1.values()), "values": b1}
    mix_runs = select(records, scenario="mix_shift", noise="low", effect=0.10)
    wrong = t(scenario="mix_shift", noise="low", effect=0.10, method="top_delta")["wrong_top"]
    shares = [r["result"]["mix_share"] for r in mix_runs]
    median_share = float(np.nanmedian(shares))
    graded["B2"] = {
        "passed": bool(wrong > 0.5 and median_share > 0.9),
        "values": {"top_delta_wrong": wrong, "median_mix_share": median_share},
    }
    b3 = {
        m: t(scenario="one_cell", noise="low", effect=0.10, method=m)["recall"]
        for m in ("adtributor", "adtributor_recursive")
    }
    graded["B3"] = {
        "passed": bool(b3["adtributor"] < 0.5 and b3["adtributor_recursive"] > 0.7),
        "values": b3,
    }
    uniform = select(records, scenario="uniform")
    b4 = float(np.mean([r["result"]["adtributor"]["named_any"] for r in uniform]))
    graded["B4"] = {"passed": b4 < 0.10, "values": {"adtributor_named_a_segment": b4}}
    null = select(records, scenario="null")
    b5 = float(np.mean([r["result"]["kept_flagged"] for r in null]))
    graded["B5"] = {"passed": b5 < 0.10, "values": {"kept_a_flagged_segment": b5}}
    b6 = {}
    for scenario in ("one_element", "two_elements", "one_cell", "mix_shift"):
        for method in METHODS:
            b6[f"{scenario}/{method}"] = t(
                scenario=scenario, noise="high", effect=0.02, method=method
            )["f1"]
    graded["B6"] = {"passed": all(v <= 0.7 for v in b6.values()), "values": b6}
    return graded


def write(records: list[dict[str, Any]], output: Path, seeds: int) -> dict[str, Any]:
    summary = summarise(records)
    graded = grade(records, summary["table"])
    result = {
        "seeds": seeds,
        "rows": ROWS,
        "n_boot": N_BOOT,
        "shapes": [f"{a}x{b}" for a, b in SHAPES],
        **summary,
        "expectations": graded,
    }
    output.write_text(json.dumps(result, indent=1) + "\n")
    lines = ["# Benchmark results", "", "| Expectation | Result |", "|---|---|"]
    for name, item in graded.items():
        lines.append(f"| {name} | {'passed' if item['passed'] else 'failed'} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whatchanged benchmark")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--shapes", type=int, default=len(SHAPES))
    parser.add_argument("--output", default=str(HERE / "results.json"))
    args = parser.parse_args()
    records = collect(args.seeds, SHAPES[: args.shapes])
    result = write(records, Path(args.output), args.seeds)
    print(json.dumps({k: v["passed"] for k, v in result["expectations"].items()}, indent=1))


if __name__ == "__main__":
    main()
