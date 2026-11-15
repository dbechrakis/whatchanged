# whatchanged

A Python package that says which segment explains a change in a metric. Point it at a table with two sides, such as two months, two regions or the two arms of a test. It tells you where the change sits, how sure the answer is, and which kind of answer you are reading.

```bash
pip install whatchanged
```

## Five-minute start

```python
import whatchanged as wc
from whatchanged.synthetic import planted

orders = planted("one_element", seed=1).table

result = wc.compare(
    orders,
    metric=wc.Ratio("profit", "sales"),
    split="side",
    baseline="a",
    current="b",
    dimensions=["d0", "d1", "d2"],
)

print(result.change.difference)
print(result.decompose(by=["d0", "d1"], missing="entry_exit").components)
print(result.explain().segments)
print(result.noise(unit="unit", n_boot=200).sampling)
print(result.summary())
```

The table here is synthetic, with a change planted in one element. With your own table, name its columns instead. `split` names a column and `baseline` and `current` name two of its values.

From a shell:

```bash
whatchanged orders.csv --metric "profit/sales" --split month --baseline 2026-09 --current 2026-10 --by region,category,channel
```

## Three kinds of answer

Every result carries a `kind`. The package never blurs them.

| Kind | Where it comes from | What it can claim |
|---|---|---|
| `exact` | `change` and `decompose` | Arithmetic. The parts add up to the change. It says where a difference sits and not why it arose. |
| `heuristic` | `explain` | A search from one paper (Adtributor, NSDI 2014) with thresholds. It can name the wrong segment, and it can say that no segment explains the change. |
| `statistical` | `noise` | A bootstrap over units and a rank among earlier period pairs. It checks sampling noise only. |

`summary()` writes at most five plain sentences. It describes and does not explain.

## Where it fails

The benchmark of [`docs/benchmark-plan.md`](docs/benchmark-plan.md) was fixed before it ran and was run once. Thresholds were not tuned afterwards. It uses synthetic tables with a planted change, 300 runs per cell. Results are in `benchmarks/results.json`.

| Expectation | Result | What it found |
|---|---|---|
| B1. On one planted element at a 10% effect and low noise, `adtributor`, `adtributor_recursive` and `decompose` have F1 above 0.9 | failed | F1 of 0.95, 0.74 and 0.99. The recursive search drops to 0.22 on five dimensions of four elements, where it descends into cells that were not planted. |
| B2. On a pure mix shift, `top_delta` names a wrong element in most runs and `decompose` puts over 90% of the change on mix | passed | `top_delta` is wrong in 100% of runs. The median mix share is 101%. |
| B3. On one planted cell, one dimension at a time has recall below 0.5 and the recursive search above 0.7 | passed | Recall of 0.00 and 0.98. |
| B4. On a uniform change, `adtributor` names a segment in under 10% of runs | failed | It names one in 72.6% of runs. With noisy rows, a change that touches every cell equally still moves some elements more than others, and the search takes that for a segment. |
| B5. On two draws of the same distribution, fewer than 10% of runs keep a flagged segment after the noise check | failed | 39.2% of runs keep one. The segments are picked for being large before they are checked, so the check is applied to a selected set and its 95% interval is too narrow for that. |
| B6. At a 2% effect and high noise, no method has F1 above 0.7 | passed | The best mean F1 on any scenario is 0.23. This is the floor of what the tool can see. |

What to do with this. Treat `explain` as a pointer to look at, not a finding. Read `decompose` as accounting. When the search names a segment on a change that is small or on noisy data, run `noise` and still doubt it. A false alarm on a uniform change is common, so a named segment is not proof that the change is localised.

## Limits

- The exact part is accounting. It says where a difference sits and not why it arose.
- The search is a heuristic from one paper, with thresholds. The benchmark shows the scenarios where it returns the wrong segment.
- Sampling noise is checked. Seasonality, data errors and definition changes are not.
- Dimensions with thousands of elements are slow in 0.1.0 and are not benchmarked.
- The decomposition allows at most five nested levels, since the number of orders grows as a factorial.
- Splitting a sum into volume and value per unit is not in 0.1.0.

## DuckDB

`pip install whatchanged[duckdb]` lets you pass a connection and a table name. The aggregation runs in SQL and only the cells come back. Identifiers are quoted and values are passed as parameters.

```python
result = wc.compare(
    con, wc.Ratio("profit", "sales"), "month", "2026-09", "2026-10", ["region"], name="orders"
)
```

## Method

The decomposition generalises the mix and rate accounting of [`superstore-shipping-region-analysis`](https://github.com/dbechrakis/superstore-shipping-region-analysis). The search follows Bhagwan, Mahajan, Vemuri and Reddy, "Adtributor: Revenue Debugging in Advertising Systems", NSDI 2014. See [`docs/adtributor-notes.md`](docs/adtributor-notes.md).

## Compared with other tools

`riskloc` collects several research algorithms as scripts and is not an installable library. PyRCA is a library built for causal graphs of service metrics and not for the dimensions of a business table. `whatchanged` is narrower. It works on a table, labels each answer, and publishes where it fails.

MIT licence.
