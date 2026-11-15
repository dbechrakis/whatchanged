# A second case: late arrivals in US flights

The table is the monthly on-time performance data of the US Bureau of Transportation Statistics, a public domain source. It is aggregated to month, reporting airline and origin airport, with the count of flights that arrived and the count that arrived at least 15 minutes late. Cancelled and diverted flights are left out. The question is which carriers and origins account for the change in the share of late arrivals between two named months. The answer describes where the change sits. It does not say why a carrier was late.

```python
import pandas as pd
import whatchanged as wc

cells = pd.read_csv("tests/data/flights_cells.csv")
result = wc.compare(
    cells, wc.Ratio("late", "flights"), "month", "2024-05", "2024-06", ["carrier", "origin"]
)
change = result.change
print(round(100 * change.baseline_value, 2), round(100 * change.current_value, 2))
parts = result.decompose(["carrier", "origin"], missing="entry_exit").components
print((parts.set_index("component")["contribution"] * 100).round(2))
print(result.explain(method="recursive").to_frame())
```

NUMBERS

The history check ranks the current contribution of each named segment among the twelve earlier pairs of consecutive months. Sampling noise is not checked here, because the table holds counts and not individual flights.
