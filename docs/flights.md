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
print(result.explain(method="recursive").message)
```

The share of late arrivals fell from 26.41% in May 2024 to 24.94% in June 2024. In exact arithmetic, in percentage points, the change splits into carrier mix +0.02, origin mix -0.06, the rate within carrier and origin cells -1.48, cells that entered +0.07 and cells that left -0.01. Almost all of the fall sits in the rate within cells, and not in a shift in which carriers or origins carried the traffic.

The search answers that no segment explains the change. The fall is spread over the carriers and the origins, with no set of fewer than 3 elements passing its thresholds. This is the third answer of the package, and here it is the useful one. A dashboard that ranks origins by their own change would have named a few airports, and the search says that doing so would single out noise.

Since no segment is named, there is nothing to rank against the twelve earlier pairs of consecutive months. Sampling noise is not checked here, since the table holds counts and not individual flights.
