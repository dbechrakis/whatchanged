# A worked case: why is Central's margin lower than West's

The table is the Sample Superstore extract, aggregated to region, category and subcategory. The question is why the profit margin of Central is lower than West's. The answer below describes where the gap sits and does not explain why a region earned less.

```python
import pandas as pd
import whatchanged as wc

cells = pd.read_csv("tests/data/superstore_cells.csv")
pair = cells[cells["region"].isin(["West", "Central"])]
result = wc.compare(
    pair, wc.Ratio("profit", "sales"), "region", "West", "Central", ["category", "subcategory"]
)
parts = result.decompose(["category", "subcategory"]).components
print((parts.set_index("component")["contribution"] * 100).round(3))
```

The gap is -7.023 percentage points. Category mix contributes +0.273 and subcategory mix +0.451. The rate within each subcategory contributes -7.747. The parts add up to the gap.

In words: Central sells a slightly better mix of products than West, and earns a much lower margin on the same products. That is where the gap sits. It does not say why Central's margins are lower. In this dataset the recorded discount explains most of the variance in line margin, so the within-product part is close to a restatement of heavier discounting.

The case uses the Superstore table only. A case on public flight data is left for a later version.
