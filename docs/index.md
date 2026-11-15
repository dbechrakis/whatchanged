# whatchanged

Point it at a table with two sides and ask what changed. It tells you where the change sits, how sure the answer is, and which kind of answer you are reading.

## Five-minute start

```bash
pip install whatchanged
```

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
print(result.explain().message)
print(result.summary())
```

The table is synthetic, with a change planted in one element. With your own table, name its columns. `split` names a column and `baseline` and `current` name two of its values.

## What next

- [The three kinds of answer](answers.md) says what each result can and cannot claim.
- [Reference](api.md) lists every function.
- [Benchmark](benchmark.md) shows where the search fails.
- [A worked case](case-study.md) decomposes a real margin gap.
