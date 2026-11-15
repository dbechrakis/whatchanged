# Benchmark plan, fixed before the first run, 2026-11-15

**Generator.** `whatchanged.synthetic.planted(...)` builds a baseline table and a current table over 3 or 5 dimensions, each with 4, 8 or 20 elements, with cell sizes drawn from a Zipf law. A cause is planted in the current table and the planted elements are recorded.

**Scenarios**, 100 seeds each:

1. `one_element`: one element of one dimension changes its rate.
2. `two_elements`: two elements of the same dimension change.
3. `one_cell`: one cell of two dimensions changes.
4. `mix_shift`: no rate changes anywhere. Volume moves from a high-rate element to a low-rate one, so the overall ratio falls while every element's own rate is unchanged.
5. `uniform`: every cell changes by the same relative amount. There is no localised cause.
6. `null`: the two tables are two draws from the same distribution.

Each scenario at two noise levels (row-level variance low and high) and two effect sizes (the planted change moves the total by 2% and by 10%).

**Methods.** `top_delta` (rank elements by absolute change of their own metric, the thing a dashboard does), `adtributor`, `adtributor_recursive`, and `decompose` followed by ranking its contributions.

**Scores.** Precision, recall and F1 of the returned elements against the planted ones, per scenario. For `uniform` and `null`, the false alarm rate: the share of runs in which a method names any segment.

**Expectations.**

- B1. On `one_element` at the 10% effect and low noise, every method except `top_delta` has F1 above 0.9.
- B2. On `mix_shift`, `top_delta` names a wrong element in more than half of the runs, and `decompose` attributes more than 90% of the change to mix.
- B3. On `one_cell`, `adtributor` (one dimension at a time) has recall below 0.5 and `adtributor_recursive` above 0.7.
- B4. On `uniform`, `adtributor` names a segment in fewer than 10% of runs.
- B5. On `null`, with `noise(unit=...)` applied, fewer than 10% of runs keep any flagged segment.
- B6. At the 2% effect and high noise, no method has F1 above 0.7 on any scenario. Reported as the floor of what the tool can see.

**What counts.** Each expectation is graded from `benchmarks/results.json`. Thresholds of the methods are not tuned after the first run. A failed expectation is reported in the README next to the passed ones, and becomes a documented limit of the method.

## Details fixed with the plan, before any run

- **Shapes.** Three, each with 100 seeds per scenario, noise level and effect: 3 dimensions of 8 elements, 5 dimensions of 4 elements, and 3 dimensions of 20 elements. Seed `i` of shape `s` is `1000 * s + i`. Each side has 5,000 rows. `null` does not depend on the effect, so it runs once per noise level.
- **Pooling.** Expectations are graded on the pool of the three shapes, 300 runs per cell. Per-shape numbers are reported next to them.
- **Settings.** Defaults of the package: `tep` 0.67, `teep` 0.10, `spread_at` 3, recursive depth 2. The noise check uses units, 200 resamples and a 95% level. Row-level noise has standard deviation 0.1 (low) or 0.5 (high). The baseline rate is 0.15.
- **What a method returns.** `adtributor` returns the elements of its top candidate set, and `adtributor_recursive` the segments it reaches. `top_delta` and `decompose` rank every element of every single dimension and return the top k, where k is the number of planted segments, and one element for `uniform` and `null`. This gives them knowledge of k that the other two do not have. `decompose` ranks by the absolute value of mix plus rate of the element in a decomposition by that one dimension. `top_delta` has no stopping rule, so its false alarm rate is 100% by construction.
- **Matching.** A returned segment counts as found when its conditions equal those of a planted segment exactly. A single element never matches a planted cell of two dimensions.
- **B2.** Graded on `mix_shift` at the 10% effect and low noise. A wrong element is one outside the two planted elements. The share of the change attributed to mix is the median over runs of the mix component over the change, in a decomposition by the first dimension, where the shift is planted.
- **B1 and B3.** Graded at the 10% effect and low noise, on mean F1 and mean recall.
- **B4.** Graded on all `uniform` runs pooled over effect and noise, with each cell reported.
- **B5.** The segments are those `adtributor` names. A run keeps a flagged segment when at least one of them has an interval that excludes zero. A run in which `adtributor` names nothing keeps none.
- **B6.** Graded on mean F1 of each method on each scenario that has planted segments, pooled over shapes.
- **CI.** A job reruns 10 seeds of the first shape and compares to the committed values within a tolerance of 0.1 on each mean.
