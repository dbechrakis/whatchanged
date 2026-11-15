# Changelog

## Unreleased (0.1.0)

- Metrics: `Sum`, `Ratio`, `Mean`.
- `decompose`: an exact mix and rate decomposition by Shapley allocation. It reproduces the published Superstore margin gap to three decimals.
- `explain`: Adtributor from the paper, single and recursive, with the answer that no segment explains the change.
- `noise`: a bootstrap over units and a rank among earlier period pairs.
- A DuckDB path that aggregates in SQL, and a command line.
- A benchmark, fixed before it was run and run once. Three expectations failed, and each is in the README.
- `Sum(column, units=...)` lets `decompose` split a change in a sum into volume and value per unit.
- Cut from this version, in the order the plan allowed: the external benchmark on `riskloc` datasets, the flight case study. The worked case uses the Superstore table.
- Not yet done: the release to PyPI. It needs the owner's accounts and the tag `v0.1.0`.
