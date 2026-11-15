# Project conventions

## Writing style
- In prose, never join or end clauses with dashes or semicolons. Use full stops and short sentences instead. Hyphens inside compound words are fine.
- This applies to the README, docs, commit messages, release notes, issue and PR text, and replies to the owner.
- Write the way a person would. Avoid filler, hype and stacked qualifiers.

## Code
- Python 3.10 to 3.13, ruff for lint and format, `mypy --strict` on `src`, pytest with hypothesis. Run `make check` before committing.
- No code comments beyond what an API docstring needs. Public functions carry docstrings because the documentation is built from them.
- Tests never touch the network and never need a real dataset.
- Core dependencies are numpy and pandas only. DuckDB is an optional extra.

## Analysis rules
- Every result says what kind of answer it is: `exact`, `heuristic` or `statistical`. Never blur the three.
- `summary()` never uses the words "cause" or "because". The package describes where a change sits and does not say why it happened.
- No number in a document without a test that checks it.
- Thresholds of the methods are not tuned after the benchmark has run. A failed expectation is reported next to the passed ones.

## Commands
- `make install`, `make check`, `make test`, `make bench`, `make docs`.
