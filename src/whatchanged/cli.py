"""The command line: read a CSV or Parquet file, compare two sides, print what changed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from whatchanged.compare import compare
from whatchanged.errors import WhatChangedError
from whatchanged.metrics import parse_metric


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise WhatChangedError(f"no such file: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise WhatChangedError("reading Parquet needs pyarrow or fastparquet") from error
    raise WhatChangedError("the file must be a .csv or a .parquet file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whatchanged", description="Which segment explains a change in a metric."
    )
    parser.add_argument("file", type=Path, help="a CSV or Parquet file")
    parser.add_argument("--metric", required=True, help="'a/b' for a ratio, 'sum:a' or 'mean:a'")
    parser.add_argument("--split", required=True, help="the column that names the two sides")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--by", required=True, help="comma-separated dimension columns")
    parser.add_argument("--unit", help="a column of units to resample for the noise check")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command. Returns the exit status."""
    args = build_parser().parse_args(argv)
    try:
        table = _read(args.file)
        if args.split in table.columns:
            table[args.split] = table[args.split].astype(str)
        dimensions = [d.strip() for d in args.by.split(",") if d.strip()]
        comparison = compare(
            table, parse_metric(args.metric), args.split, args.baseline, args.current, dimensions
        )
        if not args.json:
            print(comparison.summary())
            return 0
        out: dict[str, object] = {
            "change": comparison.change.to_dict(),
            "explain": comparison.explain().to_dict(),
        }
        try:
            out["decompose"] = comparison.decompose(missing="entry_exit").to_dict()
        except WhatChangedError as error:
            out["decompose"] = {"kind": "exact", "error": str(error)}
        if args.unit:
            out["noise"] = comparison.noise(unit=args.unit).to_dict()
        print(json.dumps(out, indent=2, default=str))
        return 0
    except WhatChangedError as error:
        print(f"whatchanged: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
