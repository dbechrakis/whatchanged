"""Download BTS on-time files and keep only counts by month, carrier and origin.

Not run in CI. It writes tests/data/flights_cells.csv, the table the worked case uses."""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
MONTHS = [(2023, m) for m in range(5, 13)] + [(2024, m) for m in range(1, 7)]
OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "flights_cells.csv"


def one(year: int, month: int) -> pd.DataFrame:
    with urllib.request.urlopen(URL.format(year=year, month=month), timeout=300) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        frame = pd.read_csv(
            archive.open(name),
            usecols=["Reporting_Airline", "Origin", "ArrDel15"],
            dtype={"Reporting_Airline": str, "Origin": str},
        )
    frame = frame.dropna(subset=["ArrDel15"])
    grouped = frame.groupby(["Reporting_Airline", "Origin"])["ArrDel15"].agg(["size", "sum"])
    out = grouped.reset_index().rename(
        columns={
            "Reporting_Airline": "carrier",
            "Origin": "origin",
            "size": "flights",
            "sum": "late",
        }
    )
    out["month"] = f"{year}-{month:02d}"
    out["late"] = out["late"].astype(int)
    return out[["month", "carrier", "origin", "flights", "late"]]


def main() -> None:
    parts = []
    for year, month in MONTHS:
        print(year, month, flush=True)
        parts.append(one(year, month))
    pd.concat(parts, ignore_index=True).to_csv(OUT, index=False)
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
