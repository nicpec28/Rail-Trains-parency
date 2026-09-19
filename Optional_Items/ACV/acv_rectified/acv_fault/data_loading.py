"""Dynamic loading of ACV case files into a per-car long format.

Each case file is a single .xlsx with 3 identifying columns (car model, train
number, time) followed by ``Car <NN> - <parameter>`` columns for every car in
the train. The parameter set is NOT fixed across files (a "standard" ~8
parameter set vs. a much richer ~60 parameter set have both been observed),
so column headers are always read from the file itself rather than assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

CAR_COL_RE = re.compile(r"^Car (\d+) - (.+)$")


@dataclass
class CaseFile:
    file_id: str
    car_ids: list[str]
    # car_id -> DataFrame indexed by row, columns = raw parameter names for that car
    per_car: dict[str, pd.DataFrame]
    n_rows: int


def load_case_file(path: str) -> CaseFile:
    """Read one case .xlsx and split it into per-car parameter frames.

    Car identifiers are kept exactly as they appear in the file's own column
    headers (e.g. ``"03"``), since that is what predictions must be reported
    in.
    """
    df = pd.read_excel(path, na_values=["None", "none", "NaN", ""])

    car_columns: dict[str, dict[str, str]] = {}
    for col in df.columns:
        m = CAR_COL_RE.match(str(col))
        if not m:
            continue
        car_id, param = m.group(1), m.group(2)
        car_columns.setdefault(car_id, {})[param] = col

    if not car_columns:
        raise ValueError(f"No 'Car <NN> - <parameter>' columns found in {path}")

    car_ids = sorted(car_columns.keys())
    per_car = {}
    for car_id in car_ids:
        cols = car_columns[car_id]
        sub = pd.DataFrame({param: df[colname] for param, colname in cols.items()})
        per_car[car_id] = sub

    return CaseFile(file_id=path, car_ids=car_ids, per_car=per_car, n_rows=len(df))
