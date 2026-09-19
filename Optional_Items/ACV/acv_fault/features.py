"""Physically-motivated, per-car feature engineering for leak localisation.

A refrigerant leak causes undercharge, which shows up as the affected car
struggling to reach its cooling setpoint (a larger indoor/cabin-temperature-
minus-target gap while actively cooling) relative to its peers on the same
train, at the same time, under the same ambient conditions. Fault/invalid
status flags are secondary, corroborating signals.

All features are computed per car from that car's own sub-frame, then
robust-z-scored *within the file* (across its 8 cars) in ``build_feature_matrix``
so they are comparable across files with very different ambient regimes and
sensor scales.

Deliberately NOT included: refrigerant-pressure and compressor-cycling
features. An earlier version added ``low_pressure_mean``, ``low_pressure_p10``,
``high_pressure_mean``, and ``compressor_cycle_rate`` after inspecting the one
training file with a richer telemetry schema and discovering what would rank
its (known, labeled) faulty car correctly. That inspection makes any resulting
validation score for those features circular: with only 6 labeled cases, and
only that one file carrying pressure/compressor data, there is no independent
file left to test whether those features generalize versus merely fit the
single example that inspired them. They were removed rather than kept on
faith, and a controlled comparison confirmed the removal costs nothing on the
6 labeled cases (see acv_fault/train.py's evaluation). If more labeled
rich-schema cases become available, pressure/compressor features should be
reconsidered and validated on cases that were not used to design them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import CaseFile

# Candidate column-name aliases, in priority order, for parameters that are
# named differently between the "standard" and "rich" telemetry schemas.
CABIN_TEMP_ALIASES = [
    "Indoor Average Temperature",
    "Passenger Cabin Temperature Detected Value",
    "Observation Area Temperature Detected Value",
]
TARGET_COOL_TEMP_ALIASES = [
    "ACV Control Temperature (Cooling)",
    "Target Temperature Value",
]
RUNNING_MODE_ALIASES = ["ACV Running Mode"]
INFO_VALID_ALIASES = ["ACV Information Valid"]
LOAD_ALIASES = ["Load Halved", "Load Shedding"]

FEATURE_NAMES = [
    "cool_gap_mean",
    "cool_gap_p90",
    "cool_gap_max",
    "fault_rate",
    "mode_invalid_rate",
    "info_invalid_rate",
    "load_active_rate",
    "missing_data_rate",
]


def _first_present(df: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    for name in aliases:
        if name in df.columns:
            return df[name]
    return None


def _numeric(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")


def _mode_contains(series: pd.Series | None, substr: str) -> pd.Series | None:
    if series is None:
        return None
    return series.astype(str).str.contains(substr, case=False, na=False)


def _active_rate(series: pd.Series | None, normal_values: set[str]) -> float:
    """Fraction of non-null rows whose value is NOT one of ``normal_values``."""
    if series is None:
        return np.nan
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    if pd.api.types.is_numeric_dtype(s):
        active = s.astype(float) != 0
    else:
        norm = s.astype(str).str.strip().str.lower()
        active = ~norm.isin(normal_values)
    return float(active.mean())


def _fault_rate(df: pd.DataFrame) -> float:
    fault_cols = [c for c in df.columns if "fault" in c.lower()]
    if not fault_cols:
        return np.nan
    rates = []
    for c in fault_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            rates.append(float((s.astype(float) != 0).mean()))
        else:
            norm = s.astype(str).str.strip().str.lower()
            rates.append(float((~norm.isin({"normal", "no fault", "0", "none", "false"})).mean()))
    return float(np.mean(rates)) if rates else np.nan


def extract_car_features(df: pd.DataFrame) -> dict[str, float]:
    """Raw (un-normalized) feature dict for a single car's parameter frame."""
    cabin = _numeric(_first_present(df, CABIN_TEMP_ALIASES))
    target = _numeric(_first_present(df, TARGET_COOL_TEMP_ALIASES))
    running_mode = _first_present(df, RUNNING_MODE_ALIASES)
    info_valid = _first_present(df, INFO_VALID_ALIASES)
    load = _first_present(df, LOAD_ALIASES)

    total_rows = len(df)
    non_null_frac = df.notna().any(axis=1).mean() if total_rows else 0.0

    feats: dict[str, float] = {}

    if cabin is not None and target is not None:
        gap = cabin - target
        cooling_active = _mode_contains(running_mode, "cool")
        gap_c = gap[cooling_active] if cooling_active is not None else gap
        gap_c = gap_c.dropna()
        feats["cool_gap_mean"] = float(gap_c.mean()) if len(gap_c) else np.nan
        feats["cool_gap_p90"] = float(gap_c.quantile(0.9)) if len(gap_c) else np.nan
        feats["cool_gap_max"] = float(gap_c.quantile(0.99)) if len(gap_c) else np.nan
    else:
        feats["cool_gap_mean"] = np.nan
        feats["cool_gap_p90"] = np.nan
        feats["cool_gap_max"] = np.nan

    feats["fault_rate"] = _fault_rate(df)

    invalid_mask = _mode_contains(running_mode, "invalid")
    feats["mode_invalid_rate"] = float(invalid_mask.mean()) if invalid_mask is not None and invalid_mask.notna().any() else np.nan

    if info_valid is not None and info_valid.notna().any():
        norm = info_valid.dropna().astype(str).str.strip().str.lower()
        feats["info_invalid_rate"] = float(norm.eq("invalid").mean())
    else:
        feats["info_invalid_rate"] = np.nan

    feats["load_active_rate"] = _active_rate(load, {"normal", "0", "false", "no"})

    feats["missing_data_rate"] = float(1.0 - non_null_frac)

    return feats


def _robust_zscore(values: pd.Series) -> pd.Series:
    """Z-score across cars within one file; NaN inputs/outputs become 0 (neutral)."""
    valid = values.dropna()
    if len(valid) < 2 or valid.std(ddof=0) == 0:
        return pd.Series(0.0, index=values.index)
    mean, std = valid.mean(), valid.std(ddof=0)
    z = (values - mean) / std
    return z.fillna(0.0)


def build_feature_matrix(case: CaseFile) -> pd.DataFrame:
    """Raw + per-file-normalized feature matrix, one row per car_id."""
    raw = pd.DataFrame(
        {car_id: extract_car_features(df) for car_id, df in case.per_car.items()}
    ).T
    raw = raw.reindex(columns=FEATURE_NAMES)
    raw.index.name = "car_id"

    z = raw.copy()
    for col in FEATURE_NAMES:
        z[col] = _robust_zscore(raw[col])

    return z
