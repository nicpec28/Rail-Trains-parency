"""
Door subsystem - full pipeline in one file.

Merges what were originally three separate scripts (segmentation.py,
features.py, train_logreg_l1.py) plus predict.py into one, run via
subcommands:

    python door_pipeline.py train
    python door_pipeline.py predict --input Test.csv --output door_predictions.csv --model door_model.pkl

Each subcommand does exactly what its original standalone script did; only
the file layout changed, not the logic. This version has NO sibling-module
imports (no "from features import ..." / "from segmentation import ...") -
everything needed lives in this one file, so it can't fail with an
ImportError/ModuleNotFoundError from a missing local file, or collide with
an unrelated PyPI package that happens to share a name like "features".

================================================================================
TRAIN/VALIDATION SPLIT - DESIGN DECISION AND JUSTIFICATION
================================================================================
(the brief explicitly leaves this open and asks us to state our assumption)

The brief suggests splitting by "operating condition, by file, or another
grouping relevant to the subsystem" where an official split isn't given. For
Door, none of those groupings exist: Train.csv is a single continuous stream
from one door, not multiple files or labelled operating conditions - there is
no natural axis to split along.

We also only have 110 labelled cycles total, 30 of them Abnormal. A single
fixed train/validation split would carve that down further (e.g. an 80/20
split leaves only ~22 validation cycles, ~6 Abnormal) - too few for a stable
estimate: with that few positives, a single flipped prediction swings F1 by
~0.15-0.3.

ASSUMPTION: with no natural grouping to split by and too little data for a
single held-out slice to be trustworthy, we use every cycle as both training
and validation data via repeated stratified k-fold (5 folds x 10 repeats = 50
train/validation splits), each fold stratified to preserve the ~27% Abnormal
ratio, so the reported performance is a distribution (mean +/- std), not one
noisy point estimate.

Pipeline:
  1. Segment Train.csv, extract ~121 candidate features per cycle, attach
     labels from Train_Segments_Answer.csv.
  2. Drop near-constant / highly-collinear features (unsupervised - uses only
     X, not the labels, so computing this on the full 110 cycles is not a
     label-leakage risk).
  3. Inside every CV fold: scale -> SMOTE-balance the TRAINING fold only ->
     SelectKBest -> Logistic Regression (L1). The held-out fold in each split
     is never resampled - it must stay representative of the real
     (imbalanced) world.
  4. Refit on the full 110-cycle dataset for deployment (predict.py runs this
     final fit on the real Test.csv).
"""

from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
STATUS_LABELS = {0: "Normal", 1: "Abnormal resistance"}


# ==============================================================================
# 1. SEGMENTATION  (was segmentation.py)
# ==============================================================================

GAP_THRESHOLD_S = 2.0  # anything under this is "still sampling"; min real gap seen is ~10s


def parse_datetime_column(series: pd.Series) -> pd.Series:
    """Parse the dataset's native 'Y-M-D-H-M-S-ms' (not zero-padded) format."""

    def _parse(s: str) -> pd.Timestamp:
        y, mo, d, h, mi, se, ms = map(int, s.split("-"))
        return pd.Timestamp(
            year=y, month=mo, day=d, hour=h, minute=mi, second=se, microsecond=ms * 1000
        )

    return series.apply(_parse)


def load_stream(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = parse_datetime_column(df["Datetime"])
    return df


def segment_stream(df: pd.DataFrame, gap_threshold_s: float = GAP_THRESHOLD_S) -> pd.DataFrame:
    """
    Assign a segment_id to every row of a continuous stream by detecting gaps
    in elapsed time larger than gap_threshold_s. Verified 110/110 exact match
    against Train_Segments_Answer.csv (see the write-up's segmentation section).

    Returns a copy of df with an added integer 'segment_id' column (0-indexed,
    in temporal order).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    dt = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
    new_segment = dt > gap_threshold_s
    segment_id = new_segment.cumsum()
    df = df.assign(segment_id=segment_id)
    return df


def summarize_segments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: start_time, end_time, n_rows (diagnostic only)."""
    g = df.groupby("segment_id")
    out = g.agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        n_rows=("timestamp", "size"),
    ).reset_index()
    return out


def format_native(ts: pd.Timestamp) -> str:
    """Round-trip back to the dataset's native, non-zero-padded timestamp format."""
    return f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-{ts.microsecond // 1000}"


# ==============================================================================
# 2. FEATURE ENGINEERING  (was features.py)
# ==============================================================================

CURRENT = "Motor current(mA)"
VOLTAGE = "Motor Voltage(10mV)"
BACK_EMF = "Motor electrodynamic force"
OPEN_TIME = "Door opening time(.1s)"
CLOSE_TIME = "Door closing time(.1s)"
CLOSE_CMD = "Close command"
OPEN_CMD = "Open command"
POSITION = "Door leaf position"
FLAG_COLS = ["DCSR", "DCSL", "DLSR", "DLSL", "Door Opened", "Door Locked", "Door is opening", "Door is closing"]

EPS = 1e-6


def _series_stats(x: np.ndarray, name: str) -> dict:
    x = x.astype(float)
    d = np.diff(x)
    return {
        f"{name}_mean": x.mean(),
        f"{name}_std": x.std(),
        f"{name}_min": x.min(),
        f"{name}_max": x.max(),
        f"{name}_median": np.median(x),
        f"{name}_range": x.max() - x.min(),
        f"{name}_first": x[0],
        f"{name}_last": x[-1],
        f"{name}_first_minus_last": x[0] - x[-1],
        f"{name}_max_abs_step": np.abs(d).max() if len(d) else 0.0,
        f"{name}_std_step": d.std() if len(d) else 0.0,
        f"{name}_mean_abs_step": np.abs(d).mean() if len(d) else 0.0,
    }


def _flag_stats(x: np.ndarray, name: str) -> dict:
    x = x.astype(int)
    transitions = int(np.sum(np.diff(x) != 0))
    return {
        f"{name}_frac_on": x.mean(),
        f"{name}_n_transitions": transitions,
        f"{name}_first": x[0],
        f"{name}_last": x[-1],
    }


def extract_features(segment_df: pd.DataFrame) -> dict:
    """segment_df: rows belonging to a single cycle, already time-sorted."""
    n = len(segment_df)
    feats: dict = {}

    feats["n_rows"] = n
    duration = (segment_df["timestamp"].iloc[-1] - segment_df["timestamp"].iloc[0]).total_seconds()
    feats["duration_s"] = duration

    for col, name in [(CURRENT, "current"), (VOLTAGE, "voltage"), (BACK_EMF, "back_emf")]:
        feats.update(_series_stats(segment_df[col].values, name))

    # "energy-like" proxy: current * voltage integrated over the cycle
    energy = (segment_df[CURRENT].values.astype(float) * segment_df[VOLTAGE].values.astype(float))
    feats["energy_proxy_mean"] = energy.mean()
    feats["energy_proxy_max"] = energy.max()
    feats["energy_proxy_sum"] = energy.sum()

    for col, name in [(OPEN_TIME, "open_time_cfg"), (CLOSE_TIME, "close_time_cfg")]:
        feats.update(_series_stats(segment_df[col].values, name))

    pos = segment_df[POSITION].values.astype(float)
    displacement = pos[-1] - pos[0]
    path_length = np.abs(np.diff(pos)).sum() if n > 1 else 0.0
    reversals = 0
    if n > 2:
        d = np.diff(pos)
        signs = np.sign(d)
        signs = signs[signs != 0]
        if len(signs) > 1:
            reversals = int(np.sum(np.diff(signs) != 0))
    feats["position_start"] = pos[0]
    feats["position_end"] = pos[-1]
    feats["position_displacement"] = displacement
    feats["position_abs_displacement"] = abs(displacement)
    feats["position_path_length"] = path_length
    feats["position_overshoot_ratio"] = path_length / (abs(displacement) + EPS)
    feats["position_reversals"] = reversals
    feats["position_min"] = pos.min()
    feats["position_max"] = pos.max()
    feats.update(_series_stats(pos, "position"))

    open_cmd = segment_df[OPEN_CMD].values.astype(int)
    close_cmd = segment_df[CLOSE_CMD].values.astype(int)
    feats["open_cmd_frac"] = open_cmd.mean()
    feats["close_cmd_frac"] = close_cmd.mean()
    feats["open_cmd_n_transitions"] = int(np.sum(np.diff(open_cmd) != 0))
    feats["close_cmd_n_transitions"] = int(np.sum(np.diff(close_cmd) != 0))
    feats["direction_is_close"] = int(close_cmd.mean() >= open_cmd.mean())

    for col in FLAG_COLS:
        feats.update(_flag_stats(segment_df[col].values, col.replace(" ", "_")))

    return feats


def build_feature_table(seg_df: pd.DataFrame) -> pd.DataFrame:
    """seg_df must have a 'segment_id' column (output of segment_stream)."""
    rows = []
    ids = []
    for sid, s in seg_df.groupby("segment_id"):
        rows.append(extract_features(s.sort_values("timestamp")))
        ids.append(sid)
    table = pd.DataFrame(rows)
    table.insert(0, "segment_id", ids)
    return table


class DoorModel:
    """Self-contained door-cycle classifier: wraps the fitted pipeline + its
    feature-column order behind a single predict_stream() call, so a caller
    (e.g. a FastAPI endpoint) can hand it one raw continuous door-sensor .csv
    and get back a per-cycle Normal/Abnormal DataFrame - no separate
    segmentation/feature-extraction/feature-column bookkeeping required."""

    def __init__(self, pipeline: ImbPipeline, feature_cols: list, model_name: str):
        self.pipeline = pipeline
        self.feature_cols = feature_cols
        self.model_name = model_name

    def predict_stream(self, csv_path: str) -> pd.DataFrame:
        """csv_path: one continuous door-sensor .csv stream (Train.csv/Test.csv
        format). Returns one row per detected cycle: start_time, end_time,
        prediction ('Normal' or 'Abnormal resistance')."""
        df = load_stream(csv_path)
        seg = segment_stream(df)
        summary = summarize_segments(seg)

        # Sanity check: warn (don't crash) if this stream doesn't show the
        # same clean "gap = boundary" structure the training data had.
        dt = df["timestamp"].diff().dt.total_seconds().dropna()
        within = dt[dt <= GAP_THRESHOLD_S]
        gaps = dt[dt > GAP_THRESHOLD_S]
        if len(within) and within.max() > 0.5:
            print(f"WARNING: within-segment sampling gaps up to {within.max():.2f}s seen "
                  f"(training data never exceeded 0.02s) - segmentation may be less reliable here.")
        if len(gaps) and gaps.min() < GAP_THRESHOLD_S:
            print("WARNING: some inter-segment gaps are close to the threshold - check for over/under-segmentation.")

        feat_table = build_feature_table(seg)
        X = feat_table[self.feature_cols]
        preds = self.pipeline.predict(X)
        labels = [STATUS_LABELS[p] for p in preds]

        return pd.DataFrame(
            {
                "start_time": [format_native(t) for t in summary["start_time"]],
                "end_time": [format_native(t) for t in summary["end_time"]],
                "prediction": labels,
            }
        )


# ==============================================================================
# 3. TRAINING - Logistic Regression (L1), the selected model
#    (was train_logreg_l1.py)
# ==============================================================================

def load_dataset(train_path: str = "Train.csv", answer_path: str = "Train_Segments_Answer.csv") -> pd.DataFrame:
    df = load_stream(train_path)
    seg = segment_stream(df)
    feat_table = build_feature_table(seg)

    answer = pd.read_csv(answer_path)
    answer = answer.reset_index(drop=True)
    # segmentation already verified 1:1, in-order correspondence with the
    # answer key (110/110 exact match) - safe to align by position.
    feat_table["status"] = answer["status"].values
    feat_table["operation"] = answer["operation"].values  # kept for analysis only, not a model feature
    feat_table["y"] = (feat_table["status"] == "Abnormal resistance").astype(int)
    return feat_table


def prune_features(train_X: pd.DataFrame, corr_threshold: float = 0.95):
    """Drop near-constant and highly-collinear columns, fit on all of X (label-free)."""
    stds = train_X.std()
    keep = stds[stds > 1e-8].index.tolist()
    dropped_constant = [c for c in train_X.columns if c not in keep]

    corr = train_X[keep].corr().abs()
    to_drop = set()
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        if c1 in to_drop:
            continue
        for c2 in cols[i + 1:]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] > corr_threshold:
                to_drop.add(c2)
    final_keep = [c for c in keep if c not in to_drop]
    return final_keep, dropped_constant, sorted(to_drop)


def make_model(n_features_selected: int) -> ImbPipeline:
    k = min(n_features_selected, 15)
    return ImbPipeline(
        [
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("select", SelectKBest(f_classif, k=k)),
            (
                "clf",
                LogisticRegression(
                    penalty="l1", solver="liblinear", C=0.5, max_iter=2000, class_weight="balanced"
                ),
            ),
        ]
    )


SCORERS = {
    "f1_abnormal": make_scorer(f1_score, pos_label=1, zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "roc_auc": "roc_auc",
}


def repeated_cv_evaluate(pipe: ImbPipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, n_repeats: int = 10) -> dict:
    """
    Many train/validation splits, not one: n_splits x n_repeats stratified
    splits over the full dataset, reported as mean +/- std so we can see how
    stable the number actually is (see module docstring for why).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS, n_jobs=-1)
    return {
        "n_val_splits": len(scores["test_f1_abnormal"]),
        "f1_abnormal_mean": float(scores["test_f1_abnormal"].mean()),
        "f1_abnormal_std": float(scores["test_f1_abnormal"].std()),
        "f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
    }


def train_main(train_path: str, answer_path: str, model_out: str, results_out: str):
    table = load_dataset(train_path, answer_path)
    feature_cols = [c for c in table.columns if c not in ("segment_id", "status", "operation", "y")]
    X_all = table[feature_cols]
    y_all = table["y"].values
    print(f"Full dataset: {len(X_all)} cycles ({int(y_all.sum())} Abnormal, {y_all.mean():.1%})")

    keep_cols, dropped_const, dropped_corr = prune_features(X_all)
    print(f"\nFeature pruning: {len(feature_cols)} candidates -> {len(keep_cols)} kept")
    print(f"  dropped (near-constant): {dropped_const}")
    print(f"  dropped (collinear, corr>0.95): {dropped_corr}")

    X_all_p = X_all[keep_cols]
    pipe = make_model(len(keep_cols))

    print(f"\nRunning repeated stratified k-fold CV (5 folds x 10 repeats = 50 train/validation "
          f"splits) over all {len(X_all_p)} cycles (SMOTE applied inside each training fold only)...")
    cv_summary = repeated_cv_evaluate(pipe, X_all_p, y_all)
    print(f"Logistic Regression (L1): F1[Abnormal] = {cv_summary['f1_abnormal_mean']:.3f} "
          f"+/- {cv_summary['f1_abnormal_std']:.3f} across {cv_summary['n_val_splits']} validation splits "
          f"(F1[macro] = {cv_summary['f1_macro_mean']:.3f}, ROC-AUC = {cv_summary['roc_auc_mean']:.3f})")

    # Final fit for deployment: train on ALL 110 labelled cycles. There's no
    # held-out slice to "spend" here by design (see module docstring) - the
    # real unbiased evaluation happens externally, when predict's output is
    # scored against the organisers' hidden Test.csv labels, which this
    # training run never sees.
    pipe.fit(X_all_p, y_all)

    model = DoorModel(pipe, keep_cols, "logreg_l1")
    joblib.dump(model, model_out)
    print(f"\nSaved trained pipeline (fit on all 110 labelled cycles) -> {model_out}")

    with open(results_out, "w") as f:
        json.dump(
            {
                "model": "logreg_l1",
                "cv_summary": cv_summary,
                "kept_features": keep_cols,
                "dropped_constant": dropped_const,
                "dropped_collinear": dropped_corr,
                "dataset_size": {"n_cycles": len(X_all_p), "n_abnormal": int(y_all.sum())},
                "validation_strategy": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10) over all cycles - see module docstring for justification",
            },
            f,
            indent=2,
        )
    print(f"Saved run summary -> {results_out}")


# ==============================================================================
# 4. INFERENCE  (was predict.py)
# ==============================================================================

def predict_main(input_path: str, output_path: str, model_path: str):
    model = joblib.load(model_path)
    out = model.predict_stream(input_path)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predicted segments -> {output_path}")
    print(out["prediction"].value_counts())
    return out


# ==============================================================================
# CLI
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Door subsystem - full pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Segment Train.csv, extract features, fit + save the L1 logistic regression model")
    p_train.add_argument("--train-csv", default="Train.csv")
    p_train.add_argument("--answers", default="Train_Segments_Answer.csv")
    p_train.add_argument("--model-out", default="door_model.pkl")
    p_train.add_argument("--results-out", default="cv_results.json")

    p_predict = sub.add_parser("predict", help="Run the saved model on a continuous Test.csv-format stream")
    p_predict.add_argument("--input", default="Test.csv")
    p_predict.add_argument("--output", default="door_predictions.csv")
    p_predict.add_argument("--model", default="door_model.pkl")

    args = ap.parse_args()

    if args.command == "train":
        train_main(args.train_csv, args.answers, args.model_out, args.results_out)
    elif args.command == "predict":
        predict_main(args.input, args.output, args.model)


if __name__ == "__main__":
    # Running this file directly (`python door_pipeline.py ...`) makes Python
    # execute it as a module named "__main__" - any class defined here (e.g.
    # DoorModel) would then get pickled as belonging to "__main__", which no
    # other script can ever resolve via `import door_pipeline; joblib.load(...)`.
    # Re-invoke through a real import of "door_pipeline" so DoorModel (and
    # anything pickled during this run) is attributed to that importable
    # module name instead, regardless of how this script was launched.
    import importlib
    import sys

    _mod = importlib.import_module("door_pipeline")
    sys.exit(_mod.main())
