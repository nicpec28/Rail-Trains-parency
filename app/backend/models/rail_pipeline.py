"""
Rail Corrugation subsystem - full pipeline in one file.

Merges what were originally four separate scripts (features.py,
build_features.py, train.py, predict.py) into one, run via subcommands:

    python rail_pipeline.py build-features --start 1 --end 272 --out train_features_full.csv
    python rail_pipeline.py train
    python rail_pipeline.py predict --input Test/ --output rail_predictions.csv --model rail_model.pkl

Each subcommand does exactly what its original standalone script did; only
the file layout changed, not the logic.

================================================================================
FEATURE ENGINEERING
================================================================================
Each file is a 1-second, 10,000 Hz recording: column 1 is the raw speed-sensor
toggle (0/1) signal, columns 2-129 are vibration+shock pairs for 64 axle-box
positions (8 cars x 8 positions/car). Positions 1,3,5,7 (per car) belong to the
Side I rail; positions 2,4,6,8 belong to Side II (see the Info Kit, Section
2.1). Side I and Side II must be judged independently from the same file.

  - Train speed is DERIVED from the raw toggle signal (not used raw): count
    rising edges (tooth passes) in the 1s window, divide by 90 teeth/rev to
    get revolutions, multiply by wheel circumference (pi * 0.85m) to get
    speed in m/s. This turns an uninterpretable 0/1 stream into a physically
    meaningful covariate (vibration energy naturally scales with speed
    regardless of fault, so the model needs speed as context).
  - For each side (I, II) and each signal type (vibration, shock), we pool
    across that side's 32 channels (4 positions x 8 cars) and compute
    standard vibration-analysis statistics per channel, then aggregate
    (mean/max) across channels: RMS, std, peak, crest factor (peak/RMS -
    sensitive to impulsive events, which is what a periodic corrugation
    impact looks like), kurtosis (impulsiveness/"peakiness" of the
    distribution - corrugation-induced periodic shocks raise this), and a
    simple frequency-domain summary (dominant FFT frequency, and the energy
    fraction in a 50-500 Hz "mid" band vs. the full spectrum).
  - This keeps the feature count (~49) reasonable relative to the 272-file
    training set (and especially relative to the 14 Side I examples), rather
    than exploding to a separate feature per one of the 64 raw channels.

================================================================================
CLASS IMBALANCE - HOW IT'S HANDLED
================================================================================
The training set is 234 Normal / 24 Side II / 14 Side I out of 272 files.
Side I is only 5.1% of the data. Three deliberate choices follow from that:

1. METRIC: the organisers score this on macro F1 (Info Kit, Section 4), used
   here as the model-selection criterion too, not accuracy or weighted F1.
   A model that never predicts Side I can still get >90% accuracy while
   scoring 0 F1 on that class - macro F1 (each class weighted equally,
   regardless of size) is specifically designed not to let that pass.

2. RESAMPLING: SMOTE is used, but capped - NOT fully balanced to match
   Normal's count. With only ~11-13 real Side I examples inside any given
   training fold, asking SMOTE to synthesise its way up to ~190 (matching
   Normal) means most of the "data" the model sees for that class would be
   interpolated between a dozen real points. Instead oversampling is capped
   at 4x each minority class's real count (Side I goes from ~12 to ~48
   synthetic-included examples, not ~190).

3. CLASS WEIGHTING as a second, complementary lever: every classifier that
   supports it also gets class_weight='balanced', so even after capped SMOTE,
   misclassifying a still-rare class costs proportionally more in training.

================================================================================
TRAIN/VALIDATION SPLIT - DESIGN DECISION AND JUSTIFICATION
================================================================================
There's no file-level grouping to split by (each file is already one
independent unit), but with only 14 Side I files total, a single fixed
train/validation split could easily leave a validation fold with 1-2 (or
even 0) Side I examples - unusably noisy. Repeated stratified k-fold (k=5,
20 repeats = 100 validation splits) is used instead over all 272 files, so
every file serves as both training and validation data across folds, and the
reported score is a mean +/- std, not one fragile point estimate. k=5 keeps
~2-3 Side I files in every single validation fold (14/5 ~ 2.8) - still few,
but the repeats average out most of the resulting fold-to-fold noise.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
import urllib.request
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
MAX_OVERSAMPLE_MULTIPLIER = 4  # cap on how far SMOTE inflates a minority class

TRAIN_BASE_URL = "https://raw.githubusercontent.com/aochinwen/NebulaX-Hackathon-ProblemStatement/main/PS3/02_Datasets/Rail_Corrugation/Train/"
TMP_FILE = "_tmp_rail_file.csv"
MAX_RETRIES = 5

N_TEETH = 90
WHEEL_DIAMETER_M = 0.85
WHEEL_CIRCUMFERENCE_M = np.pi * WHEEL_DIAMETER_M
SAMPLING_HZ = 10000
DURATION_S = 1.0

SIDE_I_POSITIONS = [1, 3, 5, 7]
SIDE_II_POSITIONS = [2, 4, 6, 8]
N_CARS = 8


# ==============================================================================
# 1. FEATURE ENGINEERING  (was features.py)
# ==============================================================================

def derive_speed_mps(toggle_signal: np.ndarray) -> float:
    """Rising-edge count -> revolutions -> m/s, from the raw 0/1 toggle column."""
    edges = np.sum((toggle_signal[1:] == 1) & (toggle_signal[:-1] == 0))
    revolutions = edges / N_TEETH
    return revolutions * WHEEL_CIRCUMFERENCE_M / DURATION_S


def _channel_cols(side_positions: list, signal_type: str) -> list:
    """signal_type: 'Vibration' or 'Shock'. Returns column names for all 8 cars."""
    cols = []
    for car in range(1, N_CARS + 1):
        for pos in side_positions:
            cols.append(f"{signal_type} of bearing in position {pos} of car {car}")
    return cols


def _fft_summary(x: np.ndarray) -> tuple:
    """Returns (dominant_frequency_hz, mid_band_energy_fraction) for one channel."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_HZ)
    mag = np.abs(np.fft.rfft(x - x.mean()))
    power = mag**2
    total_power = power.sum() + 1e-12
    dominant_freq = freqs[np.argmax(power)] if power.sum() > 0 else 0.0
    mid_band = (freqs >= 50) & (freqs <= 500)
    mid_band_frac = power[mid_band].sum() / total_power
    return dominant_freq, mid_band_frac


def _aggregate_channel_group(df: pd.DataFrame, cols: list, prefix: str) -> dict:
    rms_vals, std_vals, peak_vals, crest_vals, kurt_vals = [], [], [], [], []
    dom_freqs, mid_band_fracs = [], []

    for c in cols:
        x = df[c].values.astype(float)
        rms = np.sqrt(np.mean(x**2)) + 1e-9
        peak = np.max(np.abs(x))
        rms_vals.append(rms)
        std_vals.append(x.std())
        peak_vals.append(peak)
        crest_vals.append(peak / rms)
        kurt_vals.append(scipy_kurtosis(x, fisher=True, bias=False))
        dom_freq, mid_frac = _fft_summary(x)
        dom_freqs.append(dom_freq)
        mid_band_fracs.append(mid_frac)

    rms_vals, std_vals, peak_vals, crest_vals, kurt_vals = map(
        np.array, (rms_vals, std_vals, peak_vals, crest_vals, kurt_vals)
    )
    dom_freqs, mid_band_fracs = np.array(dom_freqs), np.array(mid_band_fracs)

    return {
        f"{prefix}_rms_mean": rms_vals.mean(),
        f"{prefix}_rms_max": rms_vals.max(),
        f"{prefix}_std_mean": std_vals.mean(),
        f"{prefix}_peak_mean": peak_vals.mean(),
        f"{prefix}_peak_max": peak_vals.max(),
        f"{prefix}_crest_mean": crest_vals.mean(),
        f"{prefix}_crest_max": crest_vals.max(),
        f"{prefix}_kurt_mean": kurt_vals.mean(),
        f"{prefix}_kurt_max": kurt_vals.max(),
        f"{prefix}_dom_freq_mean": dom_freqs.mean(),
        f"{prefix}_dom_freq_std": dom_freqs.std(),
        f"{prefix}_mid_band_frac_mean": mid_band_fracs.mean(),
    }


def extract_features(df: pd.DataFrame) -> dict:
    feats = {}
    speed = derive_speed_mps(df["Rotating speed"].values.astype(int))
    feats["speed_mps"] = speed

    for side_name, positions in [("side1", SIDE_I_POSITIONS), ("side2", SIDE_II_POSITIONS)]:
        for signal_type, sig_short in [("Vibration", "vib"), ("Shock", "shock")]:
            cols = _channel_cols(positions, signal_type)
            prefix = f"{side_name}_{sig_short}"
            feats.update(_aggregate_channel_group(df, cols, prefix))

    return feats


# ==============================================================================
# 2. DOWNLOAD + FEATURE-BUILD  (was build_features.py)
# ==============================================================================

def download_with_retry(url: str, dest: str):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
    raise last_exc


def already_done(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["filename"])
    return done


def build_features_range(start: int, end: int, out_path: str, base_url: str, name_prefix: str):
    """Downloads <name_prefix><N>.csv for N in [start, end], extracts features,
    deletes the raw file, and appends to a running feature table - so we never
    hold more than one ~17MB raw file on disk at a time. Resumable: skips
    filenames already present in out_path."""
    done = already_done(out_path)
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    t0 = time.time()
    n_processed = 0
    with open(out_path, "a", newline="") as f:
        writer = None
        for i in range(start, end + 1):
            fname = f"{name_prefix}{i}.csv"
            if fname in done:
                continue
            url = base_url + fname
            download_with_retry(url, TMP_FILE)
            df = pd.read_csv(TMP_FILE)
            feats = extract_features(df)
            feats["filename"] = fname
            if writer is None:
                fieldnames = list(feats.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
            writer.writerow(feats)
            f.flush()
            n_processed += 1
            if n_processed % 20 == 0 or i == end:
                elapsed = time.time() - t0
                print(f"  processed {fname} (this run: {n_processed}), elapsed {elapsed:.1f}s", flush=True)
    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)
    print(f"Done. {out_path} now has {len(already_done(out_path))} rows total.")


class RailModel:
    """Self-contained rail-corrugation classifier: wraps the fitted sklearn
    pipeline + its feature-column order + label encoder behind a single
    predict_file() call, so a caller (e.g. a FastAPI endpoint) doesn't need
    to separately juggle feature_cols/label_encoder - one object, one method,
    raw .csv in, decoded label out."""

    def __init__(self, pipeline, feature_cols: list, label_encoder: LabelEncoder, model_name: str):
        self.pipeline = pipeline
        self.feature_cols = feature_cols
        self.label_encoder = label_encoder
        self.model_name = model_name

    def predict_features(self, feats: dict) -> str:
        """feats: the dict extract_features() returns for one file."""
        X = pd.DataFrame([feats])[self.feature_cols]
        pred_code = self.pipeline.predict(X)[0]
        return self.label_encoder.inverse_transform([pred_code])[0]

    def predict_file(self, csv_path: str) -> str:
        """csv_path: one raw per-car rail-vibration .csv (Train<N>.csv/Test<N>.csv format)."""
        df = pd.read_csv(csv_path)
        feats = extract_features(df)
        return self.predict_features(feats)


# ==============================================================================
# 3. TRAINING / MODEL COMPARISON  (was train.py)
# ==============================================================================

def load_dataset(features_path: str = "train_features_full.csv", labels_path: str = "Train_Labels.csv") -> pd.DataFrame:
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    merged = features.merge(labels, on="filename", how="inner")
    assert len(merged) == len(features), "Some feature rows failed to match a label - check filenames."
    return merged


def prune_features(X: pd.DataFrame, corr_threshold: float = 0.95):
    stds = X.std()
    keep = stds[stds > 1e-8].index.tolist()
    dropped_constant = [c for c in X.columns if c not in keep]

    corr = X[keep].corr().abs()
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


def capped_smote(y: np.ndarray, multiplier: int = MAX_OVERSAMPLE_MULTIPLIER, random_state: int = RANDOM_STATE):
    """SMOTE with each minority class capped at `multiplier`x its own real count,
    rather than fully matched to the majority class - see module docstring."""
    counts = pd.Series(y).value_counts()
    majority_count = counts.max()
    minority_count_in_fold = counts.min()
    k_neighbors = max(1, min(5, minority_count_in_fold - 1))
    sampling_strategy = {
        cls: min(majority_count, count * multiplier) for cls, count in counts.items()
    }
    return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)


def make_candidate_models(n_features_selected: int, y_for_smote_sizing: np.ndarray):
    k = min(n_features_selected, 20)
    smote = capped_smote(y_for_smote_sizing)
    candidates = {}

    candidates["logreg"] = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", smote),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=3000, class_weight="balanced")),
    ])
    candidates["decision_tree"] = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", smote),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    candidates["random_forest"] = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", smote),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    candidates["gradient_boosting"] = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", smote),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE)),
    ])
    candidates["xgboost"] = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", smote),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1)),
    ])
    candidates["small_mlp"] = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", smote),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=3000, random_state=RANDOM_STATE, early_stopping=True)),
    ])
    return candidates


def _f1_for_class(cls):
    def scorer(estimator, X, y):
        preds = estimator.predict(X)
        return f1_score(y, preds, labels=[cls], average=None, zero_division=0)[0]
    return scorer


def _f1_macro_scorer(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average="macro", zero_division=0)


def make_scorers(side1_code: int, side2_code: int) -> dict:
    return {
        "f1_macro": _f1_macro_scorer,
        "f1_side1": _f1_for_class(side1_code),
        "f1_side2": _f1_for_class(side2_code),
    }


def repeated_cv_compare(models: dict, X: pd.DataFrame, y: np.ndarray, scorers: dict, n_splits: int = 5, n_repeats: int = 20) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scorers, n_jobs=-1, error_score="raise")
        rows.append({
            "model": name,
            "n_val_splits": len(scores["test_f1_macro"]),
            "f1_macro_mean": scores["test_f1_macro"].mean(),
            "f1_macro_std": scores["test_f1_macro"].std(),
            "f1_side1_mean": scores["test_f1_side1"].mean(),
            "f1_side2_mean": scores["test_f1_side2"].mean(),
        })
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def train_main(features_path: str, labels_path: str, model_out: str, results_out: str):
    table = load_dataset(features_path, labels_path)
    feature_cols = [c for c in table.columns if c not in ("filename", "label")]
    X_all = table[feature_cols]
    y_labels = table["label"].values

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_labels)  # XGBoost requires integer-coded classes
    side1_code = int(encoder.transform(["Side I"])[0])
    side2_code = int(encoder.transform(["Side II"])[0])
    scorers = make_scorers(side1_code, side2_code)

    print(f"Full dataset: {len(X_all)} files")
    print(pd.Series(y_labels).value_counts().to_string())

    keep_cols, dropped_const, dropped_corr = prune_features(X_all)
    print(f"\nFeature pruning: {len(feature_cols)} candidates -> {len(keep_cols)} kept")
    print(f"  dropped (near-constant): {dropped_const}")
    print(f"  dropped (collinear, corr>0.95): {dropped_corr}")

    X_all_p = X_all[keep_cols]
    models = make_candidate_models(len(keep_cols), y_all)

    print(f"\nRunning repeated stratified k-fold CV (5 folds x 20 repeats = 100 train/validation "
          f"splits) over all {len(X_all_p)} files (capped SMOTE applied inside each training fold only)...")
    cv_results = repeated_cv_compare(models, X_all_p, y_all, scorers)
    print(cv_results.to_string(index=False))

    best_name = cv_results.iloc[0]["model"]
    print(f"\n>>> Selected model: {best_name} "
          f"(mean macro F1={cv_results.iloc[0]['f1_macro_mean']:.3f} "
          f"+/- {cv_results.iloc[0]['f1_macro_std']:.3f} across 100 validation splits)")

    best_pipe = make_candidate_models(len(keep_cols), y_all)[best_name]
    best_pipe.fit(X_all_p, y_all)

    model = RailModel(best_pipe, keep_cols, encoder, best_name)
    joblib.dump(model, model_out)
    print(f"\nSaved trained pipeline (fit on all {len(X_all_p)} labelled files) -> {model_out}")

    with open(results_out, "w") as f:
        json.dump({
            "cv_results": cv_results.to_dict(orient="records"),
            "selected_model": best_name,
            "kept_features": keep_cols,
            "dropped_constant": dropped_const,
            "dropped_collinear": dropped_corr,
            "class_counts": pd.Series(y_labels).value_counts().to_dict(),
        }, f, indent=2)
    print(f"Saved run summary -> {results_out}")


# ==============================================================================
# 4. INFERENCE  (was predict.py)
# ==============================================================================

def predict_main(input_dir: str, output_path: str, model_path: str):
    model = joblib.load(model_path)

    files = sorted(
        glob.glob(os.path.join(input_dir, "*.csv")),
        key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))) if any(c.isdigit() for c in os.path.basename(p)) else 0,
    )
    rows = []
    for fp in files:
        fname = os.path.basename(fp)
        pred_label = model.predict_file(fp)
        rows.append({"file_id": fname, "prediction": pred_label})
        print(f"  {fname} -> {pred_label}")

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print(f"\nWrote {len(out)} predictions -> {output_path}")
    print(out["prediction"].value_counts())
    return out


# ==============================================================================
# CLI
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Rail Corrugation subsystem - full pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-features", help="Download Train<N>.csv files and extract features incrementally")
    p_build.add_argument("--start", type=int, required=True)
    p_build.add_argument("--end", type=int, required=True)
    p_build.add_argument("--out", default="train_features_full.csv")
    p_build.add_argument("--base-url", default=TRAIN_BASE_URL)
    p_build.add_argument("--prefix", default="Train")

    p_train = sub.add_parser("train", help="Compare candidate models via repeated CV, fit + save the best one")
    p_train.add_argument("--features", default="train_features_full.csv")
    p_train.add_argument("--labels", default="Train_Labels.csv")
    p_train.add_argument("--model-out", default="rail_model.pkl")
    p_train.add_argument("--results-out", default="cv_results.json")

    p_predict = sub.add_parser("predict", help="Run a saved model over a folder of Test<N>.csv files")
    p_predict.add_argument("--input", required=True, help="Folder containing Test<N>.csv files")
    p_predict.add_argument("--output", default="rail_predictions.csv")
    p_predict.add_argument("--model", default="rail_model.pkl")

    args = ap.parse_args()

    if args.command == "build-features":
        build_features_range(args.start, args.end, args.out, args.base_url, args.prefix)
    elif args.command == "train":
        train_main(args.features, args.labels, args.model_out, args.results_out)
    elif args.command == "predict":
        predict_main(args.input, args.output, args.model)


if __name__ == "__main__":
    # Running this file directly (`python rail_pipeline.py ...`) makes Python
    # execute it as a module named "__main__" - any class defined here (e.g.
    # RailModel) would then get pickled as belonging to "__main__", which no
    # other script can ever resolve via `import rail_pipeline; joblib.load(...)`.
    # Re-invoke through a real import of "rail_pipeline" so RailModel (and
    # anything pickled during this run) is attributed to that importable
    # module name instead, regardless of how this script was launched.
    import importlib
    import sys

    _mod = importlib.import_module("rail_pipeline")
    sys.exit(_mod.main())
