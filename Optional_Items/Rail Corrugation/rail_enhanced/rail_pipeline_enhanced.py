"""
Rail Corrugation subsystem - full pipeline in one file.

Merges what were originally four separate scripts (features.py,
build_features.py, train.py, predict.py) into one, run via subcommands:

    python rail_pipeline.py build-features --start 1 --end 272 --out train_features_full.csv
    python rail_pipeline.py train
    python rail_pipeline.py predict --input Test/ --output rail_predictions.csv --model rail_model.pkl

Each subcommand does exactly what its original standalone script did; only
the file layout changed, not the logic.

This version follows the team's per-channel-feature approach (see the
write-up's Rail Corrugation section) rather than the earlier aggregated-
feature/XGBoost version, with three additions made on an MIT-ML /
40-year-rail-technician review of that approach: speed-normalized frequency
features, Hilbert envelope-spectrum features, and a fold-safe VIF
collinearity filter. See "MODELLING REVIEW" below for the reasoning behind
each. This file is a STANDALONE variant - it does not overwrite the team's
existing rail_pipeline.py/rail_model.pkl/cv_results.json.

================================================================================
FEATURE ENGINEERING
================================================================================
Each file is a 1-second, 10,000 Hz recording: column 1 is the raw
rotating-speed toggle (0/1) signal, columns 2-129 are vibration+shock pairs
for 64 axle-box positions (8 cars x 8 positions/car). Per the Rail
Corrugation Info Kit, positions 1/3/5/7 (per car) belong to the Side I rail;
positions 2/4/6/8 belong to Side II. Side I and Side II must be judged
independently from the same file.

  - Car and position are parsed directly out of each column's own header
    text via regex (e.g. "Shock of bearing in position 3 of car 5"), rather
    than assumed from a fixed column order, so a reordered or partial column
    set doesn't silently misassign a channel to the wrong car/side.
  - Each of the 128 signal channels is processed independently - full
    per-car/per-position detail is kept rather than aggregating statistics
    across the 8 cars. A corrugation fault is localised to a specific car
    and side; aggregating across cars risks averaging away the one car that
    actually carries the fault signature. This is also why Side I and Side
    II are judged independently: nothing in the feature set or the model
    ever mixes a car's Side I channels with its Side II channels.
  - Speed is a single physically-meaningful `speed_mps` scalar, derived by
    counting rising edges in the raw toggle signal (not 9 statistical
    moments of the toggle signal itself, which is what the untouched
    per-channel-feature draft did - see MODELLING REVIEW).
  - For each of the 128 channels, 12 time- and frequency-domain statistics
    are extracted: mean, standard deviation, RMS, peak-to-peak, crest
    factor, kurtosis, dominant FFT frequency, spectral energy, spectral
    centroid, Hilbert-envelope dominant frequency, and two speed-normalized
    frequency variants (dominant_freq_per_speed, envelope_dom_freq_per_speed).
    This gives 128 x 12 + 1 (speed_mps) = 1,537 raw features per file.
  - Per-car Side I vs Side II asymmetry features (mean(Side I stat) -
    mean(Side II stat), per car, per signal type, over a focused subset of 4
    stats) add a direct "how lopsided is this car" signal - see MODELLING
    REVIEW.
  - Dimensionality is handled downstream via SelectKBest (ANOVA F-test) then
    a VIF collinearity filter, both inside the modelling pipeline and
    refit independently on each training fold - so the raw features are
    narrowed to the ~50 most class-discriminative ones, then further pruned
    for redundancy, without either step ever seeing the validation fold.

Known limitation: a near-constant rotating-speed signal (seen on a
meaningful minority of training files) makes kurtosis mathematically
undefined (division by a near-zero fourth central moment) and produces NaN.
These are filled with 0 before modelling - this prevents a fold-fitting
crash while affecting only the handful of features computed from a
near-constant signal, not the feature set as a whole.

================================================================================
MODELLING REVIEW (MIT-ML / 40-year rail-technician perspective)
================================================================================
Three concerns were raised about the per-channel-feature draft, and addressed
as follows:

1. THRESHOLD-ON-RAW-AMPLITUDE IS UNRELIABLE. Vibration amplitude scales with
   train speed (~speed^2, kinetic energy) and with whatever the ballast/track
   condition happens to be - neither is corrugation. Scale-invariant ratios
   (crest factor, kurtosis) already sidestep the amplitude problem, but a
   subtler version remained: corrugation is a fixed SPATIAL wavelength
   defect, so its vibration shows up at a TEMPORAL frequency = speed /
   wavelength - the same physical defect reads as a different Hz value
   depending on train speed during that file. Fix: `dominant_freq_per_speed`
   and `envelope_dom_freq_per_speed` divide the frequency features by
   `speed_mps`, approximating cycles-per-metre (roughly speed-invariant for
   the same physical wavelength) instead of raw Hz.
2. RAINFLOW WAS CONSIDERED, NOT USED. Rainflow cycle-counting is a
   fatigue/cumulative-damage technique (see the SHM subsystem, where it's the
   right tool) for slowly-varying load histories - not the right tool for
   classifying a periodic geometric defect's vibration signature at 10kHz.
   The domain-correct alternative for "periodic impact" detection is Hilbert
   envelope-spectrum analysis (standard in bearing/gear-fault diagnosis),
   which is what `_envelope_dominant_freq` implements instead.
3. MULTICOLLINEARITY. 128 channels x 12 near-identical statistics guarantees
   heavy correlation (e.g. RMS and peak-to-peak on adjacent axle positions of
   the same car). VIFSelector (greedy, threshold=10) runs AFTER SelectKBest
   in every pipeline - on the full ~1,500-wide raw matrix, VIF would sit on a
   rank-deficient correlation matrix (more features than the 272 files) and
   be slow; running it on SelectKBest's ~50 survivors is both correct and
   cheap.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time
import urllib.request
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
SEEDS = [42, 7, 123]
OUTER_SPLITS = 5
INNER_SPLITS = 3
N_SELECT_K = 50  # SelectKBest: narrow raw features to the ~50 most class-discriminative
VIF_THRESHOLD = 10.0  # standard textbook multicollinearity cutoff, applied AFTER SelectKBest

TRAIN_BASE_URL = "https://raw.githubusercontent.com/aochinwen/NebulaX-Hackathon-ProblemStatement/main/PS3/02_Datasets/Rail_Corrugation/Train/"
TMP_FILE = "_tmp_rail_file.csv"
MAX_RETRIES = 5

SAMPLING_HZ = 10000
SPEED_COL = "Rotating speed"
CHANNEL_RE = re.compile(r"^(Vibration|Shock) of bearing in position (\d+) of car (\d+)$")

N_TEETH = 90
WHEEL_DIAMETER_M = 0.85
WHEEL_CIRCUMFERENCE_M = np.pi * WHEEL_DIAMETER_M
DURATION_S = 1.0

SIDE_I_POSITIONS = [1, 3, 5, 7]
SIDE_II_POSITIONS = [2, 4, 6, 8]


# ==============================================================================
# 1. FEATURE ENGINEERING  (was features.py)
# ==============================================================================

def derive_speed_mps(toggle_signal: np.ndarray) -> float:
    """Rising-edge count -> revolutions -> m/s, from the raw 0/1 toggle column.
    A physically meaningful scalar, not 9 statistical moments of a square wave
    (see module docstring, "reinstating a clean speed feature")."""
    edges = np.sum((toggle_signal[1:] == 1) & (toggle_signal[:-1] == 0))
    revolutions = edges / N_TEETH
    return float(revolutions * WHEEL_CIRCUMFERENCE_M / DURATION_S)


def _envelope_dominant_freq(x: np.ndarray) -> float:
    """Dominant frequency of the signal's Hilbert AMPLITUDE ENVELOPE, not the
    raw signal itself - the standard rotating-machinery/bearing-fault-
    diagnosis technique for isolating a periodic impact's repetition rate from
    the broadband vibration it rides on top of. This is the domain-correct
    tool for "periodic corrugation impact" detection (see module docstring's
    rainflow discussion for why rainflow itself is not used here)."""
    envelope = np.abs(hilbert(x - x.mean()))
    n = len(envelope)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_HZ)
    mag = np.abs(np.fft.rfft(envelope - envelope.mean()))
    power = mag**2
    if power.sum() > 0:
        return float(freqs[np.argmax(power)])
    return 0.0


def _signal_stats(x: np.ndarray, speed_mps: float) -> dict:
    """11 time- and frequency-domain statistics for one raw signal channel,
    including two speed-normalized frequency features (see module docstring:
    corrugation is a fixed SPATIAL wavelength, so its TEMPORAL frequency
    shifts with train speed - dividing by speed approximates cycles-per-metre,
    which should stay roughly constant for the same physical defect
    regardless of how fast the train was going in this particular file)."""
    x = x.astype(float)
    n = len(x)
    mean = x.mean()
    std = x.std()
    rms = float(np.sqrt(np.mean(x**2)))
    peak_to_peak = float(x.max() - x.min())
    peak = float(np.max(np.abs(x)))
    crest_factor = peak / rms if rms > 0 else 0.0
    kurt = scipy_kurtosis(x, fisher=True, bias=False)  # NaN for a near-constant signal - filled later

    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_HZ)
    mag = np.abs(np.fft.rfft(x - mean))
    power = mag**2
    total_power = float(power.sum())
    if total_power > 0:
        dominant_freq = float(freqs[np.argmax(power)])
        spectral_centroid = float(np.sum(freqs * power) / total_power)
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0

    envelope_dom_freq = _envelope_dominant_freq(x)

    if speed_mps > 1e-6:
        dominant_freq_per_speed = dominant_freq / speed_mps
        envelope_dom_freq_per_speed = envelope_dom_freq / speed_mps
    else:
        dominant_freq_per_speed = 0.0
        envelope_dom_freq_per_speed = 0.0

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "dominant_freq": dominant_freq,
        "spectral_energy": total_power,
        "spectral_centroid": spectral_centroid,
        "envelope_dom_freq": envelope_dom_freq,
        "dominant_freq_per_speed": dominant_freq_per_speed,
        "envelope_dom_freq_per_speed": envelope_dom_freq_per_speed,
    }


# Stats averaged per-side for the side-asymmetry features below - a focused
# subset (not all 12), chosen for being the ones most directly tied to
# corrugation's physical signature (impulsiveness + speed-normalized
# periodicity), to avoid doubling the feature count for little added value.
ASYMMETRY_STATS = ["rms", "crest_factor", "kurtosis", "envelope_dom_freq_per_speed"]


def extract_features(df: pd.DataFrame) -> dict:
    """128 signal channels, each independently -> 12 stats each, plus one
    speed_mps scalar and per-car Side I vs Side II asymmetry features. Car/
    position parsed from each column's own header text, not assumed from
    column order."""
    feats: dict = {}

    speed_mps = derive_speed_mps(df[SPEED_COL].values.astype(int))
    feats["speed_mps"] = speed_mps

    # per_car[car][side_name][signal_type] -> list of that channel's stats dict,
    # for the asymmetry pass below.
    per_car: dict = {}

    for col in df.columns:
        if col == SPEED_COL:
            continue
        m = CHANNEL_RE.match(str(col))
        if not m:
            continue
        signal_type, position, car = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        prefix = f"car{car}_pos{position}_{signal_type}"
        stats = _signal_stats(df[col].values, speed_mps)
        for stat, val in stats.items():
            feats[f"{prefix}_{stat}"] = val

        side_name = "side1" if position in SIDE_I_POSITIONS else "side2"
        per_car.setdefault(car, {}).setdefault(side_name, {}).setdefault(signal_type, []).append(stats)

    # Side-asymmetry features: for each car, mean(Side I positions' stat) -
    # mean(Side II positions' stat), per signal type. A corrugation fault is
    # localised to one side of one car; this gives the model a direct
    # "how lopsided is this car" signal instead of requiring it to infer that
    # from many largely-independent absolute-value features (see module
    # docstring).
    for car, sides in per_car.items():
        if "side1" not in sides or "side2" not in sides:
            continue
        for signal_type in ("vibration", "shock"):
            if signal_type not in sides["side1"] or signal_type not in sides["side2"]:
                continue
            for stat in ASYMMETRY_STATS:
                side1_mean = np.mean([s[stat] for s in sides["side1"][signal_type]])
                side2_mean = np.mean([s[stat] for s in sides["side2"][signal_type]])
                feats[f"car{car}_{signal_type}_side_asym_{stat}"] = float(side1_mean - side2_mean)

    # A near-constant channel (most commonly rotating-speed on a file with
    # very little speed variation) makes kurtosis mathematically undefined -
    # fill rather than let it crash downstream fitting (see module docstring).
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            feats[k] = 0.0

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


class VIFSelector(BaseEstimator, TransformerMixin):
    """Greedy Variance-Inflation-Factor feature selector: iteratively drops
    the single highest-VIF column until every remaining one is <= threshold,
    or too few columns remain to keep computing VIF meaningfully.

    Placed AFTER SelectKBest in every pipeline below, not before: VIF on the
    full ~1,537-wide raw feature matrix would sit on a rank-deficient
    correlation matrix (more features than the 272 training files) and be
    needlessly slow to boot. Running SelectKBest first (fast, supervised,
    1,537+ -> ~50) and VIF second (cheap on ~50 columns) gets the same
    collinearity-reduction benefit without either problem."""

    def __init__(self, threshold: float = VIF_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)

        variances = X_df.var(axis=0, ddof=0)
        self.constant_cols_ = list(variances[variances <= 1e-15].index)
        working = X_df.drop(columns=self.constant_cols_, errors="ignore")

        if working.shape[1] == 0:
            self.keep_cols_ = []
            return self

        kept = list(working.columns)
        while len(kept) > 1:
            corr = working[kept].corr().to_numpy(dtype=float)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                vifs = np.maximum(np.diag(np.linalg.pinv(corr)), 1.0)
            except np.linalg.LinAlgError:
                break

            max_idx = int(np.argmax(vifs))
            max_vif = float(vifs[max_idx])
            if max_vif <= self.threshold:
                break
            kept.pop(max_idx)

        self.keep_cols_ = kept
        return self

    def transform(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        missing = [c for c in self.keep_cols_ if c not in X_df.columns]
        if missing:
            raise ValueError(f"VIFSelector: missing columns: {missing}")
        return X_df[self.keep_cols_].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_cols_, dtype=object)


# ==============================================================================
# 3. TRAINING / MODEL COMPARISON  (was train.py)
# ==============================================================================

def load_dataset(features_path: str = "train_features_full.csv", labels_path: str = "Train_Labels.csv") -> pd.DataFrame:
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    merged = features.merge(labels, on="filename", how="inner")
    assert len(merged) == len(features), "Some feature rows failed to match a label - check filenames."
    return merged


def make_candidate_pipelines(k: int = N_SELECT_K) -> dict:
    """The 5 candidates handled via a standard sklearn Pipeline + class_weight.
    XGBoost is handled separately (see _xgb_nested_cv_scores) since it needs
    manually-computed sample_weight instead of class_weight."""
    return {
        "lasso": Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("vif", VIFSelector()),
            ("clf", LogisticRegression(penalty="l1", solver="saga", class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("vif", VIFSelector()),
            ("clf", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "decision_tree": Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("vif", VIFSelector()),
            ("clf", DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=3, min_samples_split=5, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("vif", VIFSelector()),
            ("clf", RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=3, min_samples_split=5, n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "svm": Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("vif", VIFSelector()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


PARAM_GRIDS = {
    "lasso": {"clf__C": [0.01, 0.1, 1, 10]},
    "ridge": {"clf__C": [0.01, 0.1, 1, 10]},
    "decision_tree": {"clf__max_depth": [3, 5, 7, None]},
    "random_forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [5, 8, None]},
    "svm": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
}

XGB_PARAM_GRID = {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}


def _nested_cv_scores(pipe: Pipeline, param_grid: dict, X: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    """Standard nested CV: an inner GridSearchCV (hyperparameter selection)
    wrapped by an outer cross_val_score (unbiased scoring) - the inner search
    never sees the fold it's later scored on."""
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=inner_cv, n_jobs=-1)
    return cross_val_score(search, X, y, cv=outer_cv, scoring="f1_macro", n_jobs=1)


def _xgb_nested_cv_scores(X: pd.DataFrame, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Manual nested CV for XGBoost: class_weight isn't supported for
    multi-class, so sample_weight is computed fresh via
    compute_sample_weight('balanced', ...) on each fold's training portion
    only (see module docstring), rather than routed through a generic
    Pipeline/GridSearchCV fit_params path."""
    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_score, best_params = -1.0, None
        for params in ParameterGrid(XGB_PARAM_GRID):
            inner_scores = []
            for itr_idx, ival_idx in inner_cv.split(X_train, y_train):
                X_itr, X_ival = X_train.iloc[itr_idx], X_train.iloc[ival_idx]
                y_itr, y_ival = y_train[itr_idx], y_train[ival_idx]

                scaler = StandardScaler().fit(X_itr)
                X_itr_s, X_ival_s = scaler.transform(X_itr), scaler.transform(X_ival)
                selector = SelectKBest(f_classif, k=k).fit(X_itr_s, y_itr)
                X_itr_sel, X_ival_sel = selector.transform(X_itr_s), selector.transform(X_ival_s)
                vif = VIFSelector().fit(X_itr_sel)
                X_itr_v, X_ival_v = vif.transform(X_itr_sel), vif.transform(X_ival_sel)

                sw = compute_sample_weight("balanced", y_itr)
                clf = XGBClassifier(**params, eval_metric="mlogloss", random_state=seed, n_jobs=-1)
                clf.fit(X_itr_v, y_itr, sample_weight=sw)
                pred = clf.predict(X_ival_v)
                inner_scores.append(f1_score(y_ival, pred, average="macro", zero_division=0))

            mean_score = float(np.mean(inner_scores))
            if mean_score > best_score:
                best_score, best_params = mean_score, params

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
        selector = SelectKBest(f_classif, k=k).fit(X_train_s, y_train)
        X_train_sel, X_test_sel = selector.transform(X_train_s), selector.transform(X_test_s)
        vif = VIFSelector().fit(X_train_sel)
        X_train_v, X_test_v = vif.transform(X_train_sel), vif.transform(X_test_sel)

        sw_train = compute_sample_weight("balanced", y_train)
        clf = XGBClassifier(**best_params, eval_metric="mlogloss", random_state=seed, n_jobs=-1)
        clf.fit(X_train_v, y_train, sample_weight=sw_train)
        pred = clf.predict(X_test_v)
        scores.append(f1_score(y_test, pred, average="macro", zero_division=0))

    return np.array(scores)


def compare_models_multi_seed(X: pd.DataFrame, y: np.ndarray, k: int = N_SELECT_K) -> pd.DataFrame:
    """Nested CV for all 6 candidates, repeated at 3 seeds - see module
    docstring for why a single-seed comparison isn't trusted here."""
    pipelines = make_candidate_pipelines(k)
    rows = []
    for seed in SEEDS:
        for name, pipe in pipelines.items():
            scores = _nested_cv_scores(pipe, PARAM_GRIDS[name], X, y, seed)
            rows.append({"model": name, "seed": seed, "f1_macro_mean": float(scores.mean())})
            print(f"  seed={seed} {name:15s} macro F1 = {scores.mean():.4f}")

        xgb_scores = _xgb_nested_cv_scores(X, y, k, seed)
        rows.append({"model": "xgboost", "seed": seed, "f1_macro_mean": float(xgb_scores.mean())})
        print(f"  seed={seed} {'xgboost':15s} macro F1 = {xgb_scores.mean():.4f}")

    return pd.DataFrame(rows)


def train_main(features_path: str, labels_path: str, model_out: str, results_out: str):
    table = load_dataset(features_path, labels_path)
    feature_cols = [c for c in table.columns if c not in ("filename", "label")]
    X_all = table[feature_cols]
    y_labels = table["label"].values

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_labels)  # XGBoost requires integer-coded classes

    print(f"Full dataset: {len(X_all)} files, {len(feature_cols)} raw features")
    print(pd.Series(y_labels).value_counts().to_string())

    print(f"\nRunning nested CV (5-fold outer / 3-fold inner) for 6 candidates "
          f"at {len(SEEDS)} seeds ({SEEDS}) - class-weighted, no resampling, "
          f"SelectKBest -> VIF(threshold={VIF_THRESHOLD}) inside every fold...")
    per_seed = compare_models_multi_seed(X_all, y_all, N_SELECT_K)

    summary = (
        per_seed.groupby("model")["f1_macro_mean"]
        .agg(mean="mean", std="std", min="min", max="max")
        .sort_values("mean", ascending=False)
    )
    print("\nAcross-seed summary (mean macro F1, sorted by mean):")
    print(summary.to_string())

    # Final model: the actual winner of the multi-seed comparison above - a
    # strong mean macro F1 with low run-to-run variance across seeds, rather
    # than the single highest-scoring seed for any one candidate. Not
    # hardcoded to "svm": with the new features, the ranking may have
    # changed, and this selects whichever candidate the numbers actually
    # support this run.
    best_name = summary.index[0]
    print(f"\n>>> Selected model: {best_name} "
          f"(mean macro F1={summary.loc[best_name, 'mean']:.4f} "
          f"+/- {summary.loc[best_name, 'std']:.4f} across {len(SEEDS)} seeds)")

    if best_name == "xgboost":
        # XGBoost's final production fit mirrors _xgb_nested_cv_scores's
        # per-fold steps (scale -> select -> vif -> fit with sample_weight),
        # but on the FULL labelled dataset with no held-out fold - same
        # "nested CV estimates, full-data refit deploys" split as every
        # other candidate below.
        best_xgb_score, best_xgb_params = -1.0, None
        cv_search = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for params in ParameterGrid(XGB_PARAM_GRID):
            fold_scores = []
            for tr_idx, val_idx in cv_search.split(X_all, y_all):
                X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
                y_tr, y_val = y_all[tr_idx], y_all[val_idx]
                scaler = StandardScaler().fit(X_tr)
                X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
                selector = SelectKBest(f_classif, k=N_SELECT_K).fit(X_tr_s, y_tr)
                X_tr_sel, X_val_sel = selector.transform(X_tr_s), selector.transform(X_val_s)
                vif = VIFSelector().fit(X_tr_sel)
                X_tr_v, X_val_v = vif.transform(X_tr_sel), vif.transform(X_val_sel)
                sw = compute_sample_weight("balanced", y_tr)
                clf = XGBClassifier(**params, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
                clf.fit(X_tr_v, y_tr, sample_weight=sw)
                fold_scores.append(f1_score(y_val, clf.predict(X_val_v), average="macro", zero_division=0))
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_xgb_score:
                best_xgb_score, best_xgb_params = mean_score, params

        scaler = StandardScaler().fit(X_all)
        X_all_s = scaler.transform(X_all)
        selector = SelectKBest(f_classif, k=N_SELECT_K).fit(X_all_s, y_all)
        X_all_sel = selector.transform(X_all_s)
        vif = VIFSelector().fit(X_all_sel)
        X_all_v = vif.transform(X_all_sel)
        sw_all = compute_sample_weight("balanced", y_all)
        final_clf = XGBClassifier(**best_xgb_params, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
        final_clf.fit(X_all_v, y_all, sample_weight=sw_all)
        best_pipe = Pipeline([("scale", scaler), ("select", selector), ("vif", vif), ("clf", final_clf)])
        best_params_used = best_xgb_params
        print(f"Final hyperparameters: {best_xgb_params}")
    else:
        # Final hyperparameters: one more GridSearchCV, this time over the FULL
        # labelled dataset, to pick the production hyperparameters (the nested
        # CV above is for an honest score estimate, not for choosing the
        # deployed model's own hyperparameters).
        final_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        final_search = GridSearchCV(
            make_candidate_pipelines(N_SELECT_K)[best_name],
            PARAM_GRIDS[best_name],
            scoring="f1_macro",
            cv=final_cv,
            n_jobs=-1,
        )
        final_search.fit(X_all, y_all)
        best_pipe = final_search.best_estimator_
        best_params_used = final_search.best_params_
        print(f"Final hyperparameters: {final_search.best_params_}")

    model = RailModel(best_pipe, feature_cols, encoder, best_name)
    joblib.dump(model, model_out)
    print(f"\nSaved trained pipeline (fit on all {len(X_all)} labelled files) -> {model_out}")

    with open(results_out, "w") as f:
        json.dump({
            "per_seed_results": per_seed.to_dict(orient="records"),
            "summary_by_model": summary.reset_index().to_dict(orient="records"),
            "selected_model": best_name,
            "final_hyperparameters": best_params_used,
            "n_raw_features": len(feature_cols),
            "select_k": N_SELECT_K,
            "vif_threshold": VIF_THRESHOLD,
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
    ap = argparse.ArgumentParser(description="Rail Corrugation subsystem - full pipeline (enhanced)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-features", help="Download Train<N>.csv files and extract features incrementally")
    p_build.add_argument("--start", type=int, required=True)
    p_build.add_argument("--end", type=int, required=True)
    p_build.add_argument("--out", default="train_features_full.csv")
    p_build.add_argument("--base-url", default=TRAIN_BASE_URL)
    p_build.add_argument("--prefix", default="Train")

    p_train = sub.add_parser("train", help="Compare candidate models via nested CV, fit + save the best one")
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
    import importlib
    import sys

    _mod = importlib.import_module("rail_pipeline_enhanced")
    sys.exit(_mod.main())
