"""Train and validate the leak-localisation ranking model on the 6 labeled
training cases, using leave-one-file-out cross-validation scored with the
competition's own rank-decay metric (see scoring.py).

Per the modeling notes, the build order is: evaluate unsupervised per-file
anomaly-scoring baselines first (zero training, zero overfitting risk, works
immediately on the test file's schema), and only keep the supervised
logistic-regression combination layer if it measurably beats the best
baseline under leave-one-file-out CV. Saves the final chosen model (fit on
all 6 files) to model/acv_model.pkl.

Usage:
    python -m acv_fault.train --data-dir . --labels Train_Labels.csv --model-out model/acv_model.pkl
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .baselines import BASELINE_METHODS, BaselineModel, rank_from_score
from .data_loading import load_case_file
from .features import FEATURE_NAMES, build_feature_matrix
from .scoring import rank_decay_score


def load_training_table(data_dir: str, labels_path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    labels_df = pd.read_csv(labels_path, dtype=str)
    file_to_faulty = dict(zip(labels_df["filename"], labels_df["faulty_car"]))

    rows = []
    for fname in labels_df["filename"]:
        path = os.path.join(data_dir, fname)
        case = load_case_file(path)
        z = build_feature_matrix(case)
        z["file_id"] = fname
        z["car_id"] = z.index
        z["label"] = (z["car_id"] == file_to_faulty[fname]).astype(int)
        rows.append(z.reset_index(drop=True))

    table = pd.concat(rows, ignore_index=True)
    return table, file_to_faulty


def rank_cars_for_file(model: LogisticRegression, file_table: pd.DataFrame) -> list[str]:
    X = file_table[FEATURE_NAMES]
    scores = model.decision_function(X)
    order = np.argsort(-scores)
    return list(file_table.iloc[order]["car_id"])


def evaluate_lofo(table: pd.DataFrame, C: float) -> float:
    """Leave-one-file-out CV for a *fixed* C, scored with the competition's
    rank-decay metric. Used inside the inner loop of nested CV below, and to
    pick the C used for the final deployed model (which is allowed to use
    all 6 files — nested CV is for getting an *honest score estimate*, not
    for choosing the production hyperparameter)."""
    file_ids = table["file_id"].unique()
    scores = []
    for held_out in file_ids:
        train_df = table[table["file_id"] != held_out]
        test_df = table[table["file_id"] == held_out]

        model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        model.fit(train_df[FEATURE_NAMES], train_df["label"])

        ranked = rank_cars_for_file(model, test_df)
        true_faulty = test_df.loc[test_df["label"] == 1, "car_id"].iloc[0]
        scores.append(rank_decay_score(ranked, true_faulty))
    return float(np.mean(scores))


def evaluate_lofo_nested(table: pd.DataFrame, candidate_Cs: list[float]) -> float:
    """Nested leave-one-file-out CV: for each held-out file, C is chosen using
    only the *other 5* files (an inner LOFO over them), never using the outer
    held-out file's own score. This is the unbiased estimate of logistic
    regression's generalization — evaluate_lofo() picks one C by looking at
    all 6 outer scores at once and then reports that same number, which
    lets the outer folds influence the very hyperparameter being scored on
    them."""
    file_ids = table["file_id"].unique()
    outer_scores = []
    for held_out in file_ids:
        inner_table = table[table["file_id"] != held_out]
        inner_cv = {C: evaluate_lofo(inner_table, C) for C in candidate_Cs}
        inner_best_C = max(inner_cv, key=inner_cv.get)

        model = LogisticRegression(C=inner_best_C, max_iter=1000, class_weight="balanced")
        model.fit(inner_table[FEATURE_NAMES], inner_table["label"])

        test_df = table[table["file_id"] == held_out]
        ranked = rank_cars_for_file(model, test_df)
        true_faulty = test_df.loc[test_df["label"] == 1, "car_id"].iloc[0]
        outer_scores.append(rank_decay_score(ranked, true_faulty))
    return float(np.mean(outer_scores))


def baseline_cool_gap_score(table: pd.DataFrame) -> float:
    """Score of the single-feature heuristic (rank by cool_gap_mean alone)."""
    scores = []
    for file_id, file_table in table.groupby("file_id"):
        ranked = list(file_table.sort_values("cool_gap_mean", ascending=False)["car_id"])
        true_faulty = file_table.loc[file_table["label"] == 1, "car_id"].iloc[0]
        scores.append(rank_decay_score(ranked, true_faulty))
    return float(np.mean(scores))


def evaluate_unsupervised_baselines(table: pd.DataFrame) -> dict[str, float]:
    """Score every unsupervised anomaly-scoring method from baselines.py.

    These need no training at all (each file's 8 cars are scored purely
    against each other), so there is nothing to leave out — every file is
    just evaluated directly, independently of every other file.
    """
    results: dict[str, list[float]] = {name: [] for name in BASELINE_METHODS}
    for file_id, file_table in table.groupby("file_id"):
        Z = file_table.set_index("car_id")[FEATURE_NAMES]
        true_faulty = file_table.loc[file_table["label"] == 1, "car_id"].iloc[0]
        for name, method in BASELINE_METHODS.items():
            score = method(Z)
            ranked = rank_from_score(score)
            results[name].append(rank_decay_score(ranked, true_faulty))
    return {name: float(np.mean(scores)) for name, scores in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--labels", default="Train_Labels.csv")
    parser.add_argument("--model-out", default="model/acv_model.pkl")
    args = parser.parse_args()

    table, _ = load_training_table(args.data_dir, args.labels)

    print(f"Training table: {len(table)} car-rows across {table['file_id'].nunique()} files")

    print("\nStep 1: unsupervised baselines (no training, no labels used to fit them)")
    single_feature_score = baseline_cool_gap_score(table)
    print(f"  single-feature heuristic (cool_gap_mean only):  {single_feature_score:.4f}")
    baseline_scores = evaluate_unsupervised_baselines(table)
    for name, s in baseline_scores.items():
        print(f"  {name:22s}: {s:.4f}")
    best_baseline_name = max(baseline_scores, key=baseline_scores.get)
    best_baseline_score = baseline_scores[best_baseline_name]
    print(f"  -> best unsupervised baseline: {best_baseline_name} ({best_baseline_score:.4f})")

    print("\nStep 2: supervised combination layer")
    candidate_Cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

    # Non-nested number: picks C by looking at all 6 outer scores at once,
    # then reports that same number back — optimistic, kept only for context.
    cv_scores = {C: evaluate_lofo(table, C) for C in candidate_Cs}
    for C, s in cv_scores.items():
        print(f"  non-nested LOFO-CV score (C={C}): {s:.4f}  [optimistic — see nested score below]")
    best_C = max(cv_scores, key=cv_scores.get)

    # Honest number: each outer fold's C is chosen using only the other 5
    # files, never its own held-out score (see evaluate_lofo_nested docstring).
    nested_score = evaluate_lofo_nested(table, candidate_Cs)
    print(f"  -> nested LOFO-CV score (unbiased): {nested_score:.4f}")
    supervised_score = nested_score

    use_supervised = supervised_score > best_baseline_score
    print(
        f"\nDecision: {'supervised model' if use_supervised else best_baseline_name + ' baseline'} "
        f"wins ({max(supervised_score, best_baseline_score):.4f} vs "
        f"{min(supervised_score, best_baseline_score):.4f}) -> "
        f"{'keeping the logistic-regression combination layer' if use_supervised else 'falling back to the unsupervised baseline'}."
    )

    meta = {
        "feature_names": FEATURE_NAMES,
        "best_C": best_C,
        "non_nested_lofo_cv_score": cv_scores[best_C],
        "nested_lofo_cv_score": nested_score,
        "single_feature_baseline_score": single_feature_score,
        "unsupervised_baseline_scores": baseline_scores,
        "best_unsupervised_baseline": {"name": best_baseline_name, "score": best_baseline_score},
        "chosen_model": "logistic_regression" if use_supervised else best_baseline_name,
    }

    if use_supervised:
        final_model = LogisticRegression(C=best_C, max_iter=1000, class_weight="balanced")
        final_model.fit(table[FEATURE_NAMES], table["label"])
        meta["coefficients"] = dict(zip(FEATURE_NAMES, final_model.coef_[0].tolist()))
    else:
        final_model = BaselineModel(best_baseline_name)

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(final_model, args.model_out)
    meta_path = os.path.join(os.path.dirname(args.model_out) or ".", "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model to {args.model_out} and metadata to {meta_path}")


if __name__ == "__main__":
    main()
