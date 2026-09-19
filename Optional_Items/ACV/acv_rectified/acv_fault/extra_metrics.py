"""Classification-style metrics (F1, false-positive/negative rates) and a
rank-based prediction-error metric, computed from the deployed model's
predictions on the 6 labeled training cases.

The submitted model's native output is a ranking, not a binary
faulty/not-faulty label, so there is no F1/FPR/FNR without first choosing how
many of the ranked cars count as a "positive" prediction. Fixed at k=1 (only
the top-ranked car per file counts as "predicted faulty") since that's the
fairest comparison to the actual submission: a single best guess per file,
judged on whether it's exactly right.

Usage:
    python -m acv_fault.extra_metrics --data-dir . --labels Train_Labels.csv --model model/acv_model.pkl
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from .baselines import BaselineModel
from .features import FEATURE_NAMES
from .train import load_training_table, rank_cars_for_file


def confusion_counts_at_k(table, model, k: int) -> dict[str, int]:
    """TP/FP/TN/FN aggregated over all 6 files, treating each file's top-k
    ranked cars as 'predicted faulty' and the rest as 'predicted normal'."""
    tp = fp = tn = fn = 0
    for file_id, file_table in table.groupby("file_id"):
        ranked = rank_cars_for_file(model, file_table)
        predicted_positive = set(ranked[:k])
        true_faulty = file_table.loc[file_table["label"] == 1, "car_id"].iloc[0]
        for car_id in file_table["car_id"]:
            is_true_positive_case = car_id == true_faulty
            is_predicted_positive = car_id in predicted_positive
            if is_predicted_positive and is_true_positive_case:
                tp += 1
            elif is_predicted_positive and not is_true_positive_case:
                fp += 1
            elif not is_predicted_positive and is_true_positive_case:
                fn += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # = true positive rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall_tpr": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy": accuracy,
    }


def rank_prediction_error(table, model) -> dict[str, float]:
    """Rank displacement of the true faulty car from 1st place, and mean
    reciprocal rank (MRR) — both standard ranking-error/quality metrics,
    reported alongside the competition's own rank-decay score."""
    errors = []
    reciprocal_ranks = []
    per_file = {}
    for file_id, file_table in table.groupby("file_id"):
        ranked = rank_cars_for_file(model, file_table)
        true_faulty = file_table.loc[file_table["label"] == 1, "car_id"].iloc[0]
        rank = ranked.index(true_faulty) + 1
        errors.append(rank - 1)
        reciprocal_ranks.append(1.0 / rank)
        per_file[file_id] = rank
    return {
        "mean_absolute_rank_error": float(np.mean(errors)),
        "max_rank_error": int(np.max(errors)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "per_file_rank": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--labels", default="Train_Labels.csv")
    parser.add_argument("--model", default="model/acv_model.pkl")
    args = parser.parse_args()

    table, _ = load_training_table(args.data_dir, args.labels)
    model = joblib.load(args.model)

    print("Rank-based prediction error (6 labeled training cases):")
    rank_stats = rank_prediction_error(table, model)
    for file_id, rank in rank_stats["per_file_rank"].items():
        print(f"  {file_id}: true faulty car ranked {rank}/8")
    print(f"  mean absolute rank error: {rank_stats['mean_absolute_rank_error']:.4f}")
    print(f"  max rank error: {rank_stats['max_rank_error']}")
    print(f"  mean reciprocal rank (MRR): {rank_stats['mean_reciprocal_rank']:.4f}")

    print("\nClassification-style metrics (top-1: only the top-ranked car per file counts as 'predicted faulty'):")
    counts = confusion_counts_at_k(table, model, k=1)
    m = metrics_from_confusion(counts)
    all_results = {"rank_error": rank_stats, "top_1": {**counts, **m}}
    print(f"  TP={counts['tp']} FP={counts['fp']} TN={counts['tn']} FN={counts['fn']} "
          f"| precision={m['precision']:.3f} recall={m['recall_tpr']:.3f} f1={m['f1']:.3f} "
          f"FPR={m['false_positive_rate']:.4f} FNR={m['false_negative_rate']:.4f} acc={m['accuracy']:.3f}")

    with open("model/extra_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to model/extra_metrics.json")


if __name__ == "__main__":
    main()
