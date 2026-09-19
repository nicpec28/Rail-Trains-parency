"""Inference script for ACV refrigerant-leak localisation.

For every input .xlsx case file, ranks all cars in that file from most- to
least-likely to have the refrigerant leak fault, and writes one row per file
to the output CSV in the required submission format (see
ACV_Subsystem_Info_Kit.md, Section 3):

    file_id, ranked_cars

Usage:
    python predict.py --input acv_test_case.xlsx --output acv_predictions.csv
    python predict.py --input path/to/test_folder --output acv_predictions.csv

--input may be a single .xlsx file or a directory of .xlsx files (one row is
written per file found).
"""
from __future__ import annotations

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd

from acv_fault.data_loading import load_case_file
from acv_fault.features import FEATURE_NAMES, build_feature_matrix

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "acv_model.pkl")


def rank_cars(model, feature_matrix: pd.DataFrame) -> list[str]:
    X = feature_matrix[FEATURE_NAMES]
    scores = model.decision_function(X)
    order = np.argsort(-scores)
    return list(feature_matrix.index[order])


def predict_file(model, path: str) -> tuple[str, list[str]]:
    case = load_case_file(path)
    z = build_feature_matrix(case)
    ranked = rank_cars(model, z)
    return os.path.basename(path), ranked


def collect_input_files(input_path: str) -> list[str]:
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.xlsx")))
        if not files:
            raise FileNotFoundError(f"No .xlsx files found in directory: {input_path}")
        return files
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(f"--input path not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to a case .xlsx file, or a directory of .xlsx files")
    parser.add_argument("--output", required=True, help="Path to write acv_predictions.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model (joblib)")
    args = parser.parse_args()

    model = joblib.load(args.model)
    input_files = collect_input_files(args.input)

    rows = []
    for path in input_files:
        file_id, ranked = predict_file(model, path)
        rows.append({"file_id": file_id, "ranked_cars": "|".join(ranked)})
        print(f"{file_id}: {'|'.join(ranked)}")

    out_df = pd.DataFrame(rows, columns=["file_id", "ranked_cars"])
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
