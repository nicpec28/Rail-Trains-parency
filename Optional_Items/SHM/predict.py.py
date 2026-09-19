import os
import sys
import pandas as pd
import numpy as np
import joblib

from feature_extractor import extract_features

# Import custom classes so joblib can deserialize the saved model
from model_components import VIFSelector, MonotonicAwareRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "Test/"
MODEL_PATH = "shm_model.pkl"

OUTPUT_FILE = "shm_predictions.csv"


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Saved model not found: {MODEL_PATH}"
    )

print(f"Loading model: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)

if not isinstance(bundle, dict):
    raise ValueError(
        "The saved model file does not contain the expected model bundle."
    )

required_keys = [
    "model",
    "feature_columns",
]

missing_keys = [
    key for key in required_keys
    if key not in bundle
]

if missing_keys:
    raise ValueError(
        f"Model bundle is missing required keys: {missing_keys}"
    )

model = bundle["model"]
expected_features = bundle["feature_columns"]

print(f"Model loaded successfully.")
print(f"Expected features: {len(expected_features)}")


# ============================================================
# FIND TEST FILES
# ============================================================

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

test_files = sorted(
    [
        f
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith(".csv")
    ]
)

if not test_files:
    raise ValueError(
        f"No CSV files found in {TEST_DIR}"
    )

print(f"\nFound {len(test_files)} test files.")


# ============================================================
# EXTRACT FEATURES
# ============================================================

feature_rows = []
file_ids = []

for filename in test_files:

    filepath = os.path.join(TEST_DIR, filename)

    print(f"Extracting features: {filename}")

    try:
        features = extract_features(filepath)

        if features is None:
            print(f"[WARNING] No features returned for {filename}")
            continue

        # Make sure the extracted features are represented as a dict
        if isinstance(features, pd.Series):
            features = features.to_dict()

        elif isinstance(features, pd.DataFrame):
            if len(features) != 1:
                raise ValueError(
                    f"Expected one feature row, got {len(features)}"
                )

            features = features.iloc[0].to_dict()

        feature_rows.append(features)
        file_ids.append(filename)

    except Exception as e:
        print(
            f"[ERROR] Failed to extract features from "
            f"{filename}: {e}"
        )


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

if not feature_rows:
    raise ValueError(
        "No test files were successfully processed."
    )

X_test = pd.DataFrame(feature_rows, index=file_ids)

print(f"\nExtracted feature matrix:")
print(f"Samples:  {len(X_test)}")
print(f"Features: {X_test.shape[1]}")


# ============================================================
# ALIGN FEATURES TO TRAINING MODEL
# ============================================================

missing_features = [
    col
    for col in expected_features
    if col not in X_test.columns
]

if missing_features:
    raise ValueError(
        "\nTest data is missing features required by the model:\n"
        + "\n".join(missing_features)
    )

extra_features = [
    col
    for col in X_test.columns
    if col not in expected_features
]

if extra_features:
    print(
        f"\n[NOTE] Ignoring {len(extra_features)} "
        f"extra extracted features."
    )

# IMPORTANT:
# Use exactly the same feature order as during training.
X_test = X_test[expected_features].copy()


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if X_test.isna().any().any():

    missing = X_test.columns[
        X_test.isna().any()
    ].tolist()

    raise ValueError(
        "\nNaN values detected in test features:\n"
        + "\n".join(missing)
    )

if not np.isfinite(X_test.to_numpy(dtype=float)).all():

    raise ValueError(
        "Infinite values detected in test features."
    )


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

predictions = np.asarray(predictions).reshape(-1)


# ============================================================
# OUTPUT
# ============================================================

output_df = pd.DataFrame({
    "file_id": file_ids,
    "prediction": predictions,
})

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)

print(f"Test files processed: {len(output_df)}")
print(f"Output file:          {OUTPUT_FILE}")

print("\nPredictions:")
print(output_df.to_string(index=False))

print("\n" + "=" * 60)