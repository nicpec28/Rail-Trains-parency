from pathlib import Path
from io import BytesIO
import pickle
import traceback

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "rail": MODEL_DIR / "rail_model.pkl",
    "door": MODEL_DIR / "door_model.pkl",
    "shm": MODEL_DIR / "shm_model.pkl",
    "acv": MODEL_DIR / "acv_model.pkl",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Rail Fault Prediction API",
    version="1.0.0",
)


# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODELS
# ============================================================

MODELS = {}


def load_models():

    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    for model_name, model_path in MODEL_PATHS.items():

        if not model_path.exists():

            print(
                f"[WARNING] {model_name}: "
                f"FILE NOT FOUND -> {model_path}"
            )

            continue

        try:

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            MODELS[model_name] = model

            print(
                f"[OK] {model_name}: "
                f"{model_path.name}"
            )

            # Show information if the pickle is a dictionary
            if isinstance(model, dict):

                print(
                    f"     Keys: "
                    f"{list(model.keys())}"
                )

                if "feature_cols" in model:

                    print(
                        f"     Feature count: "
                        f"{len(model['feature_cols'])}"
                    )

                if "model_name" in model:

                    print(
                        f"     Model name: "
                        f"{model['model_name']}"
                    )

        except Exception as e:

            print(
                f"[ERROR] Could not load "
                f"{model_name}: {e}"
            )

    print("=" * 70)

    print(
        "Loaded models:",
        list(MODELS.keys())
    )

    print("=" * 70 + "\n")


load_models()


# ============================================================
# MODEL HELPERS
# ============================================================

def get_model_pipeline(model):
    """
    Supports both:

    A) Direct sklearn model/pipeline

       model = Pipeline(...)

    B) Dictionary bundle

       {
           "pipeline": ...,
           "feature_cols": [...],
           "model_name": "...",
           "label_encoder": ...
       }
    """

    if isinstance(model, dict):

        pipeline = model.get("pipeline")

        if pipeline is None:

            raise ValueError(
                "This .pkl is a dictionary but "
                "does not contain a 'pipeline'."
            )

        return pipeline

    return model


def get_feature_columns(model):

    if isinstance(model, dict):

        return model.get("feature_cols")

    return None


def get_label_encoder(model):

    if isinstance(model, dict):

        return model.get("label_encoder")

    return None


def make_json_safe(value):

    """
    Convert NumPy/sklearn objects into JSON-compatible
    Python objects.
    """

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, np.generic):

        return value.item()

    if isinstance(value, dict):

        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, tuple):

        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(value, list):

        return [
            make_json_safe(v)
            for v in value
        ]

    return value


# ============================================================
# PREDICTION
# ============================================================

async def predict_model(
    model_name: str,
    file: UploadFile,
):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model_name not in MODELS:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_name}' is not loaded. "
                f"Available models: "
                f"{list(MODELS.keys())}"
            ),
        )

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file uploaded.",
        )

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are currently supported.",
        )

    try:

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        contents = await file.read()

        df = pd.read_csv(
            BytesIO(contents)
        )

        if df.empty:

            raise ValueError(
                "The uploaded CSV is empty."
            )

        print("\n" + "=" * 70)
        print(f"RUNNING MODEL: {model_name}")
        print(f"FILE: {file.filename}")
        print(
            f"DATA: "
            f"{df.shape[0]} rows x "
            f"{df.shape[1]} columns"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Get model
        # ----------------------------------------------------

        model = MODELS[model_name]

        pipeline = get_model_pipeline(model)

        feature_cols = get_feature_columns(model)

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        X = df.copy()

        # If the pickle tells us exactly which features
        # it expects, select them.

        if feature_cols:

            missing_features = [
                feature
                for feature in feature_cols
                if feature not in X.columns
            ]

            if missing_features:

                raise ValueError(
                    "CSV is missing required features: "
                    + ", ".join(
                        map(str, missing_features)
                    )
                )

            X = X[feature_cols]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = pipeline.predict(X)

        prediction = make_json_safe(
            prediction
        )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = None
        confidence = None

        if hasattr(
            pipeline,
            "predict_proba"
        ):

            try:

                proba = pipeline.predict_proba(X)

                probabilities = make_json_safe(
                    proba
                )

                if len(proba) > 0:

                    confidence = float(
                        np.max(proba[0])
                    )

            except Exception as probability_error:

                print(
                    "[INFO] "
                    "predict_proba not available: "
                    f"{probability_error}"
                )

        # ----------------------------------------------------
        # Class names
        # ----------------------------------------------------

        classes = None

        label_encoder = get_label_encoder(
            model
        )

        if (
            label_encoder is not None
            and hasattr(
                label_encoder,
                "classes_"
            )
        ):

            classes = make_json_safe(
                label_encoder.classes_
            )

        elif hasattr(
            pipeline,
            "classes_"
        ):

            classes = make_json_safe(
                pipeline.classes_
            )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        result = {
            "status": "success",

            "model": model_name,

            "model_name": (
                model.get("model_name")
                if isinstance(model, dict)
                else model_name
            ),

            "filename": file.filename,

            "row_count": int(
                len(df)
            ),

            "feature_count": int(
                X.shape[1]
            ),

            "prediction": prediction,

            "confidence": confidence,

            "probabilities": probabilities,

            "classes": classes,

            "features_used": (
                feature_cols
                if feature_cols
                else list(X.columns)
            ),
        }

        print("\nPrediction:")
        print(prediction)

        if confidence is not None:

            print(
                f"Confidence: "
                f"{confidence:.4f}"
            )

        print("=" * 70 + "\n")

        return result

    except HTTPException:

        raise

    except Exception as e:

        print("\n" + "=" * 70)
        print(f"MODEL ERROR: {model_name}")
        print("=" * 70)

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "model": model_name,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Rail Fault Prediction API",
        "status": "running",
        "models_loaded": list(
            MODELS.keys()
        ),
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "models_loaded": list(
            MODELS.keys()
        ),
    }


@app.get("/models")
def get_models():

    return {
        "models": list(
            MODELS.keys()
        )
    }


# ============================================================
# RAIL
# ============================================================

@app.post("/rail/predict")
async def predict_rail(
    file: UploadFile = File(...)
):

    return await predict_model(
        "rail",
        file
    )


# ============================================================
# DOOR
# ============================================================

@app.post("/door/predict")
async def predict_door(
    file: UploadFile = File(...)
):

    return await predict_model(
        "door",
        file
    )


# ============================================================
# SHM
# ============================================================

@app.post("/shm/predict")
async def predict_shm(
    file: UploadFile = File(...)
):

    return await predict_model(
        "shm",
        file
    )


# ============================================================
# ACV
# ============================================================

@app.post("/acv/predict")
async def predict_acv(
    file: UploadFile = File(...)
):

    return await predict_model(
        "acv",
        file
    )