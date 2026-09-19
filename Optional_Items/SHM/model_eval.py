import os

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from scipy.stats import (
    loguniform,
    uniform,
    randint,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.svm import SVR

from sklearn.neighbors import (
    KNeighborsRegressor,
)

from sklearn.linear_model import (
    ElasticNet,
    Ridge,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.decomposition import (
    PCA,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.compose import (
    TransformedTargetRegressor,
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold,
    RepeatedKFold,
    cross_val_score,
)

from sklearn.metrics import (
    make_scorer,
    mean_absolute_percentage_error,
)

from sklearn.inspection import permutation_importance

from feature_extractor import extract_features

from model_components import (
    VIFSelector,
    MonotonicAwareRegressor,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_DIR = "Train/"
LABELS_FILE = "Train_Labels.csv"

# --------------------------------------------------------------------------
# Single reproducible random seed.
#
# The experiment uses ONE random state, but the outer CV is still repeated
# three times, giving 15 outer-fold evaluations per model.
# --------------------------------------------------------------------------

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Hyperparameter-search budget.
#
# n_iter is the number of random hyperparameter configurations sampled
# by RandomizedSearchCV.
#
# These are deliberately moderate because the dataset is small
# (~64 samples). Increasing n_iter increases computation substantially.
# --------------------------------------------------------------------------

N_ITER_SIMPLE = 20
N_ITER_MEDIUM = 30
N_ITER_COMPLEX = 40


# --------------------------------------------------------------------------
# Nested cross-validation.
#
# Inner CV:
#   Used to select hyperparameters.
#
# Outer CV:
#   Used to estimate generalization performance after hyperparameter
#   selection.
#
# 5 folds × 3 repeats = 15 outer evaluations per model.
# --------------------------------------------------------------------------

OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3


# --------------------------------------------------------------------------
# Heuristic warning threshold only.
#
# overfit_gap compares:
#
#     nested-CV MAPE
#         -
#     final-model in-sample MAPE
#
# These are NOT produced by exactly the same fitted model, so this is
# only a diagnostic prompt, not a statistical test of overfitting.
# --------------------------------------------------------------------------

OVERFIT_GAP_WARN_THRESHOLD = 0.05


# --------------------------------------------------------------------------
# Columns that VIFSelector must never eliminate.
#
# Cycle-bin histogram fractions are compositional and sum to approximately
# 1 by construction, so greedy VIF elimination is not particularly
# well-founded for them.
#
# Monotonic features must also survive so that
# MonotonicAwareRegressor can enforce a consistent constraint set.
# --------------------------------------------------------------------------

VIF_PROTECTED_PREFIXES = (
    "cycle_bin_",
)


# ============================================================================
# PHYSICS MONOTONIC CONSTRAINTS
# ============================================================================

MONOTONIC_INCREASING_FEATURES = [

    "rms_ac",

    "peak_to_peak",

    "max_cycle_range",

    "log_pseudo_damage_m3",

    "log_pseudo_damage_m5",
]


MONOTONIC_DECREASING_FEATURES = []


# ============================================================================
# LOAD DATA
# ============================================================================

def load_training_data():

    labels_df = pd.read_csv(
        LABELS_FILE
    )

    required_columns = {
        "filename",
        "damage",
    }

    missing = (
        required_columns
        - set(labels_df.columns)
    )

    if missing:

        raise ValueError(
            f"{LABELS_FILE} is missing "
            f"columns: {sorted(missing)}"
        )

    X_list = []
    y_list = []

    print(
        "Extracting features from "
        "training files..."
    )

    missing_files = []

    for _, row in labels_df.iterrows():

        filename = row["filename"]

        filepath = os.path.join(
            TRAIN_DIR,
            filename
        )

        if not os.path.exists(filepath):

            missing_files.append(
                filename
            )

            continue

        features = extract_features(
            filepath
        )

        X_list.append(
            features
        )

        y_list.append(
            float(row["damage"])
        )

    if missing_files:

        print(
            f"\n[WARNING] {len(missing_files)} file(s) listed in "
            f"{LABELS_FILE} were not found under '{TRAIN_DIR}' and "
            "were silently skipped:"
        )

        for name in missing_files:

            print(
                f"  - {name}"
            )

        print(
            "\nThis reduces the effective training set size below "
            "what the label file lists. Verify that this is expected "
            "rather than a path/data bug.\n"
        )

    if not X_list:

        raise ValueError(
            "No training files were found."
        )

    X = pd.DataFrame(
        X_list
    )

    y = np.asarray(
        y_list,
        dtype=float
    )

    # ------------------------------------------------------------------------
    # Validate target.
    # ------------------------------------------------------------------------

    if not np.all(
        np.isfinite(y)
    ):

        raise ValueError(
            "Damage labels contain "
            "NaN or infinite values."
        )

    if not np.all(
        y >= 0
    ):

        raise ValueError(
            "Found negative damage "
            f"labels: {y[y < 0][:10]}"
        )

    # ------------------------------------------------------------------------
    # Zero-damage diagnostic.
    #
    # log1p is well-defined at zero, but MAPE is problematic when the true
    # target is zero. We therefore explicitly report how many zero targets
    # exist before the model comparison.
    # ------------------------------------------------------------------------

    n_zero_damage = int(
        np.sum(y == 0)
    )

    print(
        f"\nZero-damage samples: "
        f"{n_zero_damage} / {len(y)}"
    )

    if n_zero_damage > 0:

        print(
            "[WARNING] MAPE is problematic when true damage is zero. "
            "Scikit-learn uses a small epsilon internally, which can "
            "make errors on zero-damage samples extremely large."
        )

    # ------------------------------------------------------------------------
    # Validate features.
    # ------------------------------------------------------------------------

    if X.isna().any().any():

        bad_columns = list(
            X.columns[
                X.isna().any()
            ]
        )

        raise ValueError(
            "Features contain NaN values "
            f"in: {bad_columns}"
        )

    if not np.all(
        np.isfinite(
            X.to_numpy(
                dtype=float
            )
        )
    ):

        raise ValueError(
            "Features contain "
            "infinite values."
        )

    print(
        f"\nLoaded {len(X)} samples "
        f"with {X.shape[1]} features."
    )

    return X, y


# ============================================================================
# BUILD MODEL CONFIGURATIONS
# ============================================================================

def build_algorithms(
    n_features,
    random_state,
):

    # PCA needs at least 2 features.

    if n_features < 2:

        raise ValueError(
            "Need at least 2 features "
            "for KNN + PCA."
        )

    # ------------------------------------------------------------------------
    # PCA n_components is searched as a variance-retention fraction rather
    # than a fixed integer component count.
    #
    # This is safer because VIFSelector runs before PCA and can leave a
    # different number of features in different CV folds.
    # ------------------------------------------------------------------------

    pca_distribution = uniform(
        0.70,
        0.29
    )

    return {

        # ====================================================================
        # RIDGE
        # ====================================================================

        "Ridge (L2 Linear)": {

            "estimator":
                Ridge(
                    random_state=random_state,
                    max_iter=20000,
                ),

            "use_pca": False,

            "monotonic": False,

            "distributions": {

                "regressor__model__alpha":
                    loguniform(
                        1e-2,
                        1e4
                    ),

                "regressor__vif__threshold":
                    uniform(
                        3.0,
                        27.0
                    ),
            },

            # 2 dimensions:
            # alpha + VIF threshold
            "n_iter":
                N_ITER_SIMPLE,
        },


        # ====================================================================
        # ELASTIC NET
        # ====================================================================

        "ElasticNet (L1/L2 Linear)": {

            "estimator":
                ElasticNet(
                    random_state=random_state,
                    max_iter=20000,
                ),

            "use_pca": False,

            "monotonic": False,

            "distributions": {

                "regressor__model__alpha":
                    loguniform(
                        1e-3,
                        1e2
                    ),

                "regressor__model__l1_ratio":
                    uniform(
                        0.0,
                        1.0
                    ),

                "regressor__vif__threshold":
                    uniform(
                        3.0,
                        27.0
                    ),
            },

            # 3 dimensions:
            # alpha + l1_ratio + VIF threshold
            "n_iter":
                N_ITER_SIMPLE,
        },


        # ====================================================================
        # KNN + PCA
        # ====================================================================

        "KNN + PCA (Distance)": {

            "estimator":
                KNeighborsRegressor(),

            "use_pca": True,

            "monotonic": False,

            "distributions": {

                "regressor__pca__n_components":
                    pca_distribution,

                "regressor__model__n_neighbors":
                    randint(
                        3,
                        15
                    ),

                "regressor__model__weights":
                    [
                        "uniform",
                        "distance",
                    ],

                "regressor__model__p":
                    [
                        1,
                        2,
                    ],

                "regressor__vif__threshold":
                    uniform(
                        3.0,
                        27.0
                    ),
            },

            # 5 dimensions:
            # PCA variance + neighbors + weights + p + VIF threshold
            "n_iter":
                N_ITER_SIMPLE,
        },


        # ====================================================================
        # SVR
        # ====================================================================

        "Support Vector Regressor (SVR)": {

            "estimator":
                SVR(),

            "use_pca": False,

            "monotonic": False,

            "distributions": {

                "regressor__model__C":
                    loguniform(
                        1e-2,
                        1e3
                    ),

                "regressor__model__epsilon":
                    loguniform(
                        1e-3,
                        1.0
                    ),

                "regressor__model__kernel":
                    [
                        "linear",
                        "rbf",
                    ],

                "regressor__model__gamma":
                    [
                        "scale",
                        "auto",
                        0.01,
                        0.1,
                        1,
                    ],

                "regressor__vif__threshold":
                    uniform(
                        3.0,
                        27.0
                    ),
            },

            # 5 dimensions:
            # C + epsilon + kernel + gamma + VIF threshold
            "n_iter":
                N_ITER_MEDIUM,
        },


        # ====================================================================
        # RANDOM FOREST
        # ====================================================================

        "Random Forest (Bagging)": {

            "estimator":
                RandomForestRegressor(
                    random_state=random_state,
                    n_jobs=1,
                ),

            "use_pca": False,

            "needs_scaling": False,

            "monotonic": False,

            "distributions": {

                "regressor__model__n_estimators":
                    randint(
                        50,
                        500
                    ),

                "regressor__model__max_depth":
                    [
                        3,
                        4,
                        5,
                        6,
                        8,
                        10,
                        None,
                    ],

                "regressor__model__min_samples_split":
                    randint(
                        4,
                        15
                    ),

                "regressor__model__min_samples_leaf":
                    randint(
                        2,
                        8
                    ),

                "regressor__model__max_features":
                    [
                        "sqrt",
                        "log2",
                        1.0,
                    ],

                "regressor__vif__threshold":
                    uniform(
                        3.0,
                        27.0
                    ),
            },

            # 6 dimensions
            "n_iter":
                N_ITER_COMPLEX,
        },


        # ====================================================================
        # HISTOGRAM GRADIENT BOOSTING
        # ====================================================================

        "HistGradientBoosting": {

            "estimator":
                HistGradientBoostingRegressor(
                    random_state=random_state
                ),

            "use_pca": False,

            "needs_scaling": False,

            "monotonic": True,

            "distributions": {

                "regressor__model__model__max_iter":
                    randint(
                        50,
                        300
                    ),

                "regressor__model__model__learning_rate":
                    loguniform(
                        1e-3,
                        0.2
                    ),

                "regressor__model__model__max_depth":
                    [
                        3,
                        4,
                        5,
                        6,
                    ],

                "regressor__model__model__l2_regularization":
                    loguniform(
                        1e-3,
                        10
                    ),

                "regressor__model__model__min_samples_leaf":
                    randint(
                        5,
                        20
                    ),

                "regressor__vif__threshold":
                    uniform(
                        3.0,
                        27.0
                    ),
            },

            # 6 dimensions
            "n_iter":
                N_ITER_COMPLEX,
        },


        # ====================================================================
        # XGBOOST
        # ====================================================================

        "XGBoost (Boosting)": {

            "estimator":
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=1,
                ),

            "use_pca": False,

            "needs_scaling": False,

            "monotonic": True,

            "distributions": {

                "regressor__model__model__n_estimators":
                    randint(
                        50,
                        300
                    ),

                "regressor__model__model__learning_rate":
                    loguniform(
                        0.01,
                        0.2
                    ),

                "regressor__model__model__max_depth":
                    randint(
                        2,
                        6
                    ),

                "regressor__model__model__reg_alpha":
                    loguniform(
                        1e-4,
                        10
                    ),

                "regressor__model__model__reg_lambda":
                    loguniform(
                        1e-2,
                        10
                    ),

                "regressor__model__model__subsample":
                    uniform(
                        0.5,
                        0.4
                    ),

                "regressor__model__model__colsample_bytree":
                    uniform(
                        0.5,
                        0.4
                    ),

                "regressor__model__model__gamma":
                    loguniform(
                        1e-3,
                        2.0
                    ),

                "regressor__vif__threshold":
                    uniform(
                        3.0,
                        27.0
                    ),
            },

            # 9 dimensions:
            # n_estimators + learning_rate + max_depth +
            # reg_alpha + reg_lambda + subsample +
            # colsample_bytree + gamma + VIF threshold
            "n_iter":
                N_ITER_COMPLEX,
        },
    }


# ============================================================================
# BUILD PIPELINE
# ============================================================================

def build_pipeline(
    config,
    vif_threshold=10.0,
    random_state=None
):

    vif = (
        VIFSelector(
            threshold=vif_threshold,

            protected_prefixes=
                VIF_PROTECTED_PREFIXES,

            protected_features=(
                MONOTONIC_INCREASING_FEATURES
                + MONOTONIC_DECREASING_FEATURES
            ),
        )
        .set_output(
            transform="pandas"
        )
    )

    steps = [
        (
            "vif",
            vif
        ),
    ]

    # ------------------------------------------------------------------------
    # Scaling
    #
    # Scaling matters for:
    #
    #   Ridge
    #   ElasticNet
    #   SVR
    #   KNN
    #   PCA
    #
    # It is unnecessary for tree-based models.
    # ------------------------------------------------------------------------

    if config.get(
        "needs_scaling",
        True
    ):

        scaler = (
            StandardScaler()
            .set_output(
                transform="pandas"
            )
        )

        steps.append(
            (
                "scaler",
                scaler
            )
        )

    # ------------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------------

    if config.get(
        "use_pca"
    ):

        steps.append(
            (
                "pca",

                PCA(
                    n_components=None,

                    # Float n_components in (0, 1) requires full SVD.
                    svd_solver="full",

                    random_state=random_state
                )
            )
        )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    if config.get(
        "monotonic"
    ):

        model_step = (
            MonotonicAwareRegressor(

                model=config[
                    "estimator"
                ],

                increasing_features=
                    MONOTONIC_INCREASING_FEATURES,

                decreasing_features=
                    MONOTONIC_DECREASING_FEATURES,
            )
        )

    else:

        model_step = config[
            "estimator"
        ]

    steps.append(
        (
            "model",
            model_step
        )
    )

    pipe = Pipeline(
        steps
    )

    # ------------------------------------------------------------------------
    # Target transformation.
    #
    # log1p(y) = log(1 + y)
    #
    # This works for y >= 0, including y = 0.
    # Predictions are transformed back using expm1.
    # ------------------------------------------------------------------------

    return TransformedTargetRegressor(

        regressor=pipe,

        func=np.log1p,

        inverse_func=np.expm1,
    )


# ============================================================================
# SANITY-CHECK PHYSICS ASSUMPTIONS
# ============================================================================

def check_monotonic_assumptions(
    X,
    y
):

    """
    Print empirical Spearman correlations for all features that are
    subjected to monotonic constraints.

    This is only a sanity check.

    A positive/negative Spearman correlation does NOT prove that the
    relationship is universally monotonic. It simply identifies obvious
    contradictions between the empirical data and the assumed direction.
    """

    from scipy.stats import spearmanr

    constrained = (
        [
            (
                f,
                1
            )

            for f in MONOTONIC_INCREASING_FEATURES
        ]

        +

        [
            (
                f,
                -1
            )

            for f in MONOTONIC_DECREASING_FEATURES
        ]
    )

    if not constrained:

        return

    print(
        "\nSanity-checking monotonic constraint assumptions "
        "(Spearman correlation with damage):"
    )

    for feature, expected_sign in constrained:

        if feature not in X.columns:

            print(
                f"  {feature:<28} "
                "not present in extracted features "
                "-- skipped"
            )

            continue

        rho, _ = spearmanr(
            X[feature],
            y
        )

        contradicted = (

            (
                expected_sign > 0
                and rho < 0
            )

            or

            (
                expected_sign < 0
                and rho > 0
            )
        )

        flag = (
            " <-- CONTRADICTS ASSUMED DIRECTION"
            if contradicted
            else ""
        )

        expected_str = (
            "increasing"
            if expected_sign > 0
            else "decreasing"
        )

        print(
            f"  {feature:<28} "
            f"spearman r={rho:+.3f} "
            f"(assumed {expected_str})"
            f"{flag}"
        )

    print("")


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(
    X,
    y,
    random_state
):

    print(
        f"\nDataset: {len(X)} samples"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "\nEvaluation metric: MAPE"
    )

    print(
        "SHM score = max(0, 1 - MAPE)"
    )

    print(
        "\nLower MAPE = higher SHM score.\n"
    )

    # ------------------------------------------------------------------------
    # Physics sanity check.
    # ------------------------------------------------------------------------

    check_monotonic_assumptions(
        X,
        y
    )

    # ------------------------------------------------------------------------
    # Inner CV.
    #
    # Hyperparameters are selected inside this CV.
    # ------------------------------------------------------------------------

    inner_cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state,
    )

    # ------------------------------------------------------------------------
    # Outer CV.
    #
    # 5 folds × 3 repeats = 15 held-out evaluations.
    # ------------------------------------------------------------------------

    outer_cv = RepeatedKFold(
        n_splits=OUTER_N_SPLITS,
        n_repeats=OUTER_N_REPEATS,
        random_state=random_state,
    )

    # ------------------------------------------------------------------------
    # MAPE scorer.
    #
    # greater_is_better=False means sklearn returns NEGATIVE MAPE.
    # ------------------------------------------------------------------------

    mape_scorer = make_scorer(
        mean_absolute_percentage_error,
        greater_is_better=False,
    )

    algorithms = build_algorithms(
        X.shape[1],
        random_state
    )

    results = []

    checkpoint_path = (
        f"model_comparison_checkpoint_seed_"
        f"{random_state}.pkl"
    )

    # ------------------------------------------------------------------------
    # Evaluate every model family.
    # ------------------------------------------------------------------------

    for name, config in algorithms.items():

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Evaluating {name}"
        )

        print(
            f"Random search iterations: "
            f"{config['n_iter']}"
        )

        print(
            "=" * 70
        )

        try:

            pipeline = build_pipeline(
                config,
                random_state=random_state
            )

            search = RandomizedSearchCV(

                estimator=pipeline,

                param_distributions=
                    config["distributions"],

                n_iter=
                    config["n_iter"],

                scoring=
                    mape_scorer,

                cv=
                    inner_cv,

                random_state=
                    random_state,

                n_jobs=-1,

                # A failed candidate is assigned NaN instead of
                # aborting the entire model search.
                error_score=np.nan,
            )

            # ----------------------------------------------------------------
            # Nested CV.
            # ----------------------------------------------------------------

            nested_scores = cross_val_score(

                search,

                X,

                y,

                cv=
                    outer_cv,

                scoring=
                    mape_scorer,

                # Keep outer folds sequential to avoid multiplying
                # the already-parallel inner search.
                n_jobs=1,

                error_score=np.nan,
            )

            if np.all(
                np.isnan(nested_scores)
            ):

                raise RuntimeError(
                    "All outer folds failed to produce "
                    "a score. See warnings above."
                )

            # sklearn returns NEGATIVE MAPE.
            fold_mape = -nested_scores

            n_failed_folds = int(
                np.isnan(fold_mape).sum()
            )

            if n_failed_folds:

                print(
                    f"[WARNING] {n_failed_folds} of "
                    f"{len(fold_mape)} outer folds failed "
                    f"for {name} and were excluded."
                )

            cv_mape = np.nanmean(
                fold_mape
            )

            cv_mape_std = np.nanstd(
                fold_mape
            )

            # ----------------------------------------------------------------
            # SHM score.
            # ----------------------------------------------------------------

            estimated_shm_score = max(
                0.0,
                1.0 - cv_mape
            )

            print(
                f"\nCV MAPE: "
                f"{cv_mape:.6f}"
            )

            print(
                f"MAPE std: "
                f"{cv_mape_std:.6f}"
            )

            print(
                f"Estimated SHM score: "
                f"{estimated_shm_score:.6f}"
            )

            results.append({

                "Model":
                    name,

                "CV_MAPE":
                    cv_mape,

                "CV_MAPE_std":
                    cv_mape_std,

                "Estimated_SHM_Score":
                    estimated_shm_score,

                "Search_Object":
                    search,
            })

        except Exception as exc:

            print(
                f"\n[ERROR] {name} failed and is being "
                f"excluded from the leaderboard: "
                f"{exc!r}"
            )

        # --------------------------------------------------------------------
        # Checkpoint after every model.
        # --------------------------------------------------------------------

        try:

            joblib.dump(
                results,
                checkpoint_path
            )

        except Exception as checkpoint_exc:

            print(
                f"[WARNING] Could not write checkpoint: "
                f"{checkpoint_exc!r}"
            )

    # ------------------------------------------------------------------------
    # Make sure at least one model succeeded.
    # ------------------------------------------------------------------------

    if not results:

        raise RuntimeError(
            "Every model family failed. "
            f"See errors above. Partial state was checkpointed "
            f"to {checkpoint_path}."
        )

    # =========================================================================
    # RANK BY MAPE
    # =========================================================================

    results = sorted(
        results,
        key=lambda x:
            x["CV_MAPE"]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL LEADERBOARD — RANKED BY CV MAPE"
    )

    print(
        "=" * 80
    )

    for rank, result in enumerate(
        results,
        1
    ):

        print(
            f"{rank}. "
            f"{result['Model'].ljust(32)} | "
            f"MAPE: "
            f"{result['CV_MAPE']:.6f} | "
            f"SHM score: "
            f"{result['Estimated_SHM_Score']:.6f} | "
            f"std: "
            f"{result['CV_MAPE_std']:.6f}"
        )

    print(
        "=" * 80
    )

    # =========================================================================
    # TRAIN WINNER ON ALL TRAINING DATA
    # =========================================================================

    winner = results[0]

    print(
        f"\nWinning model: "
        f"{winner['Model']}"
    )

    print(
        f"Expected CV MAPE: "
        f"{winner['CV_MAPE']:.6f}"
    )

    print(
        f"Estimated SHM score: "
        f"{winner['Estimated_SHM_Score']:.6f}"
    )

    print(
        "\nRefitting hyperparameter search "
        "on 100% of training data..."
    )

    best_search = winner[
        "Search_Object"
    ]

    best_search.fit(
        X,
        y
    )

    best_estimator = (
        best_search.best_estimator_
    )

    # =========================================================================
    # TRAIN MAPE
    # =========================================================================

    train_predictions = (
        best_estimator.predict(
            X
        )
    )

    train_mape = (
        mean_absolute_percentage_error(
            y,
            train_predictions
        )
    )

    train_shm_equivalent = max(
        0.0,
        1.0 - train_mape
    )

    overfit_gap = (
        winner["CV_MAPE"]
        - train_mape
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MODEL DIAGNOSTICS"
    )

    print(
        "=" * 70
    )

    print(
        f"Train MAPE: "
        f"{train_mape:.6f}"
    )

    print(
        f"Train SHM-equivalent: "
        f"{train_shm_equivalent:.6f}"
    )

    print(
        f"Nested CV MAPE: "
        f"{winner['CV_MAPE']:.6f}"
    )

    print(
        f"Estimated CV SHM score: "
        f"{winner['Estimated_SHM_Score']:.6f}"
    )

    print(
        f"MAPE std: "
        f"{winner['CV_MAPE_std']:.6f}"
    )

    print(
        f"Overfit gap: "
        f"{overfit_gap:.6f}"
    )

    print(
        "=" * 70
    )

    if (
        overfit_gap
        > OVERFIT_GAP_WARN_THRESHOLD
    ):

        print(
            "\n[WARNING] Train MAPE is substantially "
            "lower than CV MAPE."
        )

        print(
            "This may indicate overfitting."
        )

    # =========================================================================
    # BEST HYPERPARAMETERS
    # =========================================================================

    print(
        "\nWinning hyperparameters:"
    )

    for key, value in (
        best_search
        .best_params_
        .items()
    ):

        print(
            f"  {key}: {value}"
        )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    print(
        "\nComputing permutation feature importance on "
        "the final model (in-sample)..."
    )

    feature_importances = {}

    try:

        perm_result = permutation_importance(

            best_estimator,

            X,

            y,

            scoring=
                mape_scorer,

            n_repeats=20,

            random_state=
                random_state,

            n_jobs=1,
        )

        importance_order = np.argsort(
            perm_result.importances_mean
        )[::-1]

        print(
            "\nTop features by permutation importance "
            "(mean MAPE-score drop when shuffled):"
        )

        for rank_i, idx in enumerate(
            importance_order[:15],
            1
        ):

            col = X.columns[idx]

            mean_imp = float(
                perm_result
                .importances_mean[idx]
            )

            std_imp = float(
                perm_result
                .importances_std[idx]
            )

            feature_importances[col] = {

                "mean":
                    mean_imp,

                "std":
                    std_imp,
            }

            print(
                f"  {rank_i}. "
                f"{col:<28} "
                f"{mean_imp:+.5f} "
                f"+/- {std_imp:.5f}"
            )

    except Exception as exc:

        print(
            f"[WARNING] Permutation importance failed: "
            f"{exc!r}"
        )

    # =========================================================================
    # TRAINING RANGE ENVELOPE
    # =========================================================================

    vif_step = (
        best_estimator
        .regressor_
        .named_steps["vif"]
    )

    kept_cols = (
        vif_step.keep_cols_
    )

    train_ranges = {

        col: (

            float(
                X[col].min()
            ),

            float(
                X[col].max()
            )
        )

        for col in kept_cols
    }

    # =========================================================================
    # MULTIVARIATE JOINT EXTRAPOLATION ENVELOPE
    # =========================================================================

    train_matrix = (
        X[
            kept_cols
        ]
        .to_numpy(
            dtype=float
        )
    )

    mahalanobis_mean = (
        train_matrix.mean(
            axis=0
        )
    )

    # Small ridge term for numerical stability.
    train_cov = np.cov(
        train_matrix,
        rowvar=False
    )

    train_cov = np.atleast_2d(
        train_cov
    )

    train_cov_reg = (
        train_cov
        + np.eye(
            train_cov.shape[0]
        ) * 1e-8
    )

    mahalanobis_cov_inv = (
        np.linalg.pinv(
            train_cov_reg
        )
    )

    diff = (
        train_matrix
        - mahalanobis_mean
    )

    train_mahalanobis_sq = (
        np.einsum(
            "ij,jk,ik->i",
            diff,
            mahalanobis_cov_inv,
            diff,
        )
    )

    train_mahalanobis = (
        np.sqrt(
            np.maximum(
                train_mahalanobis_sq,
                0.0
            )
        )
    )

    # 99th percentile of the training distribution.
    mahalanobis_threshold = float(
        np.percentile(
            train_mahalanobis,
            99
        )
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    save_path = "shm_model.pkl"

    bundle = {

        "model":
            best_estimator,

        "train_ranges":
            train_ranges,

        "feature_columns":
            list(
                X.columns
            ),

        "kept_columns":
            list(
                kept_cols
            ),

        "cv_mape":
            float(
                winner["CV_MAPE"]
            ),

        "cv_mape_std":
            float(
                winner["CV_MAPE_std"]
            ),

        "estimated_shm_score":
            float(
                winner[
                    "Estimated_SHM_Score"
                ]
            ),

        "feature_importances":
            feature_importances,

        "mahalanobis_mean":
            mahalanobis_mean,

        "mahalanobis_cov_inv":
            mahalanobis_cov_inv,

        "mahalanobis_threshold":
            mahalanobis_threshold,
    }

    joblib.dump(
        bundle,
        save_path
    )

    print(
        f"\nSaved model to:"
        f"\n{save_path}"
    )

    return (
        results,
        best_estimator,
        train_ranges,
    )


# ============================================================================
# PREDICTION WITH EXTRAPOLATION CHECK
# ============================================================================

def load_model_bundle(
    model_bundle_path
):

    """
    Load a saved model bundle once.

    Pass the returned bundle to
    predict_with_extrapolation_check()
    when scoring multiple batches.
    """

    return joblib.load(
        model_bundle_path
    )


def predict_with_extrapolation_check(
    model_bundle_path,
    X_new,
    bundle=None,
):

    if bundle is None:

        bundle = load_model_bundle(
            model_bundle_path
        )

    model = bundle[
        "model"
    ]

    train_ranges = bundle[
        "train_ranges"
    ]

    X_new = pd.DataFrame(
        X_new
    )

    warnings = []

    # ------------------------------------------------------------------------
    # Marginal extrapolation check.
    # ------------------------------------------------------------------------

    for col, (
        lo,
        hi
    ) in train_ranges.items():

        if col not in X_new.columns:

            continue

        out_of_range = (

            (
                X_new[col]
                < lo
            )

            |

            (
                X_new[col]
                > hi
            )
        )

        if out_of_range.any():

            n = int(
                out_of_range.sum()
            )

            warnings.append(

                f"[EXTRAPOLATION WARNING] "
                f"{n} sample(s) have "
                f"'{col}' outside "
                f"training range "
                f"[{lo:.4g}, {hi:.4g}]."
            )

    # ------------------------------------------------------------------------
    # Joint Mahalanobis extrapolation check.
    # ------------------------------------------------------------------------

    has_joint_check = (

        "mahalanobis_mean"
        in bundle

        and

        "mahalanobis_cov_inv"
        in bundle

        and

        "mahalanobis_threshold"
        in bundle
    )

    if has_joint_check:

        kept_cols = bundle[
            "kept_columns"
        ]

        missing_cols = [

            col

            for col in kept_cols

            if col not in X_new.columns
        ]

        if missing_cols:

            warnings.append(

                "[EXTRAPOLATION CHECK SKIPPED] "
                "Cannot compute joint "
                "(Mahalanobis) extrapolation check "
                "-- missing columns: "
                f"{missing_cols}"
            )

        else:

            new_matrix = (
                X_new[
                    kept_cols
                ]
                .to_numpy(
                    dtype=float
                )
            )

            diff = (
                new_matrix
                - bundle[
                    "mahalanobis_mean"
                ]
            )

            m_dist_sq = (
                np.einsum(
                    "ij,jk,ik->i",
                    diff,
                    bundle[
                        "mahalanobis_cov_inv"
                    ],
                    diff,
                )
            )

            m_dist = (
                np.sqrt(
                    np.maximum(
                        m_dist_sq,
                        0.0
                    )
                )
            )

            threshold = bundle[
                "mahalanobis_threshold"
            ]

            out_of_envelope = (
                m_dist
                > threshold
            )

            if out_of_envelope.any():

                n = int(
                    out_of_envelope.sum()
                )

                warnings.append(

                    "[EXTRAPOLATION WARNING] "
                    f"{n} sample(s) have a "
                    "feature COMBINATION far from "
                    "anything seen in training "
                    f"(Mahalanobis distance > "
                    f"{threshold:.3g}), even though "
                    "each feature may individually "
                    "be in-range."
                )

    # ------------------------------------------------------------------------
    # Print warnings.
    # ------------------------------------------------------------------------

    for warning in warnings:

        print(
            warning
        )

    # ------------------------------------------------------------------------
    # Predict.
    # ------------------------------------------------------------------------

    predictions = (
        model.predict(
            X_new
        )
    )

    return (
        predictions,
        warnings,
    )


# ============================================================================
# TEE LOGGER
# ============================================================================

class _Tee:

    """
    Duplicate console output into a log file.
    """

    def __init__(
        self,
        *streams
    ):

        self._streams = streams

    def write(
        self,
        data
    ):

        for stream in self._streams:

            stream.write(
                data
            )

    def flush(
        self
    ):

        for stream in self._streams:

            stream.flush()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    import sys
    import datetime

    _log_filename = (

        "model_eval_"

        +

        datetime.datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )

        +

        ".log"
    )

    with open(
        _log_filename,
        "w"
    ) as _log_file:

        sys.stdout = _Tee(
            sys.stdout,
            _log_file
        )

        try:

            print(
                f"Logging this run's output to: "
                f"{_log_filename}\n"
            )

            # ----------------------------------------------------------------
            # Load feature matrix and target.
            # ----------------------------------------------------------------

            X_raw, y = (
                load_training_data()
            )

            # ----------------------------------------------------------------
            # Single evaluation run.
            # ----------------------------------------------------------------

            print("\n")
            print(
                "=" * 80
            )

            print(
                f"EVALUATION RANDOM STATE: "
                f"{RANDOM_STATE}"
            )

            print(
                "=" * 80
            )

            results, best_estimator, train_ranges = (
                compare_models(

                    X_raw,

                    y,

                    random_state=
                        RANDOM_STATE
                )
            )

        finally:

            sys.stdout = (
                sys.stdout
                ._streams[0]
            )