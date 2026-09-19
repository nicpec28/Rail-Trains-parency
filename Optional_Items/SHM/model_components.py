import numpy as np
import pandas as pd

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    clone,
)


# ============================================================================
# VIF SELECTOR
# ============================================================================

class VIFSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        threshold=10.0,
        protected_features=None,
        protected_prefixes=None,
    ):
        self.threshold = threshold

        # Columns matching either of these are never removed by the
        # greedy VIF elimination loop below, even if their VIF exceeds
        # `threshold`. This exists for two reasons:
        #
        # 1. Compositional features (e.g. a set of histogram-bin
        #    fractions that sum to ~1 by construction) are, by
        #    construction, close to linearly dependent. Their
        #    correlation matrix is then near-singular, so the VIFs
        #    computed via np.linalg.pinv below no longer have the
        #    textbook "1 / (1 - R^2)" interpretation -- greedy removal
        #    driven by those numbers can strip out most or all of a
        #    physically meaningful feature group (e.g. a load-spectrum
        #    histogram) rather than genuinely redundant scalar features.
        # 2. Features referenced by a monotonic-constrained model
        #    (see MonotonicAwareRegressor) must survive VIF selection on
        #    every CV fold, or the enforced constraint set silently
        #    changes fold to fold, undermining the point of using a
        #    monotonic model in the first place.
        self.protected_features = protected_features
        self.protected_prefixes = protected_prefixes

    def fit(self, X, y=None):

        X_df = self._to_dataframe(X)

        self.feature_names_in_ = np.asarray(
            X_df.columns,
            dtype=object
        )

        protected_features = set(
            self.protected_features or []
        )

        protected_prefixes = tuple(
            self.protected_prefixes or ()
        )

        def _is_protected(col):
            return (
                col in protected_features
                or str(col).startswith(protected_prefixes)
            )

        self.protected_cols_ = [
            col
            for col in X_df.columns
            if _is_protected(col)
        ]

        # Remove constant columns. Uses both an absolute floor (catches
        # exact/near-exact constants regardless of scale) and a
        # relative one (variance relative to the column's own
        # magnitude), since an absolute-only threshold can miss a
        # constant column whose values sit at a small scale (e.g.
        # ~1e-8) or over-flag a large-magnitude column with tiny but
        # nonzero floating-point variance.
        variances = X_df.var(
            axis=0,
            ddof=0
        )

        col_scale = np.maximum(
            X_df.abs().mean(axis=0),
            1.0
        )

        relative_variance = variances / (col_scale ** 2)

        self.constant_cols_ = list(
            variances[
                (variances <= 1e-15)
                | (relative_variance <= 1e-12)
            ].index
        )

        working = X_df.drop(
            columns=self.constant_cols_,
            errors="ignore"
        )

        if working.shape[1] == 0:
            raise ValueError(
                "VIFSelector: no non-constant features remain."
            )

        kept = list(working.columns)

        # --------------------------------------------------------------------
        # Iteratively remove highest-VIF feature (skipping protected ones).
        # --------------------------------------------------------------------

        while len(kept) > 1:

            current = working[kept]

            corr = current.corr().to_numpy(
                dtype=float
            )

            corr = np.nan_to_num(
                corr,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            try:
                inv_corr = np.linalg.pinv(corr)
                vifs = np.diag(inv_corr)

            except np.linalg.LinAlgError:
                break

            vifs = np.asarray(
                vifs,
                dtype=float
            )

            vifs = np.maximum(
                vifs,
                1.0
            )

            # Correlations still account for protected columns (they
            # stay in `current`/`corr` above), but protected columns
            # themselves are never candidates for removal.
            removable_idx = [
                i
                for i, col in enumerate(kept)
                if col not in self.protected_cols_
            ]

            if not removable_idx:
                break

            local_max = int(
                np.argmax(vifs[removable_idx])
            )

            max_idx = removable_idx[local_max]

            max_vif = float(
                vifs[max_idx]
            )

            max_feature = kept[max_idx]

            if max_vif <= self.threshold:
                break

            print(
                f"VIFSelector: dropping "
                f"'{max_feature}' "
                f"(VIF={max_vif:.2f})"
            )

            kept.pop(max_idx)

        self.keep_cols_ = kept

        # --------------------------------------------------------------------
        # Final VIF values.
        # --------------------------------------------------------------------

        self.vif_values_ = {}

        if len(self.keep_cols_) > 1:

            final_corr = (
                working[self.keep_cols_]
                .corr()
                .to_numpy(dtype=float)
            )

            final_corr = np.nan_to_num(
                final_corr,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            try:

                final_inv = np.linalg.pinv(
                    final_corr
                )

                final_vifs = np.maximum(
                    np.diag(final_inv),
                    1.0
                )

                self.vif_values_ = dict(
                    zip(
                        self.keep_cols_,
                        final_vifs
                    )
                )

            except np.linalg.LinAlgError:
                pass

        return self

    def transform(self, X):

        X_df = self._to_dataframe(X)

        missing = [
            col
            for col in self.keep_cols_
            if col not in X_df.columns
        ]

        if missing:
            raise ValueError(
                "VIFSelector: missing columns: "
                f"{missing}"
            )

        return X_df[
            self.keep_cols_
        ].copy()

    def get_feature_names_out(
        self,
        input_features=None
    ):

        return np.asarray(
            self.keep_cols_,
            dtype=object
        )

    @staticmethod
    def _to_dataframe(X):

        if isinstance(X, pd.DataFrame):
            return X.copy()

        return pd.DataFrame(X)


# ============================================================================
# MONOTONIC-AWARE REGRESSOR
# ============================================================================

class MonotonicAwareRegressor(
    BaseEstimator,
    RegressorMixin
):

    def __init__(
        self,
        model,
        increasing_features=None,
        decreasing_features=None,
    ):

        self.model = model

        self.increasing_features = (
            []
            if increasing_features is None
            else increasing_features
        )

        self.decreasing_features = (
            []
            if decreasing_features is None
            else decreasing_features
        )

    def fit(self, X, y):

        # Preserve feature names after VIF + scaling.
        if isinstance(X, pd.DataFrame):

            feature_names = list(
                X.columns
            )

        else:

            feature_names = [
                f"feature_{i}"
                for i in range(X.shape[1])
            ]

        self.feature_names_in_ = np.asarray(
            feature_names,
            dtype=object
        )

        # --------------------------------------------------------------------
        # Construct constraint vector.
        #
        # +1 = increasing
        #  0 = unconstrained
        # -1 = decreasing
        # --------------------------------------------------------------------

        constraints = []

        for feature in feature_names:

            if feature in self.increasing_features:
                constraints.append(1)

            elif feature in self.decreasing_features:
                constraints.append(-1)

            else:
                constraints.append(0)

        self.monotonic_constraints_ = constraints

        # --------------------------------------------------------------------
        # Clone underlying estimator.
        # --------------------------------------------------------------------

        self.model_ = clone(
            self.model
        )

        model_name = type(
            self.model_
        ).__name__

        # --------------------------------------------------------------------
        # Detect which monotonic-constraint parameter this estimator
        # supports by introspecting get_params(), rather than matching
        # on the class name. A string match on __name__ silently breaks
        # for any subclass, a renamed class in a future library
        # version, or a differently-imported alias -- it fails loudly
        # here (via the same TypeError as before) instead of routing to
        # the wrong branch or matching nothing unexpectedly.
        # --------------------------------------------------------------------

        if not hasattr(self.model_, "get_params"):

            raise TypeError(
                "MonotonicAwareRegressor requires an estimator "
                "exposing get_params(). "
                f"Received {model_name}"
            )

        supported_params = self.model_.get_params(deep=False)

        if "monotone_constraints" in supported_params:

            # XGBoost-style.
            self.model_.set_params(
                monotone_constraints=tuple(
                    constraints
                )
            )

        elif "monotonic_cst" in supported_params:

            # scikit-learn HistGradientBoosting-style.
            self.model_.set_params(
                monotonic_cst=constraints
            )

        else:

            raise TypeError(
                "MonotonicAwareRegressor supports estimators that "
                "expose a 'monotone_constraints' parameter "
                "(XGBoost-style) or a 'monotonic_cst' parameter "
                "(scikit-learn HistGradientBoosting-style). "
                f"Received {model_name}, whose get_params() exposes "
                f"neither: {sorted(supported_params)}"
            )

        self.model_.fit(
            X,
            y
        )

        return self

    def predict(self, X):

        return self.model_.predict(X)

    def score(self, X, y):

        return self.model_.score(
            X,
            y
        )

    def get_feature_names_out(
        self,
        input_features=None
    ):

        if hasattr(
            self,
            "feature_names_in_"
        ):
            return self.feature_names_in_

        if input_features is not None:
            return np.asarray(
                input_features,
                dtype=object
            )

        return None

    # ------------------------------------------------------------------------
    # SKLEARN PARAMETER HANDLING
    # ------------------------------------------------------------------------

    def get_params(
        self,
        deep=True
    ):

        params = {
            "model": self.model,
            "increasing_features":
                self.increasing_features,
            "decreasing_features":
                self.decreasing_features,
        }

        if deep and hasattr(
            self.model,
            "get_params"
        ):

            for key, value in (
                self.model
                .get_params(deep=True)
                .items()
            ):

                params[
                    f"model__{key}"
                ] = value

        return params

    def set_params(
        self,
        **params
    ):

        if "model" in params:

            self.model = params.pop(
                "model"
            )

        if "increasing_features" in params:

            self.increasing_features = (
                params.pop(
                    "increasing_features"
                )
            )

        if "decreasing_features" in params:

            self.decreasing_features = (
                params.pop(
                    "decreasing_features"
                )
            )

        model_params = {}

        for key, value in params.items():

            if key.startswith(
                "model__"
            ):

                model_params[
                    key[len("model__"):]
                ] = value

            else:

                raise ValueError(
                    f"Invalid parameter "
                    f"'{key}' for "
                    f"{self.__class__.__name__}"
                )

        if model_params:

            self.model.set_params(
                **model_params
            )

        return self