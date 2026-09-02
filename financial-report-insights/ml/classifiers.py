"""
Financial distress classification models for Phase 3.2.

Uses scikit-learn only (no xgboost/shap/lightgbm) and builds features inline
without depending on ml/feature_engineering.py.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning *default* when the denominator is zero or
    either operand is NaN/inf."""
    try:
        num = float(numerator) if numerator is not None else 0.0
        den = float(denominator) if denominator is not None else 0.0
    except (TypeError, ValueError):
        return default
    if den == 0.0 or not math.isfinite(den) or not math.isfinite(num):
        return default
    result = num / den
    return result if math.isfinite(result) else default


# ---------------------------------------------------------------------------
# FinancialDistressClassifier
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS = ("logistic_regression", "random_forest", "gradient_boosting")


class FinancialDistressClassifier:
    """Binary classifier that predicts financial distress (1) vs healthy (0).

    Pseudo-labels are derived from a simplified Altman Z-Score:
        Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA)
            + 0.6*(Equity/TL) + 1.0*(Revenue/TA)
    Z < 1.81  =>  distressed (1)
    Z >= 1.81 =>  healthy    (0)
    """

    _CV_FOLDS = 5

    def __init__(self, model_type: str = "random_forest") -> None:
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {_SUPPORTED_MODELS}, got {model_type!r}")
        self.model_type = model_type
        self._estimator = self._create_model(model_type)
        self._scaler = StandardScaler()
        self.model_: Any = None  # fitted estimator after train()
        self._fitted_scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # Internal factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_model(model_type: str) -> Any:
        """Return an unfitted sklearn estimator for the requested model type."""
        if model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        if model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        if model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1,
            )
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # ------------------------------------------------------------------
    # Pseudo-label generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_pseudo_labels(financial_data_list: list[dict]) -> list[int]:
        """Compute simplified Altman Z-Scores and return distress labels.

        Parameters
        ----------
        financial_data_list:
            List of dicts with (a subset of) financial fields.  Missing or
            non-numeric values are treated as 0.

        Returns
        -------
        list[int]
            1 for distressed (Z < 1.81), 0 for healthy (Z >= 1.81).
        """
        labels: list[int] = []
        for record in financial_data_list:

            def _get(key: str) -> float:
                val = record.get(key, 0) or 0
                try:
                    v = float(val)
                    return v if math.isfinite(v) else 0.0
                except (TypeError, ValueError):
                    return 0.0

            total_assets = _get("total_assets")
            total_liabilities = _get("total_liabilities")
            working_capital = _get("current_assets") - _get("current_liabilities")
            retained_earnings = _get("retained_earnings")
            ebit = _get("ebit") or (_get("operating_income"))
            equity = _get("total_equity")
            revenue = _get("revenue")

            ta = total_assets
            tl = total_liabilities

            z = (
                1.2 * _safe_ratio(working_capital, ta)
                + 1.4 * _safe_ratio(retained_earnings, ta)
                + 3.3 * _safe_ratio(ebit, ta)
                + 0.6 * _safe_ratio(equity, tl)
                + 1.0 * _safe_ratio(revenue, ta)
            )

            labels.append(1 if z < 1.81 else 0)

        return labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features: list[list[float]], labels: list[int]) -> dict[str, float]:
        """Fit the model using 5-fold stratified cross-validation.

        Parameters
        ----------
        features:
            2-D list of shape (n_samples, n_features).
        labels:
            Binary target list of length n_samples.

        Returns
        -------
        dict
            Cross-validated metrics: accuracy, precision, recall, f1, roc_auc.
            The final model (fitted on all data) is stored as ``self.model_``.
        """
        X = np.array(features, dtype=float)
        y = np.array(labels, dtype=int)

        if len(X) == 0 or len(y) == 0:
            raise ValueError("features and labels must be non-empty")
        if X.shape[0] != len(y):
            raise ValueError("features and labels must have the same length")

        # Determine safe fold count (cannot exceed number of samples or minority
        # class size)
        n_folds = min(self._CV_FOLDS, len(y), int(np.bincount(y).min()))
        n_folds = max(n_folds, 2)  # cross_validate requires at least 2

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        estimator = self._create_model(self.model_type)

        scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        cv_results = cross_validate(
            estimator,
            X_scaled,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )

        metrics = {
            "accuracy": float(cv_results["test_accuracy"].mean()),
            "precision": float(cv_results["test_precision_weighted"].mean()),
            "recall": float(cv_results["test_recall_weighted"].mean()),
            "f1": float(cv_results["test_f1_weighted"].mean()),
            "roc_auc": float(cv_results["test_roc_auc"].mean()),
        }

        # Re-fit on all data so the model is ready for inference
        final_estimator = self._create_model(self.model_type)
        final_estimator.fit(X_scaled, y)
        self.model_ = final_estimator
        self._fitted_scaler = scaler

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    def _scale(self, features: list[list[float]]) -> np.ndarray:
        X = np.array(features, dtype=float)
        if self._fitted_scaler is not None:
            return self._fitted_scaler.transform(X)
        return X

    def predict(self, features: list[list[float]]) -> list[int]:
        """Return predicted binary class labels."""
        self._require_fitted()
        X_scaled = self._scale(features)
        return self.model_.predict(X_scaled).tolist()

    def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        """Return class-probability estimates [[p_healthy, p_distressed], ...]."""
        self._require_fitted()
        X_scaled = self._scale(features)
        return self.model_.predict_proba(X_scaled).tolist()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """Return feature importances sorted descending.

        For tree ensembles uses ``.feature_importances_``.
        For logistic regression uses absolute coefficient values.

        Parameters
        ----------
        feature_names:
            Names corresponding to columns used during training.

        Returns
        -------
        list of (name, importance) sorted by importance descending.
        """
        self._require_fitted()

        if self.model_type == "logistic_regression":
            importances = np.abs(self.model_.coef_[0])
        else:
            importances = self.model_.feature_importances_

        if len(feature_names) != len(importances):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"number of model features ({len(importances)})"
            )

        pairs = list(zip(feature_names, importances.tolist()))
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs


# ---------------------------------------------------------------------------
# DistressExplainer
# ---------------------------------------------------------------------------

_DIRECTION_DESCRIPTIONS: dict[str, str] = {
    "total_assets": "Total assets — larger balance sheets reduce distress risk",
    "total_liabilities": "Total liabilities — high debt load elevates distress risk",
    "current_assets": "Current assets — short-term liquidity buffer",
    "current_liabilities": "Current liabilities — near-term cash obligations",
    "retained_earnings": "Retained earnings — accumulated profitability cushion",
    "ebit": "EBIT — core operating profitability",
    "operating_income": "Operating income — earnings before interest and taxes",
    "total_equity": "Total equity — book-value solvency indicator",
    "revenue": "Revenue — top-line scale and growth signal",
    "net_income": "Net income — bottom-line profitability",
    "cash": "Cash — most liquid distress buffer",
    "inventory": "Inventory — less liquid current asset",
    "accounts_receivable": "Accounts receivable — pending cash inflows",
    "accounts_payable": "Accounts payable — short-term supplier obligations",
    "total_debt": "Total debt — interest-bearing obligations",
    "gross_profit": "Gross profit — product/service margin",
    "operating_expenses": "Operating expenses — cost structure burden",
    "working_capital": "Working capital — net short-term liquidity",
}

_DEFAULT_DESCRIPTION = "Financial metric influencing distress classification"


class DistressExplainer:
    """Lightweight SHAP-like explainer for financial distress predictions.

    Does NOT require the ``shap`` library.  Instead it uses a feature-value
    weighted importance approach that is efficient for a single sample:

        contribution_i = feature_importance_i * abs(feature_value_i)

    The sign of the contribution (direction) is determined by the feature
    value relative to zero — positive values of high-importance features
    that are associated with risk markers increase risk, while features that
    represent buffers (e.g. equity, cash) decrease risk.

    This is an approximation; for production use, consider adding shap as a
    dependency and replacing this logic.
    """

    # Features whose higher values reduce distress risk
    _PROTECTIVE_FEATURES = {
        "cash",
        "total_equity",
        "retained_earnings",
        "current_assets",
        "gross_profit",
        "net_income",
        "ebit",
        "operating_income",
        "revenue",
        "working_capital",
        "accounts_receivable",
    }

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names

        # Pre-extract importances once
        if hasattr(model, "feature_importances_"):
            self._importances = np.array(model.feature_importances_)
        elif hasattr(model, "coef_"):
            self._importances = np.abs(model.coef_[0])
        else:
            # Fall back to uniform importances
            self._importances = np.ones(len(feature_names)) / max(len(feature_names), 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_prediction(self, features: list[float]) -> list[tuple[str, float, str]]:
        """Explain a single prediction via weighted feature contributions.

        Parameters
        ----------
        features:
            Feature vector (same order as ``feature_names``).

        Returns
        -------
        list of (feature_name, contribution, direction)
        where direction is "increases_risk" or "decreases_risk".
        """
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"features length ({len(features)}) does not match feature_names length ({len(self.feature_names)})"
            )

        x = np.array(features, dtype=float)
        # Contribution = importance * |value|  (magnitude)
        contributions = self._importances * np.abs(x)

        result: list[tuple[str, float, str]] = []
        for name, contrib, value in zip(self.feature_names, contributions, x):
            direction = self._direction(name, value)
            result.append((name, float(contrib), direction))

        return result

    def get_top_risk_factors(self, features: list[float], k: int = 5) -> list[dict]:
        """Return the top-k risk factors for a single prediction.

        Parameters
        ----------
        features:
            Feature vector for a single sample.
        k:
            Number of factors to return (clamped to available features).

        Returns
        -------
        list of dicts with keys: feature_name, value, contribution,
        direction, description.
        """
        explanations = self.explain_prediction(features)
        # Sort by contribution magnitude descending
        sorted_explanations = sorted(explanations, key=lambda t: t[1], reverse=True)
        top_k = sorted_explanations[: max(1, min(k, len(sorted_explanations)))]

        result: list[dict] = []
        for name, contrib, direction in top_k:
            idx = self.feature_names.index(name)
            value = features[idx]
            description = _DIRECTION_DESCRIPTIONS.get(name, _DEFAULT_DESCRIPTION)
            result.append(
                {
                    "feature_name": name,
                    "value": float(value),
                    "contribution": float(contrib),
                    "direction": direction,
                    "description": description,
                }
            )

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _direction(self, feature_name: str, value: float) -> str:
        """Determine if the feature pushes toward or away from distress."""
        is_protective = feature_name in self._PROTECTIVE_FEATURES
        positive_value = value > 0

        if is_protective:
            # High value of a protective feature reduces risk
            return "decreases_risk" if positive_value else "increases_risk"
        else:
            # High value of a risk feature increases risk
            return "increases_risk" if positive_value else "decreases_risk"
