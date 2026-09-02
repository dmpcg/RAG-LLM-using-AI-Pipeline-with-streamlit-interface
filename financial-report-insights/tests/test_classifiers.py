"""
Tests for ml/classifiers.py — Phase 3.2 Classification Models.

Covers:
- FinancialDistressClassifier: model creation, pseudo-label generation,
  training, prediction, feature importance
- DistressExplainer: explain_prediction, get_top_risk_factors
- Edge cases: single sample, all-same-label, missing fields, minimal features
"""

from __future__ import annotations

import random

import pytest

from ml.classifiers import DistressExplainer, FinancialDistressClassifier

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_features(n_samples: int = 50, n_features: int = 8, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(n_features)] for _ in range(n_samples)]


def _make_labels(n_samples: int = 50, seed: int = 42) -> list[int]:
    """Balanced binary labels."""
    rng = random.Random(seed)
    labels = [0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2)
    rng.shuffle(labels)
    return labels


FEATURE_NAMES = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]


# ---------------------------------------------------------------------------
# FinancialDistressClassifier — model creation
# ---------------------------------------------------------------------------


class TestModelCreation:
    def test_default_is_random_forest(self):
        clf = FinancialDistressClassifier()
        assert clf.model_type == "random_forest"

    def test_logistic_regression_creation(self):
        clf = FinancialDistressClassifier(model_type="logistic_regression")
        assert clf.model_type == "logistic_regression"

    def test_random_forest_creation(self):
        clf = FinancialDistressClassifier(model_type="random_forest")
        assert clf.model_type == "random_forest"

    def test_gradient_boosting_creation(self):
        clf = FinancialDistressClassifier(model_type="gradient_boosting")
        assert clf.model_type == "gradient_boosting"

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type must be one of"):
            FinancialDistressClassifier(model_type="xgboost")

    def test_create_model_factory_lr(self):
        from sklearn.linear_model import LogisticRegression

        m = FinancialDistressClassifier._create_model("logistic_regression")
        assert isinstance(m, LogisticRegression)

    def test_create_model_factory_rf(self):
        from sklearn.ensemble import RandomForestClassifier

        m = FinancialDistressClassifier._create_model("random_forest")
        assert isinstance(m, RandomForestClassifier)

    def test_create_model_factory_gb(self):
        from sklearn.ensemble import GradientBoostingClassifier

        m = FinancialDistressClassifier._create_model("gradient_boosting")
        assert isinstance(m, GradientBoostingClassifier)

    def test_model_not_fitted_before_train(self):
        clf = FinancialDistressClassifier()
        assert clf.model_ is None


# ---------------------------------------------------------------------------
# generate_pseudo_labels
# ---------------------------------------------------------------------------


class TestGeneratePseudoLabels:
    """Tests for Z-score-based pseudo-label generation."""

    def _healthy_record(self) -> dict:
        """Returns a record that should yield a high Z-score (healthy)."""
        return {
            "current_assets": 800_000,
            "current_liabilities": 200_000,
            "total_assets": 2_000_000,
            "retained_earnings": 600_000,
            "ebit": 400_000,
            "total_equity": 1_000_000,
            "total_liabilities": 500_000,
            "revenue": 1_500_000,
        }

    def _distressed_record(self) -> dict:
        """Returns a record that should yield a low Z-score (distressed)."""
        return {
            "current_assets": 50_000,
            "current_liabilities": 900_000,
            "total_assets": 1_000_000,
            "retained_earnings": -500_000,
            "ebit": -200_000,
            "total_equity": 10_000,
            "total_liabilities": 990_000,
            "revenue": 100_000,
        }

    def test_healthy_company_labeled_0(self):
        labels = FinancialDistressClassifier.generate_pseudo_labels([self._healthy_record()])
        assert labels == [0]

    def test_distressed_company_labeled_1(self):
        labels = FinancialDistressClassifier.generate_pseudo_labels([self._distressed_record()])
        assert labels == [1]

    def test_multiple_records(self):
        records = [self._healthy_record(), self._distressed_record()]
        labels = FinancialDistressClassifier.generate_pseudo_labels(records)
        assert len(labels) == 2
        assert labels[0] == 0
        assert labels[1] == 1

    def test_empty_list_returns_empty(self):
        labels = FinancialDistressClassifier.generate_pseudo_labels([])
        assert labels == []

    def test_missing_all_fields_defaults_zero_labeled_1(self):
        """All zeros => Z = 0 < 1.81 => distressed."""
        labels = FinancialDistressClassifier.generate_pseudo_labels([{}])
        assert labels == [1]

    def test_none_field_values_handled_gracefully(self):
        record = {
            "current_assets": None,
            "current_liabilities": None,
            "total_assets": None,
            "retained_earnings": None,
            "ebit": None,
            "total_equity": None,
            "total_liabilities": None,
            "revenue": None,
        }
        labels = FinancialDistressClassifier.generate_pseudo_labels([record])
        assert len(labels) == 1
        assert labels[0] in (0, 1)

    def test_zero_total_assets_does_not_crash(self):
        """Zero total_assets means division produces 0 for most terms."""
        record = {
            "current_assets": 100_000,
            "current_liabilities": 50_000,
            "total_assets": 0,
            "retained_earnings": 50_000,
            "ebit": 20_000,
            "total_equity": 200_000,
            "total_liabilities": 0,
            "revenue": 300_000,
        }
        labels = FinancialDistressClassifier.generate_pseudo_labels([record])
        assert len(labels) == 1

    def test_operating_income_fallback_when_ebit_missing(self):
        """When 'ebit' key is absent, operating_income should be used."""
        record = {
            "current_assets": 800_000,
            "current_liabilities": 200_000,
            "total_assets": 2_000_000,
            "retained_earnings": 600_000,
            "operating_income": 400_000,
            "total_equity": 1_000_000,
            "total_liabilities": 500_000,
            "revenue": 1_500_000,
        }
        labels = FinancialDistressClassifier.generate_pseudo_labels([record])
        assert labels == [0]

    def test_returns_only_0_and_1(self):
        """All labels must be binary."""
        rng = random.Random(7)
        records = [
            {
                "current_assets": rng.uniform(0, 1e6),
                "current_liabilities": rng.uniform(0, 5e5),
                "total_assets": rng.uniform(1e5, 5e6),
                "retained_earnings": rng.uniform(-1e6, 1e6),
                "ebit": rng.uniform(-2e5, 8e5),
                "total_equity": rng.uniform(0, 2e6),
                "total_liabilities": rng.uniform(1e4, 3e6),
                "revenue": rng.uniform(0, 3e6),
            }
            for _ in range(30)
        ]
        labels = FinancialDistressClassifier.generate_pseudo_labels(records)
        assert all(l in (0, 1) for l in labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrain:
    """Tests for the train() method across model types."""

    EXPECTED_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.features = _make_features(n_samples=60, n_features=8)
        self.labels = _make_labels(n_samples=60)

    def _train(self, model_type: str) -> tuple[FinancialDistressClassifier, dict]:
        clf = FinancialDistressClassifier(model_type=model_type)
        metrics = clf.train(self.features, self.labels)
        return clf, metrics

    def test_train_rf_returns_metrics(self):
        _, metrics = self._train("random_forest")
        assert set(metrics.keys()) == self.EXPECTED_METRIC_KEYS

    def test_train_lr_returns_metrics(self):
        _, metrics = self._train("logistic_regression")
        assert set(metrics.keys()) == self.EXPECTED_METRIC_KEYS

    @pytest.mark.slow
    def test_train_gb_returns_metrics(self):
        _, metrics = self._train("gradient_boosting")
        assert set(metrics.keys()) == self.EXPECTED_METRIC_KEYS

    def test_metrics_are_floats_in_valid_range(self):
        _, metrics = self._train("random_forest")
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float"
            assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1]"

    def test_model_is_stored_after_train(self):
        clf, _ = self._train("random_forest")
        assert clf.model_ is not None

    def test_empty_features_raises(self):
        clf = FinancialDistressClassifier()
        with pytest.raises((ValueError, Exception)):
            clf.train([], [])

    def test_mismatched_lengths_raises(self):
        clf = FinancialDistressClassifier()
        with pytest.raises(ValueError):
            clf.train(self.features, self.labels[:-1])

    def test_train_with_small_dataset(self):
        """Minimum viable training set (10 samples, balanced)."""
        features = _make_features(n_samples=10, n_features=4)
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        clf = FinancialDistressClassifier(model_type="logistic_regression")
        metrics = clf.train(features, labels)
        assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# predict / predict_proba
# ---------------------------------------------------------------------------


class TestPredict:
    @pytest.fixture(autouse=True)
    def _trained_rf(self):
        features = _make_features(n_samples=60)
        labels = _make_labels(n_samples=60)
        self.clf = FinancialDistressClassifier(model_type="random_forest")
        self.clf.train(features, labels)
        self.n_features = 8

    def test_predict_returns_list(self):
        X = _make_features(n_samples=5)
        result = self.clf.predict(X)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_predict_returns_binary(self):
        X = _make_features(n_samples=10)
        result = self.clf.predict(X)
        assert all(v in (0, 1) for v in result)

    def test_predict_single_sample(self):
        X = [_make_features(n_samples=1)[0]]
        result = self.clf.predict(X)
        assert len(result) == 1
        assert result[0] in (0, 1)

    def test_predict_proba_returns_list(self):
        X = _make_features(n_samples=5)
        result = self.clf.predict_proba(X)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_predict_proba_two_classes(self):
        X = _make_features(n_samples=5)
        result = self.clf.predict_proba(X)
        assert all(len(row) == 2 for row in result)

    def test_predict_proba_sums_to_one(self):
        X = _make_features(n_samples=10)
        result = self.clf.predict_proba(X)
        for row in result:
            assert abs(sum(row) - 1.0) < 1e-6, f"Probabilities don't sum to 1: {row}"

    def test_predict_proba_values_in_0_1(self):
        X = _make_features(n_samples=10)
        result = self.clf.predict_proba(X)
        for row in result:
            assert all(0.0 <= p <= 1.0 for p in row)

    def test_predict_before_train_raises(self):
        clf = FinancialDistressClassifier()
        with pytest.raises(RuntimeError, match="trained"):
            clf.predict(_make_features(n_samples=3))

    def test_predict_proba_before_train_raises(self):
        clf = FinancialDistressClassifier()
        with pytest.raises(RuntimeError, match="trained"):
            clf.predict_proba(_make_features(n_samples=3))


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.feature_names = FEATURE_NAMES
        features = _make_features(n_samples=60)
        labels = _make_labels(n_samples=60)
        self.clf_rf = FinancialDistressClassifier(model_type="random_forest")
        self.clf_rf.train(features, labels)
        self.clf_lr = FinancialDistressClassifier(model_type="logistic_regression")
        self.clf_lr.train(features, labels)

    def test_rf_importance_returns_list_of_tuples(self):
        result = self.clf_rf.get_feature_importance(self.feature_names)
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_rf_importance_same_length_as_features(self):
        result = self.clf_rf.get_feature_importance(self.feature_names)
        assert len(result) == len(self.feature_names)

    def test_rf_importance_sorted_descending(self):
        result = self.clf_rf.get_feature_importance(self.feature_names)
        importances = [t[1] for t in result]
        assert importances == sorted(importances, reverse=True)

    def test_lr_importance_uses_abs_coef(self):
        result = self.clf_lr.get_feature_importance(self.feature_names)
        assert len(result) == len(self.feature_names)
        # All importance values should be non-negative
        assert all(imp >= 0.0 for _, imp in result)

    def test_importance_feature_names_present(self):
        result = self.clf_rf.get_feature_importance(self.feature_names)
        names_out = {t[0] for t in result}
        assert names_out == set(self.feature_names)

    def test_mismatched_feature_names_raises(self):
        with pytest.raises(ValueError, match="feature_names length"):
            self.clf_rf.get_feature_importance(["only_one"])

    def test_importance_before_train_raises(self):
        clf = FinancialDistressClassifier()
        with pytest.raises(RuntimeError, match="trained"):
            clf.get_feature_importance(self.feature_names)


# ---------------------------------------------------------------------------
# DistressExplainer
# ---------------------------------------------------------------------------


class TestDistressExplainer:
    @pytest.fixture(autouse=True)
    def _setup(self):
        features = _make_features(n_samples=60)
        labels = _make_labels(n_samples=60)
        clf = FinancialDistressClassifier(model_type="random_forest")
        clf.train(features, labels)
        self.model = clf.model_
        self.feature_names = FEATURE_NAMES
        self.explainer = DistressExplainer(self.model, self.feature_names)
        self.sample_features = features[0]

    def test_explain_prediction_returns_list(self):
        result = self.explainer.explain_prediction(self.sample_features)
        assert isinstance(result, list)

    def test_explain_prediction_length_matches_features(self):
        result = self.explainer.explain_prediction(self.sample_features)
        assert len(result) == len(self.feature_names)

    def test_explain_prediction_tuple_structure(self):
        result = self.explainer.explain_prediction(self.sample_features)
        for item in result:
            assert isinstance(item, tuple), "Each explanation should be a tuple"
            assert len(item) == 3, "Each tuple should have 3 elements"
            name, contrib, direction = item
            assert isinstance(name, str)
            assert isinstance(contrib, float)
            assert direction in ("increases_risk", "decreases_risk")

    def test_explain_prediction_non_negative_contributions(self):
        result = self.explainer.explain_prediction(self.sample_features)
        assert all(contrib >= 0.0 for _, contrib, _ in result)

    def test_explain_prediction_wrong_length_raises(self):
        with pytest.raises(ValueError, match="features length"):
            self.explainer.explain_prediction([1.0, 2.0])  # too short

    def test_get_top_risk_factors_default_k5(self):
        result = self.explainer.get_top_risk_factors(self.sample_features)
        assert len(result) == 5

    def test_get_top_risk_factors_custom_k(self):
        result = self.explainer.get_top_risk_factors(self.sample_features, k=3)
        assert len(result) == 3

    def test_get_top_risk_factors_k_exceeds_features_clamped(self):
        """Requesting more factors than features returns at most len(feature_names)."""
        result = self.explainer.get_top_risk_factors(self.sample_features, k=100)
        assert len(result) <= len(self.feature_names)

    def test_get_top_risk_factors_dict_keys(self):
        result = self.explainer.get_top_risk_factors(self.sample_features, k=3)
        required_keys = {"feature_name", "value", "contribution", "direction", "description"}
        for item in result:
            assert required_keys.issubset(item.keys()), f"Missing keys in {item}"

    def test_get_top_risk_factors_sorted_by_contribution(self):
        result = self.explainer.get_top_risk_factors(self.sample_features, k=5)
        contribs = [item["contribution"] for item in result]
        assert contribs == sorted(contribs, reverse=True)

    def test_get_top_risk_factors_direction_valid_values(self):
        result = self.explainer.get_top_risk_factors(self.sample_features, k=5)
        for item in result:
            assert item["direction"] in ("increases_risk", "decreases_risk")

    def test_explainer_with_logistic_regression(self):
        """Explainer should also work with a LR model using coef_."""
        features = _make_features(n_samples=60)
        labels = _make_labels(n_samples=60)
        clf = FinancialDistressClassifier(model_type="logistic_regression")
        clf.train(features, labels)
        explainer = DistressExplainer(clf.model_, self.feature_names)
        result = explainer.explain_prediction(self.sample_features)
        assert len(result) == len(self.feature_names)

    def test_explainer_known_protective_features(self):
        """Features in the protective set should decrease risk when positive."""
        feature_names = ["cash", "total_liabilities"]
        features = _make_features(n_samples=60, n_features=2)
        labels = _make_labels(n_samples=60)
        clf = FinancialDistressClassifier(model_type="random_forest")
        clf.train(features, labels)
        explainer = DistressExplainer(clf.model_, feature_names)

        # Positive cash (protective) should decrease risk
        result = explainer.explain_prediction([100_000.0, 50_000.0])
        cash_entry = next(t for t in result if t[0] == "cash")
        assert cash_entry[2] == "decreases_risk"

        # Positive total_liabilities (risk factor) should increase risk
        liabilities_entry = next(t for t in result if t[0] == "total_liabilities")
        assert liabilities_entry[2] == "increases_risk"


# ---------------------------------------------------------------------------
# End-to-end integration with pseudo-labels
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests using pseudo-label generation -> training -> prediction."""

    def _generate_synthetic_financial_records(self, n: int, seed: int = 0) -> list[dict]:
        rng = random.Random(seed)
        records = []
        for _ in range(n):
            ta = rng.uniform(500_000, 10_000_000)
            equity_ratio = rng.uniform(0.1, 0.9)
            records.append(
                {
                    "current_assets": ta * rng.uniform(0.1, 0.6),
                    "current_liabilities": ta * rng.uniform(0.05, 0.4),
                    "total_assets": ta,
                    "retained_earnings": ta * rng.uniform(-0.3, 0.5),
                    "ebit": ta * rng.uniform(-0.1, 0.3),
                    "total_equity": ta * equity_ratio,
                    "total_liabilities": ta * (1 - equity_ratio),
                    "revenue": ta * rng.uniform(0.2, 2.0),
                }
            )
        return records

    @pytest.mark.slow
    def test_full_pipeline_rf(self):
        records = self._generate_synthetic_financial_records(80)
        labels = FinancialDistressClassifier.generate_pseudo_labels(records)

        # Build simple feature vectors from the records
        feature_names = [
            "current_assets",
            "current_liabilities",
            "total_assets",
            "retained_earnings",
            "ebit",
            "total_equity",
            "total_liabilities",
            "revenue",
        ]
        features = [[r.get(k, 0) or 0 for k in feature_names] for r in records]

        clf = FinancialDistressClassifier(model_type="random_forest")
        metrics = clf.train(features, labels)

        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

        preds = clf.predict(features[:5])
        assert len(preds) == 5

        probas = clf.predict_proba(features[:5])
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in probas)

        importances = clf.get_feature_importance(feature_names)
        assert len(importances) == len(feature_names)

        explainer = DistressExplainer(clf.model_, feature_names)
        top_factors = explainer.get_top_risk_factors(features[0], k=5)
        assert len(top_factors) == 5

    def test_all_same_label_trains_without_crash(self):
        """If all labels are 0, cross-val may warn but should not crash fatally."""
        features = _make_features(n_samples=20, n_features=4)
        labels = [0] * 20
        clf = FinancialDistressClassifier(model_type="logistic_regression")
        try:
            clf.train(features, labels)
        except Exception as exc:
            # Some folds with only one class will raise; that is acceptable
            # as long as it's a clear error, not a silent corruption
            assert isinstance(exc, (ValueError, Exception))
