"""Tests for ml/forecasting.py - time series forecasting module.

All tests use synthetic data only; no external datasets.
"""

import math

import numpy as np
import pytest

from ml.forecasting import (
    EnsembleForecaster,
    ExponentialSmoother,
    ForecastResult,
    SimpleARModel,
    compute_prediction_intervals,
    walk_forward_validate,
)

# ---------------------------------------------------------------------------
# SimpleARModel
# ---------------------------------------------------------------------------


class TestSimpleARModel:
    """Tests for the autoregressive model."""

    def test_fit_and_predict_growing_series(self):
        """AR should capture upward trend from a growing series."""
        values = [100, 110, 121, 133, 146, 160]
        model = SimpleARModel(order=2)
        model.fit(values)
        preds = model.predict(steps=3)
        assert len(preds) == 3
        # Growing series -> predictions should continue upward
        assert preds[0] > values[-1]

    def test_predict_returns_correct_step_count(self):
        values = [10, 20, 30, 40, 50]
        model = SimpleARModel(order=1).fit(values)
        for steps in [1, 5, 10]:
            assert len(model.predict(steps=steps)) == steps

    def test_constant_series_predicts_constant(self):
        values = [42.0] * 10
        model = SimpleARModel(order=2).fit(values)
        preds = model.predict(steps=5)
        for p in preds:
            assert abs(p - 42.0) < 1e-6

    def test_ar1_process(self):
        """Fit an AR(1) series generated with known coefficient."""
        np.random.seed(42)
        phi = 0.8
        n = 100
        y = [0.0]
        for _ in range(n - 1):
            y.append(phi * y[-1] + np.random.normal(0, 0.1))
        model = SimpleARModel(order=1).fit(y)
        coeffs = model.get_coefficients()
        # Recovered coefficient should be close to 0.8
        assert abs(coeffs[0] - phi) < 0.15

    def test_too_few_data_points_raises(self):
        with pytest.raises(ValueError, match="at least"):
            SimpleARModel(order=3).fit([1, 2, 3])

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            SimpleARModel(order=0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            SimpleARModel(order=1).predict(steps=1)

    def test_get_coefficients_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            SimpleARModel(order=1).get_coefficients()

    def test_predict_zero_steps_raises(self):
        model = SimpleARModel(order=1).fit([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="steps"):
            model.predict(steps=0)

    def test_get_residuals(self):
        values = [10, 20, 30, 40, 50]
        model = SimpleARModel(order=1).fit(values)
        residuals = model.get_residuals()
        assert len(residuals) == len(values) - 1

    def test_negative_values(self):
        values = [-50, -40, -30, -20, -10]
        model = SimpleARModel(order=1).fit(values)
        preds = model.predict(steps=2)
        assert len(preds) == 2
        # Trend is upward (less negative)
        assert preds[0] > values[-1]

    def test_alternating_values(self):
        values = [10, 20, 10, 20, 10, 20, 10, 20]
        model = SimpleARModel(order=2).fit(values)
        preds = model.predict(steps=4)
        assert len(preds) == 4


# ---------------------------------------------------------------------------
# ExponentialSmoother
# ---------------------------------------------------------------------------


class TestExponentialSmoother:
    """Tests for simple, double, and triple exponential smoothing."""

    def test_simple_constant_series(self):
        values = [50.0] * 10
        model = ExponentialSmoother(method="simple").fit(values)
        preds = model.predict(steps=3)
        for p in preds:
            assert abs(p - 50.0) < 1.0

    def test_simple_trending_data(self):
        values = [10, 12, 14, 16, 18, 20, 22, 24]
        model = ExponentialSmoother(method="simple").fit(values)
        preds = model.predict(steps=2)
        # Simple smoothing lags behind trend, but predictions should be positive
        assert all(p > 0 for p in preds)

    def test_double_trending_data(self):
        """Double smoothing should capture linear trend."""
        values = [100 + 10 * i for i in range(10)]
        model = ExponentialSmoother(method="double").fit(values)
        preds = model.predict(steps=3)
        # Should continue trending upward
        assert preds[0] > values[-1]
        assert preds[1] > preds[0]

    def test_double_predict_step_count(self):
        values = list(range(10, 110, 10))
        model = ExponentialSmoother(method="double").fit(values)
        assert len(model.predict(steps=5)) == 5

    def test_triple_seasonal_data(self):
        """Triple smoothing should handle seasonal patterns."""
        # Two full cycles of quarterly seasonal data
        values = [100, 120, 80, 100, 110, 130, 90, 110, 115, 135, 95, 115]
        model = ExponentialSmoother(method="triple").fit(values, seasonal_period=4)
        preds = model.predict(steps=4)
        assert len(preds) == 4
        # Predictions should reflect seasonal pattern (not all identical)
        assert max(preds) - min(preds) > 5

    def test_triple_too_short_raises(self):
        values = [100, 120, 80, 100]  # Only 1 season, need 2
        with pytest.raises(ValueError, match="at least"):
            ExponentialSmoother(method="triple").fit(values, seasonal_period=4)

    def test_very_short_series(self):
        values = [10, 20]
        model = ExponentialSmoother(method="simple").fit(values)
        preds = model.predict(steps=1)
        assert len(preds) == 1

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ExponentialSmoother(method="simple").fit([42])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            ExponentialSmoother(method="quadruple")

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            ExponentialSmoother(method="simple").predict(steps=1)

    def test_predict_zero_steps_raises(self):
        model = ExponentialSmoother(method="simple").fit([10, 20, 30])
        with pytest.raises(ValueError, match="steps"):
            model.predict(steps=0)

    def test_get_residuals(self):
        values = [10, 20, 30, 40, 50]
        model = ExponentialSmoother(method="double").fit(values)
        residuals = model.get_residuals()
        assert len(residuals) == len(values)

    def test_zeros_series(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.0]
        model = ExponentialSmoother(method="simple").fit(values)
        preds = model.predict(steps=2)
        for p in preds:
            assert abs(p) < 1e-6

    def test_negative_values(self):
        values = [-100, -90, -80, -70, -60]
        model = ExponentialSmoother(method="double").fit(values)
        preds = model.predict(steps=2)
        assert preds[0] > values[-1]  # trend upward


# ---------------------------------------------------------------------------
# EnsembleForecaster
# ---------------------------------------------------------------------------


class TestEnsembleForecaster:
    """Tests for the ensemble forecaster."""

    def test_fit_and_predict(self):
        values = [100, 110, 121, 133, 146, 160, 176]
        model = EnsembleForecaster().fit(values)
        result = model.predict(steps=3)
        assert isinstance(result, ForecastResult)
        assert len(result.point_forecast) == 3

    def test_returns_all_fields(self):
        values = [10 + i * 5 for i in range(12)]
        model = EnsembleForecaster().fit(values)
        result = model.predict(steps=4)
        assert len(result.point_forecast) == 4
        assert len(result.lower_bound) == 4
        assert len(result.upper_bound) == 4
        assert result.confidence_level == 0.95
        assert isinstance(result.method_weights, dict)
        assert len(result.method_weights) > 0
        assert isinstance(result.metrics, dict)

    def test_weights_sum_to_one(self):
        values = [100 + i * 10 for i in range(10)]
        model = EnsembleForecaster().fit(values)
        result = model.predict(steps=1)
        total = sum(result.method_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_prediction_intervals_bracket_forecast(self):
        values = [50 + i * 3 for i in range(15)]
        model = EnsembleForecaster().fit(values)
        result = model.predict(steps=5)
        for fc, lo, hi in zip(result.point_forecast, result.lower_bound, result.upper_bound):
            assert lo <= fc <= hi

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            EnsembleForecaster().predict(steps=1)

    def test_too_few_data_raises(self):
        with pytest.raises(ValueError, match="at least"):
            EnsembleForecaster().fit([1, 2, 3])

    def test_custom_methods_ar_only(self):
        values = [10, 20, 30, 40, 50, 60, 70]
        model = EnsembleForecaster(methods=["ar"]).fit(values)
        result = model.predict(steps=2)
        assert "ar" in result.method_weights

    def test_custom_methods_exponential_only(self):
        values = [10, 20, 30, 40, 50, 60, 70]
        model = EnsembleForecaster(methods=["exponential"]).fit(values)
        result = model.predict(steps=2)
        assert "exponential" in result.method_weights


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------


class TestForecastResult:
    """Tests for the ForecastResult dataclass."""

    def test_default_confidence(self):
        r = ForecastResult(
            point_forecast=[1.0],
            lower_bound=[0.5],
            upper_bound=[1.5],
        )
        assert r.confidence_level == 0.95

    def test_custom_fields(self):
        r = ForecastResult(
            point_forecast=[10, 20],
            lower_bound=[8, 16],
            upper_bound=[12, 24],
            confidence_level=0.90,
            method_weights={"ar": 0.6, "exp": 0.4},
            metrics={"mae": 1.5},
        )
        assert r.confidence_level == 0.90
        assert r.method_weights["ar"] == 0.6
        assert r.metrics["mae"] == 1.5


# ---------------------------------------------------------------------------
# compute_prediction_intervals
# ---------------------------------------------------------------------------


class TestComputePredictionIntervals:
    """Tests for prediction interval computation."""

    def test_bounds_contain_forecast(self):
        forecasts = [100, 110, 120]
        residuals = [1, -1, 0.5, -0.5, 0.2]
        lower, upper = compute_prediction_intervals(forecasts, residuals)
        for fc, lo, hi in zip(forecasts, lower, upper):
            assert lo < fc < hi

    def test_intervals_widen_with_horizon(self):
        forecasts = [100, 100, 100, 100]
        residuals = [1, -1, 2, -2, 0.5]
        lower, upper = compute_prediction_intervals(forecasts, residuals)
        widths = [hi - lo for lo, hi in zip(lower, upper)]
        for i in range(1, len(widths)):
            assert widths[i] > widths[i - 1]

    def test_wider_intervals_with_lower_confidence(self):
        forecasts = [50, 60, 70]
        residuals = [2, -3, 1, -1, 4]
        _, upper_95 = compute_prediction_intervals(forecasts, residuals, confidence=0.95)
        _, upper_80 = compute_prediction_intervals(forecasts, residuals, confidence=0.80)
        # 95% interval should be wider than 80%
        for u95, u80 in zip(upper_95, upper_80):
            assert u95 > u80

    def test_zero_residuals_gives_point_interval(self):
        forecasts = [10, 20]
        residuals = [0, 0, 0]
        lower, upper = compute_prediction_intervals(forecasts, residuals)
        for fc, lo, hi in zip(forecasts, lower, upper):
            assert abs(lo - fc) < 1e-6
            assert abs(hi - fc) < 1e-6

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            compute_prediction_intervals([1], [0], confidence=0.0)
        with pytest.raises(ValueError, match="confidence"):
            compute_prediction_intervals([1], [0], confidence=1.0)


# ---------------------------------------------------------------------------
# walk_forward_validate
# ---------------------------------------------------------------------------


class TestWalkForwardValidate:
    """Tests for walk-forward validation."""

    def test_returns_metrics_dict(self):
        values = [10, 20, 30, 40, 50, 60, 70]
        model = SimpleARModel(order=1)
        metrics = walk_forward_validate(model, values, test_size=3)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics

    def test_perfect_linear_has_low_error(self):
        """A perfect linear series should have low AR(1) error."""
        values = [float(i * 10) for i in range(20)]
        model = SimpleARModel(order=1)
        metrics = walk_forward_validate(model, values, test_size=3)
        assert metrics["mae"] < 5.0

    def test_test_size_too_large_raises(self):
        values = [1, 2, 3]
        with pytest.raises(ValueError, match="test_size"):
            walk_forward_validate(SimpleARModel(order=1), values, test_size=3)

    def test_with_exponential_smoother(self):
        values = [100, 110, 105, 115, 108, 118, 112, 122]
        model = ExponentialSmoother(method="double")
        metrics = walk_forward_validate(model, values, test_size=2)
        assert math.isfinite(metrics["mae"])
        assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE always

    def test_metrics_non_negative(self):
        values = [5, 10, 15, 20, 25, 30]
        model = SimpleARModel(order=1)
        metrics = walk_forward_validate(model, values, test_size=2)
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_two_values_ar1(self):
        model = SimpleARModel(order=1).fit([10, 20])
        preds = model.predict(steps=1)
        assert len(preds) == 1

    def test_large_values(self):
        values = [1e9 + i * 1e6 for i in range(10)]
        model = SimpleARModel(order=1).fit(values)
        preds = model.predict(steps=2)
        assert all(p > 1e9 for p in preds)

    def test_small_values(self):
        values = [1e-6 * (i + 1) for i in range(10)]
        model = ExponentialSmoother(method="double").fit(values)
        preds = model.predict(steps=2)
        assert all(p > 0 for p in preds)

    def test_single_value_ar_raises(self):
        with pytest.raises(ValueError):
            SimpleARModel(order=1).fit([42])

    def test_mixed_positive_negative(self):
        values = [10, -5, 20, -10, 15, -3, 25]
        model = SimpleARModel(order=2).fit(values)
        preds = model.predict(steps=3)
        assert len(preds) == 3

    def test_all_zeros(self):
        values = [0.0] * 8
        model = ExponentialSmoother(method="double").fit(values)
        preds = model.predict(steps=2)
        for p in preds:
            assert abs(p) < 1e-6

    def test_ensemble_with_noisy_data(self):
        np.random.seed(123)
        values = [100 + 5 * i + np.random.normal(0, 3) for i in range(20)]
        model = EnsembleForecaster().fit(values)
        result = model.predict(steps=4)
        assert len(result.point_forecast) == 4
        # Forecasts should be in a reasonable range
        assert all(50 < p < 300 for p in result.point_forecast)
