"""Time series forecasting using only numpy and scipy.

Provides AR models, exponential smoothing (simple/double/triple),
ensemble forecasting with walk-forward validation, and prediction intervals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast output with prediction intervals and diagnostics."""

    point_forecast: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    confidence_level: float = 0.95
    method_weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class SimpleARModel:
    """Autoregressive AR(p) model fitted via ordinary least squares.

    Parameters
    ----------
    order : int
        The number of lagged observations to use (p in AR(p)).
    """

    def __init__(self, order: int = 2) -> None:
        if order < 1:
            raise ValueError("AR order must be >= 1")
        self.order = order
        self._coefficients: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._values: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None

    def fit(self, values: list[float]) -> SimpleARModel:
        """Fit AR coefficients using numpy least squares.

        Parameters
        ----------
        values : list[float]
            Historical time series values (at least order + 1 points).

        Returns
        -------
        self
        """
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < self.order + 1:
            raise ValueError(f"Need at least {self.order + 1} data points for AR({self.order}), got {len(arr)}")

        self._values = arr

        # Build design matrix: each row is [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
        n = len(arr)
        X = np.ones((n - self.order, self.order + 1))
        y = arr[self.order :]

        for lag in range(1, self.order + 1):
            X[:, lag] = arr[self.order - lag : n - lag]

        # Solve via least squares
        result, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self._intercept = result[0]
        self._coefficients = result[1:]

        # Compute residuals for prediction intervals
        fitted = X @ result
        self._residuals = y - fitted

        return self

    def predict(self, steps: int = 1) -> list[float]:
        """Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        list[float]
            Forecasted values.
        """
        if self._coefficients is None or self._values is None:
            raise RuntimeError("Model must be fit before predicting")
        if steps < 1:
            raise ValueError("steps must be >= 1")

        history = list(self._values)
        forecasts: list[float] = []

        for step in range(steps):
            # y_hat = intercept + sum(coeff_i * y_{t-i})
            recent = history[-self.order :]
            y_hat = self._intercept + float(np.dot(self._coefficients, recent[::-1]))
            if not math.isfinite(y_hat):
                logger.warning("AR model prediction diverged at step %d", step)
                break
            forecasts.append(y_hat)
            history.append(y_hat)

        return forecasts

    def get_coefficients(self) -> list[float]:
        """Return fitted AR coefficients (excluding intercept)."""
        if self._coefficients is None:
            raise RuntimeError("Model must be fit before accessing coefficients")
        return self._coefficients.tolist()

    def get_residuals(self) -> np.ndarray:
        """Return in-sample residuals."""
        if self._residuals is None:
            raise RuntimeError("Model must be fit before accessing residuals")
        return self._residuals


class ExponentialSmoother:
    """Exponential smoothing: simple, double (Holt), or triple (Holt-Winters).

    Parameters
    ----------
    method : str
        One of "simple", "double", "triple".
    """

    VALID_METHODS = ("simple", "double", "triple")

    def __init__(self, method: str = "double") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")
        self.method = method
        self._alpha: float = 0.3
        self._beta: float = 0.1
        self._gamma: float = 0.1
        self._values: Optional[np.ndarray] = None
        self._seasonal_period: int = 4
        self._level: float = 0.0
        self._trend: float = 0.0
        self._seasonal: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted: bool = False

    @property
    def seasonal_period(self) -> int:
        """Public accessor for the seasonal period."""
        return self._seasonal_period

    def fit(self, values: list[float], seasonal_period: int = 4) -> ExponentialSmoother:
        """Fit the smoother, auto-optimizing parameters via scipy.

        Parameters
        ----------
        values : list[float]
            Historical time series.
        seasonal_period : int
            Period length for triple smoothing (ignored for simple/double).

        Returns
        -------
        self
        """
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < 2:
            raise ValueError("Need at least 2 data points")
        if self.method == "triple" and len(arr) < 2 * seasonal_period:
            raise ValueError(
                f"Triple smoothing needs at least {2 * seasonal_period} points "
                f"for seasonal_period={seasonal_period}, got {len(arr)}"
            )

        self._values = arr
        self._seasonal_period = seasonal_period

        if self.method == "simple":
            self._fit_simple_optimized()
        elif self.method == "double":
            self._fit_double_optimized()
        else:
            self._fit_triple_optimized()

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Simple exponential smoothing
    # ------------------------------------------------------------------

    def _simple_smooth(self, arr: np.ndarray, alpha: float) -> np.ndarray:
        """Return fitted values for simple exponential smoothing."""
        n = len(arr)
        fitted = np.empty(n)
        fitted[0] = arr[0]
        for t in range(1, n):
            fitted[t] = alpha * arr[t] + (1 - alpha) * fitted[t - 1]
        return fitted

    def _fit_simple_optimized(self) -> None:
        arr = self._values
        assert arr is not None

        def objective(params: np.ndarray) -> float:
            alpha = params[0]
            fitted = self._simple_smooth(arr, alpha)
            return float(np.mean((arr - fitted) ** 2))

        result = minimize(
            objective,
            x0=[0.3],
            bounds=[(0.01, 0.99)],
            method="L-BFGS-B",
        )
        self._alpha = float(result.x[0])
        fitted = self._simple_smooth(arr, self._alpha)
        self._level = fitted[-1]
        self._residuals = arr - fitted

    # ------------------------------------------------------------------
    # Double (Holt) exponential smoothing
    # ------------------------------------------------------------------

    def _double_smooth(self, arr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        n = len(arr)
        level = np.empty(n)
        trend = np.empty(n)
        fitted = np.empty(n)

        level[0] = arr[0]
        trend[0] = arr[1] - arr[0] if n > 1 else 0.0
        fitted[0] = arr[0]

        for t in range(1, n):
            level[t] = alpha * arr[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
            fitted[t] = level[t - 1] + trend[t - 1]

        self._level = level[-1]
        self._trend = trend[-1]
        return fitted

    def _fit_double_optimized(self) -> None:
        arr = self._values
        assert arr is not None

        def objective(params: np.ndarray) -> float:
            alpha, beta = params
            fitted = self._double_smooth(arr, alpha, beta)
            residuals = arr[1:] - fitted[1:]  # skip first (exact)
            return float(np.mean(residuals**2))

        result = minimize(
            objective,
            x0=[0.3, 0.1],
            bounds=[(0.01, 0.99), (0.01, 0.99)],
            method="L-BFGS-B",
        )
        self._alpha, self._beta = float(result.x[0]), float(result.x[1])
        fitted = self._double_smooth(arr, self._alpha, self._beta)
        self._residuals = arr - fitted

    # ------------------------------------------------------------------
    # Triple (Holt-Winters) exponential smoothing
    # ------------------------------------------------------------------

    def _triple_smooth(
        self,
        arr: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        m: int,
    ) -> np.ndarray:
        n = len(arr)
        level = np.empty(n)
        trend = np.empty(n)
        seasonal = np.empty(n + m)
        fitted = np.empty(n)

        # Initialize: average of first season as level, simple trend
        level[0] = np.mean(arr[:m])
        trend[0] = (np.mean(arr[m : 2 * m]) - np.mean(arr[:m])) / m

        # Initialize seasonal indices from first full season
        for j in range(m):
            seasonal[j] = arr[j] - level[0]

        fitted[0] = level[0] + seasonal[0]

        for t in range(1, n):
            level[t] = alpha * (arr[t] - seasonal[t % m]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
            seasonal[t + m] = gamma * (arr[t] - level[t]) + (1 - gamma) * seasonal[t % m]
            fitted[t] = level[t - 1] + trend[t - 1] + seasonal[t % m]

        self._level = level[-1]
        self._trend = trend[-1]
        # Store the last full season of seasonal components
        self._seasonal = seasonal[n : n + m].copy()
        if len(self._seasonal) < m:
            # Pad with most recent seasonal values
            last_full = seasonal[-(m + (n % m)) : -(n % m)] if n % m else seasonal[-m:]
            self._seasonal = last_full.copy()

        return fitted

    def _fit_triple_optimized(self) -> None:
        arr = self._values
        assert arr is not None
        m = self._seasonal_period

        def objective(params: np.ndarray) -> float:
            alpha, beta, gamma = params
            fitted = self._triple_smooth(arr, alpha, beta, gamma, m)
            residuals = arr[1:] - fitted[1:]
            return float(np.mean(residuals**2))

        result = minimize(
            objective,
            x0=[0.3, 0.1, 0.1],
            bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)],
            method="L-BFGS-B",
        )
        self._alpha = float(result.x[0])
        self._beta = float(result.x[1])
        self._gamma = float(result.x[2])
        fitted = self._triple_smooth(arr, self._alpha, self._beta, self._gamma, m)
        self._residuals = arr - fitted

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, steps: int = 1) -> list[float]:
        """Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        list[float]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before predicting")
        if steps < 1:
            raise ValueError("steps must be >= 1")

        forecasts: list[float] = []

        if self.method == "simple":
            # Simple: flat forecast at last level
            for _ in range(steps):
                forecasts.append(self._level)

        elif self.method == "double":
            # Holt: level + trend * h
            for h in range(1, steps + 1):
                forecasts.append(self._level + self._trend * h)

        else:
            # Holt-Winters: level + trend * h + seasonal
            m = self._seasonal_period
            seasonal = self._seasonal
            assert seasonal is not None
            for h in range(1, steps + 1):
                s_idx = (h - 1) % m
                forecasts.append(self._level + self._trend * h + seasonal[s_idx])

        return forecasts

    def get_residuals(self) -> np.ndarray:
        """Return in-sample residuals."""
        if self._residuals is None:
            raise RuntimeError("Model must be fit before accessing residuals")
        return self._residuals


def compute_prediction_intervals(
    forecasts: list[float],
    residuals: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute prediction intervals from residual distribution.

    Uses a normal approximation: forecast +/- z * sigma * sqrt(h) where h is
    the forecast horizon step.

    Parameters
    ----------
    forecasts : list[float]
        Point forecasts.
    residuals : array-like
        In-sample residuals from the fitted model.
    confidence : float
        Confidence level (0 < confidence < 1).

    Returns
    -------
    tuple[list[float], list[float]]
        (lower_bounds, upper_bounds)
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1 exclusive")

    from scipy.stats import norm

    resid = np.asarray(residuals, dtype=np.float64)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    z = norm.ppf((1 + confidence) / 2)

    lower: list[float] = []
    upper: list[float] = []
    for h, fc in enumerate(forecasts, start=1):
        # Intervals widen with sqrt of horizon
        width = z * sigma * np.sqrt(h)
        lower.append(fc - width)
        upper.append(fc + width)

    return lower, upper


def walk_forward_validate(
    model: SimpleARModel | ExponentialSmoother,
    values: list[float],
    test_size: int = 3,
) -> dict[str, float]:
    """Walk-forward (expanding window) validation.

    Trains on an expanding window and predicts one step ahead for each
    test observation.

    Parameters
    ----------
    model : SimpleARModel or ExponentialSmoother
        An *unfitted* model instance (will be cloned internally).
    values : list[float]
        Full time series.
    test_size : int
        Number of observations to hold out for testing.

    Returns
    -------
    dict with keys "mae", "rmse", "mape"
    """
    n = len(values)
    if test_size >= n:
        raise ValueError("test_size must be less than series length")

    train_end = n - test_size
    errors: list[float] = []
    pct_errors: list[float] = []

    for i in range(test_size):
        train = values[: train_end + i]
        actual = values[train_end + i]

        # Clone and fit
        try:
            if isinstance(model, SimpleARModel):
                clone = SimpleARModel(order=model.order)
                clone.fit(train)
            else:
                clone = ExponentialSmoother(method=model.method)
                clone.fit(train, seasonal_period=model.seasonal_period)

            pred = clone.predict(steps=1)[0]
        except ValueError:
            # Not enough data for this window size; skip
            continue

        err = actual - pred
        errors.append(err)
        if actual != 0:
            pct_errors.append(abs(err / actual))

    if not errors:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    errors_arr = np.array(errors)
    mae = float(np.mean(np.abs(errors_arr)))
    rmse = float(np.sqrt(np.mean(errors_arr**2)))
    mape = float(np.mean(pct_errors)) if pct_errors else float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape}


class EnsembleForecaster:
    """Combines AR and exponential smoothing via accuracy-weighted averaging.

    Parameters
    ----------
    methods : list[str] or None
        Methods to include. Default: ["ar", "exponential"].
    """

    DEFAULT_METHODS = ["ar", "exponential"]

    def __init__(self, methods: list[str] | None = None) -> None:
        self.methods = methods or list(self.DEFAULT_METHODS)
        self._models: dict[str, SimpleARModel | ExponentialSmoother] = {}
        self._weights: dict[str, float] = {}
        self._metrics: dict[str, float] = {}
        self._residuals: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, values: list[float], seasonal_period: int = 4) -> EnsembleForecaster:
        """Fit all component models and compute accuracy weights.

        Parameters
        ----------
        values : list[float]
            Historical time series.
        seasonal_period : int
            Seasonal period for exponential smoothing.

        Returns
        -------
        self
        """
        if len(values) < 4:
            raise ValueError("Need at least 4 data points for ensemble forecasting")

        model_scores: dict[str, float] = {}
        # Cache walk-forward results to avoid redundant validation passes.
        wf_cache: dict[str, dict[str, float]] = {}

        for method in self.methods:
            try:
                if method == "ar":
                    order = min(2, len(values) - 2)
                    m = SimpleARModel(order=max(1, order))
                    m.fit(values)
                    self._models["ar"] = m
                    metrics = walk_forward_validate(
                        SimpleARModel(order=max(1, order)),
                        values,
                        test_size=min(3, len(values) - max(1, order) - 1),
                    )
                    wf_cache["ar"] = metrics
                    model_scores["ar"] = metrics.get("mae", float("inf"))

                elif method == "exponential":
                    m = ExponentialSmoother(method="double")
                    m.fit(values, seasonal_period=seasonal_period)
                    self._models["exponential"] = m
                    metrics = walk_forward_validate(
                        ExponentialSmoother(method="double"),
                        values,
                        test_size=min(3, len(values) - 2),
                    )
                    wf_cache["exponential"] = metrics
                    model_scores["exponential"] = metrics.get("mae", float("inf"))
            except (ValueError, RuntimeError):
                continue

        if not self._models:
            raise RuntimeError("No models could be fitted successfully")

        # Compute weights: inverse MAE, normalized
        # Replace nan/inf with a large penalty
        max_score = max((s for s in model_scores.values() if np.isfinite(s)), default=1.0)
        safe_scores = {k: (v if np.isfinite(v) else max_score * 10) for k, v in model_scores.items()}

        inv_scores = {k: 1.0 / max(v, 1e-10) for k, v in safe_scores.items()}
        total = sum(inv_scores.values())
        self._weights = {k: v / total for k, v in inv_scores.items()}

        # Aggregate metrics — reuse cached walk-forward results instead of
        # running validation a second time for each model.
        all_metrics: dict[str, list[float]] = {"mae": [], "rmse": [], "mape": []}
        for method_name in self._models:
            met = wf_cache.get(method_name, {})
            for k in all_metrics:
                if np.isfinite(met.get(k, float("nan"))):
                    all_metrics[k].append(met[k])

        self._metrics = {k: float(np.mean(v)) if v else float("nan") for k, v in all_metrics.items()}

        # Combine residuals (weighted)
        residual_arrays = []
        for name, mdl in self._models.items():
            r = mdl.get_residuals()
            residual_arrays.append(r * self._weights.get(name, 0.5))
        if residual_arrays:
            min_len = min(len(r) for r in residual_arrays)
            self._residuals = sum(r[-min_len:] for r in residual_arrays)  # type: ignore[assignment]
        else:
            self._residuals = np.array([0.0])

        self._fitted = True
        return self

    def predict(self, steps: int = 1) -> ForecastResult:
        """Produce a weighted ensemble forecast with prediction intervals.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        ForecastResult
        """
        if not self._fitted:
            raise RuntimeError("Ensemble must be fit before predicting")
        if steps < 1:
            raise ValueError("steps must be >= 1")

        combined = np.zeros(steps)
        for name, mdl in self._models.items():
            preds = np.array(mdl.predict(steps=steps))
            combined += preds * self._weights.get(name, 0.5)

        point_forecast = combined.tolist()

        residuals = self._residuals if self._residuals is not None else np.array([0.0])
        lower, upper = compute_prediction_intervals(point_forecast, residuals, confidence=0.95)

        return ForecastResult(
            point_forecast=point_forecast,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=0.95,
            method_weights=dict(self._weights),
            metrics=dict(self._metrics),
        )
