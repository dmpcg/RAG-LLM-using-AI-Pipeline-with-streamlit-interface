"""Feature engineering for financial ML models.

Extracts, transforms, and selects features from ``FinancialData`` instances
and time-series data for downstream machine-learning pipelines.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Optional

from financial_analyzer import FinancialData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_divide(
    numerator: Optional[float],
    denominator: Optional[float],
    default: Optional[float] = None,
) -> Optional[float]:
    """Safely divide, returning *default* for None inputs or zero denominators."""
    if numerator is None or denominator is None:
        return default
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Scalers (lightweight, no numpy/sklearn dependency)
# ---------------------------------------------------------------------------


class _RobustScaler:
    """Scales features by removing the median and scaling to IQR."""

    def __init__(self) -> None:
        self.median_: list[float] = []
        self.iqr_: list[float] = []

    def fit(self, matrix: list[list[float]]) -> None:
        n_features = len(matrix[0]) if matrix else 0
        self.median_ = []
        self.iqr_ = []
        for j in range(n_features):
            col = sorted(row[j] for row in matrix)
            self.median_.append(_percentile(col, 50))
            q25 = _percentile(col, 25)
            q75 = _percentile(col, 75)
            self.iqr_.append(q75 - q25)

    def transform(self, vector: list[float]) -> list[float]:
        return [
            safe_divide(v - m, iqr, default=0.0)  # type: ignore[arg-type]
            for v, m, iqr in zip(vector, self.median_, self.iqr_)
        ]


class _StandardScaler:
    """Scales features to zero mean and unit variance."""

    def __init__(self) -> None:
        self.mean_: list[float] = []
        self.std_: list[float] = []

    def fit(self, matrix: list[list[float]]) -> None:
        n_features = len(matrix[0]) if matrix else 0
        self.mean_ = []
        self.std_ = []
        for j in range(n_features):
            col = [row[j] for row in matrix]
            mu = sum(col) / len(col)
            self.mean_.append(mu)
            var = sum((x - mu) ** 2 for x in col) / len(col)
            self.std_.append(math.sqrt(var))

    def transform(self, vector: list[float]) -> list[float]:
        return [
            safe_divide(v - m, s, default=0.0)  # type: ignore[arg-type]
            for v, m, s in zip(vector, self.mean_, self.std_)
        ]


class _MinMaxScaler:
    """Scales features to [0, 1] range."""

    def __init__(self) -> None:
        self.min_: list[float] = []
        self.range_: list[float] = []

    def fit(self, matrix: list[list[float]]) -> None:
        n_features = len(matrix[0]) if matrix else 0
        self.min_ = []
        self.range_ = []
        for j in range(n_features):
            col = [row[j] for row in matrix]
            mn = min(col)
            mx = max(col)
            self.min_.append(mn)
            self.range_.append(mx - mn)

    def transform(self, vector: list[float]) -> list[float]:
        return [
            safe_divide(v - mn, rng, default=0.0)  # type: ignore[arg-type]
            for v, mn, rng in zip(vector, self.min_, self.range_)
        ]


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute *pct*-th percentile from an already-sorted list."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


_SCALER_MAP = {
    "robust": _RobustScaler,
    "standard": _StandardScaler,
    "minmax": _MinMaxScaler,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDUSTRY_SECTORS = [
    "technology",
    "healthcare",
    "financials",
    "consumer_discretionary",
    "consumer_staples",
    "industrials",
    "energy",
    "materials",
    "real_estate",
    "utilities",
    "communication_services",
    "other",
]

REVENUE_THRESHOLDS = {
    "small": 50_000_000,  # < $50M
    "medium": 1_000_000_000,  # < $1B
    # anything >= $1B is "large"
}

REPORT_TYPES = ["annual", "quarterly", "interim"]


# ===================================================================
# FinancialFeatureExtractor
# ===================================================================


class FinancialFeatureExtractor:
    """Extracts, combines, and scales features from financial data."""

    def __init__(self, scaler_type: str = "robust") -> None:
        if scaler_type not in _SCALER_MAP:
            raise ValueError(f"Unknown scaler_type '{scaler_type}'. Choose from {list(_SCALER_MAP.keys())}.")
        self._scaler_type = scaler_type
        self._scaler = _SCALER_MAP[scaler_type]()
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    # Ratio features
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_ratio_features(financial_data: FinancialData) -> dict[str, Optional[float]]:
        """Extract a standardised ratio feature dict from *financial_data*.

        Returns a dict mapping feature name to value (``None`` when the
        required fields are missing or denominators are zero).
        """
        fd = financial_data

        # Derived helpers
        total_debt = fd.total_debt
        if total_debt is None and fd.total_liabilities is not None:
            total_debt = fd.total_liabilities

        ebit = fd.ebit
        if ebit is None:
            ebit = fd.operating_income

        features: dict[str, Optional[float]] = {}

        # -- Profitability ------------------------------------------------
        features["gross_margin"] = safe_divide(fd.gross_profit, fd.revenue)
        features["operating_margin"] = safe_divide(fd.operating_income, fd.revenue)
        features["net_margin"] = safe_divide(fd.net_income, fd.revenue)
        features["roa"] = safe_divide(fd.net_income, fd.total_assets)
        features["roe"] = safe_divide(fd.net_income, fd.total_equity)

        # -- Liquidity ----------------------------------------------------
        features["current_ratio"] = safe_divide(fd.current_assets, fd.current_liabilities)
        quick_assets = None
        if fd.current_assets is not None:
            inv = fd.inventory if fd.inventory is not None else 0.0
            quick_assets = fd.current_assets - inv
        features["quick_ratio"] = safe_divide(quick_assets, fd.current_liabilities)
        features["cash_ratio"] = safe_divide(fd.cash, fd.current_liabilities)

        # -- Leverage -----------------------------------------------------
        features["debt_to_equity"] = safe_divide(total_debt, fd.total_equity)
        features["debt_to_assets"] = safe_divide(total_debt, fd.total_assets)
        features["interest_coverage"] = safe_divide(ebit, fd.interest_expense)

        # -- Efficiency ---------------------------------------------------
        features["asset_turnover"] = safe_divide(fd.revenue, fd.total_assets)
        features["inventory_turnover"] = safe_divide(
            fd.cogs,
            fd.avg_inventory if fd.avg_inventory is not None else fd.inventory,
        )
        features["receivables_turnover"] = safe_divide(
            fd.revenue,
            fd.avg_receivables if fd.avg_receivables is not None else fd.accounts_receivable,
        )

        return features

    # ------------------------------------------------------------------ #
    # Temporal features
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_temporal_features(time_series: list[dict[str, Any]]) -> dict[str, Optional[float]]:
        """Derive temporal features from a chronologically-ordered list of
        period dicts.

        Each dict is expected to contain at least a ``"value"`` key with a
        numeric value (or ``None``).  Additional per-metric keys are ignored.
        """
        features: dict[str, Optional[float]] = {}

        if not time_series:
            return features

        values = [p.get("value") for p in time_series]
        numeric_values = [v for v in values if v is not None]

        if len(numeric_values) < 1:
            return features

        # -- Growth rates -------------------------------------------------
        growth_rates: list[float] = []
        for i in range(1, len(values)):
            if values[i] is not None and values[i - 1] is not None:
                gr = safe_divide(values[i] - values[i - 1], abs(values[i - 1]))
                if gr is not None:
                    growth_rates.append(gr)

        features["growth_rate_latest"] = growth_rates[-1] if growth_rates else None
        features["growth_rate_mean"] = sum(growth_rates) / len(growth_rates) if growth_rates else None

        # -- Moving averages ----------------------------------------------
        features["ma_3"] = sum(numeric_values[-3:]) / min(3, len(numeric_values)) if numeric_values else None
        features["ma_5"] = sum(numeric_values[-5:]) / min(5, len(numeric_values)) if numeric_values else None

        # -- Volatility ---------------------------------------------------
        if len(growth_rates) >= 2:
            features["volatility"] = statistics.stdev(growth_rates)
        else:
            features["volatility"] = None

        # -- Momentum (current vs N-period-ago) ---------------------------
        features["momentum_3"] = (
            safe_divide(
                numeric_values[-1] - numeric_values[-3],
                abs(numeric_values[-3]),
            )
            if len(numeric_values) >= 3
            else None
        )
        features["momentum_5"] = (
            safe_divide(
                numeric_values[-1] - numeric_values[-5],
                abs(numeric_values[-5]),
            )
            if len(numeric_values) >= 5
            else None
        )

        return features

    # ------------------------------------------------------------------ #
    # Categorical encoding
    # ------------------------------------------------------------------ #

    @staticmethod
    def encode_categorical(metadata: dict[str, Any]) -> dict[str, float]:
        """Encode categorical metadata into numeric features.

        Handles:
        * ``industry`` -- one-hot across ``INDUSTRY_SECTORS``
        * ``revenue`` -- ordinal company-size bucket
        * ``report_type`` -- one-hot across ``REPORT_TYPES``
        """
        features: dict[str, float] = {}

        # -- Industry (one-hot) -------------------------------------------
        industry = (metadata.get("industry") or "other").lower().strip()
        if industry not in INDUSTRY_SECTORS:
            industry = "other"
        for sector in INDUSTRY_SECTORS:
            features[f"industry_{sector}"] = 1.0 if sector == industry else 0.0

        # -- Company size (ordinal) ---------------------------------------
        revenue = metadata.get("revenue")
        if revenue is not None and isinstance(revenue, (int, float)):
            if revenue < REVENUE_THRESHOLDS["small"]:
                features["company_size"] = 0.0
            elif revenue < REVENUE_THRESHOLDS["medium"]:
                features["company_size"] = 1.0
            else:
                features["company_size"] = 2.0
        else:
            features["company_size"] = 0.0  # default small

        # -- Report type (one-hot) ----------------------------------------
        report_type = (metadata.get("report_type") or "annual").lower().strip()
        if report_type not in REPORT_TYPES:
            report_type = "annual"
        for rt in REPORT_TYPES:
            features[f"report_type_{rt}"] = 1.0 if rt == report_type else 0.0

        return features

    # ------------------------------------------------------------------ #
    # Build combined feature vector
    # ------------------------------------------------------------------ #

    def build_feature_vector(
        self,
        financial_data: FinancialData,
        time_series: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[float]]:
        """Combine all feature sources into parallel (names, values) lists.

        ``None`` values are replaced with ``0.0`` and a companion flag column
        ``_has_{feature}`` is added (1.0 present, 0.0 missing).
        """
        combined: dict[str, Optional[float]] = {}
        combined.update(self.extract_ratio_features(financial_data))

        if time_series is not None:
            combined.update(self.extract_temporal_features(time_series))

        if metadata is not None:
            # encode_categorical always returns float, never None
            combined.update(self.encode_categorical(metadata))

        names: list[str] = []
        values: list[float] = []

        for name, val in combined.items():
            names.append(name)
            if val is None:
                values.append(0.0)
                names.append(f"_has_{name}")
                values.append(0.0)
            else:
                values.append(float(val))
                names.append(f"_has_{name}")
                values.append(1.0)

        return names, values

    # ------------------------------------------------------------------ #
    # Scaling
    # ------------------------------------------------------------------ #

    def fit_scaler(self, feature_matrix: list[list[float]]) -> None:
        """Fit the scaler on a matrix (list of feature vectors)."""
        if not feature_matrix or not feature_matrix[0]:
            raise ValueError("Cannot fit scaler on empty feature matrix.")
        self._scaler.fit(feature_matrix)
        self._is_fitted = True

    def transform(self, feature_vector: list[float]) -> list[float]:
        """Apply fitted scaler to a single feature vector."""
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit_scaler first.")
        return self._scaler.transform(feature_vector)

    def fit_transform(self, feature_matrix: list[list[float]]) -> list[list[float]]:
        """Fit scaler and transform the matrix in one step."""
        self.fit_scaler(feature_matrix)
        return [self._scaler.transform(row) for row in feature_matrix]


# ===================================================================
# FeatureSelector
# ===================================================================


class FeatureSelector:
    """Utility methods for feature selection (variance, correlation, top-k)."""

    @staticmethod
    def select_by_variance(
        features: list[list[float]],
        names: list[str],
        threshold: float = 0.01,
    ) -> tuple[list[list[float]], list[str]]:
        """Remove features whose variance is below *threshold*.

        Parameters
        ----------
        features : list of feature vectors (rows = samples, cols = features)
        names : feature names aligned with columns
        threshold : minimum variance to keep a feature

        Returns
        -------
        (filtered_features, filtered_names)
        """
        if not features or not features[0]:
            return features, names

        n_features = len(features[0])
        keep_idx: list[int] = []

        for j in range(n_features):
            col = [row[j] for row in features]
            mu = sum(col) / len(col)
            var = sum((x - mu) ** 2 for x in col) / len(col)
            if var >= threshold:
                keep_idx.append(j)

        filtered = [[row[j] for j in keep_idx] for row in features]
        filtered_names = [names[j] for j in keep_idx]
        return filtered, filtered_names

    @staticmethod
    def select_by_correlation(
        features: list[list[float]],
        names: list[str],
        threshold: float = 0.95,
    ) -> tuple[list[list[float]], list[str]]:
        """Remove features whose absolute Pearson correlation exceeds
        *threshold* with an earlier-indexed feature (keep first).
        """
        if not features or not features[0]:
            return features, names

        n_samples = len(features)
        n_features = len(features[0])

        # Pre-compute column stats
        cols: list[list[float]] = []
        means: list[float] = []
        for j in range(n_features):
            col = [features[i][j] for i in range(n_samples)]
            cols.append(col)
            means.append(sum(col) / n_samples)

        def _pearson(j1: int, j2: int) -> float:
            m1, m2 = means[j1], means[j2]
            c1, c2 = cols[j1], cols[j2]
            num = sum((a - m1) * (b - m2) for a, b in zip(c1, c2))
            d1 = math.sqrt(sum((a - m1) ** 2 for a in c1))
            d2 = math.sqrt(sum((b - m2) ** 2 for b in c2))
            denom = d1 * d2
            if denom < 1e-12:
                return 0.0
            return num / denom

        drop: set[int] = set()
        for j in range(n_features):
            if j in drop:
                continue
            for k in range(j + 1, n_features):
                if k in drop:
                    continue
                if abs(_pearson(j, k)) > threshold:
                    drop.add(k)

        keep_idx = [j for j in range(n_features) if j not in drop]
        filtered = [[row[j] for j in keep_idx] for row in features]
        filtered_names = [names[j] for j in keep_idx]
        return filtered, filtered_names

    @staticmethod
    def select_top_k(
        features: list[list[float]],
        names: list[str],
        target: list[float],
        k: int = 10,
        method: str = "mutual_info",
    ) -> tuple[list[list[float]], list[str]]:
        """Select the top-*k* features ranked by association with *target*.

        Supported methods:

        * ``"mutual_info"`` -- discretised mutual information estimate
        * ``"correlation"`` -- absolute Pearson correlation
        """
        if not features or not features[0]:
            return features, names

        n_samples = len(features)
        n_features = len(features[0])
        k = min(k, n_features)

        scores: list[tuple[float, int]] = []

        if method == "correlation":
            t_mean = sum(target) / n_samples
            for j in range(n_features):
                col = [features[i][j] for i in range(n_samples)]
                c_mean = sum(col) / n_samples
                num = sum((a - c_mean) * (b - t_mean) for a, b in zip(col, target))
                d1 = math.sqrt(sum((a - c_mean) ** 2 for a in col))
                d2 = math.sqrt(sum((b - t_mean) ** 2 for b in target))
                denom = d1 * d2
                r = abs(num / denom) if denom > 1e-12 else 0.0
                scores.append((r, j))
        else:
            # mutual_info: simple binned MI estimate
            n_bins = max(2, int(math.sqrt(n_samples)))
            for j in range(n_features):
                col = [features[i][j] for i in range(n_samples)]
                mi = _binned_mi(col, target, n_bins)
                scores.append((mi, j))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_idx = sorted(s[1] for s in scores[:k])

        filtered = [[row[j] for j in top_idx] for row in features]
        filtered_names = [names[j] for j in top_idx]
        return filtered, filtered_names


def _binned_mi(x: list[float], y: list[float], n_bins: int) -> float:
    """Compute a simple binned mutual-information estimate."""
    n = len(x)
    if n == 0:
        return 0.0

    def _bin(vals: list[float], nb: int) -> list[int]:
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-12:
            return [0] * len(vals)
        return [min(int((v - mn) / rng * nb), nb - 1) for v in vals]

    bx = _bin(x, n_bins)
    by = _bin(y, n_bins)

    # Joint and marginal counts
    joint: dict[tuple[int, int], int] = {}
    mx_counts: dict[int, int] = {}
    my_counts: dict[int, int] = {}
    for xi, yi in zip(bx, by):
        joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
        mx_counts[xi] = mx_counts.get(xi, 0) + 1
        my_counts[yi] = my_counts.get(yi, 0) + 1

    mi = 0.0
    for (xi, yi), count in joint.items():
        p_xy = count / n
        p_x = mx_counts[xi] / n
        p_y = my_counts[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))

    return max(mi, 0.0)
