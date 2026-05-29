"""Tests for ml.feature_engineering module."""

from __future__ import annotations

import pytest

from financial_analyzer import FinancialData
from ml.feature_engineering import (
    FeatureSelector,
    FinancialFeatureExtractor,
    safe_divide,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def complete_fd():
    """A FinancialData instance with all fields populated."""
    return FinancialData(
        revenue=1_000_000,
        cogs=600_000,
        gross_profit=400_000,
        operating_income=200_000,
        operating_expenses=200_000,
        ebit=200_000,
        ebitda=250_000,
        interest_expense=20_000,
        net_income=150_000,
        total_assets=2_000_000,
        current_assets=800_000,
        cash=200_000,
        inventory=100_000,
        accounts_receivable=150_000,
        total_liabilities=1_000_000,
        current_liabilities=400_000,
        total_debt=600_000,
        total_equity=1_000_000,
        operating_cash_flow=180_000,
        avg_inventory=95_000,
        avg_receivables=140_000,
    )


@pytest.fixture
def partial_fd():
    """A FinancialData with only a few fields populated."""
    return FinancialData(
        revenue=500_000,
        net_income=50_000,
        total_assets=1_000_000,
    )


@pytest.fixture
def zero_denom_fd():
    """FinancialData where denominators are zero."""
    return FinancialData(
        revenue=0,
        net_income=100_000,
        total_assets=0,
        total_equity=0,
        current_liabilities=0,
        interest_expense=0,
    )


@pytest.fixture
def extractor():
    return FinancialFeatureExtractor(scaler_type="robust")


@pytest.fixture
def time_series_normal():
    return [
        {"value": 100},
        {"value": 110},
        {"value": 105},
        {"value": 120},
        {"value": 130},
    ]


# -----------------------------------------------------------------------
# safe_divide (local copy)
# -----------------------------------------------------------------------


class TestSafeDivide:
    def test_normal(self):
        assert safe_divide(10.0, 2.0) == 5.0

    def test_zero_denom(self):
        assert safe_divide(10.0, 0.0) is None

    def test_none_numerator(self):
        assert safe_divide(None, 5.0) is None

    def test_none_denominator(self):
        assert safe_divide(5.0, None) is None

    def test_default_value(self):
        assert safe_divide(10.0, 0.0, default=0.0) == 0.0

    def test_near_zero_denom(self):
        assert safe_divide(10.0, 1e-15) is None


# -----------------------------------------------------------------------
# extract_ratio_features
# -----------------------------------------------------------------------


class TestExtractRatioFeatures:
    def test_complete_data(self, complete_fd):
        features = FinancialFeatureExtractor.extract_ratio_features(complete_fd)

        assert features["gross_margin"] == pytest.approx(0.4)
        assert features["operating_margin"] == pytest.approx(0.2)
        assert features["net_margin"] == pytest.approx(0.15)
        assert features["roa"] == pytest.approx(0.075)
        assert features["roe"] == pytest.approx(0.15)
        assert features["current_ratio"] == pytest.approx(2.0)
        # quick_ratio = (800k - 100k) / 400k = 1.75
        assert features["quick_ratio"] == pytest.approx(1.75)
        assert features["cash_ratio"] == pytest.approx(0.5)
        assert features["debt_to_equity"] == pytest.approx(0.6)
        assert features["debt_to_assets"] == pytest.approx(0.3)
        assert features["interest_coverage"] == pytest.approx(10.0)
        assert features["asset_turnover"] == pytest.approx(0.5)
        # inventory_turnover = cogs / avg_inventory = 600k / 95k
        assert features["inventory_turnover"] == pytest.approx(600_000 / 95_000)
        # receivables_turnover = revenue / avg_receivables = 1M / 140k
        assert features["receivables_turnover"] == pytest.approx(1_000_000 / 140_000)

    def test_partial_data(self, partial_fd):
        features = FinancialFeatureExtractor.extract_ratio_features(partial_fd)
        assert features["gross_margin"] is None  # no gross_profit
        assert features["net_margin"] == pytest.approx(0.1)
        assert features["roa"] == pytest.approx(0.05)
        assert features["roe"] is None  # no equity
        assert features["current_ratio"] is None
        assert features["debt_to_equity"] is None

    def test_zero_denominators(self, zero_denom_fd):
        features = FinancialFeatureExtractor.extract_ratio_features(zero_denom_fd)
        assert features["net_margin"] is None  # revenue=0
        assert features["roa"] is None  # total_assets=0
        assert features["roe"] is None  # total_equity=0
        assert features["current_ratio"] is None  # current_liabilities=0
        assert features["interest_coverage"] is None  # interest_expense=0

    def test_returns_all_expected_keys(self, complete_fd):
        features = FinancialFeatureExtractor.extract_ratio_features(complete_fd)
        expected = {
            "gross_margin",
            "operating_margin",
            "net_margin",
            "roa",
            "roe",
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
            "debt_to_equity",
            "debt_to_assets",
            "interest_coverage",
            "asset_turnover",
            "inventory_turnover",
            "receivables_turnover",
        }
        assert set(features.keys()) == expected

    def test_no_inventory_uses_current_assets_for_quick(self):
        fd = FinancialData(current_assets=500, current_liabilities=250)
        features = FinancialFeatureExtractor.extract_ratio_features(fd)
        # inventory defaults to 0 when None, so quick = current_assets / cl
        assert features["quick_ratio"] == pytest.approx(2.0)

    def test_debt_falls_back_to_total_liabilities(self):
        fd = FinancialData(total_liabilities=800, total_equity=200, total_assets=1000)
        features = FinancialFeatureExtractor.extract_ratio_features(fd)
        # total_debt is None, falls back to total_liabilities
        assert features["debt_to_equity"] == pytest.approx(4.0)
        assert features["debt_to_assets"] == pytest.approx(0.8)

    def test_ebit_fallback_to_operating_income(self):
        fd = FinancialData(operating_income=300, interest_expense=30)
        features = FinancialFeatureExtractor.extract_ratio_features(fd)
        assert features["interest_coverage"] == pytest.approx(10.0)


# -----------------------------------------------------------------------
# extract_temporal_features
# -----------------------------------------------------------------------


class TestExtractTemporalFeatures:
    def test_normal_series(self, time_series_normal):
        features = FinancialFeatureExtractor.extract_temporal_features(time_series_normal)
        assert features["growth_rate_latest"] is not None
        assert features["ma_3"] is not None
        assert features["ma_5"] is not None
        assert features["volatility"] is not None
        assert features["momentum_3"] is not None
        assert features["momentum_5"] is not None

    def test_growth_rate_latest(self, time_series_normal):
        features = FinancialFeatureExtractor.extract_temporal_features(time_series_normal)
        # last growth = (130-120)/120
        assert features["growth_rate_latest"] == pytest.approx(10 / 120)

    def test_moving_averages(self, time_series_normal):
        features = FinancialFeatureExtractor.extract_temporal_features(time_series_normal)
        # ma_3 = mean of last 3 numeric: (105+120+130)/3
        assert features["ma_3"] == pytest.approx((105 + 120 + 130) / 3)
        # ma_5 = mean of all 5
        assert features["ma_5"] == pytest.approx((100 + 110 + 105 + 120 + 130) / 5)

    def test_single_period(self):
        features = FinancialFeatureExtractor.extract_temporal_features([{"value": 42}])
        assert features.get("growth_rate_latest") is None
        assert features.get("volatility") is None
        assert features["ma_3"] == pytest.approx(42.0)
        assert features.get("momentum_3") is None

    def test_empty_list(self):
        features = FinancialFeatureExtractor.extract_temporal_features([])
        assert features == {}

    def test_all_none_values(self):
        features = FinancialFeatureExtractor.extract_temporal_features([{"value": None}, {"value": None}])
        assert features == {}

    def test_two_periods(self):
        features = FinancialFeatureExtractor.extract_temporal_features([{"value": 100}, {"value": 150}])
        assert features["growth_rate_latest"] == pytest.approx(0.5)
        # Only one growth rate, so volatility requires >=2 -> None
        assert features["volatility"] is None

    def test_momentum_with_zero_base(self):
        ts = [{"value": 0}, {"value": 5}, {"value": 10}]
        features = FinancialFeatureExtractor.extract_temporal_features(ts)
        # momentum_3 = (10-0)/abs(0) -> None (div by zero)
        assert features["momentum_3"] is None


# -----------------------------------------------------------------------
# encode_categorical
# -----------------------------------------------------------------------


class TestEncodeCategorical:
    def test_known_industry(self):
        features = FinancialFeatureExtractor.encode_categorical({"industry": "technology"})
        assert features["industry_technology"] == 1.0
        assert features["industry_healthcare"] == 0.0
        assert sum(v for k, v in features.items() if k.startswith("industry_")) == 1.0

    def test_unknown_industry_defaults_other(self):
        features = FinancialFeatureExtractor.encode_categorical({"industry": "space_mining"})
        assert features["industry_other"] == 1.0

    def test_missing_industry_defaults_other(self):
        features = FinancialFeatureExtractor.encode_categorical({})
        assert features["industry_other"] == 1.0

    def test_company_size_small(self):
        features = FinancialFeatureExtractor.encode_categorical({"revenue": 10_000_000})
        assert features["company_size"] == 0.0

    def test_company_size_medium(self):
        features = FinancialFeatureExtractor.encode_categorical({"revenue": 500_000_000})
        assert features["company_size"] == 1.0

    def test_company_size_large(self):
        features = FinancialFeatureExtractor.encode_categorical({"revenue": 5_000_000_000})
        assert features["company_size"] == 2.0

    def test_report_type_quarterly(self):
        features = FinancialFeatureExtractor.encode_categorical({"report_type": "quarterly"})
        assert features["report_type_quarterly"] == 1.0
        assert features["report_type_annual"] == 0.0

    def test_unknown_report_type_defaults_annual(self):
        features = FinancialFeatureExtractor.encode_categorical({"report_type": "biweekly"})
        assert features["report_type_annual"] == 1.0


# -----------------------------------------------------------------------
# build_feature_vector
# -----------------------------------------------------------------------


class TestBuildFeatureVector:
    def test_parallel_lists(self, extractor, complete_fd):
        names, values = extractor.build_feature_vector(complete_fd)
        assert len(names) == len(values)
        assert all(isinstance(n, str) for n in names)
        assert all(isinstance(v, float) for v in values)

    def test_none_replacement_with_flags(self, extractor, partial_fd):
        names, values = extractor.build_feature_vector(partial_fd)
        # gross_margin is None -> value = 0.0 and _has_gross_margin = 0.0
        gm_idx = names.index("gross_margin")
        has_gm_idx = names.index("_has_gross_margin")
        assert values[gm_idx] == 0.0
        assert values[has_gm_idx] == 0.0

        # net_margin is present -> _has_net_margin = 1.0
        nm_idx = names.index("net_margin")
        has_nm_idx = names.index("_has_net_margin")
        assert values[nm_idx] == pytest.approx(0.1)
        assert values[has_nm_idx] == 1.0

    def test_with_time_series(self, extractor, complete_fd, time_series_normal):
        names, values = extractor.build_feature_vector(complete_fd, time_series=time_series_normal)
        assert "growth_rate_latest" in names
        assert "ma_3" in names

    def test_with_metadata(self, extractor, complete_fd):
        names, values = extractor.build_feature_vector(complete_fd, metadata={"industry": "technology", "revenue": 1e9})
        assert "industry_technology" in names
        assert "company_size" in names

    def test_all_sources_combined(self, extractor, complete_fd, time_series_normal):
        names, values = extractor.build_feature_vector(
            complete_fd,
            time_series=time_series_normal,
            metadata={"industry": "healthcare", "report_type": "quarterly"},
        )
        # Should contain ratio, temporal, and categorical features
        assert "gross_margin" in names
        assert "growth_rate_latest" in names
        assert "industry_healthcare" in names


# -----------------------------------------------------------------------
# Scaling
# -----------------------------------------------------------------------


class TestScaling:
    def test_fit_transform_robust(self):
        ext = FinancialFeatureExtractor(scaler_type="robust")
        matrix = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        result = ext.fit_transform(matrix)
        assert len(result) == 4
        assert len(result[0]) == 2

    def test_fit_transform_standard(self):
        ext = FinancialFeatureExtractor(scaler_type="standard")
        matrix = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        result = ext.fit_transform(matrix)
        # mean=1, std=sqrt(2/3)
        assert result[1][0] == pytest.approx(0.0, abs=1e-9)

    def test_fit_transform_minmax(self):
        ext = FinancialFeatureExtractor(scaler_type="minmax")
        matrix = [[0.0], [5.0], [10.0]]
        result = ext.fit_transform(matrix)
        assert result[0][0] == pytest.approx(0.0)
        assert result[1][0] == pytest.approx(0.5)
        assert result[2][0] == pytest.approx(1.0)

    def test_transform_before_fit_raises(self):
        ext = FinancialFeatureExtractor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            ext.transform([1.0, 2.0])

    def test_fit_empty_raises(self):
        ext = FinancialFeatureExtractor()
        with pytest.raises(ValueError, match="empty"):
            ext.fit_scaler([])

    def test_round_trip(self):
        ext = FinancialFeatureExtractor(scaler_type="minmax")
        matrix = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]
        ext.fit_scaler(matrix)
        transformed = ext.transform([20.0, 200.0])
        assert transformed[0] == pytest.approx(0.5)
        assert transformed[1] == pytest.approx(0.5)

    def test_invalid_scaler_type(self):
        with pytest.raises(ValueError, match="Unknown scaler_type"):
            FinancialFeatureExtractor(scaler_type="invalid")


# -----------------------------------------------------------------------
# FeatureSelector
# -----------------------------------------------------------------------


class TestFeatureSelectorVariance:
    def test_removes_constant_feature(self):
        features = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
        names = ["varying", "constant"]
        filtered, fnames = FeatureSelector.select_by_variance(features, names, threshold=0.01)
        assert "constant" not in fnames
        assert "varying" in fnames

    def test_keeps_all_when_above_threshold(self):
        features = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        names = ["a", "b"]
        filtered, fnames = FeatureSelector.select_by_variance(features, names, threshold=0.01)
        assert fnames == ["a", "b"]

    def test_empty_input(self):
        filtered, fnames = FeatureSelector.select_by_variance([], [], threshold=0.01)
        assert filtered == []
        assert fnames == []


class TestFeatureSelectorCorrelation:
    def test_removes_highly_correlated(self):
        # a and b are perfectly correlated (b = 2*a)
        features = [[1.0, 2.0, 10.0], [2.0, 4.0, 20.0], [3.0, 6.0, 30.0]]
        names = ["a", "b", "c"]
        filtered, fnames = FeatureSelector.select_by_correlation(features, names, threshold=0.95)
        # b should be dropped (correlated with a), c too (correlated with both)
        # Actually a, b, c are all perfectly correlated
        assert "a" in fnames
        assert "b" not in fnames

    def test_keeps_uncorrelated(self):
        features = [[1.0, -1.0], [2.0, 1.0], [3.0, -1.0], [4.0, 1.0]]
        names = ["x", "y"]
        filtered, fnames = FeatureSelector.select_by_correlation(features, names, threshold=0.95)
        assert len(fnames) == 2

    def test_empty_input(self):
        filtered, fnames = FeatureSelector.select_by_correlation([], [])
        assert filtered == []


class TestFeatureSelectorTopK:
    def test_selects_k_features(self):
        # Feature 0 is strongly correlated with target, feature 1 is random-ish
        features = [
            [1.0, 5.0, 10.0],
            [2.0, 3.0, 20.0],
            [3.0, 7.0, 30.0],
            [4.0, 2.0, 40.0],
            [5.0, 8.0, 50.0],
        ]
        names = ["strong", "weak", "also_strong"]
        target = [10.0, 20.0, 30.0, 40.0, 50.0]
        filtered, fnames = FeatureSelector.select_top_k(features, names, target, k=2, method="correlation")
        assert len(fnames) == 2
        # strong and also_strong should be picked
        assert "strong" in fnames
        assert "also_strong" in fnames

    def test_mutual_info_method(self):
        features = [
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
        ]
        names = ["informative", "constant"]
        target = [1.0, 2.0, 3.0, 4.0]
        filtered, fnames = FeatureSelector.select_top_k(features, names, target, k=1, method="mutual_info")
        assert len(fnames) == 1
        assert "informative" in fnames

    def test_k_larger_than_features(self):
        features = [[1.0], [2.0], [3.0]]
        names = ["only"]
        target = [1.0, 2.0, 3.0]
        filtered, fnames = FeatureSelector.select_top_k(features, names, target, k=10)
        assert fnames == ["only"]

    def test_empty_input(self):
        filtered, fnames = FeatureSelector.select_top_k([], [], [], k=5)
        assert filtered == []

    def test_single_feature(self):
        features = [[1.0], [2.0], [3.0], [4.0]]
        names = ["a"]
        target = [10.0, 20.0, 30.0, 40.0]
        filtered, fnames = FeatureSelector.select_top_k(features, names, target, k=1)
        assert fnames == ["a"]

    def test_all_identical_features(self):
        features = [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
        names = ["a", "b"]
        target = [1.0, 2.0, 3.0]
        filtered, fnames = FeatureSelector.select_top_k(features, names, target, k=1)
        assert len(fnames) == 1
