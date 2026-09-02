"""Phase 9 Tests: Regression Forecasting & Industry Benchmarking.

Tests for regression_forecast() and industry_benchmark() methods.
"""

import pytest

from financial_analyzer import (
    BenchmarkComparison,
    CharlieAnalyzer,
    FinancialData,
    IndustryBenchmarkResult,
    RegressionForecast,
)


@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture
def sample_data():
    """Complete FinancialData for benchmarking."""
    return FinancialData(
        revenue=1_000_000,
        cogs=600_000,
        gross_profit=400_000,
        operating_expenses=200_000,
        operating_income=200_000,
        net_income=150_000,
        ebit=200_000,
        ebitda=250_000,
        total_assets=2_000_000,
        total_liabilities=800_000,
        total_equity=1_200_000,
        current_assets=500_000,
        current_liabilities=200_000,
        inventory=100_000,
        accounts_receivable=150_000,
        accounts_payable=80_000,
        total_debt=400_000,
        retained_earnings=600_000,
        depreciation=50_000,
        interest_expense=30_000,
        operating_cash_flow=220_000,
        investing_cash_flow=-80_000,
        financing_cash_flow=-50_000,
        capex=80_000,
    )


# ===== REGRESSION FORECAST DATACLASS =====


class TestRegressionForecastDataclass:
    def test_defaults(self):
        r = RegressionForecast()
        assert r.metric_name == ""
        assert r.forecast_values == []
        assert r.r_squared is None
        assert r.method == "linear"

    def test_fields_assignable(self):
        r = RegressionForecast(metric_name="revenue", method="exponential", r_squared=0.95)
        assert r.metric_name == "revenue"
        assert r.method == "exponential"
        assert r.r_squared == 0.95


# ===== BENCHMARK COMPARISON DATACLASS =====


class TestBenchmarkComparisonDataclass:
    def test_defaults(self):
        b = BenchmarkComparison()
        assert b.metric_name == ""
        assert b.rating == ""
        assert b.percentile_rank is None

    def test_fields_assignable(self):
        b = BenchmarkComparison(metric_name="roe", company_value=0.15, rating="above average")
        assert b.metric_name == "roe"
        assert b.company_value == 0.15


# ===== INDUSTRY BENCHMARK RESULT DATACLASS =====


class TestIndustryBenchmarkResultDataclass:
    def test_defaults(self):
        r = IndustryBenchmarkResult()
        assert r.industry_name == ""
        assert r.comparisons == []
        assert r.overall_percentile is None


# ===== REGRESSION FORECAST =====


class TestRegressionForecast:
    def test_returns_regression_forecast(self, analyzer):
        values = [100, 110, 120, 130, 140]
        result = analyzer.regression_forecast(values)
        assert isinstance(result, RegressionForecast)

    def test_linear_forecast_values(self, analyzer):
        values = [100, 200, 300, 400, 500]
        result = analyzer.regression_forecast(values, periods_ahead=3, method="linear")
        assert len(result.forecast_values) == 3
        # Perfect linear: next values should be ~600, 700, 800
        assert abs(result.forecast_values[0] - 600) < 1
        assert abs(result.forecast_values[1] - 700) < 1
        assert abs(result.forecast_values[2] - 800) < 1

    def test_linear_r_squared_perfect(self, analyzer):
        values = [100, 200, 300, 400, 500]
        result = analyzer.regression_forecast(values, method="linear")
        assert result.r_squared is not None
        assert result.r_squared > 0.99

    def test_linear_slope(self, analyzer):
        values = [100, 200, 300, 400, 500]
        result = analyzer.regression_forecast(values, method="linear")
        assert result.slope is not None
        assert abs(result.slope - 100) < 1

    def test_linear_intercept(self, analyzer):
        values = [100, 200, 300, 400, 500]
        result = analyzer.regression_forecast(values, method="linear")
        assert result.intercept is not None
        assert abs(result.intercept - 100) < 1

    def test_exponential_forecast(self, analyzer):
        # Doubling each period: 100, 200, 400, 800, 1600
        values = [100, 200, 400, 800, 1600]
        result = analyzer.regression_forecast(values, periods_ahead=2, method="exponential")
        assert len(result.forecast_values) == 2
        # Next should be ~3200, 6400
        assert result.forecast_values[0] > 2000
        assert result.forecast_values[1] > result.forecast_values[0]

    def test_exponential_r_squared(self, analyzer):
        values = [100, 200, 400, 800, 1600]
        result = analyzer.regression_forecast(values, method="exponential")
        assert result.r_squared is not None
        assert result.r_squared > 0.99

    def test_polynomial_forecast(self, analyzer):
        # Quadratic: 1, 4, 9, 16, 25
        values = [1, 4, 9, 16, 25]
        result = analyzer.regression_forecast(values, periods_ahead=2, method="polynomial")
        assert len(result.forecast_values) == 2
        # Next: 36, 49
        assert result.forecast_values[0] > 30

    def test_confidence_bands(self, analyzer):
        values = [100, 110, 120, 130, 140]
        result = analyzer.regression_forecast(values, periods_ahead=3)
        assert len(result.confidence_upper) == 3
        assert len(result.confidence_lower) == 3
        # Upper > forecast > lower
        for i in range(3):
            assert result.confidence_upper[i] >= result.forecast_values[i]
            assert result.confidence_lower[i] <= result.forecast_values[i]

    def test_too_few_values(self, analyzer):
        result = analyzer.regression_forecast([100, 200], periods_ahead=3)
        assert result.forecast_values == []
        assert result.r_squared is None

    def test_noisy_data_lower_r_squared(self, analyzer):
        values = [100, 130, 95, 140, 110, 150, 100]
        result = analyzer.regression_forecast(values, method="linear")
        # Noisy data should have lower R²
        assert result.r_squared is not None
        assert result.r_squared < 0.7

    def test_metric_name_preserved(self, analyzer):
        result = analyzer.regression_forecast([10, 20, 30], metric_name="revenue")
        assert result.metric_name == "revenue"

    def test_periods_ahead_stored(self, analyzer):
        result = analyzer.regression_forecast([10, 20, 30], periods_ahead=5)
        assert result.periods_ahead == 5

    def test_method_stored(self, analyzer):
        result = analyzer.regression_forecast([10, 20, 30], method="polynomial")
        assert result.method == "polynomial"

    def test_historical_values_stored(self, analyzer):
        vals = [10, 20, 30, 40]
        result = analyzer.regression_forecast(vals)
        assert result.historical_values == vals

    def test_flat_data(self, analyzer):
        values = [100, 100, 100, 100, 100]
        result = analyzer.regression_forecast(values, periods_ahead=3)
        # Slope should be ~0, forecast should be ~100
        assert abs(result.slope) < 1e-6
        for v in result.forecast_values:
            assert abs(v - 100) < 1

    def test_exponential_with_zeros_falls_to_linear(self, analyzer):
        """Exponential with zeros falls back to linear (not all y > 0)."""
        values = [0, 10, 20, 30, 40]
        result = analyzer.regression_forecast(values, method="exponential")
        # Falls back to linear since log(0) is undefined
        assert len(result.forecast_values) == 3  # default periods_ahead
        assert result.forecast_values[0] > 40  # should continue upward


# ===== INDUSTRY BENCHMARK =====


class TestIndustryBenchmark:
    def test_returns_benchmark_result(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        assert isinstance(result, IndustryBenchmarkResult)

    def test_has_comparisons(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        assert len(result.comparisons) > 0

    def test_default_industry_general(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        assert result.industry_name == "general"

    def test_technology_industry(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data, industry="technology")
        assert result.industry_name == "technology"
        assert len(result.comparisons) > 0

    def test_manufacturing_industry(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data, industry="manufacturing")
        assert result.industry_name == "manufacturing"

    def test_retail_industry(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data, industry="retail")
        assert result.industry_name == "retail"

    def test_healthcare_industry(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data, industry="healthcare")
        assert result.industry_name == "healthcare"

    def test_unknown_industry_falls_back(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data, industry="unicorns")
        # Falls back to "general"
        assert len(result.comparisons) > 0

    def test_overall_percentile(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        assert result.overall_percentile is not None
        assert 1 <= result.overall_percentile <= 99

    def test_summary_not_empty(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        assert len(result.summary) > 0

    def test_comparison_has_company_value(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        for c in result.comparisons:
            assert c.company_value is not None

    def test_comparison_has_industry_median(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        for c in result.comparisons:
            assert c.industry_median is not None

    def test_comparison_has_percentile(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        for c in result.comparisons:
            assert c.percentile_rank is not None
            assert 1 <= c.percentile_rank <= 99

    def test_comparison_has_rating(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        for c in result.comparisons:
            assert c.rating in ("above average", "average", "below average")

    def test_strong_company_mostly_above(self, analyzer, sample_data):
        """Sample data is strong; should have above-average ratings."""
        result = analyzer.industry_benchmark(sample_data)
        above = sum(1 for c in result.comparisons if c.rating == "above average")
        assert above >= 2  # At least some above average

    def test_empty_data_no_crash(self, analyzer):
        result = analyzer.industry_benchmark(FinancialData())
        assert isinstance(result, IndustryBenchmarkResult)
        assert result.summary == "Insufficient data for industry comparison."

    def test_gross_margin_present(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        metrics = [c.metric_name for c in result.comparisons]
        assert "gross_margin" in metrics

    def test_current_ratio_present(self, analyzer, sample_data):
        result = analyzer.industry_benchmark(sample_data)
        metrics = [c.metric_name for c in result.comparisons]
        assert "current_ratio" in metrics

    def test_debt_to_equity_inverted(self, analyzer):
        """Lower D/E should rank higher (inverted metric)."""
        low_de = FinancialData(
            revenue=100_000,
            net_income=10_000,
            total_assets=200_000,
            total_liabilities=40_000,
            total_equity=160_000,
            total_debt=30_000,
            current_assets=80_000,
            current_liabilities=30_000,
        )
        high_de = FinancialData(
            revenue=100_000,
            net_income=10_000,
            total_assets=200_000,
            total_liabilities=160_000,
            total_equity=40_000,
            total_debt=150_000,
            current_assets=80_000,
            current_liabilities=30_000,
        )
        r_low = analyzer.industry_benchmark(low_de)
        r_high = analyzer.industry_benchmark(high_de)
        de_low = next((c for c in r_low.comparisons if c.metric_name == "debt_to_equity"), None)
        de_high = next((c for c in r_high.comparisons if c.metric_name == "debt_to_equity"), None)
        if de_low and de_high:
            assert de_low.percentile_rank > de_high.percentile_rank


# ===== EDGE CASES =====


class TestPhase9EdgeCases:
    def test_single_value_no_forecast(self, analyzer):
        result = analyzer.regression_forecast([100])
        assert result.forecast_values == []

    def test_three_values_minimum(self, analyzer):
        result = analyzer.regression_forecast([10, 20, 30], periods_ahead=2)
        assert len(result.forecast_values) == 2

    def test_negative_values_linear(self, analyzer):
        values = [-10, -5, 0, 5, 10]
        result = analyzer.regression_forecast(values, periods_ahead=2)
        assert len(result.forecast_values) == 2
        assert result.forecast_values[0] > 10

    def test_benchmark_partial_data(self, analyzer):
        """Only has revenue and assets - should still produce some comparisons."""
        data = FinancialData(
            revenue=100_000,
            total_assets=200_000,
            total_equity=100_000,
            total_liabilities=100_000,
        )
        result = analyzer.industry_benchmark(data)
        # Should have at least debt_to_equity and asset_turnover
        assert len(result.comparisons) >= 1

    def test_all_industries_have_benchmarks(self, analyzer):
        """Every configured industry should produce results."""
        data = FinancialData(
            revenue=100_000,
            net_income=10_000,
            ebit=15_000,
            total_assets=200_000,
            total_liabilities=80_000,
            total_equity=120_000,
            current_assets=50_000,
            current_liabilities=30_000,
            cogs=60_000,
            gross_profit=40_000,
        )
        for ind in ["general", "technology", "manufacturing", "retail", "healthcare"]:
            result = analyzer.industry_benchmark(data, industry=ind)
            assert len(result.comparisons) > 0, f"No comparisons for {ind}"
