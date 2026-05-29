"""
Edge-case tests for financial ratio calculations.
Covers: negative values, zero denominators, single data points, extreme values, CAGR guards.
"""

import pandas as pd
import pytest

from financial_analyzer import CharlieAnalyzer, FinancialData, safe_divide

# ============================================================
# safe_divide helper
# ============================================================


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == pytest.approx(5.0)

    def test_zero_denominator_returns_default(self):
        assert safe_divide(10, 0) is None
        assert safe_divide(10, 0, default=0.0) == 0.0

    def test_none_numerator(self):
        assert safe_divide(None, 5) is None

    def test_none_denominator(self):
        assert safe_divide(5, None) is None

    def test_both_none(self):
        assert safe_divide(None, None) is None

    def test_near_zero_denominator(self):
        assert safe_divide(10, 1e-15) is None

    def test_negative_values(self):
        assert safe_divide(-10, 2) == pytest.approx(-5.0)
        assert safe_divide(10, -2) == pytest.approx(-5.0)

    def test_custom_default(self):
        assert safe_divide(0, 0, default=42.0) == 42.0


# ============================================================
# Liquidity ratios edge cases
# ============================================================


class TestLiquidityEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_all_none_data(self, analyzer):
        data = FinancialData()
        ratios = analyzer.calculate_liquidity_ratios(data)
        assert ratios["current_ratio"] is None
        assert ratios["quick_ratio"] is None
        assert ratios["cash_ratio"] is None

    def test_zero_current_liabilities(self, analyzer):
        data = FinancialData(current_assets=100000, current_liabilities=0, inventory=20000, cash=30000)
        ratios = analyzer.calculate_liquidity_ratios(data)
        assert ratios["current_ratio"] is None
        assert ratios["quick_ratio"] is None
        assert ratios["cash_ratio"] is None

    def test_negative_current_liabilities(self, analyzer):
        data = FinancialData(current_assets=100000, current_liabilities=-50000, inventory=20000, cash=30000)
        ratios = analyzer.calculate_liquidity_ratios(data)
        assert ratios["current_ratio"] is not None  # negative denominator is valid math
        assert ratios["current_ratio"] == pytest.approx(-2.0)

    def test_very_large_values(self, analyzer):
        data = FinancialData(current_assets=5e12, current_liabilities=2.5e12, inventory=1e12, cash=1e12)
        ratios = analyzer.calculate_liquidity_ratios(data)
        assert ratios["current_ratio"] == pytest.approx(2.0)

    def test_very_small_values(self, analyzer):
        data = FinancialData(current_assets=0.50, current_liabilities=0.25, inventory=0.10, cash=0.15)
        ratios = analyzer.calculate_liquidity_ratios(data)
        assert ratios["current_ratio"] == pytest.approx(2.0)


# ============================================================
# Profitability ratios edge cases
# ============================================================


class TestProfitabilityEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_negative_net_income(self, analyzer):
        data = FinancialData(net_income=-500000, revenue=1000000, total_equity=1000000, total_assets=2000000)
        ratios = analyzer.calculate_profitability_ratios(data)
        assert ratios["net_margin"] == pytest.approx(-0.5)
        assert ratios["roe"] == pytest.approx(-0.5)
        assert ratios["roa"] == pytest.approx(-0.25)

    def test_negative_equity(self, analyzer):
        data = FinancialData(net_income=100000, total_equity=-200000, revenue=1000000)
        ratios = analyzer.calculate_profitability_ratios(data)
        assert ratios["roe"] == pytest.approx(-0.5)

    def test_zero_revenue(self, analyzer):
        data = FinancialData(revenue=0, cogs=0, net_income=0, operating_income=0)
        ratios = analyzer.calculate_profitability_ratios(data)
        assert ratios["gross_margin"] is None
        assert ratios["operating_margin"] is None
        assert ratios["net_margin"] is None

    def test_roic_zero_invested_capital(self, analyzer):
        data = FinancialData(operating_income=100000, total_equity=0, total_debt=0)
        ratios = analyzer.calculate_profitability_ratios(data)
        assert ratios["roic"] is None

    def test_gross_margin_via_gross_profit(self, analyzer):
        data = FinancialData(gross_profit=600000, revenue=1000000)
        ratios = analyzer.calculate_profitability_ratios(data)
        assert ratios["gross_margin"] == pytest.approx(0.6)

    def test_billion_dollar_values(self, analyzer):
        data = FinancialData(revenue=5e9, cogs=3e9, net_income=1e9, total_equity=8e9)
        ratios = analyzer.calculate_profitability_ratios(data)
        assert ratios["gross_margin"] == pytest.approx(0.4)
        assert ratios["roe"] == pytest.approx(0.125)


# ============================================================
# Leverage ratios edge cases
# ============================================================


class TestLeverageEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_zero_equity(self, analyzer):
        data = FinancialData(total_debt=500000, total_equity=0, total_assets=1000000)
        ratios = analyzer.calculate_leverage_ratios(data)
        assert ratios["debt_to_equity"] is None

    def test_zero_interest_expense(self, analyzer):
        data = FinancialData(ebit=300000, interest_expense=0)
        ratios = analyzer.calculate_leverage_ratios(data)
        assert ratios["interest_coverage"] is None

    def test_interest_coverage_via_operating_income(self, analyzer):
        data = FinancialData(operating_income=200000, interest_expense=50000)
        ratios = analyzer.calculate_leverage_ratios(data)
        assert ratios["interest_coverage"] == pytest.approx(4.0)


# ============================================================
# Efficiency ratios edge cases
# ============================================================


class TestEfficiencyEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_zero_inventory(self, analyzer):
        data = FinancialData(cogs=400000, inventory=0, avg_inventory=None)
        ratios = analyzer.calculate_efficiency_ratios(data)
        assert ratios["inventory_turnover"] is None

    def test_all_none(self, analyzer):
        data = FinancialData()
        ratios = analyzer.calculate_efficiency_ratios(data)
        assert all(v is None for v in ratios.values())


# ============================================================
# Trend analysis edge cases
# ============================================================


class TestTrendEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_single_data_point(self, analyzer):
        df = pd.DataFrame({"revenue": [100]})
        trend = analyzer.analyze_trends(df, "revenue")
        assert trend.cagr is None
        assert trend.mom_growth is None
        assert trend.trend_direction == "insufficient_data"

    def test_two_data_points(self, analyzer):
        df = pd.DataFrame({"revenue": [100, 120]})
        trend = analyzer.analyze_trends(df, "revenue")
        assert trend.mom_growth == pytest.approx(0.2)
        assert trend.cagr == pytest.approx(0.2)

    def test_cagr_negative_start(self, analyzer):
        df = pd.DataFrame({"revenue": [-100, 200]})
        trend = analyzer.analyze_trends(df, "revenue")
        assert trend.cagr is None  # negative start should be guarded

    def test_cagr_zero_start(self, analyzer):
        df = pd.DataFrame({"revenue": [0, 200]})
        trend = analyzer.analyze_trends(df, "revenue")
        assert trend.cagr is None

    def test_all_zeros(self, analyzer):
        df = pd.DataFrame({"revenue": [0, 0, 0, 0]})
        trend = analyzer.analyze_trends(df, "revenue")
        assert trend.mom_growth is None
        assert trend.cagr is None

    def test_constant_values(self, analyzer):
        df = pd.DataFrame({"revenue": [100, 100, 100, 100]})
        trend = analyzer.analyze_trends(df, "revenue")
        assert trend.mom_growth == pytest.approx(0.0)
        assert trend.trend_direction == "stable"


# ============================================================
# Forecast edge cases
# ============================================================


class TestForecastEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_empty_historical(self, analyzer):
        result = analyzer.forecast_simple([], periods=3)
        assert result.forecasted_values == []

    def test_single_value(self, analyzer):
        result = analyzer.forecast_simple([100], periods=3)
        assert result.forecasted_values == []

    def test_growth_rate_with_zero_values(self, analyzer):
        result = analyzer.forecast_simple([0, 0, 100], periods=2, method="growth_rate")
        assert len(result.forecasted_values) == 2


# ============================================================
# Cash flow / working capital edge cases
# ============================================================


class TestCashFlowEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_zero_revenue_dso(self, analyzer):
        data = FinancialData(accounts_receivable=100000, revenue=0)
        result = analyzer.analyze_cash_flow(data)
        assert result.dso is None

    def test_zero_cogs_dio(self, analyzer):
        data = FinancialData(inventory=50000, cogs=0)
        result = analyzer.analyze_cash_flow(data)
        assert result.dio is None

    def test_working_capital_zero_liabilities(self, analyzer):
        data = FinancialData(current_assets=100000, current_liabilities=0, revenue=500000)
        result = analyzer.analyze_working_capital(data)
        assert result.net_working_capital == 100000
        assert result.working_capital_ratio is None  # zero denominator


# ============================================================
# Variance calculation edge cases
# ============================================================


class TestVarianceEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return CharlieAnalyzer()

    def test_zero_budget(self, analyzer):
        result = analyzer.calculate_variance(100, 0, "Revenue")
        assert result.variance == 100
        assert result.variance_percent is None  # None instead of inf (JSON-safe)

    def test_both_zero(self, analyzer):
        result = analyzer.calculate_variance(0, 0, "Expense")
        assert result.variance == 0
        assert result.variance_percent == 0.0
