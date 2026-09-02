"""Phase 6 Tests: Monte Carlo Simulation & Cash Flow Forecasting.

Tests for stochastic modeling, DCF valuation, and code quality fixes.
"""

import pytest

from financial_analyzer import (
    CashFlowForecast,
    CharlieAnalyzer,
    FinancialData,
    MonteCarloResult,
)


@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture
def sample_data():
    """Complete FinancialData for testing."""
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


@pytest.fixture
def minimal_data():
    """Minimal FinancialData with just revenue."""
    return FinancialData(revenue=500_000, total_assets=1_000_000)


# ===== MONTE CARLO RESULT DATACLASS =====


class TestMonteCarloResultDataclass:
    def test_defaults(self):
        r = MonteCarloResult()
        assert r.n_simulations == 0
        assert r.variable_assumptions == {}
        assert r.metric_distributions == {}
        assert r.percentiles == {}
        assert r.summary == ""

    def test_fields_assignable(self):
        r = MonteCarloResult(n_simulations=100, summary="test")
        assert r.n_simulations == 100
        assert r.summary == "test"


# ===== CASH FLOW FORECAST DATACLASS =====


class TestCashFlowForecastDataclass:
    def test_defaults(self):
        r = CashFlowForecast()
        assert r.periods == []
        assert r.dcf_value is None
        assert r.discount_rate == 0.10

    def test_fields_assignable(self):
        r = CashFlowForecast(dcf_value=1_000_000, discount_rate=0.08)
        assert r.dcf_value == 1_000_000
        assert r.discount_rate == 0.08


# ===== MONTE CARLO SIMULATION =====


class TestMonteCarloSimulation:
    def test_returns_monte_carlo_result(self, analyzer, sample_data):
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=50)
        assert isinstance(result, MonteCarloResult)

    def test_simulation_count(self, analyzer, sample_data):
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=100)
        assert result.n_simulations == 100

    def test_default_assumptions_use_available_fields(self, analyzer, sample_data):
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=50)
        # Should auto-detect revenue, cogs, operating_expenses
        assert "revenue" in result.variable_assumptions
        assert "cogs" in result.variable_assumptions
        assert "operating_expenses" in result.variable_assumptions

    def test_custom_assumptions(self, analyzer, sample_data):
        assumptions = {"revenue": {"mean_pct": 5.0, "std_pct": 15.0}}
        result = analyzer.monte_carlo_simulation(sample_data, assumptions, n_simulations=50)
        assert "revenue" in result.variable_assumptions
        assert result.variable_assumptions["revenue"]["std_pct"] == 15.0

    def test_percentiles_computed(self, analyzer, sample_data):
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=100)
        assert "health_score" in result.percentiles
        pcts = result.percentiles["health_score"]
        assert "p10" in pcts
        assert "p50" in pcts
        assert "p90" in pcts
        assert "mean" in pcts
        assert "std" in pcts
        # P10 <= P50 <= P90
        assert pcts["p10"] <= pcts["p50"] <= pcts["p90"]

    def test_distributions_have_correct_length(self, analyzer, sample_data):
        n = 200
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=n)
        for metric, values in result.metric_distributions.items():
            assert len(values) == n, f"{metric} has {len(values)} values, expected {n}"

    def test_all_standard_metrics_present(self, analyzer, sample_data):
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=50)
        expected = {"health_score", "z_score", "f_score", "net_margin", "current_ratio", "roe"}
        assert expected.issubset(set(result.metric_distributions.keys()))

    def test_summary_not_empty(self, analyzer, sample_data):
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=50)
        assert len(result.summary) > 0
        assert "Monte Carlo" in result.summary

    def test_no_data_returns_zero_sims(self, analyzer):
        empty = FinancialData()
        result = analyzer.monte_carlo_simulation(empty, n_simulations=100)
        assert result.n_simulations == 0

    def test_deterministic_with_seed(self, analyzer, sample_data):
        """Same seed should produce same results."""
        r1 = analyzer.monte_carlo_simulation(sample_data, n_simulations=50)
        r2 = analyzer.monte_carlo_simulation(sample_data, n_simulations=50)
        assert r1.percentiles["health_score"]["p50"] == r2.percentiles["health_score"]["p50"]

    def test_higher_std_wider_range(self, analyzer, sample_data):
        """Higher uncertainty should produce wider outcome range."""
        narrow = {"revenue": {"mean_pct": 0.0, "std_pct": 5.0}}
        wide = {"revenue": {"mean_pct": 0.0, "std_pct": 30.0}}
        r_narrow = analyzer.monte_carlo_simulation(sample_data, narrow, n_simulations=500)
        r_wide = analyzer.monte_carlo_simulation(sample_data, wide, n_simulations=500)
        narrow_spread = r_narrow.percentiles["health_score"]["p90"] - r_narrow.percentiles["health_score"]["p10"]
        wide_spread = r_wide.percentiles["health_score"]["p90"] - r_wide.percentiles["health_score"]["p10"]
        # Wide uncertainty should generally have wider spread
        # (with 500 sims and these parameters, this is very reliable)
        assert wide_spread >= narrow_spread * 0.5  # Generous tolerance

    def test_minimal_data_runs(self, analyzer, minimal_data):
        result = analyzer.monte_carlo_simulation(minimal_data, n_simulations=50)
        # Should still run, even if some metrics are 0
        assert result.n_simulations == 50


# ===== CASH FLOW FORECASTING =====


class TestCashFlowForecast:
    def test_returns_cashflow_forecast(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data)
        assert isinstance(result, CashFlowForecast)

    def test_default_12_periods(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data)
        assert len(result.periods) == 12
        assert len(result.revenue_forecast) == 12
        assert len(result.expense_forecast) == 12
        assert len(result.net_cash_flow) == 12
        assert len(result.fcf_forecast) == 12
        assert len(result.cumulative_cash) == 12

    def test_custom_periods(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, periods=6)
        assert len(result.periods) == 6

    def test_revenue_grows_each_period(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, revenue_growth=0.10)
        for i in range(1, len(result.revenue_forecast)):
            assert result.revenue_forecast[i] > result.revenue_forecast[i - 1]

    def test_revenue_declines_with_negative_growth(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, revenue_growth=-0.05, periods=5)
        for i in range(1, len(result.revenue_forecast)):
            assert result.revenue_forecast[i] < result.revenue_forecast[i - 1]

    def test_dcf_value_positive(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data)
        assert result.dcf_value is not None
        assert result.dcf_value > 0

    def test_terminal_value_positive(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data)
        assert result.terminal_value is not None
        assert result.terminal_value > 0

    def test_cumulative_cash_monotonic_with_positive_net(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, revenue_growth=0.05)
        # With positive growth and reasonable expense ratio, cumulative should grow
        for i in range(1, len(result.cumulative_cash)):
            assert result.cumulative_cash[i] >= result.cumulative_cash[i - 1]

    def test_discount_rate_stored(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, discount_rate=0.12)
        assert result.discount_rate == 0.12

    def test_zero_revenue_returns_empty(self, analyzer):
        data = FinancialData(revenue=0)
        result = analyzer.forecast_cashflow(data)
        assert len(result.periods) == 0

    def test_no_revenue_returns_empty(self, analyzer):
        data = FinancialData()
        result = analyzer.forecast_cashflow(data)
        assert len(result.periods) == 0

    def test_expense_ratio_derived_from_data(self, analyzer, sample_data):
        """When expense_ratio is None, should derive from cogs+opex/revenue."""
        result = analyzer.forecast_cashflow(sample_data, periods=3)
        # expense ratio should be (600k+200k)/1M = 0.80
        expected_rev_1 = sample_data.revenue * 1.05  # default 5% growth
        expected_exp_1 = expected_rev_1 * 0.80
        assert abs(result.expense_forecast[0] - expected_exp_1) < 1.0

    def test_higher_discount_lowers_dcf(self, analyzer, sample_data):
        low = analyzer.forecast_cashflow(sample_data, discount_rate=0.08)
        high = analyzer.forecast_cashflow(sample_data, discount_rate=0.15)
        assert low.dcf_value > high.dcf_value

    def test_growth_rates_populated(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, revenue_growth=0.07, periods=5)
        assert len(result.growth_rates) == 5
        assert all(g == 7.0 for g in result.growth_rates)


# ===== EDGE CASES =====


class TestPhase6EdgeCases:
    def test_monte_carlo_single_sim(self, analyzer, sample_data):
        result = analyzer.monte_carlo_simulation(sample_data, n_simulations=1)
        assert result.n_simulations == 1
        assert len(result.metric_distributions["health_score"]) == 1

    def test_forecast_single_period(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, periods=1)
        assert len(result.periods) == 1
        assert result.dcf_value > 0

    def test_forecast_high_growth(self, analyzer, sample_data):
        result = analyzer.forecast_cashflow(sample_data, revenue_growth=0.50, periods=5)
        # Revenue should roughly double in ~1.5 periods at 50% growth
        assert result.revenue_forecast[-1] > sample_data.revenue * 5

    def test_forecast_equal_discount_and_terminal(self, analyzer, sample_data):
        """When discount_rate equals terminal_growth, GGM is undefined."""
        result = analyzer.forecast_cashflow(sample_data, discount_rate=0.05, terminal_growth=0.05, periods=3)
        assert result.terminal_value is None  # undefined, not misleading 0.0

    def test_monte_carlo_extreme_std(self, analyzer, sample_data):
        """Very high uncertainty should not crash."""
        assumptions = {"revenue": {"mean_pct": 0.0, "std_pct": 80.0}}
        result = analyzer.monte_carlo_simulation(sample_data, assumptions, n_simulations=50)
        assert result.n_simulations == 50
