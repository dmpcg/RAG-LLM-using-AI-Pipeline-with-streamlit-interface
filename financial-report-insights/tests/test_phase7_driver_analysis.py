"""Phase 7 Tests: Driver Analysis, Breakeven, and Covenant Monitoring.

Tests for tornado/driver ranking, breakeven analysis, and KPI threshold monitoring.
"""

import pytest

from financial_analyzer import (
    BreakevenResult,
    CharlieAnalyzer,
    CovenantCheck,
    CovenantMonitorResult,
    FinancialData,
    TornadoDriver,
    TornadoResult,
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
    """Minimal FinancialData."""
    return FinancialData(revenue=500_000, total_assets=1_000_000)


# ===== TORNADO DRIVER DATACLASS =====


class TestTornadoDriverDataclass:
    def test_defaults(self):
        d = TornadoDriver()
        assert d.variable == ""
        assert d.spread == 0.0

    def test_fields_assignable(self):
        d = TornadoDriver(variable="revenue", spread=5.0)
        assert d.variable == "revenue"
        assert d.spread == 5.0


class TestTornadoResultDataclass:
    def test_defaults(self):
        r = TornadoResult()
        assert r.target_metric == ""
        assert r.drivers == []
        assert r.top_driver == ""

    def test_fields_assignable(self):
        r = TornadoResult(target_metric="health_score", top_driver="revenue")
        assert r.target_metric == "health_score"


# ===== BREAKEVEN DATACLASS =====


class TestBreakevenResultDataclass:
    def test_defaults(self):
        r = BreakevenResult()
        assert r.breakeven_revenue is None
        assert r.margin_of_safety is None

    def test_fields_assignable(self):
        r = BreakevenResult(breakeven_revenue=500_000, margin_of_safety=0.5)
        assert r.breakeven_revenue == 500_000


# ===== COVENANT DATACLASSES =====


class TestCovenantCheckDataclass:
    def test_defaults(self):
        c = CovenantCheck()
        assert c.status == "unknown"
        assert c.direction == "above"

    def test_fields_assignable(self):
        c = CovenantCheck(name="CR", status="pass", headroom=0.5)
        assert c.name == "CR"
        assert c.headroom == 0.5


class TestCovenantMonitorResultDataclass:
    def test_defaults(self):
        r = CovenantMonitorResult()
        assert r.checks == []
        assert r.passes == 0
        assert r.breaches == 0


# ===== TORNADO ANALYSIS =====


class TestTornadoAnalysis:
    def test_returns_tornado_result(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data)
        assert isinstance(result, TornadoResult)

    def test_auto_detects_variables(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data)
        assert len(result.drivers) > 0
        # Should include revenue, cogs, etc.
        var_names = [d.variable for d in result.drivers]
        assert "revenue" in var_names
        assert "cogs" in var_names

    def test_custom_variables(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data, variables=["revenue", "cogs"])
        assert len(result.drivers) == 2
        var_names = [d.variable for d in result.drivers]
        assert "revenue" in var_names
        assert "cogs" in var_names

    def test_drivers_sorted_by_spread(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data)
        for i in range(len(result.drivers) - 1):
            assert result.drivers[i].spread >= result.drivers[i + 1].spread

    def test_top_driver_matches_first(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data)
        assert result.top_driver == result.drivers[0].variable

    def test_base_metric_value_populated(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data, target_metric="health_score")
        assert result.base_metric_value > 0

    def test_different_targets(self, analyzer, sample_data):
        """All supported target metrics should work."""
        for target in ["health_score", "z_score", "f_score", "net_margin", "current_ratio", "roe", "debt_to_equity"]:
            result = analyzer.tornado_analysis(sample_data, target_metric=target)
            assert isinstance(result, TornadoResult)
            assert result.target_metric == target

    def test_spread_is_positive(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data)
        for d in result.drivers:
            assert d.spread >= 0

    def test_higher_swing_wider_spread(self, analyzer, sample_data):
        """Wider swing should generally produce wider spreads."""
        r_small = analyzer.tornado_analysis(sample_data, variables=["revenue"], pct_swing=5.0)
        r_large = analyzer.tornado_analysis(sample_data, variables=["revenue"], pct_swing=20.0)
        assert r_large.drivers[0].spread >= r_small.drivers[0].spread

    def test_empty_data_returns_empty(self, analyzer):
        empty = FinancialData()
        result = analyzer.tornado_analysis(empty)
        assert result.drivers == []
        assert result.top_driver == ""

    def test_minimal_data(self, analyzer, minimal_data):
        result = analyzer.tornado_analysis(minimal_data)
        assert isinstance(result, TornadoResult)
        # Should still find at least revenue and total_assets
        var_names = [d.variable for d in result.drivers]
        assert "revenue" in var_names


# ===== BREAKEVEN ANALYSIS =====


class TestBreakevenAnalysis:
    def test_returns_breakeven_result(self, analyzer, sample_data):
        result = analyzer.breakeven_analysis(sample_data)
        assert isinstance(result, BreakevenResult)

    def test_breakeven_revenue_positive(self, analyzer, sample_data):
        result = analyzer.breakeven_analysis(sample_data)
        assert result.breakeven_revenue is not None
        assert result.breakeven_revenue > 0

    def test_current_revenue_matches(self, analyzer, sample_data):
        result = analyzer.breakeven_analysis(sample_data)
        assert result.current_revenue == sample_data.revenue

    def test_margin_of_safety_positive_when_profitable(self, analyzer, sample_data):
        """Sample data is profitable, so margin of safety should be positive."""
        result = analyzer.breakeven_analysis(sample_data)
        assert result.margin_of_safety is not None
        assert result.margin_of_safety > 0

    def test_variable_cost_ratio(self, analyzer, sample_data):
        result = analyzer.breakeven_analysis(sample_data)
        # COGS/Revenue = 600k/1M = 0.6
        assert result.variable_cost_ratio == 0.6

    def test_contribution_margin_ratio(self, analyzer, sample_data):
        result = analyzer.breakeven_analysis(sample_data)
        # 1 - 0.6 = 0.4
        assert result.contribution_margin_ratio == 0.4

    def test_fixed_costs_includes_interest(self, analyzer, sample_data):
        result = analyzer.breakeven_analysis(sample_data)
        # Fixed costs = opex + interest = 200k + 30k = 230k
        assert result.fixed_costs == 230_000

    def test_breakeven_formula(self, analyzer, sample_data):
        """Breakeven = fixed_costs / contribution_margin_ratio."""
        result = analyzer.breakeven_analysis(sample_data)
        expected = 230_000 / 0.4  # 575,000
        assert abs(result.breakeven_revenue - expected) < 1.0

    def test_zero_revenue_returns_empty(self, analyzer):
        data = FinancialData(revenue=0)
        result = analyzer.breakeven_analysis(data)
        assert result.breakeven_revenue is None

    def test_no_revenue_returns_empty(self, analyzer):
        data = FinancialData()
        result = analyzer.breakeven_analysis(data)
        assert result.breakeven_revenue is None

    def test_100pct_variable_costs(self, analyzer):
        """When COGS equals revenue, contribution margin is 0."""
        data = FinancialData(revenue=100_000, cogs=100_000, operating_expenses=10_000)
        result = analyzer.breakeven_analysis(data)
        assert result.contribution_margin_ratio == 0.0
        assert result.breakeven_revenue is None

    def test_no_cogs_all_fixed(self, analyzer):
        """When no COGS, all costs are fixed."""
        data = FinancialData(revenue=100_000, cogs=0, operating_expenses=50_000)
        result = analyzer.breakeven_analysis(data)
        assert result.variable_cost_ratio == 0.0
        assert result.contribution_margin_ratio == 1.0
        assert result.breakeven_revenue == 50_000


# ===== COVENANT MONITORING =====


class TestCovenantMonitor:
    def test_returns_covenant_monitor_result(self, analyzer, sample_data):
        result = analyzer.covenant_monitor(sample_data)
        assert isinstance(result, CovenantMonitorResult)

    def test_default_covenants_count(self, analyzer, sample_data):
        result = analyzer.covenant_monitor(sample_data)
        assert len(result.checks) == 5  # 5 default covenants

    def test_checks_sum_to_total(self, analyzer, sample_data):
        result = analyzer.covenant_monitor(sample_data)
        total_known = result.passes + result.warnings + result.breaches
        unknown = sum(1 for c in result.checks if c.status == "unknown")
        assert total_known + unknown == len(result.checks)

    def test_custom_covenants(self, analyzer, sample_data):
        custom = [
            {"name": "Min CR", "metric": "current_ratio", "threshold": 1.0, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert len(result.checks) == 1
        # current_ratio = 500k/200k = 2.5, threshold 1.0 => pass
        assert result.checks[0].status == "pass"

    def test_breach_detection(self, analyzer, sample_data):
        """Set a threshold that will definitely be breached."""
        custom = [
            {"name": "Impossible CR", "metric": "current_ratio", "threshold": 100.0, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.breaches == 1
        assert result.checks[0].status == "breach"

    def test_below_direction(self, analyzer, sample_data):
        """Direction 'below' means passing if value <= threshold."""
        custom = [
            {"name": "Max D/E", "metric": "debt_to_equity", "threshold": 2.0, "direction": "below"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        # D/E = 800k/1.2M = 0.667, well below 2.0 => pass
        assert result.checks[0].status == "pass"

    def test_below_breach(self, analyzer, sample_data):
        """When value exceeds threshold with 'below' direction, should breach."""
        custom = [
            {"name": "Max D/E", "metric": "debt_to_equity", "threshold": 0.1, "direction": "below"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.checks[0].status == "breach"

    def test_headroom_positive_when_passing(self, analyzer, sample_data):
        custom = [
            {"name": "Min CR", "metric": "current_ratio", "threshold": 1.0, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.checks[0].headroom > 0

    def test_headroom_negative_when_breaching(self, analyzer, sample_data):
        custom = [
            {"name": "Min CR", "metric": "current_ratio", "threshold": 100.0, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.checks[0].headroom < 0

    def test_unknown_metric(self, analyzer, sample_data):
        custom = [
            {"name": "Bad Metric", "metric": "nonexistent_metric", "threshold": 1.0, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.checks[0].status == "unknown"
        assert result.checks[0].current_value is None

    def test_summary_contains_breach_count(self, analyzer, sample_data):
        custom = [
            {"name": "Fail", "metric": "current_ratio", "threshold": 100.0, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert "BREACH" in result.summary

    def test_summary_passing_only(self, analyzer, sample_data):
        custom = [
            {"name": "Easy", "metric": "current_ratio", "threshold": 0.1, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert "passing" in result.summary

    def test_dscr_metric(self, analyzer, sample_data):
        """DSCR = EBITDA / interest_expense = 250k / 30k = 8.33."""
        custom = [
            {"name": "DSCR", "metric": "dscr", "threshold": 1.25, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.checks[0].status == "pass"
        assert result.checks[0].current_value > 8.0

    def test_health_score_metric(self, analyzer, sample_data):
        custom = [
            {"name": "Health", "metric": "health_score", "threshold": 20.0, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.checks[0].status == "pass"
        assert result.checks[0].current_value > 20.0

    def test_empty_data(self, analyzer):
        empty = FinancialData()
        result = analyzer.covenant_monitor(empty)
        # Most metrics will be unknown
        assert isinstance(result, CovenantMonitorResult)

    def test_warning_zone(self, analyzer, sample_data):
        """Test that warning is triggered when close to threshold.
        Current ratio = 2.5, threshold 2.3, warning zone = 2.3 + 10% = 2.53.
        Since 2.5 < 2.53, should be warning.
        """
        custom = [
            {"name": "Near CR", "metric": "current_ratio", "threshold": 2.3, "direction": "above"},
        ]
        result = analyzer.covenant_monitor(sample_data, custom)
        assert result.checks[0].status == "warning"
        assert result.warnings == 1


# ===== EDGE CASES =====


class TestPhase7EdgeCases:
    def test_tornado_single_variable(self, analyzer, sample_data):
        result = analyzer.tornado_analysis(sample_data, variables=["revenue"])
        assert len(result.drivers) == 1
        assert result.top_driver == "revenue"

    def test_breakeven_no_opex(self, analyzer):
        """With no opex, fixed costs are just interest."""
        data = FinancialData(revenue=100_000, cogs=40_000, interest_expense=10_000)
        result = analyzer.breakeven_analysis(data)
        assert result.fixed_costs == 10_000
        # contribution margin = 1 - 0.4 = 0.6
        assert result.breakeven_revenue == pytest.approx(10_000 / 0.6, abs=1.0)

    def test_covenant_empty_list(self, analyzer, sample_data):
        result = analyzer.covenant_monitor(sample_data, covenants=[])
        assert len(result.checks) == 0
        assert result.summary == "No covenants checked."

    def test_tornado_all_metrics_work(self, analyzer, sample_data):
        """Ensure no target metric crashes."""
        for metric in ["health_score", "z_score", "f_score", "net_margin", "current_ratio", "roe", "debt_to_equity"]:
            result = analyzer.tornado_analysis(sample_data, target_metric=metric, variables=["revenue"])
            assert result.drivers[0].spread >= 0
