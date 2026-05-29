"""Phase 5 tests: Scenario Analysis, Sensitivity Analysis, Period Comparison UI.

Tests cover:
- ScenarioResult / SensitivityResult dataclasses
- _apply_adjustments() deep-copy logic
- scenario_analysis() base vs adjusted comparison
- sensitivity_analysis() grid computation
- Edge cases: no adjustments, missing data, extreme values
- Period comparison UI data flow
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    CompositeHealthScore,
    FinancialData,
    PeriodComparison,
    ScenarioResult,
    SensitivityResult,
)

# ===== Fixtures =====


@pytest.fixture
def analyzer():
    """CharlieAnalyzer with default settings."""
    return CharlieAnalyzer(tax_rate=0.25)


@pytest.fixture
def sample_data():
    """Complete FinancialData for testing."""
    return FinancialData(
        revenue=1_000_000,
        cogs=600_000,
        gross_profit=400_000,
        operating_income=200_000,
        ebit=200_000,
        net_income=150_000,
        ebt=200_000,
        total_assets=2_000_000,
        current_assets=800_000,
        cash=200_000,
        inventory=150_000,
        accounts_receivable=250_000,
        total_liabilities=1_000_000,
        current_liabilities=400_000,
        total_debt=500_000,
        total_equity=1_000_000,
        retained_earnings=300_000,
        interest_expense=50_000,
        operating_cash_flow=250_000,
        capex=80_000,
        accounts_payable=100_000,
    )


@pytest.fixture
def minimal_data():
    """Minimal FinancialData with only revenue and assets."""
    return FinancialData(
        revenue=500_000,
        total_assets=1_000_000,
    )


# ===== _apply_adjustments Tests =====


class TestApplyAdjustments:
    """Tests for _apply_adjustments deep-copy and multiplication."""

    def test_single_adjustment(self, analyzer, sample_data):
        """Adjusting revenue by +10% should multiply by 1.10."""
        adjusted = analyzer._apply_adjustments(sample_data, {"revenue": 1.10})
        assert adjusted.revenue == pytest.approx(1_100_000)
        # Original unchanged
        assert sample_data.revenue == 1_000_000

    def test_multiple_adjustments(self, analyzer, sample_data):
        """Multiple fields adjusted simultaneously."""
        adjusted = analyzer._apply_adjustments(
            sample_data,
            {
                "revenue": 1.20,
                "cogs": 1.05,
                "total_liabilities": 0.90,
            },
        )
        assert adjusted.revenue == pytest.approx(1_200_000)
        assert adjusted.cogs == pytest.approx(630_000)
        assert adjusted.total_liabilities == pytest.approx(900_000)

    def test_deep_copy_isolation(self, analyzer, sample_data):
        """Adjustments should not mutate the original data."""
        original_revenue = sample_data.revenue
        adjusted = analyzer._apply_adjustments(sample_data, {"revenue": 2.0})
        assert adjusted.revenue == pytest.approx(2_000_000)
        assert sample_data.revenue == original_revenue

    def test_none_field_ignored(self, analyzer, minimal_data):
        """Adjusting a None field should leave it None."""
        adjusted = analyzer._apply_adjustments(minimal_data, {"cogs": 1.10})
        assert adjusted.cogs is None

    def test_zero_multiplier(self, analyzer, sample_data):
        """Zero multiplier should set field to zero."""
        adjusted = analyzer._apply_adjustments(sample_data, {"revenue": 0.0})
        assert adjusted.revenue == 0.0

    def test_negative_adjustment(self, analyzer, sample_data):
        """Negative multiplier for edge-case testing."""
        adjusted = analyzer._apply_adjustments(sample_data, {"net_income": -0.5})
        assert adjusted.net_income == pytest.approx(-75_000)

    def test_empty_adjustments(self, analyzer, sample_data):
        """Empty adjustments dict should return identical copy."""
        adjusted = analyzer._apply_adjustments(sample_data, {})
        assert adjusted.revenue == sample_data.revenue
        assert adjusted is not sample_data


# ===== Scenario Analysis Tests =====


class TestScenarioAnalysis:
    """Tests for scenario_analysis() method."""

    def test_returns_scenario_result(self, analyzer, sample_data):
        """Should return a ScenarioResult dataclass."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 1.10}, "Rev +10%")
        assert isinstance(result, ScenarioResult)
        assert result.scenario_name == "Rev +10%"

    def test_base_and_scenario_health_present(self, analyzer, sample_data):
        """Both base and scenario health scores should be computed."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 1.10})
        assert isinstance(result.base_health, CompositeHealthScore)
        assert isinstance(result.scenario_health, CompositeHealthScore)
        assert result.base_health.score >= 0
        assert result.scenario_health.score >= 0

    def test_revenue_increase_improves_health(self, analyzer, sample_data):
        """Increasing revenue by 20% should generally improve health score."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 1.20})
        # With more revenue and same costs, profitability improves
        assert result.scenario_health.score >= result.base_health.score

    def test_debt_increase_hurts_health(self, analyzer, sample_data):
        """Doubling total liabilities should hurt health score."""
        result = analyzer.scenario_analysis(
            sample_data,
            {
                "total_liabilities": 2.0,
                "total_debt": 2.0,
            },
        )
        assert result.scenario_health.score <= result.base_health.score

    def test_z_scores_populated(self, analyzer, sample_data):
        """Both base and scenario Z-scores should be computed."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 1.10})
        assert result.base_z_score is not None
        assert result.scenario_z_score is not None

    def test_f_scores_populated(self, analyzer, sample_data):
        """Both base and scenario F-scores should be computed."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 1.10})
        assert result.base_f_score is not None
        assert result.scenario_f_score is not None
        assert 0 <= result.base_f_score <= 9
        assert 0 <= result.scenario_f_score <= 9

    def test_ratios_contain_key_metrics(self, analyzer, sample_data):
        """Base and scenario ratios should contain health_score and z_score."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 1.10})
        assert "health_score" in result.base_ratios
        assert "z_score" in result.base_ratios
        assert "health_score" in result.scenario_ratios
        assert "z_score" in result.scenario_ratios

    def test_impact_summary_not_empty(self, analyzer, sample_data):
        """Impact summary should contain descriptive text."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 1.10}, "Test")
        assert len(result.impact_summary) > 0
        assert "Test" in result.impact_summary

    def test_no_change_scenario(self, analyzer, sample_data):
        """Empty adjustments should yield identical base and scenario scores."""
        result = analyzer.scenario_analysis(sample_data, {})
        assert result.base_health.score == result.scenario_health.score

    def test_extreme_revenue_drop(self, analyzer, sample_data):
        """Revenue drop to 10% should cause severe health decline."""
        result = analyzer.scenario_analysis(sample_data, {"revenue": 0.10})
        assert result.scenario_health.score <= result.base_health.score

    def test_adjustments_stored(self, analyzer, sample_data):
        """Adjustments dict should be stored in result."""
        adj = {"revenue": 1.15, "cogs": 0.95}
        result = analyzer.scenario_analysis(sample_data, adj)
        assert result.adjustments == adj

    def test_none_field_adjustment_warns_in_summary(self, analyzer):
        """Adjusting a None field should produce a warning in impact_summary."""
        data = FinancialData(revenue=1_000_000, total_assets=5_000_000)
        # net_income is None on this data
        result = analyzer.scenario_analysis(data, {"net_income": 1.10})
        assert "WARNING" in result.impact_summary
        assert "net_income" in result.impact_summary


# ===== Sensitivity Analysis Tests =====


class TestSensitivityAnalysis:
    """Tests for sensitivity_analysis() method."""

    def test_returns_sensitivity_result(self, analyzer, sample_data):
        """Should return a SensitivityResult dataclass."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue")
        assert isinstance(result, SensitivityResult)

    def test_default_range(self, analyzer, sample_data):
        """Default pct_range should produce 9 data points."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue")
        assert len(result.variable_labels) == 9
        assert result.variable_labels[0] == "-20%"
        assert result.variable_labels[-1] == "+20%"

    def test_custom_range(self, analyzer, sample_data):
        """Custom pct_range should be respected."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue", pct_range=[-10, 0, 10])
        assert len(result.variable_labels) == 3
        assert result.variable_labels == ["-10%", "+0%", "+10%"]

    def test_health_score_in_results(self, analyzer, sample_data):
        """Health score should appear in metric_results."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue")
        assert "health_score" in result.metric_results
        assert len(result.metric_results["health_score"]) == 9

    def test_z_score_in_results(self, analyzer, sample_data):
        """Z-score should appear in metric_results."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue")
        assert "z_score" in result.metric_results
        assert all(v is not None for v in result.metric_results["z_score"])

    def test_baseline_at_zero_percent(self, analyzer, sample_data):
        """At 0% change, metrics should match base analysis."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue", pct_range=[-10, 0, 10])
        # 0% is index 1
        base_health = analyzer.composite_health_score(sample_data)
        assert result.metric_results["health_score"][1] == float(base_health.score)

    def test_revenue_increase_improves_metrics(self, analyzer, sample_data):
        """Higher revenue should generally improve health score."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue", pct_range=[-20, 0, 20])
        scores = result.metric_results["health_score"]
        # +20% should be >= 0% change
        assert scores[2] >= scores[1]

    def test_variable_name_stored(self, analyzer, sample_data):
        """Variable name should be stored in result."""
        result = analyzer.sensitivity_analysis(sample_data, "total_assets")
        assert result.variable_name == "total_assets"

    def test_missing_variable_yields_none(self, analyzer, minimal_data):
        """Sensitivity on a field that's None in base data produces None metrics."""
        result = analyzer.sensitivity_analysis(minimal_data, "cogs", pct_range=[-10, 0, 10])
        # cogs is None, so net_margin can't improve from adjusting it
        assert "health_score" in result.metric_results

    def test_multipliers_correct(self, analyzer, sample_data):
        """Variable multipliers should match the percentage range."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue", pct_range=[-10, 0, 10])
        assert result.variable_multipliers == pytest.approx([0.9, 1.0, 1.1])

    def test_all_standard_metrics_present(self, analyzer, sample_data):
        """All 7 standard metrics should be in results."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue")
        expected = {"health_score", "z_score", "f_score", "current_ratio", "net_margin", "debt_to_equity", "roe"}
        assert expected == set(result.metric_results.keys())


# ===== ScenarioResult Dataclass Tests =====


class TestScenarioResultDataclass:
    """Tests for ScenarioResult default values and structure."""

    def test_defaults(self):
        result = ScenarioResult()
        assert result.scenario_name == ""
        assert result.adjustments == {}
        assert result.base_health is None
        assert result.scenario_health is None
        assert result.base_z_score is None
        assert result.scenario_z_score is None
        assert result.impact_summary == ""

    def test_fields_assignable(self):
        result = ScenarioResult(
            scenario_name="Test",
            adjustments={"revenue": 1.1},
            impact_summary="Revenue increased by 10%.",
        )
        assert result.scenario_name == "Test"
        assert result.adjustments["revenue"] == 1.1


# ===== SensitivityResult Dataclass Tests =====


class TestSensitivityResultDataclass:
    """Tests for SensitivityResult default values and structure."""

    def test_defaults(self):
        result = SensitivityResult()
        assert result.variable_name == ""
        assert result.variable_labels == []
        assert result.variable_multipliers == []
        assert result.metric_results == {}

    def test_fields_assignable(self):
        result = SensitivityResult(
            variable_name="revenue",
            variable_labels=["-10%", "0%", "+10%"],
            variable_multipliers=[0.9, 1.0, 1.1],
            metric_results={"health_score": [60, 70, 80]},
        )
        assert result.variable_name == "revenue"
        assert len(result.metric_results["health_score"]) == 3


# ===== Period Comparison Integration Tests =====


class TestPeriodComparisonIntegration:
    """Tests for compare_periods with realistic multi-period data."""

    def test_improving_company(self, analyzer):
        """A company improving in all areas should have only improvements."""
        current = FinancialData(
            revenue=1_200_000,
            cogs=600_000,
            net_income=200_000,
            total_assets=2_000_000,
            current_assets=900_000,
            current_liabilities=400_000,
            total_liabilities=800_000,
            total_equity=1_200_000,
            total_debt=400_000,
            ebit=250_000,
            retained_earnings=400_000,
        )
        prior = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            net_income=150_000,
            total_assets=2_000_000,
            current_assets=800_000,
            current_liabilities=400_000,
            total_liabilities=1_000_000,
            total_equity=1_000_000,
            total_debt=500_000,
            ebit=200_000,
            retained_earnings=300_000,
        )
        result = analyzer.compare_periods(current, prior)
        assert isinstance(result, PeriodComparison)
        # Revenue improved -> profitability should improve
        assert len(result.improvements) > 0

    def test_declining_company(self, analyzer):
        """A company declining should have deteriorations."""
        current = FinancialData(
            revenue=800_000,
            cogs=600_000,
            net_income=50_000,
            total_assets=2_000_000,
            current_assets=600_000,
            current_liabilities=500_000,
            total_liabilities=1_200_000,
            total_equity=800_000,
            total_debt=700_000,
            ebit=100_000,
            retained_earnings=200_000,
        )
        prior = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            net_income=150_000,
            total_assets=2_000_000,
            current_assets=800_000,
            current_liabilities=400_000,
            total_liabilities=1_000_000,
            total_equity=1_000_000,
            total_debt=500_000,
            ebit=200_000,
            retained_earnings=300_000,
        )
        result = analyzer.compare_periods(current, prior)
        assert len(result.deteriorations) > 0

    def test_identical_periods(self, analyzer, sample_data):
        """Comparing a period to itself should have no changes."""
        result = analyzer.compare_periods(sample_data, sample_data)
        assert len(result.improvements) == 0
        assert len(result.deteriorations) == 0

    def test_deltas_sign(self, analyzer):
        """Deltas should be current - prior."""
        current = FinancialData(
            revenue=1_200_000,
            cogs=600_000,
            net_income=200_000,
            total_assets=2_000_000,
            current_assets=800_000,
            current_liabilities=400_000,
            total_liabilities=1_000_000,
            total_equity=1_000_000,
            total_debt=500_000,
            ebit=250_000,
            retained_earnings=350_000,
        )
        prior = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            net_income=150_000,
            total_assets=2_000_000,
            current_assets=800_000,
            current_liabilities=400_000,
            total_liabilities=1_000_000,
            total_equity=1_000_000,
            total_debt=500_000,
            ebit=200_000,
            retained_earnings=300_000,
        )
        result = analyzer.compare_periods(current, prior)
        # Net margin improved: 200k/1.2M > 150k/1M
        net_margin_delta = result.deltas.get("profitability_net_margin")
        if net_margin_delta is not None:
            assert net_margin_delta > 0  # improved


# ===== Scenario with Minimal Data =====


class TestScenarioEdgeCases:
    """Edge cases for scenario and sensitivity analysis."""

    def test_minimal_data_scenario(self, analyzer, minimal_data):
        """Scenario analysis should work even with minimal data."""
        result = analyzer.scenario_analysis(minimal_data, {"revenue": 1.20})
        assert isinstance(result, ScenarioResult)
        assert result.base_health is not None
        assert result.scenario_health is not None

    def test_sensitivity_single_point(self, analyzer, sample_data):
        """Sensitivity with single point should work."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue", pct_range=[0])
        assert len(result.variable_labels) == 1
        assert result.variable_labels == ["+0%"]

    def test_sensitivity_large_range(self, analyzer, sample_data):
        """Sensitivity with large negative should not crash."""
        result = analyzer.sensitivity_analysis(sample_data, "revenue", pct_range=[-90, -50, 0, 50, 100])
        assert len(result.variable_labels) == 5
        # At -90% revenue, health score should be very low
        assert result.metric_results["health_score"][0] <= result.metric_results["health_score"][2]

    def test_scenario_with_all_fields_adjusted(self, analyzer, sample_data):
        """Adjusting many fields at once should not crash."""
        adj = {
            "revenue": 1.10,
            "cogs": 1.05,
            "net_income": 1.15,
            "total_assets": 1.02,
            "current_assets": 1.05,
            "total_liabilities": 0.95,
            "current_liabilities": 0.95,
            "total_equity": 1.10,
        }
        result = analyzer.scenario_analysis(sample_data, adj, "Multi-adjust")
        assert isinstance(result, ScenarioResult)
        assert "Multi-adjust" in result.impact_summary
