"""Tests for Phase 3 production features.

Covers:
- Composite Health Score (weighted 0-100, letter grades)
- Multi-Period Comparison (deltas, improvements, deteriorations)
- Financial Report Generation (sections, executive summary)
- analyze() includes composite_health
"""

from datetime import datetime

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    CompositeHealthScore,
    FinancialData,
    FinancialReport,
    PeriodComparison,
)


@pytest.fixture
def analyzer():
    return CharlieAnalyzer(tax_rate=0.25)


@pytest.fixture
def healthy_company():
    """A financially healthy company."""
    return FinancialData(
        revenue=10_000_000,
        cogs=6_000_000,
        gross_profit=4_000_000,
        operating_income=2_000_000,
        ebit=2_000_000,
        ebt=1_800_000,
        interest_expense=200_000,
        net_income=1_350_000,
        total_assets=20_000_000,
        current_assets=8_000_000,
        cash=3_000_000,
        inventory=2_000_000,
        accounts_receivable=2_500_000,
        total_liabilities=10_000_000,
        current_liabilities=4_000_000,
        accounts_payable=1_500_000,
        total_debt=6_000_000,
        total_equity=10_000_000,
        retained_earnings=5_000_000,
        operating_cash_flow=2_500_000,
        capex=500_000,
    )


@pytest.fixture
def distressed_company():
    """A company in financial distress."""
    return FinancialData(
        revenue=2_000_000,
        cogs=1_800_000,
        gross_profit=200_000,
        operating_income=-100_000,
        ebit=-100_000,
        ebt=-200_000,
        interest_expense=100_000,
        net_income=-250_000,
        total_assets=5_000_000,
        current_assets=1_000_000,
        cash=200_000,
        inventory=500_000,
        accounts_receivable=250_000,
        total_liabilities=6_000_000,
        current_liabilities=3_000_000,
        accounts_payable=800_000,
        total_debt=4_000_000,
        total_equity=-1_000_000,
        retained_earnings=-2_000_000,
        operating_cash_flow=-300_000,
        capex=100_000,
    )


@pytest.fixture
def prior_period():
    """Prior period data for comparison."""
    return FinancialData(
        revenue=9_000_000,
        cogs=5_600_000,
        gross_profit=3_400_000,
        operating_income=1_700_000,
        ebit=1_700_000,
        net_income=1_100_000,
        total_assets=19_000_000,
        current_assets=7_000_000,
        current_liabilities=4_200_000,
        total_debt=6_500_000,
        total_equity=9_000_000,
        total_liabilities=10_000_000,
        retained_earnings=4_000_000,
        operating_cash_flow=2_000_000,
    )


# ===== Composite Health Score Tests =====


class TestCompositeHealthScore:
    def test_returns_dataclass(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert isinstance(result, CompositeHealthScore)

    def test_score_range(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert 0 <= result.score <= 100

    def test_grade_assignment(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_healthy_company_high_score(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert result.score >= 50
        assert result.grade in ("A", "B", "C")

    def test_distressed_company_low_score(self, analyzer, distressed_company):
        result = analyzer.composite_health_score(distressed_company)
        assert result.score <= 35
        assert result.grade in ("D", "F")

    def test_all_components_present(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        expected_components = {"z_score", "f_score", "profitability", "liquidity", "leverage"}
        assert set(result.component_scores.keys()) == expected_components

    def test_components_sum_to_total(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert result.score == sum(result.component_scores.values())

    def test_z_score_component_max_25(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert 0 <= result.component_scores["z_score"] <= 25

    def test_f_score_component_max_25(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert 0 <= result.component_scores["f_score"] <= 25

    def test_profitability_component_max_20(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert 0 <= result.component_scores["profitability"] <= 20

    def test_liquidity_component_max_15(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert 0 <= result.component_scores["liquidity"] <= 15

    def test_leverage_component_max_15(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert 0 <= result.component_scores["leverage"] <= 15

    def test_interpretation_text(self, analyzer, healthy_company):
        result = analyzer.composite_health_score(healthy_company)
        assert result.interpretation is not None
        assert "Composite score" in result.interpretation
        assert "Grade" in result.interpretation
        assert "Strongest" in result.interpretation
        assert "Weakest" in result.interpretation

    def test_with_prior_data(self, analyzer, healthy_company, prior_period):
        result = analyzer.composite_health_score(healthy_company, prior_period)
        assert isinstance(result, CompositeHealthScore)
        assert result.score >= 0

    def test_empty_data(self, analyzer):
        data = FinancialData()
        result = analyzer.composite_health_score(data)
        assert isinstance(result, CompositeHealthScore)
        assert result.score >= 0
        assert result.score <= 100

    def test_grade_thresholds(self, analyzer):
        """Verify grade boundaries by checking known ranges."""
        # Very strong company -> A grade
        strong = FinancialData(
            revenue=15_000_000,
            cogs=5_000_000,
            gross_profit=10_000_000,
            operating_income=5_000_000,
            ebit=5_000_000,
            net_income=3_000_000,
            total_assets=10_000_000,
            current_assets=6_000_000,
            current_liabilities=1_500_000,
            total_liabilities=3_000_000,
            total_equity=7_000_000,
            retained_earnings=4_000_000,
            total_debt=1_000_000,
            operating_cash_flow=4_000_000,
        )
        result = analyzer.composite_health_score(strong)
        assert result.grade in ("A", "B")  # Strong company


# ===== Period Comparison Tests =====


class TestPeriodComparison:
    def test_returns_dataclass(self, analyzer, healthy_company, prior_period):
        result = analyzer.compare_periods(healthy_company, prior_period)
        assert isinstance(result, PeriodComparison)

    def test_has_current_and_prior_ratios(self, analyzer, healthy_company, prior_period):
        result = analyzer.compare_periods(healthy_company, prior_period)
        assert len(result.current_ratios) > 0
        assert len(result.prior_ratios) > 0

    def test_deltas_computed(self, analyzer, healthy_company, prior_period):
        result = analyzer.compare_periods(healthy_company, prior_period)
        assert len(result.deltas) > 0

    def test_improvements_and_deteriorations(self, analyzer, healthy_company, prior_period):
        result = analyzer.compare_periods(healthy_company, prior_period)
        # Should have at least some improvements or deteriorations
        assert len(result.improvements) + len(result.deteriorations) > 0

    def test_delta_direction_correct(self, analyzer, healthy_company, prior_period):
        result = analyzer.compare_periods(healthy_company, prior_period)
        for key in result.deltas:
            cv = result.current_ratios.get(key)
            pv = result.prior_ratios.get(key)
            if cv is not None and pv is not None:
                assert abs(result.deltas[key] - (cv - pv)) < 1e-6

    def test_includes_all_ratio_categories(self, analyzer, healthy_company, prior_period):
        result = analyzer.compare_periods(healthy_company, prior_period)
        categories = set()
        for key in result.current_ratios:
            categories.add(key.split("_")[0])
        # Should have at least liquidity, profitability, leverage
        assert "liquidity" in categories
        assert "profitability" in categories
        assert "leverage" in categories

    def test_includes_scoring_models(self, analyzer, healthy_company, prior_period):
        result = analyzer.compare_periods(healthy_company, prior_period)
        assert "altman_z_score" in result.current_ratios
        assert "piotroski_f_score" in result.current_ratios

    def test_leverage_lower_is_better(self, analyzer, healthy_company, prior_period):
        """Decreasing leverage should be classified as improvement."""
        result = analyzer.compare_periods(healthy_company, prior_period)
        for key in result.improvements:
            if "leverage_debt" in key:
                # If leverage decreased, delta should be negative
                assert result.deltas[key] < 0

    def test_identical_periods(self, analyzer, healthy_company):
        """Same data for both periods should have zero deltas."""
        result = analyzer.compare_periods(healthy_company, healthy_company)
        for delta in result.deltas.values():
            assert abs(delta) < 1e-6
        assert len(result.improvements) == 0
        assert len(result.deteriorations) == 0


# ===== Financial Report Tests =====


class TestFinancialReport:
    def test_returns_dataclass(self, analyzer, healthy_company):
        result = analyzer.generate_report(healthy_company)
        assert isinstance(result, FinancialReport)

    def test_has_executive_summary(self, analyzer, healthy_company):
        result = analyzer.generate_report(healthy_company)
        assert result.executive_summary != ""
        assert "Overall Financial Health" in result.executive_summary

    def test_has_required_sections(self, analyzer, healthy_company):
        result = analyzer.generate_report(healthy_company)
        expected_sections = [
            "executive_summary",
            "ratio_analysis",
            "scoring_models",
            "risk_assessment",
            "recommendations",
        ]
        for section in expected_sections:
            assert section in result.sections, f"Missing section: {section}"

    def test_generated_at_timestamp(self, analyzer, healthy_company):
        result = analyzer.generate_report(healthy_company)
        assert result.generated_at != ""
        # Should be parseable as ISO timestamp
        datetime.fromisoformat(result.generated_at)

    def test_scoring_models_section(self, analyzer, healthy_company):
        result = analyzer.generate_report(healthy_company)
        scoring = result.sections["scoring_models"]
        assert "Composite Health" in scoring
        assert "F-Score" in scoring

    def test_with_prior_data(self, analyzer, healthy_company, prior_period):
        result = analyzer.generate_report(healthy_company, prior_period)
        assert "period_comparison" in result.sections

    def test_without_prior_data(self, analyzer, healthy_company):
        result = analyzer.generate_report(healthy_company)
        assert "period_comparison" not in result.sections

    def test_period_comparison_content(self, analyzer, healthy_company, prior_period):
        result = analyzer.generate_report(healthy_company, prior_period)
        comp = result.sections["period_comparison"]
        # Should have Improvements or Deteriorations or "No significant changes"
        assert "Improvements" in comp or "Deteriorations" in comp or "No significant changes" in comp

    def test_risk_assessment_section(self, analyzer, healthy_company):
        result = analyzer.generate_report(healthy_company)
        assert result.sections["risk_assessment"] != ""

    def test_distressed_report(self, analyzer, distressed_company):
        result = analyzer.generate_report(distressed_company)
        assert "Grade" in result.executive_summary or "Health" in result.executive_summary
        # Distressed company should have warnings
        risk = result.sections["risk_assessment"]
        assert "WARNING" in risk or "CRITICAL" in risk or "No significant" in risk

    def test_empty_data_report(self, analyzer):
        data = FinancialData()
        result = analyzer.generate_report(data)
        assert isinstance(result, FinancialReport)
        assert result.executive_summary != ""


# ===== analyze() Integration Tests =====


class TestAnalyzeIncludesHealth:
    def test_composite_health_in_results(self, analyzer, healthy_company):
        results = analyzer.analyze(healthy_company)
        assert "composite_health" in results
        assert isinstance(results["composite_health"], CompositeHealthScore)

    def test_composite_health_score_value(self, analyzer, healthy_company):
        results = analyzer.analyze(healthy_company)
        health = results["composite_health"]
        assert 0 <= health.score <= 100
        assert health.grade in ("A", "B", "C", "D", "F")

    def test_all_analyze_keys_present(self, analyzer, healthy_company):
        results = analyzer.analyze(healthy_company)
        expected_keys = [
            "liquidity_ratios",
            "profitability_ratios",
            "leverage_ratios",
            "efficiency_ratios",
            "cash_flow",
            "working_capital",
            "dupont",
            "altman_z_score",
            "piotroski_f_score",
            "composite_health",
            "insights",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_dataframe_input_includes_health(self, analyzer):
        import pandas as pd

        df = pd.DataFrame(
            {
                "Revenue": [10_000_000],
                "Net Income": [1_000_000],
                "Total Assets": [20_000_000],
                "Total Equity": [10_000_000],
                "Current Assets": [5_000_000],
                "Current Liabilities": [3_000_000],
            }
        )
        results = analyzer.analyze(df)
        assert "composite_health" in results
        assert results["composite_health"].score >= 0
