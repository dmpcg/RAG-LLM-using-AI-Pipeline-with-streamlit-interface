"""Phase 15 Tests: Comprehensive Financial Health Score.

Tests for comprehensive_health_score(), HealthDimension, ComprehensiveHealthResult.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    ComprehensiveHealthResult,
    FinancialData,
    HealthDimension,
)


@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture
def sample_data():
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
        capex=80_000,
    )


# ===== DATACLASS TESTS =====


class TestHealthDimensionDataclass:
    def test_defaults(self):
        d = HealthDimension()
        assert d.name == ""
        assert d.score == 0.0
        assert d.max_score == 100.0
        assert d.weight == 0.0
        assert d.status == ""
        assert d.detail == ""

    def test_fields(self):
        d = HealthDimension(name="Liquidity", score=75.0, weight=0.15, status="green")
        assert d.name == "Liquidity"
        assert d.score == 75.0
        assert d.weight == 0.15


class TestComprehensiveHealthResultDataclass:
    def test_defaults(self):
        r = ComprehensiveHealthResult()
        assert r.overall_score == 0.0
        assert r.grade == ""
        assert r.dimensions == []
        assert r.summary == ""

    def test_fields(self):
        r = ComprehensiveHealthResult(overall_score=85.0, grade="A")
        assert r.overall_score == 85.0
        assert r.grade == "A"


# ===== COMPREHENSIVE HEALTH SCORE =====


class TestComprehensiveHealthScore:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        assert isinstance(result, ComprehensiveHealthResult)

    def test_has_seven_dimensions(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        assert len(result.dimensions) == 7

    def test_dimension_names(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        names = [d.name for d in result.dimensions]
        assert "Profitability" in names
        assert "Liquidity" in names
        assert "Leverage" in names
        assert "Efficiency" in names
        assert "Cash Flow" in names
        assert "Capital Efficiency" in names
        assert "Debt Service" in names

    def test_overall_score_range(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        assert 0 <= result.overall_score <= 100

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        assert result.grade in ["A+", "A", "B+", "B", "C+", "C", "D", "F"]

    def test_healthy_company_high_score(self, analyzer):
        """Strong company should get A or A+ grade."""
        data = FinancialData(
            revenue=5_000_000,
            cogs=2_000_000,
            gross_profit=3_000_000,
            operating_expenses=1_000_000,
            operating_income=2_000_000,
            net_income=1_500_000,
            ebit=2_000_000,
            ebitda=2_500_000,
            total_assets=8_000_000,
            total_liabilities=2_000_000,
            total_equity=6_000_000,
            current_assets=3_000_000,
            current_liabilities=500_000,
            inventory=200_000,
            total_debt=500_000,
            interest_expense=20_000,
            operating_cash_flow=2_000_000,
            capex=300_000,
        )
        result = analyzer.comprehensive_health_score(data)
        assert result.overall_score >= 80
        assert result.grade in ["A+", "A"]

    def test_distressed_company_low_score(self, analyzer):
        """Weak company should get D or F grade."""
        data = FinancialData(
            revenue=200_000,
            cogs=250_000,
            gross_profit=-50_000,
            operating_expenses=100_000,
            operating_income=-150_000,
            net_income=-200_000,
            ebit=-150_000,
            ebitda=-100_000,
            total_assets=500_000,
            total_liabilities=800_000,
            total_equity=-300_000,
            current_assets=50_000,
            current_liabilities=300_000,
            inventory=20_000,
            total_debt=600_000,
            interest_expense=80_000,
            operating_cash_flow=-120_000,
            capex=10_000,
        )
        result = analyzer.comprehensive_health_score(data)
        assert result.overall_score < 40
        assert result.grade in ["D", "F"]

    def test_dimension_scores_bounded(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        for d in result.dimensions:
            assert 0 <= d.score <= 100

    def test_dimension_weights_sum_to_one(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        total_weight = sum(d.weight for d in result.dimensions)
        assert abs(total_weight - 1.0) < 0.01

    def test_dimension_statuses(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        for d in result.dimensions:
            assert d.status in ["green", "yellow", "red"]

    def test_dimension_details_present(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        for d in result.dimensions:
            assert d.detail != ""

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.comprehensive_health_score(sample_data)
        assert "Overall" in result.summary
        assert result.grade in result.summary

    def test_empty_data(self, analyzer):
        result = analyzer.comprehensive_health_score(FinancialData())
        assert isinstance(result, ComprehensiveHealthResult)
        assert 0 <= result.overall_score <= 100
        assert result.grade != ""

    def test_weighted_sum_correct(self, analyzer, sample_data):
        """Overall score should be the weighted sum of dimension scores."""
        result = analyzer.comprehensive_health_score(sample_data)
        expected = sum(d.score * d.weight for d in result.dimensions)
        assert abs(result.overall_score - expected) < 1.0


# ===== GRADE MAPPING =====


class TestHealthGrade:
    def test_a_plus(self, analyzer):
        assert analyzer._health_grade(95) == "A+"

    def test_a(self, analyzer):
        assert analyzer._health_grade(85) == "A"

    def test_b_plus(self, analyzer):
        assert analyzer._health_grade(75) == "B+"

    def test_b(self, analyzer):
        assert analyzer._health_grade(65) == "B"

    def test_c_plus(self, analyzer):
        assert analyzer._health_grade(55) == "C+"

    def test_c(self, analyzer):
        assert analyzer._health_grade(45) == "C"

    def test_d(self, analyzer):
        assert analyzer._health_grade(35) == "D"

    def test_f(self, analyzer):
        assert analyzer._health_grade(15) == "F"

    def test_boundary_90(self, analyzer):
        assert analyzer._health_grade(90) == "A+"

    def test_boundary_0(self, analyzer):
        assert analyzer._health_grade(0) == "F"


# ===== TRAFFIC LIGHT =====


class TestTrafficLight:
    def test_green(self, analyzer):
        assert analyzer._traffic_light(80) == "green"

    def test_green_boundary(self, analyzer):
        assert analyzer._traffic_light(70) == "green"

    def test_yellow(self, analyzer):
        assert analyzer._traffic_light(55) == "yellow"

    def test_yellow_boundary(self, analyzer):
        assert analyzer._traffic_light(40) == "yellow"

    def test_red(self, analyzer):
        assert analyzer._traffic_light(30) == "red"

    def test_red_zero(self, analyzer):
        assert analyzer._traffic_light(0) == "red"


# ===== EDGE CASES =====


class TestPhase15EdgeCases:
    def test_partial_data(self, analyzer):
        """Only some fields populated."""
        data = FinancialData(
            revenue=500_000,
            net_income=50_000,
            total_assets=1_000_000,
            total_equity=600_000,
        )
        result = analyzer.comprehensive_health_score(data)
        assert isinstance(result, ComprehensiveHealthResult)
        assert len(result.dimensions) == 7

    def test_negative_equity(self, analyzer):
        data = FinancialData(
            revenue=100_000,
            total_equity=-50_000,
            total_debt=200_000,
            total_liabilities=300_000,
            total_assets=250_000,
        )
        result = analyzer.comprehensive_health_score(data)
        assert isinstance(result, ComprehensiveHealthResult)
        # Should still produce valid bounded scores
        for d in result.dimensions:
            assert 0 <= d.score <= 100

    def test_zero_revenue(self, analyzer):
        data = FinancialData(
            revenue=0,
            total_assets=500_000,
            current_assets=200_000,
            current_liabilities=100_000,
        )
        result = analyzer.comprehensive_health_score(data)
        assert isinstance(result, ComprehensiveHealthResult)
        assert result.grade != ""

    def test_all_none_fields(self, analyzer):
        """Completely empty data should still produce a result."""
        data = FinancialData()
        result = analyzer.comprehensive_health_score(data)
        assert len(result.dimensions) == 7
        # All dimensions start at 50 base with no adjustments
        for d in result.dimensions:
            assert d.score >= 0
