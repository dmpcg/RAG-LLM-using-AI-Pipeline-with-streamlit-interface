"""Phase 215 Tests: Cost Control Analysis.

Tests for cost_control_analysis() and CostControlResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    CostControlResult,
    FinancialData,
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
        cash=50_000,
        inventory=100_000,
        accounts_receivable=150_000,
        accounts_payable=80_000,
        total_debt=400_000,
        retained_earnings=600_000,
        depreciation=50_000,
        interest_expense=30_000,
        operating_cash_flow=220_000,
        capex=80_000,
        dividends_paid=40_000,
    )


# ===== DATACLASS TESTS =====


class TestCostControlDataclass:
    def test_defaults(self):
        r = CostControlResult()
        assert r.opex_to_revenue is None
        assert r.cogs_to_revenue is None
        assert r.sga_to_revenue is None
        assert r.operating_margin is None
        assert r.opex_to_gross_profit is None
        assert r.ebitda_margin is None
        assert r.cc_score == 0.0
        assert r.cc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CostControlResult(opex_to_revenue=0.20, cc_grade="Good")
        assert r.opex_to_revenue == 0.20
        assert r.cc_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestCostControlAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cost_control_analysis(sample_data)
        assert isinstance(result, CostControlResult)

    def test_opex_to_revenue(self, analyzer, sample_data):
        """OpEx/Rev = 200k/1M = 0.20."""
        result = analyzer.cost_control_analysis(sample_data)
        assert result.opex_to_revenue == pytest.approx(0.20, abs=0.005)

    def test_cogs_to_revenue(self, analyzer, sample_data):
        """COGS/Rev = 600k/1M = 0.60."""
        result = analyzer.cost_control_analysis(sample_data)
        assert result.cogs_to_revenue == pytest.approx(0.60, abs=0.005)

    def test_operating_margin(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.cost_control_analysis(sample_data)
        assert result.operating_margin == pytest.approx(0.20, abs=0.005)

    def test_opex_to_gross_profit(self, analyzer, sample_data):
        """OpEx/GP = 200k/400k = 0.50."""
        result = analyzer.cost_control_analysis(sample_data)
        assert result.opex_to_gross_profit == pytest.approx(0.50, abs=0.01)

    def test_ebitda_margin(self, analyzer, sample_data):
        """EBITDA/Rev = 250k/1M = 0.25."""
        result = analyzer.cost_control_analysis(sample_data)
        assert result.ebitda_margin == pytest.approx(0.25, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cost_control_analysis(sample_data)
        assert result.cc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cost_control_analysis(sample_data)
        assert "Cost Control" in result.summary


# ===== SCORING TESTS =====


class TestCostControlScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OpEx/Rev=0.20 <=0.20 => base 8.5. OI/Rev=0.20 <0.25 => no adj. COGS/Rev=0.60 >0.40 but <=0.75 => no adj. Score=8.5."""
        result = analyzer.cost_control_analysis(sample_data)
        assert result.cc_score == pytest.approx(8.5, abs=0.5)
        assert result.cc_grade == "Excellent"

    def test_excellent_cost_control(self, analyzer):
        """Very low OpEx/Revenue."""
        data = FinancialData(
            revenue=2_000_000,
            operating_expenses=200_000,
            cogs=500_000,
            gross_profit=1_500_000,
            operating_income=800_000,
            ebitda=900_000,
        )
        # OpEx/Rev=0.10 <=0.15 => base 10. OI/Rev=0.40 >=0.25 => +0.5. COGS/Rev=0.25 <=0.40 => +0.5. Score=10.
        result = analyzer.cost_control_analysis(data)
        assert result.cc_score >= 10.0
        assert result.cc_grade == "Excellent"

    def test_weak_cost_control(self, analyzer):
        """Very high OpEx/Revenue."""
        data = FinancialData(
            revenue=500_000,
            operating_expenses=400_000,
            cogs=400_000,
            gross_profit=100_000,
            operating_income=10_000,
            ebitda=20_000,
        )
        # OpEx/Rev=0.80 >0.55 => base 1.0. OI/Rev=0.02 <0.05 => -0.5. COGS/Rev=0.80 >0.75 => -0.5. Score=0.
        result = analyzer.cost_control_analysis(data)
        assert result.cc_score <= 0.5
        assert result.cc_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase215EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cost_control_analysis(FinancialData())
        assert isinstance(result, CostControlResult)
        assert result.cc_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => OpEx/Rev=None => score 0."""
        data = FinancialData(
            operating_expenses=200_000,
            cogs=600_000,
        )
        result = analyzer.cost_control_analysis(data)
        assert result.opex_to_revenue is None
        assert result.cc_score == 0.0

    def test_no_opex(self, analyzer):
        """OpEx=None => OpEx/Rev=None => score 0."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
        )
        result = analyzer.cost_control_analysis(data)
        assert result.opex_to_revenue is None
        assert result.cc_score == 0.0

    def test_no_cogs(self, analyzer):
        """COGS=None => COGS/Rev=None, but OpEx/Rev still works."""
        data = FinancialData(
            revenue=1_000_000,
            operating_expenses=200_000,
            operating_income=200_000,
        )
        result = analyzer.cost_control_analysis(data)
        assert result.cogs_to_revenue is None
        assert result.opex_to_revenue is not None
        assert result.cc_score > 0.0
