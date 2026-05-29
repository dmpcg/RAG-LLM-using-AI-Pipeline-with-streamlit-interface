"""Phase 307 Tests: CapEx to Revenue Analysis.

Tests for capex_to_revenue_analysis() and CapexToRevenueResult dataclass.
"""

import pytest

from financial_analyzer import (
    CapexToRevenueResult,
    CharlieAnalyzer,
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


class TestCapexToRevenueDataclass:
    def test_defaults(self):
        r = CapexToRevenueResult()
        assert r.capex_to_revenue is None
        assert r.capex_to_ocf is None
        assert r.capex_to_ebitda is None
        assert r.capex_to_assets is None
        assert r.investment_intensity is None
        assert r.capex_yield is None
        assert r.ctr_score == 0.0
        assert r.ctr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CapexToRevenueResult(capex_to_revenue=0.08, ctr_grade="Excellent")
        assert r.capex_to_revenue == 0.08
        assert r.ctr_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestCapexToRevenueAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.capex_to_revenue_analysis(sample_data)
        assert isinstance(result, CapexToRevenueResult)

    def test_capex_to_revenue(self, analyzer, sample_data):
        """CapEx/Rev = 80k/1M = 0.08."""
        result = analyzer.capex_to_revenue_analysis(sample_data)
        assert result.capex_to_revenue == pytest.approx(0.08, abs=0.01)

    def test_capex_to_ocf(self, analyzer, sample_data):
        """CapEx/OCF = 80k/220k = 0.364."""
        result = analyzer.capex_to_revenue_analysis(sample_data)
        assert result.capex_to_ocf == pytest.approx(0.364, abs=0.01)

    def test_capex_to_ebitda(self, analyzer, sample_data):
        """CapEx/EBITDA = 80k/250k = 0.32."""
        result = analyzer.capex_to_revenue_analysis(sample_data)
        assert result.capex_to_ebitda == pytest.approx(0.32, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.capex_to_revenue_analysis(sample_data)
        assert result.ctr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.capex_to_revenue_analysis(sample_data)
        assert "CapEx to Revenue" in result.summary


# ===== SCORING TESTS =====


class TestCapexToRevenueScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """CapEx/Rev=0.08 in [0.05,0.15]=>base 10. OCF>CapEx(+0.5). Both>0(+0.5). Score=10 (capped)."""
        result = analyzer.capex_to_revenue_analysis(sample_data)
        assert result.ctr_score >= 10.0
        assert result.ctr_grade == "Excellent"

    def test_moderate_capex(self, analyzer):
        """Moderate capex — slightly high."""
        data = FinancialData(
            revenue=1_000_000,
            capex=220_000,
            operating_cash_flow=250_000,
        )
        # CapEx/Rev=0.22 in (0.20,0.25]=>base 7.0. OCF>CapEx(+0.5). Both>0(+0.5). Score=8.0.
        result = analyzer.capex_to_revenue_analysis(data)
        assert result.ctr_score == pytest.approx(8.0, abs=0.5)
        assert result.ctr_grade == "Excellent"

    def test_excessive_capex(self, analyzer):
        """Excessive capex relative to revenue."""
        data = FinancialData(
            revenue=1_000_000,
            capex=600_000,
        )
        # CapEx/Rev=0.60>0.50=>base 1.0. No OCF(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.capex_to_revenue_analysis(data)
        assert result.ctr_score <= 2.0
        assert result.ctr_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase307EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.capex_to_revenue_analysis(FinancialData())
        assert isinstance(result, CapexToRevenueResult)
        assert result.ctr_score == 0.0

    def test_no_capex(self, analyzer):
        data = FinancialData(revenue=1_000_000)
        result = analyzer.capex_to_revenue_analysis(data)
        assert result.ctr_score == 0.0

    def test_no_revenue(self, analyzer):
        data = FinancialData(capex=80_000)
        result = analyzer.capex_to_revenue_analysis(data)
        assert result.ctr_score == 0.0
