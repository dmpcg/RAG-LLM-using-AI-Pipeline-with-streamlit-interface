"""Phase 291 Tests: Revenue Cash Realization Analysis.

Tests for revenue_cash_realization_analysis() and RevenueCashRealizationResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    RevenueCashRealizationResult,
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


class TestRevenueCashRealizationDataclass:
    def test_defaults(self):
        r = RevenueCashRealizationResult()
        assert r.cash_to_revenue is None
        assert r.ocf_to_revenue is None
        assert r.collection_rate is None
        assert r.revenue_cash_gap is None
        assert r.cash_conversion_speed is None
        assert r.revenue_quality_ratio is None
        assert r.rcr_score == 0.0
        assert r.rcr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = RevenueCashRealizationResult(ocf_to_revenue=0.22, rcr_grade="Excellent")
        assert r.ocf_to_revenue == 0.22
        assert r.rcr_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestRevenueCashRealizationAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert isinstance(result, RevenueCashRealizationResult)

    def test_ocf_to_revenue(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert result.ocf_to_revenue == pytest.approx(0.22, abs=0.01)

    def test_collection_rate(self, analyzer, sample_data):
        """(Rev-AR)/Rev = (1M-150k)/1M = 0.85."""
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert result.collection_rate == pytest.approx(0.85, abs=0.01)

    def test_revenue_cash_gap(self, analyzer, sample_data):
        """Rev-OCF = 1M-220k = 780k."""
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert result.revenue_cash_gap == pytest.approx(780_000, abs=100)

    def test_cash_conversion_speed(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert result.cash_conversion_speed == pytest.approx(0.22, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert result.rcr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert "Revenue Cash Realization" in result.summary


# ===== SCORING TESTS =====


class TestRevenueCashRealizationScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OCF/Rev=0.22 in [0.22,0.30)=>base 8.5. CollRate=0.85>=0.85(+0.5). OCF>0&Rev>0(+0.5). Score=9.5."""
        result = analyzer.revenue_cash_realization_analysis(sample_data)
        assert result.rcr_score == pytest.approx(9.5, abs=0.5)
        assert result.rcr_grade == "Excellent"

    def test_very_strong_realization(self, analyzer):
        """Very strong cash realization."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=400_000,
            accounts_receivable=50_000,
        )
        # OCF/Rev=0.40>=0.30=>base 10. CollRate=0.95>=0.85(+0.5). OCF>0&Rev>0(+0.5). Score=10 (capped).
        result = analyzer.revenue_cash_realization_analysis(data)
        assert result.rcr_score >= 10.0
        assert result.rcr_grade == "Excellent"

    def test_weak_realization(self, analyzer):
        """Weak cash realization."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=10_000,
            accounts_receivable=500_000,
        )
        # OCF/Rev=0.01<0.02=>base 1.0. CollRate=0.50<0.85(no adj). OCF>0&Rev>0(+0.5). Score=1.5.
        result = analyzer.revenue_cash_realization_analysis(data)
        assert result.rcr_score <= 2.0
        assert result.rcr_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase291EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.revenue_cash_realization_analysis(FinancialData())
        assert isinstance(result, RevenueCashRealizationResult)
        assert result.rcr_score == 0.0

    def test_no_revenue(self, analyzer):
        data = FinancialData(operating_cash_flow=220_000)
        result = analyzer.revenue_cash_realization_analysis(data)
        assert result.rcr_score == 0.0

    def test_zero_revenue(self, analyzer):
        data = FinancialData(revenue=0, operating_cash_flow=220_000)
        result = analyzer.revenue_cash_realization_analysis(data)
        assert result.rcr_score == 0.0

    def test_no_ocf(self, analyzer):
        data = FinancialData(revenue=1_000_000)
        result = analyzer.revenue_cash_realization_analysis(data)
        assert isinstance(result, RevenueCashRealizationResult)
