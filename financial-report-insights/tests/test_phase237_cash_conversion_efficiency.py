"""Phase 237 Tests: Cash Conversion Efficiency Analysis.

Tests for cash_conversion_efficiency_analysis() and CashConversionEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CashConversionEfficiencyResult,
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


class TestCashConversionEfficiencyDataclass:
    def test_defaults(self):
        r = CashConversionEfficiencyResult()
        assert r.ocf_to_oi is None
        assert r.ocf_to_ni is None
        assert r.ocf_to_revenue is None
        assert r.ocf_to_ebitda is None
        assert r.fcf_to_oi is None
        assert r.cash_to_oi is None
        assert r.cce_score == 0.0
        assert r.cce_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CashConversionEfficiencyResult(ocf_to_oi=1.10, cce_grade="Excellent")
        assert r.ocf_to_oi == 1.10
        assert r.cce_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestCashConversionEfficiencyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert isinstance(result, CashConversionEfficiencyResult)

    def test_ocf_to_oi(self, analyzer, sample_data):
        """OCF/OI = 220k/200k = 1.10."""
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert result.ocf_to_oi == pytest.approx(1.10, abs=0.01)

    def test_ocf_to_ni(self, analyzer, sample_data):
        """OCF/NI = 220k/150k = 1.467."""
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert result.ocf_to_ni == pytest.approx(1.467, abs=0.01)

    def test_ocf_to_revenue(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert result.ocf_to_revenue == pytest.approx(0.22, abs=0.005)

    def test_fcf_to_oi(self, analyzer, sample_data):
        """FCF=220k-80k=140k. FCF/OI=140k/200k=0.70."""
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert result.fcf_to_oi == pytest.approx(0.70, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert result.cce_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert "Cash Conversion Efficiency" in result.summary


# ===== SCORING TESTS =====


class TestCashConversionEfficiencyScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OCF/OI=1.10 >=1.00 => base 8.5. OCF/NI=1.467 >=1.20 => +0.5. OCF/Rev=0.22 >=0.20 => +0.5. Score=9.5."""
        result = analyzer.cash_conversion_efficiency_analysis(sample_data)
        assert result.cce_score == pytest.approx(9.5, abs=0.5)
        assert result.cce_grade == "Excellent"

    def test_excellent_conversion(self, analyzer):
        """Very high OCF/OI."""
        data = FinancialData(
            operating_cash_flow=300_000,
            operating_income=200_000,
            net_income=180_000,
            revenue=1_000_000,
        )
        # OCF/OI=1.50 >=1.20 => base 10. OCF/NI=1.67 >=1.20 => +0.5. OCF/Rev=0.30 >=0.20 => +0.5. Score=10.
        result = analyzer.cash_conversion_efficiency_analysis(data)
        assert result.cce_score >= 10.0
        assert result.cce_grade == "Excellent"

    def test_weak_conversion(self, analyzer):
        """Very low OCF/OI."""
        data = FinancialData(
            operating_cash_flow=10_000,
            operating_income=200_000,
            net_income=150_000,
            revenue=1_000_000,
        )
        # OCF/OI=0.05 <0.20 => base 1.0. OCF/NI=0.067 <0.80 => -0.5. OCF/Rev=0.01 <0.05 => -0.5. Score=0.
        result = analyzer.cash_conversion_efficiency_analysis(data)
        assert result.cce_score <= 0.5
        assert result.cce_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase237EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cash_conversion_efficiency_analysis(FinancialData())
        assert isinstance(result, CashConversionEfficiencyResult)
        assert result.cce_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => insufficient data => score 0."""
        data = FinancialData(
            operating_income=200_000,
        )
        result = analyzer.cash_conversion_efficiency_analysis(data)
        assert result.ocf_to_oi is None
        assert result.cce_score == 0.0

    def test_no_oi(self, analyzer):
        """OI=None => insufficient data => score 0."""
        data = FinancialData(
            operating_cash_flow=220_000,
        )
        result = analyzer.cash_conversion_efficiency_analysis(data)
        assert result.ocf_to_oi is None
        assert result.cce_score == 0.0

    def test_zero_oi(self, analyzer):
        """OI=0 => OCF/OI=None => score 0."""
        data = FinancialData(
            operating_cash_flow=220_000,
            operating_income=0,
        )
        result = analyzer.cash_conversion_efficiency_analysis(data)
        assert result.cce_score == 0.0
