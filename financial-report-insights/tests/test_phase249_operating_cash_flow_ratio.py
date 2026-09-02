"""Phase 249 Tests: Operating Cash Flow Ratio Analysis.

Tests for operating_cash_flow_ratio_analysis() and OperatingCashFlowRatioResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperatingCashFlowRatioResult,
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


class TestOperatingCashFlowRatioDataclass:
    def test_defaults(self):
        r = OperatingCashFlowRatioResult()
        assert r.ocf_to_cl is None
        assert r.ocf_to_tl is None
        assert r.ocf_to_revenue is None
        assert r.ocf_to_ni is None
        assert r.ocf_to_debt is None
        assert r.ocf_margin is None
        assert r.ocfr_score == 0.0
        assert r.ocfr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperatingCashFlowRatioResult(ocf_to_cl=1.5, ocfr_grade="Good")
        assert r.ocf_to_cl == 1.5
        assert r.ocfr_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestOperatingCashFlowRatioAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operating_cash_flow_ratio_analysis(sample_data)
        assert isinstance(result, OperatingCashFlowRatioResult)

    def test_ocf_to_cl(self, analyzer, sample_data):
        """OCF/CL = 220k/200k = 1.10."""
        result = analyzer.operating_cash_flow_ratio_analysis(sample_data)
        assert result.ocf_to_cl == pytest.approx(1.10, abs=0.01)

    def test_ocf_to_tl(self, analyzer, sample_data):
        """OCF/TL = 220k/800k = 0.275."""
        result = analyzer.operating_cash_flow_ratio_analysis(sample_data)
        assert result.ocf_to_tl == pytest.approx(0.275, abs=0.01)

    def test_ocf_to_revenue(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.operating_cash_flow_ratio_analysis(sample_data)
        assert result.ocf_to_revenue == pytest.approx(0.22, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operating_cash_flow_ratio_analysis(sample_data)
        assert result.ocfr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operating_cash_flow_ratio_analysis(sample_data)
        assert "Operating Cash Flow Ratio" in result.summary


# ===== SCORING TESTS =====


class TestOperatingCashFlowRatioScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OCF/CL=1.10 >=1.0 => base 7.0. OCF/Rev=0.22 >=0.20 => +0.5. OCF/NI=1.47 >=1.0 => +0.5. Score=8.0."""
        result = analyzer.operating_cash_flow_ratio_analysis(sample_data)
        assert result.ocfr_score == pytest.approx(8.0, abs=0.5)
        assert result.ocfr_grade == "Excellent"

    def test_excellent_high_coverage(self, analyzer):
        """Very high OCF coverage."""
        data = FinancialData(
            operating_cash_flow=500_000,
            current_liabilities=200_000,
            revenue=1_000_000,
            net_income=100_000,
            total_liabilities=800_000,
        )
        # OCF/CL=2.5 >=2.0 => base 10. OCF/Rev=0.5 >=0.20 => +0.5. OCF/NI=5.0 >=1.0 => +0.5. Score=10.
        result = analyzer.operating_cash_flow_ratio_analysis(data)
        assert result.ocfr_score >= 10.0
        assert result.ocfr_grade == "Excellent"

    def test_weak_low_coverage(self, analyzer):
        """Very low OCF coverage."""
        data = FinancialData(
            operating_cash_flow=30_000,
            current_liabilities=300_000,
            revenue=1_000_000,
            net_income=200_000,
            total_liabilities=800_000,
        )
        # OCF/CL=0.10 <0.25 => base 1.0. OCF/Rev=0.03 <0.05 => -0.5. Score=0.5.
        result = analyzer.operating_cash_flow_ratio_analysis(data)
        assert result.ocfr_score <= 1.5
        assert result.ocfr_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase249EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operating_cash_flow_ratio_analysis(FinancialData())
        assert isinstance(result, OperatingCashFlowRatioResult)
        assert result.ocfr_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => insufficient data => score 0."""
        data = FinancialData(current_liabilities=200_000)
        result = analyzer.operating_cash_flow_ratio_analysis(data)
        assert result.ocfr_score == 0.0

    def test_no_cl(self, analyzer):
        """CL=None => insufficient data => score 0."""
        data = FinancialData(operating_cash_flow=220_000)
        result = analyzer.operating_cash_flow_ratio_analysis(data)
        assert result.ocfr_score == 0.0

    def test_zero_cl(self, analyzer):
        """CL=0 => insufficient data => score 0."""
        data = FinancialData(
            operating_cash_flow=220_000,
            current_liabilities=0,
        )
        result = analyzer.operating_cash_flow_ratio_analysis(data)
        assert result.ocfr_score == 0.0
