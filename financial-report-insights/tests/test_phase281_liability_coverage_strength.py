"""Phase 281 Tests: Liability Coverage Strength Analysis.

Tests for liability_coverage_strength_analysis() and LiabilityCoverageStrengthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    LiabilityCoverageStrengthResult,
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


class TestLiabilityCoverageStrengthDataclass:
    def test_defaults(self):
        r = LiabilityCoverageStrengthResult()
        assert r.ocf_to_liabilities is None
        assert r.ebitda_to_liabilities is None
        assert r.assets_to_liabilities is None
        assert r.equity_to_liabilities is None
        assert r.liability_to_revenue is None
        assert r.liability_burden is None
        assert r.lcs_score == 0.0
        assert r.lcs_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = LiabilityCoverageStrengthResult(ocf_to_liabilities=0.275, lcs_grade="Good")
        assert r.ocf_to_liabilities == 0.275
        assert r.lcs_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestLiabilityCoverageStrengthAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert isinstance(result, LiabilityCoverageStrengthResult)

    def test_ocf_to_liabilities(self, analyzer, sample_data):
        """OCF/TL = 220k/800k = 0.275."""
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert result.ocf_to_liabilities == pytest.approx(0.275, abs=0.01)

    def test_assets_to_liabilities(self, analyzer, sample_data):
        """TA/TL = 2M/800k = 2.5."""
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert result.assets_to_liabilities == pytest.approx(2.5, abs=0.01)

    def test_equity_to_liabilities(self, analyzer, sample_data):
        """TE/TL = 1.2M/800k = 1.5."""
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert result.equity_to_liabilities == pytest.approx(1.5, abs=0.01)

    def test_liability_burden(self, analyzer, sample_data):
        """TL/TA = 800k/2M = 0.40."""
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert result.liability_burden == pytest.approx(0.40, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert result.lcs_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert "Liability Coverage Strength" in result.summary


# ===== SCORING TESTS =====


class TestLiabilityCoverageStrengthScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OCF/TL=0.275 in [0.25,0.35)=>base 7.0. TA/TL=2.5>=2.0(+0.5). OCF>0&TL>0(+0.5). Score=8.0."""
        result = analyzer.liability_coverage_strength_analysis(sample_data)
        assert result.lcs_score == pytest.approx(8.0, abs=0.5)
        assert result.lcs_grade == "Excellent"

    def test_strong_coverage(self, analyzer):
        """Very strong OCF relative to liabilities."""
        data = FinancialData(
            operating_cash_flow=600_000,
            total_liabilities=500_000,
            total_assets=3_000_000,
        )
        # OCF/TL=1.2>=0.50=>base 10. TA/TL=6.0>=2.0(+0.5). OCF>0&TL>0(+0.5). Score=10 (capped).
        result = analyzer.liability_coverage_strength_analysis(data)
        assert result.lcs_score >= 10.0
        assert result.lcs_grade == "Excellent"

    def test_weak_coverage(self, analyzer):
        """Very weak OCF relative to liabilities."""
        data = FinancialData(
            operating_cash_flow=30_000,
            total_liabilities=2_000_000,
            total_assets=2_500_000,
        )
        # OCF/TL=0.015<0.05=>base 1.0. TA/TL=1.25<2.0(no adj). OCF>0&TL>0(+0.5). Score=1.5.
        result = analyzer.liability_coverage_strength_analysis(data)
        assert result.lcs_score <= 2.0
        assert result.lcs_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase281EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.liability_coverage_strength_analysis(FinancialData())
        assert isinstance(result, LiabilityCoverageStrengthResult)
        assert result.lcs_score == 0.0

    def test_no_ocf(self, analyzer):
        data = FinancialData(total_liabilities=800_000)
        result = analyzer.liability_coverage_strength_analysis(data)
        assert result.lcs_score == 0.0

    def test_no_liabilities(self, analyzer):
        data = FinancialData(operating_cash_flow=220_000)
        result = analyzer.liability_coverage_strength_analysis(data)
        assert result.lcs_score == 0.0

    def test_zero_ocf(self, analyzer):
        data = FinancialData(operating_cash_flow=0, total_liabilities=800_000)
        result = analyzer.liability_coverage_strength_analysis(data)
        # Zero OCF coverage gives a low but non-zero score via _scored_analysis
        assert result.lcs_score <= 2.0
