"""Phase 185 Tests: Payout Discipline Analysis.

Tests for payout_discipline_analysis() and PayoutDisciplineResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    PayoutDisciplineResult,
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


class TestPayoutDisciplineDataclass:
    def test_defaults(self):
        r = PayoutDisciplineResult()
        assert r.cash_dividend_coverage is None
        assert r.payout_ratio is None
        assert r.retention_ratio is None
        assert r.dividend_to_ocf is None
        assert r.capex_priority is None
        assert r.free_cash_after_dividends is None
        assert r.pd_score == 0.0
        assert r.pd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = PayoutDisciplineResult(cash_dividend_coverage=5.5, pd_grade="Excellent")
        assert r.cash_dividend_coverage == 5.5
        assert r.pd_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestPayoutDisciplineAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.payout_discipline_analysis(sample_data)
        assert isinstance(result, PayoutDisciplineResult)

    def test_cash_dividend_coverage(self, analyzer, sample_data):
        """OCF/Div = 220k/40k = 5.50."""
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.cash_dividend_coverage == pytest.approx(5.50, abs=0.01)

    def test_payout_ratio(self, analyzer, sample_data):
        """Div/NI = 40k/150k = 0.267."""
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.payout_ratio == pytest.approx(0.267, abs=0.005)

    def test_retention_ratio(self, analyzer, sample_data):
        """(NI-Div)/NI = (150k-40k)/150k = 0.733."""
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.retention_ratio == pytest.approx(0.733, abs=0.005)

    def test_dividend_to_ocf(self, analyzer, sample_data):
        """Div/OCF = 40k/220k = 0.182."""
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.dividend_to_ocf == pytest.approx(0.182, abs=0.005)

    def test_capex_priority(self, analyzer, sample_data):
        """CapEx/(CapEx+Div) = 80k/(80k+40k) = 0.667."""
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.capex_priority == pytest.approx(0.667, abs=0.005)

    def test_free_cash_after_dividends(self, analyzer, sample_data):
        """(OCF-CapEx-Div)/Rev = (220k-80k-40k)/1M = 0.10."""
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.free_cash_after_dividends == pytest.approx(0.10, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.pd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.payout_discipline_analysis(sample_data)
        assert "Payout Discipline" in result.summary


# ===== SCORING TESTS =====


class TestPayoutDisciplineScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """CDC=5.5 => base 10.0. CPri=0.667 >=0.60 => +0.5. RR=0.733 >=0.60 => +0.5. Score=10.0(clamped)."""
        result = analyzer.payout_discipline_analysis(sample_data)
        assert result.pd_score == pytest.approx(10.0, abs=0.5)
        assert result.pd_grade == "Excellent"

    def test_excellent_discipline(self, analyzer):
        """Very high cash dividend coverage."""
        data = FinancialData(
            operating_cash_flow=500_000,
            dividends_paid=50_000,
            net_income=300_000,
            capex=200_000,
            revenue=1_000_000,
        )
        result = analyzer.payout_discipline_analysis(data)
        assert result.pd_score >= 10.0
        assert result.pd_grade == "Excellent"

    def test_weak_discipline(self, analyzer):
        """Very low cash dividend coverage."""
        data = FinancialData(
            operating_cash_flow=50_000,
            dividends_paid=80_000,
            net_income=60_000,
            capex=20_000,
            revenue=1_000_000,
        )
        # CDC=50k/80k=0.625 <1.0 => base 1.0. CPri=20k/(20k+80k)=0.20 <0.30 => -0.5. RR=(60k-80k)/60k=-0.33 <0.30 => -0.5. Score=0.0.
        result = analyzer.payout_discipline_analysis(data)
        assert result.pd_score <= 1.0
        assert result.pd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase185EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.payout_discipline_analysis(FinancialData())
        assert isinstance(result, PayoutDisciplineResult)
        assert result.pd_score == 0.0

    def test_no_dividends(self, analyzer):
        """Div=None => CDC=None, score 0."""
        data = FinancialData(
            operating_cash_flow=220_000,
            net_income=150_000,
        )
        result = analyzer.payout_discipline_analysis(data)
        assert result.cash_dividend_coverage is None
        assert result.pd_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => CDC=None."""
        data = FinancialData(
            dividends_paid=40_000,
            net_income=150_000,
        )
        result = analyzer.payout_discipline_analysis(data)
        assert result.cash_dividend_coverage is None
        assert result.pd_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => PR=None, RR=None."""
        data = FinancialData(
            operating_cash_flow=220_000,
            dividends_paid=40_000,
            capex=80_000,
        )
        result = analyzer.payout_discipline_analysis(data)
        assert result.payout_ratio is None
        assert result.retention_ratio is None
        assert result.cash_dividend_coverage is not None
