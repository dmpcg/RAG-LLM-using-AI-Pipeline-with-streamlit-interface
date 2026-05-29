"""Phase 251 Tests: Dividend Payout Ratio Analysis.

Tests for dividend_payout_analysis() and DividendPayoutResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DividendPayoutResult,
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


class TestDividendPayoutDataclass:
    def test_defaults(self):
        r = DividendPayoutResult()
        assert r.div_to_ni is None
        assert r.retention_ratio is None
        assert r.div_to_ocf is None
        assert r.div_to_fcf is None
        assert r.div_to_revenue is None
        assert r.div_coverage is None
        assert r.dpr_score == 0.0
        assert r.dpr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DividendPayoutResult(div_to_ni=0.30, dpr_grade="Excellent")
        assert r.div_to_ni == 0.30
        assert r.dpr_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestDividendPayoutAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.dividend_payout_analysis(sample_data)
        assert isinstance(result, DividendPayoutResult)

    def test_div_to_ni(self, analyzer, sample_data):
        """Div/NI = 40k/150k = 0.2667."""
        result = analyzer.dividend_payout_analysis(sample_data)
        assert result.div_to_ni == pytest.approx(0.2667, abs=0.01)

    def test_retention_ratio(self, analyzer, sample_data):
        """Retention = 1 - 0.2667 = 0.7333."""
        result = analyzer.dividend_payout_analysis(sample_data)
        assert result.retention_ratio == pytest.approx(0.7333, abs=0.01)

    def test_div_coverage(self, analyzer, sample_data):
        """Div Coverage = NI/Div = 150k/40k = 3.75."""
        result = analyzer.dividend_payout_analysis(sample_data)
        assert result.div_coverage == pytest.approx(3.75, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.dividend_payout_analysis(sample_data)
        assert result.dpr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.dividend_payout_analysis(sample_data)
        assert "Dividend Payout" in result.summary


# ===== SCORING TESTS =====


class TestDividendPayoutScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """Div/NI=0.267 in [0.20,0.60] => base 10. Coverage=3.75 >=2.0 => +0.5. Div/OCF=0.182 <=0.50 => +0.5. Score=10."""
        result = analyzer.dividend_payout_analysis(sample_data)
        assert result.dpr_score >= 10.0
        assert result.dpr_grade == "Excellent"

    def test_moderate_payout(self, analyzer):
        """Moderate payout in ideal range."""
        data = FinancialData(
            net_income=200_000,
            dividends_paid=80_000,
            operating_cash_flow=300_000,
            capex=50_000,
            revenue=1_000_000,
        )
        # Div/NI=0.40 in [0.20,0.60] => base 10.
        result = analyzer.dividend_payout_analysis(data)
        assert result.dpr_score >= 10.0
        assert result.dpr_grade == "Excellent"

    def test_high_payout_weak(self, analyzer):
        """Payout >90% is unsustainable."""
        data = FinancialData(
            net_income=100_000,
            dividends_paid=95_000,
            operating_cash_flow=110_000,
            revenue=1_000_000,
        )
        # Div/NI=0.95 >0.90 => base 3.0. Coverage=1.05 >=1.0 no penalty. Div/OCF=0.864 >0.50 no bonus.
        result = analyzer.dividend_payout_analysis(data)
        assert result.dpr_score <= 4.0
        assert result.dpr_grade in ["Weak", "Adequate"]


# ===== EDGE CASES =====


class TestPhase251EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.dividend_payout_analysis(FinancialData())
        assert isinstance(result, DividendPayoutResult)
        assert result.dpr_score == 0.0

    def test_no_dividends(self, analyzer):
        """Div=None => insufficient data => score 0."""
        data = FinancialData(net_income=150_000)
        result = analyzer.dividend_payout_analysis(data)
        assert result.dpr_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => insufficient data => score 0."""
        data = FinancialData(dividends_paid=40_000)
        result = analyzer.dividend_payout_analysis(data)
        assert result.dpr_score == 0.0

    def test_zero_net_income(self, analyzer):
        """NI=0 => insufficient data => score 0."""
        data = FinancialData(
            net_income=0,
            dividends_paid=40_000,
        )
        result = analyzer.dividend_payout_analysis(data)
        assert result.dpr_score == 0.0
