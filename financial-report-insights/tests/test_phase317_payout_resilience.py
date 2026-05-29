"""Phase 317 Tests: Payout Resilience Analysis.

Tests for payout_resilience_analysis() and PayoutResilienceResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    PayoutResilienceResult,
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


class TestPayoutResilienceDataclass:
    def test_defaults(self):
        r = PayoutResilienceResult()
        assert r.div_to_ni is None
        assert r.div_to_ocf is None
        assert r.div_to_revenue is None
        assert r.div_to_ebitda is None
        assert r.payout_ratio is None
        assert r.resilience_buffer is None
        assert r.prs_score == 0.0
        assert r.prs_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = PayoutResilienceResult(payout_ratio=0.30, prs_grade="Excellent")
        assert r.payout_ratio == 0.30
        assert r.prs_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestPayoutResilienceAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.payout_resilience_analysis(sample_data)
        assert isinstance(result, PayoutResilienceResult)

    def test_div_to_ni(self, analyzer, sample_data):
        """Div/NI = 40k/150k = 0.267."""
        result = analyzer.payout_resilience_analysis(sample_data)
        assert result.div_to_ni == pytest.approx(0.267, abs=0.01)

    def test_div_to_ocf(self, analyzer, sample_data):
        """Div/OCF = 40k/220k = 0.182."""
        result = analyzer.payout_resilience_analysis(sample_data)
        assert result.div_to_ocf == pytest.approx(0.182, abs=0.01)

    def test_resilience_buffer(self, analyzer, sample_data):
        """Buffer = 1 - 0.267 = 0.733."""
        result = analyzer.payout_resilience_analysis(sample_data)
        assert result.resilience_buffer == pytest.approx(0.733, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.payout_resilience_analysis(sample_data)
        assert result.prs_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.payout_resilience_analysis(sample_data)
        assert "Payout Resilience" in result.summary


# ===== SCORING TESTS =====


class TestPayoutResilienceScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """Div/NI=0.267 in [0.20,0.50]=>base 10. Div/OCF=0.182<=0.40(+0.5). Both>0(+0.5). Score=10 (capped)."""
        result = analyzer.payout_resilience_analysis(sample_data)
        assert result.prs_score >= 10.0
        assert result.prs_grade == "Excellent"

    def test_high_payout(self, analyzer):
        """Very high payout — unsustainable."""
        data = FinancialData(
            dividends_paid=140_000,
            net_income=150_000,
            operating_cash_flow=220_000,
        )
        # Div/NI=0.933 in (0.90,1.0]=>base 2.5. Div/OCF=0.636>0.40(no adj). Both>0(+0.5). Score=3.0.
        result = analyzer.payout_resilience_analysis(data)
        assert result.prs_score <= 3.5
        assert result.prs_grade == "Weak"

    def test_over_payout(self, analyzer):
        """Paying more than earned — dividend > net income."""
        data = FinancialData(
            dividends_paid=200_000,
            net_income=150_000,
            operating_cash_flow=220_000,
        )
        # Div/NI=1.333>1.0=>base 1.0. Div/OCF=0.909>0.40(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.payout_resilience_analysis(data)
        assert result.prs_score <= 2.0
        assert result.prs_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase317EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.payout_resilience_analysis(FinancialData())
        assert isinstance(result, PayoutResilienceResult)
        assert result.prs_score == 0.0

    def test_no_dividends(self, analyzer):
        data = FinancialData(net_income=150_000, operating_cash_flow=220_000)
        result = analyzer.payout_resilience_analysis(data)
        assert result.prs_score == 0.0

    def test_no_net_income(self, analyzer):
        data = FinancialData(dividends_paid=40_000, operating_cash_flow=220_000)
        result = analyzer.payout_resilience_analysis(data)
        assert result.prs_score == 0.0
