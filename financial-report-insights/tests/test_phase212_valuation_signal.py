"""Phase 212 Tests: Valuation Signal Analysis.

Tests for valuation_signal_analysis() and ValuationSignalResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ValuationSignalResult,
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


class TestValuationSignalDataclass:
    def test_defaults(self):
        r = ValuationSignalResult()
        assert r.ev_to_ebitda is None
        assert r.price_to_earnings is None
        assert r.price_to_book is None
        assert r.ev_to_revenue is None
        assert r.earnings_yield is None
        assert r.fcf_yield is None
        assert r.vsg_score == 0.0
        assert r.vsg_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ValuationSignalResult(ev_to_ebitda=8.0, vsg_grade="Good")
        assert r.ev_to_ebitda == 8.0
        assert r.vsg_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestValuationSignalAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.valuation_signal_analysis(sample_data)
        assert isinstance(result, ValuationSignalResult)

    def test_ev_to_ebitda(self, analyzer, sample_data):
        """EV=TA+TD-Cash=2M+400k-50k=2.35M. EV/EBITDA=2.35M/250k=9.40."""
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.ev_to_ebitda == pytest.approx(9.40, abs=0.05)

    def test_price_to_earnings(self, analyzer, sample_data):
        """TA/NI = 2M/150k = 13.33."""
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.price_to_earnings == pytest.approx(13.33, abs=0.05)

    def test_price_to_book(self, analyzer, sample_data):
        """TA/TE = 2M/1.2M = 1.667."""
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.price_to_book == pytest.approx(1.667, abs=0.005)

    def test_ev_to_revenue(self, analyzer, sample_data):
        """EV/Rev = 2.35M/1M = 2.35."""
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.ev_to_revenue == pytest.approx(2.35, abs=0.05)

    def test_earnings_yield(self, analyzer, sample_data):
        """NI/TA = 150k/2M = 0.075."""
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.earnings_yield == pytest.approx(0.075, abs=0.005)

    def test_fcf_yield(self, analyzer, sample_data):
        """FCF=220k-80k=140k. FCF/TA=140k/2M=0.07."""
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.fcf_yield == pytest.approx(0.07, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.vsg_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.valuation_signal_analysis(sample_data)
        assert "Valuation Signal" in result.summary


# ===== SCORING TESTS =====


class TestValuationSignalScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """EV/EBITDA=9.40 <=12 => base 7.0. EY=0.075 >=0.03 but <0.10 => no adj. P/B=1.667 >1.0 but <=5.0 => no adj. Score=7.0."""
        result = analyzer.valuation_signal_analysis(sample_data)
        assert result.vsg_score == pytest.approx(7.0, abs=0.5)
        assert result.vsg_grade in ["Good", "Excellent"]

    def test_excellent_valuation(self, analyzer):
        """Very low EV/EBITDA."""
        data = FinancialData(
            total_assets=500_000,
            total_debt=100_000,
            cash=50_000,
            ebitda=200_000,
            revenue=800_000,
            net_income=150_000,
            total_equity=400_000,
            operating_cash_flow=180_000,
            capex=20_000,
        )
        # EV=500k+100k-50k=550k. EV/EBITDA=2.75 <=5 => base 10. EY=150k/500k=0.30 >=0.10 => +0.5. P/B=500k/400k=1.25 >1 => no adj. Score=10.
        result = analyzer.valuation_signal_analysis(data)
        assert result.vsg_score >= 10.0
        assert result.vsg_grade == "Excellent"

    def test_weak_valuation(self, analyzer):
        """Very high EV/EBITDA."""
        data = FinancialData(
            total_assets=5_000_000,
            total_debt=2_000_000,
            cash=100_000,
            ebitda=100_000,
            revenue=500_000,
            net_income=20_000,
            total_equity=500_000,
            operating_cash_flow=80_000,
            capex=50_000,
        )
        # EV=5M+2M-100k=6.9M. EV/EBITDA=69 >30 => base 1.0. EY=20k/5M=0.004 <0.03 => -0.5. P/B=5M/500k=10 >5 => -0.5. Score=0.
        result = analyzer.valuation_signal_analysis(data)
        assert result.vsg_score <= 0.5
        assert result.vsg_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase212EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.valuation_signal_analysis(FinancialData())
        assert isinstance(result, ValuationSignalResult)
        assert result.vsg_score == 0.0

    def test_no_ebitda(self, analyzer):
        """EBITDA=None => EV/EBITDA=None => score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            total_debt=400_000,
            cash=50_000,
        )
        result = analyzer.valuation_signal_analysis(data)
        assert result.ev_to_ebitda is None
        assert result.vsg_score == 0.0

    def test_no_total_assets(self, analyzer):
        """TA=None => EV=None => EV/EBITDA=None."""
        data = FinancialData(
            ebitda=250_000,
            revenue=1_000_000,
        )
        result = analyzer.valuation_signal_analysis(data)
        assert result.ev_to_ebitda is None
        assert result.vsg_score == 0.0

    def test_no_debt_no_cash(self, analyzer):
        """TD=None, Cash=None => EV=TA only. Still works."""
        data = FinancialData(
            total_assets=2_000_000,
            ebitda=250_000,
            revenue=1_000_000,
            net_income=150_000,
            total_equity=1_200_000,
        )
        # EV=2M. EV/EBITDA=8.0 <=8 => base 8.5
        result = analyzer.valuation_signal_analysis(data)
        assert result.ev_to_ebitda == pytest.approx(8.0, abs=0.05)
        assert result.vsg_score > 0.0
