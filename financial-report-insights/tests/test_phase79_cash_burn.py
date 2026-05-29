"""Phase 79 Tests: Cash Burn Analysis.

Tests for cash_burn_analysis() and CashBurnResult dataclass.
"""

import pytest

from financial_analyzer import (
    CashBurnResult,
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


class TestCashBurnDataclass:
    def test_defaults(self):
        r = CashBurnResult()
        assert r.ocf_margin is None
        assert r.capex_intensity is None
        assert r.fcf_margin is None
        assert r.cash_self_sufficiency is None
        assert r.cash_runway_months is None
        assert r.net_cash_position is None
        assert r.cb_score == 0.0
        assert r.cb_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CashBurnResult(fcf_margin=0.15, cb_grade="Good")
        assert r.fcf_margin == 0.15
        assert r.cb_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestCashBurnAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cash_burn_analysis(sample_data)
        assert isinstance(result, CashBurnResult)

    def test_ocf_margin(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.cash_burn_analysis(sample_data)
        assert result.ocf_margin == pytest.approx(0.22, abs=0.01)

    def test_capex_intensity(self, analyzer, sample_data):
        """CapEx/Rev = 80k/1M = 0.08."""
        result = analyzer.cash_burn_analysis(sample_data)
        assert result.capex_intensity == pytest.approx(0.08, abs=0.01)

    def test_fcf_margin(self, analyzer, sample_data):
        """(OCF-CapEx)/Rev = (220k-80k)/1M = 0.14."""
        result = analyzer.cash_burn_analysis(sample_data)
        assert result.fcf_margin == pytest.approx(0.14, abs=0.01)

    def test_cash_self_sufficiency(self, analyzer, sample_data):
        """OCF/(CapEx+Div) = 220k/(80k+40k) = 1.833."""
        result = analyzer.cash_burn_analysis(sample_data)
        assert result.cash_self_sufficiency == pytest.approx(1.833, abs=0.01)

    def test_net_cash_position(self, analyzer, sample_data):
        """Cash-TD = 50k-400k = -350k."""
        result = analyzer.cash_burn_analysis(sample_data)
        assert result.net_cash_position == pytest.approx(-350_000, abs=100)

    def test_no_cash_runway_when_positive_fcf(self, analyzer, sample_data):
        """FCF > 0 => no cash runway needed."""
        result = analyzer.cash_burn_analysis(sample_data)
        assert result.cash_runway_months is None

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cash_burn_analysis(sample_data)
        assert result.cb_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cash_burn_analysis(sample_data)
        assert "Cash Burn" in result.summary


# ===== SCORING TESTS =====


class TestCashBurnScoring:
    def test_very_high_fcf_margin(self, analyzer):
        """FCF margin >= 0.20 => base 10."""
        data = FinancialData(
            revenue=5_000_000,
            operating_cash_flow=2_000_000,
            capex=500_000,
            dividends_paid=200_000,
            cash=1_000_000,
            total_debt=200_000,
        )
        result = analyzer.cash_burn_analysis(data)
        # FCF margin=0.30 (base 10) + CSS=2.86 (+0.5) + NCP>0 (+0.5) => 10 capped
        assert result.cb_score >= 10.0
        assert result.cb_grade == "Excellent"

    def test_moderate_fcf_margin(self, analyzer, sample_data):
        """FCF margin=0.14 => base 7.0 (falls in [0.10, 0.15) bracket)."""
        result = analyzer.cash_burn_analysis(sample_data)
        # FCF=0.14 (base 7.0) + CSS=1.83 (no adj) + NCP=-350k (no adj) => 7.0
        assert result.cb_score >= 7.0

    def test_low_fcf_margin(self, analyzer):
        """FCF margin ~ 0.02 => base 4.0."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=100_000,
            capex=80_000,
            dividends_paid=50_000,
            cash=30_000,
            total_debt=200_000,
        )
        result = analyzer.cash_burn_analysis(data)
        # FCF=0.02 (base 4.0) + CSS=0.77 (no adj) + NCP=-170k (no adj) => 4.0
        assert result.cb_score >= 3.5
        assert result.cb_score <= 5.0

    def test_negative_fcf(self, analyzer):
        """FCF margin < 0 => base 2.5 or 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=50_000,
            capex=200_000,
            dividends_paid=30_000,
            cash=100_000,
            total_debt=500_000,
        )
        result = analyzer.cash_burn_analysis(data)
        # FCF=-0.15 (base 1.0) + CSS=0.217 (<0.5 => -0.5) + NCP=-400k (no adj) => 0.0
        assert result.cb_score < 2.0
        assert result.cb_grade == "Weak"

    def test_cash_runway_when_burning(self, analyzer):
        """Negative FCF => calculate runway."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=50_000,
            capex=200_000,
            cash=300_000,
            total_debt=100_000,
        )
        result = analyzer.cash_burn_analysis(data)
        # FCF = 50k - 200k = -150k annual. Monthly burn = 12.5k. Runway = 300k/12.5k = 24.0 months
        assert result.cash_runway_months == pytest.approx(24.0, abs=1.0)

    def test_css_bonus(self, analyzer):
        """CSS >= 2.0 => +0.5."""
        data = FinancialData(
            revenue=2_000_000,
            operating_cash_flow=500_000,
            capex=100_000,
            dividends_paid=50_000,
            cash=200_000,
            total_debt=100_000,
        )
        result = analyzer.cash_burn_analysis(data)
        # FCF=0.20 (base 10) + CSS=3.33 (+0.5) + NCP>0 (+0.5) => 10 capped
        assert result.cb_score >= 10.0

    def test_negative_net_cash_penalty(self, analyzer):
        """NCP < -rev*0.5 => -0.5."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=120_000,
            capex=20_000,
            cash=10_000,
            total_debt=600_000,  # NCP = 10k - 600k = -590k < -500k
        )
        result = analyzer.cash_burn_analysis(data)
        # FCF=0.10 (base 7.0) + NCP=-590k (-0.5) => 6.5
        assert result.cb_score <= 7.0


# ===== EDGE CASES =====


class TestPhase79EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cash_burn_analysis(FinancialData())
        assert isinstance(result, CashBurnResult)
        assert result.ocf_margin is None

    def test_no_revenue(self, analyzer):
        """No revenue => empty result."""
        data = FinancialData(
            operating_cash_flow=100_000,
            capex=50_000,
        )
        result = analyzer.cash_burn_analysis(data)
        assert result.cb_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=0 => FCF margin negative."""
        data = FinancialData(
            revenue=1_000_000,
            capex=100_000,
            cash=50_000,
        )
        result = analyzer.cash_burn_analysis(data)
        # OCF=0, FCF = -100k, FCF margin = -0.10
        assert result.fcf_margin == pytest.approx(-0.10, abs=0.01)

    def test_no_capex_no_div(self, analyzer):
        """No outflows => CSS is None."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=200_000,
            cash=100_000,
        )
        result = analyzer.cash_burn_analysis(data)
        assert result.cash_self_sufficiency is None
