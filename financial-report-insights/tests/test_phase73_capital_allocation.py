"""Phase 73 Tests: Capital Allocation Analysis.

Tests for capital_allocation_analysis() and CapitalAllocationResult dataclass.
"""

import pytest

from financial_analyzer import (
    CapitalAllocationResult,
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


class TestCapitalAllocationDataclass:
    def test_defaults(self):
        r = CapitalAllocationResult()
        assert r.capex_to_revenue is None
        assert r.capex_to_ocf is None
        assert r.shareholder_return_ratio is None
        assert r.reinvestment_rate is None
        assert r.fcf_yield is None
        assert r.total_payout_to_fcf is None
        assert r.ca_score == 0.0
        assert r.ca_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CapitalAllocationResult(capex_to_revenue=0.08, ca_grade="Good")
        assert r.capex_to_revenue == 0.08
        assert r.ca_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestCapitalAllocationAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.capital_allocation_analysis(sample_data)
        assert isinstance(result, CapitalAllocationResult)

    def test_capex_to_revenue(self, analyzer, sample_data):
        """CapEx/Rev = 80k/1M = 0.08."""
        result = analyzer.capital_allocation_analysis(sample_data)
        assert result.capex_to_revenue == pytest.approx(0.08, abs=0.01)

    def test_capex_to_ocf(self, analyzer, sample_data):
        """CapEx/OCF = 80k/220k = 0.364."""
        result = analyzer.capital_allocation_analysis(sample_data)
        assert result.capex_to_ocf == pytest.approx(0.364, abs=0.01)

    def test_shareholder_return_ratio(self, analyzer, sample_data):
        """(Div+BB)/NI = (40k+0)/150k = 0.267."""
        result = analyzer.capital_allocation_analysis(sample_data)
        assert result.shareholder_return_ratio == pytest.approx(0.267, abs=0.01)

    def test_reinvestment_rate(self, analyzer, sample_data):
        """CapEx/Dep = 80k/50k = 1.6."""
        result = analyzer.capital_allocation_analysis(sample_data)
        assert result.reinvestment_rate == pytest.approx(1.6, abs=0.1)

    def test_fcf_yield(self, analyzer, sample_data):
        """FCF/Rev = (220k-80k)/1M = 0.14."""
        result = analyzer.capital_allocation_analysis(sample_data)
        assert result.fcf_yield == pytest.approx(0.14, abs=0.01)

    def test_total_payout_to_fcf(self, analyzer, sample_data):
        """(Div+BB)/FCF = 40k/140k = 0.286."""
        result = analyzer.capital_allocation_analysis(sample_data)
        assert result.total_payout_to_fcf == pytest.approx(0.286, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.capital_allocation_analysis(sample_data)
        assert result.ca_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.capital_allocation_analysis(sample_data)
        assert "Capital Allocation" in result.summary


# ===== SCORING TESTS =====


class TestCapitalAllocationScoring:
    def test_very_low_capex_ratio(self, analyzer):
        """CapEx/OCF <= 0.25 => base 10."""
        data = FinancialData(
            revenue=5_000_000,
            net_income=1_000_000,
            depreciation=200_000,
            operating_cash_flow=2_000_000,
            capex=400_000,
            dividends_paid=200_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        # CTO=0.20 (base 10) + FCF yield=0.32 (+0.5) + payout/FCF=0.125 (+0.5) => capped 10
        assert result.ca_score >= 9.0
        assert result.ca_grade == "Excellent"

    def test_moderate_capex_ratio(self, analyzer, sample_data):
        """CapEx/OCF ~ 0.364 => base 7.0 (>0.35)."""
        result = analyzer.capital_allocation_analysis(sample_data)
        # CTO=0.364 (base 7.0) + FCF yield=0.14 (+0.5) + payout/FCF=0.286 (+0.5) => 8.0
        assert result.ca_score >= 8.0

    def test_high_capex_ratio(self, analyzer):
        """CapEx/OCF = 0.70 => base 5.5."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            depreciation=80_000,
            operating_cash_flow=200_000,
            capex=140_000,
            dividends_paid=30_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        # CTO=0.70 (base 5.5) + FCF yield=0.06 (no bonus) + payout/FCF=0.50 (+0.5) => 6.0
        assert result.ca_score <= 7.0

    def test_excessive_capex(self, analyzer):
        """CapEx/OCF > 0.95 => base 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=50_000,
            depreciation=60_000,
            operating_cash_flow=100_000,
            capex=98_000,
            dividends_paid=10_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        # CTO=0.98 (base 1.0) + FCF yield=0.002 (no bonus) + payout/FCF: FCF=2k, payout=10k => no TPF since 10k/2k=5.0 >=1.0 => -0.5
        assert result.ca_score < 3.0
        assert result.ca_grade == "Weak"

    def test_fcf_yield_bonus(self, analyzer):
        """FCF yield >= 0.10 => +0.5."""
        data = FinancialData(
            revenue=2_000_000,
            net_income=400_000,
            depreciation=100_000,
            operating_cash_flow=800_000,
            capex=300_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        # CTO=0.375 (base 7.0) + FCF yield=0.25 (+0.5) => 7.5
        assert result.ca_score >= 7.0

    def test_negative_fcf_penalty(self, analyzer):
        """FCF < 0 => FCF yield < 0 => -0.5."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=50_000,
            depreciation=50_000,
            operating_cash_flow=100_000,
            capex=120_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        # CTO=1.2 (base 1.0) + FCF yield=-0.02 (-0.5) => 0.5
        assert result.ca_score < 2.0


# ===== EDGE CASES =====


class TestPhase73EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.capital_allocation_analysis(FinancialData())
        assert isinstance(result, CapitalAllocationResult)
        assert result.capex_to_revenue is None

    def test_no_capex_no_div_no_bb(self, analyzer):
        """No capital allocation activity => empty result."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=150_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        assert result.ca_score == 0.0

    def test_dividends_only(self, analyzer):
        """Only dividends, no capex."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=150_000,
            operating_cash_flow=200_000,
            dividends_paid=60_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        # capex=0, CTO = safe_divide(0, 200k) = 0.0 <= 0.25 => base 10
        assert result.ca_score >= 9.0

    def test_buybacks_included(self, analyzer):
        """Share buybacks affect shareholder return ratio."""
        data = FinancialData(
            revenue=2_000_000,
            net_income=300_000,
            depreciation=100_000,
            operating_cash_flow=500_000,
            capex=100_000,
            dividends_paid=50_000,
            share_buybacks=80_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        # SRR = (50k+80k)/300k = 0.433
        assert result.shareholder_return_ratio == pytest.approx(0.433, abs=0.01)

    def test_no_depreciation(self, analyzer):
        """No depreciation => reinvestment_rate is None."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            operating_cash_flow=200_000,
            capex=50_000,
        )
        result = analyzer.capital_allocation_analysis(data)
        assert result.reinvestment_rate is None
