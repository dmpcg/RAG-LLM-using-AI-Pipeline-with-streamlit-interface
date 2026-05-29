"""Phase 346 Tests: Net Worth Growth Analysis.

Tests for net_worth_growth_analysis() and NetWorthGrowthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    NetWorthGrowthResult,
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


class TestNetWorthGrowthDataclass:
    def test_defaults(self):
        r = NetWorthGrowthResult()
        assert r.nw_growth_ratio is None
        assert r.re_to_equity is None
        assert r.equity_to_assets is None
        assert r.ni_to_equity is None
        assert r.plowback_rate is None
        assert r.nw_spread is None
        assert r.nwg_score == 0.0
        assert r.nwg_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = NetWorthGrowthResult(re_to_equity=0.50, nwg_grade="Good")
        assert r.re_to_equity == 0.50
        assert r.nwg_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestNetWorthGrowthAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.net_worth_growth_analysis(sample_data)
        assert isinstance(result, NetWorthGrowthResult)

    def test_re_to_equity(self, analyzer, sample_data):
        """RE/TE = 600k/1.2M = 0.50."""
        result = analyzer.net_worth_growth_analysis(sample_data)
        assert result.re_to_equity == pytest.approx(0.50, abs=0.001)

    def test_equity_to_assets(self, analyzer, sample_data):
        """TE/TA = 1.2M/2M = 0.60."""
        result = analyzer.net_worth_growth_analysis(sample_data)
        assert result.equity_to_assets == pytest.approx(0.60, abs=0.001)

    def test_plowback_rate(self, analyzer, sample_data):
        """(NI - Div)/NI = (150k-40k)/150k = 0.733."""
        result = analyzer.net_worth_growth_analysis(sample_data)
        assert result.plowback_rate == pytest.approx(0.733, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.net_worth_growth_analysis(sample_data)
        assert result.nwg_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.net_worth_growth_analysis(sample_data)
        assert "Net Worth Growth" in result.summary


# ===== SCORING TESTS =====


class TestNetWorthGrowthScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """RE/TE=0.50 in [0.50,0.60)=>base 7.0. EA=0.60>=0.40(+0.5). Both>0(+0.5). Score=8.0."""
        result = analyzer.net_worth_growth_analysis(sample_data)
        assert result.nwg_score == pytest.approx(8.0, abs=0.5)
        assert result.nwg_grade in ["Good", "Excellent"]

    def test_high_net_worth_growth(self, analyzer):
        """High RE/TE — strong self-funded growth."""
        data = FinancialData(
            retained_earnings=900_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        # RE/TE=0.75>=0.70=>base 10. EA=0.60>=0.40(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.net_worth_growth_analysis(data)
        assert result.nwg_score >= 10.0
        assert result.nwg_grade == "Excellent"

    def test_low_net_worth_growth(self, analyzer):
        """Low RE/TE — weak self-funded growth."""
        data = FinancialData(
            retained_earnings=50_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        # RE/TE=0.042<0.10=>base 1.0. EA=0.60>=0.40(+0.5). Both>0(+0.5). Score=2.0.
        result = analyzer.net_worth_growth_analysis(data)
        assert result.nwg_score <= 3.0
        assert result.nwg_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase346EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.net_worth_growth_analysis(FinancialData())
        assert isinstance(result, NetWorthGrowthResult)
        assert result.nwg_score == 0.0

    def test_no_total_equity(self, analyzer):
        """TE=None => RE/TE=None => score 0."""
        data = FinancialData(retained_earnings=600_000)
        result = analyzer.net_worth_growth_analysis(data)
        assert result.re_to_equity is None
        assert result.nwg_score == 0.0

    def test_no_retained_earnings(self, analyzer):
        """RE=None => ratio=None."""
        data = FinancialData(total_equity=1_200_000)
        result = analyzer.net_worth_growth_analysis(data)
        assert result.nwg_score == 0.0
