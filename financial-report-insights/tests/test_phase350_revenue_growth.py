"""Phase 350 Tests: Revenue Growth Capacity Analysis.

Tests for revenue_growth_analysis() and RevenueGrowthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    RevenueGrowthResult,
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


class TestRevenueGrowthDataclass:
    def test_defaults(self):
        r = RevenueGrowthResult()
        assert r.rg_capacity is None
        assert r.roe is None
        assert r.plowback is None
        assert r.sustainable_growth is None
        assert r.revenue_per_asset is None
        assert r.rg_spread is None
        assert r.rg_score == 0.0
        assert r.rg_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = RevenueGrowthResult(roe=0.125, rg_grade="Good")
        assert r.roe == 0.125
        assert r.rg_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestRevenueGrowthAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.revenue_growth_analysis(sample_data)
        assert isinstance(result, RevenueGrowthResult)

    def test_roe(self, analyzer, sample_data):
        """ROE = NI/TE = 150k/1.2M = 0.125."""
        result = analyzer.revenue_growth_analysis(sample_data)
        assert result.roe == pytest.approx(0.125, abs=0.001)

    def test_plowback(self, analyzer, sample_data):
        """Plowback = (NI-Div)/NI = (150k-40k)/150k = 0.733."""
        result = analyzer.revenue_growth_analysis(sample_data)
        assert result.plowback == pytest.approx(0.733, abs=0.005)

    def test_sustainable_growth(self, analyzer, sample_data):
        """SGR = ROE * Plowback = 0.125 * 0.733 = 0.0917."""
        result = analyzer.revenue_growth_analysis(sample_data)
        assert result.sustainable_growth == pytest.approx(0.0917, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.revenue_growth_analysis(sample_data)
        assert result.rg_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.revenue_growth_analysis(sample_data)
        assert "Revenue Growth" in result.summary


# ===== SCORING TESTS =====


class TestRevenueGrowthScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """SGR=0.0917 in [0.07,0.10)=>base 5.5. ROE=0.125>=0.12(+0.5). Both>0(+0.5). Score=6.5."""
        result = analyzer.revenue_growth_analysis(sample_data)
        assert result.rg_score == pytest.approx(6.5, abs=0.5)
        assert result.rg_grade in ["Good", "Adequate"]

    def test_high_growth(self, analyzer):
        """High SGR — strong reinvestment capacity."""
        data = FinancialData(
            net_income=300_000,
            total_equity=1_200_000,
            dividends_paid=30_000,
            total_assets=2_000_000,
            revenue=1_000_000,
        )
        # ROE=0.25, Plowback=0.90, SGR=0.225>=0.15=>base 10. ROE>=0.12(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.revenue_growth_analysis(data)
        assert result.rg_score >= 10.0
        assert result.rg_grade == "Excellent"

    def test_low_growth(self, analyzer):
        """Low SGR — limited growth capacity."""
        data = FinancialData(
            net_income=20_000,
            total_equity=1_200_000,
            dividends_paid=15_000,
            total_assets=2_000_000,
            revenue=1_000_000,
        )
        # ROE=0.0167, Plowback=0.25, SGR=0.0042<0.02=>base 1.0. ROE<0.12(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.revenue_growth_analysis(data)
        assert result.rg_score <= 3.0
        assert result.rg_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase350EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.revenue_growth_analysis(FinancialData())
        assert isinstance(result, RevenueGrowthResult)
        assert result.rg_score == 0.0

    def test_no_total_equity(self, analyzer):
        """TE=None => ROE=None => SGR=None => score 0."""
        data = FinancialData(net_income=150_000)
        result = analyzer.revenue_growth_analysis(data)
        assert result.roe is None
        assert result.rg_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => ROE=None."""
        data = FinancialData(total_equity=1_200_000)
        result = analyzer.revenue_growth_analysis(data)
        assert result.rg_score == 0.0
