"""Phase 25 Tests: Sustainability & Growth Capacity.

Tests for sustainable_growth_analysis() and SustainableGrowthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    SustainableGrowthResult,
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
        inventory=100_000,
        accounts_receivable=150_000,
        accounts_payable=80_000,
        total_debt=400_000,
        retained_earnings=600_000,
        depreciation=50_000,
        interest_expense=30_000,
        operating_cash_flow=220_000,
        capex=80_000,
    )


# ===== DATACLASS TESTS =====


class TestSustainableGrowthDataclass:
    def test_defaults(self):
        r = SustainableGrowthResult()
        assert r.sustainable_growth_rate is None
        assert r.internal_growth_rate is None
        assert r.retention_ratio is None
        assert r.growth_score == 0.0
        assert r.growth_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = SustainableGrowthResult(
            sustainable_growth_rate=0.125,
            retention_ratio=1.0,
            growth_grade="High Growth",
        )
        assert r.sustainable_growth_rate == 0.125
        assert r.retention_ratio == 1.0
        assert r.growth_grade == "High Growth"


# ===== CORE COMPUTATION TESTS =====


class TestSustainableGrowth:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert isinstance(result, SustainableGrowthResult)

    def test_roe(self, analyzer, sample_data):
        """ROE = NI / Equity = 150k / 1.2M = 12.5%."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.roe == pytest.approx(150_000 / 1_200_000, rel=0.01)

    def test_roa(self, analyzer, sample_data):
        """ROA = NI / TA = 150k / 2M = 7.5%."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.roa == pytest.approx(150_000 / 2_000_000, rel=0.01)

    def test_retention_no_dividends(self, analyzer, sample_data):
        """No dividends in sample => payout=0, retention=1.0."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.payout_ratio == pytest.approx(0.0)
        assert result.retention_ratio == pytest.approx(1.0)

    def test_sgr(self, analyzer, sample_data):
        """SGR = ROE * retention = 0.125 * 1.0 = 12.5%."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.sustainable_growth_rate == pytest.approx(0.125, rel=0.01)

    def test_igr(self, analyzer, sample_data):
        """IGR = ROA*b / (1 - ROA*b) = 0.075 / 0.925 = 8.108%."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        roa_b = 0.075 * 1.0
        expected = roa_b / (1 - roa_b)
        assert result.internal_growth_rate == pytest.approx(expected, rel=0.01)

    def test_plowback_capacity(self, analyzer, sample_data):
        """Plowback = retained_earnings / NI = 600k / 150k = 4.0."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.plowback_capacity == pytest.approx(4.0, rel=0.01)

    def test_equity_growth_rate(self, analyzer, sample_data):
        """Equity growth = retained_earnings / equity = 600k / 1.2M = 50%."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.equity_growth_rate == pytest.approx(0.50, rel=0.01)

    def test_reinvestment_rate(self, analyzer, sample_data):
        """Reinvestment = capex / depreciation = 80k / 50k = 1.6."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.reinvestment_rate == pytest.approx(1.6, rel=0.01)

    def test_growth_profitability_balance(self, analyzer, sample_data):
        """GPB = SGR / 0.10 = 0.125 / 0.10 = 1.25."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.growth_profitability_balance == pytest.approx(1.25, rel=0.01)

    def test_with_dividends(self, analyzer):
        """Payout = 30k/100k = 30%, retention = 70%."""
        data = FinancialData(
            net_income=100_000,
            total_equity=500_000,
            total_assets=1_000_000,
            dividends_paid=30_000,
            retained_earnings=200_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.payout_ratio == pytest.approx(0.30, rel=0.01)
        assert result.retention_ratio == pytest.approx(0.70, rel=0.01)
        # SGR = 0.20 * 0.70 = 0.14
        assert result.sustainable_growth_rate == pytest.approx(0.14, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.growth_grade in ["High Growth", "Sustainable", "Moderate", "Constrained"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert "Growth Capacity" in result.summary


# ===== SCORING TESTS =====


class TestGrowthScoring:
    def test_high_growth(self, analyzer):
        """High SGR, high ROE, full retention, high reinvestment."""
        data = FinancialData(
            revenue=5_000_000,
            net_income=1_500_000,
            total_assets=4_000_000,
            total_equity=3_000_000,
            retained_earnings=2_000_000,
            capex=200_000,
            depreciation=100_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        # ROE=0.50 -> SGR=0.50 >=0.15: +2.0 -> 7.0
        # ROE=0.50 >=0.20: +1.0 -> 8.0
        # retention=1.0 >=0.80: +1.0 -> 9.0
        # reinvest=2.0 >=1.5: +0.5 -> 9.5
        assert result.growth_grade == "High Growth"
        assert result.growth_score >= 8.0

    def test_constrained(self, analyzer):
        """Negative NI => constrained."""
        data = FinancialData(
            revenue=500_000,
            net_income=-100_000,
            total_assets=2_000_000,
            total_equity=500_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        # ROE=-0.20 <0: -1.5 -> 3.5
        # NI<0: no payout/retention/SGR
        assert result.growth_grade in ["Constrained", "Moderate"]

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            revenue=10_000_000,
            net_income=5_000_000,
            total_assets=5_000_000,
            total_equity=4_500_000,
            retained_earnings=4_000_000,
            capex=500_000,
            depreciation=200_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.growth_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            revenue=100_000,
            net_income=-500_000,
            total_assets=1_000_000,
            total_equity=100_000,
            capex=10_000,
            depreciation=100_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.growth_score >= 0.0


# ===== EDGE CASES =====


class TestPhase25EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.sustainable_growth_analysis(FinancialData())
        assert isinstance(result, SustainableGrowthResult)
        assert result.sustainable_growth_rate is None
        assert result.internal_growth_rate is None
        assert result.roe is None
        assert result.roa is None

    def test_zero_equity(self, analyzer):
        data = FinancialData(
            net_income=100_000,
            total_assets=1_000_000,
            total_equity=0,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.roe is None
        assert result.sustainable_growth_rate is None
        assert result.equity_growth_rate is None

    def test_zero_net_income(self, analyzer):
        """NI=0 means no payout/retention."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=0,
            total_assets=2_000_000,
            total_equity=1_000_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.payout_ratio is None
        assert result.retention_ratio is None
        assert result.sustainable_growth_rate is None

    def test_no_depreciation(self, analyzer):
        """No depreciation => reinvestment rate is None."""
        data = FinancialData(
            net_income=100_000,
            total_equity=500_000,
            total_assets=1_000_000,
            capex=50_000,
            depreciation=0,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.reinvestment_rate is None

    def test_no_retained_earnings(self, analyzer):
        data = FinancialData(
            net_income=100_000,
            total_equity=500_000,
            total_assets=1_000_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.plowback_capacity is None
        assert result.equity_growth_rate is None

    def test_high_payout_ratio(self, analyzer):
        """Dividends exceed NI (>100% payout) => negative retention."""
        data = FinancialData(
            net_income=100_000,
            total_equity=500_000,
            total_assets=1_000_000,
            dividends_paid=120_000,
        )
        result = analyzer.sustainable_growth_analysis(data)
        assert result.payout_ratio == pytest.approx(1.20, rel=0.01)
        assert result.retention_ratio == pytest.approx(-0.20, rel=0.01)
        # SGR = 0.20 * (-0.20) = -0.04
        assert result.sustainable_growth_rate == pytest.approx(-0.04, rel=0.01)

    def test_sample_data_grade(self, analyzer, sample_data):
        """Sample data: SGR=0.125>=0.08(+1)->6, ROE=0.125>=0.10(+0.5)->6.5, ret=1.0>=0.80(+1)->7.5, reinvest=1.6>=1.5(+0.5)->8.0."""
        result = analyzer.sustainable_growth_analysis(sample_data)
        assert result.growth_score == pytest.approx(8.0, abs=0.1)
        assert result.growth_grade == "High Growth"
