"""Phase 159 Tests: Internal Growth Capacity Analysis.

Tests for internal_growth_capacity_analysis() and InternalGrowthCapacityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    InternalGrowthCapacityResult,
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


class TestInternalGrowthCapacityDataclass:
    def test_defaults(self):
        r = InternalGrowthCapacityResult()
        assert r.sustainable_growth_rate is None
        assert r.internal_growth_rate is None
        assert r.plowback_ratio is None
        assert r.reinvestment_rate is None
        assert r.growth_financing_ratio is None
        assert r.equity_growth_rate is None
        assert r.igc_score == 0.0
        assert r.igc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = InternalGrowthCapacityResult(growth_financing_ratio=2.0, igc_grade="Good")
        assert r.growth_financing_ratio == 2.0
        assert r.igc_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestInternalGrowthCapacityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert isinstance(result, InternalGrowthCapacityResult)

    def test_plowback_ratio(self, analyzer, sample_data):
        """(NI-Div)/NI = (150k-40k)/150k = 0.733."""
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.plowback_ratio == pytest.approx(0.733, abs=0.01)

    def test_sustainable_growth_rate(self, analyzer, sample_data):
        """ROE * Plowback = (150k/1.2M) * 0.733 = 0.125 * 0.733 = 0.0917."""
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.sustainable_growth_rate == pytest.approx(0.0917, abs=0.01)

    def test_internal_growth_rate(self, analyzer, sample_data):
        """ROA * b / (1 - ROA * b) = (150k/2M) * 0.733 / (1 - 0.075*0.733)
        = 0.055 / (1 - 0.055) = 0.055 / 0.945 = 0.0582."""
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.internal_growth_rate == pytest.approx(0.058, abs=0.01)

    def test_reinvestment_rate(self, analyzer, sample_data):
        """CapEx/D&A = 80k/50k = 1.60."""
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.reinvestment_rate == pytest.approx(1.60, abs=0.1)

    def test_growth_financing_ratio(self, analyzer, sample_data):
        """OCF / (CapEx + Div) = 220k / (80k + 40k) = 220k/120k = 1.833."""
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.growth_financing_ratio == pytest.approx(1.833, abs=0.01)

    def test_equity_growth_rate(self, analyzer, sample_data):
        """RE/TE = 600k/1.2M = 0.50."""
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.equity_growth_rate == pytest.approx(0.50, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.igc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert "Internal Growth Capacity" in result.summary


# ===== SCORING TESTS =====


class TestInternalGrowthCapacityScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """GFR=1.833 => base 7.0. PB=0.733 >=0.60 => +0.5. RR=1.60 in [1.0,2.5] => +0.5. Score=8.0."""
        result = analyzer.internal_growth_capacity_analysis(sample_data)
        assert result.igc_score == pytest.approx(8.0, abs=0.5)
        assert result.igc_grade == "Excellent"

    def test_excellent_capacity(self, analyzer):
        """GFR >= 2.5 => base 10."""
        data = FinancialData(
            net_income=200_000,
            total_equity=800_000,
            total_assets=2_000_000,
            operating_cash_flow=400_000,
            capex=100_000,
            dividends_paid=50_000,
            depreciation=80_000,
            retained_earnings=500_000,
        )
        result = analyzer.internal_growth_capacity_analysis(data)
        assert result.igc_score >= 10.0
        assert result.igc_grade == "Excellent"

    def test_weak_capacity(self, analyzer):
        """GFR < 0.5 => base 1.0."""
        data = FinancialData(
            net_income=30_000,
            total_equity=500_000,
            total_assets=1_000_000,
            operating_cash_flow=50_000,
            capex=80_000,
            dividends_paid=40_000,
            depreciation=10_000,
            retained_earnings=50_000,
        )
        # GFR=50k/(80k+40k)=0.417 => 1.0. PB=(30k-40k)/30k=-0.33 <0.20 => -0.5. RR=80k/10k=8.0 >4.0 => -0.5. Score=0.0.
        result = analyzer.internal_growth_capacity_analysis(data)
        assert result.igc_score <= 1.0
        assert result.igc_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase159EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.internal_growth_capacity_analysis(FinancialData())
        assert isinstance(result, InternalGrowthCapacityResult)
        assert result.igc_score == 0.0

    def test_no_dividends(self, analyzer):
        """Div=None => plowback treats div as 0 => full retention."""
        data = FinancialData(
            net_income=150_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
            operating_cash_flow=220_000,
            capex=80_000,
            depreciation=50_000,
        )
        result = analyzer.internal_growth_capacity_analysis(data)
        assert result.plowback_ratio == pytest.approx(1.0, abs=0.01)
        assert result.growth_financing_ratio is not None

    def test_no_ocf(self, analyzer):
        """OCF=None => GFR=None, score 0."""
        data = FinancialData(
            net_income=150_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
            capex=80_000,
        )
        result = analyzer.internal_growth_capacity_analysis(data)
        assert result.growth_financing_ratio is None
        assert result.igc_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => plowback=None, SGR=None, IGR=None."""
        data = FinancialData(
            operating_cash_flow=220_000,
            capex=80_000,
            dividends_paid=40_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        result = analyzer.internal_growth_capacity_analysis(data)
        assert result.plowback_ratio is None
        assert result.sustainable_growth_rate is None
        assert result.internal_growth_rate is None
