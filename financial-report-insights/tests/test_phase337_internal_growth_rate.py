"""Phase 337 Tests: Internal Growth Rate Analysis.

Tests for internal_growth_rate_analysis() and InternalGrowthRateResult dataclass.
"""

import pytest
from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    InternalGrowthRateResult,
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

class TestInternalGrowthRateDataclass:
    def test_defaults(self):
        r = InternalGrowthRateResult()
        assert r.igr is None
        assert r.roa is None
        assert r.retention_ratio is None
        assert r.roa_times_b is None
        assert r.sustainable_growth is None
        assert r.growth_capacity is None
        assert r.igr_score == 0.0
        assert r.igr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = InternalGrowthRateResult(igr=0.08, igr_grade="Good")
        assert r.igr == 0.08
        assert r.igr_grade == "Good"


# ===== CORE COMPUTATION TESTS =====

class TestInternalGrowthRateAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.internal_growth_rate_analysis(sample_data)
        assert isinstance(result, InternalGrowthRateResult)

    def test_roa(self, analyzer, sample_data):
        """ROA = 150k/2M = 0.075."""
        result = analyzer.internal_growth_rate_analysis(sample_data)
        assert result.roa == pytest.approx(0.075, abs=0.001)

    def test_retention_ratio(self, analyzer, sample_data):
        """b = (150k-40k)/150k = 0.7333."""
        result = analyzer.internal_growth_rate_analysis(sample_data)
        assert result.retention_ratio == pytest.approx(0.7333, abs=0.005)

    def test_igr(self, analyzer, sample_data):
        """IGR = (0.075*0.7333)/(1-0.075*0.7333) = 0.055/0.945 = 0.0582."""
        result = analyzer.internal_growth_rate_analysis(sample_data)
        assert result.igr == pytest.approx(0.0582, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.internal_growth_rate_analysis(sample_data)
        assert result.igr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.internal_growth_rate_analysis(sample_data)
        assert "Internal Growth Rate" in result.summary


# ===== SCORING TESTS =====

class TestInternalGrowthRateScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """IGR=5.82% in [5,8)=>base 5.5. b=0.733>=0.70(+0.5). Both>0(+0.5). Score=6.5."""
        result = analyzer.internal_growth_rate_analysis(sample_data)
        assert result.igr_score == pytest.approx(6.5, abs=0.5)
        assert result.igr_grade == "Good"

    def test_high_growth(self, analyzer):
        """High IGR — high ROA and high retention."""
        data = FinancialData(
            net_income=400_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
            dividends_paid=20_000,
        )
        # ROA=0.20, b=(400k-20k)/400k=0.95, ROA*b=0.19, IGR=0.19/0.81=0.2346=23.46%
        # >=15=>base 10. b=0.95>=0.70(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.internal_growth_rate_analysis(data)
        assert result.igr_score >= 10.0
        assert result.igr_grade == "Excellent"

    def test_low_growth(self, analyzer):
        """Low IGR — low ROA."""
        data = FinancialData(
            net_income=20_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
            dividends_paid=15_000,
        )
        # ROA=0.01, b=(20k-15k)/20k=0.25, ROA*b=0.0025, IGR=0.0025/0.9975=0.25%
        # 0.25% in [0,3)=>base 2.5. b=0.25<0.70(no adj). Both>0(+0.5). Score=3.0.
        result = analyzer.internal_growth_rate_analysis(data)
        assert result.igr_score <= 4.0
        assert result.igr_grade in ["Weak", "Adequate"]


# ===== EDGE CASES =====

class TestPhase337EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.internal_growth_rate_analysis(FinancialData())
        assert isinstance(result, InternalGrowthRateResult)
        assert result.igr_score == 0.0

    def test_no_assets(self, analyzer):
        """TA=0 => ROA=None => IGR=None."""
        data = FinancialData(net_income=150_000)
        result = analyzer.internal_growth_rate_analysis(data)
        assert result.igr is None

    def test_no_net_income(self, analyzer):
        """NI=0 => ROA=0 => IGR=0 => weak."""
        data = FinancialData(total_assets=2_000_000)
        result = analyzer.internal_growth_rate_analysis(data)
        assert result.igr_score == 0.0


# ===== WP-B / P0-13: safe_divide threshold band tests =====

class TestIGRSafeDivideThreshold:
    """Pin the INTENTIONAL threshold change from the hand-rolled 1e-9 guard
    to safe_divide's 1e-12 guard (D5 in WS2-ANALYTICS-PERF-PLAN.md).

    safe_divide returns default (None) only when abs(denom) < 1e-12.
    The old hand-rolled guard returned None when abs(denom) <= 1e-9.
    Band [1e-12, 1e-9) now yields a large finite IGR (accepted change).
    """

    def _make_data_for_roa_b(self, roa_b: float):
        """Construct FinancialData so that roa*b == roa_b exactly.

        We set NI = roa_b * TA (no dividends so b=1) which gives:
            ROA = NI / TA = roa_b
            b   = (NI - 0) / NI = 1.0
            roa_b_product = roa_b * 1.0 = roa_b
        """
        ta = 1_000_000.0
        ni = roa_b * ta
        return FinancialData(
            net_income=ni,
            total_assets=ta,
            total_equity=ta,
            dividends_paid=0.0,
        )

    def test_denom_below_1e12_returns_none(self, analyzer):
        """abs(1 - roa_b) < 1e-12 => safe_divide returns None."""
        # roa_b = 1.0 - 5e-13  =>  denom = 5e-13  <  1e-12  =>  None
        roa_b = 1.0 - 5e-13
        data = self._make_data_for_roa_b(roa_b)
        result = analyzer.internal_growth_rate_analysis(data)
        assert result.igr is None

    def test_denom_in_band_1e12_to_1e9_returns_finite(self, analyzer):
        """abs(1 - roa_b) in [1e-12, 1e-9) => large finite IGR (new behavior, was None).

        This documents the INTENTIONAL threshold relaxation (D5): the old
        hand-rolled guard treated this band as None; safe_divide does not.
        """
        # roa_b = 1.0 - 5e-11  =>  denom = 5e-11  in [1e-12, 1e-9)
        roa_b = 1.0 - 5e-11
        data = self._make_data_for_roa_b(roa_b)
        result = analyzer.internal_growth_rate_analysis(data)
        # Must be a large finite number, not None
        assert result.igr is not None
        assert result.igr > 1e6  # ~1/5e-11 = 2e10

    def test_normal_denom_value_unchanged(self, analyzer):
        """Normal denominator: IGR formula gives expected value."""
        # Use sample_data values: ROA=0.075, b=0.7333, roa_b~0.055
        # denom = 1 - 0.055 = 0.945, IGR ~ 0.0582
        data = FinancialData(
            net_income=150_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
            dividends_paid=40_000,
        )
        result = analyzer.internal_growth_rate_analysis(data)
        assert result.igr == pytest.approx(0.0582, abs=0.005)
