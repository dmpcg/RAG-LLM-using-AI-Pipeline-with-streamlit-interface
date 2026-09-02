"""Phase 52 Tests: EBITDA Margin Quality Analysis.

Tests for ebitda_margin_quality_analysis() and EbitdaMarginQualityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    EbitdaMarginQualityResult,
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
    )


# ===== DATACLASS TESTS =====


class TestEbitdaMarginQualityDataclass:
    def test_defaults(self):
        r = EbitdaMarginQualityResult()
        assert r.ebitda_margin_pct is None
        assert r.operating_margin_pct is None
        assert r.da_intensity is None
        assert r.ebitda_oi_spread is None
        assert r.ebitda_to_gp is None
        assert r.ebitda_score == 0.0
        assert r.ebitda_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = EbitdaMarginQualityResult(
            ebitda_margin_pct=25.0,
            ebitda_grade="Good",
        )
        assert r.ebitda_margin_pct == 25.0
        assert r.ebitda_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestEbitdaMarginQualityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert isinstance(result, EbitdaMarginQualityResult)

    def test_ebitda_margin(self, analyzer, sample_data):
        """EBITDA/Revenue = 250k/1M = 25%."""
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert result.ebitda_margin_pct == pytest.approx(25.0, abs=0.1)

    def test_operating_margin(self, analyzer, sample_data):
        """OI/Revenue = 200k/1M = 20%."""
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert result.operating_margin_pct == pytest.approx(20.0, abs=0.1)

    def test_da_intensity(self, analyzer, sample_data):
        """D&A/Revenue = 50k/1M = 5%."""
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert result.da_intensity == pytest.approx(5.0, abs=0.1)

    def test_ebitda_oi_spread(self, analyzer, sample_data):
        """EBITDA margin - OI margin = 25% - 20% = 5%."""
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert result.ebitda_oi_spread == pytest.approx(5.0, abs=0.1)

    def test_ebitda_to_gp(self, analyzer, sample_data):
        """EBITDA/GP = 250k/400k = 0.625."""
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert result.ebitda_to_gp == pytest.approx(0.625, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert result.ebitda_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert "EBITDA" in result.summary


# ===== SCORING TESTS =====


class TestEbitdaMarginQualityScoring:
    def test_high_margin(self, analyzer):
        """EBITDA 45% => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=450_000,
            operating_income=400_000,
            gross_profit=600_000,
            depreciation=50_000,
        )
        result = analyzer.ebitda_margin_quality_analysis(data)
        assert result.ebitda_score >= 8.0
        assert result.ebitda_grade == "Excellent"

    def test_moderate_margin(self, analyzer, sample_data):
        """EBITDA 25% => base 7.0 + adjustments."""
        result = analyzer.ebitda_margin_quality_analysis(sample_data)
        assert result.ebitda_score >= 6.0

    def test_low_margin(self, analyzer):
        """EBITDA 8% => base 2.5."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=80_000,
            operating_income=30_000,
            depreciation=50_000,
        )
        result = analyzer.ebitda_margin_quality_analysis(data)
        assert result.ebitda_score < 5.0

    def test_negative_ebitda(self, analyzer):
        """Negative EBITDA => base 0.5, Weak."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=-50_000,
            operating_income=-100_000,
            depreciation=50_000,
        )
        result = analyzer.ebitda_margin_quality_analysis(data)
        assert result.ebitda_score < 4.0
        assert result.ebitda_grade == "Weak"

    def test_asset_light_bonus(self, analyzer):
        """Low D&A (<3%) gets +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=250_000,
            operating_income=240_000,
            depreciation=10_000,
            gross_profit=400_000,
        )
        result = analyzer.ebitda_margin_quality_analysis(data)
        # Base 7.0 (25%), D&A <3% => +0.5, EBITDA/GP=0.625 => no adj
        assert result.ebitda_score >= 7.0


# ===== EDGE CASES =====


class TestPhase52EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.ebitda_margin_quality_analysis(FinancialData())
        assert isinstance(result, EbitdaMarginQualityResult)
        assert result.ebitda_margin_pct is None

    def test_no_revenue(self, analyzer):
        data = FinancialData(ebitda=100_000)
        result = analyzer.ebitda_margin_quality_analysis(data)
        assert result.ebitda_margin_pct is None

    def test_revenue_only(self, analyzer):
        """Revenue but no EBITDA => ebitda_margin is None."""
        data = FinancialData(revenue=500_000)
        result = analyzer.ebitda_margin_quality_analysis(data)
        assert result.ebitda_margin_pct is None

    def test_no_depreciation(self, analyzer):
        """No D&A => da_intensity None."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=200_000,
            operating_income=200_000,
        )
        result = analyzer.ebitda_margin_quality_analysis(data)
        assert result.ebitda_margin_pct == pytest.approx(20.0, abs=0.1)
        assert result.da_intensity is None
