"""Phase 341 Tests: Asset Lightness Analysis.

Tests for asset_lightness_analysis() and AssetLightnessResult dataclass.
"""

import pytest

from financial_analyzer import (
    AssetLightnessResult,
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


class TestAssetLightnessDataclass:
    def test_defaults(self):
        r = AssetLightnessResult()
        assert r.lightness_ratio is None
        assert r.ca_to_ta is None
        assert r.revenue_to_assets is None
        assert r.intangible_intensity is None
        assert r.fixed_asset_ratio is None
        assert r.lightness_spread is None
        assert r.alt_score == 0.0
        assert r.alt_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = AssetLightnessResult(lightness_ratio=0.60, alt_grade="Good")
        assert r.lightness_ratio == 0.60
        assert r.alt_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestAssetLightnessAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.asset_lightness_analysis(sample_data)
        assert isinstance(result, AssetLightnessResult)

    def test_lightness_ratio(self, analyzer, sample_data):
        """CA/TA = 500k/2M = 0.25."""
        result = analyzer.asset_lightness_analysis(sample_data)
        assert result.lightness_ratio == pytest.approx(0.25, abs=0.001)

    def test_revenue_to_assets(self, analyzer, sample_data):
        """Revenue/TA = 1M/2M = 0.50."""
        result = analyzer.asset_lightness_analysis(sample_data)
        assert result.revenue_to_assets == pytest.approx(0.50, abs=0.001)

    def test_fixed_asset_ratio(self, analyzer, sample_data):
        """Fixed ratio = 1 - 0.25 = 0.75."""
        result = analyzer.asset_lightness_analysis(sample_data)
        assert result.fixed_asset_ratio == pytest.approx(0.75, abs=0.001)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.asset_lightness_analysis(sample_data)
        assert result.alt_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.asset_lightness_analysis(sample_data)
        assert "Asset Lightness" in result.summary


# ===== SCORING TESTS =====


class TestAssetLightnessScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """CA/TA=0.25 in [0.15,0.30)=>base 2.5. Rev/TA=0.50>=0.50(+0.5). Both>0(+0.5). Score=3.5."""
        result = analyzer.asset_lightness_analysis(sample_data)
        assert result.alt_score == pytest.approx(3.5, abs=0.5)
        assert result.alt_grade in ["Weak", "Adequate"]

    def test_high_lightness(self, analyzer):
        """High CA/TA — very asset-light."""
        data = FinancialData(
            current_assets=1_500_000,
            total_assets=2_000_000,
            revenue=1_500_000,
        )
        # CA/TA=0.75>=0.70=>base 10. Rev/TA=0.75>=0.50(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.asset_lightness_analysis(data)
        assert result.alt_score >= 10.0
        assert result.alt_grade == "Excellent"

    def test_low_lightness(self, analyzer):
        """Low CA/TA — heavy fixed assets."""
        data = FinancialData(
            current_assets=100_000,
            total_assets=2_000_000,
            revenue=500_000,
        )
        # CA/TA=0.05<0.15=>base 1.0. Rev/TA=0.25<0.50(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.asset_lightness_analysis(data)
        assert result.alt_score <= 3.0
        assert result.alt_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase341EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.asset_lightness_analysis(FinancialData())
        assert isinstance(result, AssetLightnessResult)
        assert result.alt_score == 0.0

    def test_no_total_assets(self, analyzer):
        """TA=None => lightness=None."""
        data = FinancialData(current_assets=500_000)
        result = analyzer.asset_lightness_analysis(data)
        assert result.lightness_ratio is None

    def test_no_current_assets(self, analyzer):
        """CA=None => lightness=None."""
        data = FinancialData(total_assets=2_000_000)
        result = analyzer.asset_lightness_analysis(data)
        assert result.alt_score == 0.0
