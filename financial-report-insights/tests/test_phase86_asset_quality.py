"""Phase 86 Tests: Asset Quality Analysis.

Tests for asset_quality_analysis() and AssetQualityResult dataclass.
"""

import pytest

from financial_analyzer import (
    AssetQualityResult,
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


class TestAssetQualityDataclass:
    def test_defaults(self):
        r = AssetQualityResult()
        assert r.tangible_asset_ratio is None
        assert r.fixed_asset_ratio is None
        assert r.current_asset_ratio is None
        assert r.cash_to_current_assets is None
        assert r.receivables_to_assets is None
        assert r.inventory_to_assets is None
        assert r.aq_score == 0.0
        assert r.aq_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = AssetQualityResult(current_asset_ratio=0.40, aq_grade="Good")
        assert r.current_asset_ratio == 0.40
        assert r.aq_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestAssetQualityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.asset_quality_analysis(sample_data)
        assert isinstance(result, AssetQualityResult)

    def test_current_asset_ratio(self, analyzer, sample_data):
        """CA/TA = 500k/2M = 0.25."""
        result = analyzer.asset_quality_analysis(sample_data)
        assert result.current_asset_ratio == pytest.approx(0.25, abs=0.01)

    def test_fixed_asset_ratio(self, analyzer, sample_data):
        """(TA-CA)/TA = 1.5M/2M = 0.75."""
        result = analyzer.asset_quality_analysis(sample_data)
        assert result.fixed_asset_ratio == pytest.approx(0.75, abs=0.01)

    def test_cash_to_current_assets(self, analyzer, sample_data):
        """Cash/CA = 50k/500k = 0.10."""
        result = analyzer.asset_quality_analysis(sample_data)
        assert result.cash_to_current_assets == pytest.approx(0.10, abs=0.01)

    def test_receivables_to_assets(self, analyzer, sample_data):
        """AR/TA = 150k/2M = 0.075."""
        result = analyzer.asset_quality_analysis(sample_data)
        assert result.receivables_to_assets == pytest.approx(0.075, abs=0.005)

    def test_inventory_to_assets(self, analyzer, sample_data):
        """Inv/TA = 100k/2M = 0.05."""
        result = analyzer.asset_quality_analysis(sample_data)
        assert result.inventory_to_assets == pytest.approx(0.05, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.asset_quality_analysis(sample_data)
        assert result.aq_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.asset_quality_analysis(sample_data)
        assert "Asset" in result.summary or "Quality" in result.summary


# ===== SCORING TESTS =====


class TestAssetQualityScoring:
    def test_very_liquid_assets(self, analyzer):
        """CAR >= 0.60 => base 10."""
        data = FinancialData(
            total_assets=1_000_000,
            current_assets=700_000,
            cash=300_000,
            accounts_receivable=50_000,
            inventory=100_000,
        )
        result = analyzer.asset_quality_analysis(data)
        # CAR=0.70 (base 10). CCA=300k/700k=0.43 (>=0.30 => +0.5). AR/TA=0.05 (<=0.10 => +0.5) => 10 capped
        assert result.aq_score >= 10.0
        assert result.aq_grade == "Excellent"

    def test_moderate_assets(self, analyzer, sample_data):
        """CAR=0.25 => base 4.0."""
        result = analyzer.asset_quality_analysis(sample_data)
        # CAR=0.25 (base 4.0). CCA=0.10 (no adj). AR/TA=0.075 (<=0.10 => +0.5) => 4.5
        assert result.aq_score >= 4.0

    def test_low_current_assets(self, analyzer):
        """CAR < 0.10 => base 1.0."""
        data = FinancialData(
            total_assets=5_000_000,
            current_assets=300_000,
            cash=10_000,
            accounts_receivable=50_000,
            inventory=100_000,
        )
        result = analyzer.asset_quality_analysis(data)
        # CAR=0.06 (base 1.0). CCA=0.033 (<0.05 => -0.5). AR/TA=0.01 (<=0.10 => +0.5) => 1.0
        assert result.aq_score <= 2.0

    def test_high_ar_penalty(self, analyzer):
        """AR/TA > 0.40 => -0.5."""
        data = FinancialData(
            total_assets=1_000_000,
            current_assets=600_000,
            cash=50_000,
            accounts_receivable=450_000,
            inventory=50_000,
        )
        result = analyzer.asset_quality_analysis(data)
        # CAR=0.60 (base 10). CCA=50k/600k=0.083 (no adj). AR/TA=0.45 (>0.40 => -0.5) => 9.5
        assert result.aq_score <= 10.0


# ===== EDGE CASES =====


class TestPhase86EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.asset_quality_analysis(FinancialData())
        assert isinstance(result, AssetQualityResult)
        assert result.current_asset_ratio is None

    def test_no_total_assets(self, analyzer):
        """TA=0 => empty result."""
        data = FinancialData(
            current_assets=500_000,
            cash=100_000,
        )
        result = analyzer.asset_quality_analysis(data)
        assert result.aq_score == 0.0

    def test_no_current_assets(self, analyzer):
        """CA=0 => cash_to_current_assets is None."""
        data = FinancialData(
            total_assets=1_000_000,
        )
        result = analyzer.asset_quality_analysis(data)
        assert result.cash_to_current_assets is None

    def test_all_current_assets(self, analyzer):
        """CA = TA => fixed_asset_ratio = 0."""
        data = FinancialData(
            total_assets=500_000,
            current_assets=500_000,
            cash=200_000,
            accounts_receivable=100_000,
        )
        result = analyzer.asset_quality_analysis(data)
        assert result.fixed_asset_ratio == pytest.approx(0.0, abs=0.01)
        assert result.current_asset_ratio == pytest.approx(1.0, abs=0.01)
