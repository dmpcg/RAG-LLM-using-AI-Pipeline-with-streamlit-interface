"""Phase 327 Tests: Noncurrent Asset Ratio Analysis.

Tests for noncurrent_asset_ratio_analysis() and NoncurrentAssetRatioResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    NoncurrentAssetRatioResult,
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


class TestNoncurrentAssetRatioDataclass:
    def test_defaults(self):
        r = NoncurrentAssetRatioResult()
        assert r.nca_ratio is None
        assert r.current_asset_ratio is None
        assert r.nca_to_equity is None
        assert r.nca_to_debt is None
        assert r.asset_structure_spread is None
        assert r.liquidity_complement is None
        assert r.nar_score == 0.0
        assert r.nar_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = NoncurrentAssetRatioResult(nca_ratio=0.75, nar_grade="Excellent")
        assert r.nca_ratio == 0.75
        assert r.nar_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestNoncurrentAssetRatioAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert isinstance(result, NoncurrentAssetRatioResult)

    def test_nca_ratio(self, analyzer, sample_data):
        """NCA = (2M - 500k) / 2M = 1.5M / 2M = 0.75."""
        result = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert result.nca_ratio == pytest.approx(0.75, abs=0.01)

    def test_current_asset_ratio(self, analyzer, sample_data):
        """CA/TA = 500k/2M = 0.25."""
        result = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert result.current_asset_ratio == pytest.approx(0.25, abs=0.01)

    def test_nca_to_equity(self, analyzer, sample_data):
        """NCA/TE = 1.5M/1.2M = 1.25."""
        result = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert result.nca_to_equity == pytest.approx(1.25, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert result.nar_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert "Noncurrent Asset Ratio" in result.summary


# ===== SCORING TESTS =====


class TestNoncurrentAssetRatioScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """NCA=0.75 in (0.65,0.75]=>base 8.5. NCA/Eq=1.25>1.0(no adj). Both>0(+0.5). Score=9.0."""
        result = analyzer.noncurrent_asset_ratio_analysis(sample_data)
        assert result.nar_score == pytest.approx(9.0, abs=0.5)
        assert result.nar_grade == "Excellent"

    def test_balanced_structure(self, analyzer):
        """Moderate NCA ratio — ideal balance."""
        data = FinancialData(
            total_assets=1_000_000,
            current_assets=500_000,
            total_equity=600_000,
        )
        # NCA = 500k/1M = 0.50 in [0.40,0.65]=>base 10. NCA/Eq=500k/600k=0.833<=1.0(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.noncurrent_asset_ratio_analysis(data)
        assert result.nar_score >= 9.5
        assert result.nar_grade == "Excellent"

    def test_high_nca(self, analyzer):
        """Very high NCA — illiquid asset-heavy."""
        data = FinancialData(
            total_assets=1_000_000,
            current_assets=50_000,
            total_equity=300_000,
        )
        # NCA = 950k/1M = 0.95 > 0.90 => base 4.0. NCA/Eq=950k/300k=3.17>1.0(no adj). Both>0(+0.5). Score=4.5.
        result = analyzer.noncurrent_asset_ratio_analysis(data)
        assert result.nar_score <= 5.0
        assert result.nar_grade == "Adequate"


# ===== EDGE CASES =====


class TestPhase327EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.noncurrent_asset_ratio_analysis(FinancialData())
        assert isinstance(result, NoncurrentAssetRatioResult)
        assert result.nar_score == 0.0

    def test_no_current_assets(self, analyzer):
        """CA=0 => NCA ratio = 1.0 (all noncurrent)."""
        data = FinancialData(total_assets=1_000_000)
        result = analyzer.noncurrent_asset_ratio_analysis(data)
        assert result.nca_ratio == pytest.approx(1.0, abs=0.01)
        assert result.nar_score > 0.0
