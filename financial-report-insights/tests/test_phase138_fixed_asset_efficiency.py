"""Phase 138 Tests: Fixed Asset Efficiency Analysis.

Tests for fixed_asset_efficiency_analysis() and FixedAssetEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FixedAssetEfficiencyResult,
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


class TestFixedAssetEfficiencyDataclass:
    def test_defaults(self):
        r = FixedAssetEfficiencyResult()
        assert r.fixed_asset_ratio is None
        assert r.fixed_asset_turnover is None
        assert r.fixed_to_equity is None
        assert r.fixed_asset_coverage is None
        assert r.depreciation_to_fixed is None
        assert r.capex_to_fixed is None
        assert r.fae_score == 0.0
        assert r.fae_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FixedAssetEfficiencyResult(fixed_asset_turnover=3.5, fae_grade="Good")
        assert r.fixed_asset_turnover == 3.5
        assert r.fae_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestFixedAssetEfficiencyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert isinstance(result, FixedAssetEfficiencyResult)

    def test_fixed_asset_ratio(self, analyzer, sample_data):
        """FAR = (2M - 500k) / 2M = 0.75."""
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.fixed_asset_ratio == pytest.approx(0.75, abs=0.01)

    def test_fixed_asset_turnover(self, analyzer, sample_data):
        """FAT = 1M / (2M - 500k) = 0.667."""
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.fixed_asset_turnover == pytest.approx(0.667, abs=0.01)

    def test_fixed_to_equity(self, analyzer, sample_data):
        """FTE = (2M - 500k) / 1.2M = 1.25."""
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.fixed_to_equity == pytest.approx(1.25, abs=0.01)

    def test_fixed_asset_coverage(self, analyzer, sample_data):
        """FAC = 1.2M / (2M - 500k) = 0.80."""
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.fixed_asset_coverage == pytest.approx(0.80, abs=0.01)

    def test_depreciation_to_fixed(self, analyzer, sample_data):
        """DTF = 50k / 1.5M = 0.0333."""
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.depreciation_to_fixed == pytest.approx(0.0333, abs=0.01)

    def test_capex_to_fixed(self, analyzer, sample_data):
        """CTF = 80k / 1.5M = 0.0533."""
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.capex_to_fixed == pytest.approx(0.0533, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.fae_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert "Fixed Asset Efficiency" in result.summary


# ===== SCORING TESTS =====


class TestFixedAssetEfficiencyScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """FAT=0.667 => base 2.5. CTF=0.053 (neither). FAC=0.80 (neither). Score=2.5."""
        result = analyzer.fixed_asset_efficiency_analysis(sample_data)
        assert result.fae_score == pytest.approx(2.5, abs=0.5)
        assert result.fae_grade == "Weak"

    def test_excellent_efficiency(self, analyzer):
        """FAT >= 5.0 => base 10."""
        data = FinancialData(
            total_assets=1_000_000,
            current_assets=800_000,
            revenue=2_000_000,
            total_equity=500_000,
            depreciation=20_000,
            capex=30_000,
        )
        # FA=200k. FAT=2M/200k=10.0 => 10. CTF=30k/200k=0.15 >=0.10 => +0.5. FAC=500k/200k=2.5 >=1.0 => +0.5. Capped 10.
        result = analyzer.fixed_asset_efficiency_analysis(data)
        assert result.fae_score >= 10.0
        assert result.fae_grade == "Excellent"

    def test_weak_efficiency(self, analyzer):
        """FAT < 0.5 => base 1.0."""
        data = FinancialData(
            total_assets=5_000_000,
            current_assets=500_000,
            revenue=1_000_000,
            total_equity=800_000,
            depreciation=50_000,
            capex=20_000,
        )
        # FA=4.5M. FAT=1M/4.5M=0.222 <0.5 => 1.0. CTF=20k/4.5M=0.004 <0.02 => -0.5. FAC=800k/4.5M=0.178 <0.3 => -0.5. Score=0.0.
        result = analyzer.fixed_asset_efficiency_analysis(data)
        assert result.fae_score <= 1.0
        assert result.fae_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase138EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.fixed_asset_efficiency_analysis(FinancialData())
        assert isinstance(result, FixedAssetEfficiencyResult)
        assert result.fae_score == 0.0

    def test_no_current_assets(self, analyzer):
        """CA=None => FA=None => all ratios None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            revenue=1_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.fixed_asset_efficiency_analysis(data)
        assert result.fixed_asset_turnover is None
        assert result.fae_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => FAT=None, but FAR still computed."""
        data = FinancialData(
            total_assets=2_000_000,
            current_assets=500_000,
            total_equity=1_200_000,
        )
        result = analyzer.fixed_asset_efficiency_analysis(data)
        assert result.fixed_asset_turnover is None
        assert result.fixed_asset_ratio is not None

    def test_no_depreciation(self, analyzer):
        """Depr=None => DTF=None."""
        data = FinancialData(
            total_assets=2_000_000,
            current_assets=500_000,
            revenue=1_000_000,
            total_equity=1_200_000,
            capex=80_000,
        )
        result = analyzer.fixed_asset_efficiency_analysis(data)
        assert result.depreciation_to_fixed is None
        assert result.fixed_asset_turnover is not None
