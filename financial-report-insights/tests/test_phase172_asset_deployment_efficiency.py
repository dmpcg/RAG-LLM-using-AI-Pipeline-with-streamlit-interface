"""Phase 172 Tests: Asset Deployment Efficiency Analysis.

Tests for asset_deployment_efficiency_analysis() and AssetDeploymentEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    AssetDeploymentEfficiencyResult,
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


class TestAssetDeploymentEfficiencyDataclass:
    def test_defaults(self):
        r = AssetDeploymentEfficiencyResult()
        assert r.asset_turnover is None
        assert r.fixed_asset_leverage is None
        assert r.asset_income_yield is None
        assert r.asset_cash_yield is None
        assert r.inventory_velocity is None
        assert r.receivables_velocity is None
        assert r.ade_score == 0.0
        assert r.ade_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = AssetDeploymentEfficiencyResult(asset_turnover=0.50, ade_grade="Good")
        assert r.asset_turnover == 0.50
        assert r.ade_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestAssetDeploymentEfficiencyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert isinstance(result, AssetDeploymentEfficiencyResult)

    def test_asset_turnover(self, analyzer, sample_data):
        """Rev/TA = 1M/2M = 0.50."""
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.asset_turnover == pytest.approx(0.50, abs=0.01)

    def test_fixed_asset_leverage(self, analyzer, sample_data):
        """Rev/(TA-CA) = 1M/(2M-500k) = 1M/1.5M = 0.667."""
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.fixed_asset_leverage == pytest.approx(0.667, abs=0.01)

    def test_asset_income_yield(self, analyzer, sample_data):
        """OI/TA = 200k/2M = 0.10."""
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.asset_income_yield == pytest.approx(0.10, abs=0.01)

    def test_asset_cash_yield(self, analyzer, sample_data):
        """OCF/TA = 220k/2M = 0.11."""
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.asset_cash_yield == pytest.approx(0.11, abs=0.01)

    def test_inventory_velocity(self, analyzer, sample_data):
        """Rev/Inv = 1M/100k = 10.0."""
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.inventory_velocity == pytest.approx(10.0, abs=0.1)

    def test_receivables_velocity(self, analyzer, sample_data):
        """Rev/AR = 1M/150k = 6.667."""
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.receivables_velocity == pytest.approx(6.667, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.ade_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert "Asset Deployment Efficiency" in result.summary


# ===== SCORING TESTS =====


class TestAssetDeploymentEfficiencyScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """AT=0.50 => base 4.0. AIY=0.10 >=0.10 => +0.5. ACY=0.11 >=0.10 => +0.5. Score=5.0."""
        result = analyzer.asset_deployment_efficiency_analysis(sample_data)
        assert result.ade_score == pytest.approx(5.0, abs=0.5)
        assert result.ade_grade == "Adequate"

    def test_excellent_deployment(self, analyzer):
        """Very high asset turnover."""
        data = FinancialData(
            revenue=5_000_000,
            total_assets=2_000_000,
            current_assets=500_000,
            operating_income=500_000,
            operating_cash_flow=600_000,
            inventory=200_000,
            accounts_receivable=300_000,
        )
        result = analyzer.asset_deployment_efficiency_analysis(data)
        assert result.ade_score >= 10.0
        assert result.ade_grade == "Excellent"

    def test_weak_deployment(self, analyzer):
        """Very low asset turnover."""
        data = FinancialData(
            revenue=100_000,
            total_assets=5_000_000,
            current_assets=200_000,
            operating_income=5_000,
            operating_cash_flow=3_000,
            inventory=50_000,
            accounts_receivable=80_000,
        )
        # AT=100k/5M=0.02 => base 1.0. AIY=5k/5M=0.001 <0.02 => -0.5. ACY=3k/5M=0.0006 <0.02 => -0.5. Score=0.0.
        result = analyzer.asset_deployment_efficiency_analysis(data)
        assert result.ade_score <= 1.0
        assert result.ade_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase172EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.asset_deployment_efficiency_analysis(FinancialData())
        assert isinstance(result, AssetDeploymentEfficiencyResult)
        assert result.ade_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => AT=None, most ratios None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
        )
        result = analyzer.asset_deployment_efficiency_analysis(data)
        assert result.asset_turnover is None
        assert result.ade_score == 0.0

    def test_no_total_assets(self, analyzer):
        """TA=None => AT=None, AIY=None, ACY=None."""
        data = FinancialData(
            revenue=1_000_000,
            inventory=100_000,
            accounts_receivable=150_000,
        )
        result = analyzer.asset_deployment_efficiency_analysis(data)
        assert result.asset_turnover is None
        assert result.inventory_velocity is not None

    def test_no_inventory(self, analyzer):
        """Inv=None => IV=None."""
        data = FinancialData(
            revenue=1_000_000,
            total_assets=2_000_000,
            accounts_receivable=150_000,
        )
        result = analyzer.asset_deployment_efficiency_analysis(data)
        assert result.inventory_velocity is None
        assert result.receivables_velocity is not None
