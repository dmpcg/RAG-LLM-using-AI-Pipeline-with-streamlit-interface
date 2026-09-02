"""Phase 19 Tests: Asset Efficiency & Turnover Analysis.

Tests for asset_efficiency_analysis() and AssetEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    AssetEfficiencyResult,
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


class TestAssetEfficiencyDataclass:
    def test_defaults(self):
        r = AssetEfficiencyResult()
        assert r.total_asset_turnover is None
        assert r.efficiency_score == 0.0
        assert r.efficiency_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = AssetEfficiencyResult(
            total_asset_turnover=1.5,
            efficiency_grade="Good",
        )
        assert r.total_asset_turnover == 1.5
        assert r.efficiency_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestAssetEfficiency:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert isinstance(result, AssetEfficiencyResult)

    def test_total_asset_turnover(self, analyzer, sample_data):
        """Total AT = Revenue / Total Assets = 1M / 2M = 0.50."""
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.total_asset_turnover == pytest.approx(0.50, rel=0.01)

    def test_fixed_asset_turnover(self, analyzer, sample_data):
        """Fixed AT = Revenue / (TA - CA) = 1M / 1.5M = 0.667."""
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.fixed_asset_turnover == pytest.approx(1_000_000 / 1_500_000, rel=0.01)

    def test_inventory_turnover(self, analyzer, sample_data):
        """Inv Turnover = COGS / Inventory = 600k / 100k = 6.0."""
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.inventory_turnover == pytest.approx(6.0, rel=0.01)

    def test_receivables_turnover(self, analyzer, sample_data):
        """AR Turnover = Revenue / AR = 1M / 150k = 6.67."""
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.receivables_turnover == pytest.approx(1_000_000 / 150_000, rel=0.01)

    def test_payables_turnover(self, analyzer, sample_data):
        """AP Turnover = COGS / AP = 600k / 80k = 7.5."""
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.payables_turnover == pytest.approx(7.5, rel=0.01)

    def test_working_capital_turnover(self, analyzer, sample_data):
        """WC Turnover = Revenue / (CA - CL) = 1M / 300k = 3.33."""
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.working_capital_turnover == pytest.approx(1_000_000 / 300_000, rel=0.01)

    def test_equity_turnover(self, analyzer, sample_data):
        """Equity Turnover = Revenue / Equity = 1M / 1.2M = 0.833."""
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.equity_turnover == pytest.approx(1_000_000 / 1_200_000, rel=0.01)

    def test_efficiency_grade_assigned(self, analyzer, sample_data):
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert result.efficiency_grade in ["Excellent", "Good", "Average", "Below Average"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.asset_efficiency_analysis(sample_data)
        assert "Asset Efficiency" in result.summary

    def test_uses_avg_values(self, analyzer):
        """Should prefer avg values over point-in-time."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=500_000,
            total_assets=2_000_000,
            inventory=100_000,
            accounts_receivable=150_000,
            avg_total_assets=1_800_000,  # Average
            avg_inventory=90_000,  # Average
            avg_receivables=140_000,  # Average
        )
        result = analyzer.asset_efficiency_analysis(data)
        # Should use avg_total_assets
        assert result.total_asset_turnover == pytest.approx(1_000_000 / 1_800_000, rel=0.01)
        # Should use avg_inventory
        assert result.inventory_turnover == pytest.approx(500_000 / 90_000, rel=0.01)
        # Should use avg_receivables
        assert result.receivables_turnover == pytest.approx(1_000_000 / 140_000, rel=0.01)


# ===== SCORING TESTS =====


class TestEfficiencyScoring:
    def test_high_efficiency_company(self, analyzer):
        """Company with excellent turnover should score high."""
        data = FinancialData(
            revenue=5_000_000,
            cogs=3_000_000,
            total_assets=2_000_000,
            current_assets=800_000,
            current_liabilities=300_000,
            inventory=200_000,
            accounts_receivable=300_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        # AT=2.5 (high), inv=15 (high), AR=16.7 (high), fixed AT=4.17 (high)
        assert result.efficiency_grade == "Excellent"

    def test_low_efficiency_company(self, analyzer):
        """Company with poor turnover should score low."""
        data = FinancialData(
            revenue=100_000,
            cogs=60_000,
            total_assets=5_000_000,
            current_assets=2_000_000,
            current_liabilities=500_000,
            inventory=500_000,
            accounts_receivable=800_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        # AT=0.02 (very low), inv=0.12 (very low), AR=0.125 (very low)
        assert result.efficiency_grade in ["Below Average", "Average"]

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            revenue=10_000_000,
            cogs=5_000_000,
            total_assets=1_000_000,
            current_assets=400_000,
            inventory=50_000,
            accounts_receivable=80_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        assert result.efficiency_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            revenue=10_000,
            cogs=5_000,
            total_assets=50_000_000,
            current_assets=20_000_000,
            current_liabilities=5_000_000,
            inventory=10_000_000,
            accounts_receivable=15_000_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        assert result.efficiency_score >= 0.0


# ===== EDGE CASES =====


class TestPhase19EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.asset_efficiency_analysis(FinancialData())
        assert isinstance(result, AssetEfficiencyResult)
        assert result.total_asset_turnover is None

    def test_zero_revenue(self, analyzer):
        data = FinancialData(
            revenue=0,
            total_assets=1_000_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        # safe_divide(0, 1M) = 0.0 which is correct (zero revenue = zero turnover)
        assert result.total_asset_turnover == pytest.approx(0.0, abs=0.01)

    def test_zero_assets(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            total_assets=0,
        )
        result = analyzer.asset_efficiency_analysis(data)
        assert result.total_asset_turnover is None

    def test_no_inventory(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            cogs=500_000,
            total_assets=2_000_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        assert result.inventory_turnover is None
        assert result.total_asset_turnover is not None

    def test_negative_working_capital(self, analyzer):
        """Negative WC (CL > CA) means WC turnover is None."""
        data = FinancialData(
            revenue=1_000_000,
            current_assets=200_000,
            current_liabilities=500_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        assert result.working_capital_turnover is None

    def test_zero_equity(self, analyzer):
        data = FinancialData(
            revenue=500_000,
            total_equity=0,
        )
        result = analyzer.asset_efficiency_analysis(data)
        assert result.equity_turnover is None

    def test_no_fixed_assets(self, analyzer):
        """When CA = TA, fixed assets = 0, so fixed AT is None."""
        data = FinancialData(
            revenue=500_000,
            total_assets=500_000,
            current_assets=500_000,
        )
        result = analyzer.asset_efficiency_analysis(data)
        assert result.fixed_asset_turnover is None
