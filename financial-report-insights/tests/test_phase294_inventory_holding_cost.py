"""Phase 294 Tests: Inventory Holding Cost Analysis.

Tests for inventory_holding_cost_analysis() and InventoryHoldingCostResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    InventoryHoldingCostResult,
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


class TestInventoryHoldingCostDataclass:
    def test_defaults(self):
        r = InventoryHoldingCostResult()
        assert r.inventory_to_revenue is None
        assert r.inventory_to_current_assets is None
        assert r.inventory_to_total_assets is None
        assert r.inventory_days is None
        assert r.inventory_carrying_cost is None
        assert r.inventory_intensity is None
        assert r.ihc_score == 0.0
        assert r.ihc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = InventoryHoldingCostResult(inventory_to_revenue=0.10, ihc_grade="Excellent")
        assert r.inventory_to_revenue == 0.10
        assert r.ihc_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestInventoryHoldingCostAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.inventory_holding_cost_analysis(sample_data)
        assert isinstance(result, InventoryHoldingCostResult)

    def test_inventory_to_revenue(self, analyzer, sample_data):
        """Inv/Rev = 100k/1M = 0.10."""
        result = analyzer.inventory_holding_cost_analysis(sample_data)
        assert result.inventory_to_revenue == pytest.approx(0.10, abs=0.01)

    def test_inventory_to_current_assets(self, analyzer, sample_data):
        """Inv/CA = 100k/500k = 0.20."""
        result = analyzer.inventory_holding_cost_analysis(sample_data)
        assert result.inventory_to_current_assets == pytest.approx(0.20, abs=0.01)

    def test_inventory_days(self, analyzer, sample_data):
        """Inv/COGS*365 = 100k/600k*365 = 60.83 days."""
        result = analyzer.inventory_holding_cost_analysis(sample_data)
        assert result.inventory_days == pytest.approx(60.83, abs=1.0)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.inventory_holding_cost_analysis(sample_data)
        assert result.ihc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.inventory_holding_cost_analysis(sample_data)
        assert "Inventory Holding Cost" in result.summary


# ===== SCORING TESTS =====


class TestInventoryHoldingCostScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """Inv/Rev=0.10 in (0.08,0.12]=>base 7.0. Inv/CA=0.20<=0.30(+0.5). Inv>0&Rev>0(+0.5). Score=8.0."""
        result = analyzer.inventory_holding_cost_analysis(sample_data)
        assert result.ihc_score == pytest.approx(8.0, abs=0.5)
        assert result.ihc_grade == "Excellent"

    def test_very_lean_inventory(self, analyzer):
        """Very low inventory relative to revenue."""
        data = FinancialData(
            revenue=1_000_000,
            inventory=20_000,
            current_assets=500_000,
        )
        # Inv/Rev=0.02<=0.05=>base 10. Inv/CA=0.04<=0.30(+0.5). Inv>0&Rev>0(+0.5). Score=10 (capped).
        result = analyzer.inventory_holding_cost_analysis(data)
        assert result.ihc_score >= 10.0
        assert result.ihc_grade == "Excellent"

    def test_heavy_inventory(self, analyzer):
        """Very high inventory relative to revenue."""
        data = FinancialData(
            revenue=1_000_000,
            inventory=400_000,
            current_assets=500_000,
        )
        # Inv/Rev=0.40>0.35=>base 1.0. Inv/CA=0.80>0.30(no adj). Inv>0&Rev>0(+0.5). Score=1.5.
        result = analyzer.inventory_holding_cost_analysis(data)
        assert result.ihc_score <= 2.0
        assert result.ihc_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase294EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.inventory_holding_cost_analysis(FinancialData())
        assert isinstance(result, InventoryHoldingCostResult)
        assert result.ihc_score == 0.0

    def test_no_revenue(self, analyzer):
        data = FinancialData(inventory=100_000)
        result = analyzer.inventory_holding_cost_analysis(data)
        assert result.ihc_score == 0.0

    def test_zero_revenue(self, analyzer):
        data = FinancialData(revenue=0, inventory=100_000)
        result = analyzer.inventory_holding_cost_analysis(data)
        assert result.ihc_score == 0.0

    def test_no_inventory(self, analyzer):
        data = FinancialData(revenue=1_000_000)
        result = analyzer.inventory_holding_cost_analysis(data)
        assert isinstance(result, InventoryHoldingCostResult)
        assert result.ihc_score == 0.0
