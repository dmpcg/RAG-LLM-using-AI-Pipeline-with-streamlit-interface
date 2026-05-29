"""Phase 247 Tests: Inventory Turnover Analysis.

Tests for inventory_turnover_analysis() and InventoryTurnoverResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    InventoryTurnoverResult,
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


class TestInventoryTurnoverDataclass:
    def test_defaults(self):
        r = InventoryTurnoverResult()
        assert r.cogs_to_inv is None
        assert r.dio is None
        assert r.inv_to_ca is None
        assert r.inv_to_ta is None
        assert r.inv_to_revenue is None
        assert r.inv_velocity is None
        assert r.ito_score == 0.0
        assert r.ito_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = InventoryTurnoverResult(cogs_to_inv=6.0, ito_grade="Good")
        assert r.cogs_to_inv == 6.0
        assert r.ito_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestInventoryTurnoverAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.inventory_turnover_analysis(sample_data)
        assert isinstance(result, InventoryTurnoverResult)

    def test_cogs_to_inv(self, analyzer, sample_data):
        """COGS/Inv = 600k/100k = 6.0."""
        result = analyzer.inventory_turnover_analysis(sample_data)
        assert result.cogs_to_inv == pytest.approx(6.0, abs=0.01)

    def test_dio(self, analyzer, sample_data):
        """DIO = 100k*365/600k = 60.83 days."""
        result = analyzer.inventory_turnover_analysis(sample_data)
        assert result.dio == pytest.approx(60.83, abs=0.5)

    def test_inv_to_ca(self, analyzer, sample_data):
        """Inv/CA = 100k/500k = 0.20."""
        result = analyzer.inventory_turnover_analysis(sample_data)
        assert result.inv_to_ca == pytest.approx(0.20, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.inventory_turnover_analysis(sample_data)
        assert result.ito_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.inventory_turnover_analysis(sample_data)
        assert "Inventory Turnover" in result.summary


# ===== SCORING TESTS =====


class TestInventoryTurnoverScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """COGS/Inv=6.0 >=6.0 => base 7.0. DIO=60.83 (no adj). Inv/CA=0.20 <=0.15? No. Score=7.0."""
        result = analyzer.inventory_turnover_analysis(sample_data)
        assert result.ito_score == pytest.approx(7.0, abs=0.5)
        assert result.ito_grade == "Good"

    def test_excellent_fast_turns(self, analyzer):
        """Very fast inventory turns."""
        data = FinancialData(
            cogs=1_200_000,
            inventory=80_000,
            current_assets=1_000_000,
            total_assets=3_000_000,
        )
        # COGS/Inv=15.0 >=12.0 => base 10. DIO=24.3 <=30 => +0.5. Inv/CA=0.08 <=0.15 => +0.5. Score=10.
        result = analyzer.inventory_turnover_analysis(data)
        assert result.ito_score >= 10.0
        assert result.ito_grade == "Excellent"

    def test_weak_slow_turns(self, analyzer):
        """Very slow inventory turns."""
        data = FinancialData(
            cogs=200_000,
            inventory=400_000,
            current_assets=600_000,
            total_assets=2_000_000,
        )
        # COGS/Inv=0.5 <1.0 => base 1.0. DIO=730 >120 => -0.5. Inv/CA=0.667 >0.50 => -0.5. Score=0.
        result = analyzer.inventory_turnover_analysis(data)
        assert result.ito_score <= 0.5
        assert result.ito_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase247EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.inventory_turnover_analysis(FinancialData())
        assert isinstance(result, InventoryTurnoverResult)
        assert result.ito_score == 0.0

    def test_no_inventory(self, analyzer):
        """Inv=None => insufficient data => score 0."""
        data = FinancialData(cogs=600_000)
        result = analyzer.inventory_turnover_analysis(data)
        assert result.ito_score == 0.0

    def test_no_cogs(self, analyzer):
        """COGS=None => insufficient data => score 0."""
        data = FinancialData(inventory=100_000)
        result = analyzer.inventory_turnover_analysis(data)
        assert result.ito_score == 0.0

    def test_zero_inventory(self, analyzer):
        """Inv=0 => insufficient data => score 0."""
        data = FinancialData(
            cogs=600_000,
            inventory=0,
        )
        result = analyzer.inventory_turnover_analysis(data)
        assert result.ito_score == 0.0
