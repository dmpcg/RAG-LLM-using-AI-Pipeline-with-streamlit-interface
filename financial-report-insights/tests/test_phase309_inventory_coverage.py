"""Phase 309 Tests: Inventory Coverage Analysis.

Tests for inventory_coverage_analysis() and InventoryCoverageResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    InventoryCoverageResult,
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


class TestInventoryCoverageDataclass:
    def test_defaults(self):
        r = InventoryCoverageResult()
        assert r.inventory_to_revenue is None
        assert r.inventory_to_cogs is None
        assert r.inventory_to_assets is None
        assert r.inventory_to_current_assets is None
        assert r.inventory_days is None
        assert r.inventory_buffer is None
        assert r.icv_score == 0.0
        assert r.icv_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = InventoryCoverageResult(inventory_to_cogs=0.167, icv_grade="Excellent")
        assert r.inventory_to_cogs == 0.167
        assert r.icv_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestInventoryCoverageAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.inventory_coverage_analysis(sample_data)
        assert isinstance(result, InventoryCoverageResult)

    def test_inventory_to_cogs(self, analyzer, sample_data):
        """Inv/COGS = 100k/600k = 0.167."""
        result = analyzer.inventory_coverage_analysis(sample_data)
        assert result.inventory_to_cogs == pytest.approx(0.167, abs=0.01)

    def test_inventory_days(self, analyzer, sample_data):
        """Days = 100k / (600k/365) = 60.83."""
        result = analyzer.inventory_coverage_analysis(sample_data)
        assert result.inventory_days == pytest.approx(60.83, abs=1.0)

    def test_inventory_to_revenue(self, analyzer, sample_data):
        """Inv/Rev = 100k/1M = 0.10."""
        result = analyzer.inventory_coverage_analysis(sample_data)
        assert result.inventory_to_revenue == pytest.approx(0.10, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.inventory_coverage_analysis(sample_data)
        assert result.icv_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.inventory_coverage_analysis(sample_data)
        assert "Inventory Coverage" in result.summary


# ===== SCORING TESTS =====


class TestInventoryCoverageScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """Inv/COGS=0.167 in [0.08,0.20]=>base 10. Days=60.83 in [30,90](+0.5). Both>0(+0.5). Score=10 (capped)."""
        result = analyzer.inventory_coverage_analysis(sample_data)
        assert result.icv_score >= 10.0
        assert result.icv_grade == "Excellent"

    def test_high_inventory(self, analyzer):
        """High inventory relative to COGS."""
        data = FinancialData(
            inventory=350_000,
            cogs=600_000,
            revenue=1_000_000,
        )
        # Inv/COGS=0.583>0.50=>base 2.5. Days=213 not in [30,90](no adj). Both>0(+0.5). Score=3.0.
        result = analyzer.inventory_coverage_analysis(data)
        assert result.icv_score <= 3.5
        assert result.icv_grade == "Weak"

    def test_low_inventory(self, analyzer):
        """Very low inventory — possible stockout risk."""
        data = FinancialData(
            inventory=5_000,
            cogs=600_000,
            revenue=1_000_000,
        )
        # Inv/COGS=0.0083<0.02=>base 4.0. Days=3 not in [30,90](no adj). Both>0(+0.5). Score=4.5.
        result = analyzer.inventory_coverage_analysis(data)
        assert result.icv_score == pytest.approx(4.5, abs=0.5)
        assert result.icv_grade == "Adequate"


# ===== EDGE CASES =====


class TestPhase309EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.inventory_coverage_analysis(FinancialData())
        assert isinstance(result, InventoryCoverageResult)
        assert result.icv_score == 0.0

    def test_no_inventory(self, analyzer):
        data = FinancialData(cogs=600_000, revenue=1_000_000)
        result = analyzer.inventory_coverage_analysis(data)
        assert result.icv_score == 0.0

    def test_no_cogs_no_revenue(self, analyzer):
        data = FinancialData(inventory=100_000)
        result = analyzer.inventory_coverage_analysis(data)
        assert result.icv_score == 0.0
