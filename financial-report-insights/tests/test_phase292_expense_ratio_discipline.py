"""Phase 292 Tests: Expense Ratio Discipline Analysis.

Tests for expense_ratio_discipline_analysis() and ExpenseRatioDisciplineResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    ExpenseRatioDisciplineResult,
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


class TestExpenseRatioDisciplineDataclass:
    def test_defaults(self):
        r = ExpenseRatioDisciplineResult()
        assert r.opex_to_revenue is None
        assert r.cogs_to_revenue is None
        assert r.sga_to_revenue is None
        assert r.total_expense_ratio is None
        assert r.operating_margin is None
        assert r.expense_efficiency is None
        assert r.erd_score == 0.0
        assert r.erd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ExpenseRatioDisciplineResult(opex_to_revenue=0.20, erd_grade="Excellent")
        assert r.opex_to_revenue == 0.20
        assert r.erd_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestExpenseRatioDisciplineAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert isinstance(result, ExpenseRatioDisciplineResult)

    def test_opex_to_revenue(self, analyzer, sample_data):
        """OpEx/Rev = 200k/1M = 0.20."""
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert result.opex_to_revenue == pytest.approx(0.20, abs=0.01)

    def test_cogs_to_revenue(self, analyzer, sample_data):
        """COGS/Rev = 600k/1M = 0.60."""
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert result.cogs_to_revenue == pytest.approx(0.60, abs=0.01)

    def test_total_expense_ratio(self, analyzer, sample_data):
        """(OpEx+COGS)/Rev = (200k+600k)/1M = 0.80."""
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert result.total_expense_ratio == pytest.approx(0.80, abs=0.01)

    def test_operating_margin(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert result.operating_margin == pytest.approx(0.20, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert result.erd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert "Expense Ratio Discipline" in result.summary


# ===== SCORING TESTS =====


class TestExpenseRatioDisciplineScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OpEx/Rev=0.20<=0.30=>base 10. COGS/Rev=0.60<=0.60(+0.5). OI=200k>0(+0.5). Score=10 (capped)."""
        result = analyzer.expense_ratio_discipline_analysis(sample_data)
        assert result.erd_score >= 10.0
        assert result.erd_grade == "Excellent"

    def test_very_disciplined(self, analyzer):
        """Very low expense ratio."""
        data = FinancialData(
            revenue=1_000_000,
            operating_expenses=100_000,
            cogs=300_000,
            operating_income=600_000,
        )
        # OpEx/Rev=0.10<=0.30=>base 10. COGS/Rev=0.30<=0.60(+0.5). OI>0(+0.5). Score=10 (capped).
        result = analyzer.expense_ratio_discipline_analysis(data)
        assert result.erd_score >= 10.0
        assert result.erd_grade == "Excellent"

    def test_weak_discipline(self, analyzer):
        """Very high expense ratio."""
        data = FinancialData(
            revenue=1_000_000,
            operating_expenses=850_000,
            cogs=800_000,
        )
        # OpEx/Rev=0.85>0.80=>base 1.0. COGS/Rev=0.80>0.60(no adj). No OI(no adj). Score=1.0.
        result = analyzer.expense_ratio_discipline_analysis(data)
        assert result.erd_score <= 2.0
        assert result.erd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase292EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.expense_ratio_discipline_analysis(FinancialData())
        assert isinstance(result, ExpenseRatioDisciplineResult)
        assert result.erd_score == 0.0

    def test_no_revenue(self, analyzer):
        data = FinancialData(operating_expenses=200_000)
        result = analyzer.expense_ratio_discipline_analysis(data)
        assert result.erd_score == 0.0

    def test_zero_revenue(self, analyzer):
        data = FinancialData(revenue=0, operating_expenses=200_000)
        result = analyzer.expense_ratio_discipline_analysis(data)
        assert result.erd_score == 0.0

    def test_no_opex(self, analyzer):
        data = FinancialData(revenue=1_000_000)
        result = analyzer.expense_ratio_discipline_analysis(data)
        assert isinstance(result, ExpenseRatioDisciplineResult)
