"""Phase 330 Tests: Operating Expense Ratio Analysis.

Tests for operating_expense_ratio_analysis() and OperatingExpenseRatioResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperatingExpenseRatioResult,
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


class TestOperatingExpenseRatioDataclass:
    def test_defaults(self):
        r = OperatingExpenseRatioResult()
        assert r.opex_ratio is None
        assert r.opex_per_revenue is None
        assert r.opex_to_gross_profit is None
        assert r.opex_to_ebitda is None
        assert r.opex_coverage is None
        assert r.efficiency_gap is None
        assert r.oer_score == 0.0
        assert r.oer_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperatingExpenseRatioResult(opex_ratio=0.20, oer_grade="Good")
        assert r.opex_ratio == 0.20
        assert r.oer_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestOperatingExpenseRatioAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operating_expense_ratio_analysis(sample_data)
        assert isinstance(result, OperatingExpenseRatioResult)

    def test_opex_ratio(self, analyzer, sample_data):
        """OpEx/Revenue = 200k/1M = 0.20."""
        result = analyzer.operating_expense_ratio_analysis(sample_data)
        assert result.opex_ratio == pytest.approx(0.20, abs=0.01)

    def test_opex_to_gross_profit(self, analyzer, sample_data):
        """OpEx/GP = 200k/400k = 0.50."""
        result = analyzer.operating_expense_ratio_analysis(sample_data)
        assert result.opex_to_gross_profit == pytest.approx(0.50, abs=0.01)

    def test_opex_to_ebitda(self, analyzer, sample_data):
        """OpEx/EBITDA = 200k/250k = 0.80."""
        result = analyzer.operating_expense_ratio_analysis(sample_data)
        assert result.opex_to_ebitda == pytest.approx(0.80, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operating_expense_ratio_analysis(sample_data)
        assert result.oer_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operating_expense_ratio_analysis(sample_data)
        assert "Operating Expense Ratio" in result.summary


# ===== SCORING TESTS =====


class TestOperatingExpenseRatioScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OER=0.20 in (0.15,0.20]=>base 7.0. OpEx/GP=0.50<1.0(+0.5). Both>0(+0.5). Score=8.0."""
        result = analyzer.operating_expense_ratio_analysis(sample_data)
        assert result.oer_score == pytest.approx(8.0, abs=0.5)
        assert result.oer_grade in ["Excellent", "Good"]

    def test_low_opex(self, analyzer):
        """Very low OpEx — highly efficient."""
        data = FinancialData(
            revenue=1_000_000,
            operating_expenses=50_000,
            gross_profit=800_000,
        )
        # OER=0.05<=0.10=>base 10. OpEx/GP=0.0625<1.0(+0.5). Both>0(+0.5). Score=10 (capped).
        result = analyzer.operating_expense_ratio_analysis(data)
        assert result.oer_score >= 10.0
        assert result.oer_grade == "Excellent"

    def test_high_opex(self, analyzer):
        """Very high OpEx — bloated cost structure."""
        data = FinancialData(
            revenue=1_000_000,
            operating_expenses=500_000,
            gross_profit=400_000,
        )
        # OER=0.50>0.40=>base 1.0. OpEx/GP=1.25>=1.0(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.operating_expense_ratio_analysis(data)
        assert result.oer_score <= 2.0
        assert result.oer_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase330EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operating_expense_ratio_analysis(FinancialData())
        assert isinstance(result, OperatingExpenseRatioResult)
        assert result.oer_score == 0.0

    def test_no_opex(self, analyzer):
        """OpEx=None, Revenue=1M => safe_divide returns None (no OpEx data)."""
        data = FinancialData(revenue=1_000_000)
        result = analyzer.operating_expense_ratio_analysis(data)
        # With no operating_expenses attribute, opex_ratio is None
        assert result.opex_ratio is None

    def test_no_revenue(self, analyzer):
        """Revenue=0 with OpEx => safe_divide returns None => score 0.0."""
        data = FinancialData(operating_expenses=500_000)
        result = analyzer.operating_expense_ratio_analysis(data)
        assert result.oer_score == 0.0
