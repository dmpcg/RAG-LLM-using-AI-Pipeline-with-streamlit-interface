"""Phase 314 Tests: Debt Burden Index Analysis.

Tests for debt_burden_index_analysis() and DebtBurdenIndexResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtBurdenIndexResult,
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


class TestDebtBurdenIndexDataclass:
    def test_defaults(self):
        r = DebtBurdenIndexResult()
        assert r.debt_to_ebitda is None
        assert r.debt_to_assets is None
        assert r.debt_to_equity is None
        assert r.debt_to_revenue is None
        assert r.debt_ratio is None
        assert r.burden_intensity is None
        assert r.dbi_score == 0.0
        assert r.dbi_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtBurdenIndexResult(debt_to_ebitda=2.0, dbi_grade="Good")
        assert r.debt_to_ebitda == 2.0
        assert r.dbi_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestDebtBurdenIndexAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_burden_index_analysis(sample_data)
        assert isinstance(result, DebtBurdenIndexResult)

    def test_debt_to_ebitda(self, analyzer, sample_data):
        """Debt/EBITDA = 400k/250k = 1.6."""
        result = analyzer.debt_burden_index_analysis(sample_data)
        assert result.debt_to_ebitda == pytest.approx(1.6, abs=0.01)

    def test_debt_to_assets(self, analyzer, sample_data):
        """Debt/Assets = 400k/2M = 0.20."""
        result = analyzer.debt_burden_index_analysis(sample_data)
        assert result.debt_to_assets == pytest.approx(0.20, abs=0.01)

    def test_debt_to_equity(self, analyzer, sample_data):
        """Debt/Equity = 400k/1.2M = 0.333."""
        result = analyzer.debt_burden_index_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_burden_index_analysis(sample_data)
        assert result.dbi_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_burden_index_analysis(sample_data)
        assert "Debt Burden Index" in result.summary


# ===== SCORING TESTS =====


class TestDebtBurdenIndexScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """Debt/EBITDA=1.6 in (1.0,2.0]=>base 8.5. Debt/Assets=0.20<=0.40(+0.5). Both>0(+0.5). Score=9.5."""
        result = analyzer.debt_burden_index_analysis(sample_data)
        assert result.dbi_score == pytest.approx(9.5, abs=0.5)
        assert result.dbi_grade == "Excellent"

    def test_low_debt(self, analyzer):
        """Very low debt burden."""
        data = FinancialData(
            total_debt=50_000,
            ebitda=250_000,
            total_assets=2_000_000,
        )
        # Debt/EBITDA=0.20<=1.0=>base 10. Debt/Assets=0.025<=0.40(+0.5). Both>0(+0.5). Score=10 (capped).
        result = analyzer.debt_burden_index_analysis(data)
        assert result.dbi_score >= 10.0
        assert result.dbi_grade == "Excellent"

    def test_high_debt(self, analyzer):
        """Very high debt burden."""
        data = FinancialData(
            total_debt=2_000_000,
            ebitda=250_000,
            total_assets=2_000_000,
        )
        # Debt/EBITDA=8.0>6.0=>base 1.0. Debt/Assets=1.0>0.40(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.debt_burden_index_analysis(data)
        assert result.dbi_score <= 2.0
        assert result.dbi_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase314EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_burden_index_analysis(FinancialData())
        assert isinstance(result, DebtBurdenIndexResult)
        assert result.dbi_score == 0.0

    def test_no_debt(self, analyzer):
        data = FinancialData(ebitda=250_000, total_assets=2_000_000)
        result = analyzer.debt_burden_index_analysis(data)
        assert result.dbi_score == 0.0

    def test_no_ebitda_uses_revenue(self, analyzer):
        """Falls back to Debt/Revenue when EBITDA is missing."""
        data = FinancialData(total_debt=400_000, revenue=1_000_000, total_assets=2_000_000)
        result = analyzer.debt_burden_index_analysis(data)
        # Debt/Revenue=0.40<=1.0=>base 10. Debt/Assets=0.20<=0.40(+0.5). No EBITDA(no adj). Score=10 (capped).
        assert result.dbi_score >= 9.0
