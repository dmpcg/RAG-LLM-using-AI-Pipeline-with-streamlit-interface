"""Phase 101 Tests: Debt Composition Analysis.

Tests for debt_composition_analysis() and DebtCompositionResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtCompositionResult,
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


class TestDebtCompositionDataclass:
    def test_defaults(self):
        r = DebtCompositionResult()
        assert r.debt_to_equity is None
        assert r.debt_to_assets is None
        assert r.long_term_debt_ratio is None
        assert r.interest_burden is None
        assert r.debt_cost_ratio is None
        assert r.debt_coverage_margin is None
        assert r.dco_score == 0.0
        assert r.dco_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtCompositionResult(debt_to_equity=0.33, dco_grade="Excellent")
        assert r.debt_to_equity == 0.33
        assert r.dco_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestDebtCompositionAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_composition_analysis(sample_data)
        assert isinstance(result, DebtCompositionResult)

    def test_debt_to_equity(self, analyzer, sample_data):
        """D/E = 400k/1.2M = 0.333."""
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.01)

    def test_debt_to_assets(self, analyzer, sample_data):
        """D/A = 400k/2M = 0.20."""
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.debt_to_assets == pytest.approx(0.20, abs=0.01)

    def test_long_term_debt_ratio(self, analyzer, sample_data):
        """LTR = (800k-200k)/800k = 0.75."""
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.long_term_debt_ratio == pytest.approx(0.75, abs=0.01)

    def test_interest_burden(self, analyzer, sample_data):
        """IB = 30k/200k = 0.15."""
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.interest_burden == pytest.approx(0.15, abs=0.01)

    def test_debt_cost_ratio(self, analyzer, sample_data):
        """DCR = 30k/400k = 0.075."""
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.debt_cost_ratio == pytest.approx(0.075, abs=0.005)

    def test_debt_coverage_margin(self, analyzer, sample_data):
        """DCM = (220k-30k)/400k = 0.475."""
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.debt_coverage_margin == pytest.approx(0.475, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.dco_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_composition_analysis(sample_data)
        assert "Debt Composition" in result.summary


# ===== SCORING TESTS =====


class TestDebtCompositionScoring:
    def test_low_debt(self, analyzer, sample_data):
        """D/E=0.333 <= 0.50 => base 8.5. DCM=0.475 >= 0.30 => +0.5. Score=9.0."""
        result = analyzer.debt_composition_analysis(sample_data)
        assert result.dco_score >= 8.0
        assert result.dco_grade == "Excellent"

    def test_no_debt(self, analyzer):
        """TD=0, TL=0 => perfect score."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=150_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.debt_composition_analysis(data)
        assert result.dco_score == 10.0
        assert result.dco_grade == "Excellent"

    def test_high_debt(self, analyzer):
        """D/E > 3.0 => base 1.0."""
        data = FinancialData(
            revenue=500_000,
            net_income=20_000,
            ebit=50_000,
            total_assets=1_000_000,
            total_liabilities=900_000,
            total_equity=100_000,
            total_debt=800_000,
            current_liabilities=100_000,
            interest_expense=80_000,
            operating_cash_flow=30_000,
        )
        result = analyzer.debt_composition_analysis(data)
        # D/E = 800k/100k = 8.0 > 3.0 => base 1.0
        assert result.dco_score <= 2.0
        assert result.dco_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase101EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_composition_analysis(FinancialData())
        assert isinstance(result, DebtCompositionResult)
        # No debt => perfect
        assert result.dco_score == 10.0

    def test_no_equity(self, analyzer):
        """TE=0 => D/E=None, falls to D/A scoring."""
        data = FinancialData(
            revenue=500_000,
            ebit=80_000,
            total_assets=1_000_000,
            total_liabilities=500_000,
            total_debt=500_000,
            current_liabilities=100_000,
            interest_expense=40_000,
            operating_cash_flow=100_000,
        )
        result = analyzer.debt_composition_analysis(data)
        assert result.debt_to_equity is None
        # D/A = 500k/1M = 0.50 => base 4.0 (D/A scoring)
        assert result.dco_score >= 3.0

    def test_zero_interest(self, analyzer):
        """IE=0 => interest_burden=None, debt_cost_ratio=None."""
        data = FinancialData(
            revenue=1_000_000,
            ebit=200_000,
            total_assets=2_000_000,
            total_liabilities=400_000,
            total_equity=1_200_000,
            total_debt=200_000,
            current_liabilities=100_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.debt_composition_analysis(data)
        assert result.interest_burden is None
        assert result.debt_cost_ratio is None

    def test_negative_coverage_margin(self, analyzer):
        """OCF < IE => negative DCM => -0.5 penalty."""
        data = FinancialData(
            revenue=500_000,
            ebit=50_000,
            total_assets=1_000_000,
            total_liabilities=500_000,
            total_equity=500_000,
            total_debt=400_000,
            current_liabilities=100_000,
            interest_expense=80_000,
            operating_cash_flow=50_000,
        )
        result = analyzer.debt_composition_analysis(data)
        # DCM = (50k-80k)/400k = -0.075 < 0 => -0.5
        assert result.debt_coverage_margin < 0
