"""Phase 267 Tests: Debt Quality Assessment.

Tests for debt_quality_analysis() and DebtQualityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtQualityResult,
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


class TestDebtQualityDataclass:
    def test_defaults(self):
        r = DebtQualityResult()
        assert r.debt_to_equity is None
        assert r.debt_to_assets is None
        assert r.long_term_debt_ratio is None
        assert r.debt_to_ebitda is None
        assert r.interest_coverage is None
        assert r.debt_cost is None
        assert r.dq_score == 0.0
        assert r.dq_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtQualityResult(debt_to_equity=0.33, dq_grade="Excellent")
        assert r.debt_to_equity == 0.33
        assert r.dq_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestDebtQualityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_quality_analysis(sample_data)
        assert isinstance(result, DebtQualityResult)

    def test_debt_to_equity(self, analyzer, sample_data):
        """D/E = 400k/1.2M = 0.333."""
        result = analyzer.debt_quality_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.01)

    def test_debt_to_assets(self, analyzer, sample_data):
        """D/A = 400k/2M = 0.20."""
        result = analyzer.debt_quality_analysis(sample_data)
        assert result.debt_to_assets == pytest.approx(0.20, abs=0.01)

    def test_debt_to_ebitda(self, analyzer, sample_data):
        """D/EBITDA = 400k/250k = 1.60."""
        result = analyzer.debt_quality_analysis(sample_data)
        assert result.debt_to_ebitda == pytest.approx(1.60, abs=0.01)

    def test_interest_coverage(self, analyzer, sample_data):
        """IC = EBIT/IE = 200k/30k = 6.667."""
        result = analyzer.debt_quality_analysis(sample_data)
        assert result.interest_coverage == pytest.approx(6.667, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_quality_analysis(sample_data)
        assert result.dq_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_quality_analysis(sample_data)
        assert "Debt Quality" in result.summary


# ===== SCORING TESTS =====


class TestDebtQualityScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """D/E=0.333 in (0.20,0.50]=>base 8.5. IC=6.667>=5.0(+0.5). D/EBITDA=1.60<=3.0(+0.5). Score=9.5."""
        result = analyzer.debt_quality_analysis(sample_data)
        assert result.dq_score == pytest.approx(9.5, abs=0.5)
        assert result.dq_grade == "Excellent"

    def test_minimal_debt(self, analyzer):
        """Very low debt-to-equity."""
        data = FinancialData(
            total_debt=50_000,
            total_equity=1_000_000,
            total_assets=1_200_000,
            ebitda=300_000,
            ebit=250_000,
            interest_expense=5_000,
        )
        # D/E=0.05<=0.20=>base 10. IC=50.0>=5.0(+0.5). D/EBITDA=0.167<=3.0(+0.5). Score=10.
        result = analyzer.debt_quality_analysis(data)
        assert result.dq_score >= 10.0
        assert result.dq_grade == "Excellent"

    def test_heavy_debt_weak(self, analyzer):
        """Very high leverage."""
        data = FinancialData(
            total_debt=4_000_000,
            total_equity=1_000_000,
            total_assets=5_000_000,
        )
        # D/E=4.0>3.0=>base 1.0
        result = analyzer.debt_quality_analysis(data)
        assert result.dq_score <= 2.0
        assert result.dq_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase267EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_quality_analysis(FinancialData())
        assert isinstance(result, DebtQualityResult)
        assert result.dq_score == 0.0

    def test_no_equity(self, analyzer):
        data = FinancialData(total_debt=400_000)
        result = analyzer.debt_quality_analysis(data)
        assert result.dq_score == 0.0

    def test_no_debt(self, analyzer):
        data = FinancialData(total_equity=1_200_000)
        result = analyzer.debt_quality_analysis(data)
        assert result.dq_score == 0.0

    def test_zero_equity(self, analyzer):
        data = FinancialData(total_equity=0, total_debt=400_000)
        result = analyzer.debt_quality_analysis(data)
        assert result.dq_score == 0.0
