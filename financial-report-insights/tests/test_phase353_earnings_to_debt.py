"""Phase 353 Tests: Earnings To Debt Analysis.

Tests for earnings_to_debt_analysis() and EarningsToDebtResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    EarningsToDebtResult,
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


class TestEarningsToDebtDataclass:
    def test_defaults(self):
        r = EarningsToDebtResult()
        assert r.etd_ratio is None
        assert r.ni_to_interest is None
        assert r.ni_to_liabilities is None
        assert r.earnings_yield_on_debt is None
        assert r.debt_years_from_earnings is None
        assert r.etd_spread is None
        assert r.etd_score == 0.0
        assert r.etd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = EarningsToDebtResult(etd_ratio=0.375, etd_grade="Excellent")
        assert r.etd_ratio == 0.375
        assert r.etd_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestEarningsToDebtAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.earnings_to_debt_analysis(sample_data)
        assert isinstance(result, EarningsToDebtResult)

    def test_etd_ratio(self, analyzer, sample_data):
        """NI/TD = 150k/400k = 0.375."""
        result = analyzer.earnings_to_debt_analysis(sample_data)
        assert result.etd_ratio == pytest.approx(0.375, abs=0.001)

    def test_ni_to_interest(self, analyzer, sample_data):
        """NI/IE = 150k/30k = 5.0."""
        result = analyzer.earnings_to_debt_analysis(sample_data)
        assert result.ni_to_interest == pytest.approx(5.0, abs=0.01)

    def test_ni_to_liabilities(self, analyzer, sample_data):
        """NI/TL = 150k/800k = 0.1875."""
        result = analyzer.earnings_to_debt_analysis(sample_data)
        assert result.ni_to_liabilities == pytest.approx(0.1875, abs=0.001)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.earnings_to_debt_analysis(sample_data)
        assert result.etd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.earnings_to_debt_analysis(sample_data)
        assert "Earnings To Debt" in result.summary


# ===== SCORING TESTS =====


class TestEarningsToDebtScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """NI/TD=0.375 in [0.30,0.40)=>base 8.5. NI/IE=5.0>=3.0(+0.5). Both>0(+0.5). Score=9.5."""
        result = analyzer.earnings_to_debt_analysis(sample_data)
        assert result.etd_score == pytest.approx(9.5, abs=0.5)
        assert result.etd_grade in ["Excellent", "Good"]

    def test_high_earnings(self, analyzer):
        """Very high NI/TD — strong earnings capacity."""
        data = FinancialData(
            net_income=300_000,
            total_debt=400_000,
            interest_expense=30_000,
            total_liabilities=800_000,
        )
        # NI/TD=0.75>=0.40=>base 10. NI/IE=10.0>=3.0(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.earnings_to_debt_analysis(data)
        assert result.etd_score >= 10.0
        assert result.etd_grade == "Excellent"

    def test_low_earnings(self, analyzer):
        """Low NI/TD — weak earnings capacity."""
        data = FinancialData(
            net_income=10_000,
            total_debt=400_000,
            interest_expense=30_000,
            total_liabilities=800_000,
        )
        # NI/TD=0.025<0.05=>base 1.0. NI/IE=0.33<3.0(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.earnings_to_debt_analysis(data)
        assert result.etd_score <= 3.0
        assert result.etd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase353EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.earnings_to_debt_analysis(FinancialData())
        assert isinstance(result, EarningsToDebtResult)
        assert result.etd_score == 0.0

    def test_no_total_debt(self, analyzer):
        """TD=None => ratio=None => score 0."""
        data = FinancialData(net_income=150_000)
        result = analyzer.earnings_to_debt_analysis(data)
        assert result.etd_ratio is None
        assert result.etd_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => ratio=None."""
        data = FinancialData(total_debt=400_000)
        result = analyzer.earnings_to_debt_analysis(data)
        assert result.etd_score == 0.0
