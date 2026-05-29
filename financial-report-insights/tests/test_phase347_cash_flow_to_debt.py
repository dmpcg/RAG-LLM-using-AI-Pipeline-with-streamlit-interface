"""Phase 347 Tests: Cash Flow To Debt Analysis.

Tests for cash_flow_to_debt_analysis() and CashFlowToDebtResult dataclass.
"""

import pytest

from financial_analyzer import (
    CashFlowToDebtResult,
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


class TestCashFlowToDebtDataclass:
    def test_defaults(self):
        r = CashFlowToDebtResult()
        assert r.cf_to_debt is None
        assert r.ocf_to_td is None
        assert r.fcf_to_td is None
        assert r.debt_payback_years is None
        assert r.ocf_to_interest is None
        assert r.cf_debt_spread is None
        assert r.cfd_score == 0.0
        assert r.cfd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CashFlowToDebtResult(ocf_to_td=0.55, cfd_grade="Excellent")
        assert r.ocf_to_td == 0.55
        assert r.cfd_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestCashFlowToDebtAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cash_flow_to_debt_analysis(sample_data)
        assert isinstance(result, CashFlowToDebtResult)

    def test_ocf_to_td(self, analyzer, sample_data):
        """OCF/TD = 220k/400k = 0.55."""
        result = analyzer.cash_flow_to_debt_analysis(sample_data)
        assert result.ocf_to_td == pytest.approx(0.55, abs=0.001)

    def test_fcf_to_td(self, analyzer, sample_data):
        """FCF/TD = (220k-80k)/400k = 0.35."""
        result = analyzer.cash_flow_to_debt_analysis(sample_data)
        assert result.fcf_to_td == pytest.approx(0.35, abs=0.001)

    def test_debt_payback_years(self, analyzer, sample_data):
        """TD/OCF = 400k/220k = 1.818."""
        result = analyzer.cash_flow_to_debt_analysis(sample_data)
        assert result.debt_payback_years == pytest.approx(1.818, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cash_flow_to_debt_analysis(sample_data)
        assert result.cfd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cash_flow_to_debt_analysis(sample_data)
        assert "Cash Flow" in result.summary


# ===== SCORING TESTS =====


class TestCashFlowToDebtScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OCF/TD=0.55>=0.50=>base 10. OCF/Int=7.33>=3.0(+0.5). Both>0(+0.5). Score=10 clamped."""
        result = analyzer.cash_flow_to_debt_analysis(sample_data)
        assert result.cfd_score == pytest.approx(10.0, abs=0.5)
        assert result.cfd_grade == "Excellent"

    def test_high_cash_flow_to_debt(self, analyzer):
        """High OCF/TD — strong debt coverage."""
        data = FinancialData(
            operating_cash_flow=500_000,
            total_debt=400_000,
            interest_expense=30_000,
            capex=80_000,
        )
        result = analyzer.cash_flow_to_debt_analysis(data)
        assert result.cfd_score >= 10.0
        assert result.cfd_grade == "Excellent"

    def test_low_cash_flow_to_debt(self, analyzer):
        """Low OCF/TD — weak debt coverage."""
        data = FinancialData(
            operating_cash_flow=15_000,
            total_debt=400_000,
            interest_expense=30_000,
            capex=10_000,
        )
        # OCF/TD=0.0375<0.05=>base 1.0. OCF/Int=0.5<3.0(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.cash_flow_to_debt_analysis(data)
        assert result.cfd_score <= 3.0
        assert result.cfd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase347EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cash_flow_to_debt_analysis(FinancialData())
        assert isinstance(result, CashFlowToDebtResult)
        assert result.cfd_score == 0.0

    def test_no_total_debt(self, analyzer):
        """TD=None => OCF/TD=None => score 0."""
        data = FinancialData(operating_cash_flow=220_000)
        result = analyzer.cash_flow_to_debt_analysis(data)
        assert result.ocf_to_td is None
        assert result.cfd_score == 0.0

    def test_no_operating_cash_flow(self, analyzer):
        """OCF=None => ratio=None."""
        data = FinancialData(total_debt=400_000)
        result = analyzer.cash_flow_to_debt_analysis(data)
        assert result.cfd_score == 0.0
