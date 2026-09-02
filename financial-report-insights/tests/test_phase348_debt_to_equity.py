"""Phase 348 Tests: Debt To Equity Analysis.

Tests for debt_to_equity_analysis() and DebtToEquityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtToEquityResult,
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


class TestDebtToEquityDataclass:
    def test_defaults(self):
        r = DebtToEquityResult()
        assert r.dte_ratio is None
        assert r.td_to_te is None
        assert r.lt_debt_to_equity is None
        assert r.debt_to_assets is None
        assert r.equity_multiplier is None
        assert r.dte_spread is None
        assert r.dte_score == 0.0
        assert r.dte_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtToEquityResult(td_to_te=0.33, dte_grade="Excellent")
        assert r.td_to_te == 0.33
        assert r.dte_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestDebtToEquityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_to_equity_analysis(sample_data)
        assert isinstance(result, DebtToEquityResult)

    def test_td_to_te(self, analyzer, sample_data):
        """TD/TE = 400k/1.2M = 0.333."""
        result = analyzer.debt_to_equity_analysis(sample_data)
        assert result.td_to_te == pytest.approx(0.333, abs=0.001)

    def test_debt_to_assets(self, analyzer, sample_data):
        """TD/TA = 400k/2M = 0.20."""
        result = analyzer.debt_to_equity_analysis(sample_data)
        assert result.debt_to_assets == pytest.approx(0.20, abs=0.001)

    def test_equity_multiplier(self, analyzer, sample_data):
        """TA/TE = 2M/1.2M = 1.667."""
        result = analyzer.debt_to_equity_analysis(sample_data)
        assert result.equity_multiplier == pytest.approx(1.667, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_to_equity_analysis(sample_data)
        assert result.dte_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_to_equity_analysis(sample_data)
        assert "Debt to Equity" in result.summary


# ===== SCORING TESTS =====


class TestDebtToEquityScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """TD/TE=0.333<=0.50=>base 8.5. D/A=0.20<=0.50(+0.5). Both>0(+0.5). Score=9.5."""
        result = analyzer.debt_to_equity_analysis(sample_data)
        assert result.dte_score == pytest.approx(9.5, abs=0.5)
        assert result.dte_grade in ["Excellent", "Good"]

    def test_low_leverage(self, analyzer):
        """Very low TD/TE — conservative capital structure."""
        data = FinancialData(
            total_debt=100_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        # TD/TE=0.083<=0.30=>base 10. D/A=0.05<=0.50(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.debt_to_equity_analysis(data)
        assert result.dte_score >= 10.0
        assert result.dte_grade == "Excellent"

    def test_high_leverage(self, analyzer):
        """High TD/TE — aggressive leverage."""
        data = FinancialData(
            total_debt=3_000_000,
            total_equity=1_200_000,
            total_assets=4_200_000,
        )
        # TD/TE=2.50>2.00=>base 1.0. D/A=0.714>0.50(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.debt_to_equity_analysis(data)
        assert result.dte_score <= 3.0
        assert result.dte_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase348EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_to_equity_analysis(FinancialData())
        assert isinstance(result, DebtToEquityResult)
        assert result.dte_score == 0.0

    def test_no_total_equity(self, analyzer):
        """TE=None => TD/TE=None => score 0."""
        data = FinancialData(total_debt=400_000)
        result = analyzer.debt_to_equity_analysis(data)
        assert result.td_to_te is None
        assert result.dte_score == 0.0

    def test_no_total_debt(self, analyzer):
        """TD=None => ratio=None."""
        data = FinancialData(total_equity=1_200_000)
        result = analyzer.debt_to_equity_analysis(data)
        assert result.dte_score == 0.0
