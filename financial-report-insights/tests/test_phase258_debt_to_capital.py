"""Phase 258 Tests: Debt-to-Capital Analysis.

Tests for debt_to_capital_analysis() and DebtToCapitalResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtToCapitalResult,
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


class TestDebtToCapitalDataclass:
    def test_defaults(self):
        r = DebtToCapitalResult()
        assert r.debt_to_capital is None
        assert r.debt_to_equity is None
        assert r.long_term_debt_to_capital is None
        assert r.equity_ratio is None
        assert r.net_debt_to_capital is None
        assert r.financial_risk_index is None
        assert r.dtc_score == 0.0
        assert r.dtc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtToCapitalResult(debt_to_capital=0.25, dtc_grade="Excellent")
        assert r.debt_to_capital == 0.25
        assert r.dtc_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestDebtToCapitalAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert isinstance(result, DebtToCapitalResult)

    def test_debt_to_capital(self, analyzer, sample_data):
        """D/C = 400k/(400k+1.2M) = 400k/1.6M = 0.25."""
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert result.debt_to_capital == pytest.approx(0.25, abs=0.01)

    def test_debt_to_equity(self, analyzer, sample_data):
        """D/E = 400k/1.2M = 0.333."""
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.01)

    def test_equity_ratio(self, analyzer, sample_data):
        """E/C = 1.2M/1.6M = 0.75."""
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert result.equity_ratio == pytest.approx(0.75, abs=0.01)

    def test_net_debt_to_capital(self, analyzer, sample_data):
        """Net D = 400k-50k = 350k. ND/C = 350k/1.6M = 0.21875."""
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert result.net_debt_to_capital == pytest.approx(0.219, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert result.dtc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert "Debt-to-Capital" in result.summary


# ===== SCORING TESTS =====


class TestDebtToCapitalScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """DTC=0.25 in (0.20,0.30]=>base 8.5. NDC=0.219<0.25-0.05=0.20? No, 0.219<0.25-0.05=0.20 is false. ER=0.75>=0.60(+0.5). Score=9.0."""
        result = analyzer.debt_to_capital_analysis(sample_data)
        assert result.dtc_score == pytest.approx(9.0, abs=0.5)
        assert result.dtc_grade == "Excellent"

    def test_low_debt_excellent(self, analyzer):
        """Very low debt."""
        data = FinancialData(
            total_debt=100_000,
            total_equity=900_000,
            cash=80_000,
        )
        # DTC=100k/1M=0.10<=0.20=>base 10. NDC=20k/1M=0.02<0.10-0.05(+0.5). ER=0.90>=0.60(+0.5). Score=10.
        result = analyzer.debt_to_capital_analysis(data)
        assert result.dtc_score >= 10.0
        assert result.dtc_grade == "Excellent"

    def test_high_debt_weak(self, analyzer):
        """Very high debt."""
        data = FinancialData(
            total_debt=800_000,
            total_equity=200_000,
        )
        # DTC=800k/1M=0.80>0.75=>base 1.0
        result = analyzer.debt_to_capital_analysis(data)
        assert result.dtc_score <= 2.0
        assert result.dtc_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase258EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_to_capital_analysis(FinancialData())
        assert isinstance(result, DebtToCapitalResult)
        assert result.dtc_score == 0.0

    def test_no_debt(self, analyzer):
        data = FinancialData(total_equity=1_000_000)
        result = analyzer.debt_to_capital_analysis(data)
        assert result.dtc_score == 0.0

    def test_no_equity(self, analyzer):
        data = FinancialData(total_debt=400_000)
        result = analyzer.debt_to_capital_analysis(data)
        assert result.dtc_score == 0.0

    def test_negative_equity(self, analyzer):
        data = FinancialData(total_debt=400_000, total_equity=-100_000)
        result = analyzer.debt_to_capital_analysis(data)
        assert result.dtc_score == 0.0
