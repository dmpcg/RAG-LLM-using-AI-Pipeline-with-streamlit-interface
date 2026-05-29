"""Phase 286 Tests: Net Debt Position Analysis.

Tests for net_debt_position_analysis() and NetDebtPositionResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    NetDebtPositionResult,
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


class TestNetDebtPositionDataclass:
    def test_defaults(self):
        r = NetDebtPositionResult()
        assert r.net_debt is None
        assert r.net_debt_to_ebitda is None
        assert r.net_debt_to_equity is None
        assert r.net_debt_to_assets is None
        assert r.cash_to_debt is None
        assert r.net_debt_to_ocf is None
        assert r.ndp_score == 0.0
        assert r.ndp_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = NetDebtPositionResult(net_debt=350_000, ndp_grade="Good")
        assert r.net_debt == 350_000
        assert r.ndp_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestNetDebtPositionAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.net_debt_position_analysis(sample_data)
        assert isinstance(result, NetDebtPositionResult)

    def test_net_debt(self, analyzer, sample_data):
        """Net Debt = 400k - 50k = 350k."""
        result = analyzer.net_debt_position_analysis(sample_data)
        assert result.net_debt == pytest.approx(350_000, abs=100)

    def test_net_debt_to_ebitda(self, analyzer, sample_data):
        """350k/250k = 1.40."""
        result = analyzer.net_debt_position_analysis(sample_data)
        assert result.net_debt_to_ebitda == pytest.approx(1.40, abs=0.01)

    def test_cash_to_debt(self, analyzer, sample_data):
        """50k/400k = 0.125."""
        result = analyzer.net_debt_position_analysis(sample_data)
        assert result.cash_to_debt == pytest.approx(0.125, abs=0.01)

    def test_net_debt_to_equity(self, analyzer, sample_data):
        """350k/1.2M = 0.292."""
        result = analyzer.net_debt_position_analysis(sample_data)
        assert result.net_debt_to_equity == pytest.approx(0.292, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.net_debt_position_analysis(sample_data)
        assert result.ndp_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.net_debt_position_analysis(sample_data)
        assert "Net Debt Position" in result.summary


# ===== SCORING TESTS =====


class TestNetDebtPositionScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """ND/EBITDA=1.40 in (1.0,2.0]=>base 7.0. Cash/Debt=0.125<0.30(no adj). Debt>0(+0.5). Score=7.5."""
        result = analyzer.net_debt_position_analysis(sample_data)
        assert result.ndp_score == pytest.approx(7.5, abs=0.5)
        assert result.ndp_grade in ["Good", "Excellent"]

    def test_net_cash_position(self, analyzer):
        """Net cash (more cash than debt)."""
        data = FinancialData(
            total_debt=200_000,
            cash=500_000,
            ebitda=300_000,
        )
        # Net Debt=-300k<=0=>base 10. Cash/Debt=2.5>=0.30(+0.5). Debt>0(+0.5). Score=10 (capped).
        result = analyzer.net_debt_position_analysis(data)
        assert result.ndp_score >= 10.0
        assert result.ndp_grade == "Excellent"

    def test_heavily_indebted(self, analyzer):
        """Very high net debt."""
        data = FinancialData(
            total_debt=2_000_000,
            cash=50_000,
            ebitda=250_000,
        )
        # ND=1.95M. ND/EBITDA=7.8>6.0=>base 1.0. Cash/Debt=0.025<0.30(no adj). Debt>0(+0.5). Score=1.5.
        result = analyzer.net_debt_position_analysis(data)
        assert result.ndp_score <= 2.0
        assert result.ndp_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase286EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.net_debt_position_analysis(FinancialData())
        assert isinstance(result, NetDebtPositionResult)
        assert result.ndp_score == 0.0

    def test_no_debt(self, analyzer):
        data = FinancialData(cash=50_000)
        result = analyzer.net_debt_position_analysis(data)
        assert result.ndp_score == 0.0

    def test_zero_debt(self, analyzer):
        data = FinancialData(total_debt=0, cash=50_000)
        result = analyzer.net_debt_position_analysis(data)
        assert result.ndp_score == 0.0

    def test_no_cash(self, analyzer):
        data = FinancialData(total_debt=400_000, ebitda=250_000)
        result = analyzer.net_debt_position_analysis(data)
        assert isinstance(result, NetDebtPositionResult)
        assert result.net_debt == pytest.approx(400_000, abs=100)
