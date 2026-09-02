"""Phase 293 Tests: Funding Mix Balance Analysis.

Tests for funding_mix_balance_analysis() and FundingMixBalanceResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FundingMixBalanceResult,
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


class TestFundingMixBalanceDataclass:
    def test_defaults(self):
        r = FundingMixBalanceResult()
        assert r.equity_to_total_capital is None
        assert r.debt_to_equity is None
        assert r.debt_to_total_capital is None
        assert r.equity_multiplier is None
        assert r.leverage_headroom is None
        assert r.funding_stability is None
        assert r.fmb_score == 0.0
        assert r.fmb_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FundingMixBalanceResult(equity_to_total_capital=0.75, fmb_grade="Excellent")
        assert r.equity_to_total_capital == 0.75
        assert r.fmb_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestFundingMixBalanceAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.funding_mix_balance_analysis(sample_data)
        assert isinstance(result, FundingMixBalanceResult)

    def test_equity_to_total_capital(self, analyzer, sample_data):
        """E/(E+D) = 1.2M/(1.2M+400k) = 0.75."""
        result = analyzer.funding_mix_balance_analysis(sample_data)
        assert result.equity_to_total_capital == pytest.approx(0.75, abs=0.01)

    def test_debt_to_equity(self, analyzer, sample_data):
        """D/E = 400k/1.2M = 0.333."""
        result = analyzer.funding_mix_balance_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.01)

    def test_debt_to_total_capital(self, analyzer, sample_data):
        """D/(E+D) = 400k/1.6M = 0.25."""
        result = analyzer.funding_mix_balance_analysis(sample_data)
        assert result.debt_to_total_capital == pytest.approx(0.25, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.funding_mix_balance_analysis(sample_data)
        assert result.fmb_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.funding_mix_balance_analysis(sample_data)
        assert "Funding Mix Balance" in result.summary


# ===== SCORING TESTS =====


class TestFundingMixBalanceScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """E/(E+D)=0.75 in [0.70,0.80)=>base 8.5. D/E=0.333<=0.50(+0.5). E>0&D>=0(+0.5). Score=9.5."""
        result = analyzer.funding_mix_balance_analysis(sample_data)
        assert result.fmb_score == pytest.approx(9.5, abs=0.5)
        assert result.fmb_grade == "Excellent"

    def test_very_equity_heavy(self, analyzer):
        """Almost all equity funded."""
        data = FinancialData(
            total_equity=900_000,
            total_debt=100_000,
        )
        # E/(E+D)=0.90>=0.80=>base 10. D/E=0.111<=0.50(+0.5). E>0&D>=0(+0.5). Score=10 (capped).
        result = analyzer.funding_mix_balance_analysis(data)
        assert result.fmb_score >= 10.0
        assert result.fmb_grade == "Excellent"

    def test_heavily_leveraged(self, analyzer):
        """Very debt heavy."""
        data = FinancialData(
            total_equity=200_000,
            total_debt=1_800_000,
        )
        # E/(E+D)=0.10<0.30=>base 1.0. D/E=9.0>0.50(no adj). E>0&D>=0(+0.5). Score=1.5.
        result = analyzer.funding_mix_balance_analysis(data)
        assert result.fmb_score <= 2.0
        assert result.fmb_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase293EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.funding_mix_balance_analysis(FinancialData())
        assert isinstance(result, FundingMixBalanceResult)
        assert result.fmb_score == 0.0

    def test_no_equity(self, analyzer):
        data = FinancialData(total_debt=400_000)
        result = analyzer.funding_mix_balance_analysis(data)
        assert result.fmb_score == 0.0

    def test_zero_equity(self, analyzer):
        data = FinancialData(total_equity=0, total_debt=400_000)
        result = analyzer.funding_mix_balance_analysis(data)
        assert result.fmb_score == 0.0

    def test_no_debt(self, analyzer):
        data = FinancialData(total_equity=1_200_000)
        result = analyzer.funding_mix_balance_analysis(data)
        assert isinstance(result, FundingMixBalanceResult)
