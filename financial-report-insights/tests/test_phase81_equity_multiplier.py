"""Phase 81 Tests: Equity Multiplier Analysis.

Tests for equity_multiplier_analysis() and EquityMultiplierResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    EquityMultiplierResult,
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


class TestEquityMultiplierDataclass:
    def test_defaults(self):
        r = EquityMultiplierResult()
        assert r.equity_multiplier is None
        assert r.debt_ratio is None
        assert r.equity_ratio is None
        assert r.financial_leverage_index is None
        assert r.dupont_roe is None
        assert r.leverage_spread is None
        assert r.em_score == 0.0
        assert r.em_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = EquityMultiplierResult(equity_multiplier=2.0, em_grade="Good")
        assert r.equity_multiplier == 2.0
        assert r.em_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestEquityMultiplierAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert isinstance(result, EquityMultiplierResult)

    def test_equity_multiplier(self, analyzer, sample_data):
        """EM = TA/TE = 2M/1.2M = 1.667."""
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert result.equity_multiplier == pytest.approx(1.667, abs=0.01)

    def test_debt_ratio(self, analyzer, sample_data):
        """DR = TL/TA = 800k/2M = 0.40."""
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert result.debt_ratio == pytest.approx(0.40, abs=0.01)

    def test_equity_ratio(self, analyzer, sample_data):
        """ER = TE/TA = 1.2M/2M = 0.60."""
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert result.equity_ratio == pytest.approx(0.60, abs=0.01)

    def test_financial_leverage_index(self, analyzer, sample_data):
        """FLI = ROE/ROA = (150k/1.2M)/(150k/2M) = 0.125/0.075 = 1.667."""
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert result.financial_leverage_index == pytest.approx(1.667, abs=0.01)

    def test_dupont_roe(self, analyzer, sample_data):
        """DuPont = (NI/Rev)*(Rev/TA)*(TA/TE) = 0.15*0.50*1.667 = 0.125."""
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert result.dupont_roe == pytest.approx(0.125, abs=0.01)

    def test_leverage_spread(self, analyzer, sample_data):
        """LS = ROA - CoD = 0.075 - (30k/400k) = 0.075 - 0.075 = 0.0."""
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert result.leverage_spread == pytest.approx(0.0, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert result.em_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.equity_multiplier_analysis(sample_data)
        assert "Equity Multiplier" in result.summary


# ===== SCORING TESTS =====


class TestEquityMultiplierScoring:
    def test_low_leverage(self, analyzer):
        """EM <= 1.5 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=1_000_000,
            total_liabilities=200_000,
            total_equity=800_000,
            interest_expense=10_000,
            total_debt=100_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        # EM=1.25, base 10. DR=0.20 (<=0.30 => +0.5). LS=0.10-0.10=0.0 (no adj). => 10 capped
        assert result.em_score >= 10.0
        assert result.em_grade == "Excellent"

    def test_moderate_leverage(self, analyzer, sample_data):
        """EM ~ 1.667 => base 8.5."""
        result = analyzer.equity_multiplier_analysis(sample_data)
        # EM=1.667 (base 8.5). DR=0.40 (no adj). LS=0.0 (no adj). => 8.5
        assert result.em_score >= 8.0

    def test_high_leverage(self, analyzer):
        """EM ~ 4.0 => base 4.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=50_000,
            total_assets=2_000_000,
            total_liabilities=1_500_000,
            total_equity=500_000,
            interest_expense=100_000,
            total_debt=1_000_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        # EM=4.0, base 4.0. DR=0.75 (>0.70 => -0.5). LS=0.025-0.10=-0.075 (<-0.05 => -0.5). => 3.0
        assert result.em_score < 4.0
        assert result.em_grade == "Weak"

    def test_very_high_leverage(self, analyzer):
        """EM > 5.0 => base 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=20_000,
            total_assets=3_000_000,
            total_liabilities=2_500_000,
            total_equity=500_000,
            interest_expense=200_000,
            total_debt=2_000_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        # EM=6.0, base 1.0. DR=0.833 (>0.70 => -0.5). => 0.5
        assert result.em_score < 2.0
        assert result.em_grade == "Weak"

    def test_debt_ratio_bonus(self, analyzer):
        """DR <= 0.30 => +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=1_000_000,
            total_liabilities=250_000,
            total_equity=750_000,
            interest_expense=5_000,
            total_debt=50_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        # EM=1.333 (base 10). DR=0.25 (<=0.30 => +0.5). => 10 capped
        assert result.em_score >= 10.0

    def test_leverage_spread_bonus(self, analyzer):
        """LS > 0.05 => +0.5."""
        data = FinancialData(
            revenue=5_000_000,
            net_income=500_000,
            total_assets=2_000_000,
            total_liabilities=800_000,
            total_equity=1_200_000,
            interest_expense=10_000,
            total_debt=400_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        # EM=1.667 (base 8.5). ROA=0.25, CoD=0.025, LS=0.225 (>0.05 => +0.5). => 9.0
        assert result.em_score >= 9.0


# ===== EDGE CASES =====


class TestPhase81EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.equity_multiplier_analysis(FinancialData())
        assert isinstance(result, EquityMultiplierResult)
        assert result.equity_multiplier is None

    def test_no_equity(self, analyzer):
        """TE=0 => empty result."""
        data = FinancialData(
            total_assets=1_000_000,
            total_liabilities=1_000_000,
            total_equity=0,
        )
        result = analyzer.equity_multiplier_analysis(data)
        assert result.em_score == 0.0

    def test_no_total_assets(self, analyzer):
        """TA=0 => empty result."""
        data = FinancialData(
            total_equity=500_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        assert result.em_score == 0.0

    def test_no_revenue(self, analyzer):
        """No revenue => DuPont ROE is None."""
        data = FinancialData(
            net_income=100_000,
            total_assets=1_000_000,
            total_liabilities=200_000,
            total_equity=800_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        assert result.equity_multiplier is not None
        assert result.dupont_roe is None

    def test_no_debt(self, analyzer):
        """No debt => leverage spread is None."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=1_000_000,
            total_liabilities=200_000,
            total_equity=800_000,
        )
        result = analyzer.equity_multiplier_analysis(data)
        assert result.leverage_spread is None
