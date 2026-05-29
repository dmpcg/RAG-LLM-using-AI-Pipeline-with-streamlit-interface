"""Phase 236 Tests: Fixed Cost Leverage Ratio Analysis.

Tests for fixed_cost_leverage_ratio_analysis() and FixedCostLeverageRatioResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FixedCostLeverageRatioResult,
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


class TestFixedCostLeverageRatioDataclass:
    def test_defaults(self):
        r = FixedCostLeverageRatioResult()
        assert r.dol is None
        assert r.contribution_margin is None
        assert r.oi_to_revenue is None
        assert r.cogs_to_revenue is None
        assert r.opex_to_revenue is None
        assert r.breakeven_proxy is None
        assert r.fclr_score == 0.0
        assert r.fclr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FixedCostLeverageRatioResult(dol=2.0, fclr_grade="Excellent")
        assert r.dol == 2.0
        assert r.fclr_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestFixedCostLeverageRatioAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert isinstance(result, FixedCostLeverageRatioResult)

    def test_dol(self, analyzer, sample_data):
        """DOL = GP/OI = 400k/200k = 2.0."""
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert result.dol == pytest.approx(2.0, abs=0.05)

    def test_contribution_margin(self, analyzer, sample_data):
        """CM = GP/Rev = 400k/1M = 0.40."""
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert result.contribution_margin == pytest.approx(0.40, abs=0.005)

    def test_oi_to_revenue(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert result.oi_to_revenue == pytest.approx(0.20, abs=0.005)

    def test_cogs_to_revenue(self, analyzer, sample_data):
        """COGS/Rev = 600k/1M = 0.60."""
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert result.cogs_to_revenue == pytest.approx(0.60, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert result.fclr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert "Fixed Cost Leverage Ratio" in result.summary


# ===== SCORING TESTS =====


class TestFixedCostLeverageRatioScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """DOL=2.0 <=2.0 => base 8.5. OI/Rev=0.20 >=0.15 => +0.5. COGS/Rev=0.60 no adj. Score=9.0."""
        result = analyzer.fixed_cost_leverage_ratio_analysis(sample_data)
        assert result.fclr_score == pytest.approx(9.0, abs=0.5)
        assert result.fclr_grade == "Excellent"

    def test_low_leverage(self, analyzer):
        """Very low DOL + low cost structure."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=700_000,
            operating_income=500_000,
            cogs=300_000,
            operating_expenses=200_000,
        )
        # DOL=700k/500k=1.4 <=1.5 => base 10. OI/Rev=0.50 >=0.15 => +0.5. COGS/Rev=0.30 <=0.50 => +0.5. Score=10.
        result = analyzer.fixed_cost_leverage_ratio_analysis(data)
        assert result.fclr_score >= 10.0
        assert result.fclr_grade == "Excellent"

    def test_high_leverage(self, analyzer):
        """Very high DOL."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=500_000,
            operating_income=50_000,
            cogs=500_000,
            operating_expenses=450_000,
        )
        # DOL=500k/50k=10.0 >5.0 => base 1.0. OI/Rev=0.05 <0.05 not triggered (==0.05). COGS/Rev=0.50 <=0.50 => +0.5. Score=1.5.
        result = analyzer.fixed_cost_leverage_ratio_analysis(data)
        assert result.fclr_score <= 2.0
        assert result.fclr_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase236EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.fixed_cost_leverage_ratio_analysis(FinancialData())
        assert isinstance(result, FixedCostLeverageRatioResult)
        assert result.fclr_score == 0.0

    def test_no_gp(self, analyzer):
        """GP=None => insufficient data => score 0."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
        )
        result = analyzer.fixed_cost_leverage_ratio_analysis(data)
        assert result.dol is None
        assert result.fclr_score == 0.0

    def test_no_oi(self, analyzer):
        """OI=None => insufficient data => score 0."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=400_000,
        )
        result = analyzer.fixed_cost_leverage_ratio_analysis(data)
        assert result.dol is None
        assert result.fclr_score == 0.0

    def test_zero_oi(self, analyzer):
        """OI=0 => DOL=None (division by zero) => score 0."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=400_000,
            operating_income=0,
        )
        result = analyzer.fixed_cost_leverage_ratio_analysis(data)
        assert result.fclr_score == 0.0
