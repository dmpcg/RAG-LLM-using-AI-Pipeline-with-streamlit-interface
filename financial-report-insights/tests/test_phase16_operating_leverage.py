"""Phase 16 Tests: Operating Leverage & Break-Even Analysis.

Tests for operating_leverage_analysis() and OperatingLeverageResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperatingLeverageResult,
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
        inventory=100_000,
        accounts_receivable=150_000,
        accounts_payable=80_000,
        total_debt=400_000,
        retained_earnings=600_000,
        depreciation=50_000,
        interest_expense=30_000,
        operating_cash_flow=220_000,
        capex=80_000,
    )


# ===== DATACLASS TESTS =====


class TestOperatingLeverageDataclass:
    def test_defaults(self):
        r = OperatingLeverageResult()
        assert r.degree_of_operating_leverage is None
        assert r.contribution_margin is None
        assert r.break_even_revenue is None
        assert r.margin_of_safety is None
        assert r.cost_structure == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperatingLeverageResult(
            degree_of_operating_leverage=2.5,
            cost_structure="High Fixed",
        )
        assert r.degree_of_operating_leverage == 2.5
        assert r.cost_structure == "High Fixed"


# ===== OPERATING LEVERAGE ANALYSIS =====


class TestOperatingLeverageAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert isinstance(result, OperatingLeverageResult)

    def test_contribution_margin_computed(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.contribution_margin is not None
        assert result.contribution_margin > 0

    def test_cm_ratio_computed(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.contribution_margin_ratio is not None
        assert 0 < result.contribution_margin_ratio < 1

    def test_dol_computed(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.degree_of_operating_leverage is not None
        assert result.degree_of_operating_leverage > 0

    def test_break_even_computed(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.break_even_revenue is not None
        assert result.break_even_revenue > 0

    def test_break_even_below_revenue(self, analyzer, sample_data):
        """Profitable company: break-even should be below actual revenue."""
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.break_even_revenue < sample_data.revenue

    def test_margin_of_safety_positive(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.margin_of_safety is not None
        assert result.margin_of_safety > 0

    def test_margin_of_safety_pct(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.margin_of_safety_pct is not None
        assert 0 < result.margin_of_safety_pct < 1

    def test_fixed_costs_estimated(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.estimated_fixed_costs is not None
        assert result.estimated_fixed_costs > 0

    def test_variable_costs_estimated(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.estimated_variable_costs is not None
        assert result.estimated_variable_costs > 0

    def test_cost_structure_assigned(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert result.cost_structure in ["High Fixed", "Balanced", "High Variable", "N/A"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operating_leverage_analysis(sample_data)
        assert "Operating Leverage" in result.summary

    def test_dol_is_cm_over_oi(self, analyzer, sample_data):
        """DOL = Contribution Margin / Operating Income."""
        result = analyzer.operating_leverage_analysis(sample_data)
        if result.degree_of_operating_leverage is not None and result.contribution_margin is not None:
            expected = result.contribution_margin / sample_data.operating_income
            assert abs(result.degree_of_operating_leverage - expected) < 0.01

    def test_costs_sum_correctly(self, analyzer, sample_data):
        """Fixed + Variable should equal total estimated costs."""
        result = analyzer.operating_leverage_analysis(sample_data)
        total = (result.estimated_fixed_costs or 0) + (result.estimated_variable_costs or 0)
        # Total should equal cogs + opex
        expected = sample_data.cogs + sample_data.operating_expenses
        assert abs(total - expected) < 1


# ===== HIGH FIXED COST COMPANY =====


class TestHighFixedCostCompany:
    def test_high_depreciation_means_high_fixed(self, analyzer):
        """Company with large depreciation has high fixed costs."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=200_000,
            operating_expenses=500_000,
            operating_income=300_000,
            depreciation=400_000,  # Large depreciation
        )
        result = analyzer.operating_leverage_analysis(data)
        assert result.cost_structure in ["High Fixed", "Balanced"]

    def test_high_dol_with_high_fixed(self, analyzer):
        """High fixed cost company should have higher DOL."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=100_000,
            operating_expenses=600_000,
            operating_income=300_000,
            depreciation=500_000,
        )
        result = analyzer.operating_leverage_analysis(data)
        if result.degree_of_operating_leverage is not None:
            assert result.degree_of_operating_leverage > 1.0


# ===== EDGE CASES =====


class TestPhase16EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operating_leverage_analysis(FinancialData())
        assert isinstance(result, OperatingLeverageResult)
        # No revenue means no break-even
        assert result.break_even_revenue is None

    def test_zero_operating_income(self, analyzer):
        """Zero OI means DOL is undefined."""
        data = FinancialData(
            revenue=500_000,
            cogs=400_000,
            operating_expenses=100_000,
            operating_income=0,
        )
        result = analyzer.operating_leverage_analysis(data)
        assert result.degree_of_operating_leverage is None

    def test_negative_operating_income(self, analyzer):
        """Negative OI gives negative DOL."""
        data = FinancialData(
            revenue=500_000,
            cogs=400_000,
            operating_expenses=200_000,
            operating_income=-100_000,
        )
        result = analyzer.operating_leverage_analysis(data)
        # DOL could be negative (contribution margin / negative OI)
        assert result.degree_of_operating_leverage is not None

    def test_zero_revenue(self, analyzer):
        data = FinancialData(
            revenue=0,
            cogs=0,
            operating_expenses=50_000,
        )
        result = analyzer.operating_leverage_analysis(data)
        assert isinstance(result, OperatingLeverageResult)
        assert result.contribution_margin is None or result.contribution_margin == 0

    def test_no_depreciation(self, analyzer):
        """No depreciation: all opex split 50/50."""
        data = FinancialData(
            revenue=800_000,
            cogs=300_000,
            operating_expenses=200_000,
            operating_income=300_000,
            depreciation=0,
        )
        result = analyzer.operating_leverage_analysis(data)
        assert result.estimated_fixed_costs is not None
        # Fixed = 0 (dep) + 200k * 0.5 = 100k
        assert abs(result.estimated_fixed_costs - 100_000) < 1

    def test_margin_of_safety_zero_at_breakeven(self, analyzer):
        """Company exactly at break-even should have ~0 margin of safety."""
        # Build data where revenue ~= break-even
        data = FinancialData(
            revenue=200_000,
            cogs=150_000,
            operating_expenses=50_000,
            operating_income=0,
            depreciation=10_000,
        )
        result = analyzer.operating_leverage_analysis(data)
        # With 0 OI, margin of safety should be near zero or break-even near revenue
        if result.margin_of_safety is not None:
            assert abs(result.margin_of_safety) < 50_000  # Roughly near zero

    def test_pure_variable_cost(self, analyzer):
        """Company with no fixed costs (no depreciation, no opex)."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=700_000,
            operating_expenses=0,
            operating_income=300_000,
            depreciation=0,
        )
        result = analyzer.operating_leverage_analysis(data)
        assert result.cost_structure == "High Variable"
        assert result.estimated_fixed_costs == 0
