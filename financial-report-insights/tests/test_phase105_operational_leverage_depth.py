"""Phase 105 Tests: Operational Leverage Depth Analysis.

Tests for operational_leverage_depth_analysis() and OperationalLeverageDepthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperationalLeverageDepthResult,
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


class TestOperationalLeverageDepthDataclass:
    def test_defaults(self):
        r = OperationalLeverageDepthResult()
        assert r.fixed_cost_ratio is None
        assert r.variable_cost_ratio is None
        assert r.contribution_margin is None
        assert r.dol_proxy is None
        assert r.breakeven_coverage is None
        assert r.cost_flexibility is None
        assert r.old_score == 0.0
        assert r.old_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperationalLeverageDepthResult(contribution_margin=0.40, old_grade="Good")
        assert r.contribution_margin == 0.40
        assert r.old_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestOperationalLeverageDepthAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert isinstance(result, OperationalLeverageDepthResult)

    def test_fixed_cost_ratio(self, analyzer, sample_data):
        """FCR = (OpEx+D&A) / (COGS+OpEx) = (200k+50k) / (600k+200k) = 250k/800k = 0.3125."""
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.fixed_cost_ratio == pytest.approx(0.3125, abs=0.01)

    def test_variable_cost_ratio(self, analyzer, sample_data):
        """VCR = COGS / (COGS+OpEx) = 600k/800k = 0.75."""
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.variable_cost_ratio == pytest.approx(0.75, abs=0.01)

    def test_contribution_margin(self, analyzer, sample_data):
        """CM = GP/Rev = 400k/1M = 0.40."""
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.contribution_margin == pytest.approx(0.40, abs=0.01)

    def test_dol_proxy(self, analyzer, sample_data):
        """DOL = GP/OI = 400k/200k = 2.0."""
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.dol_proxy == pytest.approx(2.0, abs=0.01)

    def test_breakeven_coverage(self, analyzer, sample_data):
        """BC = Rev / (COGS+OpEx) = 1M/800k = 1.25."""
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.breakeven_coverage == pytest.approx(1.25, abs=0.01)

    def test_cost_flexibility(self, analyzer, sample_data):
        """CF = COGS / (COGS+OpEx) = 600k/800k = 0.75."""
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.cost_flexibility == pytest.approx(0.75, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.old_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert "Operational Leverage Depth" in result.summary


# ===== SCORING TESTS =====


class TestOperationalLeverageDepthScoring:
    def test_good_margins(self, analyzer, sample_data):
        """CM=0.40 => base 7.0. BC=1.25 (not >=1.5). CF=0.75 >=0.60 => +0.5. Score=7.5."""
        result = analyzer.operational_leverage_depth_analysis(sample_data)
        assert result.old_score >= 7.0
        assert result.old_grade in ["Good", "Excellent"]

    def test_excellent_margins(self, analyzer):
        """CM >= 0.60 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=200_000,
            gross_profit=800_000,
            operating_expenses=100_000,
            operating_income=700_000,
            depreciation=20_000,
        )
        result = analyzer.operational_leverage_depth_analysis(data)
        # CM=800k/1M=0.80 => base 10. BC=1M/300k=3.33 >=1.5 => +0.5. CF=200k/300k=0.67 >=0.60 => +0.5. Score=10(capped).
        assert result.old_score >= 10.0
        assert result.old_grade == "Excellent"

    def test_weak_margins(self, analyzer):
        """CM < 0.10 => base 1.0."""
        data = FinancialData(
            revenue=500_000,
            cogs=480_000,
            gross_profit=20_000,
            operating_expenses=50_000,
            operating_income=-30_000,
        )
        result = analyzer.operational_leverage_depth_analysis(data)
        # CM=20k/500k=0.04 => base 1.0. BC=500k/530k=0.94 <1.0 => -0.5. CF=480k/530k=0.91 >=0.60 => +0.5. Score=1.0.
        assert result.old_score <= 2.0
        assert result.old_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase105EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operational_leverage_depth_analysis(FinancialData())
        assert isinstance(result, OperationalLeverageDepthResult)
        assert result.old_score == 0.0

    def test_revenue_only(self, analyzer):
        """Rev only => total_costs=0 => early return."""
        data = FinancialData(revenue=1_000_000)
        result = analyzer.operational_leverage_depth_analysis(data)
        assert result.old_score == 0.0

    def test_no_operating_income(self, analyzer):
        """OI=0 => dol_proxy=None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            gross_profit=400_000,
            operating_expenses=400_000,
        )
        result = analyzer.operational_leverage_depth_analysis(data)
        assert result.dol_proxy is None

    def test_high_breakeven_bonus(self, analyzer):
        """BC >= 1.5 => +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=300_000,
            gross_profit=700_000,
            operating_expenses=200_000,
            operating_income=500_000,
            depreciation=30_000,
        )
        result = analyzer.operational_leverage_depth_analysis(data)
        # BC = 1M/500k = 2.0 >= 1.5 => +0.5
        assert result.breakeven_coverage >= 1.5
