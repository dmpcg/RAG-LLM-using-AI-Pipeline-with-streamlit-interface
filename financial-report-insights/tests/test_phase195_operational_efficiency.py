"""Phase 195 Tests: Operational Efficiency Analysis.

Tests for operational_efficiency_analysis() and OperationalEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperationalEfficiencyResult,
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


class TestOperationalEfficiencyDataclass:
    def test_defaults(self):
        r = OperationalEfficiencyResult()
        assert r.oi_margin is None
        assert r.revenue_to_assets is None
        assert r.gross_profit_per_asset is None
        assert r.opex_efficiency is None
        assert r.asset_utilization is None
        assert r.income_per_liability is None
        assert r.oe_score == 0.0
        assert r.oe_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperationalEfficiencyResult(oi_margin=0.20, oe_grade="Excellent")
        assert r.oi_margin == 0.20
        assert r.oe_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestOperationalEfficiencyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert isinstance(result, OperationalEfficiencyResult)

    def test_oi_margin(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.oi_margin == pytest.approx(0.20, abs=0.005)

    def test_revenue_to_assets(self, analyzer, sample_data):
        """Rev/TA = 1M/2M = 0.50."""
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.revenue_to_assets == pytest.approx(0.50, abs=0.005)

    def test_gross_profit_per_asset(self, analyzer, sample_data):
        """GP/TA = 400k/2M = 0.20."""
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.gross_profit_per_asset == pytest.approx(0.20, abs=0.005)

    def test_opex_efficiency(self, analyzer, sample_data):
        """Rev/OpEx = 1M/200k = 5.00."""
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.opex_efficiency == pytest.approx(5.00, abs=0.01)

    def test_asset_utilization(self, analyzer, sample_data):
        """Rev/CA = 1M/500k = 2.00."""
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.asset_utilization == pytest.approx(2.00, abs=0.01)

    def test_income_per_liability(self, analyzer, sample_data):
        """OI/TL = 200k/800k = 0.25."""
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.income_per_liability == pytest.approx(0.25, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.oe_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert "Operational Efficiency" in result.summary


# ===== SCORING TESTS =====


class TestOperationalEfficiencyScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OIM=0.20 => base 8.5. RtA=0.50 no adj (<0.80). OE=5.00 >=5.0 => +0.5. Score=9.0."""
        result = analyzer.operational_efficiency_analysis(sample_data)
        assert result.oe_score == pytest.approx(9.0, abs=0.5)
        assert result.oe_grade == "Excellent"

    def test_excellent_efficiency(self, analyzer):
        """Very high OI margin."""
        data = FinancialData(
            operating_income=400_000,
            revenue=1_000_000,
            total_assets=1_000_000,
            gross_profit=600_000,
            operating_expenses=100_000,
            current_assets=400_000,
            total_liabilities=300_000,
        )
        result = analyzer.operational_efficiency_analysis(data)
        assert result.oe_score >= 10.0
        assert result.oe_grade == "Excellent"

    def test_weak_efficiency(self, analyzer):
        """Very low OI margin."""
        data = FinancialData(
            operating_income=10_000,
            revenue=1_000_000,
            total_assets=5_000_000,
            gross_profit=200_000,
            operating_expenses=600_000,
            current_assets=1_000_000,
            total_liabilities=3_000_000,
        )
        # OIM=0.01 <0.02 => base 1.0. RtA=1M/5M=0.20 <0.30 => -0.5. OE=1M/600k=1.67 <2.0 => -0.5. Score=0.0.
        result = analyzer.operational_efficiency_analysis(data)
        assert result.oe_score <= 0.5
        assert result.oe_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase195EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operational_efficiency_analysis(FinancialData())
        assert isinstance(result, OperationalEfficiencyResult)
        assert result.oe_score == 0.0

    def test_no_oi(self, analyzer):
        """OI=None => OIM=None, score 0."""
        data = FinancialData(
            revenue=1_000_000,
            total_assets=2_000_000,
        )
        result = analyzer.operational_efficiency_analysis(data)
        assert result.oi_margin is None
        assert result.oe_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => OIM=None."""
        data = FinancialData(
            operating_income=200_000,
            total_assets=2_000_000,
        )
        result = analyzer.operational_efficiency_analysis(data)
        assert result.oi_margin is None
        assert result.oe_score == 0.0

    def test_no_assets(self, analyzer):
        """TA=None => RtA=None, but OIM still works."""
        data = FinancialData(
            operating_income=200_000,
            revenue=1_000_000,
        )
        result = analyzer.operational_efficiency_analysis(data)
        assert result.revenue_to_assets is None
        assert result.oi_margin is not None
