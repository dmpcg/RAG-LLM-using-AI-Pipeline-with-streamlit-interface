"""Phase 92 Tests: Operational Risk Analysis.

Tests for operational_risk_analysis() and OperationalRiskResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperationalRiskResult,
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


class TestOperationalRiskDataclass:
    def test_defaults(self):
        r = OperationalRiskResult()
        assert r.operating_leverage is None
        assert r.cost_rigidity is None
        assert r.breakeven_ratio is None
        assert r.margin_of_safety is None
        assert r.cash_burn_ratio is None
        assert r.risk_buffer is None
        assert r.or_score == 0.0
        assert r.or_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperationalRiskResult(margin_of_safety=0.20, or_grade="Good")
        assert r.margin_of_safety == 0.20
        assert r.or_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestOperationalRiskAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operational_risk_analysis(sample_data)
        assert isinstance(result, OperationalRiskResult)

    def test_breakeven_ratio(self, analyzer, sample_data):
        """(COGS+OpEx)/Rev = 800k/1M = 0.80."""
        result = analyzer.operational_risk_analysis(sample_data)
        assert result.breakeven_ratio == pytest.approx(0.80, abs=0.01)

    def test_margin_of_safety(self, analyzer, sample_data):
        """1 - 0.80 = 0.20."""
        result = analyzer.operational_risk_analysis(sample_data)
        assert result.margin_of_safety == pytest.approx(0.20, abs=0.01)

    def test_cost_rigidity(self, analyzer, sample_data):
        """OpEx / (COGS+OpEx) = 200k/800k = 0.25."""
        result = analyzer.operational_risk_analysis(sample_data)
        assert result.cost_rigidity == pytest.approx(0.25, abs=0.01)

    def test_operating_leverage(self, analyzer, sample_data):
        """GP/OI = 400k/200k = 2.0."""
        result = analyzer.operational_risk_analysis(sample_data)
        assert result.operating_leverage == pytest.approx(2.0, abs=0.1)

    def test_risk_buffer(self, analyzer, sample_data):
        """(OCF+Cash)/OpEx = (220k+50k)/200k = 1.35."""
        result = analyzer.operational_risk_analysis(sample_data)
        assert result.risk_buffer == pytest.approx(1.35, abs=0.05)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operational_risk_analysis(sample_data)
        assert result.or_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operational_risk_analysis(sample_data)
        assert "Operational Risk" in result.summary


# ===== SCORING TESTS =====


class TestOperationalRiskScoring:
    def test_low_risk(self, analyzer):
        """MoS >= 0.30 => base 10."""
        data = FinancialData(
            revenue=2_000_000,
            cogs=800_000,
            operating_expenses=200_000,
            gross_profit=1_200_000,
            operating_income=1_000_000,
            operating_cash_flow=900_000,
            cash=200_000,
        )
        result = analyzer.operational_risk_analysis(data)
        # MoS=1-0.50=0.50 (base 10). RB=(900k+200k)/200k=5.5 (>=2.0 => +0.5). CR=200k/1M=0.20 (<=0.20 => +0.5) => 10 capped
        assert result.or_score >= 10.0
        assert result.or_grade == "Excellent"

    def test_moderate_risk(self, analyzer, sample_data):
        """MoS~0.20 => base 5.5 or 7.0 (floating point boundary)."""
        result = analyzer.operational_risk_analysis(sample_data)
        assert result.or_score >= 5.0

    def test_high_risk(self, analyzer):
        """MoS < 0.05 => base 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=700_000,
            operating_expenses=280_000,
            gross_profit=300_000,
            operating_income=20_000,
            operating_cash_flow=30_000,
            cash=10_000,
        )
        result = analyzer.operational_risk_analysis(data)
        # MoS=1-0.98=0.02 (base 1.0)
        assert result.or_score < 3.0
        assert result.or_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase92EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operational_risk_analysis(FinancialData())
        assert isinstance(result, OperationalRiskResult)
        assert result.margin_of_safety is None

    def test_no_revenue(self, analyzer):
        """Rev=0 => empty result."""
        data = FinancialData(cogs=500_000, operating_expenses=200_000)
        result = analyzer.operational_risk_analysis(data)
        assert result.or_score == 0.0

    def test_no_opex(self, analyzer):
        """OpEx=0 => risk_buffer is None, cost_rigidity=0."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            gross_profit=400_000,
            operating_income=400_000,
        )
        result = analyzer.operational_risk_analysis(data)
        assert result.cost_rigidity == pytest.approx(0.0, abs=0.01)

    def test_zero_operating_income(self, analyzer):
        """OI=0 => operating_leverage is None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            gross_profit=400_000,
            operating_expenses=400_000,
        )
        result = analyzer.operational_risk_analysis(data)
        assert result.operating_leverage is None
