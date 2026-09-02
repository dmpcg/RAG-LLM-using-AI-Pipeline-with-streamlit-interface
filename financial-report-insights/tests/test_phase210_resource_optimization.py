"""Phase 210 Tests: Resource Optimization Analysis.

Tests for resource_optimization_analysis() and ResourceOptimizationResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ResourceOptimizationResult,
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


class TestResourceOptimizationDataclass:
    def test_defaults(self):
        r = ResourceOptimizationResult()
        assert r.fcf_to_revenue is None
        assert r.ocf_to_revenue is None
        assert r.capex_to_revenue is None
        assert r.ocf_to_assets is None
        assert r.fcf_to_assets is None
        assert r.dividend_payout_ratio is None
        assert r.ro_score == 0.0
        assert r.ro_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ResourceOptimizationResult(fcf_to_revenue=0.14, ro_grade="Good")
        assert r.fcf_to_revenue == 0.14
        assert r.ro_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestResourceOptimizationAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.resource_optimization_analysis(sample_data)
        assert isinstance(result, ResourceOptimizationResult)

    def test_fcf_to_revenue(self, analyzer, sample_data):
        """FCF=OCF-CapEx=220k-80k=140k. FCF/Rev=140k/1M=0.14."""
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.fcf_to_revenue == pytest.approx(0.14, abs=0.005)

    def test_ocf_to_revenue(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.ocf_to_revenue == pytest.approx(0.22, abs=0.005)

    def test_capex_to_revenue(self, analyzer, sample_data):
        """CapEx/Rev = 80k/1M = 0.08."""
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.capex_to_revenue == pytest.approx(0.08, abs=0.005)

    def test_ocf_to_assets(self, analyzer, sample_data):
        """OCF/TA = 220k/2M = 0.11."""
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.ocf_to_assets == pytest.approx(0.11, abs=0.005)

    def test_fcf_to_assets(self, analyzer, sample_data):
        """FCF/TA = 140k/2M = 0.07."""
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.fcf_to_assets == pytest.approx(0.07, abs=0.005)

    def test_dividend_payout_ratio(self, analyzer, sample_data):
        """Div/NI = 40k/150k = 0.267."""
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.dividend_payout_ratio == pytest.approx(0.267, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.ro_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.resource_optimization_analysis(sample_data)
        assert "Resource Optimization" in result.summary


# ===== SCORING TESTS =====


class TestResourceOptimizationScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """FCF/Rev=0.14 >=0.12 => base 7.0. OCF/Rev=0.22 >=0.20 => +0.5. CapEx/Rev=0.08 <=0.05? No. >0.20? No => no adj. Score=7.5."""
        result = analyzer.resource_optimization_analysis(sample_data)
        assert result.ro_score == pytest.approx(7.5, abs=0.5)
        assert result.ro_grade in ["Good", "Excellent"]

    def test_excellent_optimization(self, analyzer):
        """Very high FCF/Revenue."""
        data = FinancialData(
            revenue=2_000_000,
            operating_cash_flow=800_000,
            capex=50_000,
            total_assets=3_000_000,
            net_income=500_000,
        )
        result = analyzer.resource_optimization_analysis(data)
        assert result.ro_score >= 10.0
        assert result.ro_grade == "Excellent"

    def test_weak_optimization(self, analyzer):
        """Very low FCF/Revenue."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=30_000,
            capex=25_000,
            total_assets=2_000_000,
            net_income=10_000,
        )
        # FCF=30k-25k=5k. FCF/Rev=0.005 <0.02 => base 1.0.
        # OCF/Rev=0.03 <0.08 => -0.5. CapEx/Rev=0.025 <=0.05 => +0.5. Score=1.0.
        result = analyzer.resource_optimization_analysis(data)
        assert result.ro_score <= 1.5
        assert result.ro_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase210EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.resource_optimization_analysis(FinancialData())
        assert isinstance(result, ResourceOptimizationResult)
        assert result.ro_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => FCF=None => score 0."""
        data = FinancialData(
            revenue=1_000_000,
            capex=80_000,
            total_assets=2_000_000,
        )
        result = analyzer.resource_optimization_analysis(data)
        assert result.fcf_to_revenue is None
        assert result.ro_score == 0.0

    def test_no_capex(self, analyzer):
        """CapEx=None => FCF=None => score 0."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=220_000,
            total_assets=2_000_000,
        )
        result = analyzer.resource_optimization_analysis(data)
        assert result.fcf_to_revenue is None
        assert result.ro_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => FCF/Rev=None."""
        data = FinancialData(
            operating_cash_flow=220_000,
            capex=80_000,
            total_assets=2_000_000,
        )
        result = analyzer.resource_optimization_analysis(data)
        assert result.fcf_to_revenue is None
        assert result.ro_score == 0.0

    def test_zero_capex_value(self, analyzer):
        """CapEx=0 => FCF=OCF. FCF/Rev calculated."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=220_000,
            capex=0,
            total_assets=2_000_000,
        )
        result = analyzer.resource_optimization_analysis(data)
        assert result.fcf_to_revenue == pytest.approx(0.22, abs=0.005)
        assert result.ro_score > 0.0
