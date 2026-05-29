"""Phase 131 Tests: Funding Efficiency Analysis.

Tests for funding_efficiency_analysis() and FundingEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FundingEfficiencyResult,
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


class TestFundingEfficiencyDataclass:
    def test_defaults(self):
        r = FundingEfficiencyResult()
        assert r.debt_to_capitalization is None
        assert r.equity_multiplier is None
        assert r.interest_coverage_ebitda is None
        assert r.cost_of_debt is None
        assert r.weighted_funding_cost is None
        assert r.funding_spread is None
        assert r.fe_score == 0.0
        assert r.fe_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FundingEfficiencyResult(cost_of_debt=0.05, fe_grade="Good")
        assert r.cost_of_debt == 0.05
        assert r.fe_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestFundingEfficiencyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert isinstance(result, FundingEfficiencyResult)

    def test_debt_to_capitalization(self, analyzer, sample_data):
        """DC = 400k / (400k + 1.2M) = 400k / 1.6M = 0.25."""
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.debt_to_capitalization == pytest.approx(0.25, abs=0.01)

    def test_equity_multiplier(self, analyzer, sample_data):
        """EM = 2M / 1.2M = 1.667."""
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.equity_multiplier == pytest.approx(1.667, abs=0.01)

    def test_interest_coverage_ebitda(self, analyzer, sample_data):
        """IC = 250k / 30k = 8.333."""
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.interest_coverage_ebitda == pytest.approx(8.333, abs=0.01)

    def test_cost_of_debt(self, analyzer, sample_data):
        """CoD = 30k / 400k = 0.075."""
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.cost_of_debt == pytest.approx(0.075, abs=0.001)

    def test_weighted_funding_cost(self, analyzer, sample_data):
        """WFC = 0.075 * 0.25 = 0.01875."""
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.weighted_funding_cost == pytest.approx(0.01875, abs=0.001)

    def test_funding_spread(self, analyzer, sample_data):
        """FS = ROA - WFC = (150k/2M) - 0.01875 = 0.075 - 0.01875 = 0.05625."""
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.funding_spread == pytest.approx(0.05625, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.fe_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert "Funding Efficiency" in result.summary


# ===== SCORING TESTS =====


class TestFundingEfficiencyScoring:
    def test_strong_efficiency(self, analyzer, sample_data):
        """IC=8.333 => base 8.5. DC=0.25 <=0.30 => +0.5. FS=0.056 >=0.05 => +0.5. Score=9.5."""
        result = analyzer.funding_efficiency_analysis(sample_data)
        assert result.fe_score == pytest.approx(9.5, abs=0.5)
        assert result.fe_grade == "Excellent"

    def test_excellent_efficiency(self, analyzer):
        """IC >= 10.0 => base 10."""
        data = FinancialData(
            ebitda=500_000,
            interest_expense=20_000,
            total_debt=200_000,
            total_equity=1_500_000,
            total_assets=2_000_000,
            net_income=400_000,
        )
        # IC=25.0 => 10. DC=200k/1.7M=0.118 <=0.30 => +0.5. FS=ROA(0.20)-WFC(0.01)=0.19 >=0.05 => +0.5. Capped 10.
        result = analyzer.funding_efficiency_analysis(data)
        assert result.fe_score >= 10.0
        assert result.fe_grade == "Excellent"

    def test_weak_efficiency(self, analyzer):
        """IC < 1.0 => base 1.0."""
        data = FinancialData(
            ebitda=50_000,
            interest_expense=80_000,
            total_debt=800_000,
            total_equity=200_000,
            total_assets=1_000_000,
            net_income=10_000,
        )
        # IC=0.625 => 1.0. DC=800k/1M=0.80 >0.60 => -0.5. FS=0.01-0.08=-0.07 <0 => -0.5. Score=0.0 clamped.
        result = analyzer.funding_efficiency_analysis(data)
        assert result.fe_score <= 1.0
        assert result.fe_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase131EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.funding_efficiency_analysis(FinancialData())
        assert isinstance(result, FundingEfficiencyResult)
        assert result.fe_score == 0.0

    def test_no_ebitda(self, analyzer):
        """EBITDA=None => IC=None => score 0."""
        data = FinancialData(
            total_assets=1_000_000,
            total_equity=600_000,
            total_debt=300_000,
            interest_expense=20_000,
        )
        result = analyzer.funding_efficiency_analysis(data)
        assert result.interest_coverage_ebitda is None
        assert result.fe_score == 0.0

    def test_no_interest_expense(self, analyzer):
        """IE=None => IC=None, CoD=None."""
        data = FinancialData(
            ebitda=250_000,
            total_assets=1_000_000,
            total_equity=600_000,
            total_debt=300_000,
        )
        result = analyzer.funding_efficiency_analysis(data)
        assert result.interest_coverage_ebitda is None
        assert result.cost_of_debt is None

    def test_no_total_debt(self, analyzer):
        """TD=None => DC=None, CoD=None."""
        data = FinancialData(
            ebitda=250_000,
            interest_expense=20_000,
            total_assets=1_000_000,
            total_equity=600_000,
        )
        result = analyzer.funding_efficiency_analysis(data)
        assert result.debt_to_capitalization is None
        assert result.cost_of_debt is None
