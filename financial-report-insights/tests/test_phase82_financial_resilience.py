"""Phase 82 Tests: Financial Resilience Analysis.

Tests for financial_resilience_analysis() and FinancialResilienceResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FinancialResilienceResult,
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


class TestFinancialResilienceDataclass:
    def test_defaults(self):
        r = FinancialResilienceResult()
        assert r.cash_to_assets is None
        assert r.cash_to_debt is None
        assert r.operating_cash_coverage is None
        assert r.interest_coverage_cash is None
        assert r.free_cash_margin is None
        assert r.resilience_buffer is None
        assert r.fr_score == 0.0
        assert r.fr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FinancialResilienceResult(cash_to_assets=0.10, fr_grade="Good")
        assert r.cash_to_assets == 0.10
        assert r.fr_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestFinancialResilienceAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.financial_resilience_analysis(sample_data)
        assert isinstance(result, FinancialResilienceResult)

    def test_cash_to_assets(self, analyzer, sample_data):
        """Cash/TA = 50k/2M = 0.025."""
        result = analyzer.financial_resilience_analysis(sample_data)
        assert result.cash_to_assets == pytest.approx(0.025, abs=0.001)

    def test_cash_to_debt(self, analyzer, sample_data):
        """Cash/TD = 50k/400k = 0.125."""
        result = analyzer.financial_resilience_analysis(sample_data)
        assert result.cash_to_debt == pytest.approx(0.125, abs=0.01)

    def test_operating_cash_coverage(self, analyzer, sample_data):
        """OCF/TL = 220k/800k = 0.275."""
        result = analyzer.financial_resilience_analysis(sample_data)
        assert result.operating_cash_coverage == pytest.approx(0.275, abs=0.01)

    def test_interest_coverage_cash(self, analyzer, sample_data):
        """OCF/IE = 220k/30k = 7.333."""
        result = analyzer.financial_resilience_analysis(sample_data)
        assert result.interest_coverage_cash == pytest.approx(7.333, abs=0.1)

    def test_free_cash_margin(self, analyzer, sample_data):
        """FCF/Rev = (220k-80k)/1M = 0.14."""
        result = analyzer.financial_resilience_analysis(sample_data)
        assert result.free_cash_margin == pytest.approx(0.14, abs=0.01)

    def test_resilience_buffer(self, analyzer, sample_data):
        """(Cash+OCF)/AnnualOpEx = (50k+220k)/(600k+200k) = 270k/800k = 0.3375."""
        result = analyzer.financial_resilience_analysis(sample_data)
        assert result.resilience_buffer == pytest.approx(0.3375, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.financial_resilience_analysis(sample_data)
        assert result.fr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.financial_resilience_analysis(sample_data)
        assert "Resilience" in result.summary


# ===== SCORING TESTS =====


class TestFinancialResilienceScoring:
    def test_very_resilient(self, analyzer):
        """RB >= 1.0 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=400_000,
            operating_expenses=200_000,
            total_assets=2_000_000,
            total_liabilities=300_000,
            total_debt=200_000,
            cash=500_000,
            operating_cash_flow=600_000,
            capex=50_000,
            interest_expense=10_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        # RB=(500k+600k)/600k=1.833 (base 10). CD=500k/200k=2.5 (>=1.0 => +0.5). ICC=60 (>=5 => +0.5). => 10 capped
        assert result.fr_score >= 10.0
        assert result.fr_grade == "Excellent"

    def test_moderate_resilience(self, analyzer, sample_data):
        """RB ~ 0.34 => base 4.0 (falls in [0.20, 0.35) bracket)."""
        result = analyzer.financial_resilience_analysis(sample_data)
        # RB=0.3375 (base 4.0). CD=0.125 (no adj). ICC=7.33 (>=5 => +0.5). => 4.5
        assert result.fr_score >= 4.0

    def test_weak_resilience(self, analyzer):
        """RB < 0.10 => base 1.0."""
        data = FinancialData(
            revenue=2_000_000,
            cogs=1_500_000,
            operating_expenses=400_000,
            total_assets=3_000_000,
            total_liabilities=2_500_000,
            total_debt=2_000_000,
            cash=20_000,
            operating_cash_flow=50_000,
            capex=100_000,
            interest_expense=200_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        # RB=(20k+50k)/1.9M=0.037 (base 1.0). CD=0.01 (<0.10 => -0.5). ICC=0.25 (<1.5 => -0.5). => 0.0
        assert result.fr_score < 2.0
        assert result.fr_grade == "Weak"

    def test_cash_to_debt_bonus(self, analyzer):
        """CD >= 1.0 => +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=400_000,
            operating_expenses=200_000,
            total_assets=1_000_000,
            total_liabilities=300_000,
            total_debt=100_000,
            cash=200_000,
            operating_cash_flow=300_000,
            capex=50_000,
            interest_expense=5_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        # RB=(200k+300k)/600k=0.833 (base 8.5). CD=2.0 (+0.5). ICC=60 (+0.5). => 9.5
        assert result.fr_score >= 9.0

    def test_interest_coverage_penalty(self, analyzer):
        """ICC < 1.5 => -0.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=500_000,
            operating_expenses=300_000,
            total_assets=1_500_000,
            total_liabilities=800_000,
            total_debt=500_000,
            cash=50_000,
            operating_cash_flow=100_000,
            capex=30_000,
            interest_expense=80_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        # RB=(50k+100k)/800k=0.1875 (base 2.5). ICC=1.25 (<1.5 => -0.5). => 2.0
        assert result.fr_score <= 3.0


# ===== EDGE CASES =====


class TestPhase82EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.financial_resilience_analysis(FinancialData())
        assert isinstance(result, FinancialResilienceResult)
        assert result.cash_to_assets is None

    def test_no_total_assets(self, analyzer):
        """TA=0 => empty result."""
        data = FinancialData(
            cash=100_000,
            total_debt=50_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        assert result.fr_score == 0.0

    def test_no_debt(self, analyzer):
        """No debt => cash_to_debt is None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=500_000,
            operating_expenses=200_000,
            total_assets=1_000_000,
            total_liabilities=200_000,
            cash=200_000,
            operating_cash_flow=300_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        assert result.cash_to_debt is None

    def test_no_interest_expense(self, analyzer):
        """No IE => interest_coverage_cash is None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=500_000,
            operating_expenses=200_000,
            total_assets=1_000_000,
            total_liabilities=200_000,
            cash=100_000,
            operating_cash_flow=200_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        assert result.interest_coverage_cash is None

    def test_no_opex_fallback(self, analyzer):
        """No COGS/OpEx => resilience_buffer is None, fallback to cash_to_assets scoring."""
        data = FinancialData(
            total_assets=1_000_000,
            cash=200_000,
        )
        result = analyzer.financial_resilience_analysis(data)
        assert result.resilience_buffer is None
        # Fallback: cash_to_assets = 0.20, >= 0.15 => base 5.5
        assert result.fr_score >= 5.0
