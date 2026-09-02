"""Phase 90 Tests: Financial Health Score Analysis.

Tests for financial_health_score_analysis() and FinancialHealthScoreResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FinancialHealthScoreResult,
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


class TestFinancialHealthScoreDataclass:
    def test_defaults(self):
        r = FinancialHealthScoreResult()
        assert r.profitability_score is None
        assert r.liquidity_score is None
        assert r.solvency_score is None
        assert r.efficiency_score is None
        assert r.coverage_score is None
        assert r.composite_score is None
        assert r.fh_score == 0.0
        assert r.fh_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FinancialHealthScoreResult(composite_score=7.5, fh_grade="Good")
        assert r.composite_score == 7.5
        assert r.fh_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestFinancialHealthScoreAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.financial_health_score_analysis(sample_data)
        assert isinstance(result, FinancialHealthScoreResult)

    def test_profitability_score(self, analyzer, sample_data):
        """NM=0.15 => profitability=10."""
        result = analyzer.financial_health_score_analysis(sample_data)
        assert result.profitability_score == 10.0

    def test_liquidity_score(self, analyzer, sample_data):
        """CR=500k/200k=2.5 => liquidity=10."""
        result = analyzer.financial_health_score_analysis(sample_data)
        assert result.liquidity_score == 10.0

    def test_solvency_score(self, analyzer, sample_data):
        """D/E=400k/1.2M=0.333 => solvency=8."""
        result = analyzer.financial_health_score_analysis(sample_data)
        assert result.solvency_score == 8.0

    def test_efficiency_score(self, analyzer, sample_data):
        """AT=1M/2M=0.5 => efficiency=6."""
        result = analyzer.financial_health_score_analysis(sample_data)
        assert result.efficiency_score == 6.0

    def test_coverage_score(self, analyzer, sample_data):
        """IC=200k/30k=6.67 => coverage=8."""
        result = analyzer.financial_health_score_analysis(sample_data)
        assert result.coverage_score == 8.0

    def test_composite_score(self, analyzer, sample_data):
        """Weighted average of all 5 pillars."""
        result = analyzer.financial_health_score_analysis(sample_data)
        # P=10*0.25 + L=10*0.20 + S=8*0.20 + E=6*0.15 + C=8*0.20 = 2.5+2.0+1.6+0.9+1.6 = 8.6
        assert result.composite_score == pytest.approx(8.6, abs=0.1)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.financial_health_score_analysis(sample_data)
        assert result.fh_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.financial_health_score_analysis(sample_data)
        assert "Financial Health" in result.summary


# ===== SCORING TESTS =====


class TestFinancialHealthScoring:
    def test_excellent_health(self, analyzer):
        """All pillars max."""
        data = FinancialData(
            revenue=5_000_000,
            net_income=1_000_000,
            ebit=1_200_000,
            total_assets=3_000_000,
            total_equity=2_500_000,
            total_debt=300_000,
            current_assets=2_000_000,
            current_liabilities=500_000,
            interest_expense=100_000,
        )
        result = analyzer.financial_health_score_analysis(data)
        assert result.fh_score >= 9.0
        assert result.fh_grade == "Excellent"

    def test_weak_health(self, analyzer):
        """All pillars low."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=-100_000,
            ebit=-50_000,
            total_assets=5_000_000,
            total_equity=500_000,
            total_debt=2_000_000,
            current_assets=200_000,
            current_liabilities=600_000,
            interest_expense=200_000,
        )
        result = analyzer.financial_health_score_analysis(data)
        assert result.fh_score < 4.0
        assert result.fh_grade == "Weak"

    def test_partial_pillars(self, analyzer):
        """Only some pillars available."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
        )
        result = analyzer.financial_health_score_analysis(data)
        # Only profitability + efficiency available
        assert result.profitability_score is not None
        assert result.efficiency_score is not None
        assert result.liquidity_score is None
        assert result.composite_score is not None


# ===== EDGE CASES =====


class TestPhase90EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.financial_health_score_analysis(FinancialData())
        assert isinstance(result, FinancialHealthScoreResult)
        assert result.composite_score is None

    def test_only_revenue(self, analyzer):
        """Only rev => profitability pillar."""
        data = FinancialData(revenue=1_000_000)
        result = analyzer.financial_health_score_analysis(data)
        assert result.profitability_score is not None
        assert result.liquidity_score is None

    def test_zero_interest(self, analyzer):
        """IE=0 => coverage_score is None."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=150_000,
            ebit=200_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
            total_debt=400_000,
            current_assets=500_000,
            current_liabilities=200_000,
        )
        result = analyzer.financial_health_score_analysis(data)
        assert result.coverage_score is None

    def test_zero_equity(self, analyzer):
        """TE=0 => solvency_score is None."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
            current_assets=500_000,
            current_liabilities=200_000,
        )
        result = analyzer.financial_health_score_analysis(data)
        assert result.solvency_score is None
