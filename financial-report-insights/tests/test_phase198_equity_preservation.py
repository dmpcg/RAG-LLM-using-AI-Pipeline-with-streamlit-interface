"""Phase 198 Tests: Equity Preservation Analysis.

Tests for equity_preservation_analysis() and EquityPreservationResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    EquityPreservationResult,
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


class TestEquityPreservationDataclass:
    def test_defaults(self):
        r = EquityPreservationResult()
        assert r.equity_to_assets is None
        assert r.retained_to_equity is None
        assert r.equity_growth_capacity is None
        assert r.equity_to_liabilities is None
        assert r.tangible_equity_ratio is None
        assert r.equity_per_revenue is None
        assert r.ep_score == 0.0
        assert r.ep_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = EquityPreservationResult(equity_to_assets=0.60, ep_grade="Excellent")
        assert r.equity_to_assets == 0.60
        assert r.ep_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestEquityPreservationAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.equity_preservation_analysis(sample_data)
        assert isinstance(result, EquityPreservationResult)

    def test_equity_to_assets(self, analyzer, sample_data):
        """TE/TA = 1.2M/2M = 0.60."""
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.equity_to_assets == pytest.approx(0.60, abs=0.005)

    def test_retained_to_equity(self, analyzer, sample_data):
        """RE/TE = 600k/1.2M = 0.50."""
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.retained_to_equity == pytest.approx(0.50, abs=0.005)

    def test_equity_growth_capacity(self, analyzer, sample_data):
        """NI/TE = 150k/1.2M = 0.125."""
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.equity_growth_capacity == pytest.approx(0.125, abs=0.005)

    def test_equity_to_liabilities(self, analyzer, sample_data):
        """TE/TL = 1.2M/800k = 1.50."""
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.equity_to_liabilities == pytest.approx(1.50, abs=0.01)

    def test_tangible_equity_ratio(self, analyzer, sample_data):
        """TE/TA = 1.2M/2M = 0.60."""
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.tangible_equity_ratio == pytest.approx(0.60, abs=0.005)

    def test_equity_per_revenue(self, analyzer, sample_data):
        """TE/Rev = 1.2M/1M = 1.20."""
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.equity_per_revenue == pytest.approx(1.20, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.ep_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.equity_preservation_analysis(sample_data)
        assert "Equity Preservation" in result.summary


# ===== SCORING TESTS =====


class TestEquityPreservationScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """EtA=0.60 => base 10.0. RtE=0.50 no adj (<0.60). EGC=0.125 no adj (<0.15). Score=10.0."""
        result = analyzer.equity_preservation_analysis(sample_data)
        assert result.ep_score == pytest.approx(10.0, abs=0.5)
        assert result.ep_grade == "Excellent"

    def test_excellent_preservation(self, analyzer):
        """Very high equity-to-assets."""
        data = FinancialData(
            total_equity=2_000_000,
            total_assets=2_500_000,
            retained_earnings=1_500_000,
            net_income=400_000,
            total_liabilities=500_000,
            revenue=1_500_000,
        )
        result = analyzer.equity_preservation_analysis(data)
        assert result.ep_score >= 10.0
        assert result.ep_grade == "Excellent"

    def test_weak_preservation(self, analyzer):
        """Very low equity-to-assets."""
        data = FinancialData(
            total_equity=100_000,
            total_assets=2_000_000,
            retained_earnings=10_000,
            net_income=3_000,
            total_liabilities=1_900_000,
            revenue=1_000_000,
        )
        # EtA=100k/2M=0.05 <0.10 => base 1.0. RtE=10k/100k=0.10 <0.20 => -0.5. EGC=3k/100k=0.03 <0.05 => -0.5. Score=0.0.
        result = analyzer.equity_preservation_analysis(data)
        assert result.ep_score <= 0.5
        assert result.ep_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase198EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.equity_preservation_analysis(FinancialData())
        assert isinstance(result, EquityPreservationResult)
        assert result.ep_score == 0.0

    def test_no_equity(self, analyzer):
        """TE=None => EtA=None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            net_income=150_000,
        )
        result = analyzer.equity_preservation_analysis(data)
        assert result.equity_to_assets is None
        assert result.ep_score == 0.0

    def test_no_assets(self, analyzer):
        """TA=None => EtA=None."""
        data = FinancialData(
            total_equity=1_200_000,
            net_income=150_000,
        )
        result = analyzer.equity_preservation_analysis(data)
        assert result.equity_to_assets is None
        assert result.ep_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => EGC=None, but EtA still works."""
        data = FinancialData(
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        result = analyzer.equity_preservation_analysis(data)
        assert result.equity_growth_capacity is None
        assert result.equity_to_assets is not None
