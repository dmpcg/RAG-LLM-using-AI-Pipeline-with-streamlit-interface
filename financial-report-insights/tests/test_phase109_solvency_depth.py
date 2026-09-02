"""Phase 109 Tests: Solvency Depth Analysis.

Tests for solvency_depth_analysis() and SolvencyDepthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    SolvencyDepthResult,
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


class TestSolvencyDepthDataclass:
    def test_defaults(self):
        r = SolvencyDepthResult()
        assert r.debt_to_equity is None
        assert r.debt_to_assets is None
        assert r.equity_to_assets is None
        assert r.interest_coverage_ratio is None
        assert r.debt_to_ebitda is None
        assert r.financial_leverage is None
        assert r.sd_score == 0.0
        assert r.sd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = SolvencyDepthResult(debt_to_ebitda=1.5, sd_grade="Good")
        assert r.debt_to_ebitda == 1.5
        assert r.sd_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestSolvencyDepthAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.solvency_depth_analysis(sample_data)
        assert isinstance(result, SolvencyDepthResult)

    def test_debt_to_equity(self, analyzer, sample_data):
        """D/E = 400k/1.2M = 0.333."""
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.01)

    def test_debt_to_assets(self, analyzer, sample_data):
        """D/A = 400k/2M = 0.20."""
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.debt_to_assets == pytest.approx(0.20, abs=0.01)

    def test_equity_to_assets(self, analyzer, sample_data):
        """E/A = 1.2M/2M = 0.60."""
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.equity_to_assets == pytest.approx(0.60, abs=0.01)

    def test_interest_coverage_ratio(self, analyzer, sample_data):
        """IC = 200k/30k = 6.667."""
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.interest_coverage_ratio == pytest.approx(6.667, abs=0.01)

    def test_debt_to_ebitda(self, analyzer, sample_data):
        """D/EBITDA = 400k/250k = 1.60."""
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.debt_to_ebitda == pytest.approx(1.60, abs=0.01)

    def test_financial_leverage(self, analyzer, sample_data):
        """FL = 2M/1.2M = 1.667."""
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.financial_leverage == pytest.approx(1.667, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.sd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.solvency_depth_analysis(sample_data)
        assert "Solvency Depth" in result.summary


# ===== SCORING TESTS =====


class TestSolvencyDepthScoring:
    def test_good_solvency(self, analyzer, sample_data):
        """D/EBITDA=1.60 => base 8.5. D/E=0.333 <=1.0 => +0.5. IC=6.667 >=5.0 => +0.5. Score=9.5."""
        result = analyzer.solvency_depth_analysis(sample_data)
        assert result.sd_score >= 9.5
        assert result.sd_grade == "Excellent"

    def test_excellent_solvency(self, analyzer):
        """D/EBITDA <= 1.0 => base 10."""
        data = FinancialData(
            total_debt=200_000,
            total_equity=2_000_000,
            total_assets=2_200_000,
            ebit=300_000,
            ebitda=400_000,
            interest_expense=20_000,
        )
        result = analyzer.solvency_depth_analysis(data)
        # D/EBITDA=0.5 => base 10. D/E=0.1 <=1.0 => +0.5. IC=15 >=5.0 => +0.5. Score=10(capped).
        assert result.sd_score >= 10.0
        assert result.sd_grade == "Excellent"

    def test_weak_solvency(self, analyzer):
        """D/EBITDA > 6.0 => base 1.0."""
        data = FinancialData(
            total_debt=1_000_000,
            total_equity=200_000,
            total_assets=1_500_000,
            ebit=50_000,
            ebitda=100_000,
            interest_expense=80_000,
        )
        result = analyzer.solvency_depth_analysis(data)
        # D/EBITDA=10.0 => base 1.0. D/E=5.0 >3.0 => -0.5. IC=0.625 <2.0 => -0.5. Score=0.0(clamped).
        assert result.sd_score <= 0.5
        assert result.sd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase109EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.solvency_depth_analysis(FinancialData())
        assert isinstance(result, SolvencyDepthResult)
        assert result.sd_score == 0.0

    def test_no_debt(self, analyzer):
        """TD=0 => D/E=0, D/EBITDA=0."""
        data = FinancialData(
            total_equity=1_000_000,
            total_assets=1_000_000,
            ebitda=200_000,
        )
        result = analyzer.solvency_depth_analysis(data)
        assert result.debt_to_ebitda == pytest.approx(0.0, abs=0.01)
        assert result.sd_score >= 10.0

    def test_no_interest_expense(self, analyzer):
        """IE=0 => IC=None."""
        data = FinancialData(
            total_debt=300_000,
            total_equity=700_000,
            total_assets=1_000_000,
            ebit=200_000,
            ebitda=250_000,
        )
        result = analyzer.solvency_depth_analysis(data)
        assert result.interest_coverage_ratio is None

    def test_no_ebitda(self, analyzer):
        """EBITDA=0 => D/EBITDA=None, fallback to D/A scoring."""
        data = FinancialData(
            total_debt=400_000,
            total_equity=600_000,
            total_assets=1_000_000,
        )
        result = analyzer.solvency_depth_analysis(data)
        assert result.debt_to_ebitda is None
        # D/A = 0.40, fallback => 5.0. D/E=0.667 <=1.0 => +0.5. Score=5.5
        assert result.sd_score >= 5.0
