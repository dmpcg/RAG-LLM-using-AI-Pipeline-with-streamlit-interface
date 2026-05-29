"""Phase 36 Tests: Interest Coverage & Debt Capacity.

Tests for interest_coverage_analysis() and InterestCoverageResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    InterestCoverageResult,
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
    )


# ===== DATACLASS TESTS =====


class TestInterestCoverageDataclass:
    def test_defaults(self):
        r = InterestCoverageResult()
        assert r.ebit_coverage is None
        assert r.ebitda_coverage is None
        assert r.debt_to_ebitda is None
        assert r.coverage_score == 0.0
        assert r.coverage_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = InterestCoverageResult(
            ebit_coverage=6.5,
            coverage_grade="Excellent",
        )
        assert r.ebit_coverage == 6.5
        assert r.coverage_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestInterestCoverage:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.interest_coverage_analysis(sample_data)
        assert isinstance(result, InterestCoverageResult)

    def test_ebit_coverage(self, analyzer, sample_data):
        """EBIT/Interest = 200k/30k = 6.67x."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.ebit_coverage == pytest.approx(6.667, rel=0.01)

    def test_ebitda_coverage(self, analyzer, sample_data):
        """EBITDA/Interest = 250k/30k = 8.33x."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.ebitda_coverage == pytest.approx(8.333, rel=0.01)

    def test_debt_to_ebitda(self, analyzer, sample_data):
        """Debt/EBITDA = 400k/250k = 1.6x."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.debt_to_ebitda == pytest.approx(1.6, rel=0.01)

    def test_ocf_to_debt(self, analyzer, sample_data):
        """OCF/Debt = 220k/400k = 0.55."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.ocf_to_debt == pytest.approx(0.55, rel=0.01)

    def test_fcf_to_debt(self, analyzer, sample_data):
        """FCF/Debt = (220k-80k)/400k = 0.35."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.fcf_to_debt == pytest.approx(0.35, rel=0.01)

    def test_interest_to_revenue(self, analyzer, sample_data):
        """Interest/Revenue = 30k/1M = 0.03."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.interest_to_revenue == pytest.approx(0.03, rel=0.01)

    def test_debt_to_equity(self, analyzer, sample_data):
        """Debt/Equity = 400k/1.2M = 0.333."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, rel=0.01)

    def test_max_debt_capacity(self, analyzer, sample_data):
        """Max capacity = 3 * EBITDA = 3 * 250k = 750k."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.max_debt_capacity == pytest.approx(750_000, rel=0.01)

    def test_spare_debt_capacity(self, analyzer, sample_data):
        """Spare = 750k - 400k = 350k."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.spare_debt_capacity == pytest.approx(350_000, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.coverage_grade in ["Excellent", "Adequate", "Strained", "Critical"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.interest_coverage_analysis(sample_data)
        assert "Interest" in result.summary or "coverage" in result.summary.lower()


# ===== SCORING TESTS =====


class TestInterestCoverageScoring:
    def test_excellent_coverage(self, analyzer):
        """Very high coverage ratios => Excellent."""
        data = FinancialData(
            ebit=500_000,
            ebitda=600_000,
            interest_expense=50_000,
            total_debt=300_000,
            operating_cash_flow=400_000,
            capex=50_000,
            revenue=2_000_000,
            total_equity=1_000_000,
        )
        result = analyzer.interest_coverage_analysis(data)
        # EBIT/Int=10.0 (≥5: +2.0), D/EBITDA=0.5 (≤1.5: +1.0), OCF/Debt=1.33 (≥0.30: +0.5)
        assert result.coverage_score >= 8.0
        assert result.coverage_grade == "Excellent"

    def test_critical_coverage(self, analyzer):
        """Very low coverage => Critical."""
        data = FinancialData(
            ebit=20_000,
            ebitda=30_000,
            interest_expense=50_000,
            total_debt=500_000,
            operating_cash_flow=10_000,
            capex=5_000,
            revenue=200_000,
            total_equity=100_000,
        )
        result = analyzer.interest_coverage_analysis(data)
        # EBIT/Int=0.4 (<1.0: -2.0), D/EBITDA=16.67 (>5.0: -1.0), OCF/Debt=0.02 (<0.10: -0.5)
        assert result.coverage_score < 4.0
        assert result.coverage_grade == "Critical"

    def test_adequate_coverage(self, analyzer):
        """Mid-range coverage => Adequate."""
        data = FinancialData(
            ebit=150_000,
            ebitda=200_000,
            interest_expense=30_000,
            total_debt=500_000,
            operating_cash_flow=100_000,
            capex=30_000,
            revenue=800_000,
            total_equity=600_000,
        )
        result = analyzer.interest_coverage_analysis(data)
        # EBIT/Int=5.0 (≥5.0: +2.0), D/EBITDA=2.5 (≤3.0: +0.5), OCF/Debt=0.20 (not ≥0.30 and not <0.10: no adj)
        assert result.coverage_score >= 6.0
        assert result.coverage_grade in ["Excellent", "Adequate"]


# ===== EDGE CASES =====


class TestPhase36EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.interest_coverage_analysis(FinancialData())
        assert isinstance(result, InterestCoverageResult)
        assert result.ebit_coverage is None
        assert result.ebitda_coverage is None

    def test_no_interest_expense(self, analyzer):
        """No interest => coverage ratios are None."""
        data = FinancialData(
            ebit=200_000,
            ebitda=250_000,
            total_debt=400_000,
        )
        result = analyzer.interest_coverage_analysis(data)
        assert result.ebit_coverage is None
        assert result.ebitda_coverage is None

    def test_no_debt(self, analyzer):
        """No debt => debt ratios are None."""
        data = FinancialData(
            ebit=200_000,
            ebitda=250_000,
            interest_expense=30_000,
        )
        result = analyzer.interest_coverage_analysis(data)
        assert result.debt_to_ebitda is None
        assert result.ocf_to_debt is None

    def test_no_capex(self, analyzer):
        """No capex => fcf_to_debt uses OCF only (capex=0)."""
        data = FinancialData(
            ebit=200_000,
            ebitda=250_000,
            interest_expense=30_000,
            total_debt=400_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.interest_coverage_analysis(data)
        # FCF = OCF - 0 = OCF; FCF/Debt = 220k/400k = 0.55
        if result.fcf_to_debt is not None:
            assert result.fcf_to_debt == pytest.approx(0.55, rel=0.01)

    def test_zero_interest(self, analyzer):
        """Zero interest expense => coverage None (division by zero)."""
        data = FinancialData(
            ebit=200_000,
            ebitda=250_000,
            interest_expense=0,
            total_debt=400_000,
        )
        result = analyzer.interest_coverage_analysis(data)
        assert result.ebit_coverage is None

    def test_sample_data_score(self, analyzer, sample_data):
        """EBIT_cov=6.67 (≥5.0:+2.0), D/EBITDA=1.6 (≤3.0:+0.5), OCF/Debt=0.55 (≥0.30:+0.5).
        Score = 5.0+2.0+0.5+0.5 = 8.0 => Excellent."""
        result = analyzer.interest_coverage_analysis(sample_data)
        assert result.coverage_score == pytest.approx(8.0, abs=0.1)
        assert result.coverage_grade == "Excellent"
