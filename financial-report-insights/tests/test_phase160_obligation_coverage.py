"""Phase 160 Tests: Obligation Coverage Analysis.

Tests for obligation_coverage_analysis() and ObligationCoverageResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ObligationCoverageResult,
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


class TestObligationCoverageDataclass:
    def test_defaults(self):
        r = ObligationCoverageResult()
        assert r.ebitda_interest_coverage is None
        assert r.cash_interest_coverage is None
        assert r.debt_amortization_capacity is None
        assert r.fixed_charge_coverage is None
        assert r.debt_burden_ratio is None
        assert r.interest_to_revenue is None
        assert r.oc_score == 0.0
        assert r.oc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ObligationCoverageResult(ebitda_interest_coverage=8.0, oc_grade="Excellent")
        assert r.ebitda_interest_coverage == 8.0
        assert r.oc_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestObligationCoverageAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert isinstance(result, ObligationCoverageResult)

    def test_ebitda_interest_coverage(self, analyzer, sample_data):
        """EBITDA/IE = 250k/30k = 8.333."""
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.ebitda_interest_coverage == pytest.approx(8.333, abs=0.01)

    def test_cash_interest_coverage(self, analyzer, sample_data):
        """OCF/IE = 220k/30k = 7.333."""
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.cash_interest_coverage == pytest.approx(7.333, abs=0.01)

    def test_debt_amortization_capacity(self, analyzer, sample_data):
        """(OCF-CapEx)/TD = (220k-80k)/400k = 140k/400k = 0.35."""
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.debt_amortization_capacity == pytest.approx(0.35, abs=0.01)

    def test_fixed_charge_coverage(self, analyzer, sample_data):
        """EBITDA/(IE+CapEx) = 250k/(30k+80k) = 250k/110k = 2.273."""
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.fixed_charge_coverage == pytest.approx(2.273, abs=0.01)

    def test_debt_burden_ratio(self, analyzer, sample_data):
        """TD/EBITDA = 400k/250k = 1.60."""
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.debt_burden_ratio == pytest.approx(1.60, abs=0.01)

    def test_interest_to_revenue(self, analyzer, sample_data):
        """IE/Rev = 30k/1M = 0.03."""
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.interest_to_revenue == pytest.approx(0.03, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.oc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert "Obligation Coverage" in result.summary


# ===== SCORING TESTS =====


class TestObligationCoverageScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """EIC=8.333 => base 8.5. DBR=1.60 <=2.0 => +0.5. CIC=7.333 <8.0 => no adj. Score=9.0."""
        result = analyzer.obligation_coverage_analysis(sample_data)
        assert result.oc_score == pytest.approx(9.0, abs=0.5)
        assert result.oc_grade == "Excellent"

    def test_excellent_coverage(self, analyzer):
        """EIC >= 10.0 => base 10."""
        data = FinancialData(
            ebitda=500_000,
            interest_expense=30_000,
            operating_cash_flow=400_000,
            capex=100_000,
            total_debt=300_000,
            revenue=2_000_000,
        )
        result = analyzer.obligation_coverage_analysis(data)
        assert result.oc_score >= 10.0
        assert result.oc_grade == "Excellent"

    def test_weak_coverage(self, analyzer):
        """EIC < 1.0 => base 1.0."""
        data = FinancialData(
            ebitda=20_000,
            interest_expense=30_000,
            operating_cash_flow=15_000,
            capex=10_000,
            total_debt=500_000,
            revenue=200_000,
        )
        # EIC=20k/30k=0.667 => 1.0. DBR=500k/20k=25.0 >=5.0 => -0.5. CIC=15k/30k=0.5 <1.5 => -0.5. Score=0.0.
        result = analyzer.obligation_coverage_analysis(data)
        assert result.oc_score <= 1.0
        assert result.oc_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase160EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.obligation_coverage_analysis(FinancialData())
        assert isinstance(result, ObligationCoverageResult)
        assert result.oc_score == 0.0

    def test_no_interest_expense(self, analyzer):
        """IE=None => EIC=None, score 0."""
        data = FinancialData(
            ebitda=250_000,
            operating_cash_flow=220_000,
            total_debt=400_000,
        )
        result = analyzer.obligation_coverage_analysis(data)
        assert result.ebitda_interest_coverage is None
        assert result.oc_score == 0.0

    def test_no_debt(self, analyzer):
        """TD=None => DBR=None, DAC=None."""
        data = FinancialData(
            ebitda=250_000,
            interest_expense=30_000,
            operating_cash_flow=220_000,
            capex=80_000,
            revenue=1_000_000,
        )
        result = analyzer.obligation_coverage_analysis(data)
        assert result.debt_burden_ratio is None
        assert result.debt_amortization_capacity is None
        assert result.ebitda_interest_coverage is not None

    def test_no_ebitda(self, analyzer):
        """EBITDA=None => EIC=None, FCC=None."""
        data = FinancialData(
            interest_expense=30_000,
            operating_cash_flow=220_000,
            capex=80_000,
            total_debt=400_000,
            revenue=1_000_000,
        )
        result = analyzer.obligation_coverage_analysis(data)
        assert result.ebitda_interest_coverage is None
        assert result.fixed_charge_coverage is None
        assert result.oc_score == 0.0
