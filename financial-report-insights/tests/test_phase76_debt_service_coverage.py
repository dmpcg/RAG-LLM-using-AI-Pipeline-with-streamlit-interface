"""Phase 76 Tests: Debt Service Coverage Analysis.

Tests for debt_service_coverage_analysis() and DebtServiceCoverageResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtServiceCoverageResult,
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


class TestDebtServiceCoverageDataclass:
    def test_defaults(self):
        r = DebtServiceCoverageResult()
        assert r.dscr is None
        assert r.ocf_to_debt_service is None
        assert r.ebitda_to_interest is None
        assert r.fcf_to_debt_service is None
        assert r.debt_service_to_revenue is None
        assert r.coverage_cushion is None
        assert r.dsc_score == 0.0
        assert r.dsc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtServiceCoverageResult(dscr=2.5, dsc_grade="Good")
        assert r.dscr == 2.5
        assert r.dsc_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestDebtServiceCoverageAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert isinstance(result, DebtServiceCoverageResult)

    def test_dscr(self, analyzer, sample_data):
        """EBITDA/IE = 250k/30k = 8.33."""
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert result.dscr == pytest.approx(8.33, abs=0.1)

    def test_ocf_to_debt_service(self, analyzer, sample_data):
        """OCF/IE = 220k/30k = 7.33."""
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert result.ocf_to_debt_service == pytest.approx(7.33, abs=0.1)

    def test_ebitda_to_interest(self, analyzer, sample_data):
        """EBITDA/IE = 250k/30k = 8.33."""
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert result.ebitda_to_interest == pytest.approx(8.33, abs=0.1)

    def test_fcf_to_debt_service(self, analyzer, sample_data):
        """FCF/IE = (220k-80k)/30k = 4.67."""
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert result.fcf_to_debt_service == pytest.approx(4.67, abs=0.1)

    def test_debt_service_to_revenue(self, analyzer, sample_data):
        """IE/Rev = 30k/1M = 0.03."""
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert result.debt_service_to_revenue == pytest.approx(0.03, abs=0.01)

    def test_coverage_cushion(self, analyzer, sample_data):
        """DSCR - 1.0 = 8.33 - 1.0 = 7.33."""
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert result.coverage_cushion == pytest.approx(7.33, abs=0.1)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert result.dsc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_service_coverage_analysis(sample_data)
        assert "Debt Service Coverage" in result.summary


# ===== SCORING TESTS =====


class TestDebtServiceCoverageScoring:
    def test_very_high_dscr(self, analyzer, sample_data):
        """DSCR >= 4.0 => base 10."""
        result = analyzer.debt_service_coverage_analysis(sample_data)
        # DSCR=8.33 (base 10) + OCF/DS=7.33 (+0.5) + DS/Rev=0.03 (no adj) => 10 capped
        assert result.dsc_score >= 9.0
        assert result.dsc_grade == "Excellent"

    def test_moderate_dscr(self, analyzer):
        """DSCR ~ 2.5 => base 7.0."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=250_000,
            operating_cash_flow=200_000,
            capex=50_000,
            interest_expense=100_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        # DSCR=2.5 (base 7.0) + OCF/DS=2.0 (no adj) + DS/Rev=0.10 (-0.5) => 6.5
        assert result.dsc_score >= 6.0

    def test_low_dscr(self, analyzer):
        """DSCR ~ 1.5 => base 4.0."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=150_000,
            operating_cash_flow=120_000,
            capex=30_000,
            interest_expense=100_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        # DSCR=1.5 (base 4.0) + OCF/DS=1.2 (no adj) + DS/Rev=0.10 (-0.5) => 3.5
        assert result.dsc_score <= 5.0

    def test_below_one_dscr(self, analyzer):
        """DSCR < 1.0 => base 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=80_000,
            operating_cash_flow=70_000,
            capex=20_000,
            interest_expense=100_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        assert result.dsc_score < 3.0
        assert result.dsc_grade == "Weak"

    def test_ocf_coverage_bonus(self, analyzer):
        """OCF/DS >= 5.0 => +0.5."""
        data = FinancialData(
            revenue=5_000_000,
            ebitda=1_500_000,
            operating_cash_flow=1_200_000,
            capex=200_000,
            interest_expense=200_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        # DSCR=7.5 (base 10) + OCF/DS=6.0 (+0.5) + DS/Rev=0.04 (no adj) => 10 capped
        assert result.dsc_score >= 10.0

    def test_low_debt_service_burden_bonus(self, analyzer):
        """DS/Rev <= 0.02 => +0.5."""
        data = FinancialData(
            revenue=5_000_000,
            ebitda=1_000_000,
            operating_cash_flow=800_000,
            capex=100_000,
            interest_expense=80_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        # DSCR=12.5 (base 10) + OCF/DS=10.0 (+0.5) + DS/Rev=0.016 (+0.5) => 10 capped
        assert result.dsc_score >= 10.0


# ===== EDGE CASES =====


class TestPhase76EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_service_coverage_analysis(FinancialData())
        assert isinstance(result, DebtServiceCoverageResult)
        assert result.dscr is None

    def test_no_interest_expense(self, analyzer):
        """No IE => empty result."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=250_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        assert result.dsc_score == 0.0

    def test_no_ebitda(self, analyzer):
        """EBITDA=0 => DSCR=0."""
        data = FinancialData(
            revenue=1_000_000,
            interest_expense=50_000,
            operating_cash_flow=100_000,
            capex=30_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        # EBITDA=0, DSCR=safe_divide(0, 50k)=0.0
        assert result.dscr == pytest.approx(0.0, abs=0.01)

    def test_no_revenue(self, analyzer):
        """No revenue => DS/Rev is None."""
        data = FinancialData(
            ebitda=200_000,
            operating_cash_flow=150_000,
            interest_expense=50_000,
        )
        result = analyzer.debt_service_coverage_analysis(data)
        assert result.debt_service_to_revenue is None  # safe_divide(50k, 0) => None
