"""Phase 197 Tests: Debt Management Analysis.

Tests for debt_management_analysis() and DebtManagementResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtManagementResult,
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


class TestDebtManagementDataclass:
    def test_defaults(self):
        r = DebtManagementResult()
        assert r.debt_to_operating_income is None
        assert r.debt_to_ocf is None
        assert r.interest_to_revenue is None
        assert r.debt_to_gross_profit is None
        assert r.net_debt_ratio is None
        assert r.debt_coverage_ratio is None
        assert r.dm_score == 0.0
        assert r.dm_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtManagementResult(debt_to_operating_income=2.0, dm_grade="Good")
        assert r.debt_to_operating_income == 2.0
        assert r.dm_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestDebtManagementAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_management_analysis(sample_data)
        assert isinstance(result, DebtManagementResult)

    def test_debt_to_operating_income(self, analyzer, sample_data):
        """TD/OI = 400k/200k = 2.00."""
        result = analyzer.debt_management_analysis(sample_data)
        assert result.debt_to_operating_income == pytest.approx(2.00, abs=0.01)

    def test_debt_to_ocf(self, analyzer, sample_data):
        """TD/OCF = 400k/220k = 1.818."""
        result = analyzer.debt_management_analysis(sample_data)
        assert result.debt_to_ocf == pytest.approx(1.818, abs=0.01)

    def test_interest_to_revenue(self, analyzer, sample_data):
        """IE/Rev = 30k/1M = 0.03."""
        result = analyzer.debt_management_analysis(sample_data)
        assert result.interest_to_revenue == pytest.approx(0.03, abs=0.005)

    def test_debt_to_gross_profit(self, analyzer, sample_data):
        """TD/GP = 400k/400k = 1.00."""
        result = analyzer.debt_management_analysis(sample_data)
        assert result.debt_to_gross_profit == pytest.approx(1.00, abs=0.01)

    def test_net_debt_ratio(self, analyzer, sample_data):
        """(TD-Cash)/TA = (400k-50k)/2M = 0.175."""
        result = analyzer.debt_management_analysis(sample_data)
        assert result.net_debt_ratio == pytest.approx(0.175, abs=0.005)

    def test_debt_coverage_ratio(self, analyzer, sample_data):
        """EBITDA/(IE+TD*0.1) = 250k/(30k+40k) = 250k/70k = 3.571."""
        result = analyzer.debt_management_analysis(sample_data)
        assert result.debt_coverage_ratio == pytest.approx(3.571, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_management_analysis(sample_data)
        assert result.dm_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_management_analysis(sample_data)
        assert "Debt Management" in result.summary


# ===== SCORING TESTS =====


class TestDebtManagementScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """DtOI=2.00 => base 8.5. ItR=0.03 <=0.03 => +0.5. DCR=3.571 >=3.0 => +0.5. Score=9.5."""
        result = analyzer.debt_management_analysis(sample_data)
        assert result.dm_score == pytest.approx(9.5, abs=0.5)
        assert result.dm_grade == "Excellent"

    def test_excellent_management(self, analyzer):
        """Very low debt-to-OI."""
        data = FinancialData(
            total_debt=100_000,
            operating_income=500_000,
            operating_cash_flow=600_000,
            interest_expense=5_000,
            revenue=2_000_000,
            gross_profit=800_000,
            cash=200_000,
            total_assets=3_000_000,
            ebitda=600_000,
        )
        result = analyzer.debt_management_analysis(data)
        assert result.dm_score >= 10.0
        assert result.dm_grade == "Excellent"

    def test_weak_management(self, analyzer):
        """Very high debt-to-OI."""
        data = FinancialData(
            total_debt=2_000_000,
            operating_income=200_000,
            operating_cash_flow=180_000,
            interest_expense=150_000,
            revenue=1_000_000,
            gross_profit=400_000,
            cash=20_000,
            total_assets=3_000_000,
            ebitda=250_000,
        )
        # DtOI=2M/200k=10.0 >7.0 => base 1.0. ItR=150k/1M=0.15 >0.10 => -0.5. DCR=250k/(150k+200k)=250k/350k=0.714 <1.5 => -0.5. Score=0.0.
        result = analyzer.debt_management_analysis(data)
        assert result.dm_score <= 0.5
        assert result.dm_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase197EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_management_analysis(FinancialData())
        assert isinstance(result, DebtManagementResult)
        assert result.dm_score == 0.0

    def test_no_debt(self, analyzer):
        """TD=None => DtOI=None, score 0."""
        data = FinancialData(
            operating_income=200_000,
            revenue=1_000_000,
        )
        result = analyzer.debt_management_analysis(data)
        assert result.debt_to_operating_income is None
        assert result.dm_score == 0.0

    def test_no_operating_income(self, analyzer):
        """OI=None => DtOI=None."""
        data = FinancialData(
            total_debt=400_000,
            revenue=1_000_000,
        )
        result = analyzer.debt_management_analysis(data)
        assert result.debt_to_operating_income is None
        assert result.dm_score == 0.0

    def test_no_interest_expense(self, analyzer):
        """IE=None => ItR=None, but DtOI still works."""
        data = FinancialData(
            total_debt=400_000,
            operating_income=200_000,
        )
        result = analyzer.debt_management_analysis(data)
        assert result.interest_to_revenue is None
        assert result.debt_to_operating_income is not None
