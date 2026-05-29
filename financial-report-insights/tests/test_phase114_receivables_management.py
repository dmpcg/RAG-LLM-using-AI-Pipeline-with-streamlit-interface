"""Phase 114 Tests: Receivables Management Analysis.

Tests for receivables_management_analysis() and ReceivablesManagementResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ReceivablesManagementResult,
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


class TestReceivablesManagementDataclass:
    def test_defaults(self):
        r = ReceivablesManagementResult()
        assert r.dso is None
        assert r.ar_to_revenue is None
        assert r.ar_to_current_assets is None
        assert r.receivables_turnover is None
        assert r.collection_effectiveness is None
        assert r.ar_concentration is None
        assert r.rm_score == 0.0
        assert r.rm_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ReceivablesManagementResult(dso=30.0, rm_grade="Excellent")
        assert r.dso == 30.0
        assert r.rm_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestReceivablesManagementAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.receivables_management_analysis(sample_data)
        assert isinstance(result, ReceivablesManagementResult)

    def test_dso(self, analyzer, sample_data):
        """DSO = 150k/1M * 365 = 54.75 days."""
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.dso == pytest.approx(54.75, abs=0.5)

    def test_ar_to_revenue(self, analyzer, sample_data):
        """AR/Rev = 150k/1M = 0.15."""
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.ar_to_revenue == pytest.approx(0.15, abs=0.005)

    def test_ar_to_current_assets(self, analyzer, sample_data):
        """AR/CA = 150k/500k = 0.30."""
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.ar_to_current_assets == pytest.approx(0.30, abs=0.01)

    def test_receivables_turnover(self, analyzer, sample_data):
        """RT = 1M/150k = 6.67."""
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.receivables_turnover == pytest.approx(6.67, abs=0.1)

    def test_collection_effectiveness(self, analyzer, sample_data):
        """CE = (1M - 150k)/1M = 0.85."""
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.collection_effectiveness == pytest.approx(0.85, abs=0.01)

    def test_ar_concentration(self, analyzer, sample_data):
        """ARC = 150k/(150k+50k) = 0.75."""
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.ar_concentration == pytest.approx(0.75, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.rm_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.receivables_management_analysis(sample_data)
        assert "Receivables Management" in result.summary


# ===== SCORING TESTS =====


class TestReceivablesManagementScoring:
    def test_moderate_dso(self, analyzer, sample_data):
        """DSO=54.75 in 45-60 => base 7.0. RT=6.67 (4-12) => no adj. CE=0.85 (0.70-0.90) => no adj. Score=7.0."""
        result = analyzer.receivables_management_analysis(sample_data)
        assert result.rm_score == pytest.approx(7.0, abs=0.5)
        assert result.rm_grade == "Good"

    def test_excellent_dso(self, analyzer):
        """DSO <= 30 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            accounts_receivable=50_000,
            current_assets=300_000,
            cash=100_000,
        )
        # DSO = 50k/1M * 365 = 18.25 => base 10. RT=20 >=12 => +0.5. CE=0.95 >=0.90 => +0.5. Score=10(capped).
        result = analyzer.receivables_management_analysis(data)
        assert result.rm_score >= 10.0
        assert result.rm_grade == "Excellent"

    def test_weak_dso(self, analyzer):
        """DSO > 120 => base 1.0."""
        data = FinancialData(
            revenue=500_000,
            accounts_receivable=300_000,
            current_assets=400_000,
            cash=20_000,
        )
        # DSO = 300k/500k * 365 = 219 => base 1.0. RT=1.67 <4 => -0.5. CE=0.40 <0.70 => -0.5. Score=0.0.
        result = analyzer.receivables_management_analysis(data)
        assert result.rm_score <= 1.0
        assert result.rm_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase114EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.receivables_management_analysis(FinancialData())
        assert isinstance(result, ReceivablesManagementResult)
        assert result.rm_score == 0.0

    def test_no_current_assets(self, analyzer):
        """CA=0 => AR/CA=None."""
        data = FinancialData(
            revenue=500_000,
            accounts_receivable=50_000,
            cash=30_000,
        )
        result = analyzer.receivables_management_analysis(data)
        assert result.ar_to_current_assets is None

    def test_no_cash(self, analyzer):
        """Cash=0 => ARC=AR/(AR+0) = 1.0."""
        data = FinancialData(
            revenue=500_000,
            accounts_receivable=50_000,
            current_assets=200_000,
        )
        result = analyzer.receivables_management_analysis(data)
        assert result.ar_concentration == pytest.approx(1.0, abs=0.01)

    def test_low_ar(self, analyzer):
        """Very low AR => excellent DSO."""
        data = FinancialData(
            revenue=1_000_000,
            accounts_receivable=10_000,
            current_assets=500_000,
            cash=200_000,
        )
        result = analyzer.receivables_management_analysis(data)
        # DSO = 10k/1M * 365 = 3.65 => base 10
        assert result.rm_score >= 10.0
        assert result.rm_grade == "Excellent"
