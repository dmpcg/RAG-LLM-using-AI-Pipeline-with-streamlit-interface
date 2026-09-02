"""Phase 245 Tests: Receivables Turnover Analysis.

Tests for receivables_turnover_analysis() and ReceivablesTurnoverResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ReceivablesTurnoverResult,
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


class TestReceivablesTurnoverDataclass:
    def test_defaults(self):
        r = ReceivablesTurnoverResult()
        assert r.rev_to_ar is None
        assert r.dso is None
        assert r.ar_to_ca is None
        assert r.ar_to_ta is None
        assert r.ar_to_revenue is None
        assert r.collection_efficiency is None
        assert r.rto_score == 0.0
        assert r.rto_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ReceivablesTurnoverResult(rev_to_ar=10.0, rto_grade="Excellent")
        assert r.rev_to_ar == 10.0
        assert r.rto_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestReceivablesTurnoverAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.receivables_turnover_analysis(sample_data)
        assert isinstance(result, ReceivablesTurnoverResult)

    def test_rev_to_ar(self, analyzer, sample_data):
        """Rev/AR = 1M/150k = 6.667."""
        result = analyzer.receivables_turnover_analysis(sample_data)
        assert result.rev_to_ar == pytest.approx(6.667, abs=0.01)

    def test_dso(self, analyzer, sample_data):
        """DSO = 150k*365/1M = 54.75 days."""
        result = analyzer.receivables_turnover_analysis(sample_data)
        assert result.dso == pytest.approx(54.75, abs=0.5)

    def test_ar_to_revenue(self, analyzer, sample_data):
        """AR/Rev = 150k/1M = 0.15."""
        result = analyzer.receivables_turnover_analysis(sample_data)
        assert result.ar_to_revenue == pytest.approx(0.15, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.receivables_turnover_analysis(sample_data)
        assert result.rto_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.receivables_turnover_analysis(sample_data)
        assert "Receivables Turnover" in result.summary


# ===== SCORING TESTS =====


class TestReceivablesTurnoverScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """Rev/AR=6.667 >=6.0 => base 5.5. DSO=54.75 (no adj). AR/Rev=0.15 (no adj). Score=5.5."""
        result = analyzer.receivables_turnover_analysis(sample_data)
        assert result.rto_score == pytest.approx(5.5, abs=0.5)
        assert result.rto_grade in ["Good", "Adequate"]

    def test_excellent_fast_collection(self, analyzer):
        """Very fast collection."""
        data = FinancialData(
            revenue=10_000_000,
            accounts_receivable=500_000,
            current_assets=2_000_000,
            total_assets=5_000_000,
        )
        # Rev/AR=20.0 >=15.0 => base 10. DSO=18.25 <=30 => +0.5. AR/Rev=0.05 <=0.05 => +0.5. Score=10.
        result = analyzer.receivables_turnover_analysis(data)
        assert result.rto_score >= 10.0
        assert result.rto_grade == "Excellent"

    def test_weak_slow_collection(self, analyzer):
        """Very slow collection."""
        data = FinancialData(
            revenue=1_000_000,
            accounts_receivable=600_000,
            current_assets=800_000,
            total_assets=2_000_000,
        )
        # Rev/AR=1.667 <2.0 => base 1.0. DSO=219 >90 => -0.5. AR/Rev=0.60 >0.25 => -0.5. Score=0.
        result = analyzer.receivables_turnover_analysis(data)
        assert result.rto_score <= 0.5
        assert result.rto_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase245EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.receivables_turnover_analysis(FinancialData())
        assert isinstance(result, ReceivablesTurnoverResult)
        assert result.rto_score == 0.0

    def test_no_ar(self, analyzer):
        """AR=None => insufficient data => score 0."""
        data = FinancialData(revenue=1_000_000)
        result = analyzer.receivables_turnover_analysis(data)
        assert result.rto_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => insufficient data => score 0."""
        data = FinancialData(accounts_receivable=150_000)
        result = analyzer.receivables_turnover_analysis(data)
        assert result.rto_score == 0.0

    def test_zero_ar(self, analyzer):
        """AR=0 => insufficient data => score 0."""
        data = FinancialData(
            revenue=1_000_000,
            accounts_receivable=0,
        )
        result = analyzer.receivables_turnover_analysis(data)
        assert result.rto_score == 0.0
