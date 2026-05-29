"""Phase 279 Tests: Capital Adequacy Analysis.

Tests for capital_adequacy_analysis() and CapitalAdequacyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CapitalAdequacyResult,
    CharlieAnalyzer,
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


class TestCapitalAdequacyDataclass:
    def test_defaults(self):
        r = CapitalAdequacyResult()
        assert r.equity_ratio is None
        assert r.equity_to_debt is None
        assert r.retained_to_equity is None
        assert r.equity_to_liabilities is None
        assert r.tangible_equity_ratio is None
        assert r.capital_buffer is None
        assert r.caq_score == 0.0
        assert r.caq_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CapitalAdequacyResult(equity_ratio=0.60, caq_grade="Excellent")
        assert r.equity_ratio == 0.60
        assert r.caq_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestCapitalAdequacyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert isinstance(result, CapitalAdequacyResult)

    def test_equity_ratio(self, analyzer, sample_data):
        """TE/TA = 1.2M/2M = 0.60."""
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert result.equity_ratio == pytest.approx(0.60, abs=0.01)

    def test_equity_to_debt(self, analyzer, sample_data):
        """TE/TD = 1.2M/400k = 3.0."""
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert result.equity_to_debt == pytest.approx(3.0, abs=0.01)

    def test_retained_to_equity(self, analyzer, sample_data):
        """RE/TE = 600k/1.2M = 0.50."""
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert result.retained_to_equity == pytest.approx(0.50, abs=0.01)

    def test_equity_to_liabilities(self, analyzer, sample_data):
        """TE/TL = 1.2M/800k = 1.50."""
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert result.equity_to_liabilities == pytest.approx(1.50, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert result.caq_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert "Capital Adequacy" in result.summary


# ===== SCORING TESTS =====


class TestCapitalAdequacyScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """ER=0.60>=0.60=>base 10. RE/TE=0.50>=0.50(+0.5). Equity>0&TA>0(+0.5). Score=10 (capped)."""
        result = analyzer.capital_adequacy_analysis(sample_data)
        assert result.caq_score >= 10.0
        assert result.caq_grade == "Excellent"

    def test_moderate_equity(self, analyzer):
        """Moderate equity ratio."""
        data = FinancialData(
            total_equity=800_000,
            total_assets=2_000_000,
            total_liabilities=1_200_000,
            retained_earnings=200_000,
        )
        # ER=0.40 in [0.40,0.50)=>base 7.0. RE/TE=0.25<0.50(no adj). Equity>0&TA>0(+0.5). Score=7.5.
        result = analyzer.capital_adequacy_analysis(data)
        assert result.caq_score == pytest.approx(7.5, abs=0.5)
        assert result.caq_grade in ["Excellent", "Good"]

    def test_thin_equity_weak(self, analyzer):
        """Very thin equity base."""
        data = FinancialData(
            total_equity=100_000,
            total_assets=2_000_000,
            total_liabilities=1_900_000,
        )
        # ER=0.05<0.10=>base 1.0. No RE(no adj). Equity>0&TA>0(+0.5). Score=1.5.
        result = analyzer.capital_adequacy_analysis(data)
        assert result.caq_score <= 2.0
        assert result.caq_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase279EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.capital_adequacy_analysis(FinancialData())
        assert isinstance(result, CapitalAdequacyResult)
        assert result.caq_score == 0.0

    def test_no_equity(self, analyzer):
        data = FinancialData(total_assets=2_000_000)
        result = analyzer.capital_adequacy_analysis(data)
        assert result.caq_score == 0.0

    def test_no_total_assets(self, analyzer):
        data = FinancialData(total_equity=1_200_000)
        result = analyzer.capital_adequacy_analysis(data)
        assert result.caq_score == 0.0

    def test_zero_equity(self, analyzer):
        data = FinancialData(total_equity=0, total_assets=2_000_000)
        result = analyzer.capital_adequacy_analysis(data)
        # Zero equity ratio gives a low but non-zero score via _scored_analysis
        assert result.caq_score <= 2.0
