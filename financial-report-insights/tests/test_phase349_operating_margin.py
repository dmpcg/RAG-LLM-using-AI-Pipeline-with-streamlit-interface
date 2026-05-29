"""Phase 349 Tests: Operating Margin Analysis.

Tests for operating_margin_analysis() and OperatingMarginResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperatingMarginResult,
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


class TestOperatingMarginDataclass:
    def test_defaults(self):
        r = OperatingMarginResult()
        assert r.operating_margin is None
        assert r.oi_to_revenue is None
        assert r.ebit_margin is None
        assert r.ebitda_margin is None
        assert r.margin_trend is None
        assert r.opm_spread is None
        assert r.opm_score == 0.0
        assert r.opm_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperatingMarginResult(oi_to_revenue=0.20, opm_grade="Excellent")
        assert r.oi_to_revenue == 0.20
        assert r.opm_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestOperatingMarginAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operating_margin_analysis(sample_data)
        assert isinstance(result, OperatingMarginResult)

    def test_oi_to_revenue(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.operating_margin_analysis(sample_data)
        assert result.oi_to_revenue == pytest.approx(0.20, abs=0.001)

    def test_ebit_margin(self, analyzer, sample_data):
        """EBIT/Rev = 200k/1M = 0.20."""
        result = analyzer.operating_margin_analysis(sample_data)
        assert result.ebit_margin == pytest.approx(0.20, abs=0.001)

    def test_ebitda_margin(self, analyzer, sample_data):
        """EBITDA/Rev = 250k/1M = 0.25."""
        result = analyzer.operating_margin_analysis(sample_data)
        assert result.ebitda_margin == pytest.approx(0.25, abs=0.001)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operating_margin_analysis(sample_data)
        assert result.opm_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operating_margin_analysis(sample_data)
        assert "Operating Margin" in result.summary


# ===== SCORING TESTS =====


class TestOperatingMarginScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OI/Rev=0.20 in [0.20,0.25)=>base 8.5. EBITDA=0.25>=0.20(+0.5). Both>0(+0.5). Score=9.5."""
        result = analyzer.operating_margin_analysis(sample_data)
        assert result.opm_score == pytest.approx(9.5, abs=0.5)
        assert result.opm_grade in ["Excellent", "Good"]

    def test_high_margin(self, analyzer):
        """High OI/Rev — very profitable operations."""
        data = FinancialData(
            operating_income=300_000,
            revenue=1_000_000,
            ebit=300_000,
            ebitda=350_000,
        )
        # OI/Rev=0.30>=0.25=>base 10. EBITDA=0.35>=0.20(+0.5). Both>0(+0.5). Score=10.
        result = analyzer.operating_margin_analysis(data)
        assert result.opm_score >= 10.0
        assert result.opm_grade == "Excellent"

    def test_low_margin(self, analyzer):
        """Low OI/Rev — thin margins."""
        data = FinancialData(
            operating_income=10_000,
            revenue=1_000_000,
            ebit=10_000,
            ebitda=30_000,
        )
        # OI/Rev=0.01<0.02=>base 1.0. EBITDA=0.03<0.20(no adj). Both>0(+0.5). Score=1.5.
        result = analyzer.operating_margin_analysis(data)
        assert result.opm_score <= 3.0
        assert result.opm_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase349EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operating_margin_analysis(FinancialData())
        assert isinstance(result, OperatingMarginResult)
        assert result.opm_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => OI/Rev=None => score 0."""
        data = FinancialData(operating_income=200_000)
        result = analyzer.operating_margin_analysis(data)
        assert result.oi_to_revenue is None
        assert result.opm_score == 0.0

    def test_no_operating_income(self, analyzer):
        """OI=None => ratio=None."""
        data = FinancialData(revenue=1_000_000)
        result = analyzer.operating_margin_analysis(data)
        assert result.opm_score == 0.0
