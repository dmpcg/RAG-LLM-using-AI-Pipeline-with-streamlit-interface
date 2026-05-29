"""Phase 275 Tests: Operating Income Quality Analysis.

Tests for operating_income_quality_analysis() and OperatingIncomeQualityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperatingIncomeQualityResult,
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


class TestOperatingIncomeQualityDataclass:
    def test_defaults(self):
        r = OperatingIncomeQualityResult()
        assert r.oi_to_revenue is None
        assert r.oi_to_ebitda is None
        assert r.oi_to_ocf is None
        assert r.oi_to_total_assets is None
        assert r.operating_spread is None
        assert r.oi_cash_backing is None
        assert r.oiq_score == 0.0
        assert r.oiq_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperatingIncomeQualityResult(oi_to_revenue=0.20, oiq_grade="Excellent")
        assert r.oi_to_revenue == 0.20
        assert r.oiq_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestOperatingIncomeQualityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert isinstance(result, OperatingIncomeQualityResult)

    def test_oi_to_revenue(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert result.oi_to_revenue == pytest.approx(0.20, abs=0.01)

    def test_oi_to_ebitda(self, analyzer, sample_data):
        """OI/EBITDA = 200k/250k = 0.80."""
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert result.oi_to_ebitda == pytest.approx(0.80, abs=0.01)

    def test_oi_to_ocf(self, analyzer, sample_data):
        """OI/OCF = 200k/220k = 0.909."""
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert result.oi_to_ocf == pytest.approx(0.909, abs=0.01)

    def test_oi_cash_backing(self, analyzer, sample_data):
        """OCF/OI = 220k/200k = 1.10."""
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert result.oi_cash_backing == pytest.approx(1.10, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert result.oiq_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert "Operating Income Quality" in result.summary


# ===== SCORING TESTS =====


class TestOperatingIncomeQualityScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OI/Rev=0.20 in [0.20,0.30)=>base 8.5. OCF(220k)>OI(200k)(+0.5). OI>0&Rev>0(+0.5). Score=9.5."""
        result = analyzer.operating_income_quality_analysis(sample_data)
        assert result.oiq_score == pytest.approx(9.5, abs=0.5)
        assert result.oiq_grade == "Excellent"

    def test_very_high_margin(self, analyzer):
        """Very high operating margin."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=400_000,
            operating_cash_flow=500_000,
            ebitda=450_000,
        )
        # OI/Rev=0.40>=0.30=>base 10. OCF>OI(+0.5). OI>0&Rev>0(+0.5). Score=10.
        result = analyzer.operating_income_quality_analysis(data)
        assert result.oiq_score >= 10.0
        assert result.oiq_grade == "Excellent"

    def test_thin_margin_weak(self, analyzer):
        """Very thin operating margin."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=10_000,
        )
        # OI/Rev=0.01<0.02=>base 1.0. No OCF(no adj). OI>0&Rev>0(+0.5). Score=1.5.
        result = analyzer.operating_income_quality_analysis(data)
        assert result.oiq_score <= 2.0
        assert result.oiq_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase275EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operating_income_quality_analysis(FinancialData())
        assert isinstance(result, OperatingIncomeQualityResult)
        assert result.oiq_score == 0.0

    def test_no_operating_income(self, analyzer):
        data = FinancialData(revenue=1_000_000)
        result = analyzer.operating_income_quality_analysis(data)
        assert result.oiq_score == 0.0

    def test_no_revenue(self, analyzer):
        data = FinancialData(operating_income=200_000)
        result = analyzer.operating_income_quality_analysis(data)
        assert result.oiq_score == 0.0

    def test_negative_oi(self, analyzer):
        data = FinancialData(operating_income=-50_000, revenue=1_000_000)
        result = analyzer.operating_income_quality_analysis(data)
        # Negative OI gives a low but non-zero score via _scored_analysis
        assert result.oiq_score <= 2.0
