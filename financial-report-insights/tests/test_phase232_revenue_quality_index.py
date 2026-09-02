"""Phase 232 Tests: Revenue Quality Index Analysis.

Tests for revenue_quality_index_analysis() and RevenueQualityIndexResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    RevenueQualityIndexResult,
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


class TestRevenueQualityIndexDataclass:
    def test_defaults(self):
        r = RevenueQualityIndexResult()
        assert r.ocf_to_revenue is None
        assert r.gross_margin is None
        assert r.ni_to_revenue is None
        assert r.ebitda_to_revenue is None
        assert r.ar_to_revenue is None
        assert r.cash_to_revenue is None
        assert r.rqi_score == 0.0
        assert r.rqi_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = RevenueQualityIndexResult(ocf_to_revenue=0.22, rqi_grade="Excellent")
        assert r.ocf_to_revenue == 0.22
        assert r.rqi_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestRevenueQualityIndexAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert isinstance(result, RevenueQualityIndexResult)

    def test_ocf_to_revenue(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert result.ocf_to_revenue == pytest.approx(0.22, abs=0.005)

    def test_gross_margin(self, analyzer, sample_data):
        """GM = 400k/1M = 0.40."""
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert result.gross_margin == pytest.approx(0.40, abs=0.005)

    def test_ni_to_revenue(self, analyzer, sample_data):
        """NI/Rev = 150k/1M = 0.15."""
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert result.ni_to_revenue == pytest.approx(0.15, abs=0.005)

    def test_ebitda_to_revenue(self, analyzer, sample_data):
        """EBITDA/Rev = 250k/1M = 0.25."""
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert result.ebitda_to_revenue == pytest.approx(0.25, abs=0.005)

    def test_ar_to_revenue(self, analyzer, sample_data):
        """AR/Rev = 150k/1M = 0.15."""
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert result.ar_to_revenue == pytest.approx(0.15, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert result.rqi_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert "Revenue Quality Index" in result.summary


# ===== SCORING TESTS =====


class TestRevenueQualityIndexScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OCF/Rev=0.22 >=0.20 => base 8.5. GM=0.40 <0.50 no bonus. AR/Rev=0.15 >0.10 no bonus. Score=8.5."""
        result = analyzer.revenue_quality_index_analysis(sample_data)
        assert result.rqi_score == pytest.approx(8.5, abs=0.5)
        assert result.rqi_grade == "Excellent"

    def test_excellent_quality(self, analyzer):
        """Very high OCF/Revenue + high margin + low AR."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=300_000,
            gross_profit=600_000,
            accounts_receivable=50_000,
        )
        # OCF/Rev=0.30 >=0.25 => base 10. GM=0.60 >=0.50 => +0.5. AR/Rev=0.05 <=0.10 => +0.5. Score=10.
        result = analyzer.revenue_quality_index_analysis(data)
        assert result.rqi_score >= 10.0
        assert result.rqi_grade == "Excellent"

    def test_weak_quality(self, analyzer):
        """Negative OCF + low margin + high AR."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=-50_000,
            gross_profit=150_000,
            accounts_receivable=400_000,
        )
        # OCF/Rev=-0.05 <0.00 => base 1.0. GM=0.15 <0.20 => -0.5. AR/Rev=0.40 >0.30 => -0.5. Score=0.
        result = analyzer.revenue_quality_index_analysis(data)
        assert result.rqi_score <= 0.5
        assert result.rqi_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase232EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.revenue_quality_index_analysis(FinancialData())
        assert isinstance(result, RevenueQualityIndexResult)
        assert result.rqi_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => insufficient data => score 0."""
        data = FinancialData(
            operating_cash_flow=220_000,
        )
        result = analyzer.revenue_quality_index_analysis(data)
        assert result.ocf_to_revenue is None
        assert result.rqi_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => insufficient data => score 0."""
        data = FinancialData(
            revenue=1_000_000,
        )
        result = analyzer.revenue_quality_index_analysis(data)
        assert result.ocf_to_revenue is None
        assert result.rqi_score == 0.0

    def test_no_gross_profit(self, analyzer):
        """GP=None => gross_margin=None, OCF still works."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.revenue_quality_index_analysis(data)
        assert result.ocf_to_revenue is not None
        assert result.gross_margin is None
        assert result.rqi_score > 0.0
