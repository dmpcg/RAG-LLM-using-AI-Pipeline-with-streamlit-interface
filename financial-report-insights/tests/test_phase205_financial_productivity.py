"""Phase 205 Tests: Financial Productivity Analysis.

Tests for financial_productivity_analysis() and FinancialProductivityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FinancialProductivityResult,
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


class TestFinancialProductivityDataclass:
    def test_defaults(self):
        r = FinancialProductivityResult()
        assert r.revenue_per_asset is None
        assert r.revenue_per_equity is None
        assert r.ebitda_per_employee_proxy is None
        assert r.operating_income_per_asset is None
        assert r.net_income_per_revenue is None
        assert r.cash_flow_per_asset is None
        assert r.fp_score == 0.0
        assert r.fp_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FinancialProductivityResult(revenue_per_asset=0.50, fp_grade="Good")
        assert r.revenue_per_asset == 0.50
        assert r.fp_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestFinancialProductivityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.financial_productivity_analysis(sample_data)
        assert isinstance(result, FinancialProductivityResult)

    def test_revenue_per_asset(self, analyzer, sample_data):
        """Rev/TA = 1M/2M = 0.50."""
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.revenue_per_asset == pytest.approx(0.50, abs=0.005)

    def test_revenue_per_equity(self, analyzer, sample_data):
        """Rev/TE = 1M/1.2M = 0.833."""
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.revenue_per_equity == pytest.approx(0.833, abs=0.005)

    def test_ebitda_per_opex(self, analyzer, sample_data):
        """EBITDA/OpEx = 250k/200k = 1.25."""
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.ebitda_per_employee_proxy == pytest.approx(1.25, abs=0.01)

    def test_operating_income_per_asset(self, analyzer, sample_data):
        """OI/TA = 200k/2M = 0.10."""
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.operating_income_per_asset == pytest.approx(0.10, abs=0.005)

    def test_net_income_per_revenue(self, analyzer, sample_data):
        """NI/Rev = 150k/1M = 0.15."""
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.net_income_per_revenue == pytest.approx(0.15, abs=0.005)

    def test_cash_flow_per_asset(self, analyzer, sample_data):
        """OCF/TA = 220k/2M = 0.11."""
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.cash_flow_per_asset == pytest.approx(0.11, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.fp_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.financial_productivity_analysis(sample_data)
        assert "Financial Productivity" in result.summary


# ===== SCORING TESTS =====


class TestFinancialProductivityScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """RPA=0.50 >=0.40 => base 4.0. EPO=1.25 <1.5 no adj. CFPA=0.11 <0.12 no adj. Score=4.0."""
        result = analyzer.financial_productivity_analysis(sample_data)
        assert result.fp_score == pytest.approx(4.0, abs=0.5)
        assert result.fp_grade == "Adequate"

    def test_excellent_productivity(self, analyzer):
        """Very high asset turnover."""
        data = FinancialData(
            revenue=5_000_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
            ebitda=800_000,
            operating_expenses=400_000,
            operating_income=600_000,
            net_income=400_000,
            operating_cash_flow=700_000,
        )
        result = analyzer.financial_productivity_analysis(data)
        assert result.fp_score >= 10.0
        assert result.fp_grade == "Excellent"

    def test_weak_productivity(self, analyzer):
        """Very low asset turnover."""
        data = FinancialData(
            revenue=100_000,
            total_assets=3_000_000,
            total_equity=1_000_000,
            ebitda=20_000,
            operating_expenses=80_000,
            operating_income=10_000,
            net_income=5_000,
            operating_cash_flow=30_000,
        )
        # RPA=100k/3M=0.033 <0.20 => base 1.0. EPO=20k/80k=0.25 <0.50 => -0.5. CFPA=30k/3M=0.01 <0.05 => -0.5. Score=0.0.
        result = analyzer.financial_productivity_analysis(data)
        assert result.fp_score <= 0.5
        assert result.fp_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase205EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.financial_productivity_analysis(FinancialData())
        assert isinstance(result, FinancialProductivityResult)
        assert result.fp_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => RPA=None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.financial_productivity_analysis(data)
        assert result.revenue_per_asset is None
        assert result.fp_score == 0.0

    def test_no_assets(self, analyzer):
        """TA=None => RPA=None."""
        data = FinancialData(
            revenue=1_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.financial_productivity_analysis(data)
        assert result.revenue_per_asset is None
        assert result.fp_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => CFPA=None, but RPA still works."""
        data = FinancialData(
            revenue=1_000_000,
            total_assets=2_000_000,
            ebitda=250_000,
            operating_expenses=200_000,
        )
        result = analyzer.financial_productivity_analysis(data)
        assert result.cash_flow_per_asset is None
        assert result.revenue_per_asset is not None
