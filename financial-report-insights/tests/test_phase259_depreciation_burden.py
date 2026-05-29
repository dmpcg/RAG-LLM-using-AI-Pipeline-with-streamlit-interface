"""Phase 259 Tests: Depreciation Burden Analysis.

Tests for depreciation_burden_analysis() and DepreciationBurdenResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DepreciationBurdenResult,
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


class TestDepreciationBurdenDataclass:
    def test_defaults(self):
        r = DepreciationBurdenResult()
        assert r.dep_to_revenue is None
        assert r.dep_to_assets is None
        assert r.dep_to_ebitda is None
        assert r.dep_to_gross_profit is None
        assert r.ebitda_to_ebit_spread is None
        assert r.asset_age_proxy is None
        assert r.db_score == 0.0
        assert r.db_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DepreciationBurdenResult(dep_to_revenue=0.05, db_grade="Excellent")
        assert r.dep_to_revenue == 0.05
        assert r.db_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestDepreciationBurdenAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert isinstance(result, DepreciationBurdenResult)

    def test_dep_to_revenue(self, analyzer, sample_data):
        """D&A/Rev = 50k/1M = 0.05."""
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert result.dep_to_revenue == pytest.approx(0.05, abs=0.01)

    def test_dep_to_assets(self, analyzer, sample_data):
        """D&A/TA = 50k/2M = 0.025."""
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert result.dep_to_assets == pytest.approx(0.025, abs=0.001)

    def test_dep_to_ebitda(self, analyzer, sample_data):
        """D&A/EBITDA = 50k/250k = 0.20."""
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert result.dep_to_ebitda == pytest.approx(0.20, abs=0.01)

    def test_dep_to_gross_profit(self, analyzer, sample_data):
        """D&A/GP = 50k/400k = 0.125."""
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert result.dep_to_gross_profit == pytest.approx(0.125, abs=0.01)

    def test_ebitda_to_ebit_spread(self, analyzer, sample_data):
        """EBITDA/EBIT = 250k/200k = 1.25."""
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert result.ebitda_to_ebit_spread == pytest.approx(1.25, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert result.db_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert "Depreciation Burden" in result.summary


# ===== SCORING TESTS =====


class TestDepreciationBurdenScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """DTR=0.05 in (0.03,0.05]=>base 8.5. D&A/EBITDA=0.20<=0.20(+0.5). D&A/TA=0.025<=0.03(+0.5). Score=9.5."""
        result = analyzer.depreciation_burden_analysis(sample_data)
        assert result.db_score == pytest.approx(9.5, abs=0.5)
        assert result.db_grade == "Excellent"

    def test_very_light_depreciation(self, analyzer):
        """Very low depreciation."""
        data = FinancialData(
            revenue=1_000_000,
            depreciation=20_000,
            ebitda=300_000,
            ebit=280_000,
            total_assets=3_000_000,
            gross_profit=500_000,
        )
        # DTR=0.02<=0.03=>base 10. D&A/EBITDA=0.067<=0.20(+0.5). D&A/TA=0.0067<=0.03(+0.5). Score=10.
        result = analyzer.depreciation_burden_analysis(data)
        assert result.db_score >= 10.0
        assert result.db_grade == "Excellent"

    def test_heavy_depreciation_weak(self, analyzer):
        """Heavy depreciation burden."""
        data = FinancialData(
            revenue=1_000_000,
            depreciation=300_000,
            ebitda=350_000,
            total_assets=1_000_000,
        )
        # DTR=0.30>0.25=>base 1.0
        result = analyzer.depreciation_burden_analysis(data)
        assert result.db_score <= 2.0
        assert result.db_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase259EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.depreciation_burden_analysis(FinancialData())
        assert isinstance(result, DepreciationBurdenResult)
        assert result.db_score == 0.0

    def test_no_revenue(self, analyzer):
        data = FinancialData(depreciation=50_000)
        result = analyzer.depreciation_burden_analysis(data)
        assert result.db_score == 0.0

    def test_no_depreciation(self, analyzer):
        data = FinancialData(revenue=1_000_000)
        result = analyzer.depreciation_burden_analysis(data)
        assert result.db_score == 0.0

    def test_zero_depreciation(self, analyzer):
        data = FinancialData(revenue=1_000_000, depreciation=0)
        result = analyzer.depreciation_burden_analysis(data)
        # Zero depreciation burden is excellent (lower is better)
        assert result.db_score == 10.0
