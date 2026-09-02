"""Phase 55 Tests: Return on Assets Quality Analysis.

Tests for roa_quality_analysis() and ROAQualityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ROAQualityResult,
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
    )


# ===== DATACLASS TESTS =====


class TestROAQualityDataclass:
    def test_defaults(self):
        r = ROAQualityResult()
        assert r.roa_pct is None
        assert r.operating_roa_pct is None
        assert r.cash_roa_pct is None
        assert r.asset_turnover is None
        assert r.fixed_asset_turnover is None
        assert r.capital_intensity is None
        assert r.roa_score == 0.0
        assert r.roa_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ROAQualityResult(
            roa_pct=7.5,
            roa_grade="Good",
        )
        assert r.roa_pct == 7.5
        assert r.roa_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestROAQualityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.roa_quality_analysis(sample_data)
        assert isinstance(result, ROAQualityResult)

    def test_roa(self, analyzer, sample_data):
        """NI/TA = 150k/2M = 7.5%."""
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.roa_pct == pytest.approx(7.5, abs=0.1)

    def test_operating_roa(self, analyzer, sample_data):
        """OI/TA = 200k/2M = 10%."""
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.operating_roa_pct == pytest.approx(10.0, abs=0.1)

    def test_cash_roa(self, analyzer, sample_data):
        """OCF/TA = 220k/2M = 11%."""
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.cash_roa_pct == pytest.approx(11.0, abs=0.1)

    def test_asset_turnover(self, analyzer, sample_data):
        """Rev/TA = 1M/2M = 0.50."""
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.asset_turnover == pytest.approx(0.50, abs=0.01)

    def test_fixed_asset_turnover(self, analyzer, sample_data):
        """Rev/(TA-CA) = 1M/(2M-500k) = 0.667."""
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.fixed_asset_turnover == pytest.approx(0.667, abs=0.01)

    def test_capital_intensity(self, analyzer, sample_data):
        """TA/Rev = 2M/1M = 2.0."""
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.capital_intensity == pytest.approx(2.0, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.roa_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.roa_quality_analysis(sample_data)
        assert "Return on Assets" in result.summary


# ===== SCORING TESTS =====


class TestROAQualityScoring:
    def test_high_roa(self, analyzer):
        """ROA 20% => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=400_000,
            total_assets=2_000_000,
            operating_income=500_000,
            operating_cash_flow=450_000,
            current_assets=800_000,
        )
        result = analyzer.roa_quality_analysis(data)
        assert result.roa_score >= 8.0
        assert result.roa_grade == "Excellent"

    def test_moderate_roa(self, analyzer, sample_data):
        """ROA 7.5% => base 7.0."""
        result = analyzer.roa_quality_analysis(sample_data)
        assert result.roa_score >= 6.0

    def test_low_roa(self, analyzer):
        """ROA 2% => base 3.5."""
        data = FinancialData(
            revenue=500_000,
            net_income=40_000,
            total_assets=2_000_000,
        )
        result = analyzer.roa_quality_analysis(data)
        assert result.roa_score < 6.0

    def test_negative_roa(self, analyzer):
        """ROA -10% => base 0.5, Weak."""
        data = FinancialData(
            revenue=500_000,
            net_income=-200_000,
            total_assets=2_000_000,
        )
        result = analyzer.roa_quality_analysis(data)
        assert result.roa_score < 4.0
        assert result.roa_grade == "Weak"

    def test_cash_roa_bonus(self, analyzer):
        """Cash ROA > ROA => +0.5 bonus."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=200_000,
            total_assets=2_000_000,
            operating_cash_flow=300_000,
        )
        result = analyzer.roa_quality_analysis(data)
        # ROA=10%, cash ROA=15% > ROA => +0.5
        assert result.roa_score >= 9.0


# ===== EDGE CASES =====


class TestPhase55EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.roa_quality_analysis(FinancialData())
        assert isinstance(result, ROAQualityResult)
        assert result.roa_pct is None

    def test_no_assets(self, analyzer):
        data = FinancialData(revenue=1_000_000, net_income=100_000)
        result = analyzer.roa_quality_analysis(data)
        assert result.roa_pct is None

    def test_zero_assets(self, analyzer):
        data = FinancialData(revenue=1_000_000, net_income=100_000, total_assets=0)
        result = analyzer.roa_quality_analysis(data)
        assert result.roa_pct is None

    def test_no_net_income(self, analyzer):
        """Assets but no NI => roa_pct is None."""
        data = FinancialData(revenue=500_000, total_assets=2_000_000)
        result = analyzer.roa_quality_analysis(data)
        assert result.roa_pct is None
