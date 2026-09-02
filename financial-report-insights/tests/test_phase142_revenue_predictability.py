"""Phase 142 Tests: Revenue Predictability Analysis.

Tests for revenue_predictability_analysis() and RevenuePredictabilityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    RevenuePredictabilityResult,
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


class TestRevenuePredictabilityDataclass:
    def test_defaults(self):
        r = RevenuePredictabilityResult()
        assert r.revenue_to_assets is None
        assert r.revenue_to_equity is None
        assert r.revenue_to_debt is None
        assert r.gross_margin is None
        assert r.operating_margin is None
        assert r.net_margin is None
        assert r.rp_score == 0.0
        assert r.rp_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = RevenuePredictabilityResult(operating_margin=0.25, rp_grade="Excellent")
        assert r.operating_margin == 0.25
        assert r.rp_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestRevenuePredictabilityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert isinstance(result, RevenuePredictabilityResult)

    def test_revenue_to_assets(self, analyzer, sample_data):
        """R/A = 1M / 2M = 0.50."""
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.revenue_to_assets == pytest.approx(0.50, abs=0.01)

    def test_revenue_to_equity(self, analyzer, sample_data):
        """R/E = 1M / 1.2M = 0.833."""
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.revenue_to_equity == pytest.approx(0.833, abs=0.01)

    def test_revenue_to_debt(self, analyzer, sample_data):
        """R/D = 1M / 400k = 2.50."""
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.revenue_to_debt == pytest.approx(2.50, abs=0.01)

    def test_gross_margin(self, analyzer, sample_data):
        """GM = 400k / 1M = 0.40."""
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.gross_margin == pytest.approx(0.40, abs=0.01)

    def test_operating_margin(self, analyzer, sample_data):
        """OM = 200k / 1M = 0.20."""
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.operating_margin == pytest.approx(0.20, abs=0.01)

    def test_net_margin(self, analyzer, sample_data):
        """NM = 150k / 1M = 0.15."""
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.net_margin == pytest.approx(0.15, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.rp_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert "Revenue Predictability" in result.summary


# ===== SCORING TESTS =====


class TestRevenuePredictabilityScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OM=0.20 => base 7.0. GM=0.40 (neither). NM=0.15 >=0.15 => +0.5. Score=7.5."""
        result = analyzer.revenue_predictability_analysis(sample_data)
        assert result.rp_score == pytest.approx(7.5, abs=0.5)
        assert result.rp_grade == "Good"

    def test_excellent_predictability(self, analyzer):
        """OM >= 0.30 => base 10."""
        data = FinancialData(
            revenue=2_000_000,
            gross_profit=1_200_000,
            operating_income=700_000,
            net_income=500_000,
            total_assets=3_000_000,
            total_equity=2_000_000,
            total_debt=200_000,
        )
        # OM=0.35 => 10. GM=0.60 >=0.50 => +0.5. NM=0.25 >=0.15 => +0.5. Capped 10.
        result = analyzer.revenue_predictability_analysis(data)
        assert result.rp_score >= 10.0
        assert result.rp_grade == "Excellent"

    def test_weak_predictability(self, analyzer):
        """OM < 0.05 => base 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=150_000,
            operating_income=20_000,
            net_income=5_000,
            total_assets=2_000_000,
            total_equity=800_000,
            total_debt=500_000,
        )
        # OM=0.02 <0.05 => 1.0. GM=0.15 <0.20 => -0.5. NM=0.005 <0.03 => -0.5. Score=0.0.
        result = analyzer.revenue_predictability_analysis(data)
        assert result.rp_score <= 1.0
        assert result.rp_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase142EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.revenue_predictability_analysis(FinancialData())
        assert isinstance(result, RevenuePredictabilityResult)
        assert result.rp_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => most ratios None, score 0."""
        data = FinancialData(
            gross_profit=400_000,
            operating_income=200_000,
            net_income=150_000,
            total_assets=2_000_000,
        )
        result = analyzer.revenue_predictability_analysis(data)
        assert result.operating_margin is None
        assert result.rp_score == 0.0

    def test_no_operating_income(self, analyzer):
        """OI=None => OM=None => score 0, but GM still computed."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=400_000,
            net_income=150_000,
            total_assets=2_000_000,
        )
        result = analyzer.revenue_predictability_analysis(data)
        assert result.operating_margin is None
        assert result.gross_margin is not None

    def test_no_net_income(self, analyzer):
        """NI=None => NM=None."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=400_000,
            operating_income=200_000,
            total_assets=2_000_000,
        )
        result = analyzer.revenue_predictability_analysis(data)
        assert result.net_margin is None
        assert result.operating_margin is not None
