"""Phase 102 Tests: Revenue Efficiency Analysis.

Tests for revenue_efficiency_analysis() and RevenueEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    RevenueEfficiencyResult,
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


class TestRevenueEfficiencyDataclass:
    def test_defaults(self):
        r = RevenueEfficiencyResult()
        assert r.revenue_per_asset is None
        assert r.cash_conversion_efficiency is None
        assert r.gross_margin_efficiency is None
        assert r.operating_leverage_ratio is None
        assert r.revenue_to_equity is None
        assert r.net_revenue_retention is None
        assert r.rev_eff_score == 0.0
        assert r.rev_eff_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = RevenueEfficiencyResult(cash_conversion_efficiency=0.22, rev_eff_grade="Excellent")
        assert r.cash_conversion_efficiency == 0.22
        assert r.rev_eff_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestRevenueEfficiencyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert isinstance(result, RevenueEfficiencyResult)

    def test_revenue_per_asset(self, analyzer, sample_data):
        """RPA = 1M/2M = 0.50."""
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.revenue_per_asset == pytest.approx(0.50, abs=0.01)

    def test_cash_conversion_efficiency(self, analyzer, sample_data):
        """CCE = 220k/1M = 0.22."""
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.cash_conversion_efficiency == pytest.approx(0.22, abs=0.01)

    def test_gross_margin_efficiency(self, analyzer, sample_data):
        """GME = 400k/1M = 0.40."""
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.gross_margin_efficiency == pytest.approx(0.40, abs=0.01)

    def test_operating_leverage_ratio(self, analyzer, sample_data):
        """OLR = 200k/400k = 0.50."""
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.operating_leverage_ratio == pytest.approx(0.50, abs=0.01)

    def test_revenue_to_equity(self, analyzer, sample_data):
        """RTE = 1M/1.2M = 0.833."""
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.revenue_to_equity == pytest.approx(0.833, abs=0.01)

    def test_net_revenue_retention(self, analyzer, sample_data):
        """NRR = 150k/1M = 0.15."""
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.net_revenue_retention == pytest.approx(0.15, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.rev_eff_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert "Revenue Efficiency" in result.summary


# ===== SCORING TESTS =====


class TestRevenueEfficiencyScoring:
    def test_good_conversion(self, analyzer, sample_data):
        """CCE=0.22 >= 0.20 => base 8.5. OLR=0.50 >= 0.50 => +0.5. Score=9.0."""
        result = analyzer.revenue_efficiency_analysis(sample_data)
        assert result.rev_eff_score >= 8.0
        assert result.rev_eff_grade == "Excellent"

    def test_excellent_conversion(self, analyzer):
        """CCE >= 0.25 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=400_000,
            gross_profit=600_000,
            operating_income=400_000,
            net_income=200_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
            operating_cash_flow=300_000,
        )
        result = analyzer.revenue_efficiency_analysis(data)
        # CCE = 300k/1M = 0.30 >= 0.25 => base 10. GME=0.60 >= 0.50 => +0.5. Score=10(capped).
        assert result.rev_eff_score >= 10.0
        assert result.rev_eff_grade == "Excellent"

    def test_poor_conversion(self, analyzer):
        """CCE negative => base 1.0."""
        data = FinancialData(
            revenue=500_000,
            cogs=400_000,
            gross_profit=100_000,
            operating_income=-50_000,
            net_income=-80_000,
            total_assets=1_000_000,
            total_equity=500_000,
            operating_cash_flow=-50_000,
        )
        result = analyzer.revenue_efficiency_analysis(data)
        assert result.rev_eff_score <= 2.0
        assert result.rev_eff_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase102EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.revenue_efficiency_analysis(FinancialData())
        assert isinstance(result, RevenueEfficiencyResult)
        # Rev=0 => early return
        assert result.rev_eff_score == 0.0

    def test_no_assets(self, analyzer):
        """TA=0 => revenue_per_asset=None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            gross_profit=400_000,
            operating_income=200_000,
            net_income=150_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.revenue_efficiency_analysis(data)
        assert result.revenue_per_asset is None

    def test_zero_ocf(self, analyzer):
        """OCF=0 => CCE=0 => base 2.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            gross_profit=400_000,
            operating_income=200_000,
            net_income=150_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
        )
        result = analyzer.revenue_efficiency_analysis(data)
        assert result.cash_conversion_efficiency == 0.0

    def test_high_gross_margin_bonus(self, analyzer):
        """GME >= 0.50 => +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=300_000,
            gross_profit=700_000,
            operating_income=300_000,
            net_income=200_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
            operating_cash_flow=200_000,
        )
        result = analyzer.revenue_efficiency_analysis(data)
        assert result.gross_margin_efficiency >= 0.50
