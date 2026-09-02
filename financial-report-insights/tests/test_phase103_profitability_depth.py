"""Phase 103 Tests: Profitability Depth Analysis.

Tests for profitability_depth_analysis() and ProfitabilityDepthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ProfitabilityDepthResult,
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


class TestProfitabilityDepthDataclass:
    def test_defaults(self):
        r = ProfitabilityDepthResult()
        assert r.gross_margin is None
        assert r.operating_margin is None
        assert r.ebitda_margin is None
        assert r.net_margin is None
        assert r.margin_spread is None
        assert r.profit_retention_ratio is None
        assert r.pd_score == 0.0
        assert r.pd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ProfitabilityDepthResult(operating_margin=0.20, pd_grade="Excellent")
        assert r.operating_margin == 0.20
        assert r.pd_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestProfitabilityDepthAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.profitability_depth_analysis(sample_data)
        assert isinstance(result, ProfitabilityDepthResult)

    def test_gross_margin(self, analyzer, sample_data):
        """GM = 400k/1M = 0.40."""
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.gross_margin == pytest.approx(0.40, abs=0.01)

    def test_operating_margin(self, analyzer, sample_data):
        """OM = 200k/1M = 0.20."""
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.operating_margin == pytest.approx(0.20, abs=0.01)

    def test_ebitda_margin(self, analyzer, sample_data):
        """EM = 250k/1M = 0.25."""
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.ebitda_margin == pytest.approx(0.25, abs=0.01)

    def test_net_margin(self, analyzer, sample_data):
        """NM = 150k/1M = 0.15."""
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.net_margin == pytest.approx(0.15, abs=0.01)

    def test_margin_spread(self, analyzer, sample_data):
        """Spread = 0.40 - 0.15 = 0.25."""
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.margin_spread == pytest.approx(0.25, abs=0.01)

    def test_profit_retention_ratio(self, analyzer, sample_data):
        """PRR = 150k/400k = 0.375."""
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.profit_retention_ratio == pytest.approx(0.375, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.pd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.profitability_depth_analysis(sample_data)
        assert "Profitability Depth" in result.summary


# ===== SCORING TESTS =====


class TestProfitabilityDepthScoring:
    def test_good_margins(self, analyzer, sample_data):
        """OM=0.20 => base 8.5. Score=8.5."""
        result = analyzer.profitability_depth_analysis(sample_data)
        assert result.pd_score >= 8.0
        assert result.pd_grade == "Excellent"

    def test_excellent_margins(self, analyzer):
        """OM >= 0.25 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=300_000,
            gross_profit=700_000,
            operating_income=300_000,
            ebitda=350_000,
            net_income=250_000,
        )
        result = analyzer.profitability_depth_analysis(data)
        # OM=0.30 => base 10. GM=0.70 >= 0.50 => +0.5. PRR=250k/700k=0.357 (no bonus). Score=10 (capped).
        assert result.pd_score >= 10.0
        assert result.pd_grade == "Excellent"

    def test_negative_margins(self, analyzer):
        """OM < 0 => base 1.0."""
        data = FinancialData(
            revenue=500_000,
            cogs=400_000,
            gross_profit=100_000,
            operating_income=-50_000,
            net_income=-80_000,
        )
        result = analyzer.profitability_depth_analysis(data)
        assert result.pd_score <= 2.0
        assert result.pd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase103EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.profitability_depth_analysis(FinancialData())
        assert isinstance(result, ProfitabilityDepthResult)
        assert result.pd_score == 0.0

    def test_revenue_only(self, analyzer):
        """Only revenue => NM=0, OM=0."""
        data = FinancialData(revenue=1_000_000)
        result = analyzer.profitability_depth_analysis(data)
        assert result.net_margin == 0.0

    def test_no_gross_profit(self, analyzer):
        """No GP/COGS => gross_margin=None."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
            net_income=150_000,
        )
        result = analyzer.profitability_depth_analysis(data)
        assert result.gross_margin is None

    def test_high_retention_bonus(self, analyzer):
        """PRR >= 0.40 => +0.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=400_000,
            gross_profit=600_000,
            operating_income=250_000,
            net_income=300_000,
        )
        result = analyzer.profitability_depth_analysis(data)
        # PRR = 300k/600k = 0.50 >= 0.40 => +0.5
        assert result.profit_retention_ratio >= 0.40
