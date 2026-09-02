"""Phase 78 Tests: Profit Retention Analysis.

Tests for profit_retention_analysis() and ProfitRetentionResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ProfitRetentionResult,
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


class TestProfitRetentionDataclass:
    def test_defaults(self):
        r = ProfitRetentionResult()
        assert r.retention_ratio is None
        assert r.payout_ratio is None
        assert r.re_to_equity is None
        assert r.sustainable_growth_rate is None
        assert r.internal_growth_rate is None
        assert r.plowback_amount is None
        assert r.pr_score == 0.0
        assert r.pr_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ProfitRetentionResult(retention_ratio=0.75, pr_grade="Good")
        assert r.retention_ratio == 0.75
        assert r.pr_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestProfitRetentionAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.profit_retention_analysis(sample_data)
        assert isinstance(result, ProfitRetentionResult)

    def test_retention_ratio(self, analyzer, sample_data):
        """(NI-Div)/NI = (150k-40k)/150k = 0.733."""
        result = analyzer.profit_retention_analysis(sample_data)
        assert result.retention_ratio == pytest.approx(0.733, abs=0.01)

    def test_payout_ratio(self, analyzer, sample_data):
        """Div/NI = 40k/150k = 0.267."""
        result = analyzer.profit_retention_analysis(sample_data)
        assert result.payout_ratio == pytest.approx(0.267, abs=0.01)

    def test_re_to_equity(self, analyzer, sample_data):
        """RE/TE = 600k/1.2M = 0.50."""
        result = analyzer.profit_retention_analysis(sample_data)
        assert result.re_to_equity == pytest.approx(0.50, abs=0.01)

    def test_sustainable_growth_rate(self, analyzer, sample_data):
        """ROE * retention = (150k/1.2M) * 0.733 = 0.125 * 0.733 = 0.0917."""
        result = analyzer.profit_retention_analysis(sample_data)
        assert result.sustainable_growth_rate == pytest.approx(0.0917, abs=0.01)

    def test_internal_growth_rate(self, analyzer, sample_data):
        """ROA=150k/2M=0.075, rr=0.733, ROA*rr=0.055, IGR=0.055/(1-0.055)=0.0582."""
        result = analyzer.profit_retention_analysis(sample_data)
        assert result.internal_growth_rate == pytest.approx(0.058, abs=0.01)

    def test_plowback_amount(self, analyzer, sample_data):
        """NI - Div = 150k - 40k = 110k."""
        result = analyzer.profit_retention_analysis(sample_data)
        assert result.plowback_amount == pytest.approx(110_000, abs=100)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.profit_retention_analysis(sample_data)
        assert result.pr_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.profit_retention_analysis(sample_data)
        assert "Profit Retention" in result.summary


# ===== SCORING TESTS =====


class TestProfitRetentionScoring:
    def test_very_high_retention(self, analyzer):
        """RR >= 0.80 => base 10."""
        data = FinancialData(
            revenue=5_000_000,
            net_income=500_000,
            dividends_paid=50_000,  # RR = 0.90
            total_equity=2_000_000,
            total_assets=5_000_000,
            retained_earnings=1_500_000,  # RE/TE = 0.75
        )
        result = analyzer.profit_retention_analysis(data)
        # RR=0.90 (base 10) + RE/TE=0.75 (+0.5) + SGR=0.225 (+0.5) => 10 capped
        assert result.pr_score >= 10.0
        assert result.pr_grade == "Excellent"

    def test_moderate_retention(self, analyzer, sample_data):
        """RR=0.733 => base 8.5."""
        result = analyzer.profit_retention_analysis(sample_data)
        # RR=0.733 (base 8.5) + RE/TE=0.50 (no adj) + SGR=0.0917 (no adj) => 8.5
        assert result.pr_score >= 8.0

    def test_low_retention(self, analyzer):
        """RR ~ 0.40 => base 4.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            dividends_paid=60_000,  # RR = 0.40
            total_equity=500_000,
            total_assets=1_000_000,
            retained_earnings=30_000,  # RE/TE = 0.06
        )
        result = analyzer.profit_retention_analysis(data)
        # RR=0.40 (base 4.0) + RE/TE=0.06 (<0.10 => -0.5) + SGR=0.08 (no adj) => 3.5
        assert result.pr_score <= 4.5

    def test_very_low_retention(self, analyzer):
        """RR < 0.10 => base 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            dividends_paid=95_000,  # RR = 0.05
            total_equity=500_000,
            total_assets=1_000_000,
            retained_earnings=20_000,  # RE/TE = 0.04
        )
        result = analyzer.profit_retention_analysis(data)
        assert result.pr_score < 2.0
        assert result.pr_grade == "Weak"

    def test_re_equity_bonus(self, analyzer):
        """RE/TE >= 0.60 => +0.5."""
        data = FinancialData(
            revenue=2_000_000,
            net_income=300_000,
            dividends_paid=60_000,  # RR = 0.80
            total_equity=1_000_000,
            total_assets=2_000_000,
            retained_earnings=700_000,  # RE/TE = 0.70
        )
        result = analyzer.profit_retention_analysis(data)
        # RR=0.80 (base 10) + RE/TE=0.70 (+0.5) + SGR=0.24 (+0.5) => 10 capped
        assert result.pr_score >= 10.0

    def test_sgr_bonus(self, analyzer):
        """SGR >= 0.15 => +0.5."""
        data = FinancialData(
            revenue=2_000_000,
            net_income=400_000,
            dividends_paid=80_000,  # RR = 0.80
            total_equity=1_000_000,  # ROE = 0.40
            total_assets=3_000_000,
            retained_earnings=400_000,
        )
        result = analyzer.profit_retention_analysis(data)
        # SGR = 0.40 * 0.80 = 0.32 => +0.5
        assert result.pr_score >= 10.0


# ===== EDGE CASES =====


class TestPhase78EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.profit_retention_analysis(FinancialData())
        assert isinstance(result, ProfitRetentionResult)
        assert result.retention_ratio is None

    def test_no_net_income(self, analyzer):
        """NI=0 => empty result."""
        data = FinancialData(
            revenue=1_000_000,
            dividends_paid=50_000,
        )
        result = analyzer.profit_retention_analysis(data)
        assert result.pr_score == 0.0

    def test_no_dividends(self, analyzer):
        """No dividends => RR=1.0, payout=0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=200_000,
            total_equity=800_000,
            total_assets=1_500_000,
            retained_earnings=500_000,
        )
        result = analyzer.profit_retention_analysis(data)
        assert result.retention_ratio == pytest.approx(1.0, abs=0.01)
        assert result.payout_ratio == pytest.approx(0.0, abs=0.01)

    def test_no_equity(self, analyzer):
        """TE=0 => RE/TE and SGR are None."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            dividends_paid=20_000,
            total_assets=500_000,
        )
        result = analyzer.profit_retention_analysis(data)
        assert result.re_to_equity is None
        assert result.sustainable_growth_rate is None
