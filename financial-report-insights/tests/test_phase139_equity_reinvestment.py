"""Phase 139 Tests: Equity Reinvestment Analysis.

Tests for equity_reinvestment_analysis() and EquityReinvestmentResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    EquityReinvestmentResult,
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


class TestEquityReinvestmentDataclass:
    def test_defaults(self):
        r = EquityReinvestmentResult()
        assert r.retention_ratio is None
        assert r.reinvestment_rate is None
        assert r.equity_growth_proxy is None
        assert r.plowback_to_assets is None
        assert r.internal_growth_rate is None
        assert r.dividend_coverage is None
        assert r.er_score == 0.0
        assert r.er_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = EquityReinvestmentResult(retention_ratio=0.75, er_grade="Good")
        assert r.retention_ratio == 0.75
        assert r.er_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestEquityReinvestmentAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert isinstance(result, EquityReinvestmentResult)

    def test_retention_ratio(self, analyzer, sample_data):
        """RR = (150k - 40k) / 150k = 0.733."""
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert result.retention_ratio == pytest.approx(0.733, abs=0.01)

    def test_reinvestment_rate(self, analyzer, sample_data):
        """ReinvRate = 80k / 150k = 0.533."""
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert result.reinvestment_rate == pytest.approx(0.533, abs=0.01)

    def test_equity_growth_proxy(self, analyzer, sample_data):
        """EGP = 600k / 1.2M = 0.50."""
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert result.equity_growth_proxy == pytest.approx(0.50, abs=0.01)

    def test_plowback_to_assets(self, analyzer, sample_data):
        """PBA = (150k - 40k) / 2M = 0.055."""
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert result.plowback_to_assets == pytest.approx(0.055, abs=0.01)

    def test_dividend_coverage(self, analyzer, sample_data):
        """DC = 150k / 40k = 3.75."""
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert result.dividend_coverage == pytest.approx(3.75, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert result.er_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert "Equity Reinvestment" in result.summary


# ===== SCORING TESTS =====


class TestEquityReinvestmentScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """RR=0.733 => base 7.0. DC=3.75 >=3.0 => +0.5. EGP=0.50 >=0.50 => +0.5. Score=8.0."""
        result = analyzer.equity_reinvestment_analysis(sample_data)
        assert result.er_score == pytest.approx(8.0, abs=0.5)
        assert result.er_grade == "Excellent"

    def test_excellent_reinvestment(self, analyzer):
        """RR >= 0.90 => base 10."""
        data = FinancialData(
            net_income=500_000,
            dividends_paid=20_000,
            capex=200_000,
            retained_earnings=1_500_000,
            total_equity=2_000_000,
            total_assets=3_000_000,
        )
        # RR=(500k-20k)/500k=0.96 => 10. DC=500k/20k=25 >=3 => +0.5. EGP=0.75 >=0.50 => +0.5. Capped 10.
        result = analyzer.equity_reinvestment_analysis(data)
        assert result.er_score >= 10.0
        assert result.er_grade == "Excellent"

    def test_weak_reinvestment(self, analyzer):
        """RR < 0.30 => base 1.0."""
        data = FinancialData(
            net_income=100_000,
            dividends_paid=80_000,
            capex=10_000,
            retained_earnings=50_000,
            total_equity=500_000,
            total_assets=2_000_000,
        )
        # RR=(100k-80k)/100k=0.20 <0.30 => 1.0. DC=100k/80k=1.25 (neither). EGP=0.10 <0.20 => -0.5. Score=0.5.
        result = analyzer.equity_reinvestment_analysis(data)
        assert result.er_score <= 1.0
        assert result.er_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase139EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.equity_reinvestment_analysis(FinancialData())
        assert isinstance(result, EquityReinvestmentResult)
        assert result.er_score == 0.0

    def test_no_dividends(self, analyzer):
        """Div=None => DC=None, but RR still needs special handling."""
        data = FinancialData(
            net_income=150_000,
            capex=80_000,
            retained_earnings=600_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        result = analyzer.equity_reinvestment_analysis(data)
        assert result.dividend_coverage is None
        assert result.reinvestment_rate is not None

    def test_no_net_income(self, analyzer):
        """NI=None => RR=None, score 0."""
        data = FinancialData(
            dividends_paid=40_000,
            retained_earnings=600_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        result = analyzer.equity_reinvestment_analysis(data)
        assert result.retention_ratio is None
        assert result.er_score == 0.0

    def test_no_retained_earnings(self, analyzer):
        """RE=None => EGP=None, IGR=None."""
        data = FinancialData(
            net_income=150_000,
            dividends_paid=40_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        result = analyzer.equity_reinvestment_analysis(data)
        assert result.equity_growth_proxy is None
        assert result.internal_growth_rate is None
        assert result.retention_ratio is not None
