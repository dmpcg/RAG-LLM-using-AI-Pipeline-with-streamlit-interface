"""Phase 22 Tests: Risk-Adjusted Performance Metrics.

Tests for risk_adjusted_performance() and RiskAdjustedResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    RiskAdjustedResult,
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


class TestRiskAdjustedDataclass:
    def test_defaults(self):
        r = RiskAdjustedResult()
        assert r.return_on_risk is None
        assert r.risk_adjusted_roe is None
        assert r.risk_adjusted_roa is None
        assert r.risk_score == 0.0
        assert r.risk_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = RiskAdjustedResult(
            return_on_risk=0.075,
            margin_of_safety=5.67,
            risk_grade="Superior",
        )
        assert r.return_on_risk == 0.075
        assert r.margin_of_safety == 5.67
        assert r.risk_grade == "Superior"


# ===== CORE COMPUTATION TESTS =====


class TestRiskAdjustedPerformance:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.risk_adjusted_performance(sample_data)
        assert isinstance(result, RiskAdjustedResult)

    def test_return_on_risk(self, analyzer, sample_data):
        """Return on risk = ROE / leverage = 12.5% / 1.667 = 7.5%."""
        result = analyzer.risk_adjusted_performance(sample_data)
        roe = 150_000 / 1_200_000
        leverage = 2_000_000 / 1_200_000
        expected = roe / leverage
        assert result.return_on_risk == pytest.approx(expected, rel=0.01)

    def test_risk_adjusted_roe(self, analyzer, sample_data):
        """Risk-adj ROE = ROE × (Equity/TA) = 12.5% × 0.6 = 7.5%."""
        result = analyzer.risk_adjusted_performance(sample_data)
        expected = (150_000 / 1_200_000) * (1_200_000 / 2_000_000)
        assert result.risk_adjusted_roe == pytest.approx(expected, rel=0.01)

    def test_risk_adjusted_roa(self, analyzer, sample_data):
        """Risk-adj ROA = ROA × (1 - Debt/TA) = 7.5% × 0.8 = 6.0%."""
        result = analyzer.risk_adjusted_performance(sample_data)
        roa = 150_000 / 2_000_000
        debt_ratio = 400_000 / 2_000_000
        expected = roa * (1 - debt_ratio)
        assert result.risk_adjusted_roa == pytest.approx(expected, rel=0.01)

    def test_debt_adjusted_return(self, analyzer, sample_data):
        """Debt-adj return = NI / (Equity + Debt) = 150k / 1.6M = 9.375%."""
        result = analyzer.risk_adjusted_performance(sample_data)
        expected = 150_000 / (1_200_000 + 400_000)
        assert result.debt_adjusted_return == pytest.approx(expected, rel=0.01)

    def test_volatility_proxy(self, analyzer, sample_data):
        """Vol proxy = |op_margin - net_margin| = |0.20 - 0.15| = 0.05."""
        result = analyzer.risk_adjusted_performance(sample_data)
        expected = abs(200_000 / 1_000_000 - 150_000 / 1_000_000)
        assert result.volatility_proxy == pytest.approx(expected, rel=0.01)

    def test_downside_risk(self, analyzer, sample_data):
        """Downside = Debt / EBITDA = 400k / 250k = 1.6."""
        result = analyzer.risk_adjusted_performance(sample_data)
        expected = 400_000 / 250_000
        assert result.downside_risk == pytest.approx(expected, rel=0.01)

    def test_margin_of_safety(self, analyzer, sample_data):
        """Safety margin = (EBIT - Interest) / Interest = (200k-30k)/30k = 5.667."""
        result = analyzer.risk_adjusted_performance(sample_data)
        expected = (200_000 - 30_000) / 30_000
        assert result.margin_of_safety == pytest.approx(expected, rel=0.01)

    def test_return_per_unit_risk(self, analyzer, sample_data):
        """Return/risk = ROE / DE_ratio = 12.5% / 0.333 = 0.375."""
        result = analyzer.risk_adjusted_performance(sample_data)
        roe = 150_000 / 1_200_000
        de_ratio = 400_000 / 1_200_000
        expected = roe / de_ratio
        assert result.return_per_unit_risk == pytest.approx(expected, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.risk_adjusted_performance(sample_data)
        assert result.risk_grade in ["Superior", "Favorable", "Neutral", "Elevated"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.risk_adjusted_performance(sample_data)
        assert "Risk-Adjusted" in result.summary


# ===== SCORING TESTS =====


class TestRiskAdjustedScoring:
    def test_superior_risk(self, analyzer):
        """High return on risk, high safety, low downside = Superior."""
        data = FinancialData(
            revenue=5_000_000,
            operating_income=2_000_000,
            ebit=2_000_000,
            ebitda=2_500_000,
            net_income=1_500_000,
            total_assets=4_000_000,
            total_equity=3_500_000,
            total_debt=200_000,
            interest_expense=10_000,
        )
        result = analyzer.risk_adjusted_performance(data)
        # return_on_risk = (1.5M/3.5M) / (4M/3.5M) = 0.4286/1.1429 = 0.375 -> >=0.10: +1.5
        # margin_safety = (2M-10k)/10k = 199 -> >=5: +2.0
        # downside = 200k/2.5M = 0.08 -> <=1.0: +1.0
        # risk_adj_roe = 0.4286*(3.5M/4M) = 0.375 -> >=0.12: +1.0
        # Total: 5 + 1.5 + 2 + 1 + 1 = 10.5 -> capped 10.0
        assert result.risk_grade == "Superior"
        assert result.risk_score >= 8.0

    def test_elevated_risk(self, analyzer):
        """Negative returns, high leverage = Elevated."""
        data = FinancialData(
            revenue=500_000,
            operating_income=-100_000,
            ebit=-100_000,
            ebitda=50_000,
            net_income=-200_000,
            total_assets=5_000_000,
            total_equity=500_000,
            total_debt=4_000_000,
            interest_expense=300_000,
        )
        result = analyzer.risk_adjusted_performance(data)
        # return_on_risk = (-0.4)/(10) = -0.04 -> <0: -1.5
        # margin_safety = (-100k-300k)/300k = -1.33 -> <1.0: -1.5
        # downside = 4M/50k = 80 -> >5: -1.5
        # risk_adj_roe = -0.4*(500k/5M) = -0.04 -> <0: -1.0
        # Total: 5 - 1.5 - 1.5 - 1.5 - 1.0 = -0.5 -> floored 0.0
        assert result.risk_grade == "Elevated"
        assert result.risk_score < 4.0

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            revenue=10_000_000,
            operating_income=5_000_000,
            ebit=5_000_000,
            ebitda=6_000_000,
            net_income=4_000_000,
            total_assets=5_000_000,
            total_equity=4_500_000,
            total_debt=100_000,
            interest_expense=5_000,
        )
        result = analyzer.risk_adjusted_performance(data)
        assert result.risk_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            revenue=100_000,
            operating_income=-500_000,
            ebit=-500_000,
            ebitda=-400_000,
            net_income=-600_000,
            total_assets=5_000_000,
            total_equity=300_000,
            total_debt=4_500_000,
            interest_expense=500_000,
        )
        result = analyzer.risk_adjusted_performance(data)
        assert result.risk_score >= 0.0


# ===== EDGE CASES =====


class TestPhase22EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.risk_adjusted_performance(FinancialData())
        assert isinstance(result, RiskAdjustedResult)
        assert result.return_on_risk is None
        assert result.risk_adjusted_roe is None
        assert result.risk_adjusted_roa is None

    def test_zero_equity(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
            total_equity=0,
        )
        result = analyzer.risk_adjusted_performance(data)
        assert result.return_on_risk is None
        assert result.risk_adjusted_roe is None
        assert result.return_per_unit_risk is None

    def test_no_debt(self, analyzer):
        """Company with no debt: DE ratio is 0, return_per_risk is None."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
            ebit=200_000,
            ebitda=250_000,
            net_income=150_000,
            total_assets=1_500_000,
            total_equity=1_500_000,
            total_debt=0,
        )
        result = analyzer.risk_adjusted_performance(data)
        # DE ratio = 0, so return_per_risk is None (can't divide by 0)
        assert result.return_per_unit_risk is None
        # But debt_adj_return = NI / equity (since debt=0)
        assert result.debt_adjusted_return == pytest.approx(150_000 / 1_500_000, rel=0.01)

    def test_no_interest(self, analyzer):
        """Without interest expense, margin of safety is None."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
            ebit=200_000,
            net_income=150_000,
            total_assets=2_000_000,
            total_equity=1_200_000,
            total_debt=400_000,
            interest_expense=0,
        )
        result = analyzer.risk_adjusted_performance(data)
        assert result.margin_of_safety is None

    def test_no_ebitda(self, analyzer):
        """Without EBITDA, downside_risk is None."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
            total_debt=500_000,
        )
        result = analyzer.risk_adjusted_performance(data)
        assert result.downside_risk is None

    def test_zero_revenue(self, analyzer):
        """Zero revenue: vol proxy is None (no margin calc)."""
        data = FinancialData(
            revenue=0,
            total_assets=1_000_000,
            total_equity=500_000,
        )
        result = analyzer.risk_adjusted_performance(data)
        assert result.volatility_proxy is None

    def test_negative_ebitda(self, analyzer):
        """Negative EBITDA: downside_risk is None (EBITDA not > 0)."""
        data = FinancialData(
            revenue=500_000,
            ebit=-50_000,
            ebitda=-20_000,
            net_income=-100_000,
            total_assets=2_000_000,
            total_equity=800_000,
            total_debt=800_000,
            interest_expense=60_000,
        )
        result = analyzer.risk_adjusted_performance(data)
        assert result.downside_risk is None
        # Margin of safety still computed: (ebit - interest) / interest
        assert result.margin_of_safety == pytest.approx((-50_000 - 60_000) / 60_000, rel=0.01)

    def test_sample_data_grade(self, analyzer, sample_data):
        """Sample data should produce Superior grade (score=8.0)."""
        result = analyzer.risk_adjusted_performance(sample_data)
        # Score breakdown:
        # return_on_risk=0.075 -> >=0.05: +0.5 -> 5.5
        # margin_safety=5.667 -> >=5.0: +2.0 -> 7.5
        # downside=1.6 -> <=3.0: +0.0 -> 7.5
        # risk_adj_roe=0.075 -> >=0.06: +0.5 -> 8.0
        assert result.risk_score == pytest.approx(8.0, abs=0.1)
        assert result.risk_grade == "Superior"
