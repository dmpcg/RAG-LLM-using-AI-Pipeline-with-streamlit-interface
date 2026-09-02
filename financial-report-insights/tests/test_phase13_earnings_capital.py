"""Phase 13 Tests: Earnings Quality & Capital Efficiency.

Tests for earnings_quality(), capital_efficiency() and related dataclasses.
"""

import pytest

from financial_analyzer import (
    CapitalEfficiencyResult,
    CharlieAnalyzer,
    EarningsQualityResult,
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


class TestEarningsQualityDataclass:
    def test_defaults(self):
        r = EarningsQualityResult()
        assert r.quality_score == 0.0
        assert r.quality_grade == ""
        assert r.indicators == []

    def test_fields(self):
        r = EarningsQualityResult(quality_grade="High", quality_score=8.5)
        assert r.quality_grade == "High"
        assert r.quality_score == 8.5


class TestCapitalEfficiencyDataclass:
    def test_defaults(self):
        r = CapitalEfficiencyResult()
        assert r.roic is None
        assert r.wacc_estimate == 0.10
        assert r.summary == ""

    def test_fields(self):
        r = CapitalEfficiencyResult(roic=0.15, eva=50_000)
        assert r.roic == 0.15
        assert r.eva == 50_000


# ===== EARNINGS QUALITY TESTS =====


class TestEarningsQuality:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.earnings_quality(sample_data)
        assert isinstance(result, EarningsQualityResult)

    def test_score_range(self, analyzer, sample_data):
        result = analyzer.earnings_quality(sample_data)
        assert 0 <= result.quality_score <= 10

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.earnings_quality(sample_data)
        assert result.quality_grade in ["High", "Moderate", "Low"]

    def test_accrual_ratio_computed(self, analyzer, sample_data):
        result = analyzer.earnings_quality(sample_data)
        # accrual_ratio = (150k - 220k) / 2M = -0.035
        assert result.accrual_ratio is not None
        assert abs(result.accrual_ratio - (-0.035)) < 0.001

    def test_cash_to_income_computed(self, analyzer, sample_data):
        result = analyzer.earnings_quality(sample_data)
        # cash_to_income = 220k / 150k ≈ 1.467
        assert result.cash_to_income_ratio is not None
        assert abs(result.cash_to_income_ratio - 1.4667) < 0.01

    def test_high_quality_company(self, analyzer):
        """Company with strong cash backing gets High quality."""
        data = FinancialData(
            net_income=100_000,
            operating_cash_flow=150_000,
            total_assets=1_000_000,
        )
        result = analyzer.earnings_quality(data)
        assert result.quality_grade == "High"

    def test_low_quality_company(self, analyzer):
        """Company with cash flow well below income gets Low quality."""
        data = FinancialData(
            net_income=200_000,
            operating_cash_flow=30_000,
            total_assets=1_000_000,
        )
        result = analyzer.earnings_quality(data)
        assert result.quality_score < 5.0

    def test_negative_ocf(self, analyzer):
        """Negative operating cash flow should lower score."""
        data = FinancialData(
            net_income=50_000,
            operating_cash_flow=-20_000,
            total_assets=500_000,
        )
        result = analyzer.earnings_quality(data)
        assert result.quality_score < 5.0

    def test_negative_ni_positive_ocf(self, analyzer):
        """Positive OCF despite loss should be noted."""
        data = FinancialData(
            net_income=-10_000,
            operating_cash_flow=50_000,
            total_assets=500_000,
        )
        result = analyzer.earnings_quality(data)
        # Should get credit for positive OCF
        assert any("Positive cash flow despite net loss" in i for i in result.indicators)

    def test_indicators_present(self, analyzer, sample_data):
        result = analyzer.earnings_quality(sample_data)
        assert len(result.indicators) > 0

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.earnings_quality(sample_data)
        assert "Earnings Quality" in result.summary

    def test_empty_data(self, analyzer):
        result = analyzer.earnings_quality(FinancialData())
        assert result.quality_grade == "N/A"
        assert "Insufficient" in result.summary

    def test_only_net_income(self, analyzer):
        """With only NI, no OCF — limited but not N/A."""
        data = FinancialData(net_income=100_000)
        result = analyzer.earnings_quality(data)
        # Should still return something (score at midpoint)
        assert result.quality_grade != "N/A"

    def test_high_accrual_penalty(self, analyzer):
        """High accrual ratio should penalize score."""
        data = FinancialData(
            net_income=200_000,
            operating_cash_flow=10_000,  # very low OCF
            total_assets=1_000_000,
        )
        result = analyzer.earnings_quality(data)
        # accrual = (200k - 10k) / 1M = 0.19 -> high accrual penalty
        assert result.accrual_ratio > 0.15
        assert result.quality_score < 5.0

    def test_zero_total_assets(self, analyzer):
        """Zero total assets should still work (skip accrual ratio)."""
        data = FinancialData(
            net_income=100_000,
            operating_cash_flow=120_000,
            total_assets=0,
        )
        result = analyzer.earnings_quality(data)
        assert result.accrual_ratio is None
        assert result.quality_grade != "N/A"

    def test_score_clamped_at_ten(self, analyzer):
        """Score should never exceed 10."""
        data = FinancialData(
            net_income=10_000,
            operating_cash_flow=50_000,
            total_assets=10_000_000,
        )
        result = analyzer.earnings_quality(data)
        assert result.quality_score <= 10.0


# ===== CAPITAL EFFICIENCY TESTS =====


class TestCapitalEfficiency:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        assert isinstance(result, CapitalEfficiencyResult)

    def test_invested_capital(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        # equity 1.2M + debt 400k = 1.6M
        assert result.invested_capital == 1_600_000

    def test_nopat_computed(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        # EBIT=200k, tax_rate estimated from NI/EBIT: 1-(150/200)=0.25
        # NOPAT = 200k * (1 - 0.25) = 150k
        assert result.nopat is not None
        assert abs(result.nopat - 150_000) < 1

    def test_roic_computed(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        # ROIC = 150k / 1.6M = 0.09375
        assert result.roic is not None
        assert abs(result.roic - 0.09375) < 0.001

    def test_eva_computed(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        # EVA = 150k - (0.10 * 1.6M) = 150k - 160k = -10k
        assert result.eva is not None
        assert abs(result.eva - (-10_000)) < 1

    def test_capital_turnover(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        # 1M / 1.6M = 0.625
        assert result.capital_turnover is not None
        assert abs(result.capital_turnover - 0.625) < 0.001

    def test_reinvestment_rate(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        # capex 80k / nopat 150k ≈ 0.5333
        assert result.reinvestment_rate is not None
        assert abs(result.reinvestment_rate - (80_000 / 150_000)) < 0.01

    def test_custom_wacc(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data, cost_of_capital=0.08)
        assert result.wacc_estimate == 0.08
        # EVA = 150k - (0.08 * 1.6M) = 150k - 128k = 22k
        assert result.eva is not None
        assert abs(result.eva - 22_000) < 1

    def test_high_roic_creates_value(self, analyzer):
        """High ROIC above WACC creates positive EVA."""
        data = FinancialData(
            revenue=5_000_000,
            ebit=1_000_000,
            net_income=750_000,
            total_equity=2_000_000,
            total_debt=500_000,
            capex=200_000,
        )
        result = analyzer.capital_efficiency(data, cost_of_capital=0.10)
        assert result.eva > 0
        assert result.roic > 0.10

    def test_low_roic_destroys_value(self, analyzer):
        """Low ROIC below WACC creates negative EVA."""
        data = FinancialData(
            revenue=1_000_000,
            ebit=50_000,
            net_income=30_000,
            total_equity=2_000_000,
            total_debt=1_000_000,
        )
        result = analyzer.capital_efficiency(data, cost_of_capital=0.10)
        assert result.eva < 0

    def test_empty_data(self, analyzer):
        result = analyzer.capital_efficiency(FinancialData())
        assert result.roic is None
        assert "Insufficient" in result.summary

    def test_no_ebit(self, analyzer):
        """Without EBIT, NOPAT is None."""
        data = FinancialData(
            total_equity=500_000,
            total_debt=200_000,
            revenue=1_000_000,
        )
        result = analyzer.capital_efficiency(data)
        assert result.nopat is None
        assert result.roic is None

    def test_uses_operating_income_fallback(self, analyzer):
        """Falls back to operating_income when ebit is None."""
        data = FinancialData(
            operating_income=300_000,
            net_income=225_000,
            total_equity=1_000_000,
            total_debt=500_000,
            revenue=2_000_000,
        )
        result = analyzer.capital_efficiency(data)
        assert result.nopat is not None
        assert result.roic is not None

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.capital_efficiency(sample_data)
        assert "ROIC" in result.summary

    def test_negative_ebit_tax_rate(self, analyzer):
        """Negative EBIT shouldn't produce unreasonable tax rate."""
        data = FinancialData(
            ebit=-50_000,
            net_income=-80_000,
            total_equity=500_000,
            total_debt=200_000,
        )
        result = analyzer.capital_efficiency(data)
        # NOPAT should still be computed with default tax rate
        assert result.nopat is not None

    def test_zero_equity_zero_debt(self, analyzer):
        """Zero invested capital returns insufficient data."""
        data = FinancialData(
            revenue=1_000_000,
            ebit=100_000,
            total_equity=0,
            total_debt=0,
        )
        result = analyzer.capital_efficiency(data)
        assert "Insufficient" in result.summary


# ===== EDGE CASES =====


class TestPhase13EdgeCases:
    def test_earnings_quality_very_negative_ocf(self, analyzer):
        data = FinancialData(
            net_income=100_000,
            operating_cash_flow=-200_000,
            total_assets=1_000_000,
        )
        result = analyzer.earnings_quality(data)
        assert result.quality_score >= 0

    def test_capital_efficiency_large_debt(self, analyzer):
        data = FinancialData(
            revenue=500_000,
            ebit=50_000,
            net_income=20_000,
            total_equity=100_000,
            total_debt=2_000_000,
        )
        result = analyzer.capital_efficiency(data, cost_of_capital=0.08)
        assert result.invested_capital == 2_100_000
        assert result.eva < 0  # huge capital base with small returns

    def test_earnings_quality_score_clamped_floor(self, analyzer):
        """Score should never go below 0."""
        data = FinancialData(
            net_income=500_000,
            operating_cash_flow=-500_000,
            total_assets=500_000,
        )
        result = analyzer.earnings_quality(data)
        assert result.quality_score >= 0.0

    def test_capital_efficiency_implied_tax_capped(self, analyzer):
        """Implied tax rate beyond 60% falls back to default 25%."""
        data = FinancialData(
            ebit=100_000,
            net_income=10_000,  # implied tax = 90% -> use default 25%
            total_equity=500_000,
            total_debt=200_000,
        )
        result = analyzer.capital_efficiency(data)
        # With 25% default: NOPAT = 100k * 0.75 = 75k
        assert abs(result.nopat - 75_000) < 1
