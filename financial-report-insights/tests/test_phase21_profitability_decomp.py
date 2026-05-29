"""Phase 21 Tests: Profitability Decomposition Analysis.

Tests for profitability_decomposition() and ProfitabilityDecompResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ProfitabilityDecompResult,
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


class TestProfitabilityDecompDataclass:
    def test_defaults(self):
        r = ProfitabilityDecompResult()
        assert r.roe is None
        assert r.roa is None
        assert r.roic is None
        assert r.profitability_score == 0.0
        assert r.profitability_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ProfitabilityDecompResult(
            roe=0.15,
            roic=0.12,
            profitability_grade="Strong",
        )
        assert r.roe == 0.15
        assert r.roic == 0.12
        assert r.profitability_grade == "Strong"


# ===== CORE COMPUTATION TESTS =====


class TestProfitabilityDecomp:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.profitability_decomposition(sample_data)
        assert isinstance(result, ProfitabilityDecompResult)

    def test_roe(self, analyzer, sample_data):
        """ROE = NI / Equity = 150k / 1.2M = 12.5%."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.roe == pytest.approx(150_000 / 1_200_000, rel=0.01)

    def test_roa(self, analyzer, sample_data):
        """ROA = NI / TA = 150k / 2M = 7.5%."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.roa == pytest.approx(150_000 / 2_000_000, rel=0.01)

    def test_invested_capital(self, analyzer, sample_data):
        """IC = Equity + Debt = 1.2M + 400k = 1.6M."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.invested_capital == pytest.approx(1_600_000, rel=0.01)

    def test_nopat(self, analyzer, sample_data):
        """NOPAT = EBIT × (1 - tax_rate) = 200k × (1-0.25) = 150k."""
        result = analyzer.profitability_decomposition(sample_data)
        expected_nopat = 200_000 * (1 - 0.25)  # default tax rate is 0.25
        assert result.nopat == pytest.approx(expected_nopat, rel=0.01)

    def test_roic(self, analyzer, sample_data):
        """ROIC = NOPAT / IC = 150k / 1.6M = 9.375%."""
        result = analyzer.profitability_decomposition(sample_data)
        expected_nopat = 200_000 * (1 - 0.25)
        expected_roic = expected_nopat / 1_600_000
        assert result.roic == pytest.approx(expected_roic, rel=0.01)

    def test_spread_computed(self, analyzer, sample_data):
        """Spread = ROIC - WACC proxy."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.spread is not None

    def test_economic_profit_computed(self, analyzer, sample_data):
        """EP = Spread × IC."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.economic_profit is not None
        if result.spread is not None and result.invested_capital is not None:
            expected = result.spread * result.invested_capital
            assert result.economic_profit == pytest.approx(expected, rel=0.01)

    def test_asset_turnover(self, analyzer, sample_data):
        """AT = Revenue / TA = 1M / 2M = 0.50."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.asset_turnover == pytest.approx(0.50, rel=0.01)

    def test_financial_leverage(self, analyzer, sample_data):
        """FL = TA / Equity = 2M / 1.2M = 1.667."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.financial_leverage == pytest.approx(2_000_000 / 1_200_000, rel=0.01)

    def test_capital_intensity(self, analyzer, sample_data):
        """CI = TA / Revenue = 2M / 1M = 2.0."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.capital_intensity == pytest.approx(2.0, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.profitability_decomposition(sample_data)
        assert result.profitability_grade in ["Elite", "Strong", "Adequate", "Poor"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.profitability_decomposition(sample_data)
        assert "Profitability" in result.summary


# ===== SCORING TESTS =====


class TestProfitabilityScoring:
    def test_elite_company(self, analyzer):
        """High ROE, high ROIC, positive spread = Elite."""
        data = FinancialData(
            revenue=5_000_000,
            operating_income=1_500_000,
            ebit=1_500_000,
            net_income=1_200_000,
            total_assets=4_000_000,
            total_equity=3_000_000,
            total_debt=500_000,
            interest_expense=25_000,
        )
        result = analyzer.profitability_decomposition(data)
        # ROE=40% (+2), ROIC high (+1.5), spread positive (+1), ROA=30% (+0.5) = 10
        assert result.profitability_grade == "Elite"

    def test_poor_company(self, analyzer):
        """Negative returns = Poor."""
        data = FinancialData(
            revenue=500_000,
            operating_income=-100_000,
            ebit=-100_000,
            net_income=-150_000,
            total_assets=3_000_000,
            total_equity=1_000_000,
            total_debt=1_500_000,
            interest_expense=100_000,
        )
        result = analyzer.profitability_decomposition(data)
        # ROE<0 (-2), ROIC<0 (-1.5), spread<-5% (-1), ROA<0 (-0.5) = 0
        assert result.profitability_grade == "Poor"

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            revenue=10_000_000,
            operating_income=4_000_000,
            ebit=4_000_000,
            net_income=3_000_000,
            total_assets=5_000_000,
            total_equity=4_000_000,
            total_debt=200_000,
            interest_expense=10_000,
        )
        result = analyzer.profitability_decomposition(data)
        assert result.profitability_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            revenue=100_000,
            operating_income=-200_000,
            ebit=-200_000,
            net_income=-300_000,
            total_assets=5_000_000,
            total_equity=500_000,
            total_debt=4_000_000,
            interest_expense=300_000,
        )
        result = analyzer.profitability_decomposition(data)
        assert result.profitability_score >= 0.0


# ===== EDGE CASES =====


class TestPhase21EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.profitability_decomposition(FinancialData())
        assert isinstance(result, ProfitabilityDecompResult)
        assert result.roe is None
        assert result.roa is None
        assert result.roic is None

    def test_zero_equity(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
            total_equity=0,
        )
        result = analyzer.profitability_decomposition(data)
        assert result.roe is None
        assert result.financial_leverage is None

    def test_no_debt(self, analyzer):
        """Company with no debt: IC = equity only."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
            ebit=200_000,
            net_income=150_000,
            total_assets=1_500_000,
            total_equity=1_500_000,
            total_debt=0,
        )
        result = analyzer.profitability_decomposition(data)
        assert result.invested_capital == pytest.approx(1_500_000, rel=0.01)
        assert result.roic is not None

    def test_no_ebit(self, analyzer):
        """Without EBIT, NOPAT is None."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
        )
        result = analyzer.profitability_decomposition(data)
        assert result.nopat is None
        assert result.roic is None

    def test_zero_revenue(self, analyzer):
        data = FinancialData(
            revenue=0,
            total_assets=1_000_000,
            total_equity=500_000,
        )
        result = analyzer.profitability_decomposition(data)
        assert result.asset_turnover is None
        assert result.capital_intensity is None

    def test_invested_capital_fallback(self, analyzer):
        """When equity is None but TA and CL available, IC = TA - CL."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
            ebit=200_000,
            total_assets=2_000_000,
            current_liabilities=300_000,
        )
        result = analyzer.profitability_decomposition(data)
        assert result.invested_capital == pytest.approx(1_700_000, rel=0.01)

    def test_tax_efficiency(self, analyzer, sample_data):
        """Tax efficiency is NI/EBT."""
        result = analyzer.profitability_decomposition(sample_data)
        assert result.tax_efficiency is not None
