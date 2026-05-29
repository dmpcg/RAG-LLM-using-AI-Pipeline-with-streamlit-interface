"""Phase 24 Tests: Valuation Indicators.

Tests for valuation_indicators() and ValuationIndicatorsResult dataclass.
"""

import math

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ValuationIndicatorsResult,
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


class TestValuationIndicatorsDataclass:
    def test_defaults(self):
        r = ValuationIndicatorsResult()
        assert r.earnings_yield is None
        assert r.ev_to_ebitda is None
        assert r.valuation_score == 0.0
        assert r.valuation_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ValuationIndicatorsResult(
            ev_to_ebitda=5.2,
            earnings_yield=0.125,
            valuation_grade="Undervalued",
        )
        assert r.ev_to_ebitda == 5.2
        assert r.earnings_yield == 0.125
        assert r.valuation_grade == "Undervalued"


# ===== CORE COMPUTATION TESTS =====


class TestValuationIndicators:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.valuation_indicators(sample_data)
        assert isinstance(result, ValuationIndicatorsResult)

    def test_earnings_yield(self, analyzer, sample_data):
        """Earnings yield = NI / Equity = 150k / 1.2M = 12.5%."""
        result = analyzer.valuation_indicators(sample_data)
        assert result.earnings_yield == pytest.approx(150_000 / 1_200_000, rel=0.01)

    def test_book_value_proxy(self, analyzer, sample_data):
        """BV proxy = Equity = 1.2M."""
        result = analyzer.valuation_indicators(sample_data)
        assert result.book_value_per_share_proxy == pytest.approx(1_200_000, rel=0.01)

    def test_ev_proxy(self, analyzer, sample_data):
        """EV = Equity + Debt - Cash = 1.2M + 400k - 300k = 1.3M."""
        result = analyzer.valuation_indicators(sample_data)
        cash = 500_000 - 200_000
        expected = 1_200_000 + 400_000 - cash
        assert result.ev_proxy == pytest.approx(expected, rel=0.01)

    def test_ev_to_ebitda(self, analyzer, sample_data):
        """EV/EBITDA = 1.3M / 250k = 5.2x."""
        result = analyzer.valuation_indicators(sample_data)
        ev = 1_200_000 + 400_000 - 300_000
        expected = ev / 250_000
        assert result.ev_to_ebitda == pytest.approx(expected, rel=0.01)

    def test_ev_to_revenue(self, analyzer, sample_data):
        """EV/Revenue = 1.3M / 1M = 1.3x."""
        result = analyzer.valuation_indicators(sample_data)
        expected = 1_300_000 / 1_000_000
        assert result.ev_to_revenue == pytest.approx(expected, rel=0.01)

    def test_ev_to_ebit(self, analyzer, sample_data):
        """EV/EBIT = 1.3M / 200k = 6.5x."""
        result = analyzer.valuation_indicators(sample_data)
        expected = 1_300_000 / 200_000
        assert result.ev_to_ebit == pytest.approx(expected, rel=0.01)

    def test_price_to_book_proxy(self, analyzer, sample_data):
        """P/B proxy = EV / Equity = 1.3M / 1.2M = 1.083."""
        result = analyzer.valuation_indicators(sample_data)
        expected = 1_300_000 / 1_200_000
        assert result.price_to_book_proxy == pytest.approx(expected, rel=0.01)

    def test_roic(self, analyzer, sample_data):
        """ROIC = NOPAT / IC = 150k / 1.6M = 9.375%."""
        result = analyzer.valuation_indicators(sample_data)
        nopat = 200_000 * (1 - 0.25)
        ic = 1_200_000 + 400_000
        expected = nopat / ic
        assert result.return_on_invested_capital == pytest.approx(expected, rel=0.01)

    def test_graham_number(self, analyzer, sample_data):
        """Graham = sqrt(22.5 * NI * BV)."""
        result = analyzer.valuation_indicators(sample_data)
        expected = math.sqrt(22.5 * 150_000 * 1_200_000)
        assert result.graham_number_proxy == pytest.approx(expected, rel=0.01)

    def test_intrinsic_value(self, analyzer, sample_data):
        """Intrinsic = NI / 0.10 = 150k / 0.10 = 1.5M."""
        result = analyzer.valuation_indicators(sample_data)
        assert result.intrinsic_value_proxy == pytest.approx(1_500_000, rel=0.01)

    def test_margin_of_safety(self, analyzer, sample_data):
        """MoS = (1.5M - 1.2M) / 1.5M = 20%."""
        result = analyzer.valuation_indicators(sample_data)
        expected = (1_500_000 - 1_200_000) / 1_500_000
        assert result.margin_of_safety_pct == pytest.approx(expected, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.valuation_indicators(sample_data)
        assert result.valuation_grade in ["Undervalued", "Fair Value", "Fully Valued", "Overvalued"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.valuation_indicators(sample_data)
        assert "Valuation" in result.summary


# ===== SCORING TESTS =====


class TestValuationScoring:
    def test_undervalued(self, analyzer):
        """Low multiples, high yields = Undervalued."""
        data = FinancialData(
            revenue=5_000_000,
            operating_income=2_000_000,
            ebit=2_000_000,
            ebitda=2_500_000,
            net_income=1_500_000,
            total_assets=4_000_000,
            total_equity=3_000_000,
            total_debt=200_000,
            interest_expense=10_000,
            current_assets=1_000_000,
            current_liabilities=200_000,
        )
        result = analyzer.valuation_indicators(data)
        # EV = 3M+200k-800k=2.4M, EV/EBITDA=0.96 <6: +1.5
        # earnings_yield=1.5M/3M=0.50 >=0.15: +2.0
        # ROIC=1.5M/3.2M=0.469 >=0.15: +1.0
        # mos=(15M-3M)/15M=0.8 >=0.30: +0.5
        assert result.valuation_grade == "Undervalued"
        assert result.valuation_score >= 8.0

    def test_overvalued(self, analyzer):
        """Negative NI = poor valuation."""
        data = FinancialData(
            revenue=500_000,
            operating_income=-100_000,
            ebit=-100_000,
            ebitda=50_000,
            net_income=-200_000,
            total_assets=5_000_000,
            total_equity=2_000_000,
            total_debt=2_500_000,
            interest_expense=200_000,
            current_assets=300_000,
            current_liabilities=500_000,
        )
        result = analyzer.valuation_indicators(data)
        # earnings_yield=-200k/2M=-0.10 <0: -1.5
        # ROIC=-75k/4.5M<0: -1.0
        # No graham (NI<0), no intrinsic (NI<0)
        assert result.valuation_grade in ["Overvalued", "Fully Valued"]

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            revenue=10_000_000,
            ebit=5_000_000,
            ebitda=6_000_000,
            net_income=4_000_000,
            total_assets=5_000_000,
            total_equity=4_500_000,
            total_debt=50_000,
            interest_expense=2_000,
            current_assets=2_000_000,
            current_liabilities=200_000,
        )
        result = analyzer.valuation_indicators(data)
        assert result.valuation_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            revenue=100_000,
            ebit=-500_000,
            ebitda=-400_000,
            net_income=-600_000,
            total_assets=5_000_000,
            total_equity=500_000,
            total_debt=4_000_000,
            interest_expense=400_000,
        )
        result = analyzer.valuation_indicators(data)
        assert result.valuation_score >= 0.0


# ===== EDGE CASES =====


class TestPhase24EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.valuation_indicators(FinancialData())
        assert isinstance(result, ValuationIndicatorsResult)
        assert result.earnings_yield is None
        assert result.ev_proxy is None

    def test_zero_equity(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
            total_equity=0,
        )
        result = analyzer.valuation_indicators(data)
        assert result.earnings_yield is None
        assert result.book_value_per_share_proxy is None

    def test_no_ebitda(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_equity=500_000,
        )
        result = analyzer.valuation_indicators(data)
        assert result.ev_to_ebitda is None

    def test_negative_ni(self, analyzer):
        """Negative NI: no graham, no intrinsic."""
        data = FinancialData(
            revenue=1_000_000,
            ebit=50_000,
            ebitda=100_000,
            net_income=-50_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
            total_debt=500_000,
        )
        result = analyzer.valuation_indicators(data)
        assert result.graham_number_proxy is None
        assert result.intrinsic_value_proxy is None
        assert result.margin_of_safety_pct is None

    def test_cash_field_used(self, analyzer):
        """When cash field is provided, use it for EV."""
        data = FinancialData(
            revenue=1_000_000,
            ebitda=200_000,
            net_income=100_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
            total_debt=500_000,
            cash=300_000,
        )
        result = analyzer.valuation_indicators(data)
        expected_ev = 1_000_000 + 500_000 - 300_000
        assert result.ev_proxy == pytest.approx(expected_ev, rel=0.01)

    def test_zero_revenue(self, analyzer):
        data = FinancialData(
            revenue=0,
            total_equity=500_000,
            total_debt=200_000,
        )
        result = analyzer.valuation_indicators(data)
        assert result.ev_to_revenue is None

    def test_sample_data_grade(self, analyzer, sample_data):
        """Sample data should produce Undervalued grade (score=8.0)."""
        result = analyzer.valuation_indicators(sample_data)
        # EV/EBITDA=5.2 <6.0: +1.5 -> 6.5
        # earnings_yield=0.125 >=0.10: +1.0 -> 7.5
        # ROIC=0.09375 >=0.08: +0.5 -> 8.0
        # mos_pct=0.20 <0.30: +0.0 -> 8.0
        assert result.valuation_score == pytest.approx(8.0, abs=0.1)
        assert result.valuation_grade == "Undervalued"
