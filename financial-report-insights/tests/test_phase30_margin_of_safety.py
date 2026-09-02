"""Phase 30 Tests: Margin of Safety Analysis.

Tests for margin_of_safety_analysis() and MarginOfSafetyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    MarginOfSafetyResult,
)


@pytest.fixture
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture
def sample_data():
    """Company with known market data for margin of safety."""
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
        shares_outstanding=100_000,
        share_price=10.0,
    )


# ===== DATACLASS TESTS =====


class TestMarginOfSafetyDataclass:
    def test_defaults(self):
        r = MarginOfSafetyResult()
        assert r.earnings_yield is None
        assert r.book_value_per_share is None
        assert r.price_to_book is None
        assert r.intrinsic_margin is None
        assert r.safety_score == 0.0
        assert r.safety_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = MarginOfSafetyResult(
            earnings_yield=0.12,
            price_to_book=0.8,
            safety_grade="Wide Margin",
        )
        assert r.earnings_yield == 0.12
        assert r.price_to_book == 0.8
        assert r.safety_grade == "Wide Margin"


# ===== CORE COMPUTATION TESTS =====


class TestMarginOfSafety:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert isinstance(result, MarginOfSafetyResult)

    def test_market_cap(self, analyzer, sample_data):
        """MCap = 100k shares * $10 = $1M."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.market_cap == pytest.approx(1_000_000)

    def test_earnings_yield(self, analyzer, sample_data):
        """Earnings yield = NI / MCap = 150k / 1M = 15%."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.earnings_yield == pytest.approx(0.15, rel=0.01)

    def test_book_value_per_share(self, analyzer, sample_data):
        """BV/share = Equity / Shares = 1.2M / 100k = $12."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.book_value_per_share == pytest.approx(12.0, rel=0.01)

    def test_price_to_book(self, analyzer, sample_data):
        """P/B = Price / BV_per_share = 10 / 12 = 0.833."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.price_to_book == pytest.approx(0.833, rel=0.01)

    def test_book_value_discount(self, analyzer, sample_data):
        """BV discount = 1 - P/B = 1 - 0.833 = 0.167 (trading below book)."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.book_value_discount == pytest.approx(0.167, rel=0.01)

    def test_intrinsic_value(self, analyzer, sample_data):
        """IV = NI / 0.10 = 150k / 0.10 = $1.5M."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.intrinsic_value_estimate == pytest.approx(1_500_000)

    def test_intrinsic_margin(self, analyzer, sample_data):
        """IV margin = (IV - MCap) / IV = (1.5M - 1M) / 1.5M = 33.3%."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.intrinsic_margin == pytest.approx(0.333, rel=0.01)

    def test_tangible_bv(self, analyzer, sample_data):
        """TBV = equity (approx) = 1.2M."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.tangible_bv == pytest.approx(1_200_000)

    def test_liquidation_value(self, analyzer, sample_data):
        """Liquidation = TBV * 0.70 = 1.2M * 0.70 = 840k."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.liquidation_value == pytest.approx(840_000)

    def test_ncav(self, analyzer, sample_data):
        """NCAV = CA - TL = 500k - 800k = -300k."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.net_current_asset_value == pytest.approx(-300_000)

    def test_ncav_per_share(self, analyzer, sample_data):
        """NCAV/share = -300k / 100k = -$3.00."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.ncav_per_share == pytest.approx(-3.0, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.safety_grade in ["Wide Margin", "Adequate", "Thin", "No Margin"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert "Margin of Safety" in result.summary


# ===== SCORING TESTS =====


class TestMarginOfSafetyScoring:
    def test_wide_margin(self, analyzer):
        """Very cheap: high earnings yield, big IV margin, below book, net-net."""
        data = FinancialData(
            net_income=500_000,
            total_equity=2_000_000,
            total_liabilities=300_000,
            current_assets=1_500_000,
            shares_outstanding=100_000,
            share_price=2.0,  # MCap=200k, EY=250%, IV=5M, P/B=0.1
        )
        result = analyzer.margin_of_safety_analysis(data)
        # EY=2.5 (>=0.15: +2.0 -> 7.0)
        # IV margin=(5M-200k)/5M=0.96 (>=0.50: +1.5 -> 8.5)
        # BV discount=1-0.1=0.9 (>0: +0.5 -> 9.0)
        # NCAV/share=(1.5M-300k)/100k=12.0 >= price 2.0 (+1.0 -> 10.0)
        assert result.safety_grade == "Wide Margin"
        assert result.safety_score >= 8.0

    def test_no_margin(self, analyzer):
        """Overvalued: negative earnings, trading well above book."""
        data = FinancialData(
            net_income=-100_000,
            total_equity=200_000,
            total_liabilities=1_000_000,
            current_assets=300_000,
            shares_outstanding=10_000,
            share_price=100.0,  # MCap=1M, P/B=50.0
        )
        result = analyzer.margin_of_safety_analysis(data)
        # EY = -100k/1M = -0.10 (<0: -1.5 -> 3.5)
        # IV: NI<0, no IV -> no IV margin adjustment
        # BV discount = 1-50 = -49 (<-1.0: -0.5 -> 3.0)
        # NCAV = 300k-1M = -700k, NCAV/share=-70 < price: no bonus
        assert result.safety_grade in ["No Margin", "Thin"]
        assert result.safety_score <= 4.0

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            net_income=10_000_000,
            total_equity=50_000_000,
            total_liabilities=1_000_000,
            current_assets=20_000_000,
            shares_outstanding=100_000,
            share_price=1.0,
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.safety_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            net_income=-5_000_000,
            total_equity=100_000,
            total_liabilities=10_000_000,
            current_assets=50_000,
            shares_outstanding=1_000,
            share_price=1000.0,
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.safety_score >= 0.0


# ===== EDGE CASES =====


class TestPhase30EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.margin_of_safety_analysis(FinancialData())
        assert isinstance(result, MarginOfSafetyResult)
        assert result.earnings_yield is None
        assert result.market_cap is None
        assert result.intrinsic_value_estimate is None

    def test_no_shares(self, analyzer):
        """No shares outstanding => no per-share metrics."""
        data = FinancialData(
            net_income=150_000,
            total_equity=1_000_000,
            total_liabilities=500_000,
            current_assets=400_000,
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.market_cap is None
        assert result.book_value_per_share is None
        assert result.ncav_per_share is None
        # IV should still compute (doesn't need shares)
        assert result.intrinsic_value_estimate == pytest.approx(1_500_000)

    def test_no_price(self, analyzer):
        """No share price => no market-based metrics."""
        data = FinancialData(
            net_income=150_000,
            total_equity=1_000_000,
            shares_outstanding=100_000,
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.market_cap is None
        assert result.price_to_book is None
        assert result.earnings_yield is None
        assert result.book_value_per_share is not None  # Only needs equity+shares

    def test_negative_net_income(self, analyzer):
        """NI<0 => no intrinsic value (can't capitalize losses)."""
        data = FinancialData(
            net_income=-50_000,
            total_equity=500_000,
            shares_outstanding=10_000,
            share_price=20.0,
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.intrinsic_value_estimate is None
        assert result.intrinsic_margin is None
        assert result.earnings_yield is not None  # NI/MCap works even if negative

    def test_zero_equity(self, analyzer):
        """Zero equity => BV/share is 0, P/B not computable (div by 0)."""
        data = FinancialData(
            net_income=50_000,
            total_equity=0,
            shares_outstanding=10_000,
            share_price=5.0,
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.book_value_per_share == pytest.approx(0.0)
        assert result.price_to_book is None  # BV=0, can't divide

    def test_negative_ncav(self, analyzer):
        """TL > CA => negative NCAV."""
        data = FinancialData(
            current_assets=200_000,
            total_liabilities=1_000_000,
            shares_outstanding=50_000,
            share_price=10.0,
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.net_current_asset_value == pytest.approx(-800_000)
        assert result.ncav_per_share == pytest.approx(-16.0)

    def test_trading_above_book(self, analyzer):
        """P/B > 1 means negative book value discount."""
        data = FinancialData(
            net_income=100_000,
            total_equity=500_000,
            shares_outstanding=10_000,
            share_price=100.0,  # P/B = 100/50 = 2.0
        )
        result = analyzer.margin_of_safety_analysis(data)
        assert result.price_to_book == pytest.approx(2.0, rel=0.01)
        assert result.book_value_discount == pytest.approx(-1.0, rel=0.01)

    def test_sample_data_score(self, analyzer, sample_data):
        """EY=15% (>=0.15: +2.0->7.0), IV margin=33.3% (>=0.25: +1.0->8.0),
        BV discount=0.167 (>0: +0.5->8.5), NCAV/share=-3 < price 10 (no bonus) => 8.5 Wide Margin."""
        result = analyzer.margin_of_safety_analysis(sample_data)
        assert result.safety_score == pytest.approx(8.5, abs=0.1)
        assert result.safety_grade == "Wide Margin"
