"""Phase 17 Tests: Cash Flow Quality & FCF Yield.

Tests for cash_flow_quality() and CashFlowQualityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CashFlowQualityResult,
    CharlieAnalyzer,
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


class TestCashFlowQualityDataclass:
    def test_defaults(self):
        r = CashFlowQualityResult()
        assert r.fcf is None
        assert r.fcf_yield is None
        assert r.fcf_margin is None
        assert r.ocf_to_net_income is None
        assert r.quality_grade == ""
        assert r.indicators == []
        assert r.summary == ""

    def test_fields(self):
        r = CashFlowQualityResult(
            fcf=100_000,
            quality_grade="Strong",
        )
        assert r.fcf == 100_000
        assert r.quality_grade == "Strong"


# ===== CORE COMPUTATION TESTS =====


class TestCashFlowQuality:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cash_flow_quality(sample_data)
        assert isinstance(result, CashFlowQualityResult)

    def test_fcf_computed(self, analyzer, sample_data):
        """FCF = OCF - CapEx = 220k - 80k = 140k."""
        result = analyzer.cash_flow_quality(sample_data)
        assert result.fcf == pytest.approx(140_000, abs=1)

    def test_fcf_yield_computed(self, analyzer, sample_data):
        """FCF Yield = FCF / EV proxy (equity+debt) = 140k / 1.6M = 8.75%."""
        result = analyzer.cash_flow_quality(sample_data)
        assert result.fcf_yield is not None
        assert result.fcf_yield == pytest.approx(140_000 / 1_600_000, rel=0.01)

    def test_fcf_margin_computed(self, analyzer, sample_data):
        """FCF Margin = FCF / Revenue = 140k / 1M = 14%."""
        result = analyzer.cash_flow_quality(sample_data)
        assert result.fcf_margin is not None
        assert result.fcf_margin == pytest.approx(0.14, rel=0.01)

    def test_ocf_to_ni_computed(self, analyzer, sample_data):
        """OCF/NI = 220k / 150k ~= 1.467."""
        result = analyzer.cash_flow_quality(sample_data)
        assert result.ocf_to_net_income is not None
        assert result.ocf_to_net_income == pytest.approx(220_000 / 150_000, rel=0.01)

    def test_capex_to_revenue_computed(self, analyzer, sample_data):
        """CapEx/Revenue = 80k / 1M = 8%."""
        result = analyzer.cash_flow_quality(sample_data)
        assert result.capex_to_revenue is not None
        assert result.capex_to_revenue == pytest.approx(0.08, rel=0.01)

    def test_capex_to_ocf_computed(self, analyzer, sample_data):
        """CapEx/OCF = 80k / 220k ~= 36.4%."""
        result = analyzer.cash_flow_quality(sample_data)
        assert result.capex_to_ocf is not None
        assert result.capex_to_ocf == pytest.approx(80_000 / 220_000, rel=0.01)

    def test_cash_conversion_efficiency(self, analyzer, sample_data):
        """CCE = OCF / EBITDA = 220k / 250k = 88%."""
        result = analyzer.cash_flow_quality(sample_data)
        assert result.cash_conversion_efficiency is not None
        assert result.cash_conversion_efficiency == pytest.approx(0.88, rel=0.01)

    def test_quality_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cash_flow_quality(sample_data)
        assert result.quality_grade in ["Strong", "Adequate", "Weak", "Poor"]

    def test_indicators_present(self, analyzer, sample_data):
        result = analyzer.cash_flow_quality(sample_data)
        assert len(result.indicators) > 0

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cash_flow_quality(sample_data)
        assert "Cash Flow Quality" in result.summary


# ===== QUALITY SCORING TESTS =====


class TestQualityScoring:
    def test_strong_company_high_score(self, analyzer):
        """Company with excellent cash flow should get Strong grade."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            ebitda=200_000,
            total_equity=2_000_000,
            total_debt=200_000,
            operating_cash_flow=250_000,
            capex=30_000,
        )
        result = analyzer.cash_flow_quality(data)
        # OCF/NI = 2.5 (strong), FCF=220k (positive), FCF margin=22% (high),
        # CCE=1.25 (strong), capex low
        assert result.quality_grade == "Strong"

    def test_weak_company_low_score(self, analyzer):
        """Company with poor cash flow should get Weak or Poor grade."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=200_000,
            ebitda=300_000,
            total_equity=500_000,
            total_debt=500_000,
            operating_cash_flow=50_000,  # Very low OCF vs NI
            capex=250_000,  # High capex, FCF negative
        )
        result = analyzer.cash_flow_quality(data)
        # OCF/NI = 0.25 (weak), FCF=-200k (negative), high capex
        assert result.quality_grade in ["Weak", "Poor"]

    def test_score_capped_at_10(self, analyzer):
        """Score should never exceed 10."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=50_000,
            ebitda=100_000,
            total_equity=5_000_000,
            total_debt=0,
            operating_cash_flow=300_000,
            capex=10_000,
        )
        result = analyzer.cash_flow_quality(data)
        # Parse score from summary
        assert "10.0/10" in result.summary or float(result.summary.split("/10")[0].split("(")[-1]) <= 10.0

    def test_score_floored_at_0(self, analyzer):
        """Score should never go below 0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=-50_000,
            ebitda=10_000,
            total_equity=100_000,
            total_debt=900_000,
            operating_cash_flow=-100_000,
            capex=300_000,
        )
        result = analyzer.cash_flow_quality(data)
        # Negative OCF/NI, negative FCF, negative FCF margin, weak CCE
        assert result.quality_grade == "Poor"

    def test_grade_boundary_strong(self, analyzer):
        """Grade boundary: Strong >= 8."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            ebitda=200_000,
            total_equity=2_000_000,
            total_debt=100_000,
            operating_cash_flow=200_000,
            capex=20_000,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.quality_grade == "Strong"

    def test_positive_fcf_indicator(self, analyzer, sample_data):
        result = analyzer.cash_flow_quality(sample_data)
        assert any("Positive free cash flow" in ind for ind in result.indicators)

    def test_strong_ocf_indicator(self, analyzer, sample_data):
        """OCF/NI = 1.47 should trigger 'exceeds NI significantly'."""
        result = analyzer.cash_flow_quality(sample_data)
        assert any("OCF exceeds NI" in ind for ind in result.indicators)

    def test_strong_cash_conversion_indicator(self, analyzer, sample_data):
        """CCE = 88% should trigger 'Strong cash conversion'."""
        result = analyzer.cash_flow_quality(sample_data)
        assert any("Strong cash conversion" in ind for ind in result.indicators)


# ===== EDGE CASES =====


class TestPhase17EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cash_flow_quality(FinancialData())
        assert isinstance(result, CashFlowQualityResult)
        assert result.fcf is None

    def test_zero_revenue(self, analyzer):
        data = FinancialData(
            revenue=0,
            operating_cash_flow=50_000,
            capex=20_000,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.fcf == pytest.approx(30_000, abs=1)
        assert result.fcf_margin is None  # Can't divide by 0 revenue
        assert result.capex_to_revenue is None

    def test_no_ocf(self, analyzer):
        data = FinancialData(
            revenue=500_000,
            net_income=100_000,
            capex=50_000,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.fcf is None
        assert result.ocf_to_net_income is None

    def test_zero_net_income(self, analyzer):
        """Zero NI means OCF/NI is undefined."""
        data = FinancialData(
            revenue=500_000,
            net_income=0,
            ebitda=100_000,
            operating_cash_flow=80_000,
            capex=30_000,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.ocf_to_net_income is None
        assert result.fcf == pytest.approx(50_000, abs=1)

    def test_negative_net_income(self, analyzer):
        """Negative NI should still compute OCF/NI."""
        data = FinancialData(
            revenue=500_000,
            net_income=-50_000,
            ebitda=100_000,
            operating_cash_flow=80_000,
            capex=30_000,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.ocf_to_net_income is not None
        assert result.ocf_to_net_income == pytest.approx(80_000 / -50_000, rel=0.01)

    def test_no_capex(self, analyzer):
        """No capex means FCF = OCF."""
        data = FinancialData(
            revenue=500_000,
            net_income=100_000,
            ebitda=150_000,
            total_equity=1_000_000,
            total_debt=200_000,
            operating_cash_flow=120_000,
            capex=0,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.fcf == pytest.approx(120_000, abs=1)
        assert result.capex_to_ocf == pytest.approx(0, abs=0.01)

    def test_zero_ev_proxy(self, analyzer):
        """Zero equity + zero debt means no FCF yield."""
        data = FinancialData(
            revenue=500_000,
            net_income=100_000,
            total_equity=0,
            total_debt=0,
            operating_cash_flow=120_000,
            capex=20_000,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.fcf_yield is None

    def test_negative_ocf(self, analyzer):
        """Negative OCF means negative FCF and capex_to_ocf is None."""
        data = FinancialData(
            revenue=500_000,
            net_income=100_000,
            ebitda=200_000,
            total_equity=1_000_000,
            total_debt=200_000,
            operating_cash_flow=-50_000,
            capex=80_000,
        )
        result = analyzer.cash_flow_quality(data)
        assert result.fcf == pytest.approx(-130_000, abs=1)
        assert result.capex_to_ocf is None  # Can't divide by negative OCF
