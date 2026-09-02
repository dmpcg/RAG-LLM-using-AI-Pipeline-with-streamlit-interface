"""Phase 39 Tests: Free Cash Flow Yield.

Tests for fcf_yield_analysis() and FCFYieldResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FCFYieldResult,
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
    )


# ===== DATACLASS TESTS =====


class TestFCFYieldDataclass:
    def test_defaults(self):
        r = FCFYieldResult()
        assert r.fcf is None
        assert r.fcf_margin is None
        assert r.fcf_to_net_income is None
        assert r.fcf_score == 0.0
        assert r.fcf_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FCFYieldResult(
            fcf=140_000,
            fcf_grade="Healthy",
        )
        assert r.fcf == 140_000
        assert r.fcf_grade == "Healthy"


# ===== CORE COMPUTATION TESTS =====


class TestFCFYieldAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.fcf_yield_analysis(sample_data)
        assert isinstance(result, FCFYieldResult)

    def test_fcf(self, analyzer, sample_data):
        """FCF = OCF - CapEx = 220k - 80k = 140k."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf == pytest.approx(140_000, rel=0.01)

    def test_fcf_margin(self, analyzer, sample_data):
        """FCF margin = 140k / 1M = 14%."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf_margin == pytest.approx(0.14, abs=0.01)

    def test_fcf_to_net_income(self, analyzer, sample_data):
        """FCF conversion = 140k / 150k = 0.933."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf_to_net_income == pytest.approx(0.933, abs=0.02)

    def test_fcf_yield_on_capital(self, analyzer, sample_data):
        """FCF yield on capital = 140k / (2M - 200k) = 7.78%."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf_yield_on_capital == pytest.approx(0.0778, abs=0.005)

    def test_fcf_yield_on_equity(self, analyzer, sample_data):
        """FCF yield on equity = 140k / 1.2M = 11.67%."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf_yield_on_equity == pytest.approx(0.1167, abs=0.005)

    def test_fcf_to_debt(self, analyzer, sample_data):
        """FCF to debt = 140k / 400k = 0.35."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf_to_debt == pytest.approx(0.35, abs=0.02)

    def test_capex_to_ocf(self, analyzer, sample_data):
        """CapEx/OCF = 80k / 220k = 36.4%."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.capex_to_ocf == pytest.approx(0.364, abs=0.02)

    def test_capex_to_revenue(self, analyzer, sample_data):
        """CapEx/Revenue = 80k / 1M = 8%."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.capex_to_revenue == pytest.approx(0.08, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf_grade in ["Strong", "Healthy", "Weak", "Negative"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.fcf_yield_analysis(sample_data)
        assert "FCF" in result.summary


# ===== SCORING TESTS =====


class TestFCFYieldScoring:
    def test_strong_fcf(self, analyzer):
        """High FCF margin + high conversion => Strong."""
        data = FinancialData(
            operating_cash_flow=500_000,
            capex=50_000,
            revenue=2_000_000,
            net_income=300_000,
            total_assets=1_500_000,
            current_liabilities=200_000,
            total_equity=800_000,
            total_debt=300_000,
        )
        result = analyzer.fcf_yield_analysis(data)
        # FCF=450k, margin=22.5%>0.15(+1.0), conversion=1.5>1.0(+0.5),
        # capex/ocf=10%<0.30(+0.5), FCF>0(+1.5)
        # Score = 5.0 + 1.5 + 1.0 + 0.5 + 0.5 = 8.5
        assert result.fcf > 0
        assert result.fcf_score >= 8.0
        assert result.fcf_grade == "Strong"

    def test_negative_fcf(self, analyzer):
        """Negative FCF => Negative grade."""
        data = FinancialData(
            operating_cash_flow=50_000,
            capex=200_000,
            revenue=1_000_000,
            net_income=100_000,
            total_assets=2_000_000,
            current_liabilities=200_000,
            total_equity=1_000_000,
            total_debt=500_000,
        )
        result = analyzer.fcf_yield_analysis(data)
        # FCF=-150k<0(-2.0), margin<0(-1.0), capex/ocf=4.0>0.70(-0.5)
        # Score = 5.0 - 2.0 - 1.0 - 0.5 = 1.5
        assert result.fcf < 0
        assert result.fcf_score < 4.0
        assert result.fcf_grade == "Negative"


# ===== EDGE CASES =====


class TestPhase39EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.fcf_yield_analysis(FinancialData())
        assert isinstance(result, FCFYieldResult)
        assert result.fcf is None

    def test_no_ocf(self, analyzer):
        """No OCF => cannot compute FCF."""
        data = FinancialData(
            capex=80_000,
            revenue=1_000_000,
        )
        result = analyzer.fcf_yield_analysis(data)
        assert result.fcf is None

    def test_no_capex(self, analyzer):
        """No CapEx => FCF = OCF."""
        data = FinancialData(
            operating_cash_flow=200_000,
            revenue=1_000_000,
            net_income=150_000,
            total_assets=1_000_000,
            current_liabilities=100_000,
            total_equity=600_000,
        )
        result = analyzer.fcf_yield_analysis(data)
        assert result.fcf == pytest.approx(200_000, rel=0.01)

    def test_no_debt(self, analyzer):
        """No debt => fcf_to_debt is None."""
        data = FinancialData(
            operating_cash_flow=200_000,
            capex=50_000,
            revenue=1_000_000,
            net_income=100_000,
            total_assets=1_000_000,
            current_liabilities=100_000,
            total_equity=900_000,
        )
        result = analyzer.fcf_yield_analysis(data)
        assert result.fcf is not None
        assert result.fcf_to_debt is None

    def test_sample_data_score(self, analyzer, sample_data):
        """FCF>0: +1.5, margin 14% >0.08: +0.5, conversion ~0.93 (no adj),
        capex/ocf 36.4% (no adj). Score = 5.0+1.5+0.5 = 7.0 => Healthy."""
        result = analyzer.fcf_yield_analysis(sample_data)
        assert result.fcf_score == pytest.approx(7.0, abs=0.3)
        assert result.fcf_grade == "Healthy"
