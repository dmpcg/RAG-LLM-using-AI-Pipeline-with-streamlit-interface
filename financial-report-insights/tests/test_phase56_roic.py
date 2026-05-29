"""Phase 56 Tests: Return on Invested Capital (ROIC) Analysis.

Tests for roic_analysis() and ROICResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ROICResult,
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


class TestROICDataclass:
    def test_defaults(self):
        r = ROICResult()
        assert r.roic_pct is None
        assert r.nopat is None
        assert r.invested_capital is None
        assert r.roic_roa_spread is None
        assert r.roic_wacc_spread is None
        assert r.capital_efficiency is None
        assert r.roic_score == 0.0
        assert r.roic_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ROICResult(
            roic_pct=15.0,
            roic_grade="Excellent",
        )
        assert r.roic_pct == 15.0
        assert r.roic_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestROICAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.roic_analysis(sample_data)
        assert isinstance(result, ROICResult)

    def test_invested_capital(self, analyzer, sample_data):
        """IC = TE + TD - Cash = 1.2M + 400k - 50k = 1.55M."""
        result = analyzer.roic_analysis(sample_data)
        assert result.invested_capital == pytest.approx(1_550_000, rel=0.01)

    def test_nopat(self, analyzer, sample_data):
        """EBT = 200k - 30k = 170k. Tax rate = 1 - 150k/170k = 0.1176.
        NOPAT = 200k * (1 - 0.1176) = 176,471."""
        result = analyzer.roic_analysis(sample_data)
        assert result.nopat is not None
        assert result.nopat > 0

    def test_roic_calculated(self, analyzer, sample_data):
        """ROIC = NOPAT / IC."""
        result = analyzer.roic_analysis(sample_data)
        assert result.roic_pct is not None
        # NOPAT / 1.55M * 100 should give a reasonable ROIC
        assert result.roic_pct > 0

    def test_capital_efficiency(self, analyzer, sample_data):
        """Rev/IC = 1M/1.55M = 0.645."""
        result = analyzer.roic_analysis(sample_data)
        assert result.capital_efficiency == pytest.approx(0.645, abs=0.02)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.roic_analysis(sample_data)
        assert result.roic_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.roic_analysis(sample_data)
        assert "ROIC" in result.summary


# ===== SCORING TESTS =====


class TestROICScoring:
    def test_high_roic(self, analyzer):
        """ROIC ~30% => base 10."""
        data = FinancialData(
            revenue=2_000_000,
            ebit=600_000,
            net_income=450_000,
            interest_expense=10_000,
            total_equity=1_000_000,
            total_debt=200_000,
            cash=50_000,
            total_assets=1_500_000,
        )
        result = analyzer.roic_analysis(data)
        assert result.roic_score >= 8.0
        assert result.roic_grade == "Excellent"

    def test_moderate_roic(self, analyzer, sample_data):
        result = analyzer.roic_analysis(sample_data)
        assert result.roic_score >= 5.0

    def test_low_roic(self, analyzer):
        """ROIC ~3% => low score."""
        data = FinancialData(
            revenue=500_000,
            ebit=50_000,
            net_income=30_000,
            interest_expense=10_000,
            total_equity=800_000,
            total_debt=400_000,
            cash=50_000,
            total_assets=1_500_000,
        )
        result = analyzer.roic_analysis(data)
        assert result.roic_score < 6.0

    def test_negative_roic(self, analyzer):
        """Negative EBIT => negative NOPAT => Weak."""
        data = FinancialData(
            revenue=500_000,
            ebit=-100_000,
            net_income=-150_000,
            total_equity=800_000,
            total_debt=200_000,
            cash=50_000,
            total_assets=1_200_000,
        )
        result = analyzer.roic_analysis(data)
        assert result.roic_score < 4.0
        assert result.roic_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase56EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.roic_analysis(FinancialData())
        assert isinstance(result, ROICResult)
        assert result.roic_pct is None

    def test_no_equity(self, analyzer):
        data = FinancialData(ebit=200_000, net_income=150_000, total_assets=2_000_000)
        result = analyzer.roic_analysis(data)
        assert result.roic_pct is None

    def test_no_ebit(self, analyzer):
        data = FinancialData(net_income=100_000, total_equity=1_000_000, total_assets=2_000_000)
        result = analyzer.roic_analysis(data)
        assert result.roic_pct is None

    def test_zero_invested_capital(self, analyzer):
        """IC = TE + TD - Cash = 100 + 0 - 200 = -100 => skip."""
        data = FinancialData(
            ebit=50_000,
            net_income=40_000,
            total_equity=100,
            total_debt=0,
            cash=200,
            total_assets=500,
        )
        result = analyzer.roic_analysis(data)
        assert result.roic_pct is None
