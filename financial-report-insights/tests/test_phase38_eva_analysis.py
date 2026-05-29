"""Phase 38 Tests: Economic Value Added (EVA).

Tests for eva_analysis() and EVAResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    EVAResult,
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


class TestEVADataclass:
    def test_defaults(self):
        r = EVAResult()
        assert r.eva is None
        assert r.nopat is None
        assert r.invested_capital is None
        assert r.eva_score == 0.0
        assert r.eva_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = EVAResult(
            eva=100_000,
            eva_grade="Value Creator",
        )
        assert r.eva == 100_000
        assert r.eva_grade == "Value Creator"


# ===== CORE COMPUTATION TESTS =====


class TestEVAAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.eva_analysis(sample_data)
        assert isinstance(result, EVAResult)

    def test_invested_capital(self, analyzer, sample_data):
        """IC = TA - CL = 2M - 200k = 1.8M."""
        result = analyzer.eva_analysis(sample_data)
        assert result.invested_capital == pytest.approx(1_800_000, rel=0.01)

    def test_nopat(self, analyzer, sample_data):
        """NOPAT = EBIT * (1 - tax). tax=11.8%, NOPAT = 200k * 0.882 = 176,400."""
        result = analyzer.eva_analysis(sample_data)
        assert result.nopat == pytest.approx(176_400, rel=0.02)

    def test_wacc_used(self, analyzer, sample_data):
        """Should use WACC from wacc_analysis (~13.3%)."""
        result = analyzer.eva_analysis(sample_data)
        assert result.wacc_used == pytest.approx(0.133, abs=0.01)

    def test_capital_charge(self, analyzer, sample_data):
        """Capital Charge = IC * WACC = 1.8M * 0.133 = ~239k."""
        result = analyzer.eva_analysis(sample_data)
        assert result.capital_charge is not None
        assert result.capital_charge > 200_000

    def test_eva_value(self, analyzer, sample_data):
        """EVA = NOPAT - Capital Charge. With these numbers, EVA is negative."""
        result = analyzer.eva_analysis(sample_data)
        assert result.eva is not None
        assert result.eva < 0  # Capital charge exceeds NOPAT

    def test_roic(self, analyzer, sample_data):
        """ROIC = NOPAT / IC = ~176k / 1.8M = ~9.8%."""
        result = analyzer.eva_analysis(sample_data)
        assert result.roic == pytest.approx(0.098, abs=0.01)

    def test_roic_wacc_spread(self, analyzer, sample_data):
        """Spread = ROIC - WACC = ~9.8% - ~13.3% = ~-3.5%."""
        result = analyzer.eva_analysis(sample_data)
        assert result.roic_wacc_spread is not None
        assert result.roic_wacc_spread < 0

    def test_eva_margin(self, analyzer, sample_data):
        """EVA/Revenue."""
        result = analyzer.eva_analysis(sample_data)
        assert result.eva_margin is not None

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.eva_analysis(sample_data)
        assert result.eva_grade in ["Value Creator", "Adequate", "Marginal", "Value Destroyer"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.eva_analysis(sample_data)
        assert "EVA" in result.summary


# ===== SCORING TESTS =====


class TestEVAScoring:
    def test_value_creator(self, analyzer):
        """High positive EVA => Value Creator."""
        data = FinancialData(
            ebit=500_000,
            total_assets=1_000_000,
            current_liabilities=100_000,
            total_debt=200_000,
            total_equity=700_000,
            interest_expense=10_000,
            net_income=400_000,
            revenue=2_000_000,
        )
        result = analyzer.eva_analysis(data)
        # NOPAT ~= 500k * (1-tax), IC = 1M - 100k = 900k
        # With low WACC, EVA should be very positive
        assert result.eva > 0
        assert result.eva_score >= 7.0
        assert result.eva_grade in ["Value Creator", "Adequate"]

    def test_value_destroyer(self, analyzer):
        """Negative EVA with large spread => Value Destroyer."""
        data = FinancialData(
            ebit=30_000,
            total_assets=2_000_000,
            current_liabilities=100_000,
            total_debt=800_000,
            total_equity=1_100_000,
            interest_expense=80_000,
            net_income=5_000,
            revenue=500_000,
        )
        result = analyzer.eva_analysis(data)
        # Low EBIT on large capital base => negative EVA
        assert result.eva < 0
        assert result.eva_score < 4.0
        assert result.eva_grade == "Value Destroyer"


# ===== EDGE CASES =====


class TestPhase38EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.eva_analysis(FinancialData())
        assert isinstance(result, EVAResult)
        assert result.eva is None

    def test_no_ebit(self, analyzer):
        """No EBIT => cannot compute NOPAT."""
        data = FinancialData(
            total_assets=1_000_000,
            current_liabilities=200_000,
            total_debt=300_000,
            total_equity=500_000,
        )
        result = analyzer.eva_analysis(data)
        assert result.nopat is None

    def test_no_total_assets_uses_debt_equity(self, analyzer):
        """Without TA/CL, fall back to Debt+Equity as invested capital."""
        data = FinancialData(
            ebit=100_000,
            total_debt=300_000,
            total_equity=700_000,
            interest_expense=15_000,
            net_income=70_000,
        )
        result = analyzer.eva_analysis(data)
        assert result.invested_capital == pytest.approx(1_000_000, rel=0.01)

    def test_sample_data_score(self, analyzer, sample_data):
        """EVA<0: -2.0, spread~-3.5% (<0 but >=-0.05: -0.5).
        Score = 5.0-2.0-0.5 = 2.5 => Value Destroyer."""
        result = analyzer.eva_analysis(sample_data)
        assert result.eva_score == pytest.approx(2.5, abs=0.3)
        assert result.eva_grade == "Value Destroyer"
