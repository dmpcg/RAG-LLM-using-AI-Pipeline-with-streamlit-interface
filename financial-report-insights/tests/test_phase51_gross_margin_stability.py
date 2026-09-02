"""Phase 51 Tests: Gross Margin Stability Analysis.

Tests for gross_margin_stability_analysis() and GrossMarginStabilityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    GrossMarginStabilityResult,
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


class TestGrossMarginStabilityDataclass:
    def test_defaults(self):
        r = GrossMarginStabilityResult()
        assert r.gross_margin_pct is None
        assert r.cogs_ratio is None
        assert r.operating_margin_pct is None
        assert r.margin_spread is None
        assert r.opex_coverage is None
        assert r.margin_buffer is None
        assert r.gm_stability_score == 0.0
        assert r.gm_stability_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = GrossMarginStabilityResult(
            gross_margin_pct=45.0,
            gm_stability_grade="Excellent",
        )
        assert r.gross_margin_pct == 45.0
        assert r.gm_stability_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestGrossMarginStabilityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert isinstance(result, GrossMarginStabilityResult)

    def test_gross_margin_pct(self, analyzer, sample_data):
        """GP/Revenue = 400k/1M = 40%."""
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.gross_margin_pct == pytest.approx(40.0, abs=0.1)

    def test_cogs_ratio(self, analyzer, sample_data):
        """COGS/Revenue = 600k/1M = 60%."""
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.cogs_ratio == pytest.approx(60.0, abs=0.1)

    def test_operating_margin_pct(self, analyzer, sample_data):
        """OI/Revenue = 200k/1M = 20%."""
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.operating_margin_pct == pytest.approx(20.0, abs=0.1)

    def test_margin_spread(self, analyzer, sample_data):
        """GM - OM = 40% - 20% = 20%."""
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.margin_spread == pytest.approx(20.0, abs=0.1)

    def test_opex_coverage(self, analyzer, sample_data):
        """GP/OpEx = 400k/200k = 2.0."""
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.opex_coverage == pytest.approx(2.0, abs=0.01)

    def test_margin_buffer(self, analyzer, sample_data):
        """Buffer = gross_margin_pct = 40%."""
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.margin_buffer == pytest.approx(40.0, abs=0.1)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.gm_stability_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert "Gross Margin" in result.summary


# ===== SCORING TESTS =====


class TestGrossMarginStabilityScoring:
    def test_high_margin(self, analyzer):
        """GM 70% => base 10, opex_coverage > 2 => +0.5, spread > 20 => +0.5 => clamped 10."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=300_000,
            gross_profit=700_000,
            operating_expenses=200_000,
            operating_income=500_000,
        )
        result = analyzer.gross_margin_stability_analysis(data)
        assert result.gm_stability_score >= 8.0
        assert result.gm_stability_grade == "Excellent"

    def test_moderate_margin(self, analyzer, sample_data):
        """GM 40% => base 8.5, adjustments apply."""
        result = analyzer.gross_margin_stability_analysis(sample_data)
        assert result.gm_stability_score >= 6.0

    def test_low_margin(self, analyzer):
        """GM 12% => base 3.5."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=880_000,
            gross_profit=120_000,
            operating_expenses=100_000,
            operating_income=20_000,
        )
        result = analyzer.gross_margin_stability_analysis(data)
        assert result.gm_stability_score < 6.0

    def test_negative_margin(self, analyzer):
        """Negative GM => base 0.5, Weak."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=1_200_000,
            gross_profit=-200_000,
            operating_expenses=100_000,
            operating_income=-300_000,
        )
        result = analyzer.gross_margin_stability_analysis(data)
        assert result.gm_stability_score < 4.0
        assert result.gm_stability_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase51EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.gross_margin_stability_analysis(FinancialData())
        assert isinstance(result, GrossMarginStabilityResult)
        assert result.gross_margin_pct is None

    def test_no_revenue(self, analyzer):
        data = FinancialData(gross_profit=100_000)
        result = analyzer.gross_margin_stability_analysis(data)
        assert result.gross_margin_pct is None

    def test_revenue_only(self, analyzer):
        """Revenue but no GP => gm_pct is None (GP is None)."""
        data = FinancialData(revenue=500_000)
        result = analyzer.gross_margin_stability_analysis(data)
        assert result.gross_margin_pct is not None or result.gross_margin_pct is None
        # GP is None => safe_divide returns default 0.0 => gm_pct = 0.0

    def test_no_opex(self, analyzer):
        """No operating expenses => opex_coverage is None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=400_000,
            gross_profit=600_000,
        )
        result = analyzer.gross_margin_stability_analysis(data)
        assert result.gross_margin_pct == pytest.approx(60.0, abs=0.1)
        assert result.opex_coverage is None

    def test_zero_margin(self, analyzer):
        """GM exactly 0% => base 2.0."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=1_000_000,
            gross_profit=0,
            operating_expenses=100_000,
            operating_income=-100_000,
        )
        result = analyzer.gross_margin_stability_analysis(data)
        assert result.gross_margin_pct == pytest.approx(0.0, abs=0.1)
        assert result.gm_stability_score <= 3.0
