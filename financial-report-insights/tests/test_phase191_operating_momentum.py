"""Phase 191 Tests: Operating Momentum Analysis.

Tests for operating_momentum_analysis() and OperatingMomentumResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    OperatingMomentumResult,
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
        dividends_paid=40_000,
    )


# ===== DATACLASS TESTS =====


class TestOperatingMomentumDataclass:
    def test_defaults(self):
        r = OperatingMomentumResult()
        assert r.ebitda_margin is None
        assert r.ebit_margin is None
        assert r.ocf_margin is None
        assert r.gross_to_operating_conversion is None
        assert r.operating_cash_conversion is None
        assert r.overhead_absorption is None
        assert r.om_score == 0.0
        assert r.om_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = OperatingMomentumResult(ebitda_margin=0.25, om_grade="Good")
        assert r.ebitda_margin == 0.25
        assert r.om_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestOperatingMomentumAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.operating_momentum_analysis(sample_data)
        assert isinstance(result, OperatingMomentumResult)

    def test_ebitda_margin(self, analyzer, sample_data):
        """EBITDA/Rev = 250k/1M = 0.25."""
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.ebitda_margin == pytest.approx(0.25, abs=0.005)

    def test_ebit_margin(self, analyzer, sample_data):
        """EBIT/Rev = 200k/1M = 0.20."""
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.ebit_margin == pytest.approx(0.20, abs=0.005)

    def test_ocf_margin(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.ocf_margin == pytest.approx(0.22, abs=0.005)

    def test_gross_to_operating_conversion(self, analyzer, sample_data):
        """OI/GP = 200k/400k = 0.50."""
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.gross_to_operating_conversion == pytest.approx(0.50, abs=0.005)

    def test_operating_cash_conversion(self, analyzer, sample_data):
        """OCF/OI = 220k/200k = 1.10."""
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.operating_cash_conversion == pytest.approx(1.10, abs=0.01)

    def test_overhead_absorption(self, analyzer, sample_data):
        """OI/OpEx = 200k/200k = 1.00."""
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.overhead_absorption == pytest.approx(1.00, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.om_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.operating_momentum_analysis(sample_data)
        assert "Operating Momentum" in result.summary


# ===== SCORING TESTS =====


class TestOperatingMomentumScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """EM=0.25 => base 8.5. GtOC=0.50 >=0.50 => +0.5. OCC=1.10 >=1.0 => +0.5. Score=9.5."""
        result = analyzer.operating_momentum_analysis(sample_data)
        assert result.om_score == pytest.approx(9.5, abs=0.5)
        assert result.om_grade == "Excellent"

    def test_excellent_momentum(self, analyzer):
        """Very high EBITDA margin."""
        data = FinancialData(
            ebitda=400_000,
            revenue=1_000_000,
            ebit=350_000,
            operating_cash_flow=380_000,
            gross_profit=600_000,
            operating_income=350_000,
            operating_expenses=250_000,
        )
        result = analyzer.operating_momentum_analysis(data)
        assert result.om_score >= 10.0
        assert result.om_grade == "Excellent"

    def test_weak_momentum(self, analyzer):
        """Very low EBITDA margin."""
        data = FinancialData(
            ebitda=30_000,
            revenue=1_000_000,
            ebit=20_000,
            operating_cash_flow=25_000,
            gross_profit=200_000,
            operating_income=20_000,
            operating_expenses=180_000,
        )
        # EM=0.03 <0.05 => base 1.0. GtOC=20k/200k=0.10 <0.25 => -0.5. OCC=25k/20k=1.25 >=1.0 => +0.5. Score=1.0.
        result = analyzer.operating_momentum_analysis(data)
        assert result.om_score <= 2.0
        assert result.om_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase191EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.operating_momentum_analysis(FinancialData())
        assert isinstance(result, OperatingMomentumResult)
        assert result.om_score == 0.0

    def test_no_ebitda(self, analyzer):
        """EBITDA=None => EM=None, score 0."""
        data = FinancialData(
            revenue=1_000_000,
            ebit=200_000,
        )
        result = analyzer.operating_momentum_analysis(data)
        assert result.ebitda_margin is None
        assert result.om_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => EM=None."""
        data = FinancialData(
            ebitda=250_000,
            ebit=200_000,
        )
        result = analyzer.operating_momentum_analysis(data)
        assert result.ebitda_margin is None
        assert result.om_score == 0.0

    def test_no_gross_profit(self, analyzer):
        """GP=None => GtOC=None, but EM still works."""
        data = FinancialData(
            ebitda=250_000,
            revenue=1_000_000,
        )
        result = analyzer.operating_momentum_analysis(data)
        assert result.gross_to_operating_conversion is None
        assert result.ebitda_margin is not None
