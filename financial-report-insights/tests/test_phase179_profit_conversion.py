"""Phase 179 Tests: Profit Conversion Analysis.

Tests for profit_conversion_analysis() and ProfitConversionResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ProfitConversionResult,
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


class TestProfitConversionDataclass:
    def test_defaults(self):
        r = ProfitConversionResult()
        assert r.gross_conversion is None
        assert r.operating_conversion is None
        assert r.net_conversion is None
        assert r.ebitda_conversion is None
        assert r.cash_conversion is None
        assert r.profit_to_cash_ratio is None
        assert r.pc_score == 0.0
        assert r.pc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ProfitConversionResult(gross_conversion=0.40, pc_grade="Good")
        assert r.gross_conversion == 0.40
        assert r.pc_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestProfitConversionAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.profit_conversion_analysis(sample_data)
        assert isinstance(result, ProfitConversionResult)

    def test_gross_conversion(self, analyzer, sample_data):
        """GP/Rev = 400k/1M = 0.40."""
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.gross_conversion == pytest.approx(0.40, abs=0.005)

    def test_operating_conversion(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.operating_conversion == pytest.approx(0.20, abs=0.005)

    def test_net_conversion(self, analyzer, sample_data):
        """NI/Rev = 150k/1M = 0.15."""
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.net_conversion == pytest.approx(0.15, abs=0.005)

    def test_ebitda_conversion(self, analyzer, sample_data):
        """EBITDA/Rev = 250k/1M = 0.25."""
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.ebitda_conversion == pytest.approx(0.25, abs=0.005)

    def test_cash_conversion(self, analyzer, sample_data):
        """OCF/Rev = 220k/1M = 0.22."""
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.cash_conversion == pytest.approx(0.22, abs=0.005)

    def test_profit_to_cash_ratio(self, analyzer, sample_data):
        """OCF/NI = 220k/150k = 1.467."""
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.profit_to_cash_ratio == pytest.approx(1.467, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.pc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.profit_conversion_analysis(sample_data)
        assert "Profit Conversion" in result.summary


# ===== SCORING TESTS =====


class TestProfitConversionScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """GC=0.40 => base 7.0. OC=0.20 >=0.20 => +0.5. CC=0.22 >=0.20 => +0.5. Score=8.0."""
        result = analyzer.profit_conversion_analysis(sample_data)
        assert result.pc_score == pytest.approx(8.0, abs=0.5)
        assert result.pc_grade == "Excellent"

    def test_excellent_conversion(self, analyzer):
        """Very high gross conversion."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=700_000,
            operating_income=400_000,
            net_income=300_000,
            ebitda=450_000,
            operating_cash_flow=350_000,
        )
        result = analyzer.profit_conversion_analysis(data)
        assert result.pc_score >= 10.0
        assert result.pc_grade == "Excellent"

    def test_weak_conversion(self, analyzer):
        """Very low gross conversion."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=80_000,
            operating_income=20_000,
            net_income=10_000,
            ebitda=40_000,
            operating_cash_flow=30_000,
        )
        # GC=80k/1M=0.08 <0.10 => base 1.0. OC=20k/1M=0.02 <0.05 => -0.5. CC=30k/1M=0.03 <0.05 => -0.5. Score=0.0.
        result = analyzer.profit_conversion_analysis(data)
        assert result.pc_score <= 1.0
        assert result.pc_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase179EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.profit_conversion_analysis(FinancialData())
        assert isinstance(result, ProfitConversionResult)
        assert result.pc_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => GC=None, score 0."""
        data = FinancialData(
            gross_profit=400_000,
            operating_income=200_000,
        )
        result = analyzer.profit_conversion_analysis(data)
        assert result.gross_conversion is None
        assert result.pc_score == 0.0

    def test_no_gross_profit(self, analyzer):
        """GP=None => GC=None."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.profit_conversion_analysis(data)
        assert result.gross_conversion is None
        assert result.operating_conversion is not None

    def test_no_ocf(self, analyzer):
        """OCF=None => CC=None, PtCR=None."""
        data = FinancialData(
            revenue=1_000_000,
            gross_profit=400_000,
            operating_income=200_000,
            net_income=150_000,
        )
        result = analyzer.profit_conversion_analysis(data)
        assert result.cash_conversion is None
        assert result.profit_to_cash_ratio is None
        assert result.gross_conversion is not None
