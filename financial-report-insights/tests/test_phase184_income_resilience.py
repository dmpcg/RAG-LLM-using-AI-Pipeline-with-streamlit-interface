"""Phase 184 Tests: Income Resilience Analysis.

Tests for income_resilience_analysis() and IncomeResilienceResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    IncomeResilienceResult,
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


class TestIncomeResilienceDataclass:
    def test_defaults(self):
        r = IncomeResilienceResult()
        assert r.operating_income_stability is None
        assert r.ebit_coverage is None
        assert r.net_margin_resilience is None
        assert r.depreciation_buffer is None
        assert r.tax_interest_drag is None
        assert r.ebitda_cushion is None
        assert r.ir_score == 0.0
        assert r.ir_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = IncomeResilienceResult(operating_income_stability=0.20, ir_grade="Excellent")
        assert r.operating_income_stability == 0.20
        assert r.ir_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestIncomeResilienceAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.income_resilience_analysis(sample_data)
        assert isinstance(result, IncomeResilienceResult)

    def test_operating_income_stability(self, analyzer, sample_data):
        """OI/Rev = 200k/1M = 0.20."""
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.operating_income_stability == pytest.approx(0.20, abs=0.005)

    def test_ebit_coverage(self, analyzer, sample_data):
        """EBIT/IE = 200k/30k = 6.667."""
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.ebit_coverage == pytest.approx(6.667, abs=0.01)

    def test_net_margin_resilience(self, analyzer, sample_data):
        """NI/OI = 150k/200k = 0.75."""
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.net_margin_resilience == pytest.approx(0.75, abs=0.005)

    def test_depreciation_buffer(self, analyzer, sample_data):
        """D&A/OI = 50k/200k = 0.25."""
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.depreciation_buffer == pytest.approx(0.25, abs=0.005)

    def test_tax_interest_drag(self, analyzer, sample_data):
        """(OI-NI)/OI = (200k-150k)/200k = 0.25."""
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.tax_interest_drag == pytest.approx(0.25, abs=0.005)

    def test_ebitda_cushion(self, analyzer, sample_data):
        """EBITDA/IE = 250k/30k = 8.333."""
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.ebitda_cushion == pytest.approx(8.333, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.ir_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.income_resilience_analysis(sample_data)
        assert "Income Resilience" in result.summary


# ===== SCORING TESTS =====


class TestIncomeResilienceScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """OIS=0.20 => base 8.5. EC=6.67 >=5.0 => +0.5. NMR=0.75 >=0.70 => +0.5. Score=9.5."""
        result = analyzer.income_resilience_analysis(sample_data)
        assert result.ir_score == pytest.approx(9.5, abs=0.5)
        assert result.ir_grade == "Excellent"

    def test_excellent_resilience(self, analyzer):
        """Very high OI stability."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=350_000,
            ebit=350_000,
            interest_expense=20_000,
            net_income=280_000,
            depreciation=50_000,
            ebitda=400_000,
        )
        result = analyzer.income_resilience_analysis(data)
        assert result.ir_score >= 10.0
        assert result.ir_grade == "Excellent"

    def test_weak_resilience(self, analyzer):
        """Very low OI stability."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=20_000,
            ebit=20_000,
            interest_expense=15_000,
            net_income=3_000,
            depreciation=10_000,
            ebitda=30_000,
        )
        # OIS=20k/1M=0.02 <0.03 => base 1.0. EC=20k/15k=1.33 <2.0 => -0.5. NMR=3k/20k=0.15 <0.40 => -0.5. Score=0.0.
        result = analyzer.income_resilience_analysis(data)
        assert result.ir_score <= 1.0
        assert result.ir_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase184EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.income_resilience_analysis(FinancialData())
        assert isinstance(result, IncomeResilienceResult)
        assert result.ir_score == 0.0

    def test_no_operating_income(self, analyzer):
        """OI=None => OIS=None, score 0."""
        data = FinancialData(
            revenue=1_000_000,
            ebit=200_000,
        )
        result = analyzer.income_resilience_analysis(data)
        assert result.operating_income_stability is None
        assert result.ir_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => OIS=None."""
        data = FinancialData(
            operating_income=200_000,
            ebit=200_000,
        )
        result = analyzer.income_resilience_analysis(data)
        assert result.operating_income_stability is None
        assert result.ir_score == 0.0

    def test_no_interest_expense(self, analyzer):
        """IE=None => EC=None, EBITDA cushion=None."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=200_000,
            ebit=200_000,
            net_income=150_000,
        )
        result = analyzer.income_resilience_analysis(data)
        assert result.ebit_coverage is None
        assert result.ebitda_cushion is None
        assert result.operating_income_stability is not None
