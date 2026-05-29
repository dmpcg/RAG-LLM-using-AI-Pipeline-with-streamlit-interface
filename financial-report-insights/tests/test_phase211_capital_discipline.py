"""Phase 211 Tests: Capital Discipline Analysis.

Tests for capital_discipline_analysis() and CapitalDisciplineResult dataclass.
"""

import pytest

from financial_analyzer import (
    CapitalDisciplineResult,
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


class TestCapitalDisciplineDataclass:
    def test_defaults(self):
        r = CapitalDisciplineResult()
        assert r.retained_to_equity is None
        assert r.retained_to_assets is None
        assert r.dividend_payout is None
        assert r.capex_to_ocf is None
        assert r.debt_to_equity is None
        assert r.ocf_to_debt is None
        assert r.cd_score == 0.0
        assert r.cd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CapitalDisciplineResult(retained_to_equity=0.50, cd_grade="Good")
        assert r.retained_to_equity == 0.50
        assert r.cd_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestCapitalDisciplineAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.capital_discipline_analysis(sample_data)
        assert isinstance(result, CapitalDisciplineResult)

    def test_retained_to_equity(self, analyzer, sample_data):
        """RE/TE = 600k/1.2M = 0.50."""
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.retained_to_equity == pytest.approx(0.50, abs=0.005)

    def test_retained_to_assets(self, analyzer, sample_data):
        """RE/TA = 600k/2M = 0.30."""
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.retained_to_assets == pytest.approx(0.30, abs=0.005)

    def test_dividend_payout(self, analyzer, sample_data):
        """Div/NI = 40k/150k = 0.267."""
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.dividend_payout == pytest.approx(0.267, abs=0.005)

    def test_capex_to_ocf(self, analyzer, sample_data):
        """CapEx/OCF = 80k/220k = 0.364."""
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.capex_to_ocf == pytest.approx(0.364, abs=0.005)

    def test_debt_to_equity(self, analyzer, sample_data):
        """TD/TE = 400k/1.2M = 0.333."""
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.005)

    def test_ocf_to_debt(self, analyzer, sample_data):
        """OCF/TD = 220k/400k = 0.55."""
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.ocf_to_debt == pytest.approx(0.55, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.cd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.capital_discipline_analysis(sample_data)
        assert "Capital Discipline" in result.summary


# ===== SCORING TESTS =====


class TestCapitalDisciplineScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """RE/TE=0.50 >=0.50 => base 7.0. OCF/TD=0.55 >=0.50 => +0.5. CapEx/OCF=0.364 >0.30 no adj. Score=7.5."""
        result = analyzer.capital_discipline_analysis(sample_data)
        assert result.cd_score == pytest.approx(7.5, abs=0.5)
        assert result.cd_grade in ["Good", "Excellent"]

    def test_excellent_discipline(self, analyzer):
        """Very high RE/TE."""
        data = FinancialData(
            retained_earnings=900_000,
            total_equity=1_000_000,
            total_assets=2_000_000,
            operating_cash_flow=600_000,
            total_debt=200_000,
            capex=50_000,
            net_income=400_000,
            dividends_paid=50_000,
        )
        result = analyzer.capital_discipline_analysis(data)
        assert result.cd_score >= 10.0
        assert result.cd_grade == "Excellent"

    def test_weak_discipline(self, analyzer):
        """Very low RE/TE."""
        data = FinancialData(
            retained_earnings=50_000,
            total_equity=1_000_000,
            total_assets=2_000_000,
            operating_cash_flow=50_000,
            total_debt=800_000,
            capex=45_000,
            net_income=30_000,
            dividends_paid=25_000,
        )
        # RE/TE=0.05 <0.15 => base 1.0. OCF/TD=50k/800k=0.0625 <0.15 => -0.5. CapEx/OCF=45k/50k=0.90 >0.80 => -0.5. Score=0.0.
        result = analyzer.capital_discipline_analysis(data)
        assert result.cd_score <= 0.5
        assert result.cd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase211EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.capital_discipline_analysis(FinancialData())
        assert isinstance(result, CapitalDisciplineResult)
        assert result.cd_score == 0.0

    def test_no_retained_earnings(self, analyzer):
        """RE=None => RE/TE=None => score 0."""
        data = FinancialData(
            total_equity=1_200_000,
            total_assets=2_000_000,
        )
        result = analyzer.capital_discipline_analysis(data)
        assert result.retained_to_equity is None
        assert result.cd_score == 0.0

    def test_no_equity(self, analyzer):
        """TE=None => RE/TE=None."""
        data = FinancialData(
            retained_earnings=600_000,
            total_assets=2_000_000,
        )
        result = analyzer.capital_discipline_analysis(data)
        assert result.retained_to_equity is None
        assert result.cd_score == 0.0

    def test_no_debt(self, analyzer):
        """TD=None => OCF/TD=None, but RE/TE still works."""
        data = FinancialData(
            retained_earnings=600_000,
            total_equity=1_200_000,
            total_assets=2_000_000,
            operating_cash_flow=220_000,
            capex=80_000,
        )
        result = analyzer.capital_discipline_analysis(data)
        assert result.ocf_to_debt is None
        assert result.retained_to_equity is not None
