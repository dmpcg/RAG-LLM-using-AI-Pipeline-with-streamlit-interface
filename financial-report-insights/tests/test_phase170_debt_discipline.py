"""Phase 170 Tests: Debt Discipline Analysis.

Tests for debt_discipline_analysis() and DebtDisciplineResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    DebtDisciplineResult,
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


class TestDebtDisciplineDataclass:
    def test_defaults(self):
        r = DebtDisciplineResult()
        assert r.debt_prudence_ratio is None
        assert r.debt_servicing_power is None
        assert r.debt_coverage_spread is None
        assert r.debt_to_equity_leverage is None
        assert r.interest_absorption is None
        assert r.debt_repayment_capacity is None
        assert r.dd_score == 0.0
        assert r.dd_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = DebtDisciplineResult(debt_prudence_ratio=0.20, dd_grade="Excellent")
        assert r.debt_prudence_ratio == 0.20
        assert r.dd_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestDebtDisciplineAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.debt_discipline_analysis(sample_data)
        assert isinstance(result, DebtDisciplineResult)

    def test_debt_prudence_ratio(self, analyzer, sample_data):
        """TD/TA = 400k/2M = 0.20."""
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.debt_prudence_ratio == pytest.approx(0.20, abs=0.01)

    def test_debt_servicing_power(self, analyzer, sample_data):
        """EBITDA/(IE+TD/5) = 250k/(30k+80k) = 250k/110k = 2.27."""
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.debt_servicing_power == pytest.approx(2.27, abs=0.05)

    def test_debt_coverage_spread(self, analyzer, sample_data):
        """OCF/TD = 220k/400k = 0.55."""
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.debt_coverage_spread == pytest.approx(0.55, abs=0.01)

    def test_debt_to_equity_leverage(self, analyzer, sample_data):
        """TD/TE = 400k/1.2M = 0.333."""
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.debt_to_equity_leverage == pytest.approx(0.333, abs=0.01)

    def test_interest_absorption(self, analyzer, sample_data):
        """IE/Rev = 30k/1M = 0.03."""
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.interest_absorption == pytest.approx(0.03, abs=0.01)

    def test_debt_repayment_capacity(self, analyzer, sample_data):
        """(OCF-CapEx)/TD = (220k-80k)/400k = 0.35."""
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.debt_repayment_capacity == pytest.approx(0.35, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.dd_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.debt_discipline_analysis(sample_data)
        assert "Debt Discipline" in result.summary


# ===== SCORING TESTS =====


class TestDebtDisciplineScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """DPR=0.20 => base 8.5. DCS=0.55 >=0.50 => +0.5. DEL=0.333 no adj. Score=9.0."""
        result = analyzer.debt_discipline_analysis(sample_data)
        assert result.dd_score == pytest.approx(9.0, abs=0.5)
        assert result.dd_grade == "Excellent"

    def test_excellent_discipline(self, analyzer):
        """Very low debt."""
        data = FinancialData(
            total_debt=50_000,
            total_assets=2_000_000,
            ebitda=500_000,
            interest_expense=2_000,
            operating_cash_flow=400_000,
            total_equity=1_900_000,
            revenue=1_500_000,
            capex=50_000,
        )
        result = analyzer.debt_discipline_analysis(data)
        assert result.dd_score >= 10.0
        assert result.dd_grade == "Excellent"

    def test_weak_discipline(self, analyzer):
        """Extremely high debt."""
        data = FinancialData(
            total_debt=1_800_000,
            total_assets=2_000_000,
            ebitda=100_000,
            interest_expense=80_000,
            operating_cash_flow=50_000,
            total_equity=200_000,
            revenue=500_000,
            capex=30_000,
        )
        # DPR=1.8M/2M=0.90 >0.70 => base 1.0. DCS=50k/1.8M=0.028 <0.10 => -0.5. DEL=1.8M/200k=9.0 >=1.50 => -0.5. Score=0.0.
        result = analyzer.debt_discipline_analysis(data)
        assert result.dd_score <= 1.0
        assert result.dd_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase170EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.debt_discipline_analysis(FinancialData())
        assert isinstance(result, DebtDisciplineResult)
        assert result.dd_score == 0.0

    def test_no_total_debt(self, analyzer):
        """TD=None => DPR=None, DCS=None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            ebitda=250_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.debt_discipline_analysis(data)
        assert result.debt_prudence_ratio is None
        assert result.dd_score == 0.0

    def test_no_total_assets(self, analyzer):
        """TA=None => DPR=None."""
        data = FinancialData(
            total_debt=400_000,
            total_equity=1_200_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.debt_discipline_analysis(data)
        assert result.debt_prudence_ratio is None
        assert result.debt_coverage_spread is not None

    def test_no_ebitda(self, analyzer):
        """EBITDA=None => DSP=None."""
        data = FinancialData(
            total_debt=400_000,
            total_assets=2_000_000,
            operating_cash_flow=220_000,
            total_equity=1_200_000,
        )
        result = analyzer.debt_discipline_analysis(data)
        assert result.debt_servicing_power is None
        assert result.debt_prudence_ratio is not None
