"""Phase 134 Tests: Income Stability Analysis.

Tests for income_stability_analysis() and IncomeStabilityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    IncomeStabilityResult,
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


class TestIncomeStabilityDataclass:
    def test_defaults(self):
        r = IncomeStabilityResult()
        assert r.net_income_margin is None
        assert r.retained_earnings_ratio is None
        assert r.operating_income_cushion is None
        assert r.net_to_gross_ratio is None
        assert r.ebitda_margin is None
        assert r.income_resilience is None
        assert r.is_score == 0.0
        assert r.is_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = IncomeStabilityResult(net_income_margin=0.15, is_grade="Good")
        assert r.net_income_margin == 0.15
        assert r.is_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestIncomeStabilityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.income_stability_analysis(sample_data)
        assert isinstance(result, IncomeStabilityResult)

    def test_net_income_margin(self, analyzer, sample_data):
        """NIM = 150k / 1M = 0.15."""
        result = analyzer.income_stability_analysis(sample_data)
        assert result.net_income_margin == pytest.approx(0.15, abs=0.01)

    def test_retained_earnings_ratio(self, analyzer, sample_data):
        """RER = 600k / 2M = 0.30."""
        result = analyzer.income_stability_analysis(sample_data)
        assert result.retained_earnings_ratio == pytest.approx(0.30, abs=0.01)

    def test_operating_income_cushion(self, analyzer, sample_data):
        """OIC = 200k / 30k = 6.667."""
        result = analyzer.income_stability_analysis(sample_data)
        assert result.operating_income_cushion == pytest.approx(6.667, abs=0.01)

    def test_net_to_gross_ratio(self, analyzer, sample_data):
        """NTG = 150k / 400k = 0.375."""
        result = analyzer.income_stability_analysis(sample_data)
        assert result.net_to_gross_ratio == pytest.approx(0.375, abs=0.01)

    def test_ebitda_margin(self, analyzer, sample_data):
        """EM = 250k / 1M = 0.25."""
        result = analyzer.income_stability_analysis(sample_data)
        assert result.ebitda_margin == pytest.approx(0.25, abs=0.01)

    def test_income_resilience(self, analyzer, sample_data):
        """IR = 220k / 150k = 1.467."""
        result = analyzer.income_stability_analysis(sample_data)
        assert result.income_resilience == pytest.approx(1.467, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.income_stability_analysis(sample_data)
        assert result.is_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.income_stability_analysis(sample_data)
        assert "Income Stability" in result.summary


# ===== SCORING TESTS =====


class TestIncomeStabilityScoring:
    def test_good_stability(self, analyzer, sample_data):
        """OIC=6.667 => base 7.0. NIM=0.15 >=0.15 => +0.5. IR=1.467 >=1.0 => +0.5. Score=8.0."""
        result = analyzer.income_stability_analysis(sample_data)
        assert result.is_score == pytest.approx(8.0, abs=0.5)
        assert result.is_grade == "Excellent"

    def test_excellent_stability(self, analyzer):
        """OIC >= 10 => base 10."""
        data = FinancialData(
            operating_income=500_000,
            interest_expense=30_000,
            net_income=400_000,
            revenue=2_000_000,
            operating_cash_flow=500_000,
            total_assets=3_000_000,
            retained_earnings=1_000_000,
            gross_profit=1_000_000,
            ebitda=600_000,
        )
        # OIC=16.67 => 10. NIM=0.20 >=0.15 => +0.5. IR=1.25 >=1.0 => +0.5. Capped 10.
        result = analyzer.income_stability_analysis(data)
        assert result.is_score >= 10.0
        assert result.is_grade == "Excellent"

    def test_weak_stability(self, analyzer):
        """OIC < 1.0 => base 1.0."""
        data = FinancialData(
            operating_income=20_000,
            interest_expense=50_000,
            net_income=5_000,
            revenue=1_000_000,
            operating_cash_flow=2_000,
            total_assets=2_000_000,
            gross_profit=200_000,
            ebitda=50_000,
        )
        # OIC=0.4 => 1.0. NIM=0.005 <0.02 => -0.5. IR=0.4 <0.5 => -0.5. Score=0.0.
        result = analyzer.income_stability_analysis(data)
        assert result.is_score <= 1.0
        assert result.is_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase134EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.income_stability_analysis(FinancialData())
        assert isinstance(result, IncomeStabilityResult)
        assert result.is_score == 0.0

    def test_no_interest_expense(self, analyzer):
        """IE=None => OIC=None => score 0."""
        data = FinancialData(
            operating_income=200_000,
            net_income=150_000,
            revenue=1_000_000,
            total_assets=2_000_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.income_stability_analysis(data)
        assert result.operating_income_cushion is None
        assert result.is_score == 0.0

    def test_no_revenue(self, analyzer):
        """Rev=None => NIM=None, EM=None, but OIC still computed."""
        data = FinancialData(
            operating_income=200_000,
            interest_expense=30_000,
            net_income=150_000,
            total_assets=2_000_000,
            operating_cash_flow=220_000,
            gross_profit=400_000,
            ebitda=250_000,
        )
        result = analyzer.income_stability_analysis(data)
        assert result.net_income_margin is None
        assert result.ebitda_margin is None
        assert result.operating_income_cushion is not None

    def test_no_net_income(self, analyzer):
        """NI=None => NIM=None, NTG=None, IR=None."""
        data = FinancialData(
            operating_income=200_000,
            interest_expense=30_000,
            revenue=1_000_000,
            total_assets=2_000_000,
            gross_profit=400_000,
            ebitda=250_000,
        )
        result = analyzer.income_stability_analysis(data)
        assert result.net_income_margin is None
        assert result.net_to_gross_ratio is None
        assert result.income_resilience is None
        assert result.operating_income_cushion is not None
