"""Phase 70 Tests: Tax Efficiency Analysis.

Tests for tax_efficiency_analysis() and TaxEfficiencyResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    TaxEfficiencyResult,
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
        ebt=170_000,
        tax_expense=20_000,
    )


# ===== DATACLASS TESTS =====


class TestTaxEfficiencyDataclass:
    def test_defaults(self):
        r = TaxEfficiencyResult()
        assert r.effective_tax_rate is None
        assert r.tax_to_revenue is None
        assert r.tax_to_ebitda is None
        assert r.after_tax_margin is None
        assert r.tax_shield_ratio is None
        assert r.pretax_to_ebit is None
        assert r.te_score == 0.0
        assert r.te_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = TaxEfficiencyResult(effective_tax_rate=0.20, te_grade="Good")
        assert r.effective_tax_rate == 0.20
        assert r.te_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestTaxEfficiencyAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert isinstance(result, TaxEfficiencyResult)

    def test_effective_tax_rate(self, analyzer, sample_data):
        """ETR = TaxExp/EBT = 20k/170k = 0.1176."""
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert result.effective_tax_rate == pytest.approx(0.1176, abs=0.01)

    def test_tax_to_revenue(self, analyzer, sample_data):
        """TaxExp/Rev = 20k/1M = 0.02."""
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert result.tax_to_revenue == pytest.approx(0.02, abs=0.01)

    def test_tax_to_ebitda(self, analyzer, sample_data):
        """TaxExp/EBITDA = 20k/250k = 0.08."""
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert result.tax_to_ebitda == pytest.approx(0.08, abs=0.01)

    def test_after_tax_margin(self, analyzer, sample_data):
        """NI/Rev = 150k/1M = 0.15."""
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert result.after_tax_margin == pytest.approx(0.15, abs=0.01)

    def test_pretax_to_ebit(self, analyzer, sample_data):
        """EBT/EBIT = 170k/200k = 0.85."""
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert result.pretax_to_ebit == pytest.approx(0.85, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert result.te_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert "Tax Efficiency" in result.summary


# ===== SCORING TESTS =====


class TestTaxEfficiencyScoring:
    def test_very_low_etr(self, analyzer):
        """ETR <= 0.15 => base 10."""
        data = FinancialData(
            revenue=5_000_000,
            net_income=1_000_000,
            ebit=1_200_000,
            ebitda=1_500_000,
            ebt=1_170_000,
            tax_expense=150_000,
            interest_expense=30_000,
        )
        result = analyzer.tax_efficiency_analysis(data)
        # ETR=0.128 (base 10) + ATM=0.20 (+0.5) + TTR=0.03 (+0.5) => capped 10
        assert result.te_score >= 9.0
        assert result.te_grade == "Excellent"

    def test_moderate_etr(self, analyzer, sample_data):
        """ETR ~ 0.12 => base 10."""
        result = analyzer.tax_efficiency_analysis(sample_data)
        assert result.te_score >= 8.0

    def test_high_etr(self, analyzer):
        """ETR = 0.35 => base 4.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=65_000,
            ebit=200_000,
            ebitda=250_000,
            ebt=100_000,
            tax_expense=35_000,
            interest_expense=100_000,
        )
        result = analyzer.tax_efficiency_analysis(data)
        assert result.te_score <= 5.0

    def test_very_high_etr(self, analyzer):
        """ETR > 0.40 => base 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=30_000,
            ebit=200_000,
            ebitda=250_000,
            ebt=60_000,
            tax_expense=30_000,
            interest_expense=140_000,
        )
        result = analyzer.tax_efficiency_analysis(data)
        assert result.te_score < 3.0
        assert result.te_grade == "Weak"

    def test_strong_margin_bonus(self, analyzer):
        """ATM >= 0.15 => +0.5."""
        data = FinancialData(
            revenue=2_000_000,
            net_income=400_000,
            ebit=600_000,
            ebitda=700_000,
            ebt=500_000,
            tax_expense=100_000,
            interest_expense=100_000,
        )
        result = analyzer.tax_efficiency_analysis(data)
        # ETR=0.20 (base 8.5) + ATM=0.20 (+0.5) + TTR=0.05 (no adj) => 9.0
        assert result.te_score >= 9.0


# ===== EDGE CASES =====


class TestPhase70EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.tax_efficiency_analysis(FinancialData())
        assert isinstance(result, TaxEfficiencyResult)
        assert result.effective_tax_rate is None

    def test_no_tax_expense(self, analyzer):
        """No tax expense => derive from EBT - NI."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=150_000,
            ebt=170_000,
            ebit=200_000,
            ebitda=250_000,
        )
        result = analyzer.tax_efficiency_analysis(data)
        # Derived tax = 170k - 150k = 20k, ETR = 20k/170k = 0.1176
        assert result.effective_tax_rate == pytest.approx(0.1176, abs=0.01)

    def test_no_ebt(self, analyzer):
        """No EBT => derive from EBIT - IE."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=150_000,
            ebit=200_000,
            ebitda=250_000,
            tax_expense=20_000,
            interest_expense=30_000,
        )
        result = analyzer.tax_efficiency_analysis(data)
        # EBT = EBIT - IE = 170k, ETR = 20k/170k = 0.1176
        assert result.effective_tax_rate == pytest.approx(0.1176, abs=0.01)

    def test_no_revenue(self, analyzer):
        """No revenue => tax ratios still computed if tax data exists."""
        data = FinancialData(
            ebt=100_000,
            tax_expense=25_000,
            net_income=75_000,
            ebit=120_000,
            ebitda=150_000,
        )
        result = analyzer.tax_efficiency_analysis(data)
        assert result.effective_tax_rate == pytest.approx(0.25, abs=0.01)
        assert result.tax_to_revenue is None  # safe_divide(25k, 0) => None (no revenue)
