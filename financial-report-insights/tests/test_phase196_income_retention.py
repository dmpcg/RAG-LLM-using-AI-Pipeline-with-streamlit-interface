"""Phase 196 Tests: Income Retention Analysis.

Tests for income_retention_analysis() and IncomeRetentionResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    IncomeRetentionResult,
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


class TestIncomeRetentionDataclass:
    def test_defaults(self):
        r = IncomeRetentionResult()
        assert r.net_to_gross_ratio is None
        assert r.net_to_operating_ratio is None
        assert r.net_to_ebitda_ratio is None
        assert r.retention_rate is None
        assert r.income_to_asset_generation is None
        assert r.after_tax_margin is None
        assert r.ir_score == 0.0
        assert r.ir_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = IncomeRetentionResult(net_to_gross_ratio=0.40, ir_grade="Excellent")
        assert r.net_to_gross_ratio == 0.40
        assert r.ir_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestIncomeRetentionAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.income_retention_analysis(sample_data)
        assert isinstance(result, IncomeRetentionResult)

    def test_net_to_gross_ratio(self, analyzer, sample_data):
        """NI/GP = 150k/400k = 0.375."""
        result = analyzer.income_retention_analysis(sample_data)
        assert result.net_to_gross_ratio == pytest.approx(0.375, abs=0.005)

    def test_net_to_operating_ratio(self, analyzer, sample_data):
        """NI/OI = 150k/200k = 0.75."""
        result = analyzer.income_retention_analysis(sample_data)
        assert result.net_to_operating_ratio == pytest.approx(0.75, abs=0.005)

    def test_net_to_ebitda_ratio(self, analyzer, sample_data):
        """NI/EBITDA = 150k/250k = 0.60."""
        result = analyzer.income_retention_analysis(sample_data)
        assert result.net_to_ebitda_ratio == pytest.approx(0.60, abs=0.005)

    def test_retention_rate(self, analyzer, sample_data):
        """RE/NI = 600k/150k = 4.00."""
        result = analyzer.income_retention_analysis(sample_data)
        assert result.retention_rate == pytest.approx(4.00, abs=0.01)

    def test_income_to_asset_generation(self, analyzer, sample_data):
        """NI/TA = 150k/2M = 0.075."""
        result = analyzer.income_retention_analysis(sample_data)
        assert result.income_to_asset_generation == pytest.approx(0.075, abs=0.005)

    def test_after_tax_margin(self, analyzer, sample_data):
        """NI/Rev = 150k/1M = 0.15."""
        result = analyzer.income_retention_analysis(sample_data)
        assert result.after_tax_margin == pytest.approx(0.15, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.income_retention_analysis(sample_data)
        assert result.ir_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.income_retention_analysis(sample_data)
        assert "Income Retention" in result.summary


# ===== SCORING TESTS =====


class TestIncomeRetentionScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """NtGR=0.375 => base 8.5. NtOR=0.75 no adj (<0.80). ATM=0.15 >=0.15 => +0.5. Score=9.0."""
        result = analyzer.income_retention_analysis(sample_data)
        assert result.ir_score == pytest.approx(9.0, abs=0.5)
        assert result.ir_grade == "Excellent"

    def test_excellent_retention(self, analyzer):
        """Very high NI/GP."""
        data = FinancialData(
            net_income=500_000,
            gross_profit=800_000,
            operating_income=600_000,
            ebitda=700_000,
            retained_earnings=2_000_000,
            total_assets=3_000_000,
            revenue=2_000_000,
        )
        result = analyzer.income_retention_analysis(data)
        assert result.ir_score >= 10.0
        assert result.ir_grade == "Excellent"

    def test_weak_retention(self, analyzer):
        """Very low NI/GP."""
        data = FinancialData(
            net_income=10_000,
            gross_profit=400_000,
            operating_income=50_000,
            ebitda=80_000,
            retained_earnings=20_000,
            total_assets=2_000_000,
            revenue=1_000_000,
        )
        # NtGR=10k/400k=0.025 <0.05 => base 1.0. NtOR=10k/50k=0.20 <0.50 => -0.5. ATM=10k/1M=0.01 <0.05 => -0.5. Score=0.0.
        result = analyzer.income_retention_analysis(data)
        assert result.ir_score <= 0.5
        assert result.ir_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase196EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.income_retention_analysis(FinancialData())
        assert isinstance(result, IncomeRetentionResult)
        assert result.ir_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => NtGR=None, score 0."""
        data = FinancialData(
            gross_profit=400_000,
            total_assets=2_000_000,
        )
        result = analyzer.income_retention_analysis(data)
        assert result.net_to_gross_ratio is None
        assert result.ir_score == 0.0

    def test_no_gross_profit(self, analyzer):
        """GP=None => NtGR=None."""
        data = FinancialData(
            net_income=150_000,
            total_assets=2_000_000,
        )
        result = analyzer.income_retention_analysis(data)
        assert result.net_to_gross_ratio is None
        assert result.ir_score == 0.0

    def test_no_operating_income(self, analyzer):
        """OI=None => NtOR=None, but NtGR still works."""
        data = FinancialData(
            net_income=150_000,
            gross_profit=400_000,
        )
        result = analyzer.income_retention_analysis(data)
        assert result.net_to_operating_ratio is None
        assert result.net_to_gross_ratio is not None
