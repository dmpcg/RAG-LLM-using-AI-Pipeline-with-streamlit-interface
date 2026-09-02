"""Phase 182 Tests: Structural Strength Analysis.

Tests for structural_strength_analysis() and StructuralStrengthResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    StructuralStrengthResult,
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


class TestStructuralStrengthDataclass:
    def test_defaults(self):
        r = StructuralStrengthResult()
        assert r.equity_multiplier is None
        assert r.debt_to_equity is None
        assert r.liability_composition is None
        assert r.equity_cushion is None
        assert r.fixed_asset_coverage is None
        assert r.financial_leverage_ratio is None
        assert r.ss_score == 0.0
        assert r.ss_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = StructuralStrengthResult(equity_multiplier=1.67, ss_grade="Good")
        assert r.equity_multiplier == 1.67
        assert r.ss_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestStructuralStrengthAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.structural_strength_analysis(sample_data)
        assert isinstance(result, StructuralStrengthResult)

    def test_equity_multiplier(self, analyzer, sample_data):
        """TA/TE = 2M/1.2M = 1.667."""
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.equity_multiplier == pytest.approx(1.667, abs=0.005)

    def test_debt_to_equity(self, analyzer, sample_data):
        """TD/TE = 400k/1.2M = 0.333."""
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.debt_to_equity == pytest.approx(0.333, abs=0.005)

    def test_liability_composition(self, analyzer, sample_data):
        """CL/TL = 200k/800k = 0.25."""
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.liability_composition == pytest.approx(0.25, abs=0.005)

    def test_equity_cushion(self, analyzer, sample_data):
        """(TE-TD)/TA = (1.2M-400k)/2M = 0.40."""
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.equity_cushion == pytest.approx(0.40, abs=0.005)

    def test_fixed_asset_coverage(self, analyzer, sample_data):
        """TE/(TA-CA) = 1.2M/(2M-500k) = 1.2M/1.5M = 0.80."""
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.fixed_asset_coverage == pytest.approx(0.80, abs=0.01)

    def test_financial_leverage_ratio(self, analyzer, sample_data):
        """TL/TE = 800k/1.2M = 0.667."""
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.financial_leverage_ratio == pytest.approx(0.667, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.ss_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.structural_strength_analysis(sample_data)
        assert "Structural Strength" in result.summary


# ===== SCORING TESTS =====


class TestStructuralStrengthScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """EM=1.667 => base 7.0. DtE=0.333 <=0.50 => +0.5. EC=0.40 >=0.30 => +0.5. Score=8.0."""
        result = analyzer.structural_strength_analysis(sample_data)
        assert result.ss_score == pytest.approx(8.0, abs=0.5)
        assert result.ss_grade == "Excellent"

    def test_excellent_strength(self, analyzer):
        """Very low equity multiplier."""
        data = FinancialData(
            total_assets=1_000_000,
            total_equity=900_000,
            total_debt=50_000,
            total_liabilities=100_000,
            current_liabilities=50_000,
            current_assets=400_000,
        )
        result = analyzer.structural_strength_analysis(data)
        assert result.ss_score >= 10.0
        assert result.ss_grade == "Excellent"

    def test_weak_strength(self, analyzer):
        """Very high equity multiplier."""
        data = FinancialData(
            total_assets=2_000_000,
            total_equity=400_000,
            total_debt=1_200_000,
            total_liabilities=1_600_000,
            current_liabilities=800_000,
            current_assets=500_000,
        )
        # EM=2M/400k=5.0 >3.50 => base 1.0. DtE=1.2M/400k=3.0 >=2.00 => -0.5. EC=(400k-1.2M)/2M=-0.40 <0.10 => -0.5. Score=0.0.
        result = analyzer.structural_strength_analysis(data)
        assert result.ss_score <= 1.0
        assert result.ss_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase182EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.structural_strength_analysis(FinancialData())
        assert isinstance(result, StructuralStrengthResult)
        assert result.ss_score == 0.0

    def test_no_total_equity(self, analyzer):
        """TE=None => EM=None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            total_debt=400_000,
        )
        result = analyzer.structural_strength_analysis(data)
        assert result.equity_multiplier is None
        assert result.ss_score == 0.0

    def test_no_total_assets(self, analyzer):
        """TA=None => EM=None."""
        data = FinancialData(
            total_equity=1_200_000,
            total_debt=400_000,
        )
        result = analyzer.structural_strength_analysis(data)
        assert result.equity_multiplier is None
        assert result.ss_score == 0.0

    def test_no_debt(self, analyzer):
        """TD=None => DtE=None, EC=None."""
        data = FinancialData(
            total_assets=2_000_000,
            total_equity=1_200_000,
            total_liabilities=800_000,
            current_liabilities=200_000,
            current_assets=500_000,
        )
        result = analyzer.structural_strength_analysis(data)
        assert result.debt_to_equity is None
        assert result.equity_cushion is None
        assert result.equity_multiplier is not None
