"""Phase 124 Tests: Income Quality Analysis.

Tests for income_quality_analysis() and IncomeQualityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    IncomeQualityResult,
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


class TestIncomeQualityDataclass:
    def test_defaults(self):
        r = IncomeQualityResult()
        assert r.ocf_to_net_income is None
        assert r.accruals_ratio is None
        assert r.cash_earnings_ratio is None
        assert r.non_cash_ratio is None
        assert r.earnings_persistence is None
        assert r.operating_income_ratio is None
        assert r.iq_score == 0.0
        assert r.iq_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = IncomeQualityResult(ocf_to_net_income=1.5, iq_grade="Excellent")
        assert r.ocf_to_net_income == 1.5
        assert r.iq_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestIncomeQualityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.income_quality_analysis(sample_data)
        assert isinstance(result, IncomeQualityResult)

    def test_ocf_to_net_income(self, analyzer, sample_data):
        """OCF/NI = 220k / 150k = 1.467."""
        result = analyzer.income_quality_analysis(sample_data)
        assert result.ocf_to_net_income == pytest.approx(1.467, abs=0.01)

    def test_accruals_ratio(self, analyzer, sample_data):
        """AR = (150k - 220k) / 2M = -0.035."""
        result = analyzer.income_quality_analysis(sample_data)
        assert result.accruals_ratio == pytest.approx(-0.035, abs=0.01)

    def test_cash_earnings_ratio(self, analyzer, sample_data):
        """CER = 220k / 250k = 0.88."""
        result = analyzer.income_quality_analysis(sample_data)
        assert result.cash_earnings_ratio == pytest.approx(0.88, abs=0.01)

    def test_non_cash_ratio(self, analyzer, sample_data):
        """NCR = 50k / 150k = 0.333."""
        result = analyzer.income_quality_analysis(sample_data)
        assert result.non_cash_ratio == pytest.approx(0.333, abs=0.01)

    def test_earnings_persistence(self, analyzer, sample_data):
        """EP = 200k / 1M = 0.20."""
        result = analyzer.income_quality_analysis(sample_data)
        assert result.earnings_persistence == pytest.approx(0.20, abs=0.01)

    def test_operating_income_ratio(self, analyzer, sample_data):
        """OIR = 200k / 150k = 1.333."""
        result = analyzer.income_quality_analysis(sample_data)
        assert result.operating_income_ratio == pytest.approx(1.333, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.income_quality_analysis(sample_data)
        assert result.iq_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.income_quality_analysis(sample_data)
        assert "Income Quality" in result.summary


# ===== SCORING TESTS =====


class TestIncomeQualityScoring:
    def test_strong_quality(self, analyzer, sample_data):
        """OCF/NI=1.467 => base 8.5. Accruals=-0.035 (not <=-0.05) => 0. CER=0.88 (not >=0.90, not <0.50) => 0. Score=8.5."""
        result = analyzer.income_quality_analysis(sample_data)
        assert result.iq_score == pytest.approx(8.5, abs=0.5)
        assert result.iq_grade == "Excellent"

    def test_excellent_quality(self, analyzer):
        """OCF/NI >= 1.5 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=250_000,
            net_income=100_000,
            operating_cash_flow=200_000,
            ebitda=200_000,
            depreciation=40_000,
            total_assets=1_000_000,
        )
        # OCF/NI=2.0 => 10. Accruals=(100k-200k)/1M=-0.10 <=-0.05 => +0.5. CER=200k/200k=1.0 >=0.90 => +0.5. Capped 10.
        result = analyzer.income_quality_analysis(data)
        assert result.iq_score >= 10.0
        assert result.iq_grade == "Excellent"

    def test_poor_quality(self, analyzer):
        """OCF/NI < 0.5 => base 4.0 or less."""
        data = FinancialData(
            revenue=500_000,
            operating_income=50_000,
            net_income=100_000,
            operating_cash_flow=30_000,
            ebitda=80_000,
            depreciation=10_000,
            total_assets=1_000_000,
        )
        # OCF/NI=0.3 => 2.5. Accruals=(100k-30k)/1M=0.07 (not >0.10) => 0. CER=30k/80k=0.375 <0.50 => -0.5. Score=2.0.
        result = analyzer.income_quality_analysis(data)
        assert result.iq_score <= 3.0
        assert result.iq_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase124EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.income_quality_analysis(FinancialData())
        assert isinstance(result, IncomeQualityResult)
        assert result.iq_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => OCF/NI=None."""
        data = FinancialData(
            revenue=500_000,
            net_income=100_000,
            total_assets=1_000_000,
        )
        result = analyzer.income_quality_analysis(data)
        assert result.ocf_to_net_income is None
        assert result.iq_score == 0.0

    def test_zero_net_income(self, analyzer):
        """NI=0 => OCF/NI=None, all NI-based=None."""
        data = FinancialData(
            revenue=500_000,
            net_income=0,
            operating_cash_flow=100_000,
            total_assets=1_000_000,
        )
        result = analyzer.income_quality_analysis(data)
        assert result.ocf_to_net_income is None
        assert result.iq_score == 0.0

    def test_negative_ocf(self, analyzer):
        """OCF < 0 => OCF/NI < 0 => score 1.0."""
        data = FinancialData(
            revenue=500_000,
            net_income=50_000,
            operating_cash_flow=-20_000,
            ebitda=60_000,
            total_assets=1_000_000,
        )
        result = analyzer.income_quality_analysis(data)
        assert result.ocf_to_net_income < 0
        assert result.iq_score <= 1.5
