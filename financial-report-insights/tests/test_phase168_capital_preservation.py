"""Phase 168 Tests: Capital Preservation Analysis.

Tests for capital_preservation_analysis() and CapitalPreservationResult dataclass.
"""

import pytest

from financial_analyzer import (
    CapitalPreservationResult,
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


class TestCapitalPreservationDataclass:
    def test_defaults(self):
        r = CapitalPreservationResult()
        assert r.retained_earnings_power is None
        assert r.capital_erosion_rate is None
        assert r.asset_integrity_ratio is None
        assert r.operating_capital_ratio is None
        assert r.net_worth_growth_proxy is None
        assert r.capital_buffer is None
        assert r.cp_score == 0.0
        assert r.cp_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CapitalPreservationResult(retained_earnings_power=0.30, cp_grade="Good")
        assert r.retained_earnings_power == 0.30
        assert r.cp_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestCapitalPreservationAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.capital_preservation_analysis(sample_data)
        assert isinstance(result, CapitalPreservationResult)

    def test_retained_earnings_power(self, analyzer, sample_data):
        """RE/TA = 600k/2M = 0.30."""
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.retained_earnings_power == pytest.approx(0.30, abs=0.01)

    def test_capital_erosion_rate(self, analyzer, sample_data):
        """(TL-Cash)/TE = (800k-50k)/1.2M = 0.625."""
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.capital_erosion_rate == pytest.approx(0.625, abs=0.01)

    def test_asset_integrity_ratio(self, analyzer, sample_data):
        """(TA-TL)/TA = (2M-800k)/2M = 0.60."""
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.asset_integrity_ratio == pytest.approx(0.60, abs=0.01)

    def test_operating_capital_ratio(self, analyzer, sample_data):
        """OCF/TD = 220k/400k = 0.55."""
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.operating_capital_ratio == pytest.approx(0.55, abs=0.01)

    def test_net_worth_growth_proxy(self, analyzer, sample_data):
        """NI/TE = 150k/1.2M = 0.125."""
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.net_worth_growth_proxy == pytest.approx(0.125, abs=0.01)

    def test_capital_buffer(self, analyzer, sample_data):
        """(CA-CL)/TA = (500k-200k)/2M = 0.15."""
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.capital_buffer == pytest.approx(0.15, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.cp_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.capital_preservation_analysis(sample_data)
        assert "Capital Preservation" in result.summary


# ===== SCORING TESTS =====


class TestCapitalPreservationScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """REP=0.30 => base 7.0. CER=0.625 no adj. OCR=0.55 >=0.50 => +0.5. Score=7.5."""
        result = analyzer.capital_preservation_analysis(sample_data)
        assert result.cp_score == pytest.approx(7.5, abs=0.5)
        assert result.cp_grade == "Good"

    def test_excellent_preservation(self, analyzer):
        """Very high retained earnings power."""
        data = FinancialData(
            retained_earnings=900_000,
            total_assets=1_500_000,
            total_liabilities=300_000,
            total_equity=1_200_000,
            cash=200_000,
            operating_cash_flow=500_000,
            total_debt=200_000,
            net_income=300_000,
            current_assets=600_000,
            current_liabilities=100_000,
        )
        result = analyzer.capital_preservation_analysis(data)
        assert result.cp_score >= 10.0
        assert result.cp_grade == "Excellent"

    def test_weak_preservation(self, analyzer):
        """Very low retained earnings power."""
        data = FinancialData(
            retained_earnings=50_000,
            total_assets=2_000_000,
            total_liabilities=1_800_000,
            total_equity=200_000,
            cash=10_000,
            operating_cash_flow=20_000,
            total_debt=1_500_000,
            net_income=-50_000,
            current_assets=100_000,
            current_liabilities=300_000,
        )
        # REP=50k/2M=0.025 => base 1.0. CER=(1.8M-10k)/200k=8.95 >1.50 => -0.5. OCR=20k/1.5M=0.013 <0.10 => -0.5. Score=0.0.
        result = analyzer.capital_preservation_analysis(data)
        assert result.cp_score <= 1.0
        assert result.cp_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase168EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.capital_preservation_analysis(FinancialData())
        assert isinstance(result, CapitalPreservationResult)
        assert result.cp_score == 0.0

    def test_no_retained_earnings(self, analyzer):
        """RE=None => REP=None, score 0."""
        data = FinancialData(
            total_assets=2_000_000,
            total_liabilities=800_000,
            total_equity=1_200_000,
        )
        result = analyzer.capital_preservation_analysis(data)
        assert result.retained_earnings_power is None
        assert result.cp_score == 0.0

    def test_no_total_assets(self, analyzer):
        """TA=None => REP=None, AIR=None, CB=None."""
        data = FinancialData(
            retained_earnings=600_000,
            total_liabilities=800_000,
            total_equity=1_200_000,
        )
        result = analyzer.capital_preservation_analysis(data)
        assert result.retained_earnings_power is None
        assert result.asset_integrity_ratio is None
        assert result.capital_buffer is None

    def test_no_total_debt(self, analyzer):
        """TD=None => OCR=None."""
        data = FinancialData(
            retained_earnings=600_000,
            total_assets=2_000_000,
            operating_cash_flow=220_000,
        )
        result = analyzer.capital_preservation_analysis(data)
        assert result.operating_capital_ratio is None
        assert result.retained_earnings_power is not None
