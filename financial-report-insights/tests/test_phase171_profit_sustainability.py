"""Phase 171 Tests: Profit Sustainability Analysis.

Tests for profit_sustainability_analysis() and ProfitSustainabilityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ProfitSustainabilityResult,
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


class TestProfitSustainabilityDataclass:
    def test_defaults(self):
        r = ProfitSustainabilityResult()
        assert r.profit_cash_backing is None
        assert r.profit_margin_depth is None
        assert r.profit_reinvestment is None
        assert r.profit_to_asset is None
        assert r.profit_stability_proxy is None
        assert r.profit_leverage is None
        assert r.ps_score == 0.0
        assert r.ps_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ProfitSustainabilityResult(profit_cash_backing=1.20, ps_grade="Excellent")
        assert r.profit_cash_backing == 1.20
        assert r.ps_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestProfitSustainabilityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert isinstance(result, ProfitSustainabilityResult)

    def test_profit_cash_backing(self, analyzer, sample_data):
        """OCF/NI = 220k/150k = 1.467."""
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.profit_cash_backing == pytest.approx(1.467, abs=0.01)

    def test_profit_margin_depth(self, analyzer, sample_data):
        """NI/Rev = 150k/1M = 0.15."""
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.profit_margin_depth == pytest.approx(0.15, abs=0.01)

    def test_profit_reinvestment(self, analyzer, sample_data):
        """(NI-Div)/NI = (150k-40k)/150k = 0.733."""
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.profit_reinvestment == pytest.approx(0.733, abs=0.01)

    def test_profit_to_asset(self, analyzer, sample_data):
        """NI/TA = 150k/2M = 0.075."""
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.profit_to_asset == pytest.approx(0.075, abs=0.005)

    def test_profit_stability_proxy(self, analyzer, sample_data):
        """OI/EBITDA = 200k/250k = 0.80."""
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.profit_stability_proxy == pytest.approx(0.80, abs=0.01)

    def test_profit_leverage(self, analyzer, sample_data):
        """NI/OI = 150k/200k = 0.75."""
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.profit_leverage == pytest.approx(0.75, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.ps_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert "Profit Sustainability" in result.summary


# ===== SCORING TESTS =====


class TestProfitSustainabilityScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """PCB=1.467 => base 8.5. PMD=0.15 >=0.15 => +0.5. PRI=0.733 >=0.70 => +0.5. Score=9.5."""
        result = analyzer.profit_sustainability_analysis(sample_data)
        assert result.ps_score == pytest.approx(9.5, abs=0.5)
        assert result.ps_grade == "Excellent"

    def test_excellent_sustainability(self, analyzer):
        """Very high cash backing."""
        data = FinancialData(
            operating_cash_flow=500_000,
            net_income=200_000,
            revenue=2_000_000,
            dividends_paid=20_000,
            total_assets=1_000_000,
            operating_income=300_000,
            ebitda=400_000,
        )
        result = analyzer.profit_sustainability_analysis(data)
        assert result.ps_score >= 10.0
        assert result.ps_grade == "Excellent"

    def test_weak_sustainability(self, analyzer):
        """Very low cash backing."""
        data = FinancialData(
            operating_cash_flow=10_000,
            net_income=200_000,
            revenue=1_000_000,
            dividends_paid=180_000,
            total_assets=5_000_000,
            operating_income=250_000,
            ebitda=300_000,
        )
        # PCB=10k/200k=0.05 => base 1.0. PMD=200k/1M=0.20 >=0.15 => +0.5. PRI=(200k-180k)/200k=0.10 <0.30 => -0.5. Score=1.0.
        result = analyzer.profit_sustainability_analysis(data)
        assert result.ps_score <= 1.5
        assert result.ps_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase171EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.profit_sustainability_analysis(FinancialData())
        assert isinstance(result, ProfitSustainabilityResult)
        assert result.ps_score == 0.0

    def test_no_net_income(self, analyzer):
        """NI=None => PCB=None, PMD=None, PRI=None, score 0."""
        data = FinancialData(
            operating_cash_flow=220_000,
            revenue=1_000_000,
        )
        result = analyzer.profit_sustainability_analysis(data)
        assert result.profit_cash_backing is None
        assert result.ps_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => PCB=None."""
        data = FinancialData(
            net_income=150_000,
            revenue=1_000_000,
            dividends_paid=40_000,
        )
        result = analyzer.profit_sustainability_analysis(data)
        assert result.profit_cash_backing is None
        assert result.profit_margin_depth is not None

    def test_no_dividends(self, analyzer):
        """Div=None => PRI=None."""
        data = FinancialData(
            operating_cash_flow=220_000,
            net_income=150_000,
            revenue=1_000_000,
        )
        result = analyzer.profit_sustainability_analysis(data)
        assert result.profit_reinvestment is None
        assert result.profit_cash_backing is not None
