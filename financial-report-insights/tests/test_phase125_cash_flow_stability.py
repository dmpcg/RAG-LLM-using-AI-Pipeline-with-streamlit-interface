"""Phase 125 Tests: Cash Flow Stability Analysis.

Tests for cash_flow_stability_analysis() and CashFlowStabilityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CashFlowStabilityResult,
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


class TestCashFlowStabilityDataclass:
    def test_defaults(self):
        r = CashFlowStabilityResult()
        assert r.ocf_margin is None
        assert r.ocf_to_ebitda is None
        assert r.ocf_to_debt_service is None
        assert r.capex_to_ocf is None
        assert r.dividend_coverage is None
        assert r.cash_flow_sufficiency is None
        assert r.cfs_score == 0.0
        assert r.cfs_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CashFlowStabilityResult(ocf_margin=0.22, cfs_grade="Excellent")
        assert r.ocf_margin == 0.22
        assert r.cfs_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestCashFlowStabilityAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert isinstance(result, CashFlowStabilityResult)

    def test_ocf_margin(self, analyzer, sample_data):
        """OCF Margin = 220k / 1M = 0.22."""
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.ocf_margin == pytest.approx(0.22, abs=0.01)

    def test_ocf_to_ebitda(self, analyzer, sample_data):
        """OCF/EBITDA = 220k / 250k = 0.88."""
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.ocf_to_ebitda == pytest.approx(0.88, abs=0.01)

    def test_ocf_to_debt_service(self, analyzer, sample_data):
        """OCF/DS = 220k / 30k = 7.33."""
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.ocf_to_debt_service == pytest.approx(7.33, abs=0.1)

    def test_capex_to_ocf(self, analyzer, sample_data):
        """CapEx/OCF = 80k / 220k = 0.364."""
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.capex_to_ocf == pytest.approx(0.364, abs=0.01)

    def test_dividend_coverage(self, analyzer, sample_data):
        """DC = 220k / 40k = 5.5."""
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.dividend_coverage == pytest.approx(5.5, abs=0.1)

    def test_cash_flow_sufficiency(self, analyzer, sample_data):
        """CFS = 220k / (80k + 40k + 30k) = 220k / 150k = 1.467."""
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.cash_flow_sufficiency == pytest.approx(1.467, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.cfs_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert "Cash Flow Stability" in result.summary


# ===== SCORING TESTS =====


class TestCashFlowStabilityScoring:
    def test_strong_stability(self, analyzer, sample_data):
        """OCF Margin=0.22 => base 8.5. OCF/EBITDA=0.88 (not >=0.90) => 0. CFS=1.467 (not >=1.5) => 0. Score=8.5."""
        result = analyzer.cash_flow_stability_analysis(sample_data)
        assert result.cfs_score == pytest.approx(8.5, abs=0.5)
        assert result.cfs_grade == "Excellent"

    def test_excellent_stability(self, analyzer):
        """OCF Margin >= 0.25 => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=300_000,
            ebitda=300_000,
            interest_expense=10_000,
            capex=50_000,
            dividends_paid=20_000,
        )
        # OCF Margin=0.30 => 10. OCF/EBITDA=1.0 >=0.90 => +0.5. CFS=300k/80k=3.75 >=1.5 => +0.5. Capped 10.
        result = analyzer.cash_flow_stability_analysis(data)
        assert result.cfs_score >= 10.0
        assert result.cfs_grade == "Excellent"

    def test_weak_stability(self, analyzer):
        """OCF Margin < 0.05 => base 4.0 or less."""
        data = FinancialData(
            revenue=1_000_000,
            operating_cash_flow=20_000,
            ebitda=50_000,
            interest_expense=15_000,
            capex=30_000,
            dividends_paid=10_000,
        )
        # OCF Margin=0.02 => 2.5. OCF/EBITDA=0.40 <0.50 => -0.5. CFS=20k/55k=0.36 <1.0 => -0.5. Score=1.5.
        result = analyzer.cash_flow_stability_analysis(data)
        assert result.cfs_score <= 2.0
        assert result.cfs_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase125EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cash_flow_stability_analysis(FinancialData())
        assert isinstance(result, CashFlowStabilityResult)
        assert result.cfs_score == 0.0

    def test_no_ocf(self, analyzer):
        """OCF=None => all metrics None."""
        data = FinancialData(
            revenue=500_000,
            ebitda=100_000,
        )
        result = analyzer.cash_flow_stability_analysis(data)
        assert result.ocf_margin is None
        assert result.cfs_score == 0.0

    def test_no_interest(self, analyzer):
        """IE=None => OCF/DS=None."""
        data = FinancialData(
            revenue=500_000,
            operating_cash_flow=100_000,
            ebitda=120_000,
            capex=30_000,
        )
        result = analyzer.cash_flow_stability_analysis(data)
        assert result.ocf_to_debt_service is None
        assert result.ocf_margin is not None

    def test_no_dividends(self, analyzer):
        """Div=None => dividend_coverage=None, sufficiency still computed."""
        data = FinancialData(
            revenue=500_000,
            operating_cash_flow=100_000,
            ebitda=120_000,
            capex=30_000,
            interest_expense=10_000,
        )
        result = analyzer.cash_flow_stability_analysis(data)
        assert result.dividend_coverage is None
        assert result.cash_flow_sufficiency is not None
