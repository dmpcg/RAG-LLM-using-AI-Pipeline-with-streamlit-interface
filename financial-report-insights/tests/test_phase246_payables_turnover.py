"""Phase 246 Tests: Payables Turnover Analysis.

Tests for payables_turnover_analysis() and PayablesTurnoverResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    PayablesTurnoverResult,
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


class TestPayablesTurnoverDataclass:
    def test_defaults(self):
        r = PayablesTurnoverResult()
        assert r.cogs_to_ap is None
        assert r.dpo is None
        assert r.ap_to_cl is None
        assert r.ap_to_tl is None
        assert r.ap_to_cogs is None
        assert r.payment_velocity is None
        assert r.pto_score == 0.0
        assert r.pto_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = PayablesTurnoverResult(cogs_to_ap=8.0, pto_grade="Excellent")
        assert r.cogs_to_ap == 8.0
        assert r.pto_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestPayablesTurnoverAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.payables_turnover_analysis(sample_data)
        assert isinstance(result, PayablesTurnoverResult)

    def test_cogs_to_ap(self, analyzer, sample_data):
        """COGS/AP = 600k/80k = 7.5."""
        result = analyzer.payables_turnover_analysis(sample_data)
        assert result.cogs_to_ap == pytest.approx(7.5, abs=0.01)

    def test_dpo(self, analyzer, sample_data):
        """DPO = 80k*365/600k = 48.67 days."""
        result = analyzer.payables_turnover_analysis(sample_data)
        assert result.dpo == pytest.approx(48.67, abs=0.5)

    def test_ap_to_cl(self, analyzer, sample_data):
        """AP/CL = 80k/200k = 0.40."""
        result = analyzer.payables_turnover_analysis(sample_data)
        assert result.ap_to_cl == pytest.approx(0.40, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.payables_turnover_analysis(sample_data)
        assert result.pto_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.payables_turnover_analysis(sample_data)
        assert "Payables Turnover" in result.summary


# ===== SCORING TESTS =====


class TestPayablesTurnoverScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """COGS/AP=7.5 in [6,12] => base 10. DPO=48.67 in [30,60] => +0.5. AP/CL=0.40 <=0.30? No. Score~10.5 clamped to 10."""
        result = analyzer.payables_turnover_analysis(sample_data)
        assert result.pto_score >= 10.0
        assert result.pto_grade == "Excellent"

    def test_excellent_ideal_range(self, analyzer):
        """COGS/AP in ideal 6-12 range."""
        data = FinancialData(
            cogs=900_000,
            accounts_payable=100_000,
            current_liabilities=500_000,
        )
        # COGS/AP=9.0 in [6,12] => base 10. DPO=40.6 in [30,60] => +0.5. AP/CL=0.20 <=0.30 => +0.5. Score=10.
        result = analyzer.payables_turnover_analysis(data)
        assert result.pto_score >= 10.0
        assert result.pto_grade == "Excellent"

    def test_weak_very_slow(self, analyzer):
        """Very slow payment (low COGS/AP)."""
        data = FinancialData(
            cogs=200_000,
            accounts_payable=300_000,
            current_liabilities=400_000,
            total_liabilities=500_000,
        )
        # COGS/AP=0.667 <1.0 => base 1.0. DPO=548 >120 => -0.5. AP/CL=0.75 >0.70 => -0.5. Score=0.
        result = analyzer.payables_turnover_analysis(data)
        assert result.pto_score <= 0.5
        assert result.pto_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase246EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.payables_turnover_analysis(FinancialData())
        assert isinstance(result, PayablesTurnoverResult)
        assert result.pto_score == 0.0

    def test_no_ap(self, analyzer):
        """AP=None => insufficient data => score 0."""
        data = FinancialData(cogs=600_000)
        result = analyzer.payables_turnover_analysis(data)
        assert result.pto_score == 0.0

    def test_no_cogs(self, analyzer):
        """COGS=None => insufficient data => score 0."""
        data = FinancialData(accounts_payable=80_000)
        result = analyzer.payables_turnover_analysis(data)
        assert result.pto_score == 0.0

    def test_zero_ap(self, analyzer):
        """AP=0 => insufficient data => score 0."""
        data = FinancialData(
            cogs=600_000,
            accounts_payable=0,
        )
        result = analyzer.payables_turnover_analysis(data)
        assert result.pto_score == 0.0
