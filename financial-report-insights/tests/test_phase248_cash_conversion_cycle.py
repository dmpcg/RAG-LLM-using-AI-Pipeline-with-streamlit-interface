"""Phase 248 Tests: Cash Conversion Cycle Analysis.

Tests for cash_conversion_cycle_analysis() and CashConversionCycleResult dataclass.
"""

import pytest

from financial_analyzer import (
    CashConversionCycleResult,
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


class TestCashConversionCycleDataclass:
    def test_defaults(self):
        r = CashConversionCycleResult()
        assert r.dso is None
        assert r.dio is None
        assert r.dpo is None
        assert r.ccc is None
        assert r.ccc_to_revenue is None
        assert r.working_cap_days is None
        assert r.ccc_score == 0.0
        assert r.ccc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CashConversionCycleResult(ccc=45.0, ccc_grade="Good")
        assert r.ccc == 45.0
        assert r.ccc_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestCashConversionCycleAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert isinstance(result, CashConversionCycleResult)

    def test_dso(self, analyzer, sample_data):
        """DSO = AR*365/Rev = 150k*365/1M = 54.75 days."""
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert result.dso == pytest.approx(54.75, abs=0.5)

    def test_dio(self, analyzer, sample_data):
        """DIO = Inv*365/COGS = 100k*365/600k = 60.83 days."""
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert result.dio == pytest.approx(60.83, abs=0.5)

    def test_dpo(self, analyzer, sample_data):
        """DPO = AP*365/COGS = 80k*365/600k = 48.67 days."""
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert result.dpo == pytest.approx(48.67, abs=0.5)

    def test_ccc(self, analyzer, sample_data):
        """CCC = DSO + DIO - DPO = 54.75 + 60.83 - 48.67 = 66.92 days."""
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert result.ccc == pytest.approx(66.92, abs=1.0)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert result.ccc_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert "Cash Conversion Cycle" in result.summary


# ===== SCORING TESTS =====


class TestCashConversionCycleScoring:
    def test_sample_data_score(self, analyzer, sample_data):
        """CCC=66.92 in (60,90] => base 5.0. DPO=48.67>=30 => +0.5. DSO=54.75>30 => no. Score=5.5."""
        result = analyzer.cash_conversion_cycle_analysis(sample_data)
        assert result.ccc_score == pytest.approx(5.5, abs=0.5)
        assert result.ccc_grade == "Adequate"

    def test_negative_ccc_excellent(self, analyzer):
        """Negative CCC = prepaid model = excellent."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            accounts_receivable=10_000,
            inventory=20_000,
            accounts_payable=200_000,
        )
        # DSO=3.65, DIO=12.17, DPO=121.67. CCC=3.65+12.17-121.67=-105.85 <=0 => base 10.
        result = analyzer.cash_conversion_cycle_analysis(data)
        assert result.ccc < 0
        assert result.ccc_score >= 10.0
        assert result.ccc_grade == "Excellent"

    def test_weak_long_cycle(self, analyzer):
        """Very long CCC = weak."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            accounts_receivable=500_000,
            inventory=400_000,
            accounts_payable=20_000,
        )
        # DSO=182.5, DIO=243.33, DPO=12.17. CCC=413.67 >120 => base 1.5.
        result = analyzer.cash_conversion_cycle_analysis(data)
        assert result.ccc > 120
        assert result.ccc_score <= 2.0
        assert result.ccc_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase248EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cash_conversion_cycle_analysis(FinancialData())
        assert isinstance(result, CashConversionCycleResult)
        assert result.ccc_score == 0.0

    def test_no_revenue(self, analyzer):
        """Revenue=None => insufficient data => score 0."""
        data = FinancialData(cogs=600_000, accounts_receivable=100_000)
        result = analyzer.cash_conversion_cycle_analysis(data)
        assert result.ccc_score == 0.0

    def test_no_cogs(self, analyzer):
        """COGS=None => insufficient data => score 0."""
        data = FinancialData(revenue=1_000_000, accounts_receivable=100_000)
        result = analyzer.cash_conversion_cycle_analysis(data)
        assert result.ccc_score == 0.0

    def test_only_ar(self, analyzer):
        """Only AR present (no inv, no AP) => CCC = DSO."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            accounts_receivable=100_000,
        )
        result = analyzer.cash_conversion_cycle_analysis(data)
        assert result.dso is not None
        assert result.dio is None
        assert result.dpo is None
        assert result.ccc == pytest.approx(result.dso, abs=0.1)
