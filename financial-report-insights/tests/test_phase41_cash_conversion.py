"""Phase 41 Tests: Cash Conversion Efficiency.

Tests for cash_conversion_analysis() and CashConversionResult dataclass.
"""

import pytest

from financial_analyzer import (
    CashConversionResult,
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
    )


# ===== DATACLASS TESTS =====


class TestCashConversionDataclass:
    def test_defaults(self):
        r = CashConversionResult()
        assert r.ccc is None
        assert r.dso is None
        assert r.dio is None
        assert r.dpo is None
        assert r.cash_conversion_score == 0.0
        assert r.cash_conversion_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = CashConversionResult(
            ccc=45.0,
            cash_conversion_grade="Good",
        )
        assert r.ccc == 45.0
        assert r.cash_conversion_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestCashConversionAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.cash_conversion_analysis(sample_data)
        assert isinstance(result, CashConversionResult)

    def test_dso(self, analyzer, sample_data):
        """DSO = 150k / 1M * 365 = 54.75 days."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.dso == pytest.approx(54.75, abs=1.0)

    def test_dio(self, analyzer, sample_data):
        """DIO = 100k / 600k * 365 = 60.83 days."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.dio == pytest.approx(60.83, abs=1.0)

    def test_dpo(self, analyzer, sample_data):
        """DPO = 80k / 600k * 365 = 48.67 days."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.dpo == pytest.approx(48.67, abs=1.0)

    def test_ccc(self, analyzer, sample_data):
        """CCC = DSO + DIO - DPO = 54.75 + 60.83 - 48.67 = 66.92 days."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.ccc == pytest.approx(66.92, abs=2.0)

    def test_cash_to_revenue(self, analyzer, sample_data):
        """Cash/Revenue = 50k / 1M = 5%."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.cash_to_revenue == pytest.approx(0.05, abs=0.005)

    def test_ocf_to_revenue(self, analyzer, sample_data):
        """OCF/Revenue = 220k / 1M = 22%."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.ocf_to_revenue == pytest.approx(0.22, abs=0.01)

    def test_ocf_to_ebitda(self, analyzer, sample_data):
        """OCF/EBITDA = 220k / 250k = 88%."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.ocf_to_ebitda == pytest.approx(0.88, abs=0.02)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.cash_conversion_grade in ["Excellent", "Good", "Fair", "Poor"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.cash_conversion_analysis(sample_data)
        assert "CCC" in result.summary or "Cash Conversion" in result.summary


# ===== SCORING TESTS =====


class TestCashConversionScoring:
    def test_excellent(self, analyzer):
        """Short CCC + high OCF ratios => Excellent."""
        data = FinancialData(
            revenue=2_000_000,
            cogs=1_000_000,
            accounts_receivable=50_000,  # DSO = 50k/2M*365 = 9.1 days (<30: +0.5)
            inventory=30_000,  # DIO = 30k/1M*365 = 10.95 days
            accounts_payable=100_000,  # DPO = 100k/1M*365 = 36.5 days
            # CCC = 9.1 + 10.95 - 36.5 = -16.4 (<30: +2.0)
            operating_cash_flow=500_000,  # OCF/Rev = 25% (>=0.20: +1.0)
            ebitda=400_000,  # OCF/EBITDA = 125% (>=0.80: +0.5)
            cash=200_000,
        )
        result = analyzer.cash_conversion_analysis(data)
        # Score = 5+2+1+0.5+0.5 = 9.0
        assert result.cash_conversion_score >= 8.0
        assert result.cash_conversion_grade == "Excellent"

    def test_poor(self, analyzer):
        """Long CCC + low OCF => Poor."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=500_000,
            accounts_receivable=350_000,  # DSO = 127.75 days (>90: -0.5)
            inventory=200_000,  # DIO = 146 days
            accounts_payable=30_000,  # DPO = 21.9 days
            # CCC = 127.75+146-21.9 = 251.8 (>120: -2.0)
            operating_cash_flow=-50_000,  # OCF/Rev = -5% (<0: -1.0)
            ebitda=100_000,  # OCF/EBITDA = -50% (<0.50: -0.5)
            cash=10_000,
        )
        result = analyzer.cash_conversion_analysis(data)
        # Score = 5-2-1-0.5-0.5 = 1.0
        assert result.cash_conversion_score < 4.0
        assert result.cash_conversion_grade == "Poor"


# ===== EDGE CASES =====


class TestPhase41EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.cash_conversion_analysis(FinancialData())
        assert isinstance(result, CashConversionResult)
        assert result.ccc is None

    def test_no_ar(self, analyzer):
        """No AR => DSO is None, CCC partial."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            inventory=100_000,
            accounts_payable=80_000,
        )
        result = analyzer.cash_conversion_analysis(data)
        assert result.dso is None
        # Without DSO, CCC should be None
        assert result.ccc is None

    def test_no_inventory(self, analyzer):
        """No inventory => DIO is None."""
        data = FinancialData(
            revenue=1_000_000,
            cogs=600_000,
            accounts_receivable=150_000,
            accounts_payable=80_000,
        )
        result = analyzer.cash_conversion_analysis(data)
        assert result.dio is None

    def test_no_cogs(self, analyzer):
        """No COGS => DIO and DPO are None."""
        data = FinancialData(
            revenue=1_000_000,
            accounts_receivable=100_000,
            inventory=50_000,
            accounts_payable=30_000,
            operating_cash_flow=200_000,
            ebitda=250_000,
        )
        result = analyzer.cash_conversion_analysis(data)
        assert result.dso is not None
        assert result.dio is None
        assert result.dpo is None

    def test_sample_data_score(self, analyzer, sample_data):
        """CCC ~67 days (60-90: no adj), OCF/Rev 22% (>=0.20: +1.0),
        OCF/EBITDA 88% (>=0.80: +0.5), DSO ~55 (30-90: no adj).
        Score = 5.0 + 0 + 1.0 + 0.5 + 0 = 6.5 => Good."""
        result = analyzer.cash_conversion_analysis(sample_data)
        assert result.cash_conversion_score == pytest.approx(6.5, abs=0.3)
        assert result.cash_conversion_grade == "Good"
