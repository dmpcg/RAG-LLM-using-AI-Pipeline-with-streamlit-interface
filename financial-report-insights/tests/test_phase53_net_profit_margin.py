"""Phase 53 Tests: Net Profit Margin Analysis.

Tests for net_profit_margin_analysis() and NetProfitMarginResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    NetProfitMarginResult,
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


class TestNetProfitMarginDataclass:
    def test_defaults(self):
        r = NetProfitMarginResult()
        assert r.net_margin_pct is None
        assert r.ebitda_margin_pct is None
        assert r.ebit_margin_pct is None
        assert r.tax_burden is None
        assert r.interest_burden is None
        assert r.net_to_ebitda is None
        assert r.npm_score == 0.0
        assert r.npm_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = NetProfitMarginResult(
            net_margin_pct=15.0,
            npm_grade="Excellent",
        )
        assert r.net_margin_pct == 15.0
        assert r.npm_grade == "Excellent"


# ===== CORE COMPUTATION TESTS =====


class TestNetProfitMarginAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert isinstance(result, NetProfitMarginResult)

    def test_net_margin(self, analyzer, sample_data):
        """NI/Revenue = 150k/1M = 15%."""
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.net_margin_pct == pytest.approx(15.0, abs=0.1)

    def test_ebitda_margin(self, analyzer, sample_data):
        """EBITDA/Revenue = 250k/1M = 25%."""
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.ebitda_margin_pct == pytest.approx(25.0, abs=0.1)

    def test_ebit_margin(self, analyzer, sample_data):
        """EBIT/Revenue = 200k/1M = 20%."""
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.ebit_margin_pct == pytest.approx(20.0, abs=0.1)

    def test_tax_burden(self, analyzer, sample_data):
        """EBT = EBIT - IE = 200k - 30k = 170k. Tax burden = NI/EBT = 150k/170k = 0.882."""
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.tax_burden == pytest.approx(0.882, abs=0.01)

    def test_interest_burden(self, analyzer, sample_data):
        """EBT/EBIT = 170k/200k = 0.85."""
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.interest_burden == pytest.approx(0.85, abs=0.01)

    def test_net_to_ebitda(self, analyzer, sample_data):
        """NI/EBITDA = 150k/250k = 0.60."""
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.net_to_ebitda == pytest.approx(0.60, abs=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.npm_grade in ["Excellent", "Good", "Adequate", "Weak"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert "Net Profit Margin" in result.summary


# ===== SCORING TESTS =====


class TestNetProfitMarginScoring:
    def test_high_margin(self, analyzer):
        """NM 30% => base 10."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=300_000,
            ebit=400_000,
            ebitda=450_000,
            interest_expense=10_000,
        )
        result = analyzer.net_profit_margin_analysis(data)
        assert result.npm_score >= 8.0
        assert result.npm_grade == "Excellent"

    def test_moderate_margin(self, analyzer, sample_data):
        """NM 15% => base 8.5."""
        result = analyzer.net_profit_margin_analysis(sample_data)
        assert result.npm_score >= 6.0

    def test_thin_margin(self, analyzer):
        """NM 3% => base 3.5."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=30_000,
            ebit=100_000,
            ebitda=150_000,
            interest_expense=50_000,
        )
        result = analyzer.net_profit_margin_analysis(data)
        assert result.npm_score < 6.0

    def test_negative_margin(self, analyzer):
        """NM -15% => base 0.5, Weak."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=-150_000,
            ebit=-100_000,
            ebitda=-50_000,
            interest_expense=50_000,
        )
        result = analyzer.net_profit_margin_analysis(data)
        assert result.npm_score < 4.0
        assert result.npm_grade == "Weak"


# ===== EDGE CASES =====


class TestPhase53EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.net_profit_margin_analysis(FinancialData())
        assert isinstance(result, NetProfitMarginResult)
        assert result.net_margin_pct is None

    def test_no_revenue(self, analyzer):
        data = FinancialData(net_income=100_000)
        result = analyzer.net_profit_margin_analysis(data)
        assert result.net_margin_pct is None

    def test_no_net_income(self, analyzer):
        """Revenue but no NI => net_margin is None."""
        data = FinancialData(revenue=500_000, ebitda=100_000)
        result = analyzer.net_profit_margin_analysis(data)
        assert result.net_margin_pct is None

    def test_no_interest_expense(self, analyzer):
        """No IE => EBT = EBIT, interest_burden = 1.0."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=150_000,
            ebit=200_000,
            ebitda=250_000,
        )
        result = analyzer.net_profit_margin_analysis(data)
        assert result.interest_burden == pytest.approx(1.0, abs=0.01)
