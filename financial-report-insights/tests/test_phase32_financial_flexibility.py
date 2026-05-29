"""Phase 32 Tests: Financial Flexibility Analysis.

Tests for financial_flexibility_analysis() and FinancialFlexibilityResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    FinancialFlexibilityResult,
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


class TestFinancialFlexibilityDataclass:
    def test_defaults(self):
        r = FinancialFlexibilityResult()
        assert r.cash_to_assets is None
        assert r.debt_headroom is None
        assert r.flexibility_score == 0.0
        assert r.flexibility_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = FinancialFlexibilityResult(
            cash_to_assets=0.15,
            flexibility_grade="Flexible",
        )
        assert r.cash_to_assets == 0.15
        assert r.flexibility_grade == "Flexible"


# ===== CORE COMPUTATION TESTS =====


class TestFinancialFlexibility:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert isinstance(result, FinancialFlexibilityResult)

    def test_cash_to_assets(self, analyzer, sample_data):
        """Cash/TA = 50k / 2M = 2.5%."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.cash_to_assets == pytest.approx(0.025, rel=0.01)

    def test_cash_to_debt(self, analyzer, sample_data):
        """Cash/Debt = 50k / 400k = 12.5%."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.cash_to_debt == pytest.approx(0.125, rel=0.01)

    def test_cash_to_revenue(self, analyzer, sample_data):
        """Cash/Revenue = 50k / 1M = 5%."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.cash_to_revenue == pytest.approx(0.05, rel=0.01)

    def test_fcf_to_revenue(self, analyzer, sample_data):
        """FCF/Rev = (220k-80k)/1M = 14%."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.fcf_to_revenue == pytest.approx(0.14, rel=0.01)

    def test_spare_borrowing_capacity(self, analyzer, sample_data):
        """Spare = (TA*0.50 - Debt) / TA = (2M*0.50 - 400k) / 2M = 30%."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.spare_borrowing_capacity == pytest.approx(0.30, rel=0.01)

    def test_unencumbered_assets(self, analyzer, sample_data):
        """TA - TL = 2M - 800k = 1.2M."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.unencumbered_assets == pytest.approx(1_200_000)

    def test_financial_slack(self, analyzer, sample_data):
        """(Cash + FCF) / Debt = (50k + 140k) / 400k = 0.475x."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.financial_slack == pytest.approx(0.475, rel=0.01)

    def test_debt_headroom(self, analyzer, sample_data):
        """3x EBITDA - Debt = 3*250k - 400k = 350k."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.debt_headroom == pytest.approx(350_000)

    def test_retained_earnings_ratio(self, analyzer, sample_data):
        """RE/Equity = 600k / 1.2M = 50%."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.retained_earnings_ratio == pytest.approx(0.50, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.flexibility_grade in ["Highly Flexible", "Flexible", "Constrained", "Rigid"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert "Financial Flexibility" in result.summary


# ===== SCORING TESTS =====


class TestFinancialFlexibilityScoring:
    def test_highly_flexible(self, analyzer):
        """High cash, strong FCF, plenty of debt headroom."""
        data = FinancialData(
            cash=5_000_000,
            total_assets=10_000_000,
            total_debt=1_000_000,
            revenue=8_000_000,
            operating_cash_flow=2_000_000,
            capex=500_000,
            ebitda=3_000_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        # Cash/TA=50% (>=0.20: +1.5->6.5)
        # FCF/Rev=(2M-500k)/8M=18.75% (>=0.15: +1.5->8.0)
        # Headroom=9M-1M=8M (>0: +1.0->9.0)
        # Slack=(5M+1.5M)/1M=6.5 (>=1.0: +0.5->9.5)
        assert result.flexibility_grade == "Highly Flexible"
        assert result.flexibility_score >= 8.0

    def test_rigid(self, analyzer):
        """Minimal cash, negative FCF, over-leveraged."""
        data = FinancialData(
            cash=10_000,
            total_assets=2_000_000,
            total_debt=3_000_000,
            revenue=1_000_000,
            operating_cash_flow=50_000,
            capex=100_000,
            ebitda=200_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        # Cash/TA=0.5% (<0.03: -1.0->4.0)
        # FCF/Rev=-5% (<0: -1.0->3.0)
        # Headroom=600k-3M=-2.4M (<0: -1.0->2.0)
        # Slack=(10k+(-50k))/3M=-0.013 (<0.2: -0.5->1.5)
        assert result.flexibility_grade in ["Rigid", "Constrained"]
        assert result.flexibility_score <= 4.0

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            cash=10_000_000,
            total_assets=15_000_000,
            total_debt=500_000,
            revenue=5_000_000,
            operating_cash_flow=3_000_000,
            capex=200_000,
            ebitda=4_000_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        assert result.flexibility_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            cash=1_000,
            total_assets=500_000,
            total_debt=5_000_000,
            revenue=100_000,
            operating_cash_flow=-50_000,
            capex=30_000,
            ebitda=20_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        assert result.flexibility_score >= 0.0


# ===== EDGE CASES =====


class TestPhase32EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.financial_flexibility_analysis(FinancialData())
        assert isinstance(result, FinancialFlexibilityResult)
        assert result.cash_to_assets is None
        assert result.debt_headroom is None

    def test_no_debt(self, analyzer):
        """No debt => debt ratios None, but cash metrics still work."""
        data = FinancialData(
            cash=500_000,
            total_assets=2_000_000,
            revenue=1_000_000,
            operating_cash_flow=300_000,
            capex=50_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        assert result.cash_to_debt is None
        assert result.financial_slack is None
        assert result.cash_to_assets is not None

    def test_no_cash_fallback_to_nwc(self, analyzer):
        """No cash field => cash = CA - CL."""
        data = FinancialData(
            current_assets=400_000,
            current_liabilities=100_000,
            total_assets=2_000_000,
            total_debt=300_000,
            revenue=1_000_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        # Cash fallback = 400k - 100k = 300k
        # Cash/TA = 300k / 2M = 15%
        assert result.cash_to_assets == pytest.approx(0.15, rel=0.01)

    def test_no_capex(self, analyzer):
        """No capex => FCF is None."""
        data = FinancialData(
            cash=200_000,
            total_assets=1_000_000,
            total_debt=300_000,
            operating_cash_flow=150_000,
            revenue=500_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        assert result.fcf_to_revenue is None
        assert result.financial_slack is None  # needs FCF

    def test_negative_debt_headroom(self, analyzer):
        """Debt > 3x EBITDA => negative headroom."""
        data = FinancialData(
            total_debt=2_000_000,
            ebitda=500_000,
            cash=100_000,
            total_assets=3_000_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        # Headroom = 3*500k - 2M = -500k
        assert result.debt_headroom == pytest.approx(-500_000)

    def test_no_ebitda(self, analyzer):
        """No EBITDA => headroom is None."""
        data = FinancialData(
            cash=200_000,
            total_assets=1_000_000,
            total_debt=300_000,
        )
        result = analyzer.financial_flexibility_analysis(data)
        assert result.debt_headroom is None

    def test_sample_data_score(self, analyzer, sample_data):
        """Cash/TA=2.5% (<3%: -1.0->4.0), FCF/Rev=14% (>=5%: +0.5->4.5),
        Headroom=350k (>0: +1.0->5.5), Slack=0.475 (>=0.2 but <1.0: no adj) => 5.5 Constrained."""
        result = analyzer.financial_flexibility_analysis(sample_data)
        assert result.flexibility_score == pytest.approx(5.5, abs=0.1)
        assert result.flexibility_grade == "Constrained"
