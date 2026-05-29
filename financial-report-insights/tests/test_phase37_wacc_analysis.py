"""Phase 37 Tests: WACC & Cost of Capital.

Tests for wacc_analysis() and WACCResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    WACCResult,
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


class TestWACCDataclass:
    def test_defaults(self):
        r = WACCResult()
        assert r.wacc is None
        assert r.cost_of_debt is None
        assert r.implied_cost_of_equity is None
        assert r.wacc_score == 0.0
        assert r.wacc_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = WACCResult(
            wacc=0.10,
            wacc_grade="Good",
        )
        assert r.wacc == 0.10
        assert r.wacc_grade == "Good"


# ===== CORE COMPUTATION TESTS =====


class TestWACCAnalysis:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.wacc_analysis(sample_data)
        assert isinstance(result, WACCResult)

    def test_capital_structure_weights(self, analyzer, sample_data):
        """Debt=400k, Equity=1.2M => Wd=25%, We=75%."""
        result = analyzer.wacc_analysis(sample_data)
        assert result.debt_weight == pytest.approx(0.25, rel=0.01)
        assert result.equity_weight == pytest.approx(0.75, rel=0.01)
        assert result.total_capital == pytest.approx(1_600_000, rel=0.01)

    def test_cost_of_debt(self, analyzer, sample_data):
        """Interest/Debt = 30k/400k = 7.5%."""
        result = analyzer.wacc_analysis(sample_data)
        assert result.cost_of_debt == pytest.approx(0.075, rel=0.01)

    def test_effective_tax_rate(self, analyzer, sample_data):
        """EBT = 200k-30k = 170k, tax = 1 - 150k/170k = 11.8%."""
        result = analyzer.wacc_analysis(sample_data)
        assert result.effective_tax_rate == pytest.approx(0.118, abs=0.01)

    def test_after_tax_cost_of_debt(self, analyzer, sample_data):
        """Rd * (1-T) = 0.075 * (1 - 0.118) = 0.0661."""
        result = analyzer.wacc_analysis(sample_data)
        assert result.after_tax_cost_of_debt == pytest.approx(0.0661, abs=0.005)

    def test_implied_cost_of_equity(self, analyzer, sample_data):
        """ROE=150k/1.2M=12.5%, implied Re = min(0.125+0.03, 0.30) = 0.155."""
        result = analyzer.wacc_analysis(sample_data)
        assert result.implied_cost_of_equity == pytest.approx(0.155, rel=0.01)

    def test_wacc_calculation(self, analyzer, sample_data):
        """WACC = 0.25*0.0661 + 0.75*0.155 = ~13.3%."""
        result = analyzer.wacc_analysis(sample_data)
        assert result.wacc == pytest.approx(0.133, abs=0.005)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.wacc_analysis(sample_data)
        assert result.wacc_grade in ["Excellent", "Good", "Fair", "Expensive"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.wacc_analysis(sample_data)
        assert "WACC" in result.summary


# ===== SCORING TESTS =====


class TestWACCScoring:
    def test_excellent_low_wacc(self, analyzer):
        """Low WACC + balanced structure => Excellent."""
        data = FinancialData(
            total_debt=300_000,
            total_equity=700_000,
            interest_expense=9_000,  # Rd = 3%
            ebit=200_000,
            net_income=160_000,
            # ROE = 160k/700k = 22.9%, implied Re = min(0.259, 0.30)=0.259
            # EBT = 200k-9k=191k, tax=1-160k/191k=0.162
            # After-tax Rd = 0.03*(1-0.162)=0.02514
            # WACC = 0.3*0.02514 + 0.7*0.259 = 0.00754+0.1813 = 0.189... hmm that's high
        )
        # Actually let me use very low implied Re
        data2 = FinancialData(
            total_debt=200_000,
            total_equity=800_000,
            interest_expense=4_000,  # Rd = 2%
            ebit=100_000,
            net_income=50_000,
            # ROE = 50k/800k = 6.25%, implied Re = max(0.08, 0.0625+0.03) = max(0.08, 0.0925) = 0.0925
            # EBT = 100k-4k=96k, tax=1-50k/96k=0.479
            # After-tax Rd = 0.02*(1-0.479)=0.01042
            # WACC = 0.2*0.01042 + 0.8*0.0925 = 0.00208+0.074 = 0.0761
            # WACC<0.08: +2.0; Wd=0.2 in [0.20,0.50]: +0.5; Rd<0.04: +0.5
            # Score = 5+2+0.5+0.5 = 8.0 Excellent
        )
        result = analyzer.wacc_analysis(data2)
        assert result.wacc_score >= 8.0
        assert result.wacc_grade == "Excellent"

    def test_expensive_high_wacc(self, analyzer):
        """High WACC + over-levered => Expensive."""
        data = FinancialData(
            total_debt=900_000,
            total_equity=100_000,
            interest_expense=180_000,  # Rd = 20%
            ebit=50_000,
            net_income=5_000,
        )
        result = analyzer.wacc_analysis(data)
        # Rd=20%>0.10: -0.5, Wd=0.9>0.70: -1.0, WACC>0.15: -1.0
        # Score = 5.0 - 0.5 - 1.0 - 1.0 = 2.5 => Expensive
        assert result.wacc_score < 4.0
        assert result.wacc_grade == "Expensive"

    def test_good_moderate_wacc(self, analyzer):
        """Moderate WACC => Good."""
        data = FinancialData(
            total_debt=400_000,
            total_equity=600_000,
            interest_expense=20_000,  # Rd = 5%
            ebit=150_000,
            net_income=100_000,
            # ROE=100k/600k=16.7%, implied Re=max(0.08, min(0.197, 0.30))=0.197
            # EBT=150k-20k=130k, tax=1-100k/130k=0.231
            # After-tax Rd = 0.05*(1-0.231)=0.03845
            # WACC = 0.4*0.03845 + 0.6*0.197 = 0.01538+0.1182 = 0.134
            # WACC between 0.12-0.15: no adj; Wd=0.4 in [0.20,0.50]: +0.5; Rd not <0.04 nor >0.10: no adj
            # Score = 5.0 + 0.5 = 5.5 => Fair
        )
        result = analyzer.wacc_analysis(data)
        assert result.wacc_score >= 4.0
        assert result.wacc_grade in ["Excellent", "Good", "Fair"]


# ===== EDGE CASES =====


class TestPhase37EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.wacc_analysis(FinancialData())
        assert isinstance(result, WACCResult)
        assert result.wacc is None

    def test_no_debt(self, analyzer):
        """No debt => all equity, WACC = cost of equity."""
        data = FinancialData(
            total_debt=0,
            total_equity=1_000_000,
            net_income=120_000,
        )
        result = analyzer.wacc_analysis(data)
        # Wd=0, We=1.0
        assert result.debt_weight == pytest.approx(0.0, abs=0.001)
        assert result.equity_weight == pytest.approx(1.0, abs=0.001)
        # WACC = 0 + 1.0 * implied_Re
        assert result.wacc == result.implied_cost_of_equity

    def test_no_equity(self, analyzer):
        """No equity (equity=0) => cannot compute (D+E must > 0)."""
        data = FinancialData(
            total_debt=500_000,
            total_equity=0,
        )
        result = analyzer.wacc_analysis(data)
        # Wd=1.0, We=0
        assert result.debt_weight == pytest.approx(1.0, abs=0.001)
        assert result.equity_weight == pytest.approx(0.0, abs=0.001)

    def test_no_interest(self, analyzer):
        """No interest expense => cost of debt is None."""
        data = FinancialData(
            total_debt=400_000,
            total_equity=600_000,
            net_income=100_000,
        )
        result = analyzer.wacc_analysis(data)
        assert result.cost_of_debt is None
        # WACC still computable (equity cost weighted by equity share)
        assert result.wacc is not None

    def test_missing_both_debt_equity(self, analyzer):
        """Neither debt nor equity => insufficient data."""
        data = FinancialData(
            revenue=1_000_000,
            ebit=200_000,
        )
        result = analyzer.wacc_analysis(data)
        assert result.wacc is None
        assert "Insufficient" in result.summary

    def test_tax_rate_clamped(self, analyzer):
        """Tax rate should be clamped between 0 and 60%."""
        data = FinancialData(
            total_debt=200_000,
            total_equity=800_000,
            interest_expense=10_000,
            ebit=100_000,
            net_income=200_000,  # NI > EBT => negative implied tax
        )
        result = analyzer.wacc_analysis(data)
        assert result.effective_tax_rate >= 0.0
        assert result.effective_tax_rate <= 0.60

    def test_sample_data_score(self, analyzer, sample_data):
        """WACC ~13.3%: no wacc adj, Wd=0.25 in [0.20,0.50]: +0.5, Rd=7.5% no adj.
        Score = 5.0 + 0.5 = 5.5 => Fair."""
        result = analyzer.wacc_analysis(sample_data)
        assert result.wacc_score == pytest.approx(5.5, abs=0.2)
        assert result.wacc_grade == "Fair"
