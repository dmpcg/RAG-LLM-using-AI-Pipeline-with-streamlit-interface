"""Phase 26 Tests: Concentration & Structural Risk Analysis.

Tests for concentration_risk_analysis() and ConcentrationRiskResult dataclass.
"""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    ConcentrationRiskResult,
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


class TestConcentrationRiskDataclass:
    def test_defaults(self):
        r = ConcentrationRiskResult()
        assert r.revenue_asset_intensity is None
        assert r.operating_dependency is None
        assert r.concentration_score == 0.0
        assert r.concentration_grade == ""
        assert r.summary == ""

    def test_fields(self):
        r = ConcentrationRiskResult(
            revenue_asset_intensity=0.50,
            operating_dependency=0.20,
            concentration_grade="Balanced",
        )
        assert r.revenue_asset_intensity == 0.50
        assert r.operating_dependency == 0.20
        assert r.concentration_grade == "Balanced"


# ===== CORE COMPUTATION TESTS =====


class TestConcentrationRisk:
    def test_returns_result(self, analyzer, sample_data):
        result = analyzer.concentration_risk_analysis(sample_data)
        assert isinstance(result, ConcentrationRiskResult)

    def test_revenue_asset_intensity(self, analyzer, sample_data):
        """Revenue/TA = 1M/2M = 0.50."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.revenue_asset_intensity == pytest.approx(0.50, rel=0.01)

    def test_operating_dependency(self, analyzer, sample_data):
        """OI/Revenue = 200k/1M = 20%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.operating_dependency == pytest.approx(0.20, rel=0.01)

    def test_asset_composition_current(self, analyzer, sample_data):
        """CA/TA = 500k/2M = 25%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.asset_composition_current == pytest.approx(0.25, rel=0.01)

    def test_asset_composition_fixed(self, analyzer, sample_data):
        """Fixed = 1 - 0.25 = 75%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.asset_composition_fixed == pytest.approx(0.75, rel=0.01)

    def test_liability_structure_current(self, analyzer, sample_data):
        """CL/TL = 200k/800k = 25%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.liability_structure_current == pytest.approx(0.25, rel=0.01)

    def test_earnings_retention_ratio(self, analyzer, sample_data):
        """NI/EBITDA = 150k/250k = 60%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.earnings_retention_ratio == pytest.approx(0.60, rel=0.01)

    def test_working_capital_intensity(self, analyzer, sample_data):
        """NWC/Revenue = 300k/1M = 30%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.working_capital_intensity == pytest.approx(0.30, rel=0.01)

    def test_capex_intensity(self, analyzer, sample_data):
        """Capex/Revenue = 80k/1M = 8%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.capex_intensity == pytest.approx(0.08, rel=0.01)

    def test_interest_burden(self, analyzer, sample_data):
        """Interest/EBIT = 30k/200k = 15%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.interest_burden == pytest.approx(0.15, rel=0.01)

    def test_cash_conversion_efficiency(self, analyzer, sample_data):
        """OCF/NI = 220k/150k = 1.467."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.cash_conversion_efficiency == pytest.approx(220_000 / 150_000, rel=0.01)

    def test_fixed_asset_ratio(self, analyzer, sample_data):
        """(TA-CA)/Equity = 1.5M/1.2M = 1.25."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.fixed_asset_ratio == pytest.approx(1.25, rel=0.01)

    def test_debt_concentration(self, analyzer, sample_data):
        """Debt/TL = 400k/800k = 50%."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.debt_concentration == pytest.approx(0.50, rel=0.01)

    def test_grade_assigned(self, analyzer, sample_data):
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.concentration_grade in ["Well Diversified", "Balanced", "Concentrated", "Highly Concentrated"]

    def test_summary_present(self, analyzer, sample_data):
        result = analyzer.concentration_risk_analysis(sample_data)
        assert "Concentration Risk" in result.summary


# ===== SCORING TESTS =====


class TestConcentrationScoring:
    def test_well_diversified(self, analyzer):
        """Balanced assets, healthy margins, strong cash conversion."""
        data = FinancialData(
            revenue=5_000_000,
            operating_income=750_000,
            ebit=750_000,
            ebitda=1_000_000,
            net_income=500_000,
            total_assets=10_000_000,
            total_liabilities=4_000_000,
            total_equity=6_000_000,
            current_assets=4_000_000,
            current_liabilities=1_500_000,
            total_debt=2_000_000,
            interest_expense=100_000,
            operating_cash_flow=700_000,
            capex=300_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        # asset_current=0.40: in [0.30,0.60] -> +1.0 -> 6.0
        # op_dep=0.15: in [0.10,0.30] -> +1.0 -> 7.0
        # cash_conv=700k/500k=1.4: >=1.2 -> +1.5 -> 8.5
        # int_burden=100k/750k=0.133: <=0.15 -> +0.5 -> 9.0
        # wc_intensity=2.5M/5M=0.50: not in [0.05,0.25] -> +0.0 -> 9.0
        assert result.concentration_grade == "Well Diversified"
        assert result.concentration_score >= 8.0

    def test_highly_concentrated(self, analyzer):
        """Operating loss, poor cash conversion, heavy interest."""
        data = FinancialData(
            revenue=500_000,
            operating_income=-100_000,
            ebit=-100_000,
            ebitda=50_000,
            net_income=-200_000,
            total_assets=5_000_000,
            total_liabilities=4_000_000,
            total_equity=1_000_000,
            current_assets=200_000,
            current_liabilities=1_500_000,
            total_debt=3_000_000,
            interest_expense=400_000,
            operating_cash_flow=-100_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        # asset_current=200k/5M=0.04: <0.10 -> -1.0 -> 4.0
        # op_dep=-0.20: <0 -> -1.5 -> 2.5
        # cash_conv: NI<0 -> None -> +0
        # int_burden: EBIT<0 -> None -> +0
        # wc_intensity=(200k-1.5M)/500k<0 -> -0.5 -> 2.0
        assert result.concentration_grade in ["Highly Concentrated", "Concentrated"]

    def test_score_capped_at_10(self, analyzer):
        data = FinancialData(
            revenue=10_000_000,
            operating_income=2_000_000,
            ebit=2_000_000,
            ebitda=3_000_000,
            net_income=1_500_000,
            total_assets=20_000_000,
            total_liabilities=8_000_000,
            total_equity=12_000_000,
            current_assets=8_000_000,
            current_liabilities=3_000_000,
            total_debt=4_000_000,
            interest_expense=200_000,
            operating_cash_flow=2_000_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        assert result.concentration_score <= 10.0

    def test_score_floored_at_0(self, analyzer):
        data = FinancialData(
            revenue=100_000,
            operating_income=-500_000,
            ebit=-500_000,
            net_income=-600_000,
            total_assets=5_000_000,
            total_liabilities=4_500_000,
            total_equity=500_000,
            current_assets=100_000,
            current_liabilities=2_000_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        assert result.concentration_score >= 0.0


# ===== EDGE CASES =====


class TestPhase26EdgeCases:
    def test_empty_data(self, analyzer):
        result = analyzer.concentration_risk_analysis(FinancialData())
        assert isinstance(result, ConcentrationRiskResult)
        assert result.revenue_asset_intensity is None
        assert result.operating_dependency is None

    def test_zero_revenue(self, analyzer):
        data = FinancialData(
            revenue=0,
            total_assets=1_000_000,
            current_assets=400_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        # safe_divide(0, 1M) = 0.0 (zero revenue = zero turnover, valid result)
        assert result.revenue_asset_intensity == pytest.approx(0.0)
        assert result.operating_dependency is None
        assert result.working_capital_intensity is None

    def test_zero_total_assets(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            total_assets=0,
        )
        result = analyzer.concentration_risk_analysis(data)
        assert result.revenue_asset_intensity is None
        assert result.asset_composition_current is None

    def test_no_liabilities(self, analyzer):
        data = FinancialData(
            revenue=1_000_000,
            total_assets=2_000_000,
            total_equity=2_000_000,
            total_liabilities=0,
        )
        result = analyzer.concentration_risk_analysis(data)
        assert result.liability_structure_current is None
        assert result.debt_concentration is None

    def test_negative_ni(self, analyzer):
        """Negative NI: no cash conversion ratio."""
        data = FinancialData(
            revenue=1_000_000,
            operating_income=50_000,
            ebit=50_000,
            net_income=-50_000,
            total_assets=2_000_000,
            total_equity=1_000_000,
            operating_cash_flow=100_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        assert result.cash_conversion_efficiency is None

    def test_no_capex(self, analyzer):
        """Zero capex => capex_intensity is None."""
        data = FinancialData(
            revenue=1_000_000,
            total_assets=2_000_000,
            current_assets=500_000,
            current_liabilities=200_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        assert result.capex_intensity is None

    def test_negative_working_capital(self, analyzer):
        """CL > CA => negative NWC."""
        data = FinancialData(
            revenue=1_000_000,
            total_assets=2_000_000,
            current_assets=200_000,
            current_liabilities=500_000,
        )
        result = analyzer.concentration_risk_analysis(data)
        assert result.working_capital_intensity == pytest.approx(-0.30, rel=0.01)

    def test_sample_data_grade(self, analyzer, sample_data):
        """Sample: asset_current=0.25(+0), op_dep=0.20(+1)->6, cash_conv=1.47(+1.5)->7.5, int=0.15(+0.5)->8.0."""
        result = analyzer.concentration_risk_analysis(sample_data)
        assert result.concentration_score == pytest.approx(8.0, abs=0.1)
        assert result.concentration_grade == "Well Diversified"
