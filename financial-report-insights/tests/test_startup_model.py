"""Tests for startup financial modeling module."""

import pytest

from financial_analyzer import FinancialData
from startup_model import (
    BurnRunway,
    FundingScenario,
    SaaSMetrics,
    StartupAnalyzer,
    StartupReport,
    UnitEconomics,
)


# ---------------------------------------------------------------------------
# SaaS metrics
# ---------------------------------------------------------------------------


class TestSaaSMetrics:
    @pytest.fixture
    def analyzer(self):
        return StartupAnalyzer()

    @pytest.fixture
    def saas_company(self):
        return FinancialData(
            monthly_recurring_revenue=500_000,
            annual_recurring_revenue=6_000_000,
            customer_count=1000,
            churned_customers=30,
            revenue=6_000_000,
            gross_profit=4_200_000,
            cash=3_000_000,
        )

    def test_saas_metrics_basic(self, analyzer, saas_company):
        m = analyzer.saas_metrics(saas_company)
        assert m.mrr == 500_000
        assert m.arr == 6_000_000
        assert m.arpu is not None
        assert m.gross_churn_rate is not None
        assert m.customers == 1000

    def test_saas_metrics_arr_from_mrr(self, analyzer):
        data = FinancialData(monthly_recurring_revenue=100_000)
        m = analyzer.saas_metrics(data)
        assert m.arr == 1_200_000

    def test_saas_metrics_mrr_from_arr(self, analyzer):
        data = FinancialData(annual_recurring_revenue=1_200_000)
        m = analyzer.saas_metrics(data)
        assert m.mrr == 100_000

    def test_saas_metrics_empty_data(self, analyzer):
        m = analyzer.saas_metrics(FinancialData())
        assert m.mrr is None
        assert m.arr is None

    def test_churn_calculation(self, analyzer, saas_company):
        m = analyzer.saas_metrics(saas_company)
        assert m.gross_churn_rate == pytest.approx(0.03)  # 30/1000

    def test_arpu_calculation(self, analyzer, saas_company):
        m = analyzer.saas_metrics(saas_company)
        assert m.arpu == pytest.approx(500.0)  # 500k / 1000

    def test_grr_estimate(self, analyzer, saas_company):
        """P0-9: 1 - gross_churn is GRR, not NRR."""
        m = analyzer.saas_metrics(saas_company)
        # GRR = 1 - gross_churn = 1 - 0.03 = 0.97
        assert m.gross_revenue_retention == pytest.approx(0.97)

    def test_nrr_is_none_without_expansion_data(self, analyzer, saas_company):
        """P0-9: NRR requires expansion-revenue data; without it, must be None."""
        m = analyzer.saas_metrics(saas_company)
        assert m.net_revenue_retention is None

    def test_p0_9_grr_field_exists_on_dataclass(self):
        """P0-9: SaaSMetrics exposes gross_revenue_retention separately from NRR."""
        sm = SaaSMetrics()
        assert hasattr(sm, "gross_revenue_retention")
        assert sm.gross_revenue_retention is None
        assert sm.net_revenue_retention is None

    def test_mrr_growth_unavailable(self, analyzer, saas_company):
        m = analyzer.saas_metrics(saas_company)
        assert m.mrr_growth_rate is None

    def test_interpretation_nonempty(self, analyzer, saas_company):
        m = analyzer.saas_metrics(saas_company)
        assert len(m.interpretation) > 0
        assert "MRR" in m.interpretation


# ---------------------------------------------------------------------------
# Unit economics
# ---------------------------------------------------------------------------


class TestUnitEconomics:
    @pytest.fixture
    def analyzer(self):
        return StartupAnalyzer()

    def test_unit_economics_healthy(self, analyzer):
        data = FinancialData(
            monthly_recurring_revenue=500_000,
            customer_count=1000,
            churned_customers=20,
            customer_acquisition_cost=5000,
            lifetime_value=25000,
            gross_profit=350_000,
            revenue=500_000,
        )
        ue = analyzer.unit_economics(data)
        assert ue.ltv_to_cac == pytest.approx(5.0)
        assert ue.cac == 5000
        assert "Healthy" in ue.interpretation

    def test_unit_economics_needs_improvement(self, analyzer):
        data = FinancialData(
            customer_acquisition_cost=10000,
            lifetime_value=20000,
            gross_profit=6000,
            revenue=10000,
        )
        ue = analyzer.unit_economics(data)
        assert ue.ltv_to_cac == pytest.approx(2.0)
        assert "improvement" in ue.interpretation

    def test_unit_economics_unsustainable(self, analyzer):
        data = FinancialData(
            customer_acquisition_cost=10000,
            lifetime_value=5000,
            gross_profit=3000,
            revenue=10000,
        )
        ue = analyzer.unit_economics(data)
        assert ue.ltv_to_cac == pytest.approx(0.5)
        assert "Unsustainable" in ue.interpretation

    def test_unit_economics_empty(self, analyzer):
        ue = analyzer.unit_economics(FinancialData())
        assert ue.ltv_to_cac is None
        assert "Insufficient" in ue.interpretation

    def test_ltv_derived_from_arpu_churn(self, analyzer):
        data = FinancialData(
            monthly_recurring_revenue=100_000,
            customer_count=100,
            churned_customers=5,
            customer_acquisition_cost=2000,
        )
        ue = analyzer.unit_economics(data)
        # ARPU = 100k/100 = 1000, churn = 5/100 = 0.05, LTV = 1000/0.05 = 20000
        assert ue.ltv == pytest.approx(20_000)
        assert ue.ltv_to_cac == pytest.approx(10.0)

    def test_gross_margin_adjusted_ltv(self, analyzer):
        data = FinancialData(
            customer_acquisition_cost=1000,
            lifetime_value=10000,
            gross_profit=7000,
            revenue=10000,
        )
        ue = analyzer.unit_economics(data)
        # GM = 0.7, GM-adjusted LTV = 10000 * 0.7 = 7000
        assert ue.gross_margin_adjusted_ltv == pytest.approx(7000)

    def test_payback_months(self, analyzer):
        data = FinancialData(
            monthly_recurring_revenue=100_000,
            customer_count=100,
            customer_acquisition_cost=5000,
            lifetime_value=50000,
        )
        ue = analyzer.unit_economics(data)
        # Monthly ARPU = 100k/100 = 1000, payback = 5000/1000 = 5
        assert ue.payback_months == pytest.approx(5.0)

    def test_magic_number_unavailable(self, analyzer):
        data = FinancialData(lifetime_value=10000, customer_acquisition_cost=2000)
        ue = analyzer.unit_economics(data)
        assert ue.magic_number is None


# ---------------------------------------------------------------------------
# Burn / Runway
# ---------------------------------------------------------------------------


class TestBurnRunway:
    @pytest.fixture
    def analyzer(self):
        return StartupAnalyzer()

    def test_burn_runway_critical(self, analyzer):
        data = FinancialData(
            cash=500_000,
            monthly_burn_rate=150_000,
            monthly_recurring_revenue=50_000,
        )
        br = analyzer.burn_runway(data)
        assert br.net_burn == pytest.approx(100_000)
        assert br.runway_months == pytest.approx(5.0)
        assert br.category == "critical"

    def test_burn_runway_caution(self, analyzer):
        data = FinancialData(
            cash=1_000_000,
            monthly_burn_rate=150_000,
            monthly_recurring_revenue=50_000,
        )
        br = analyzer.burn_runway(data)
        assert br.net_burn == pytest.approx(100_000)
        assert br.runway_months == pytest.approx(10.0)
        assert br.category == "caution"

    def test_burn_runway_comfortable(self, analyzer):
        data = FinancialData(
            cash=2_000_000,
            monthly_burn_rate=200_000,
            monthly_recurring_revenue=50_000,
        )
        br = analyzer.burn_runway(data)
        # net_burn = 150k, runway = 2M/150k = 13.3
        assert br.category == "comfortable"

    def test_burn_runway_strong(self, analyzer):
        data = FinancialData(
            cash=3_000_000,
            monthly_burn_rate=150_000,
            monthly_recurring_revenue=50_000,
        )
        br = analyzer.burn_runway(data)
        # net_burn = 100k, runway = 3M/100k = 30
        assert br.category == "strong"

    def test_burn_runway_profitable(self, analyzer):
        data = FinancialData(
            cash=1_000_000,
            monthly_burn_rate=100_000,
            monthly_recurring_revenue=150_000,
        )
        br = analyzer.burn_runway(data)
        assert br.net_burn is not None
        assert br.net_burn <= 0
        assert br.runway_months is None
        assert br.is_cash_flow_positive is True
        assert br.category == "strong"

    def test_burn_runway_no_data(self, analyzer):
        br = analyzer.burn_runway(FinancialData())
        assert br.runway_months is None
        assert br.category == ""

    def test_burn_from_opex(self, analyzer):
        data = FinancialData(
            cash=600_000,
            operating_expenses=1_200_000,  # annual -> 100k/month
            revenue=600_000,  # annual -> 50k/month
        )
        br = analyzer.burn_runway(data)
        assert br.gross_burn == pytest.approx(100_000)
        assert br.monthly_revenue == pytest.approx(50_000)
        assert br.net_burn == pytest.approx(50_000)
        assert br.runway_months == pytest.approx(12.0)

    def test_breakeven_revenue_needed(self, analyzer):
        data = FinancialData(
            cash=1_000_000,
            monthly_burn_rate=200_000,
            monthly_recurring_revenue=50_000,
        )
        br = analyzer.burn_runway(data)
        assert br.breakeven_revenue_needed == pytest.approx(200_000)

    def test_cash_on_hand(self, analyzer):
        data = FinancialData(cash=5_000_000, monthly_burn_rate=100_000)
        br = analyzer.burn_runway(data)
        assert br.cash_on_hand == 5_000_000


# ---------------------------------------------------------------------------
# Funding scenarios
# ---------------------------------------------------------------------------


class TestFundingScenarios:
    @pytest.fixture
    def analyzer(self):
        return StartupAnalyzer()

    def test_funding_scenarios(self, analyzer):
        data = FinancialData(
            cash=1_000_000,
            monthly_burn_rate=200_000,
            monthly_recurring_revenue=100_000,
            annual_recurring_revenue=1_200_000,
        )
        scenarios = [
            {"raise_amount": 5_000_000, "pre_money_valuation": 20_000_000},
            {"raise_amount": 10_000_000, "pre_money_valuation": 40_000_000},
        ]
        results = analyzer.funding_scenarios(data, scenarios)
        assert len(results) == 2
        assert results[0].dilution_pct == pytest.approx(0.20)  # 5M / 25M
        assert results[1].dilution_pct == pytest.approx(0.20)  # 10M / 50M

    def test_funding_post_money(self, analyzer):
        data = FinancialData(cash=500_000, monthly_burn_rate=100_000)
        scenarios = [{"raise_amount": 3_000_000, "pre_money_valuation": 12_000_000}]
        results = analyzer.funding_scenarios(data, scenarios)
        assert results[0].post_money_valuation == pytest.approx(15_000_000)

    def test_funding_extended_runway(self, analyzer):
        data = FinancialData(
            cash=1_000_000,
            monthly_burn_rate=200_000,
            monthly_recurring_revenue=100_000,
        )
        # net_burn = 100k, current runway = 1M/100k = 10
        scenarios = [{"raise_amount": 5_000_000, "pre_money_valuation": 20_000_000}]
        results = analyzer.funding_scenarios(data, scenarios)
        # new cash = 6M, new_runway = 6M/100k = 60
        assert results[0].new_runway_months == pytest.approx(60.0)
        assert results[0].extended_months == pytest.approx(50.0)

    def test_funding_implied_arr_multiple(self, analyzer):
        data = FinancialData(
            cash=1_000_000,
            monthly_burn_rate=100_000,
            annual_recurring_revenue=2_000_000,
        )
        scenarios = [{"raise_amount": 5_000_000, "pre_money_valuation": 20_000_000}]
        results = analyzer.funding_scenarios(data, scenarios)
        assert results[0].implied_arr_multiple == pytest.approx(10.0)  # 20M / 2M

    def test_funding_scenario_empty(self, analyzer):
        results = analyzer.funding_scenarios(FinancialData(), [])
        assert results == []

    def test_funding_scenario_name_default(self, analyzer):
        data = FinancialData(cash=1_000_000, monthly_burn_rate=100_000)
        scenarios = [{"raise_amount": 1_000_000, "pre_money_valuation": 5_000_000}]
        results = analyzer.funding_scenarios(data, scenarios)
        assert results[0].scenario_name == "Scenario 1"


# ---------------------------------------------------------------------------
# Full startup analysis
# ---------------------------------------------------------------------------


class TestFullStartupAnalysis:
    @pytest.fixture
    def analyzer(self):
        return StartupAnalyzer()

    def test_full_analysis(self, analyzer):
        data = FinancialData(
            monthly_recurring_revenue=500_000,
            annual_recurring_revenue=6_000_000,
            customer_count=1000,
            churned_customers=30,
            customer_acquisition_cost=5000,
            lifetime_value=25000,
            cash=5_000_000,
            monthly_burn_rate=600_000,
            revenue=6_000_000,
            gross_profit=4_200_000,
            total_funding_raised=10_000_000,
        )
        report = analyzer.full_startup_analysis(data)
        assert isinstance(report, StartupReport)
        # Verify sub-reports contain real computed values, not just not-None
        assert report.saas_metrics.mrr > 0
        assert report.unit_economics.ltv_to_cac is not None
        assert report.burn_runway.runway_months > 0
        assert report.stage in ("pre-seed", "seed", "series-a", "series-b", "growth", "profitable")

    def test_full_analysis_with_funding(self, analyzer):
        data = FinancialData(
            cash=2_000_000,
            monthly_burn_rate=200_000,
            monthly_recurring_revenue=100_000,
            annual_recurring_revenue=1_200_000,
        )
        scenarios = [{"raise_amount": 5_000_000, "pre_money_valuation": 20_000_000}]
        report = analyzer.full_startup_analysis(data, scenarios)
        assert len(report.funding_scenarios) == 1

    def test_stage_detection_profitable(self, analyzer):
        data = FinancialData(
            monthly_recurring_revenue=500_000,
            monthly_burn_rate=300_000,
            cash=1_000_000,
        )
        report = analyzer.full_startup_analysis(data)
        assert report.stage == "profitable"

    def test_stage_detection_growth(self, analyzer):
        data = FinancialData(
            annual_recurring_revenue=15_000_000,
            monthly_burn_rate=1_000_000,
            cash=10_000_000,
        )
        report = analyzer.full_startup_analysis(data)
        assert report.stage == "growth"

    def test_stage_detection_series_b(self, analyzer):
        data = FinancialData(
            annual_recurring_revenue=7_000_000,
            monthly_burn_rate=500_000,
            cash=5_000_000,
        )
        report = analyzer.full_startup_analysis(data)
        assert report.stage == "series-b"

    def test_stage_detection_series_a(self, analyzer):
        data = FinancialData(
            annual_recurring_revenue=2_000_000,
            monthly_burn_rate=200_000,
            cash=2_000_000,
        )
        report = analyzer.full_startup_analysis(data)
        assert report.stage == "series-a"

    def test_stage_detection_seed(self, analyzer):
        data = FinancialData(
            annual_recurring_revenue=500_000,
            monthly_burn_rate=100_000,
            cash=1_000_000,
        )
        report = analyzer.full_startup_analysis(data)
        assert report.stage == "seed"

    def test_stage_detection_pre_seed(self, analyzer):
        data = FinancialData()
        report = analyzer.full_startup_analysis(data)
        assert report.stage == "pre-seed"

    def test_summary_nonempty(self, analyzer):
        data = FinancialData(
            monthly_recurring_revenue=100_000,
            cash=1_000_000,
            monthly_burn_rate=150_000,
        )
        report = analyzer.full_startup_analysis(data)
        assert "Stage:" in report.summary

    def test_empty_data(self, analyzer):
        report = analyzer.full_startup_analysis(FinancialData())
        assert isinstance(report, StartupReport)
        assert report.stage == "pre-seed"


# ---------------------------------------------------------------------------
# Zero-customer and breakeven edge cases
# ---------------------------------------------------------------------------


class TestStartupEdgeCases:
    @pytest.fixture()
    def analyzer(self):
        return StartupAnalyzer()

    def test_saas_metrics_zero_customers_no_crash(self, analyzer):
        """ARPU with zero customers should return None, not ZeroDivisionError."""
        data = FinancialData(monthly_recurring_revenue=50_000, customer_count=0)
        metrics = analyzer.saas_metrics(data)
        assert metrics.arpu is None
        assert metrics.mrr == 50_000

    def test_saas_metrics_none_customers_no_crash(self, analyzer):
        """ARPU with None customers should return None gracefully."""
        data = FinancialData(monthly_recurring_revenue=50_000, customer_count=None)
        metrics = analyzer.saas_metrics(data)
        assert metrics.arpu is None

    def test_burn_runway_exact_breakeven(self, analyzer):
        """When net_burn=0 (exact breakeven), runway should be None not infinity."""
        data = FinancialData(
            cash=1_000_000,
            monthly_burn_rate=150_000,
            monthly_recurring_revenue=150_000,
        )
        br = analyzer.burn_runway(data)
        assert br.is_cash_flow_positive is True
        assert br.runway_months is None  # Not float('inf')

    def test_unit_economics_zero_churn(self, analyzer):
        """Zero churn should not cause division errors in LTV calculation."""
        data = FinancialData(
            monthly_recurring_revenue=100_000,
            customer_count=100,
            churned_customers=0,
        )
        ue = analyzer.unit_economics(data)
        # Zero churn -> infinite LTV or capped value, not crash
        assert ue.ltv is None or ue.ltv > 0
