"""Edge-case tests for defensive code paths across modules.

Regression guards that verify safe handling of zeros, Nones, empty data,
negative values, and boundary conditions.
"""

import math

import pytest

from compliance_scorer import ComplianceScorer
from financial_analyzer import FinancialData, safe_divide
from portfolio_analyzer import PortfolioAnalyzer, _hhi
from startup_model import StartupAnalyzer
from underwriting import UnderwritingAnalyzer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def startup_analyzer():
    return StartupAnalyzer()


@pytest.fixture
def compliance_scorer():
    return ComplianceScorer()


@pytest.fixture
def portfolio_analyzer():
    return PortfolioAnalyzer()


@pytest.fixture
def underwriting_analyzer():
    return UnderwritingAnalyzer()


# ---------------------------------------------------------------------------
# 1. safe_divide edge cases
# ---------------------------------------------------------------------------


class TestSafeDivideEdgeCases:
    def test_zero_denominator_returns_default(self):
        assert safe_divide(100, 0) is None

    def test_none_denominator_returns_default(self):
        assert safe_divide(100, None) is None

    def test_none_numerator_returns_default(self):
        assert safe_divide(None, 100) is None

    def test_tiny_denominator_returns_default(self):
        assert safe_divide(100, 1e-15) is None

    def test_negative_values_work(self):
        assert safe_divide(-100, 50) == -2.0

    def test_custom_default(self):
        assert safe_divide(100, 0, default=0.0) == 0.0

    def test_both_none_returns_default(self):
        assert safe_divide(None, None) is None


# ---------------------------------------------------------------------------
# 2. Startup SaaS metrics -- zero/None customer edge cases
# ---------------------------------------------------------------------------


class TestStartupZeroCustomer:
    def test_arpu_zero_customers(self, startup_analyzer):
        data = FinancialData(
            monthly_recurring_revenue=10_000,
            customer_count=0,
        )
        result = startup_analyzer.saas_metrics(data)
        assert result.arpu is None

    def test_arpu_none_customers(self, startup_analyzer):
        data = FinancialData(
            monthly_recurring_revenue=10_000,
            customer_count=None,
        )
        result = startup_analyzer.saas_metrics(data)
        assert result.arpu is None

    def test_churn_zero_customers(self, startup_analyzer):
        data = FinancialData(
            monthly_recurring_revenue=10_000,
            churned_customers=5,
            customer_count=0,
        )
        result = startup_analyzer.saas_metrics(data)
        assert result.gross_churn_rate is None


# ---------------------------------------------------------------------------
# 3. Burn / runway edge cases
# ---------------------------------------------------------------------------


class TestBurnRunwayEdgeCases:
    def test_exact_breakeven_net_burn_zero(self, startup_analyzer):
        data = FinancialData(
            monthly_recurring_revenue=5_000,
            monthly_burn_rate=5_000,
            cash=100_000,
        )
        result = startup_analyzer.burn_runway(data)
        assert result.net_burn == 0
        assert result.is_cash_flow_positive is True
        assert result.runway_months is None

    def test_negative_net_burn_profitable(self, startup_analyzer):
        data = FinancialData(
            monthly_recurring_revenue=10_000,
            monthly_burn_rate=5_000,
            cash=100_000,
        )
        result = startup_analyzer.burn_runway(data)
        assert result.is_cash_flow_positive is True

    def test_zero_cash_positive_burn(self, startup_analyzer):
        data = FinancialData(
            monthly_recurring_revenue=1_000,
            monthly_burn_rate=5_000,
            cash=0,
        )
        result = startup_analyzer.burn_runway(data)
        # net_burn = 5000 - 1000 = 4000; runway = 0 / 4000 = 0
        assert result.runway_months is not None
        assert result.runway_months == pytest.approx(0.0, abs=0.01)

    def test_none_cash_burn(self, startup_analyzer):
        data = FinancialData(
            cash=None,
            monthly_burn_rate=None,
            monthly_recurring_revenue=None,
        )
        result = startup_analyzer.burn_runway(data)
        assert result.gross_burn is None
        assert result.net_burn is None


# ---------------------------------------------------------------------------
# 4. Compliance score boundaries
# ---------------------------------------------------------------------------


class TestComplianceScoreBoundaries:
    def test_perfect_compliance(self, compliance_scorer):
        data = FinancialData(
            total_assets=10_000_000,
            current_assets=4_000_000,
            cash=2_000_000,
            inventory=500_000,
            total_liabilities=3_000_000,
            current_liabilities=1_000_000,
            total_debt=1_000_000,
            total_equity=7_000_000,
            revenue=15_000_000,
            cogs=9_000_000,
            gross_profit=6_000_000,
            operating_income=3_000_000,
            net_income=2_500_000,
            ebit=3_000_000,
            ebitda=4_000_000,
            interest_expense=100_000,
            operating_cash_flow=4_000_000,
            capex=500_000,
            retained_earnings=5_000_000,
        )
        report = compliance_scorer.full_compliance_report(data)
        assert report.audit_risk is not None
        assert report.audit_risk.score >= 80

    def test_worst_compliance(self, compliance_scorer):
        data = FinancialData(
            total_assets=1_000_000,
            current_assets=100_000,
            cash=5_000,
            total_liabilities=2_000_000,
            current_liabilities=500_000,
            total_debt=1_500_000,
            total_equity=-1_000_000,
            revenue=0,
            operating_income=-500_000,
            net_income=-800_000,
            operating_cash_flow=-200_000,
        )
        report = compliance_scorer.full_compliance_report(data)
        assert report.audit_risk is not None
        # Worst-case company should score poorly (D or F grade territory)
        assert report.audit_risk.score <= 50
        assert report.audit_risk.going_concern_risk is True

    def test_empty_data_no_crash(self, compliance_scorer):
        data = FinancialData()
        report = compliance_scorer.full_compliance_report(data)
        assert report.summary != ""


# ---------------------------------------------------------------------------
# 5. Portfolio HHI edge cases
# ---------------------------------------------------------------------------


class TestPortfolioHHIEdgeCases:
    def test_hhi_all_zeros(self):
        assert _hhi([0, 0, 0]) == 1.0

    def test_hhi_single_value(self):
        assert _hhi([100]) == 1.0

    def test_hhi_equal_values(self):
        result = _hhi([100, 100, 100, 100])
        assert result == pytest.approx(0.25, abs=1e-9)

    def test_hhi_empty_list(self):
        assert _hhi([]) == 1.0

    def test_diversification_zero_revenue_companies(self, portfolio_analyzer):
        companies = {
            "A": FinancialData(revenue=0, total_assets=0),
            "B": FinancialData(revenue=0, total_assets=0),
            "C": FinancialData(revenue=0, total_assets=0),
        }
        result = portfolio_analyzer.diversification_score(companies)
        assert not math.isnan(result.overall_score)


# ---------------------------------------------------------------------------
# 6. Underwriting edge cases
# ---------------------------------------------------------------------------


class TestUnderwritingEdgeCases:
    def test_negative_equity_score_zero(self, underwriting_analyzer):
        data = FinancialData(
            total_debt=1_000_000,
            total_equity=-500_000,
            total_assets=500_000,
            revenue=1_000_000,
            net_income=50_000,
        )
        result = underwriting_analyzer.credit_scorecard(data)
        assert result.category_scores["leverage"] == 0

    def test_zero_revenue_company(self, underwriting_analyzer):
        data = FinancialData(
            total_assets=1_000_000,
            total_equity=500_000,
            revenue=0,
            net_income=0,
        )
        result = underwriting_analyzer.credit_scorecard(data)
        assert isinstance(result.total_score, int)

    def test_empty_financial_data(self, underwriting_analyzer):
        data = FinancialData()
        result = underwriting_analyzer.credit_scorecard(data)
        assert result.total_score >= 0
        assert result.grade in ("A", "B", "C", "D", "F")
