"""
Startup financial modeling module.
SaaS metrics, unit economics, burn/runway analysis, and funding scenarios.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from financial_analyzer import FinancialData, safe_divide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SaaSMetrics:
    """Core SaaS business metrics."""

    mrr: Optional[float] = None  # Monthly Recurring Revenue
    arr: Optional[float] = None  # Annual Recurring Revenue
    mrr_growth_rate: Optional[float] = None  # MoM growth
    # P0-9: Real NRR requires multi-period expansion data (upgrades + downgrades).
    # Without it, set to None to avoid mislabeling 1-gross_churn as NRR.
    net_revenue_retention: Optional[float] = None  # NRR % (true NRR with expansion data)
    # GRR = 1 - gross_churn -- the upper bound on NRR when no expansion data is available.
    gross_revenue_retention: Optional[float] = None  # GRR %
    gross_churn_rate: Optional[float] = None  # Customer churn %
    revenue_churn_rate: Optional[float] = None  # Revenue churn %
    arpu: Optional[float] = None  # Average Revenue Per User
    customers: Optional[int] = None
    interpretation: str = ""


@dataclass
class UnitEconomics:
    """Unit economics analysis."""

    cac: Optional[float] = None  # Customer Acquisition Cost
    ltv: Optional[float] = None  # Lifetime Value
    ltv_to_cac: Optional[float] = None  # LTV/CAC ratio
    payback_months: Optional[float] = None  # CAC payback period
    magic_number: Optional[float] = None  # SaaS Magic Number
    gross_margin_adjusted_ltv: Optional[float] = None
    interpretation: str = ""


@dataclass
class BurnRunway:
    """Cash burn and runway analysis."""

    gross_burn: Optional[float] = None  # Total monthly spend
    net_burn: Optional[float] = None  # Net monthly cash decrease
    runway_months: Optional[float] = None
    is_cash_flow_positive: bool = False
    category: str = ""  # critical (<6mo), caution (6-12), comfortable (12-18), strong (18+)
    cash_on_hand: Optional[float] = None
    monthly_revenue: Optional[float] = None
    breakeven_revenue_needed: Optional[float] = None
    interpretation: str = ""


@dataclass
class FundingScenario:
    """A single funding scenario with dilution analysis."""

    scenario_name: str = ""
    raise_amount: float = 0.0
    pre_money_valuation: float = 0.0
    post_money_valuation: float = 0.0
    dilution_pct: float = 0.0
    new_runway_months: Optional[float] = None
    extended_months: Optional[float] = None  # Additional months gained
    implied_arr_multiple: Optional[float] = None


@dataclass
class StartupReport:
    """Complete startup analysis report."""

    saas_metrics: Optional[SaaSMetrics] = None
    unit_economics: Optional[UnitEconomics] = None
    burn_runway: Optional[BurnRunway] = None
    funding_scenarios: List[FundingScenario] = field(default_factory=list)
    stage: str = ""  # pre-seed, seed, series-a, series-b, growth, profitable
    summary: str = ""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class StartupAnalyzer:
    """Startup-specific financial analysis for SaaS and venture-backed companies."""

    def __init__(self) -> None:
        pass

    # -- SaaS metrics -------------------------------------------------------

    def saas_metrics(self, data: FinancialData) -> SaaSMetrics:
        """Compute core SaaS metrics from *data*."""

        mrr = data.monthly_recurring_revenue
        if mrr is None:
            mrr = safe_divide(data.annual_recurring_revenue, 12)

        arr = data.annual_recurring_revenue
        if arr is None and mrr is not None:
            arr = mrr * 12

        arpu: Optional[float] = None
        if mrr is not None and data.customer_count is not None and data.customer_count > 0:
            arpu = safe_divide(mrr, data.customer_count)

        gross_churn: Optional[float] = None
        if data.churned_customers is not None and data.customer_count is not None and data.customer_count > 0:
            gross_churn = safe_divide(data.churned_customers, data.customer_count)

        # P0-9: 1 - gross_churn is GROSS revenue retention (GRR), not NRR.
        # True NRR requires multi-period expansion (upgrade) data we do not have here,
        # so net_revenue_retention is left None until that data is supplied.
        grr: Optional[float] = None
        if gross_churn is not None:
            grr = 1.0 - gross_churn
        nrr: Optional[float] = None  # Requires expansion data; intentionally None.

        # MRR growth rate requires multi-period data
        mrr_growth_rate: Optional[float] = None

        # Build interpretation
        parts: List[str] = []
        if mrr is not None:
            parts.append(f"MRR ${mrr:,.0f}")
        if arr is not None:
            parts.append(f"ARR ${arr:,.0f}")
        if gross_churn is not None:
            pct = gross_churn * 100
            health = "healthy" if pct < 5 else ("moderate" if pct < 10 else "high")
            parts.append(f"Gross churn {pct:.1f}% ({health})")
        if mrr_growth_rate is None:
            parts.append("MRR growth rate unavailable (requires multi-period data)")
        interpretation = ". ".join(parts) + "." if parts else "No SaaS metrics available."

        return SaaSMetrics(
            mrr=mrr,
            arr=arr,
            mrr_growth_rate=mrr_growth_rate,
            net_revenue_retention=nrr,
            gross_revenue_retention=grr,
            gross_churn_rate=gross_churn,
            revenue_churn_rate=None,  # requires revenue-level churn data
            arpu=arpu,
            customers=data.customer_count,
            interpretation=interpretation,
        )

    # -- Unit economics -----------------------------------------------------

    def unit_economics(self, data: FinancialData) -> UnitEconomics:
        """Compute unit economics from *data*."""

        cac = data.customer_acquisition_cost

        # Derive MRR once: prefer explicit MRR > ARR/12 > revenue/12 (last resort)
        mrr = data.monthly_recurring_revenue
        if mrr is None:
            mrr = safe_divide(data.annual_recurring_revenue, 12)
        mrr_from_revenue = mrr is None  # track if we fell back to revenue/12
        if mrr is None:
            mrr = safe_divide(data.revenue, 12)

        # Monthly ARPU (used for both LTV and payback)
        arpu = safe_divide(mrr, data.customer_count) if mrr and data.customer_count else None

        # LTV: use explicit value, or derive from ARPU / churn
        ltv = data.lifetime_value
        if ltv is None:
            gross_churn = (
                safe_divide(data.churned_customers, data.customer_count)
                if data.churned_customers is not None and data.customer_count
                else None
            )
            if arpu is not None and gross_churn is not None and gross_churn > 0:
                ltv = safe_divide(arpu, gross_churn)

        # Gross margin
        gross_margin = safe_divide(data.gross_profit, data.revenue)

        # Gross-margin-adjusted LTV
        gm_ltv: Optional[float] = None
        if ltv is not None and gross_margin is not None:
            gm_ltv = ltv * gross_margin

        # LTV / CAC ratio
        ltv_to_cac = safe_divide(ltv, cac)

        # Payback months = CAC / monthly ARPU (uses same MRR/ARPU as LTV)
        payback: Optional[float] = None
        if cac is not None:
            payback = safe_divide(cac, arpu)

        # Magic number requires prior-period data
        magic_number: Optional[float] = None

        # Interpretation
        if ltv_to_cac is not None:
            if ltv_to_cac >= 3:
                interp = f"Healthy unit economics (LTV/CAC = {ltv_to_cac:.1f}x, target >3x)."
            elif ltv_to_cac >= 1:
                interp = f"Unit economics need improvement (LTV/CAC = {ltv_to_cac:.1f}x, target >3x)."
            else:
                interp = f"Unsustainable unit economics (LTV/CAC = {ltv_to_cac:.1f}x, below 1x)."
        else:
            interp = "Insufficient data to assess unit economics."

        if payback is not None:
            interp += f" CAC payback: {payback:.0f} months."
        if magic_number is None:
            interp += " Magic number unavailable (requires prior-period S&M spend)."

        return UnitEconomics(
            cac=cac,
            ltv=ltv,
            ltv_to_cac=ltv_to_cac,
            payback_months=payback,
            magic_number=magic_number,
            gross_margin_adjusted_ltv=gm_ltv,
            interpretation=interp,
        )

    # -- Burn / runway ------------------------------------------------------

    def burn_runway(self, data: FinancialData) -> BurnRunway:
        """Compute cash burn rate and runway from *data*."""

        monthly_revenue = data.monthly_recurring_revenue or safe_divide(data.revenue, 12)
        gross_burn = data.monthly_burn_rate or safe_divide(data.operating_expenses, 12)

        net_burn: Optional[float] = None
        if gross_burn is not None:
            net_burn = gross_burn - (monthly_revenue or 0)

        # Determine runway
        runway: Optional[float] = None
        category = ""
        is_cfp = False

        if net_burn is not None and net_burn <= 0:
            # Cash-flow positive — runway is effectively infinite
            is_cfp = True
            runway = None
            category = "strong"
        elif net_burn is not None and net_burn > 0:
            runway = data.cash_runway_months or safe_divide(data.cash, net_burn)

        if runway is not None and category == "":
            if runway < 6:
                category = "critical"
            elif runway < 12:
                category = "caution"
            elif runway < 18:
                category = "comfortable"
            else:
                category = "strong"

        breakeven_needed = gross_burn  # revenue must match burn

        # Interpretation
        if is_cfp:
            interp = "Company is cash-flow positive. No burn concerns."
        elif category == "critical":
            interp = f"Critical: ~{runway:.0f} months of runway. Immediate fundraising or cost reduction needed."
        elif category == "caution":
            interp = f"Caution: ~{runway:.0f} months of runway. Begin fundraising planning."
        elif category == "comfortable":
            interp = f"Comfortable: ~{runway:.0f} months of runway."
        elif category == "strong":
            interp = f"Strong: ~{runway:.0f} months of runway."
        else:
            interp = "Insufficient data to assess burn and runway."

        return BurnRunway(
            gross_burn=gross_burn,
            net_burn=net_burn,
            runway_months=runway,
            is_cash_flow_positive=is_cfp,
            category=category,
            cash_on_hand=data.cash,
            monthly_revenue=monthly_revenue,
            breakeven_revenue_needed=breakeven_needed,
            interpretation=interp,
        )

    # -- Funding scenarios --------------------------------------------------

    def funding_scenarios(
        self,
        data: FinancialData,
        scenarios: List[Dict[str, float]],
    ) -> List[FundingScenario]:
        """Evaluate a list of funding scenarios against *data*."""

        if not scenarios:
            return []

        # Pre-compute current burn info for runway extension calc
        burn = self.burn_runway(data)
        current_runway = burn.runway_months
        net_burn = burn.net_burn

        results: List[FundingScenario] = []
        for i, s in enumerate(scenarios, 1):
            raise_amount = s.get("raise_amount", 0)
            pre_money = s.get("pre_money_valuation", 0)
            post_money = pre_money + raise_amount
            dilution = safe_divide(raise_amount, post_money) if post_money > 0 else 0.0

            # New runway after injection
            new_runway: Optional[float] = None
            extended: Optional[float] = None
            if net_burn is not None and net_burn > 0:
                new_cash = (data.cash or 0) + raise_amount
                new_runway = safe_divide(new_cash, net_burn)
                if new_runway is not None and current_runway is not None:
                    extended = new_runway - current_runway

            # Implied ARR multiple
            arr = data.annual_recurring_revenue
            if arr is None and data.monthly_recurring_revenue is not None:
                arr = data.monthly_recurring_revenue * 12
            implied_arr_mult = safe_divide(pre_money, arr) if arr and arr > 0 else None

            results.append(
                FundingScenario(
                    scenario_name=s.get("name", f"Scenario {i}"),
                    raise_amount=raise_amount,
                    pre_money_valuation=pre_money,
                    post_money_valuation=post_money,
                    dilution_pct=dilution or 0.0,
                    new_runway_months=new_runway,
                    extended_months=extended,
                    implied_arr_multiple=implied_arr_mult,
                )
            )

        return results

    # -- Full analysis report -----------------------------------------------

    def full_startup_analysis(
        self,
        data: FinancialData,
        funding_scenarios_input: Optional[List[Dict[str, float]]] = None,
    ) -> StartupReport:
        """Run a complete startup analysis and return a *StartupReport*."""

        saas = self.saas_metrics(data)
        unit_econ = self.unit_economics(data)
        burn = self.burn_runway(data)
        funding = self.funding_scenarios(data, funding_scenarios_input) if funding_scenarios_input else []

        # Stage detection
        arr = saas.arr
        net_burn = burn.net_burn

        if net_burn is not None and net_burn <= 0:
            stage = "profitable"
        elif arr is not None and arr > 10_000_000:
            stage = "growth"
        elif arr is not None and arr > 5_000_000:
            stage = "series-b"
        elif arr is not None and arr > 1_000_000:
            stage = "series-a"
        elif arr is not None and arr > 0:
            stage = "seed"
        else:
            stage = "pre-seed"

        # Summary
        summary_parts: List[str] = [f"Stage: {stage}."]
        if saas.mrr is not None:
            summary_parts.append(f"MRR: ${saas.mrr:,.0f}.")
        if unit_econ.ltv_to_cac is not None:
            summary_parts.append(f"LTV/CAC: {unit_econ.ltv_to_cac:.1f}x.")
        if burn.is_cash_flow_positive:
            summary_parts.append("Cash-flow positive.")
        elif burn.runway_months is not None:
            summary_parts.append(f"Runway: {burn.runway_months:.0f} months ({burn.category}).")
        if funding:
            summary_parts.append(f"{len(funding)} funding scenario(s) evaluated.")
        summary = " ".join(summary_parts)

        return StartupReport(
            saas_metrics=saas,
            unit_economics=unit_econ,
            burn_runway=burn,
            funding_scenarios=funding,
            stage=stage,
            summary=summary,
        )
