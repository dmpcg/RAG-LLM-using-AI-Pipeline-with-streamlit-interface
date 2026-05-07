"""
Underwriting and credit analysis module.

Composes existing CharlieAnalyzer ratios with scoring rubrics for loan decisions.
Does NOT define new ratios -- it reuses safe_divide and FinancialData from
financial_analyzer and layers credit-scoring logic on top.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from export_utils import score_to_grade as _score_to_grade
from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    safe_divide,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LoanStructure:
    """Proposed loan parameters."""

    principal: float = 0.0
    annual_rate: float = 0.05  # 5 % default
    term_years: int = 5
    amortization_years: int = 5
    loan_type: str = "term"  # term, revolver, bridge
    # P0-6: borrower existing annual debt service (interest + scheduled principal).
    # When None, debt_capacity() estimates from FinancialData fields.
    existing_debt_service: Optional[float] = None


@dataclass
class CreditScorecard:
    """Credit assessment with 0-100 score and letter grade."""

    total_score: int = 0
    grade: str = "F"  # A, B, C, D, F
    category_scores: Dict[str, int] = field(default_factory=dict)
    # 5 categories x 20 points each = 100 max
    recommendation: str = "decline"  # approve, conditional, decline
    conditions: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class DebtCapacityResult:
    """Analysis of additional debt capacity."""

    current_total_debt: Optional[float] = None
    current_ebitda: Optional[float] = None
    current_leverage: Optional[float] = None  # debt / EBITDA
    max_leverage_target: float = 3.5  # conservative target
    max_additional_debt: Optional[float] = None
    pro_forma_debt: Optional[float] = None
    pro_forma_leverage: Optional[float] = None
    pro_forma_dscr: Optional[float] = None  # debt service coverage ratio
    headroom_pct: Optional[float] = None  # % capacity remaining
    assessment: str = ""


@dataclass
class CovenantPackage:
    """Recommended financial covenants for loan agreement."""

    financial_covenants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # e.g. {"min_current_ratio": {"threshold": 1.25, "frequency": "quarterly"}}
    reporting_requirements: List[str] = field(default_factory=list)
    events_of_default: List[str] = field(default_factory=list)
    covenant_tier: str = "standard"  # light, standard, heavy


@dataclass
class UnderwritingReport:
    """Complete underwriting analysis report."""

    scorecard: Optional[CreditScorecard] = None
    debt_capacity: Optional[DebtCapacityResult] = None
    covenants: Optional[CovenantPackage] = None
    loan: Optional[LoanStructure] = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Scoring helpers (pure functions)
# ---------------------------------------------------------------------------

def _grade_to_recommendation(grade: str) -> str:
    """Map letter grade to a lending recommendation."""
    if grade in ("A", "B"):
        return "approve"
    if grade == "C":
        return "conditional"
    return "decline"


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------


class UnderwritingAnalyzer:
    """Credit analysis and loan underwriting using CharlieAnalyzer as foundation."""

    def __init__(self, analyzer: Optional[CharlieAnalyzer] = None):
        self._analyzer = analyzer or CharlieAnalyzer()

    # ------------------------------------------------------------------
    # 1. Credit Scorecard
    # ------------------------------------------------------------------

    def credit_scorecard(self, data: FinancialData) -> CreditScorecard:
        """Score the borrower across five categories (each 0-20, total 0-100)."""

        scores: Dict[str, int] = {}

        # --- Profitability (20 pts) ---
        net_margin = safe_divide(data.net_income, data.revenue)
        roa = safe_divide(data.net_income, data.total_assets)
        scores["profitability"] = self._score_profitability(net_margin, roa)

        # --- Leverage (20 pts) ---
        d_e = safe_divide(data.total_debt, data.total_equity)
        d_a = safe_divide(data.total_debt, data.total_assets)
        scores["leverage"] = self._score_leverage(d_e, d_a)

        # --- Liquidity (20 pts) ---
        cr = safe_divide(data.current_assets, data.current_liabilities)
        cash_r = safe_divide(data.cash, data.current_liabilities)
        scores["liquidity"] = self._score_liquidity(cr, cash_r)

        # --- Cash Flow (20 pts) ---
        ocf_debt = safe_divide(data.operating_cash_flow, data.total_debt)
        if data.operating_cash_flow is not None:
            fcf = (data.operating_cash_flow or 0) - (data.capex or 0)
            fcf_margin = safe_divide(fcf, data.revenue)
        else:
            fcf_margin = None
        scores["cash_flow"] = self._score_cash_flow(ocf_debt, fcf_margin)

        # --- Stability (20 pts) ---
        ic = safe_divide(data.ebit, data.interest_expense)
        ebitda_margin = safe_divide(data.ebitda, data.revenue)
        scores["stability"] = self._score_stability(ic, ebitda_margin)

        total = sum(scores.values())
        grade = _score_to_grade(total)
        recommendation = _grade_to_recommendation(grade)

        strengths = [cat for cat, pts in scores.items() if pts >= 15]
        weaknesses = [cat for cat, pts in scores.items() if pts <= 5]

        conditions: List[str] = []
        if recommendation == "conditional":
            if "leverage" in weaknesses:
                conditions.append("Require debt reduction plan within 12 months")
            if "liquidity" in weaknesses:
                conditions.append("Maintain minimum cash reserve equal to 3 months debt service")
            if "cash_flow" in weaknesses:
                conditions.append("Provide monthly cash-flow forecasts for first year")
            if not conditions:
                conditions.append("Subject to enhanced monitoring and quarterly review")

        return CreditScorecard(
            total_score=total,
            grade=grade,
            category_scores=scores,
            recommendation=recommendation,
            conditions=conditions,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    # --- individual scoring helpers (static-like, kept as methods for readability) ---

    @staticmethod
    def _score_profitability(
        net_margin: Optional[float], roa: Optional[float]
    ) -> int:
        nm = net_margin or 0
        r = roa or 0
        if nm > 0.10 and r > 0.08:
            return 20
        if nm > 0.05 and r > 0.04:
            return 15
        if nm > 0.02 and r > 0.02:
            return 10
        if nm > 0 and r > 0:
            return 5
        return 0

    @staticmethod
    def _score_leverage(
        d_e: Optional[float], d_a: Optional[float]
    ) -> int:
        de = d_e if d_e is not None else 999
        da = d_a if d_a is not None else 999
        # Negative D/E means negative equity (worst case), not low leverage
        if de < 0:
            return 0
        if de < 0.5 and da < 0.3:
            return 20
        if de < 1.0 and da < 0.5:
            return 15
        if de < 2.0 and da < 0.6:
            return 10
        if de < 3.0:
            return 5
        return 0

    @staticmethod
    def _score_liquidity(
        current_ratio: Optional[float], cash_ratio: Optional[float]
    ) -> int:
        cr = current_ratio or 0
        cashr = cash_ratio or 0
        if cr > 2.0 and cashr > 0.5:
            return 20
        if cr > 1.5 and cashr > 0.3:
            return 15
        if cr > 1.2 and cashr > 0.1:
            return 10
        if cr > 1.0:
            return 5
        return 0

    @staticmethod
    def _score_cash_flow(
        ocf_debt: Optional[float], fcf_margin: Optional[float]
    ) -> int:
        od = ocf_debt or 0
        fm = fcf_margin or 0
        if od > 0.4 and fm > 0.10:
            return 20
        if od > 0.25 and fm > 0.05:
            return 15
        if od > 0.15:
            return 10
        if od > 0.05:
            return 5
        return 0

    @staticmethod
    def _score_stability(
        interest_coverage: Optional[float], ebitda_margin: Optional[float]
    ) -> int:
        ic = interest_coverage or 0
        em = ebitda_margin or 0
        if ic > 6.0 and em > 0.20:
            return 20
        if ic > 4.0 and em > 0.15:
            return 15
        if ic > 2.5 and em > 0.10:
            return 10
        if ic > 1.5:
            return 5
        return 0

    # ------------------------------------------------------------------
    # 2. Debt Capacity
    # ------------------------------------------------------------------

    def debt_capacity(
        self,
        data: FinancialData,
        proposed_loan: Optional[LoanStructure] = None,
    ) -> DebtCapacityResult:
        """Estimate how much additional debt the company can support.

        P0-7: When EBITDA is not strictly positive, leverage-based capacity is
        undefined; max_additional_debt is returned as None (not 0)
        with an explicit assessment string.

        P0-6: Pro-forma DSCR includes the borrower existing annual debt
        service in addition to service on the proposed loan. The existing
        service is taken from LoanStructure.existing_debt_service when
        provided, otherwise estimated from
        FinancialData.interest_expense + total_debt / term_years.
        """

        target = 3.5  # max debt/EBITDA

        current_debt = data.total_debt or 0
        ebitda_raw = data.ebitda

        current_leverage = safe_divide(data.total_debt, data.ebitda)

        # P0-7: short-circuit on non-positive EBITDA -- leverage-based
        # capacity is undefined for borrowers with no/negative cash earnings.
        if ebitda_raw is None or ebitda_raw <= 0:
            max_additional: Optional[float] = None
        else:
            max_capacity = target * ebitda_raw
            max_additional = max(0.0, max_capacity - current_debt)

        result = DebtCapacityResult(
            current_total_debt=data.total_debt,
            current_ebitda=data.ebitda,
            current_leverage=current_leverage,
            max_leverage_target=target,
            max_additional_debt=max_additional,
        )

        if proposed_loan is not None:
            pro_forma_debt = current_debt + proposed_loan.principal
            result.pro_forma_debt = pro_forma_debt
            result.pro_forma_leverage = safe_divide(pro_forma_debt, data.ebitda)

            # Annual debt service on the *proposed* loan
            term = proposed_loan.term_years if proposed_loan.term_years > 0 else 1
            annual_repayment = safe_divide(proposed_loan.principal, term, default=0.0) or 0.0
            annual_interest = proposed_loan.principal * proposed_loan.annual_rate
            new_loan_service = annual_repayment + annual_interest

            # P0-6: include borrower existing debt service so DSCR reflects
            # total burden, not just the new loan.
            existing_service = self._estimate_existing_debt_service(data, proposed_loan)

            total_debt_service = new_loan_service + (existing_service or 0.0)
            result.pro_forma_dscr = safe_divide(ebitda_raw, total_debt_service)

            if max_additional is not None and max_additional > 0:
                remaining = max_additional - proposed_loan.principal
                result.headroom_pct = safe_divide(remaining, max_additional)
            elif max_additional == 0:
                result.headroom_pct = 0.0
            else:
                # Capacity undefined (non-positive EBITDA)
                result.headroom_pct = None

        # Build assessment text
        result.assessment = self._build_capacity_assessment(result)
        return result

    @staticmethod
    def _estimate_existing_debt_service(
        data: FinancialData,
        proposed_loan: LoanStructure,
    ) -> Optional[float]:
        """Return the borrower annual existing debt service.

        Order of precedence:
        1. proposed_loan.existing_debt_service if explicitly set (>=0).
        2. interest_expense + total_debt / term_years estimate.
        3. None when neither path has data.
        """
        explicit = getattr(proposed_loan, "existing_debt_service", None)
        if explicit is not None and explicit >= 0:
            return float(explicit)

        interest = data.interest_expense
        existing_debt = data.total_debt
        term = proposed_loan.term_years if proposed_loan.term_years > 0 else 1

        if interest is None and (existing_debt is None or existing_debt <= 0):
            return None

        principal_amort = (
            (existing_debt / term) if existing_debt and existing_debt > 0 else 0.0
        )
        interest_part = float(interest) if interest is not None else 0.0
        return interest_part + principal_amort


    @staticmethod
    def _build_capacity_assessment(r: DebtCapacityResult) -> str:
        """Produce a human-readable assessment string."""
        parts: List[str] = []

        if r.current_leverage is not None:
            parts.append(
                f"Current leverage is {r.current_leverage:.2f}x debt/EBITDA "
                f"(target max {r.max_leverage_target:.1f}x)."
            )
        else:
            parts.append("Insufficient data to calculate current leverage.")

        if r.max_additional_debt is None:
            # P0-7: explicit message for non-positive EBITDA borrowers
            parts.append(
                "Additional debt capacity is undefined: EBITDA is not "
                "positive, so leverage-based capacity cannot be calculated. "
                "Underwriting must rely on collateral and alternative "
                "coverage measures."
            )
        elif r.max_additional_debt > 0:
            parts.append(
                f"Estimated additional debt capacity: "
                f"${r.max_additional_debt:,.0f}."
            )
        else:
            parts.append(
                "Company appears fully leveraged relative to the "
                f"{r.max_leverage_target:.1f}x target."
            )

        if r.pro_forma_dscr is not None:
            if r.pro_forma_dscr >= 1.5:
                parts.append(
                    f"Pro-forma DSCR of {r.pro_forma_dscr:.2f}x is strong."
                )
            elif r.pro_forma_dscr >= 1.2:
                parts.append(
                    f"Pro-forma DSCR of {r.pro_forma_dscr:.2f}x is adequate."
                )
            elif r.pro_forma_dscr >= 1.0:
                parts.append(
                    f"Pro-forma DSCR of {r.pro_forma_dscr:.2f}x is marginal."
                )
            else:
                parts.append(
                    f"Pro-forma DSCR of {r.pro_forma_dscr:.2f}x is below 1.0x "
                    "-- debt service exceeds EBITDA."
                )

        if r.pro_forma_leverage is not None and r.pro_forma_leverage > r.max_leverage_target:
            parts.append(
                "WARNING: pro-forma leverage exceeds target -- "
                "loan may not be supportable."
            )

        return " ".join(parts) if parts else "Insufficient data for assessment."

    # ------------------------------------------------------------------
    # 3. Covenant Recommendations
    # ------------------------------------------------------------------

    def recommend_covenants(
        self,
        data: FinancialData,
        scorecard: CreditScorecard,
    ) -> CovenantPackage:
        """Recommend financial covenants calibrated to credit quality."""

        tier = self._covenant_tier(scorecard.grade)

        covenants = self._base_financial_covenants(tier)

        if tier == "heavy":
            covenants["min_fixed_charge_coverage"] = {
                "threshold": 1.10,
                "frequency": "quarterly",
                "description": "Minimum fixed charge coverage ratio",
            }
            if data.capex is not None:
                covenants["max_capex"] = {
                    "threshold": round(data.capex * 1.10, 2),
                    "frequency": "annual",
                    "description": "Maximum annual capital expenditure",
                }

        reporting = self._reporting_requirements(tier)
        eod = self._events_of_default()

        return CovenantPackage(
            financial_covenants=covenants,
            reporting_requirements=reporting,
            events_of_default=eod,
            covenant_tier=tier,
        )

    @staticmethod
    def _covenant_tier(grade: str) -> str:
        if grade == "A":
            return "light"
        if grade == "B":
            return "standard"
        return "heavy"

    @staticmethod
    def _base_financial_covenants(tier: str) -> Dict[str, Dict[str, Any]]:
        thresholds = {
            "light":    {"cr": 1.10, "de": 4.0, "ic": 2.0},
            "standard": {"cr": 1.25, "de": 3.5, "ic": 2.5},
            "heavy":    {"cr": 1.50, "de": 3.0, "ic": 3.0},
        }
        t = thresholds.get(tier, thresholds["standard"])
        return {
            "min_current_ratio": {
                "threshold": t["cr"],
                "frequency": "quarterly",
                "description": "Minimum current ratio",
            },
            "max_debt_to_ebitda": {
                "threshold": t["de"],
                "frequency": "quarterly",
                "description": "Maximum total debt to EBITDA",
            },
            "min_interest_coverage": {
                "threshold": t["ic"],
                "frequency": "quarterly",
                "description": "Minimum interest coverage ratio",
            },
        }

    @staticmethod
    def _reporting_requirements(tier: str) -> List[str]:
        reqs = [
            "Quarterly unaudited financial statements within 45 days of quarter-end",
            "Annual audited financial statements within 90 days of fiscal year-end",
        ]
        if tier == "heavy":
            reqs.insert(
                0,
                "Monthly management accounts within 30 days of month-end",
            )
        return reqs

    @staticmethod
    def _events_of_default() -> List[str]:
        return [
            "Payment default (principal or interest)",
            "Financial covenant breach not cured within 30 days",
            "Material adverse change in financial condition",
            "Cross-default on other indebtedness exceeding $500,000",
        ]

    # ------------------------------------------------------------------
    # 4. Full Underwriting Report
    # ------------------------------------------------------------------

    def full_underwriting(
        self,
        data: FinancialData,
        loan: Optional[LoanStructure] = None,
    ) -> UnderwritingReport:
        """Run full underwriting pipeline and return consolidated report."""

        sc = self.credit_scorecard(data)
        dc = self.debt_capacity(data, loan)
        cov = self.recommend_covenants(data, sc)

        summary = self._build_summary(sc, dc, cov, loan)

        return UnderwritingReport(
            scorecard=sc,
            debt_capacity=dc,
            covenants=cov,
            loan=loan,
            summary=summary,
        )

    @staticmethod
    def _build_summary(
        sc: CreditScorecard,
        dc: DebtCapacityResult,
        cov: CovenantPackage,
        loan: Optional[LoanStructure],
    ) -> str:
        lines: List[str] = [
            f"Credit Score: {sc.total_score}/100 (Grade {sc.grade}) "
            f"-- Recommendation: {sc.recommendation.upper()}.",
        ]

        if sc.strengths:
            lines.append(f"Strengths: {', '.join(sc.strengths)}.")
        if sc.weaknesses:
            lines.append(f"Weaknesses: {', '.join(sc.weaknesses)}.")

        if dc.current_leverage is not None:
            lines.append(
                f"Current leverage: {dc.current_leverage:.2f}x debt/EBITDA."
            )

        if loan is not None:
            lines.append(
                f"Proposed loan: ${loan.principal:,.0f} at {loan.annual_rate:.1%} "
                f"for {loan.term_years} years ({loan.loan_type})."
            )
            if dc.pro_forma_dscr is not None:
                lines.append(f"Pro-forma DSCR: {dc.pro_forma_dscr:.2f}x.")
            if dc.pro_forma_leverage is not None:
                lines.append(
                    f"Pro-forma leverage: {dc.pro_forma_leverage:.2f}x."
                )

        lines.append(f"Covenant tier: {cov.covenant_tier}.")

        if sc.conditions:
            lines.append("Conditions: " + "; ".join(sc.conditions) + ".")

        return " ".join(lines)
