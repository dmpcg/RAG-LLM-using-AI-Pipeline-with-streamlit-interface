"""
Regulatory compliance and audit risk scoring module.

Evaluates existing financial data against SOX compliance heuristics,
SEC filing quality indicators, regulatory ratio thresholds (Basel III /
Dodd-Frank inspired), and audit risk assessment.  Does NOT define new
RATIO_CATALOG entries -- all ratios are computed via safe_divide on
FinancialData fields already tracked by the system.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from export_utils import score_to_grade as _score_to_grade
from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    safe_divide,
)

logger = logging.getLogger(__name__)

# Shared balance-sheet-imbalance tolerance: |Assets - (Liabilities + Equity)|
# is considered material when it exceeds this fraction of total assets (1%).
# Referenced by BOTH sec_filing_quality's consistency check and the SOX-local
# imbalance penalty so there is one tolerance, not two divergent ones.
_BS_IMBALANCE_TOLERANCE = 0.01

# Local penalty (points off SOX risk_score) applied when the balance sheet
# does not balance. This adjusts SOX's OWN risk_score only -- it is NOT added
# to material_weakness_indicators or any list consumed by audit_risk_assessment
# (the imbalance already flows into overall audit risk via sec.red_flags), so
# the signal is never double-counted.
_BS_IMBALANCE_SOX_PENALTY = 15


@dataclass
class RegulatoryThreshold:
    """A single regulatory ratio check result."""

    rule_name: str = ""
    framework: str = ""  # e.g. "Basel III", "Dodd-Frank"
    metric_name: str = ""
    current_value: Optional[float] = None
    threshold_value: float = 0.0
    passes: Optional[bool] = False  # None = insufficient data
    severity: str = "warning"  # info, warning, critical


@dataclass
class SOXComplianceResult:
    """Sarbanes-Oxley compliance heuristic assessment."""

    overall_risk: str = "moderate"  # low, moderate, high
    risk_score: int = 50  # 0 (worst) to 100 (best)
    flags: List[str] = field(default_factory=list)
    material_weakness_indicators: List[str] = field(default_factory=list)
    significant_deficiency_indicators: List[str] = field(default_factory=list)
    checks_performed: int = 0
    checks_passed: int = 0
    # Local penalty (points) deducted from risk_score for a balance-sheet
    # imbalance. Tracked separately so it is NOT re-flowed into audit risk.
    bs_imbalance_penalty: int = 0


@dataclass
class SECFilingQuality:
    """SEC filing data quality and completeness assessment."""

    disclosure_score: int = 0  # 0-100
    grade: str = "F"
    data_completeness_pct: float = 0.0
    missing_critical_fields: List[str] = field(default_factory=list)
    missing_optional_fields: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    consistency_checks_passed: int = 0
    consistency_checks_total: int = 0


@dataclass
class RegulatoryRatioReport:
    """Results of checking regulatory ratio thresholds."""

    thresholds_checked: List[RegulatoryThreshold] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    compliance_pct: Optional[float] = 0.0  # None when all checks had insufficient data
    critical_failures: List[str] = field(default_factory=list)


@dataclass
class AuditRiskAssessment:
    """Overall audit risk assessment combining all compliance factors."""

    risk_level: str = "moderate"  # low, moderate, high, critical
    score: int = 50  # 0-100 (higher = better)
    grade: str = "C"
    going_concern_risk: bool = False
    restatement_risk_indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ComplianceReport:
    """Complete regulatory compliance report."""

    sox: Optional[SOXComplianceResult] = None
    sec: Optional[SECFilingQuality] = None
    regulatory: Optional[RegulatoryRatioReport] = None
    audit_risk: Optional[AuditRiskAssessment] = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class ComplianceScorer:
    """Regulatory compliance scoring using CharlieAnalyzer as foundation."""

    # Regulatory thresholds: (rule_name, framework, metric_name,
    #                         num_field, den_field, operator, threshold, severity)
    # operator: "gte" = current >= threshold is passing, "lte" = current <= threshold is passing
    _REGULATORY_THRESHOLDS = [
        (
            "Minimum Equity Ratio",
            "Basel III (inspired)",
            "equity_ratio",
            "total_equity",
            "total_assets",
            "gte",
            0.06,
            "critical",
        ),
        (
            "Maximum Leverage Ratio",
            "Basel III (inspired)",
            "leverage_ratio",
            "total_debt",
            "total_assets",
            "lte",
            0.80,
            "critical",
        ),
        (
            "Minimum Liquidity Ratio",
            "Basel III (inspired)",
            "current_ratio",
            "current_assets",
            "current_liabilities",
            "gte",
            1.0,
            "warning",
        ),
        (
            "Minimum Interest Coverage",
            "Dodd-Frank (inspired)",
            "interest_coverage",
            "ebit",
            "interest_expense",
            "gte",
            1.5,
            "critical",
        ),
        (
            "Maximum Debt/EBITDA",
            "Dodd-Frank (inspired)",
            "debt_to_ebitda",
            "total_debt",
            "ebitda",
            "lte",
            6.0,
            "warning",
        ),
        (
            "Positive Net Margin",
            "General Prudential",
            "net_margin",
            "net_income",
            "revenue",
            "gte",
            0.0,
            "warning",
        ),
    ]

    # Fields that should be present for a credible filing
    _CRITICAL_FIELDS = [
        "revenue",
        "net_income",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "operating_cash_flow",
        "current_assets",
        "current_liabilities",
    ]

    _OPTIONAL_FIELDS = [
        "ebit",
        "ebitda",
        "gross_profit",
        "operating_income",
        "total_debt",
        "interest_expense",
        "cogs",
        "depreciation",
        "inventory",
        "accounts_receivable",
        "accounts_payable",
        "capex",
    ]

    def __init__(self, analyzer: Optional[CharlieAnalyzer] = None):
        self._analyzer = analyzer or CharlieAnalyzer()

    # ------------------------------------------------------------------
    # 1. SOX Compliance
    # ------------------------------------------------------------------

    def sox_compliance(self, data: FinancialData) -> SOXComplianceResult:
        """Heuristic SOX compliance assessment.

        Checks for red flags that could indicate material weakness or
        significant deficiency in internal controls:
          - OCF / Net Income divergence (earnings quality)
          - High AR / Revenue ratio (revenue recognition risk)
          - Weak interest coverage (going concern)
          - Negative equity (solvency risk)
          - Operating loss (sustainability risk)
          - Accrual gap (earnings manipulation signal)

        Args:
            data: FinancialData instance.

        Returns:
            SOXComplianceResult with flags and risk score.
        """
        flags: List[str] = []
        material_weakness: List[str] = []
        significant_deficiency: List[str] = []
        checks_done = 0
        checks_passed = 0

        # Check 1: OCF / Net Income divergence
        checks_done += 1
        ocf_ni = safe_divide(data.operating_cash_flow, data.net_income)
        if ocf_ni is not None:
            # When NI is negative, a positive ratio means OCF is also negative
            # (both negative -> positive ratio), which is not a sign of quality
            ni_negative = data.net_income is not None and data.net_income < 0
            if ni_negative:
                # Negative NI: flag as concern regardless of OCF/NI ratio
                flags.append(f"Negative net income (${data.net_income:,.0f}) -- earnings quality indeterminate")
                significant_deficiency.append("Net income is negative; OCF/NI ratio is unreliable")
            elif ocf_ni < 0.5:
                flags.append(f"OCF/NI ratio is {ocf_ni:.2f} (below 0.50) -- earnings quality concern")
                significant_deficiency.append("Cash flow diverges significantly from reported earnings")
            else:
                checks_passed += 1
        else:
            checks_passed += 1  # Insufficient data, no flag

        # Check 2: AR / Revenue ratio
        checks_done += 1
        ar_rev = safe_divide(data.accounts_receivable, data.revenue)
        if ar_rev is not None:
            if ar_rev > 0.40:
                flags.append(f"AR/Revenue is {ar_rev:.2f} (above 0.40) -- potential revenue recognition risk")
                significant_deficiency.append("Unusually high receivables relative to revenue")
            else:
                checks_passed += 1
        else:
            checks_passed += 1

        # Check 3: Interest coverage
        checks_done += 1
        ic = safe_divide(data.ebit, data.interest_expense)
        if ic is not None:
            if ic < 1.0:
                flags.append(f"Interest coverage is {ic:.2f}x (below 1.0) -- cannot cover interest payments")
                material_weakness.append("EBIT insufficient to cover interest expense")
            elif ic < 2.0:
                flags.append(f"Interest coverage is {ic:.2f}x (below 2.0) -- thin debt service margin")
                significant_deficiency.append("Interest coverage below prudent threshold")
            else:
                checks_passed += 1
        else:
            checks_passed += 1

        # Check 4: Negative equity
        checks_done += 1
        if data.total_equity is not None and data.total_equity < 0:
            flags.append(f"Negative equity (${data.total_equity:,.0f}) -- insolvency risk")
            material_weakness.append("Company has negative shareholders' equity")
        else:
            checks_passed += 1

        # Check 5: Operating loss
        checks_done += 1
        if data.operating_income is not None and data.operating_income < 0:
            flags.append(f"Operating loss (${data.operating_income:,.0f}) -- core business unprofitable")
            significant_deficiency.append("Operating income is negative")
        else:
            checks_passed += 1

        # Check 6: Accrual gap (NI >> OCF or OCF >> NI)
        checks_done += 1
        if (
            data.net_income is not None
            and data.operating_cash_flow is not None
            and data.revenue is not None
            and data.revenue > 0
        ):
            accrual_gap = safe_divide(abs(data.net_income - data.operating_cash_flow), data.revenue, default=0.0)
            if accrual_gap > 0.15:
                flags.append(f"Accrual gap is {accrual_gap:.1%} of revenue -- potential earnings manipulation signal")
                significant_deficiency.append("Large gap between accrual earnings and cash earnings")
            else:
                checks_passed += 1
        else:
            checks_passed += 1

        # Check 7: Balance-sheet imbalance (SOX-LOCAL penalty only).
        # Reads the SAME imbalance condition as sec_filing_quality's consistency
        # check (|Assets - (L+E)| / Assets >= _BS_IMBALANCE_TOLERANCE) and lowers
        # SOX's own risk_score. Deliberately NOT appended to material_weakness or
        # significant_deficiency -- the imbalance already reaches overall audit
        # risk via sec.red_flags, so adding it here would double-count it.
        bs_imbalance_penalty = 0
        if data.total_assets is not None and data.total_liabilities is not None and data.total_equity is not None:
            bs_sum = data.total_liabilities + data.total_equity
            bs_diff = abs(data.total_assets - bs_sum)
            if safe_divide(bs_diff, data.total_assets, default=1.0) >= _BS_IMBALANCE_TOLERANCE:
                bs_imbalance_penalty = _BS_IMBALANCE_SOX_PENALTY
                flags.append(
                    f"Balance sheet does not balance: Assets=${data.total_assets:,.0f} "
                    f"vs L+E=${bs_sum:,.0f} (diff=${bs_diff:,.0f}) -- "
                    "internal-control concern"
                )

        # Score: start from 100, subtract penalties
        score = 100
        score -= len(material_weakness) * 20
        score -= len(significant_deficiency) * 10
        score -= bs_imbalance_penalty
        score = max(0, min(100, score))

        if material_weakness:
            risk = "high"
        elif significant_deficiency:
            risk = "moderate"
        else:
            risk = "low"

        return SOXComplianceResult(
            overall_risk=risk,
            risk_score=score,
            flags=flags,
            material_weakness_indicators=material_weakness,
            significant_deficiency_indicators=significant_deficiency,
            checks_performed=checks_done,
            checks_passed=checks_passed,
            bs_imbalance_penalty=bs_imbalance_penalty,
        )

    # ------------------------------------------------------------------
    # 2. SEC Filing Quality
    # ------------------------------------------------------------------

    def sec_filing_quality(self, data: FinancialData) -> SECFilingQuality:
        """Assess SEC filing data quality and completeness.

        Scoring breakdown:
          - Data completeness (50 pts): percentage of all fields present
          - Critical field coverage (30 pts): critical fields present
          - Consistency checks (20 pts): balance sheet equation, gross profit calc

        Args:
            data: FinancialData instance.

        Returns:
            SECFilingQuality with disclosure score and flags.
        """
        # Completeness check: all fields
        all_fields = self._CRITICAL_FIELDS + self._OPTIONAL_FIELDS
        present_count = sum(1 for f in all_fields if getattr(data, f, None) is not None)
        completeness_pct = present_count / len(all_fields) if all_fields else 0.0
        completeness_pts = int(50 * completeness_pct)

        # Critical fields
        missing_critical = [f for f in self._CRITICAL_FIELDS if getattr(data, f, None) is None]
        missing_optional = [f for f in self._OPTIONAL_FIELDS if getattr(data, f, None) is None]
        critical_coverage = (
            (len(self._CRITICAL_FIELDS) - len(missing_critical)) / len(self._CRITICAL_FIELDS)
            if self._CRITICAL_FIELDS
            else 0.0
        )
        critical_pts = int(30 * critical_coverage)

        # Consistency checks
        consistency_total = 0
        consistency_passed = 0
        red_flags: List[str] = []

        # Check 1: Balance sheet equation (Assets = Liabilities + Equity)
        consistency_total += 1
        if data.total_assets is not None and data.total_liabilities is not None and data.total_equity is not None:
            bs_sum = data.total_liabilities + data.total_equity
            bs_diff = abs(data.total_assets - bs_sum)
            # Allow 1% tolerance (shared with the SOX-local imbalance check)
            if safe_divide(bs_diff, data.total_assets, default=1.0) < _BS_IMBALANCE_TOLERANCE:
                consistency_passed += 1
            else:
                red_flags.append(
                    f"Balance sheet does not balance: Assets=${data.total_assets:,.0f} "
                    f"vs L+E=${bs_sum:,.0f} (diff=${bs_diff:,.0f})"
                )

        # Check 2: Gross profit = Revenue - COGS
        consistency_total += 1
        if data.revenue is not None and data.cogs is not None and data.gross_profit is not None:
            expected_gp = data.revenue - data.cogs
            gp_diff = abs(data.gross_profit - expected_gp)
            if safe_divide(gp_diff, data.revenue, default=1.0) < 0.01:
                consistency_passed += 1
            else:
                red_flags.append(
                    f"Gross profit inconsistency: reported=${data.gross_profit:,.0f} "
                    f"vs Revenue-COGS=${expected_gp:,.0f}"
                )

        # Check 3: Operating income should be <= Gross profit
        consistency_total += 1
        if data.operating_income is not None and data.gross_profit is not None and data.gross_profit > 0:
            if data.operating_income <= data.gross_profit * 1.01:
                consistency_passed += 1
            else:
                red_flags.append("Operating income exceeds gross profit -- unusual accounting treatment")
        else:
            consistency_passed += 1  # Can't check, assume OK

        consistency_pct = consistency_passed / consistency_total if consistency_total else 1.0
        consistency_pts = int(20 * consistency_pct)

        total = completeness_pts + critical_pts + consistency_pts
        total = max(0, min(100, total))

        return SECFilingQuality(
            disclosure_score=total,
            grade=_score_to_grade(total),
            data_completeness_pct=round(completeness_pct * 100, 1),
            missing_critical_fields=missing_critical,
            missing_optional_fields=missing_optional,
            red_flags=red_flags,
            consistency_checks_passed=consistency_passed,
            consistency_checks_total=consistency_total,
        )

    # ------------------------------------------------------------------
    # 3. Regulatory Ratios
    # ------------------------------------------------------------------

    def regulatory_ratios(self, data: FinancialData) -> RegulatoryRatioReport:
        """Evaluate financial data against regulatory ratio thresholds.

        Args:
            data: FinancialData instance.

        Returns:
            RegulatoryRatioReport with pass/fail results.
        """
        results: List[RegulatoryThreshold] = []
        pass_count = 0
        fail_count = 0
        critical_failures: List[str] = []

        for (
            rule_name,
            framework,
            metric_name,
            num_field,
            den_field,
            operator,
            threshold,
            severity,
        ) in self._REGULATORY_THRESHOLDS:
            num = getattr(data, num_field, None)
            den = getattr(data, den_field, None)
            current = safe_divide(num, den)

            if current is not None:
                if operator == "gte":
                    passes = current >= threshold
                else:  # lte
                    passes = current <= threshold
            else:
                passes = None  # Insufficient data -- excluded from counts

            if passes is True:
                pass_count += 1
            elif passes is False:
                fail_count += 1
                if severity == "critical":
                    critical_failures.append(
                        f"{rule_name}: {metric_name}={current:.4f} (threshold {operator} {threshold})"
                    )

            results.append(
                RegulatoryThreshold(
                    rule_name=rule_name,
                    framework=framework,
                    metric_name=metric_name,
                    current_value=round(current, 4) if current is not None else None,
                    threshold_value=threshold,
                    passes=passes,
                    severity=severity,
                )
            )

        total = pass_count + fail_count
        compliance_pct = (pass_count / total * 100) if total > 0 else None

        return RegulatoryRatioReport(
            thresholds_checked=results,
            pass_count=pass_count,
            fail_count=fail_count,
            compliance_pct=round(compliance_pct, 1) if compliance_pct is not None else None,
            critical_failures=critical_failures,
        )

    # ------------------------------------------------------------------
    # 4. Audit Risk Assessment
    # ------------------------------------------------------------------

    def audit_risk_assessment(
        self,
        data: FinancialData,
        sox: Optional[SOXComplianceResult] = None,
        sec: Optional[SECFilingQuality] = None,
        reg: Optional[RegulatoryRatioReport] = None,
    ) -> AuditRiskAssessment:
        """Compute overall audit risk by combining SOX, SEC, and regulatory results.

        Scoring breakdown:
          - SOX component:         40 pts (from sox.risk_score)
          - SEC component:         30 pts (from sec.disclosure_score)
          - Regulatory component:  30 pts (from reg.compliance_pct)

        Going concern risk triggered if:
          - Negative equity, OR
          - Z-score < 1.81 (distress zone), OR
          - 2+ critical regulatory failures

        Args:
            data: FinancialData instance.
            sox: Pre-computed SOX result, or None to compute.
            sec: Pre-computed SEC result, or None to compute.
            reg: Pre-computed regulatory result, or None to compute.

        Returns:
            AuditRiskAssessment with score, grade, and flags.
        """
        if sox is None:
            sox = self.sox_compliance(data)
        if sec is None:
            sec = self.sec_filing_quality(data)
        if reg is None:
            reg = self.regulatory_ratios(data)

        # Component scores
        sox_pts = int(40 * sox.risk_score / 100)
        sec_pts = int(30 * sec.disclosure_score / 100)
        reg_pts = int(30 * reg.compliance_pct / 100) if reg.compliance_pct is not None else 0

        total = sox_pts + sec_pts + reg_pts
        total = max(0, min(100, total))

        # Going concern assessment
        going_concern = False
        restatement_indicators: List[str] = []
        recommendations: List[str] = []

        # Check 1: Negative equity
        if data.total_equity is not None and data.total_equity < 0:
            going_concern = True
            restatement_indicators.append("Negative shareholders' equity")

        # Check 2: Z-score distress
        try:
            z = self._analyzer.altman_z_score(data)
            if z and z.z_score is not None and z.z_score < 1.81:
                going_concern = True
                restatement_indicators.append(f"Altman Z-score {z.z_score:.2f} in distress zone")
        except (AttributeError, TypeError, ValueError):
            pass  # Z-score unavailable for this data

        # Check 3: Critical regulatory failures >= 2
        if len(reg.critical_failures) >= 2:
            going_concern = True
            restatement_indicators.append(f"{len(reg.critical_failures)} critical regulatory threshold failures")

        # SOX material weaknesses
        if sox.material_weakness_indicators:
            restatement_indicators.extend(sox.material_weakness_indicators)

        # SEC red flags
        if sec.red_flags:
            restatement_indicators.extend(sec.red_flags)

        # Build recommendations
        if going_concern:
            recommendations.append(
                "Going concern risk identified -- auditor should consider going concern opinion paragraph"
            )
        if sox.overall_risk == "high":
            recommendations.append("High SOX risk -- recommend enhanced internal controls review")
        if sec.disclosure_score < 50:
            recommendations.append("Low filing quality score -- recommend improving data completeness and consistency")
        if reg.critical_failures:
            recommendations.append("Critical regulatory threshold failures -- recommend immediate remediation plan")
        if not recommendations:
            recommendations.append("No critical findings -- standard audit procedures recommended")

        # Risk level
        if going_concern or total < 30:
            risk_level = "critical"
        elif total < 50:
            risk_level = "high"
        elif total < 70:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return AuditRiskAssessment(
            risk_level=risk_level,
            score=total,
            grade=_score_to_grade(total),
            going_concern_risk=going_concern,
            restatement_risk_indicators=restatement_indicators,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # 5. Full Compliance Report
    # ------------------------------------------------------------------

    def full_compliance_report(self, data: FinancialData) -> ComplianceReport:
        """Run the complete compliance analysis pipeline.

        Args:
            data: FinancialData instance.

        Returns:
            ComplianceReport with SOX, SEC, regulatory, and audit risk results.
        """
        sox = self.sox_compliance(data)
        sec = self.sec_filing_quality(data)
        reg = self.regulatory_ratios(data)
        audit = self.audit_risk_assessment(data, sox, sec, reg)

        summary_parts = [
            f"SOX risk: {sox.overall_risk} (score {sox.risk_score}/100).",
            f"SEC filing quality: {sec.disclosure_score}/100 (Grade {sec.grade}).",
            f"Regulatory compliance: "
            f"{reg.compliance_pct:.0f}% ({reg.pass_count}/{reg.pass_count + reg.fail_count} passed)."
            if reg.compliance_pct is not None
            else "Regulatory compliance: N/A (insufficient data for all checks).",
            f"Audit risk: {audit.risk_level} (score {audit.score}/100, Grade {audit.grade}).",
        ]
        if audit.going_concern_risk:
            summary_parts.append("GOING CONCERN RISK IDENTIFIED.")
        if reg.critical_failures:
            summary_parts.append(f"{len(reg.critical_failures)} critical regulatory failure(s).")

        return ComplianceReport(
            sox=sox,
            sec=sec,
            regulatory=reg,
            audit_risk=audit,
            summary=" ".join(summary_parts),
        )
