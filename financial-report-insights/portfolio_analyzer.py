"""
Portfolio and multi-company financial analysis module.

Composes existing CharlieAnalyzer ratios with portfolio-level metrics:
correlation matrices, diversification scoring, and aggregate risk dashboards.
Does NOT define new RATIO_CATALOG entries -- it evaluates existing ratios
across multiple companies and layers portfolio-level logic on top.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from export_utils import score_to_grade as _score_to_grade
from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    safe_divide,
)

logger = logging.getLogger(__name__)


@dataclass
class CompanySnapshot:
    """Summary snapshot of a single company's financial health."""

    name: str = ""
    health_score: int = 0
    health_grade: str = "F"
    key_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    z_score: Optional[float] = None
    z_zone: str = ""


@dataclass
class CorrelationMatrix:
    """Pairwise correlation of financial ratios across companies."""

    company_names: List[str] = field(default_factory=list)
    ratio_names: List[str] = field(default_factory=list)
    matrix: List[List[float]] = field(default_factory=list)
    avg_correlation: float = 0.0
    interpretation: str = ""


@dataclass
class DiversificationScore:
    """Portfolio diversification assessment (0-100)."""

    overall_score: int = 0
    grade: str = "F"
    hhi_revenue: float = 0.0
    hhi_assets: float = 0.0
    correlation_penalty: float = 0.0
    revenue_concentration: str = ""
    asset_concentration: str = ""
    interpretation: str = ""


@dataclass
class PortfolioRiskSummary:
    """Aggregate risk assessment across all portfolio companies."""

    num_companies: int = 0
    avg_health_score: float = 0.0
    min_health_score: int = 0
    max_health_score: int = 0
    weakest_company: str = ""
    strongest_company: str = ""
    distress_count: int = 0
    distress_companies: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    overall_risk_level: str = "moderate"  # low, moderate, high, critical


@dataclass
class PortfolioReport:
    """Complete portfolio analysis report."""

    snapshots: List[CompanySnapshot] = field(default_factory=list)
    correlation: Optional[CorrelationMatrix] = None
    diversification: Optional[DiversificationScore] = None
    risk_summary: Optional[PortfolioRiskSummary] = None
    num_companies: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Core ratios used for cross-company comparison
# ---------------------------------------------------------------------------

_CORE_RATIOS = [
    ("net_margin", "net_income", "revenue"),
    ("roa", "net_income", "total_assets"),
    ("current_ratio", "current_assets", "current_liabilities"),
    ("debt_to_assets", "total_debt", "total_assets"),
    ("interest_coverage", "ebit", "interest_expense"),
]

_CORE_RATIO_NAMES = [r[0] for r in _CORE_RATIOS]


# ---------------------------------------------------------------------------
# HHI helpers
# ---------------------------------------------------------------------------


def _hhi(values: List[float]) -> float:
    """Compute the Herfindahl-Hirschman Index (sum of squared market shares).

    Args:
        values: Absolute values (e.g. revenues). Must be non-negative.

    Returns:
        HHI in [0, 1] where 1 = fully concentrated and 1/n = perfectly even.
        Returns 1.0 if only one entry or all zeros.
    """
    # Use absolute values so negative-revenue companies are not silently excluded
    filtered = [abs(v) for v in values if v is not None and v != 0]
    if len(filtered) <= 1:
        return 1.0
    total = sum(filtered)
    if total < 1e-12:
        return 1.0
    shares = [v / total for v in filtered]
    return sum(s * s for s in shares)


def _hhi_label(hhi_val: float) -> str:
    """Human-readable HHI interpretation."""
    if hhi_val < 0.15:
        return "low concentration"
    if hhi_val < 0.25:
        return "moderate concentration"
    return "high concentration"


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class PortfolioAnalyzer:
    """Multi-company portfolio analysis using CharlieAnalyzer as foundation."""

    def __init__(self, analyzer: Optional[CharlieAnalyzer] = None):
        self._analyzer = analyzer or CharlieAnalyzer()

    # ------------------------------------------------------------------
    # 1. Company Snapshot
    # ------------------------------------------------------------------

    def company_snapshot(self, name: str, data: FinancialData) -> CompanySnapshot:
        """Extract a summary snapshot with health score and key ratios.

        Args:
            name: Company identifier.
            data: FinancialData instance.

        Returns:
            CompanySnapshot with health score, grade, and core ratios.
        """
        health = self._analyzer.composite_health_score(data)

        key_ratios: Dict[str, Optional[float]] = {}
        for ratio_name, num_field, den_field in _CORE_RATIOS:
            num = getattr(data, num_field, None)
            den = getattr(data, den_field, None)
            key_ratios[ratio_name] = safe_divide(num, den)

        z = self._analyzer.altman_z_score(data)

        return CompanySnapshot(
            name=name,
            health_score=health.score,
            health_grade=health.grade,
            key_ratios=key_ratios,
            z_score=z.z_score if z else None,
            z_zone=z.zone if z else "",
        )

    # ------------------------------------------------------------------
    # 2. Correlation Matrix
    # ------------------------------------------------------------------

    def correlation_matrix(self, companies: Dict[str, FinancialData]) -> CorrelationMatrix:
        """Compute pairwise correlation of core ratios across companies.

        Each company becomes a "sample" with 5 core ratio values. The
        correlation matrix shows how similarly each pair of companies scores
        on those ratios.

        Args:
            companies: Dict mapping company name -> FinancialData.

        Returns:
            CorrelationMatrix with NxN company correlation matrix.
        """
        names = list(companies.keys())
        n = len(names)

        if n < 2:
            return CorrelationMatrix(
                company_names=names,
                ratio_names=_CORE_RATIO_NAMES,
                matrix=[[1.0]] if n == 1 else [],
                avg_correlation=1.0 if n == 1 else 0.0,
                interpretation="Need at least 2 companies for correlation analysis.",
            )

        # Build ratio matrix: rows=companies, cols=ratios
        ratio_matrix = np.zeros((n, len(_CORE_RATIOS)))
        for i, name in enumerate(names):
            data = companies[name]
            for j, (_, num_field, den_field) in enumerate(_CORE_RATIOS):
                num = getattr(data, num_field, None)
                den = getattr(data, den_field, None)
                val = safe_divide(num, den)
                ratio_matrix[i, j] = val if val is not None else 0.0

        # Compute company-vs-company correlation (transpose: ratio vecs per company)
        # Use corrcoef on the rows (each row = one company's ratio profile)
        # Suppress expected RuntimeWarning when a company has constant (all-zero) ratios
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = np.corrcoef(ratio_matrix)

        # Average off-diagonal correlation, excluding NaN pairs
        if n > 1:
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            off_diag = corr[mask]
            valid = off_diag[~np.isnan(off_diag)]
            avg_corr = float(valid.mean()) if len(valid) > 0 else 0.0
        else:
            avg_corr = 0.0

        # Warn about companies with no computable ratios (all-zero vectors)
        zero_companies = [names[i] for i in range(n) if np.allclose(ratio_matrix[i], 0.0)]

        # Replace NaN with 0.0 only for the output matrix (display purposes)
        corr = np.nan_to_num(corr, nan=0.0)
        matrix_list = corr.tolist()

        if avg_corr > 0.7:
            interp = (
                f"High average correlation ({avg_corr:.2f}): portfolio companies "
                "move similarly, reducing diversification benefit."
            )
        elif avg_corr > 0.3:
            interp = f"Moderate correlation ({avg_corr:.2f}): reasonable diversification across portfolio companies."
        else:
            interp = (
                f"Low correlation ({avg_corr:.2f}): strong diversification effect -- "
                "companies have distinct financial profiles."
            )

        if zero_companies:
            interp += (
                f" NOTE: {', '.join(zero_companies)} had no computable ratios "
                "and were excluded from correlation calculation."
            )

        return CorrelationMatrix(
            company_names=names,
            ratio_names=_CORE_RATIO_NAMES,
            matrix=matrix_list,
            avg_correlation=avg_corr,
            interpretation=interp,
        )

    # ------------------------------------------------------------------
    # 3. Diversification Score
    # ------------------------------------------------------------------

    def diversification_score(
        self,
        companies: Dict[str, FinancialData],
        correlation: Optional[CorrelationMatrix] = None,
    ) -> DiversificationScore:
        """Score portfolio diversification on a 0-100 scale.

        Scoring breakdown:
          - Revenue HHI component:     40 pts (lower HHI = higher score)
          - Asset HHI component:       20 pts (lower HHI = higher score)
          - Correlation component:     40 pts (lower avg corr = higher score)

        Args:
            companies: Dict mapping company name -> FinancialData.
            correlation: Pre-computed CorrelationMatrix, or None to compute.

        Returns:
            DiversificationScore with overall 0-100 rating.
        """
        # Single or empty portfolio: no diversification possible
        if len(companies) <= 1:
            return DiversificationScore(
                overall_score=0,
                grade="F",
                hhi_revenue=1.0,
                hhi_assets=1.0,
                correlation_penalty=0.0,
                revenue_concentration="High",
                asset_concentration="High",
                interpretation=(
                    "Diversification score 0/100 (Grade F). "
                    "A portfolio with 1 or fewer companies has no diversification."
                ),
            )

        if correlation is None:
            correlation = self.correlation_matrix(companies)

        # Revenue HHI
        revenues = [getattr(d, "revenue", None) or 0.0 for d in companies.values()]
        hhi_rev = _hhi(revenues)

        # Asset HHI
        assets = [getattr(d, "total_assets", None) or 0.0 for d in companies.values()]
        hhi_ast = _hhi(assets)

        # Score components
        # Perfect diversification (HHI = 1/n) gets max score
        n = max(len(companies), 1)
        ideal_hhi = 1.0 / n

        # Revenue component (40 pts): linear scale from HHI=1 (0 pts) to HHI=ideal (40 pts)
        if hhi_rev <= ideal_hhi + 1e-9:
            rev_pts = 40
        elif hhi_rev >= 1.0:
            rev_pts = 0
        else:
            rev_pts = int(40 * (1.0 - hhi_rev) / (1.0 - ideal_hhi))

        # Asset component (20 pts)
        if hhi_ast <= ideal_hhi + 1e-9:
            ast_pts = 20
        elif hhi_ast >= 1.0:
            ast_pts = 0
        else:
            ast_pts = int(20 * (1.0 - hhi_ast) / (1.0 - ideal_hhi))

        # Correlation component (40 pts): avg_corr in [-1, 1]
        # -1 (perfect negative) -> 40, 0 -> 20, 1 (perfect positive) -> 0
        avg_corr = correlation.avg_correlation
        corr_pts = max(0, min(40, int(40 * (1.0 - avg_corr) / 2.0)))

        total = rev_pts + ast_pts + corr_pts
        total = max(0, min(100, total))

        return DiversificationScore(
            overall_score=total,
            grade=_score_to_grade(total),
            hhi_revenue=round(hhi_rev, 4),
            hhi_assets=round(hhi_ast, 4),
            correlation_penalty=round(avg_corr, 4),
            revenue_concentration=_hhi_label(hhi_rev),
            asset_concentration=_hhi_label(hhi_ast),
            interpretation=(
                f"Diversification score {total}/100 (Grade {_score_to_grade(total)}). "
                f"Revenue HHI={hhi_rev:.3f} ({_hhi_label(hhi_rev)}), "
                f"Asset HHI={hhi_ast:.3f} ({_hhi_label(hhi_ast)}), "
                f"Avg correlation={avg_corr:.2f}."
            ),
        )

    # ------------------------------------------------------------------
    # 4. Portfolio Risk Summary
    # ------------------------------------------------------------------

    def portfolio_risk_summary(
        self,
        snapshots: List[CompanySnapshot],
        companies: Dict[str, FinancialData],
    ) -> PortfolioRiskSummary:
        """Build an aggregate risk summary for the portfolio.

        Args:
            snapshots: Pre-computed CompanySnapshot list.
            companies: Dict mapping company name -> FinancialData.

        Returns:
            PortfolioRiskSummary with risk flags and distress count.
        """
        n = len(snapshots)
        if n == 0:
            return PortfolioRiskSummary(
                risk_flags=["No companies in portfolio"],
                overall_risk_level="critical",
            )

        scores = [s.health_score for s in snapshots]
        avg_score = sum(scores) / n
        min_score = min(scores)
        max_score = max(scores)

        weakest = min(snapshots, key=lambda s: s.health_score)
        strongest = max(snapshots, key=lambda s: s.health_score)

        # Distress detection: Z-score in distress zone OR health score < 30
        distress = [s for s in snapshots if s.z_zone == "distress" or s.health_score < 30]

        risk_flags: List[str] = []

        if len(distress) > 0:
            risk_flags.append(
                f"{len(distress)} company(ies) in financial distress: " + ", ".join(d.name for d in distress)
            )

        if avg_score < 40:
            risk_flags.append(f"Low average health score ({avg_score:.0f}/100) across portfolio")

        # Check for negative equity in any company
        for name, data in companies.items():
            equity = getattr(data, "total_equity", None)
            if equity is not None and equity < 0:
                risk_flags.append(f"{name}: negative equity (${equity:,.0f})")

        # Interest coverage warnings
        for s in snapshots:
            ic = s.key_ratios.get("interest_coverage")
            if ic is not None and ic < 1.5:
                risk_flags.append(f"{s.name}: weak interest coverage ({ic:.2f}x)")

        # Determine overall risk level
        if len(distress) >= max(1, n // 2) or avg_score < 30:
            risk_level = "critical"
        elif len(distress) > 0 or avg_score < 50:
            risk_level = "high"
        elif avg_score < 65:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return PortfolioRiskSummary(
            num_companies=n,
            avg_health_score=round(avg_score, 1),
            min_health_score=min_score,
            max_health_score=max_score,
            weakest_company=weakest.name,
            strongest_company=strongest.name,
            distress_count=len(distress),
            distress_companies=[d.name for d in distress],
            risk_flags=risk_flags,
            overall_risk_level=risk_level,
        )

    # ------------------------------------------------------------------
    # 5. Full Portfolio Analysis
    # ------------------------------------------------------------------

    def full_portfolio_analysis(self, companies: Dict[str, FinancialData]) -> PortfolioReport:
        """Run the complete portfolio analysis pipeline.

        Args:
            companies: Dict mapping company name -> FinancialData.

        Returns:
            PortfolioReport with snapshots, correlation, diversification,
            and risk summary.
        """
        # 1. Build snapshots
        snapshots = [self.company_snapshot(name, data) for name, data in companies.items()]

        # 2. Correlation matrix
        correlation = self.correlation_matrix(companies)

        # 3. Diversification score
        diversification = self.diversification_score(companies, correlation)

        # 4. Risk summary
        risk_summary = self.portfolio_risk_summary(snapshots, companies)

        # 5. Build summary text
        summary_parts = [
            f"Portfolio of {len(companies)} companies.",
            f"Average health score: {risk_summary.avg_health_score:.0f}/100.",
            f"Strongest: {risk_summary.strongest_company} ({risk_summary.max_health_score}/100).",
            f"Weakest: {risk_summary.weakest_company} ({risk_summary.min_health_score}/100).",
            f"Diversification: {diversification.overall_score}/100 (Grade {diversification.grade}).",
            f"Overall risk level: {risk_summary.overall_risk_level}.",
        ]
        if risk_summary.distress_count > 0:
            summary_parts.append(f"WARNING: {risk_summary.distress_count} company(ies) in financial distress.")

        return PortfolioReport(
            snapshots=snapshots,
            correlation=correlation,
            diversification=diversification,
            risk_summary=risk_summary,
            num_companies=len(companies),
            summary=" ".join(summary_parts),
        )
