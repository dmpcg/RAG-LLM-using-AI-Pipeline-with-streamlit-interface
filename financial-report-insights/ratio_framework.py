"""
Generic parameterized ratio analysis framework.

Replaces 300+ repetitive ratio methods with a declarative configuration-driven approach.
Each ratio is defined once with scoring thresholds and adjustments, then computed generically.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from financial_analyzer import FinancialData, safe_divide


class Operator(Enum):
    """Comparison operators for ratio adjustments."""

    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="


@dataclass
class Adjustment:
    """Defines a scoring adjustment based on a secondary ratio."""

    field: str  # Field name to check (e.g., "current_ratio")
    operator: Operator
    threshold: float
    delta: float  # Score adjustment amount (can be negative)
    description: str = ""


@dataclass
class RatioDefinition:
    """Declarative definition of a financial ratio with scoring rules."""

    name: str
    description: str
    numerator_field: str  # Field name on FinancialData
    denominator_field: str
    higher_is_better: bool = True
    scoring_thresholds: List[Tuple[float, float]] = field(default_factory=list)  # [(threshold, score), ...] descending
    adjustments: List[Adjustment] = field(default_factory=list)
    # One-line provenance/rationale for the scoring_thresholds cut-offs above.
    # Documentation only -- does NOT affect computed scores. Lets a reviewer
    # see why a threshold band was chosen rather than treating it as a magic
    # number. Empty string means "undocumented" (acceptable for ad-hoc/test
    # definitions; catalog entries should populate it).
    threshold_source: str = ""
    grade_map: Dict[Tuple[float, float], str] = field(
        default_factory=lambda: {
            (8.0, 10.0): "Excellent",
            (6.0, 8.0): "Good",
            (4.0, 6.0): "Adequate",
            (0.0, 4.0): "Weak",
        }
    )
    unit: str = ""  # e.g., "%", "x", "days"

    def get_grade(self, score: float) -> str:
        """Map a score to a grade label."""
        for (low, high), grade in self.grade_map.items():
            if low <= score < high:
                return grade
        # Handle edge case for perfect score
        if score >= 10.0:
            return "Excellent"
        return "Weak"


@dataclass
class RatioResult:
    """Result of computing a single ratio."""

    name: str
    value: Optional[float]
    score: float
    grade: str
    summary: str
    secondary_ratios: Dict[str, Optional[float]] = field(default_factory=dict)


def _get_field_value(data: FinancialData, field_name: str) -> Optional[float]:
    """Safely extract a field value from FinancialData."""
    return getattr(data, field_name, None)


def _apply_operator(value: float, operator: Operator, threshold: float) -> bool:
    """Apply comparison operator."""
    if operator == Operator.GT:
        return value > threshold
    elif operator == Operator.GTE:
        return value >= threshold
    elif operator == Operator.LT:
        return value < threshold
    elif operator == Operator.LTE:
        return value <= threshold
    elif operator == Operator.EQ:
        return abs(value - threshold) < 1e-9
    return False


def _compute_base_score(
    ratio_value: float, scoring_thresholds: List[Tuple[float, float]], higher_is_better: bool
) -> float:
    """
    Compute base score from ratio value and thresholds.

    Args:
        ratio_value: The computed ratio value
        scoring_thresholds: List of (threshold, score) tuples in DESCENDING order
        higher_is_better: If False, invert the comparison logic

    Returns:
        Base score (0-10 range before adjustments)
    """
    if not scoring_thresholds:
        return 5.0  # Default mid-range score

    # Thresholds must be pre-sorted descending by threshold value
    sorted_thresholds = scoring_thresholds

    if higher_is_better:
        # Standard logic: higher ratio = higher score
        for threshold, score in sorted_thresholds:
            if ratio_value >= threshold:
                return score
        # Below all thresholds
        return sorted_thresholds[-1][1] - 1.0
    else:
        # Inverted logic: lower ratio = higher score
        for threshold, score in sorted_thresholds:
            if ratio_value <= threshold:
                return score
        # Above all thresholds
        return sorted_thresholds[-1][1] - 1.0


def _apply_adjustments(
    base_score: float, adjustments: List[Adjustment], data: FinancialData, computed_ratios: Dict[str, Optional[float]]
) -> float:
    """
    Apply scoring adjustments based on secondary ratios.

    Args:
        base_score: Initial score before adjustments
        adjustments: List of Adjustment rules
        data: FinancialData instance
        computed_ratios: Dict of already-computed secondary ratios

    Returns:
        Adjusted score (clamped to 0-10)
    """
    score = base_score

    for adj in adjustments:
        # Try to get value from computed_ratios first, then from data fields
        value = computed_ratios.get(adj.field)
        if value is None:
            value = _get_field_value(data, adj.field)

        if value is not None and _apply_operator(value, adj.operator, adj.threshold):
            score += adj.delta

    # Clamp to valid range
    return max(0.0, min(10.0, score))


def _build_summary(name: str, value: Optional[float], score: float, grade: str, unit: str, description: str) -> str:
    """Build a human-readable summary string."""
    if value is None:
        return f"{name}: Insufficient data for calculation"

    value_str = f"{value:.2f}{unit}" if unit else f"{value:.2f}"
    return f"{name}: {value_str} | Score: {score:.1f}/10 ({grade}) | {description}"


def compute_ratio(
    data: FinancialData, definition: RatioDefinition, computed_ratios: Optional[Dict[str, Optional[float]]] = None
) -> RatioResult:
    """
    Generic ratio computation engine.

    Implements the complete scoring pipeline:
    1. Extract numerator/denominator from FinancialData
    2. Compute ratio using safe_divide
    3. Apply scoring thresholds
    4. Apply adjustments based on secondary ratios
    5. Assign grade
    6. Build summary

    Args:
        data: FinancialData instance
        definition: RatioDefinition configuration
        computed_ratios: Optional dict of pre-computed ratios for adjustments

    Returns:
        RatioResult with value, score, grade, and summary
    """
    computed_ratios = computed_ratios or {}

    # Step 1: Extract fields
    numerator = _get_field_value(data, definition.numerator_field)
    denominator = _get_field_value(data, definition.denominator_field)

    # Step 2: Compute ratio
    ratio_value = safe_divide(numerator, denominator)

    if ratio_value is None:
        return RatioResult(
            name=definition.name,
            value=None,
            score=0.0,
            grade="Insufficient Data",
            summary=f"{definition.name}: Insufficient data",
            secondary_ratios={},
        )

    # Step 3: Base score from thresholds
    base_score = _compute_base_score(ratio_value, definition.scoring_thresholds, definition.higher_is_better)

    # Step 4: Apply adjustments
    final_score = _apply_adjustments(base_score, definition.adjustments, data, computed_ratios)

    # Step 5: Assign grade
    grade = definition.get_grade(final_score)

    # Step 6: Build summary
    summary = _build_summary(definition.name, ratio_value, final_score, grade, definition.unit, definition.description)

    # Track secondary ratios used in adjustments
    secondary = {
        adj.field: computed_ratios.get(adj.field) or _get_field_value(data, adj.field) for adj in definition.adjustments
    }

    return RatioResult(
        name=definition.name,
        value=ratio_value,
        score=final_score,
        grade=grade,
        summary=summary,
        secondary_ratios=secondary,
    )


# ============================================================================
# RATIO CATALOG: Canonical Financial Ratios
# ============================================================================

RATIO_CATALOG: Dict[str, RatioDefinition] = {
    # --- PROFITABILITY RATIOS ---
    "roa": RatioDefinition(
        name="Return on Assets (ROA)",
        description="Measures how efficiently assets generate profit",
        threshold_source=(
            "Bands reflect common cross-industry ROA conventions: >15% top-tier, "
            "10-15% strong, 5-10% average, 2-5% weak (e.g. CFA curriculum / "
            "Damodaran sector medians)."
        ),
        numerator_field="net_income",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (0.15, 10.0),  # Excellent: >15%
            (0.10, 8.0),  # Good: 10-15%
            (0.05, 6.0),  # Adequate: 5-10%
            (0.02, 4.0),  # Weak: 2-5%
        ],
        adjustments=[
            Adjustment("operating_income", Operator.GT, 0, 0.5, "Positive operating income"),
            # NOTE: Removed total_debt < 0.5 adjustment -- total_debt is a dollar
            # amount, not a ratio, so comparing against 0.5 is always false for
            # real companies (dead code).
        ],
        unit="%",
    ),
    "roe": RatioDefinition(
        name="Return on Equity (ROE)",
        description="Measures return generated on shareholders' equity",
        threshold_source=(
            "ROE bands track equity-return norms: >20% excellent, 15-20% strong, "
            "10-15% adequate (near typical cost of equity), 5-10% weak. Leverage "
            "adjustment flags debt-inflated ROE."
        ),
        numerator_field="net_income",
        denominator_field="total_equity",
        higher_is_better=True,
        scoring_thresholds=[
            (0.20, 10.0),  # Excellent: >20%
            (0.15, 8.0),  # Good: 15-20%
            (0.10, 6.0),  # Adequate: 10-15%
            (0.05, 4.0),  # Weak: 5-10%
        ],
        adjustments=[
            Adjustment("total_debt", Operator.GT, 2.0, -0.5, "High leverage inflates ROE"),
        ],
        unit="%",
    ),
    # NOTE: True ROIC = EBIT / Invested Capital (equity + debt - cash).
    # The framework only has total_assets as a denominator field, so this is
    # EBIT/Total Assets, not ROIC.  Renamed to avoid misrepresentation.
    "ebit_to_total_assets": RatioDefinition(
        name="EBIT / Total Assets",
        description="Measures operating return on total assets (not true ROIC)",
        threshold_source=(
            "Mirrors the ROA bands (operating-return basis): >15% excellent down "
            "to 5% weak. Used as an operating-efficiency proxy since invested "
            "capital is unavailable in this data model."
        ),
        numerator_field="ebit",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (0.15, 10.0),
            (0.10, 8.0),
            (0.07, 6.0),
            (0.05, 4.0),
        ],
        unit="%",
    ),
    "gross_margin": RatioDefinition(
        name="Gross Profit Margin",
        description="Measures profitability after direct costs",
        threshold_source=(
            "Generalized gross-margin tiers spanning capital-light software (>50%) "
            "down to thin-margin distribution/retail (15-25%); midpoints chosen to "
            "separate healthy from commoditized businesses."
        ),
        numerator_field="gross_profit",
        denominator_field="revenue",
        higher_is_better=True,
        scoring_thresholds=[
            (0.50, 10.0),  # >50%
            (0.35, 8.0),  # 35-50%
            (0.25, 6.0),  # 25-35%
            (0.15, 4.0),  # 15-25%
        ],
        unit="%",
    ),
    "operating_margin": RatioDefinition(
        name="Operating Profit Margin",
        description="Measures profitability from operations",
        threshold_source=(
            "Operating-margin bands: >20% excellent, 15-20% strong, 10-15% "
            "adequate, 5-10% weak -- typical S&P operating-margin distribution "
            "where double-digit margins indicate durable operations."
        ),
        numerator_field="operating_income",
        denominator_field="revenue",
        higher_is_better=True,
        scoring_thresholds=[
            (0.20, 10.0),  # >20%
            (0.15, 8.0),
            (0.10, 6.0),
            (0.05, 4.0),
        ],
        unit="%",
    ),
    "net_margin": RatioDefinition(
        name="Net Profit Margin",
        description="Measures bottom-line profitability",
        threshold_source=(
            "Net-margin tiers: >15% excellent, 10-15% strong, 5-10% adequate, "
            "2-5% weak -- consistent with broad-market net-margin norms after tax "
            "and interest."
        ),
        numerator_field="net_income",
        denominator_field="revenue",
        higher_is_better=True,
        scoring_thresholds=[
            (0.15, 10.0),  # >15%
            (0.10, 8.0),
            (0.05, 6.0),
            (0.02, 4.0),
        ],
        unit="%",
    ),
    # --- LIQUIDITY RATIOS ---
    "current_ratio": RatioDefinition(
        name="Current Ratio",
        description="Measures ability to meet short-term obligations",
        threshold_source=(
            "Textbook liquidity rule of thumb: ~2.0 healthy, 1.5-2.0 good, "
            "1.0-1.5 adequate, <1.0 indicates a working-capital shortfall "
            "(standard analyst convention)."
        ),
        numerator_field="current_assets",
        denominator_field="current_liabilities",
        higher_is_better=True,
        scoring_thresholds=[
            (2.0, 10.0),  # Excellent: >2.0
            (1.5, 8.0),  # Good: 1.5-2.0
            (1.0, 6.0),  # Adequate: 1.0-1.5
            (0.75, 4.0),  # Weak: 0.75-1.0
        ],
        adjustments=[
            Adjustment("cash", Operator.GT, 0.3, 0.5, "Strong cash position"),
        ],
        unit="x",
    ),
    # NOTE: quick_ratio removed from catalog -- the simple A/B framework cannot
    # express (Current Assets - Inventory) / Current Liabilities.  The correct
    # quick ratio is computed in CharlieAnalyzer.calculate_liquidity_ratios().
    "cash_ratio": RatioDefinition(
        name="Cash Ratio",
        description="Most conservative liquidity measure",
        threshold_source=(
            "Most conservative liquidity test (cash only). Bands set stricter than "
            "current ratio: >=0.75 excellent, 0.30-0.50 adequate; values <0.15 "
            "signal heavy reliance on receivables/inventory to meet obligations."
        ),
        numerator_field="cash",
        denominator_field="current_liabilities",
        higher_is_better=True,
        scoring_thresholds=[
            (0.75, 10.0),
            (0.50, 8.0),
            (0.30, 6.0),
            (0.15, 4.0),
        ],
        unit="x",
    ),
    # --- LEVERAGE RATIOS ---
    "debt_to_equity": RatioDefinition(
        name="Debt-to-Equity Ratio",
        description="Measures financial leverage",
        threshold_source=(
            "Lower-is-better leverage bands: <0.3 excellent, 0.3-0.5 good, "
            "0.5-1.0 adequate, 1.0-2.0 weak -- aligns with conservative "
            "investment-grade gearing expectations for non-financial firms."
        ),
        numerator_field="total_debt",
        denominator_field="total_equity",
        higher_is_better=False,  # Lower is better
        scoring_thresholds=[
            (0.3, 10.0),  # <0.3 Excellent
            (0.5, 8.0),  # 0.3-0.5 Good
            (1.0, 6.0),  # 0.5-1.0 Adequate
            (2.0, 4.0),  # 1.0-2.0 Weak
        ],
        adjustments=[
            Adjustment("ebitda", Operator.GT, 0, 0.5, "Positive cash generation"),
        ],
        unit="x",
    ),
    "debt_to_ebitda": RatioDefinition(
        name="Debt-to-EBITDA Ratio",
        description="Measures debt coverage by earnings",
        threshold_source=(
            "Leverage-multiple bands common in credit/LBO underwriting: <2x "
            "excellent, 2-3x good, 3-4x adequate, 4-5x weak; >5x is typically "
            "covenant-stressed (lower is better)."
        ),
        numerator_field="total_debt",
        denominator_field="ebitda",
        higher_is_better=False,
        scoring_thresholds=[
            (2.0, 10.0),  # <2x Excellent
            (3.0, 8.0),  # 2-3x Good
            (4.0, 6.0),  # 3-4x Adequate
            (5.0, 4.0),  # 4-5x Weak
        ],
        unit="x",
    ),
    "interest_coverage": RatioDefinition(
        name="Interest Coverage Ratio",
        description="Measures ability to pay interest",
        threshold_source=(
            "Times-interest-earned bands: >8x excellent, 5-8x good, 2.5-5x "
            "adequate, 1.5-2.5x weak; <1.5x is a distress signal -- consistent "
            "with rating-agency coverage guidelines."
        ),
        numerator_field="ebit",
        denominator_field="interest_expense",
        higher_is_better=True,
        scoring_thresholds=[
            (8.0, 10.0),  # >8x Excellent
            (5.0, 8.0),  # 5-8x Good
            (2.5, 6.0),  # 2.5-5x Adequate
            (1.5, 4.0),  # 1.5-2.5x Weak
        ],
        unit="x",
    ),
    # --- EFFICIENCY RATIOS ---
    "asset_turnover": RatioDefinition(
        name="Asset Turnover Ratio",
        description="Measures efficiency of asset utilization",
        threshold_source=(
            "Generalized turnover bands (>2x excellent ... 0.5x weak). Note "
            "turnover is strongly industry-dependent (retail high, capital-heavy "
            "low); these are broad defaults, not sector-tuned."
        ),
        numerator_field="revenue",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (2.0, 10.0),  # >2x Excellent
            (1.5, 8.0),
            (1.0, 6.0),
            (0.5, 4.0),
        ],
        unit="x",
    ),
    "inventory_turnover": RatioDefinition(
        name="Inventory Turnover",
        description="Measures how quickly inventory is sold",
        threshold_source=(
            "COGS/inventory turns: >=12x (roughly monthly) excellent, down to 4x "
            "weak. Heuristic defaults; optimal turns vary widely by industry "
            "(grocery vs. heavy equipment)."
        ),
        numerator_field="cogs",
        denominator_field="inventory",
        higher_is_better=True,
        scoring_thresholds=[
            (12.0, 10.0),  # >12x (monthly turnover)
            (8.0, 8.0),
            (6.0, 6.0),
            (4.0, 4.0),
        ],
        unit="x",
    ),
    "receivables_turnover": RatioDefinition(
        name="Receivables Turnover",
        description="Measures collection efficiency",
        threshold_source=(
            "Revenue/AR turns proxy collection speed: >=12x (~30-day DSO) "
            "excellent, 6x (~60-day DSO) weak. Bands map turns to implied "
            "days-sales-outstanding (heuristic)."
        ),
        numerator_field="revenue",
        denominator_field="accounts_receivable",
        higher_is_better=True,
        scoring_thresholds=[
            (12.0, 10.0),  # >12x
            (10.0, 8.0),
            (8.0, 6.0),
            (6.0, 4.0),
        ],
        unit="x",
    ),
    # --- CASH FLOW RATIOS ---
    "fcf_yield": RatioDefinition(
        name="Free Cash Flow Yield",
        description="FCF as % of enterprise value (simplified using assets)",
        threshold_source=(
            "OCF/assets yield bands: >10% excellent down to 3% weak. Simplified "
            "proxy (assets stand in for enterprise value); cut-offs are heuristic "
            "yield tiers, not market-derived."
        ),
        numerator_field="operating_cash_flow",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (0.10, 10.0),  # >10%
            (0.07, 8.0),
            (0.05, 6.0),
            (0.03, 4.0),
        ],
        unit="%",
    ),
    "ocf_to_ni": RatioDefinition(
        name="Operating Cash Flow to Net Income",
        description="Measures earnings quality",
        threshold_source=(
            "Earnings-quality (accruals) bands: OCF/NI >=1.2 excellent, ~1.0 "
            "good, <1.0 raises accrual concerns. Anchored on the principle that "
            "high-quality earnings convert to cash (>=1x)."
        ),
        numerator_field="operating_cash_flow",
        denominator_field="net_income",
        higher_is_better=True,
        scoring_thresholds=[
            (1.2, 10.0),  # OCF > 120% of NI (excellent quality)
            (1.0, 8.0),  # OCF = NI
            (0.8, 6.0),  # OCF < NI (some concerns)
            (0.6, 4.0),
        ],
        unit="x",
    ),
    "cash_conversion_cycle": RatioDefinition(
        name="Cash Conversion Cycle",
        description="Days to convert operations to cash (simplified)",
        threshold_source=(
            "Simplified CCC proxy (inventory/revenue, lower is better): <5% of "
            "revenue excellent, >20% weak. Heuristic proxy -- the full DSO+DIO-DPO "
            "CCC is computed in CharlieAnalyzer, not here."
        ),
        numerator_field="inventory",
        denominator_field="revenue",  # Simplified CCC proxy
        higher_is_better=False,  # Lower is better
        scoring_thresholds=[
            (0.05, 10.0),  # <5% of revenue
            (0.10, 8.0),
            (0.15, 6.0),
            (0.20, 4.0),
        ],
        unit="days",
    ),
}


def run_all_ratios(data: FinancialData) -> Dict[str, RatioResult]:
    """
    Compute all ratios in the catalog.

    Handles dependencies by computing in multiple passes:
    1. First pass: compute all ratios
    2. Adjustments can reference previously computed ratios

    Args:
        data: FinancialData instance

    Returns:
        Dict mapping ratio name to RatioResult
    """
    results: Dict[str, RatioResult] = {}
    computed_values: Dict[str, Optional[float]] = {}

    # Single pass: adjustments reference FinancialData fields (not other catalog
    # ratios), so one iteration with the shared computed_values dict is sufficient.
    for ratio_key, definition in RATIO_CATALOG.items():
        result = compute_ratio(data, definition, computed_values)
        results[ratio_key] = result
        computed_values[ratio_key] = result.value

    return results


def get_ratio_by_category() -> Dict[str, List[str]]:
    """
    Group ratios by category for organized display.

    Returns:
        Dict mapping category name to list of ratio keys
    """
    categories = {
        "Profitability": ["roa", "roe", "ebit_to_total_assets", "gross_margin", "operating_margin", "net_margin"],
        "Liquidity": ["current_ratio", "cash_ratio"],
        "Leverage": ["debt_to_equity", "debt_to_ebitda", "interest_coverage"],
        "Efficiency": ["asset_turnover", "inventory_turnover", "receivables_turnover"],
        "Cash Flow": ["fcf_yield", "ocf_to_ni", "cash_conversion_cycle"],
    }
    return categories


def compute_category(data: FinancialData, category: str) -> Dict[str, RatioResult]:
    """
    Compute all ratios in a specific category.

    Args:
        data: FinancialData instance
        category: Category name (e.g., "Profitability", "Liquidity")

    Returns:
        Dict mapping ratio name to RatioResult for the category
    """
    categories = get_ratio_by_category()
    ratio_keys = categories.get(category, [])

    computed_values: Dict[str, Optional[float]] = {}
    results: Dict[str, RatioResult] = {}

    for ratio_key in ratio_keys:
        definition = RATIO_CATALOG.get(ratio_key)
        if definition:
            result = compute_ratio(data, definition, computed_values)
            results[ratio_key] = result
            computed_values[ratio_key] = result.value

    return results
