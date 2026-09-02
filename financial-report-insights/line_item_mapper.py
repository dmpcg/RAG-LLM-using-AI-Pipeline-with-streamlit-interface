"""
Line Item Mapper for Financial Documents.

Maps raw financial line item labels to canonical field names.
Trained on 20+ real financial models with non-standard naming.

Uses longest-pattern-first substring matching with fuzzy fallback.
Handles row-based and column-based layouts, parenthetical qualifiers,
abbreviations, and industry-specific terminology.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MappedItem:
    """A mapped financial line item."""

    canonical_name: str
    original_label: str
    category: str
    confidence: float  # 0.0 - 1.0
    value: Any = None
    period: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# CANONICAL FIELD MAPPING
# Keys = canonical name, Values = list of patterns (longest first within each field)
# Patterns are matched case-insensitively as substrings after normalization.
# ──────────────────────────────────────────────────────────────────────

FIELD_MAPPINGS: Dict[str, Dict[str, Any]] = {
    # ── INCOME STATEMENT ─────────────────────────────────────────────
    "revenue": {
        "category": "income_statement",
        "patterns": [
            "total revenue",
            "net revenue",
            "gross revenue",
            "net sales",
            "total sales",
            "total net revenue",
            "total net sales",
            "operating revenue",
            "service revenue",
            "product revenue",
            "subscription revenue",
            "recurring revenue",
            "revenue",
            "sales",
        ],
    },
    "cogs": {
        "category": "income_statement",
        "patterns": [
            "cost of goods sold",
            "cost of revenue",
            "cost of sales",
            "cost of goods sold- direct",
            "cost of goods sold- indirect",
            "cost of products sold",
            "cost of services",
            "direct costs",
            "cogs",
        ],
    },
    "gross_profit": {
        "category": "income_statement",
        "patterns": [
            "gross profit",
            "gross margin",
            "gross income",
        ],
    },
    "sga": {
        "category": "income_statement",
        "patterns": [
            "selling, general and administrative",
            "selling general and administrative",
            "selling, general & administrative",
            "sg&a",
            "sga",
            "sg & a",
            "general and administrative",
            "general & administrative",
            "g&a",
            "g & a",
        ],
    },
    "research_development": {
        "category": "income_statement",
        "patterns": [
            "research and development",
            "research & development",
            "r&d",
            "r & d",
            "r&d expense",
        ],
    },
    "depreciation_amortization": {
        "category": "income_statement",
        "patterns": [
            "depreciation and amortization",
            "depreciation & amortization",
            "depreciation/amortization",
            "d&a",
            "d & a",
            "da",
            "depreciation",
            "amortization",
        ],
    },
    "operating_income": {
        "category": "income_statement",
        "patterns": [
            "operating income",
            "operating profit",
            "operating earnings",
            "income from operations",
            "ebit",
        ],
    },
    "ebitda": {
        "category": "income_statement",
        "patterns": [
            "ebitda",
            "adjusted ebitda",
            "normalized ebitda",
        ],
    },
    "interest_expense": {
        "category": "income_statement",
        "patterns": [
            "interest expense",
            "interest cost",
            "interest charges",
            "interest expense, net",
            "net interest expense",
            "interest income (expense)",
            "debt & interest",
        ],
    },
    "interest_income": {
        "category": "income_statement",
        "patterns": [
            "interest income",
            "interest earned",
        ],
    },
    "other_income": {
        "category": "income_statement",
        "patterns": [
            "other income",
            "other income (expense)",
            "non-operating income",
            "other income, net",
            "other (income) expense",
        ],
    },
    "income_before_tax": {
        "category": "income_statement",
        "patterns": [
            "income before income taxes",
            "income before tax",
            "earnings before tax",
            "pre-tax income",
            "pretax income",
            "income before provision for income taxes",
        ],
    },
    "tax_expense": {
        "category": "income_statement",
        "patterns": [
            "income tax expense",
            "provision for income taxes",
            "income taxes",
            "tax expense",
            "tax provision",
        ],
    },
    "net_income": {
        "category": "income_statement",
        "patterns": [
            "net income",
            "net earnings",
            "net profit",
            "net income (loss)",
            "net earnings (loss)",
            "net income attributable",
        ],
    },
    "eps_basic": {
        "category": "income_statement",
        "patterns": [
            "basic earnings per share",
            "basic eps",
            "earnings per share - basic",
            "eps basic",
        ],
    },
    "eps_diluted": {
        "category": "income_statement",
        "patterns": [
            "diluted earnings per share",
            "diluted eps",
            "earnings per share - diluted",
            "eps diluted",
        ],
    },
    # ── Payroll / Labor (Cannabis, Staffing) ────────────────────────
    "payroll": {
        "category": "income_statement",
        "patterns": [
            "payroll and related payroll taxes",
            "payroll",
            "salaries and wages",
            "compensation and benefits",
            "employee compensation",
            "wages",
            "salaries",
        ],
    },
    "contract_labor": {
        "category": "income_statement",
        "patterns": [
            "contract labor",
            "temp labor",
            "temporary labor",
            "contracted services",
            "outsourced labor",
        ],
    },
    "sales_commissions": {
        "category": "income_statement",
        "patterns": [
            "sales commissions",
            "commissions",
            "broker commissions",
            "outside broker commission",
        ],
    },
    # ── BALANCE SHEET: ASSETS ────────────────────────────────────────
    "cash": {
        "category": "balance_sheet",
        "patterns": [
            "cash and cash equivalents",
            "cash & cash equivalents",
            "cash and equivalents",
            "cash",
        ],
    },
    "accounts_receivable": {
        "category": "balance_sheet",
        "patterns": [
            "accounts receivable",
            "trade receivables",
            "net receivables",
            "a/r",
            "ar",
        ],
    },
    "inventory": {
        "category": "balance_sheet",
        "patterns": [
            "inventories",
            "inventory",
            "merchandise inventory",
            "finished goods",
            "raw materials",
        ],
    },
    "prepaid_expenses": {
        "category": "balance_sheet",
        "patterns": [
            "prepaid expenses",
            "prepaid assets",
            "prepaid",
        ],
    },
    "total_current_assets": {
        "category": "balance_sheet",
        "patterns": [
            "total current assets",
            "current assets",
        ],
    },
    "ppe_net": {
        "category": "balance_sheet",
        "patterns": [
            "property, plant and equipment, net",
            "property plant and equipment net",
            "property, plant & equipment",
            "net ppe",
            "ppe net",
            "ppe",
            "fixed assets",
            "property and equipment",
        ],
    },
    "goodwill": {
        "category": "balance_sheet",
        "patterns": [
            "goodwill",
        ],
    },
    "intangible_assets": {
        "category": "balance_sheet",
        "patterns": [
            "intangible assets",
            "intangibles",
            "other intangible assets",
        ],
    },
    "total_assets": {
        "category": "balance_sheet",
        "patterns": [
            "total assets",
        ],
    },
    # ── BALANCE SHEET: LIABILITIES ───────────────────────────────────
    "accounts_payable": {
        "category": "balance_sheet",
        "patterns": [
            "accounts payable",
            "trade payables",
            "a/p",
            "ap",
        ],
    },
    "accrued_expenses": {
        "category": "balance_sheet",
        "patterns": [
            "accrued expenses",
            "accrued liabilities",
            "accrued expenses and other",
        ],
    },
    "current_debt": {
        "category": "balance_sheet",
        "patterns": [
            "current portion of long-term debt",
            "short-term debt",
            "current debt",
            "notes payable",
            "short-term borrowings",
        ],
    },
    "total_current_liabilities": {
        "category": "balance_sheet",
        "patterns": [
            "total current liabilities",
            "current liabilities",
        ],
    },
    "long_term_debt": {
        "category": "balance_sheet",
        "patterns": [
            "long-term debt",
            "long term debt",
            "total long-term debt",
            "senior debt",
        ],
    },
    "total_liabilities": {
        "category": "balance_sheet",
        "patterns": [
            "total liabilities",
        ],
    },
    "total_equity": {
        "category": "balance_sheet",
        "patterns": [
            "total stockholders' equity",
            "total shareholders' equity",
            "total equity",
            "stockholders' equity",
            "shareholders' equity",
            "owners' equity",
            "total stockholders equity",
            "total shareholders equity",
        ],
    },
    "retained_earnings": {
        "category": "balance_sheet",
        "patterns": [
            "retained earnings",
            "accumulated deficit",
            "retained earnings (accumulated deficit)",
        ],
    },
    # ── CASH FLOW ────────────────────────────────────────────────────
    "cfo": {
        "category": "cash_flow",
        "patterns": [
            "net cash provided by operating activities",
            "cash from operations",
            "operating cash flow",
            "net cash from operating",
            "cfo",
        ],
    },
    "capex": {
        "category": "cash_flow",
        "patterns": [
            "capital expenditures",
            "purchases of property",
            "capital spending",
            "capex",
            "cap ex",
        ],
    },
    "cfi": {
        "category": "cash_flow",
        "patterns": [
            "net cash used in investing activities",
            "cash from investing",
            "investing cash flow",
            "net cash from investing",
            "cfi",
        ],
    },
    "cff": {
        "category": "cash_flow",
        "patterns": [
            "net cash provided by financing activities",
            "net cash used in financing activities",
            "cash from financing",
            "financing cash flow",
            "net cash from financing",
            "cff",
        ],
    },
    "dividends_paid": {
        "category": "cash_flow",
        "patterns": [
            "dividends paid",
            "cash dividends paid",
            "dividend payments",
            "distributions",
        ],
    },
    "share_repurchases": {
        "category": "cash_flow",
        "patterns": [
            "repurchase of common stock",
            "share repurchases",
            "treasury stock purchased",
            "stock buyback",
            "repurchases of common stock",
        ],
    },
    "fcf": {
        "category": "cash_flow",
        "patterns": [
            "free cash flow",
            "unlevered free cash flow",
            "levered free cash flow",
            "equity free cash flow",
            "unlevered fcf",
            "levered fcf",
            "equity fcf",
            "fcf",
        ],
    },
    # ── VALUATION / LBO ──────────────────────────────────────────────
    "enterprise_value": {
        "category": "valuation",
        "patterns": [
            "enterprise value",
            "ev",
            "total enterprise value",
        ],
    },
    "purchase_multiple": {
        "category": "valuation",
        "patterns": [
            "purchase multiple",
            "entry multiple",
            "acquisition multiple",
        ],
    },
    "exit_multiple": {
        "category": "valuation",
        "patterns": [
            "exit multiple",
            "terminal multiple",
        ],
    },
    "irr": {
        "category": "valuation",
        "patterns": [
            "irr",
            "internal rate of return",
            "gross irr",
            "net irr",
        ],
    },
    "moic": {
        "category": "valuation",
        "patterns": [
            "moic",
            "multiple on invested capital",
            "multiple of invested capital",
            "money multiple",
        ],
    },
    "discount_rate": {
        "category": "valuation",
        "patterns": [
            "discount rate",
            "wacc",
            "weighted average cost of capital",
            "cost of capital",
            "hurdle rate",
        ],
    },
    "terminal_growth_rate": {
        "category": "valuation",
        "patterns": [
            "terminal growth rate",
            "perpetuity growth rate",
            "long-term growth rate",
        ],
    },
    # ── DEBT / CREDIT ────────────────────────────────────────────────
    "dscr": {
        "category": "debt",
        "patterns": [
            "debt service coverage ratio",
            "dscr",
        ],
    },
    "interest_coverage": {
        "category": "debt",
        "patterns": [
            "interest coverage ratio",
            "interest coverage",
            "times interest earned",
        ],
    },
    "leverage_ratio": {
        "category": "debt",
        "patterns": [
            "leverage ratio",
            "debt/ebitda",
            "debt to ebitda",
            "net debt/ebitda",
            "total leverage",
        ],
    },
    "debt_balance": {
        "category": "debt",
        "patterns": [
            "debt balance",
            "total debt",
            "net debt",
            "principal outstanding",
            "outstanding balance",
        ],
    },
    # ── CANNABIS-SPECIFIC ────────────────────────────────────────────
    "wholesale_revenue": {
        "category": "cannabis",
        "patterns": [
            "wholesale revenue",
            "wholesale - bigs",
            "wholesale - smalls",
            "wholesale - trim",
            "wholesale revenue - bigs",
            "wholesale revenue - smalls",
        ],
    },
    "retail_revenue": {
        "category": "cannabis",
        "patterns": [
            "retail revenue",
            "dispensary revenue",
            "adult use revenue",
            "medical revenue",
        ],
    },
    "cost_per_lb": {
        "category": "cannabis",
        "patterns": [
            "cost per lb",
            "cost/lb",
            "cost per pound",
        ],
    },
    "realized_price_per_lb": {
        "category": "cannabis",
        "patterns": [
            "realized price/lb",
            "realized price per lb",
            "avg price per lb",
            "average price per pound",
        ],
    },
    "metrc_tags": {
        "category": "cannabis",
        "patterns": [
            "metrc tags",
            "metrc",
        ],
    },
    "tax_280e_rate": {
        "category": "cannabis",
        "patterns": [
            "280e tax rate",
            "280e rate",
            "normalized tax rate",
            "effective 280e",
        ],
    },
    "seeds_clones": {
        "category": "cannabis",
        "patterns": [
            "seeds/clones",
            "seeds and clones",
            "clones",
            "seeds",
            "propagation",
        ],
    },
    "nutrients": {
        "category": "cannabis",
        "patterns": [
            "nutrients",
            "nutrient cost",
        ],
    },
    "ipm": {
        "category": "cannabis",
        "patterns": [
            "ipm",
            "integrated pest management",
            "pest management",
        ],
    },
    # ── CASH FORECAST ────────────────────────────────────────────────
    "beginning_cash": {
        "category": "cash_forecast",
        "patterns": [
            "beginning cash",
            "beg cash",
            "opening cash",
            "beginning cash balance",
        ],
    },
    "ending_cash": {
        "category": "cash_forecast",
        "patterns": [
            "ending cash",
            "closing cash",
            "ending cash balance",
            "weekly cf fcst - ending",
        ],
    },
    "ar_collections": {
        "category": "cash_forecast",
        "patterns": [
            "ar collections",
            "a/r collections",
            "accounts receivable collections",
            "forecasted ar",
        ],
    },
    "total_cash_inflow": {
        "category": "cash_forecast",
        "patterns": [
            "total cash inflow",
            "total cash revenues",
            "total cash receipts",
            "total inflows",
        ],
    },
    "total_cash_outflow": {
        "category": "cash_forecast",
        "patterns": [
            "total cash outflow",
            "total cash disbursements",
            "total outflows",
        ],
    },
    # ── KPI / OPERATING METRICS ──────────────────────────────────────
    "revenue_per_sqft": {
        "category": "kpi",
        "patterns": [
            "revenue per sqft",
            "revenue/sqft",
            "revenue per square foot",
            "sales per sqft",
        ],
    },
    "daily_transaction_count": {
        "category": "kpi",
        "patterns": [
            "daily transaction count",
            "daily transactions",
            "transaction count",
        ],
    },
    "avg_transaction_value": {
        "category": "kpi",
        "patterns": [
            "average transaction value",
            "avg transaction value",
            "average ticket",
            "avg ticket size",
        ],
    },
    "inventory_days": {
        "category": "kpi",
        "patterns": [
            "inventory days on hand",
            "days inventory",
            "days inventory outstanding",
            "dio",
        ],
    },
    "cash_runway": {
        "category": "kpi",
        "patterns": [
            "cash runway",
            "runway",
            "months of runway",
        ],
    },
    "capacity_utilization": {
        "category": "kpi",
        "patterns": [
            "capacity utilization",
            "utilization rate",
            "capacity %",
        ],
    },
    # ── STARTUP / SAAS ───────────────────────────────────────────────
    "mrr": {
        "category": "saas",
        "patterns": [
            "monthly recurring revenue",
            "mrr",
        ],
    },
    "arr": {
        "category": "saas",
        "patterns": [
            "annual recurring revenue",
            "arr",
        ],
    },
    "cac": {
        "category": "saas",
        "patterns": [
            "customer acquisition cost",
            "cac",
        ],
    },
    "ltv": {
        "category": "saas",
        "patterns": [
            "lifetime value",
            "customer lifetime value",
            "ltv",
            "clv",
        ],
    },
    "churn_rate": {
        "category": "saas",
        "patterns": [
            "churn rate",
            "monthly churn",
            "annual churn",
            "customer churn",
            "revenue churn",
        ],
    },
    "burn_rate": {
        "category": "saas",
        "patterns": [
            "burn rate",
            "monthly burn",
            "net burn",
            "cash burn",
        ],
    },
    # ── CONSTRUCTION / RE ────────────────────────────────────────────
    "original_budget": {
        "category": "construction",
        "patterns": [
            "original budget",
        ],
    },
    "current_budget": {
        "category": "construction",
        "patterns": [
            "current budget",
            "revised budget",
        ],
    },
    "in_process": {
        "category": "construction",
        "patterns": [
            "in process",
            "work in progress",
            "wip",
        ],
    },
    "complete_pct": {
        "category": "construction",
        "patterns": [
            "complete %",
            "% complete",
            "completion %",
            "percent complete",
        ],
    },
    # ── FUND REPORTING ───────────────────────────────────────────────
    "beginning_capital": {
        "category": "fund",
        "patterns": [
            "beginning capital balance",
            "beginning capital",
            "opening balance",
        ],
    },
    "ending_capital": {
        "category": "fund",
        "patterns": [
            "ending capital balance",
            "ending capital",
            "closing balance",
        ],
    },
    "contributions": {
        "category": "fund",
        "patterns": [
            "contributions",
            "capital contributions",
            "capital calls",
        ],
    },
    "distributions_fund": {
        "category": "fund",
        "patterns": [
            "distributions",
            "capital distributions",
            "return of capital",
        ],
    },
}


def _normalize_label(label: str) -> str:
    """Normalize a label for matching: lowercase, collapse whitespace, strip parens."""
    if not isinstance(label, str):
        return ""
    label = label.lower().strip()
    # Remove parenthetical qualifiers like (formula), (see below), etc.
    # Greedily remove outermost parenthetical groups (handles nested parens)
    while "(" in label and ")" in label:
        new_label = re.sub(r"\([^()]*\)", "", label)
        if new_label == label:
            break
        label = new_label
    # Collapse multiple whitespace
    label = re.sub(r"\s+", " ", label).strip()
    return label


# Pre-compute sorted patterns (longest first) for each canonical field
_SORTED_MAPPINGS: List[Tuple[str, str, str]] = []  # (pattern, canonical_name, category)


def _build_sorted_mappings() -> None:
    """Build the sorted pattern list (longest pattern first)."""
    global _SORTED_MAPPINGS
    if _SORTED_MAPPINGS:
        return
    all_patterns = []
    for canonical_name, field_info in FIELD_MAPPINGS.items():
        category = field_info["category"]
        for pattern in field_info["patterns"]:
            all_patterns.append((pattern.lower(), canonical_name, category))
    # Sort by pattern length descending (longest first to avoid greedy matches)
    all_patterns.sort(key=lambda x: len(x[0]), reverse=True)
    _SORTED_MAPPINGS = all_patterns


def map_label(label: str, section_context: Optional[str] = None) -> Optional[MappedItem]:
    """Map a raw financial label to its canonical name.

    Args:
        label: The raw label text from the document.
        section_context: Optional section type hint for disambiguation.

    Returns:
        MappedItem if matched, None otherwise.
    """
    _build_sorted_mappings()
    normalized = _normalize_label(label)
    if not normalized:
        return None

    for pattern, canonical_name, category in _SORTED_MAPPINGS:
        if pattern in normalized:
            # Boost confidence if section context matches
            confidence = 0.9
            if section_context and category == section_context:
                confidence = 0.95
            return MappedItem(
                canonical_name=canonical_name,
                original_label=label,
                category=category,
                confidence=confidence,
            )

    return None


def map_labels_batch(
    labels: List[str],
    section_context: Optional[str] = None,
) -> Dict[str, MappedItem]:
    """Map a batch of labels, returning only successful matches.

    Args:
        labels: List of raw label strings.
        section_context: Optional section type for disambiguation.

    Returns:
        Dict mapping original label -> MappedItem.
    """
    results: Dict[str, MappedItem] = {}
    seen_canonical: Set[str] = set()

    for label in labels:
        mapped = map_label(label, section_context)
        if mapped and mapped.canonical_name not in seen_canonical:
            results[label] = mapped
            seen_canonical.add(mapped.canonical_name)

    return results


def detect_periods(columns: List[str]) -> List[str]:
    """Detect time period columns from column headers.

    Handles formats like:
    - "2024", "2025E", "2024A"
    - "Q1 2024", "Q2 2025"
    - "Jan-24", "Feb-24"
    - "YTD Feb", "MTD"
    - "Week 1", "Week 2"
    - "Month to Date", "Year to Date"
    """
    period_patterns = [
        re.compile(r"^(?:FY\s*)?(\d{4})\s*[AaEe]?$"),  # 2024, 2024A, 2025E, FY2024
        re.compile(r"^Q[1-4]\s+\d{4}$", re.IGNORECASE),  # Q1 2024
        re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.IGNORECASE),  # Jan-24
        re.compile(r"^(?:YTD|MTD|QTD)\b", re.IGNORECASE),  # YTD Feb
        re.compile(r"^Week\s+\d+", re.IGNORECASE),  # Week 1
        re.compile(r"^(?:Month|Year|Inception)\s+to\s+Date$", re.IGNORECASE),  # Month to Date
        re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),  # 1/1/2024
        re.compile(r"^(?:Actual|Forecast|Budget|Plan)\s*$", re.IGNORECASE),
    ]

    periods = []
    for col in columns:
        col_str = str(col).strip()
        for pat in period_patterns:
            if pat.match(col_str):
                periods.append(col_str)
                break

    return periods
