"""Shared helpers for financial analysis modules.

Contains key-name classification utilities, ratio category grouping,
and common scoring functions used across multiple modules.
"""

from typing import Dict, List, Tuple

from structured_types import RatioCategory


# ---------------------------------------------------------------------------
# Common scoring helpers
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS_100: List[Tuple[int, str]] = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
]


def score_to_grade(score: int) -> str:
    """Map a 0-100 integer score to a letter grade (A/B/C/D/F)."""
    if not isinstance(score, (int, float)):
        raise TypeError(f"score_to_grade expects int/float, got {type(score).__name__}")
    score = max(0, min(100, int(score)))
    for threshold, grade in _GRADE_THRESHOLDS_100:
        if score >= threshold:
            return grade
    return "F"


# ---------------------------------------------------------------------------
# Key-name classification helpers
# ---------------------------------------------------------------------------

_PERCENT_KEYWORDS = (
    "margin", "return", "yield", "roe", "roa", "roic",
    "turnover", "coverage", "rate",
)
# Multiplier-style metrics rendered as "X.XXx" (e.g. current_ratio, debt_to_equity).
# These are decimal multiples, NOT percentages -- 1.5 must render as "1.50x" not "150%".
_RATIO_KEYWORDS = (
    "ratio", "to_equity", "to_assets", "multiplier",
)
_DOLLAR_KEYWORDS = (
    "revenue", "income", "assets", "debt", "equity", "cash",
    "expense", "liabilities", "ebit", "ebitda", "capex",
    "profit", "payable", "receivable", "inventory",
)


def _is_percent_key(key: str) -> bool:
    """Return True if *key* looks like a percentage metric."""
    lower = key.lower()
    return any(kw in lower for kw in _PERCENT_KEYWORDS)


def _is_ratio_key(key: str) -> bool:
    """Return True if *key* looks like a multiplier-style ratio (rendered as 'X.XXx').

    Distinguishes true ratios (e.g. current_ratio = 1.5 -> "1.50x") from
    percentage metrics (e.g. gross_margin = 0.45 -> "45.00%").
    """
    lower = key.lower()
    return any(kw in lower for kw in _RATIO_KEYWORDS)


def _is_dollar_key(key: str) -> bool:
    """Return True if *key* looks like a dollar-denominated metric."""
    lower = key.lower()
    return any(kw in lower for kw in _DOLLAR_KEYWORDS)


# ---------------------------------------------------------------------------
# Ratio category grouping
# ---------------------------------------------------------------------------

_CATEGORY_MAP: Dict[str, RatioCategory] = {
    # Liquidity
    "current_ratio": RatioCategory.LIQUIDITY,
    "quick_ratio": RatioCategory.LIQUIDITY,
    "cash_ratio": RatioCategory.LIQUIDITY,
    "working_capital": RatioCategory.LIQUIDITY,
    # Profitability
    "gross_margin": RatioCategory.PROFITABILITY,
    "operating_margin": RatioCategory.PROFITABILITY,
    "net_margin": RatioCategory.PROFITABILITY,
    "roe": RatioCategory.PROFITABILITY,
    "roa": RatioCategory.PROFITABILITY,
    "roic": RatioCategory.PROFITABILITY,
    # Leverage
    "debt_to_equity": RatioCategory.LEVERAGE,
    "debt_to_assets": RatioCategory.LEVERAGE,
    "debt_ratio": RatioCategory.LEVERAGE,
    "equity_multiplier": RatioCategory.LEVERAGE,
    "interest_coverage": RatioCategory.LEVERAGE,
    # Efficiency
    "asset_turnover": RatioCategory.EFFICIENCY,
    "inventory_turnover": RatioCategory.EFFICIENCY,
    "receivables_turnover": RatioCategory.EFFICIENCY,
    "payables_turnover": RatioCategory.EFFICIENCY,
}


def _categorize(key: str) -> str:
    """Return the category for a ratio key, falling back to 'Other'."""
    return _CATEGORY_MAP.get(key, RatioCategory.OTHER).value
