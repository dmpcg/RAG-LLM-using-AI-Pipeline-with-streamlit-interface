"""Standalone tool functions for specialized financial agents.

These are pure computation functions that perform actual financial calculations.
They do not depend on any LLM and return formatted string results suitable for
agent consumption.  Each function handles missing/invalid inputs gracefully and
never raises on bad data.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_div(num: Optional[float], den: Optional[float], default: float = 0.0) -> float:
    """Divide two numbers, returning *default* on zero/None denominator."""
    try:
        n = float(num) if num is not None else None
        d = float(den) if den is not None else None
    except (TypeError, ValueError):
        return default
    if n is None or d is None or abs(d) < 1e-12 or not math.isfinite(d) or not math.isfinite(n):
        return default
    result = n / d
    return result if math.isfinite(result) else default


def _fmt(value: float, decimals: int = 4) -> str:
    """Format a float to a fixed number of decimal places."""
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Ratio definitions
# ---------------------------------------------------------------------------

_RATIO_DEFINITIONS: dict[str, dict[str, Any]] = {
    "current_ratio": {
        "formula": "current_assets / current_liabilities",
        "numerator": "current_assets",
        "denominator": "current_liabilities",
        "description": "Measures short-term liquidity; ability to pay current obligations.",
        "good_range": "1.5 – 3.0",
        "low_concern": "Below 1.0 signals potential liquidity problems.",
        "high_concern": "Above 3.0 may indicate inefficient asset use.",
    },
    "quick_ratio": {
        "formula": "(current_assets - inventory) / current_liabilities",
        "numerator": None,  # computed specially
        "denominator": "current_liabilities",
        "description": "Liquidity excluding inventory (acid test).",
        "good_range": "1.0 – 2.0",
        "low_concern": "Below 1.0 indicates over-reliance on inventory to meet obligations.",
        "high_concern": "Rarely a concern when very high.",
    },
    "cash_ratio": {
        "formula": "cash / current_liabilities",
        "numerator": "cash",
        "denominator": "current_liabilities",
        "description": "Most conservative liquidity measure using only cash & equivalents.",
        "good_range": "0.2 – 0.5",
        "low_concern": "Below 0.1 suggests very tight liquidity.",
        "high_concern": "Very high cash ratios may indicate under-investment.",
    },
    "debt_to_equity": {
        "formula": "total_debt / total_equity",
        "numerator": "total_debt",
        "denominator": "total_equity",
        "description": "Financial leverage — how much debt is used relative to equity.",
        "good_range": "0.5 – 1.5 (varies by industry)",
        "low_concern": "Very low may indicate under-leveraging.",
        "high_concern": "Above 2.0 suggests high financial risk.",
    },
    "debt_ratio": {
        "formula": "total_liabilities / total_assets",
        "numerator": "total_liabilities",
        "denominator": "total_assets",
        "description": "Proportion of assets financed by debt.",
        "good_range": "0.3 – 0.6",
        "low_concern": "Extremely low indicates very conservative financing.",
        "high_concern": "Above 0.7 suggests high leverage risk.",
    },
    "roe": {
        "formula": "net_income / total_equity",
        "numerator": "net_income",
        "denominator": "total_equity",
        "description": "Return on Equity — profitability relative to shareholder investment.",
        "good_range": "0.10 – 0.20 (10%–20%)",
        "low_concern": "Below 0.05 indicates poor returns for shareholders.",
        "high_concern": "Extremely high ROE may reflect high leverage rather than efficiency.",
    },
    "roa": {
        "formula": "net_income / total_assets",
        "numerator": "net_income",
        "denominator": "total_assets",
        "description": "Return on Assets — efficiency of asset utilization to generate profit.",
        "good_range": "0.05 – 0.15 (5%–15%)",
        "low_concern": "Below 0.02 indicates poor asset utilization.",
        "high_concern": "Investigate sustainability if consistently above 0.20.",
    },
    "gross_margin": {
        "formula": "gross_profit / revenue",
        "numerator": "gross_profit",
        "denominator": "revenue",
        "description": "Percentage of revenue retained after cost of goods sold.",
        "good_range": "0.30 – 0.60 (30%–60%, industry-dependent)",
        "low_concern": "Below 0.20 indicates pricing pressure or high COGS.",
        "high_concern": "Very high gross margins are generally positive.",
    },
    "net_margin": {
        "formula": "net_income / revenue",
        "numerator": "net_income",
        "denominator": "revenue",
        "description": "Net profit as a percentage of revenue.",
        "good_range": "0.05 – 0.20 (5%–20%)",
        "low_concern": "Below 0.02 signals thin margins.",
        "high_concern": "Consistently above 0.25 may attract competition.",
    },
    "asset_turnover": {
        "formula": "revenue / total_assets",
        "numerator": "revenue",
        "denominator": "total_assets",
        "description": "Efficiency of asset utilization to generate revenue.",
        "good_range": "0.5 – 2.0 (varies greatly by industry)",
        "low_concern": "Below 0.3 may indicate excess assets or poor utilization.",
        "high_concern": "Very high ratios can indicate under-investment in assets.",
    },
    "interest_coverage": {
        "formula": "ebit / interest_expense",
        "numerator": "ebit",
        "denominator": "interest_expense",
        "description": "Ability to service interest payments from operating earnings.",
        "good_range": "3.0 – 10.0",
        "low_concern": "Below 1.5 signals potential inability to service debt.",
        "high_concern": "Very high is generally positive.",
    },
    "inventory_turnover": {
        "formula": "cogs / inventory",
        "numerator": "cogs",
        "denominator": "inventory",
        "description": "How many times inventory is sold and replaced in a period.",
        "good_range": "4 – 12 (industry-dependent)",
        "low_concern": "Below 2 suggests overstocking or slow-moving inventory.",
        "high_concern": "Very high may indicate insufficient inventory leading to stockouts.",
    },
}


# ---------------------------------------------------------------------------
# tool_calculate_ratio
# ---------------------------------------------------------------------------


def tool_calculate_ratio(ratio_name: str, **financial_fields: Any) -> str:
    """Compute a named financial ratio from provided field values.

    Args:
        ratio_name: Name of the ratio (e.g. ``"current_ratio"``, ``"roe"``).
        **financial_fields: Named financial statement fields as keyword arguments
            (e.g. ``current_assets=800000, current_liabilities=400000``).

    Returns:
        Formatted string with the computed ratio value and a brief interpretation.
    """
    ratio_name_lower = ratio_name.lower().replace(" ", "_").replace("-", "_")
    defn = _RATIO_DEFINITIONS.get(ratio_name_lower)

    if defn is None:
        # Generic fallback: try num/denom from kwargs if provided
        available = ", ".join(sorted(_RATIO_DEFINITIONS.keys()))
        return f"Unknown ratio: '{ratio_name}'. Supported ratios: {available}."

    # Special case: quick ratio uses derived numerator
    if ratio_name_lower == "quick_ratio":
        ca = financial_fields.get("current_assets")
        inv = financial_fields.get("inventory", 0.0)
        cl = financial_fields.get("current_liabilities")
        if ca is None or cl is None:
            return (
                "Cannot compute quick_ratio: need 'current_assets' and 'current_liabilities' (optionally 'inventory')."
            )
        try:
            numerator_val = float(ca) - float(inv if inv is not None else 0.0)
            denom_val = float(cl)
        except (TypeError, ValueError):
            return "Cannot compute quick_ratio: invalid numeric values provided."
        value = _safe_div(numerator_val, denom_val)
    else:
        num_key: str = defn["numerator"]
        den_key: str = defn["denominator"]
        raw_num = financial_fields.get(num_key)
        raw_den = financial_fields.get(den_key)

        if raw_num is None or raw_den is None:
            missing = []
            if raw_num is None:
                missing.append(f"'{num_key}'")
            if raw_den is None:
                missing.append(f"'{den_key}'")
            return f"Cannot compute '{ratio_name}': missing field(s) {', '.join(missing)}. Formula: {defn['formula']}."

        try:
            value = _safe_div(float(raw_num), float(raw_den))
        except (TypeError, ValueError):
            return f"Cannot compute '{ratio_name}': non-numeric values provided."

    interpretation = _interpret_ratio(ratio_name_lower, value, defn)
    return (
        f"Ratio: {ratio_name}\n"
        f"Value: {_fmt(value, 4)}\n"
        f"Formula: {defn['formula']}\n"
        f"Interpretation: {interpretation}\n"
        f"Typical range: {defn['good_range']}"
    )


def _interpret_ratio(name: str, value: float, defn: dict[str, Any]) -> str:
    """Return a brief qualitative interpretation string."""
    # Parse good_range into low/high rough thresholds when possible
    desc = defn["description"]
    if value == 0.0:
        return f"{desc} — value is zero; check input data."
    # Custom thresholds per ratio
    _thresholds: dict[str, tuple[float, float]] = {
        "current_ratio": (1.0, 3.0),
        "quick_ratio": (1.0, 2.0),
        "cash_ratio": (0.1, 0.5),
        "debt_to_equity": (0.5, 2.0),
        "debt_ratio": (0.3, 0.7),
        "roe": (0.05, 0.30),
        "roa": (0.02, 0.20),
        "gross_margin": (0.20, 0.80),
        "net_margin": (0.02, 0.30),
        "asset_turnover": (0.3, 3.0),
        "interest_coverage": (1.5, 15.0),
        "inventory_turnover": (2.0, 20.0),
    }
    low, high = _thresholds.get(name, (0.0, float("inf")))
    if value < low:
        concern = defn.get("low_concern", "Below typical range.")
        return f"Below typical range ({_fmt(value, 2)}). {concern}"
    if value > high:
        concern = defn.get("high_concern", "Above typical range.")
        return f"Above typical range ({_fmt(value, 2)}). {concern}"
    return f"Within typical range ({_fmt(value, 2)}). {desc}"


# ---------------------------------------------------------------------------
# tool_compare_ratios
# ---------------------------------------------------------------------------


def tool_compare_ratios(
    ratio_name: str,
    value_a: float,
    value_b: float,
    label_a: str = "Period A",
    label_b: str = "Period B",
) -> str:
    """Compare two ratio values and return a comparison analysis string.

    Args:
        ratio_name: Name of the ratio being compared.
        value_a: First ratio value.
        value_b: Second ratio value.
        label_a: Label for the first value.
        label_b: Label for the second value.

    Returns:
        Formatted comparison analysis string.
    """
    try:
        a = float(value_a)
        b = float(value_b)
    except (TypeError, ValueError):
        return f"Cannot compare '{ratio_name}': non-numeric values provided."

    if abs(a) < 1e-12:
        change_pct = 0.0
        direction = "unchanged"
    else:
        change_pct = (b - a) / abs(a) * 100.0
        if abs(change_pct) < 0.01:
            direction = "unchanged"
        elif change_pct > 0:
            direction = "increased"
        else:
            direction = "decreased"

    abs_change = b - a
    lines = [
        f"Ratio comparison: {ratio_name}",
        f"  {label_a}: {_fmt(a, 4)}",
        f"  {label_b}: {_fmt(b, 4)}",
        f"  Change: {_fmt(abs_change, 4)} ({_fmt(change_pct, 2)}% {direction})",
    ]

    # Add qualitative commentary
    ratio_lower = ratio_name.lower().replace(" ", "_").replace("-", "_")
    defn = _RATIO_DEFINITIONS.get(ratio_lower)
    if defn:
        lines.append(f"  Context: {defn['description']}")
        lines.append(f"  Typical range: {defn['good_range']}")

    if direction == "unchanged":
        lines.append("  Assessment: No meaningful change between periods.")
    elif change_pct > 20:
        lines.append(
            f"  Assessment: Significant {direction} of {_fmt(abs(change_pct), 1)}% warrants further investigation."
        )
    elif change_pct > 5:
        lines.append(f"  Assessment: Moderate {direction} of {_fmt(abs(change_pct), 1)}%.")
    else:
        lines.append(f"  Assessment: Minor {direction} of {_fmt(abs(change_pct), 1)}%.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool_explain_ratio
# ---------------------------------------------------------------------------


def tool_explain_ratio(ratio_name: str) -> str:
    """Return a comprehensive explanation of a financial ratio.

    Args:
        ratio_name: Name of the ratio to explain.

    Returns:
        Multi-line string explaining the ratio, its formula, typical ranges,
        and what high/low values indicate.
    """
    ratio_lower = ratio_name.lower().replace(" ", "_").replace("-", "_")
    defn = _RATIO_DEFINITIONS.get(ratio_lower)

    if defn is None:
        available = ", ".join(sorted(_RATIO_DEFINITIONS.keys()))
        return f"No explanation available for '{ratio_name}'. Known ratios: {available}."

    return (
        f"Ratio: {ratio_name}\n"
        f"Formula: {defn['formula']}\n"
        f"Description: {defn['description']}\n"
        f"Typical good range: {defn['good_range']}\n"
        f"When low: {defn['low_concern']}\n"
        f"When high: {defn['high_concern']}"
    )


# ---------------------------------------------------------------------------
# tool_assess_distress  (Altman Z-Score based)
# ---------------------------------------------------------------------------


def tool_assess_distress(**financial_fields: Any) -> str:
    """Assess financial distress using the Altman Z-Score model.

    Computes Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5 where:
      X1 = Working Capital / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Book Equity / Total Liabilities
      X5 = Revenue / Total Assets

    Args:
        **financial_fields: Named fields including ``current_assets``,
            ``current_liabilities``, ``total_assets``, ``retained_earnings``
            (or ``net_income`` as proxy), ``ebit`` (or ``operating_income``),
            ``total_equity``, ``total_liabilities``, ``revenue``.

    Returns:
        Formatted string with Z-Score, zone classification, and risk assessment.
    """

    def _get(key: str, *aliases: str) -> Optional[float]:
        for k in (key, *aliases):
            v = financial_fields.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    total_assets = _get("total_assets")
    if total_assets is None or total_assets <= 0:
        return "Cannot compute distress score: 'total_assets' is required and must be positive."

    current_assets = _get("current_assets") or 0.0
    current_liabilities = _get("current_liabilities") or 0.0
    retained_earnings = _get("retained_earnings", "net_income") or 0.0
    ebit = _get("ebit", "operating_income", "ebit_or_operating_income") or 0.0
    total_equity = _get("total_equity") or 0.0
    total_liabilities = _get("total_liabilities") or 0.0
    revenue = _get("revenue") or 0.0

    working_capital = current_assets - current_liabilities

    x1 = _safe_div(working_capital, total_assets)
    x2 = _safe_div(retained_earnings, total_assets)
    x3 = _safe_div(ebit, total_assets)
    x4 = _safe_div(total_equity, max(total_liabilities, 1e-6))
    x5 = _safe_div(revenue, total_assets)

    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z_score >= 2.99:
        zone = "Safe"
        risk_level = "LOW"
        summary = "Company appears financially healthy with low distress risk."
    elif z_score >= 1.81:
        zone = "Grey"
        risk_level = "MODERATE"
        summary = "Company is in the grey zone; monitor trends closely."
    else:
        zone = "Distress"
        risk_level = "HIGH"
        summary = "Company shows signs of financial distress. Immediate attention recommended."

    return (
        f"Altman Z-Score Analysis\n"
        f"Z-Score: {_fmt(z_score, 3)}\n"
        f"Zone: {zone}  |  Risk Level: {risk_level}\n"
        f"Components:\n"
        f"  X1 (Working Capital/TA): {_fmt(x1, 4)}\n"
        f"  X2 (Retained Earnings/TA): {_fmt(x2, 4)}\n"
        f"  X3 (EBIT/TA): {_fmt(x3, 4)}\n"
        f"  X4 (Equity/Liabilities): {_fmt(x4, 4)}\n"
        f"  X5 (Revenue/TA): {_fmt(x5, 4)}\n"
        f"Assessment: {summary}"
    )


# ---------------------------------------------------------------------------
# tool_check_anomalies
# ---------------------------------------------------------------------------


def tool_check_anomalies(metric_values: list[float], metric_name: str = "metric") -> str:
    """Identify statistical outliers in a list of metric values.

    Uses Z-score method: values with |z| > 2.0 are flagged as anomalies.

    Args:
        metric_values: List of numeric metric observations.
        metric_name: Name of the metric for display purposes.

    Returns:
        Formatted string describing any anomalies found.
    """
    if not metric_values:
        return f"No values provided for '{metric_name}' anomaly check."

    cleaned: list[float] = []
    for v in metric_values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                cleaned.append(fv)
        except (TypeError, ValueError):
            continue

    if len(cleaned) < 2:
        return f"Need at least 2 finite values to detect anomalies in '{metric_name}'."

    mean_val = statistics.mean(cleaned)
    if len(cleaned) >= 2:
        stdev_val = statistics.stdev(cleaned)
    else:
        stdev_val = 0.0

    if stdev_val < 1e-12:
        return f"Anomaly check for '{metric_name}': all values identical ({_fmt(mean_val, 4)}). No anomalies detected."

    anomalies: list[tuple[int, float, float]] = []
    for i, v in enumerate(cleaned):
        z = (v - mean_val) / stdev_val
        if abs(z) > 2.0:
            anomalies.append((i, v, z))

    lines = [
        f"Anomaly Check: {metric_name}",
        f"  Values analyzed: {len(cleaned)}",
        f"  Mean: {_fmt(mean_val, 4)}  |  StdDev: {_fmt(stdev_val, 4)}",
        "  Detection threshold: |z| > 2.0",
    ]

    if not anomalies:
        lines.append("  Result: No anomalies detected.")
    else:
        lines.append(f"  Anomalies found: {len(anomalies)}")
        for idx, val, z_val in anomalies:
            direction = "high" if z_val > 0 else "low"
            lines.append(f"    Index {idx}: value={_fmt(val, 4)}, z={_fmt(z_val, 2)} ({direction} outlier)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool_evaluate_leverage
# ---------------------------------------------------------------------------


def tool_evaluate_leverage(
    total_debt: Optional[float] = None,
    total_equity: Optional[float] = None,
    total_assets: Optional[float] = None,
    ebit: Optional[float] = None,
    interest_expense: Optional[float] = None,
) -> str:
    """Evaluate a company's leverage position using multiple metrics.

    Args:
        total_debt: Total debt obligations.
        total_equity: Total shareholder equity.
        total_assets: Total assets.
        ebit: Earnings before interest and taxes.
        interest_expense: Annual interest expense.

    Returns:
        Multi-line leverage assessment string.
    """
    lines = ["Leverage Assessment"]
    computed_any = False

    # Debt-to-Equity
    if total_debt is not None and total_equity is not None:
        try:
            dte = _safe_div(float(total_debt), float(total_equity))
            risk = "low" if dte < 0.5 else ("moderate" if dte < 1.5 else "high")
            lines.append(f"  Debt-to-Equity: {_fmt(dte, 3)} (leverage risk: {risk})")
            computed_any = True
        except (TypeError, ValueError):
            pass

    # Debt Ratio
    if total_assets is not None and total_debt is not None:
        try:
            dr = _safe_div(float(total_debt), float(total_assets))
            lines.append(f"  Debt Ratio: {_fmt(dr, 3)}")
            computed_any = True
        except (TypeError, ValueError):
            pass

    # Interest Coverage
    if ebit is not None and interest_expense is not None:
        try:
            ic = _safe_div(float(ebit), float(interest_expense))
            safety = "adequate" if ic >= 3.0 else ("borderline" if ic >= 1.5 else "inadequate")
            lines.append(f"  Interest Coverage: {_fmt(ic, 2)}x (debt service: {safety})")
            computed_any = True
        except (TypeError, ValueError):
            pass

    if not computed_any:
        return (
            "Insufficient data for leverage assessment. Provide at least two of: "
            "total_debt, total_equity, total_assets, ebit, interest_expense."
        )

    # Overall assessment
    dte_val: Optional[float] = None
    ic_val: Optional[float] = None
    if total_debt is not None and total_equity is not None:
        try:
            dte_val = _safe_div(float(total_debt), float(total_equity))
        except (TypeError, ValueError):
            pass
    if ebit is not None and interest_expense is not None:
        try:
            ic_val = _safe_div(float(ebit), float(interest_expense))
        except (TypeError, ValueError):
            pass

    if dte_val is not None and dte_val > 2.0:
        lines.append("  Overall: HIGH leverage risk — debt significantly exceeds equity.")
    elif ic_val is not None and ic_val < 1.5:
        lines.append("  Overall: MODERATE-HIGH risk — interest coverage is thin.")
    else:
        lines.append("  Overall: Leverage appears manageable based on available data.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool_forecast
# ---------------------------------------------------------------------------


def tool_forecast(values: list[float], steps: int = 4) -> str:
    """Forecast future values using an AR(2) model with prediction intervals.

    Args:
        values: Historical time series values (at least 3 data points required).
        steps: Number of future periods to forecast (default 4).

    Returns:
        Formatted string with point forecasts and 95% prediction intervals.
    """
    if not values:
        return "No values provided for forecasting."

    cleaned: list[float] = []
    for v in values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                cleaned.append(fv)
        except (TypeError, ValueError):
            continue

    if len(cleaned) < 3:
        return f"Need at least 3 finite data points for forecasting; got {len(cleaned)}."

    try:
        steps_int = max(1, int(steps))
    except (TypeError, ValueError):
        steps_int = 4

    try:
        from ml.forecasting import SimpleARModel, compute_prediction_intervals

        order = min(2, len(cleaned) - 1)
        model = SimpleARModel(order=max(1, order))
        model.fit(cleaned)
        forecasts = model.predict(steps_int)
        residuals = model.get_residuals()
        lower, upper = compute_prediction_intervals(forecasts, residuals, confidence=0.95)
        method = f"AR({max(1, order)})"
    except ImportError:
        logger.debug("Forecasting module not available, using linear extrapolation")
        # Fallback: simple linear extrapolation
        n = len(cleaned)
        if n >= 2:
            slope = (cleaned[-1] - cleaned[0]) / (n - 1)
        else:
            slope = 0.0
        last_val = cleaned[-1]
        forecasts = [last_val + slope * (i + 1) for i in range(steps_int)]
        # Estimate rough interval from historical variance
        try:
            std_val = statistics.stdev(cleaned)
        except statistics.StatisticsError:
            std_val = abs(last_val) * 0.1
        lower = [f - 1.96 * std_val for f in forecasts]
        upper = [f + 1.96 * std_val for f in forecasts]
        method = "Linear Extrapolation"
    except Exception as exc:
        logger.warning("Forecast tool failed: %s", exc)
        return f"Forecast error: {type(exc).__name__}"

    lines = [
        f"Forecast ({method}, {steps_int} periods):",
        f"  Historical: {', '.join(_fmt(v, 2) for v in cleaned[-5:])} (last 5 shown)",
    ]
    for i, (fc, lo, hi) in enumerate(zip(forecasts, lower, upper), start=1):
        lines.append(f"  Period +{i}: {_fmt(fc, 4)}  [95% CI: {_fmt(lo, 4)} – {_fmt(hi, 4)}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool_analyze_trend
# ---------------------------------------------------------------------------


def tool_analyze_trend(values: list[float]) -> str:
    """Analyze trend direction, strength, and inflection points in a series.

    Args:
        values: List of sequential numeric observations.

    Returns:
        Formatted string with trend direction, growth rates, and inflections.
    """
    if not values:
        return "No values provided for trend analysis."

    cleaned: list[float] = []
    for v in values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                cleaned.append(fv)
        except (TypeError, ValueError):
            continue

    if len(cleaned) < 2:
        return f"Need at least 2 values for trend analysis; got {len(cleaned)}."

    n = len(cleaned)
    first_val = cleaned[0]
    last_val = cleaned[-1]

    # Period-over-period changes
    changes = [cleaned[i] - cleaned[i - 1] for i in range(1, n)]
    positive_changes = sum(1 for c in changes if c > 0)
    negative_changes = sum(1 for c in changes if c < 0)
    flat_changes = len(changes) - positive_changes - negative_changes

    # Determine direction
    pct_positive = positive_changes / len(changes) * 100
    if pct_positive >= 70:
        direction = "Upward"
    elif pct_positive <= 30:
        direction = "Downward"
    else:
        direction = "Mixed/Stable"

    # Overall change
    overall_change_abs = last_val - first_val
    if abs(first_val) > 1e-12:
        overall_change_pct = overall_change_abs / abs(first_val) * 100.0
    else:
        overall_change_pct = 0.0

    # CAGR if positive values
    cagr_str = "N/A"
    if first_val > 0 and last_val > 0 and n >= 2:
        try:
            cagr = (last_val / first_val) ** (1.0 / (n - 1)) - 1.0
            cagr_str = f"{cagr * 100:.2f}%"
        except (ZeroDivisionError, ValueError):
            cagr_str = "N/A"

    # Detect inflection points (sign changes in first derivative)
    inflections: list[int] = []
    for i in range(1, len(changes)):
        if changes[i - 1] * changes[i] < 0:
            inflections.append(i)

    # Strength: R^2 of linear fit using numpy-free method
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(cleaned)
    x_vals = list(range(n))
    ss_xx = sum((xi - x_mean) ** 2 for xi in x_vals)
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_vals, cleaned))
    ss_yy = sum((yi - y_mean) ** 2 for yi in cleaned)

    if ss_xx > 1e-12 and ss_yy > 1e-12:
        r_sq = (ss_xy**2) / (ss_xx * ss_yy)
        strength = "strong" if r_sq >= 0.7 else ("moderate" if r_sq >= 0.4 else "weak")
        r_sq_str = f"{r_sq:.3f}"
    else:
        r_sq_str = "N/A"
        strength = "N/A"

    lines = [
        "Trend Analysis",
        f"  Data points: {n}",
        f"  Direction: {direction} ({positive_changes} up, {negative_changes} down, {flat_changes} flat)",
        f"  Overall change: {_fmt(overall_change_abs, 4)} ({_fmt(overall_change_pct, 2)}%)",
        f"  CAGR (period): {cagr_str}",
        f"  Linear fit R²: {r_sq_str} (trend strength: {strength})",
    ]

    if inflections:
        lines.append(f"  Inflection points at indices: {inflections}")
    else:
        lines.append("  No inflection points detected (consistent trend).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool_detect_seasonality
# ---------------------------------------------------------------------------


def tool_detect_seasonality(values: list[float], period: int = 4) -> str:
    """Detect seasonal patterns in a time series.

    Uses simple seasonal strength calculation by comparing within-period
    variance to total variance.

    Args:
        values: List of sequential numeric observations.
        period: Assumed seasonal period (default 4 for quarterly data).

    Returns:
        Formatted string describing detected seasonality.
    """
    if not values:
        return "No values provided for seasonality detection."

    cleaned: list[float] = []
    for v in values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                cleaned.append(fv)
        except (TypeError, ValueError):
            continue

    n = len(cleaned)
    if n < 2 * period:
        return f"Need at least {2 * period} values to detect seasonality with period={period}; got {n}."

    try:
        p = max(2, int(period))
    except (TypeError, ValueError):
        p = 4

    # Compute per-position averages across complete cycles
    n_complete = (n // p) * p
    complete = cleaned[:n_complete]

    pos_means: list[float] = []
    for pos in range(p):
        pos_vals = [complete[i] for i in range(pos, n_complete, p)]
        pos_means.append(statistics.mean(pos_vals))

    overall_mean = statistics.mean(complete)

    # Seasonal deviations
    deviations = [pm - overall_mean for pm in pos_means]
    max_dev = max(abs(d) for d in deviations)

    if abs(overall_mean) > 1e-12:
        max_dev_pct = max_dev / abs(overall_mean) * 100.0
    else:
        max_dev_pct = 0.0

    # Classify seasonal strength
    if max_dev_pct >= 15.0:
        strength = "Strong"
        detected = True
    elif max_dev_pct >= 5.0:
        strength = "Moderate"
        detected = True
    else:
        strength = "Weak/None"
        detected = False

    lines = [
        f"Seasonality Detection (period={p})",
        f"  Data points: {n} ({n // p} complete cycles + {n % p} partial)",
        f"  Max seasonal deviation: {_fmt(max_dev, 4)} ({_fmt(max_dev_pct, 2)}% of mean)",
        f"  Seasonal strength: {strength}",
        f"  Seasonality detected: {'Yes' if detected else 'No'}",
        "  Per-position means:",
    ]
    for pos, pm in enumerate(pos_means):
        dev = pm - overall_mean
        lines.append(f"    Position {pos + 1}: {_fmt(pm, 4)} (deviation: {_fmt(dev, 4)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool_search_documents
# ---------------------------------------------------------------------------


def tool_search_documents(query: str, rag_instance: Any = None) -> str:
    """Retrieve relevant document chunks matching a query.

    Args:
        query: The search query string.
        rag_instance: Optional ``SimpleRAG`` instance. When ``None``, returns
            an informational message indicating no RAG is configured.

    Returns:
        Formatted string with retrieved document excerpts and metadata.
    """
    if not query or not query.strip():
        return "Cannot search: query string is empty."

    if rag_instance is None:
        return (
            f"Document search for '{query}': No RAG instance configured. "
            "Provide a SimpleRAG instance via rag_instance parameter."
        )

    try:
        results = rag_instance.retrieve(query, top_k=3)
    except Exception as exc:
        return f"Document search failed: {exc}"

    if not results:
        return f"No documents found matching: '{query}'"

    lines = [f"Document search results for: '{query}'", f"Found {len(results)} result(s):"]
    for i, doc in enumerate(results, start=1):
        if isinstance(doc, dict):
            content = doc.get("content", doc.get("text", str(doc)))
            source = doc.get("source", doc.get("filename", "unknown"))
            score = doc.get("score", doc.get("similarity", ""))
            score_str = f"  (score: {_fmt(float(score), 4)})" if score != "" else ""
            lines.append(f"\n[{i}] Source: {source}{score_str}")
            # Truncate long content for readability
            excerpt = str(content)[:400]
            if len(str(content)) > 400:
                excerpt += "..."
            lines.append(f"    {excerpt}")
        else:
            lines.append(f"\n[{i}] {str(doc)[:400]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool_format_section
# ---------------------------------------------------------------------------


def tool_format_section(content: str, section_type: str = "analysis") -> str:
    """Format raw content into a structured report section.

    Args:
        content: The raw text content to format.
        section_type: Type of section — one of ``"executive_summary"``,
            ``"analysis"``, ``"risk"``, ``"recommendation"``, ``"data"``.

    Returns:
        Formatted report section string with appropriate headers and structure.
    """
    if not content or not content.strip():
        return f"[Empty {section_type} section]"

    section_headers: dict[str, str] = {
        "executive_summary": "EXECUTIVE SUMMARY",
        "analysis": "FINANCIAL ANALYSIS",
        "risk": "RISK ASSESSMENT",
        "recommendation": "RECOMMENDATIONS",
        "data": "DATA SUMMARY",
        "trend": "TREND ANALYSIS",
        "forecast": "FORECAST",
    }

    header = section_headers.get(section_type.lower(), section_type.upper())
    separator = "=" * (len(header) + 4)

    return f"{separator}\n  {header}\n{separator}\n\n{content.strip()}\n\n{'-' * len(separator)}"
