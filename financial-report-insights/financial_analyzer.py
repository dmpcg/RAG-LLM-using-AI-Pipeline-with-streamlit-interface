"""
Charlie-Style Financial Intelligence Module.
CFO-grade financial analysis with automatic metric extraction and insight generation.
Named after Charlie Munger who embodied the principles of rigorous financial analysis.
"""

import ast
import logging
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import re

import pandas as pd
import numpy as np

from structured_types import (
    AnalysisResults,
    EfficiencyRatios,
    LeverageRatios,
    LiquidityRatios,
    ProfitabilityRatios,
)

logger = logging.getLogger(__name__)


def safe_divide(
    numerator: Optional[float],
    denominator: Optional[float],
    default: Optional[float] = None,
) -> Optional[float]:
    """Safely divide two numbers, returning *default* when division is impossible.

    Handles ``None`` inputs, zero denominators, and near-zero denominators
    (``abs(denominator) < 1e-12``) to avoid ``ZeroDivisionError`` and ``inf``
    results across all financial ratio calculations.
    """
    if numerator is None or denominator is None:
        return default
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


@dataclass
class VarianceResult:
    """Result of a variance calculation."""
    actual: float
    budget: float
    variance: float
    variance_percent: float
    favorable: bool
    category: str
    explanation: Optional[str] = None


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""
    metric_name: str
    values: List[float]
    periods: List[str]
    yoy_growth: Optional[float]
    qoq_growth: Optional[float]
    mom_growth: Optional[float]
    cagr: Optional[float]
    moving_avg_3: List[float]
    moving_avg_12: List[float]
    trend_direction: str  # 'up', 'down', 'stable'
    seasonality_detected: bool


@dataclass
class CashFlowAnalysis:
    """Cash flow analysis results."""
    operating_cf: Optional[float]
    investing_cf: Optional[float]
    financing_cf: Optional[float]
    free_cash_flow: Optional[float]
    dso: Optional[float]  # Days Sales Outstanding
    dio: Optional[float]  # Days Inventory Outstanding
    dpo: Optional[float]  # Days Payables Outstanding
    cash_conversion_cycle: Optional[float]


@dataclass
class WorkingCapitalAnalysis:
    """Working capital analysis results."""
    current_assets: Optional[float]
    current_liabilities: Optional[float]
    net_working_capital: Optional[float]
    working_capital_ratio: Optional[float]
    working_capital_turnover: Optional[float]


@dataclass
class BudgetAnalysis:
    """Comprehensive budget analysis."""
    total_budget: float
    total_actual: float
    total_variance: float
    total_variance_percent: float
    line_items: List[VarianceResult]
    favorable_items: List[VarianceResult]
    unfavorable_items: List[VarianceResult]
    largest_variances: List[VarianceResult]
    unmatched_items: List[str] = field(default_factory=list)


@dataclass
class Insight:
    """A generated insight."""
    category: str  # liquidity, profitability, efficiency, risk, trend
    severity: str  # info, warning, critical
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class Anomaly:
    """A detected anomaly."""
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    z_score: float
    description: str


@dataclass
class Forecast:
    """Simple forecast result."""
    metric_name: str
    historical_values: List[float]
    forecasted_values: List[float]
    forecast_periods: List[str]
    method: str
    confidence_interval: Tuple[float, float]


@dataclass
class DuPontAnalysis:
    """DuPont decomposition of Return on Equity.

    3-factor: ROE = Net Margin × Asset Turnover × Equity Multiplier
    5-factor: adds Tax Burden (NI/EBT) and Interest Burden (EBT/EBIT)
    """
    roe: Optional[float] = None
    net_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    equity_multiplier: Optional[float] = None
    # 5-factor extensions
    tax_burden: Optional[float] = None       # NI / EBT
    interest_burden: Optional[float] = None  # EBT / EBIT
    # Diagnosis
    primary_driver: Optional[str] = None  # which factor drives ROE most
    interpretation: Optional[str] = None


@dataclass
class AltmanZScore:
    """Altman Z-Score bankruptcy prediction model.

    Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
    Where:
      X1 = Working Capital / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Market Value of Equity / Total Liabilities (book value used as proxy)
      X5 = Sales / Total Assets
    """
    z_score: Optional[float] = None
    zone: str = "unknown"  # 'safe', 'grey', 'distress'
    components: Dict[str, Optional[float]] = field(default_factory=dict)
    interpretation: Optional[str] = None


@dataclass
class PiotroskiFScore:
    """Piotroski F-Score financial strength model (0-9 scale).

    Profitability (4 points): ROA>0, Operating CF>0, Delta ROA>0, Accruals
    Leverage/Liquidity (3 points): Delta Leverage<0, Delta Current Ratio>0, No dilution
    Operating Efficiency (2 points): Delta Gross Margin>0, Delta Asset Turnover>0
    """
    score: int = 0
    max_score: int = 9
    criteria: Dict[str, bool] = field(default_factory=dict)
    interpretation: Optional[str] = None


@dataclass
class CompositeHealthScore:
    """Weighted 0-100 composite financial health score with letter grade.

    Components (100 total):
      - Z-Score zone:     25 pts
      - F-Score:          25 pts
      - Profitability:    20 pts
      - Liquidity:        15 pts
      - Leverage:         15 pts
    """
    score: int = 0
    grade: str = "F"  # A, B, C, D, F
    component_scores: Dict[str, int] = field(default_factory=dict)
    interpretation: Optional[str] = None


@dataclass
class PeriodComparison:
    """Result of comparing two financial periods."""
    current_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    prior_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)
    improvements: List[str] = field(default_factory=list)
    deteriorations: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of a what-if scenario analysis comparing base vs adjusted data."""
    scenario_name: str = ""
    adjustments: Dict[str, float] = field(default_factory=dict)
    base_health: Optional[CompositeHealthScore] = None
    scenario_health: Optional[CompositeHealthScore] = None
    base_z_score: Optional[float] = None
    scenario_z_score: Optional[float] = None
    base_f_score: Optional[int] = None
    scenario_f_score: Optional[int] = None
    base_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    scenario_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    impact_summary: str = ""


@dataclass
class ProbabilityWeightedResult:
    """Result of probability-weighted multi-scenario analysis."""
    scenarios: List[ScenarioResult] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    scenario_names: List[str] = field(default_factory=list)
    expected_health_score: Optional[float] = None
    expected_z_score: Optional[float] = None
    distress_probability: Optional[float] = None  # P(Z < 1.81)
    summary: str = ""


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis across a range of adjustments."""
    variable_name: str = ""
    variable_labels: List[str] = field(default_factory=list)
    variable_multipliers: List[float] = field(default_factory=list)
    metric_results: Dict[str, List[Optional[float]]] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo simulation on financial metrics."""
    n_simulations: int = 0
    variable_assumptions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metric_distributions: Dict[str, List[float]] = field(default_factory=dict)
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: str = ""


@dataclass
class CashFlowForecast:
    """Multi-period cash flow projection."""
    periods: List[str] = field(default_factory=list)
    revenue_forecast: List[float] = field(default_factory=list)
    expense_forecast: List[float] = field(default_factory=list)
    net_cash_flow: List[float] = field(default_factory=list)
    cumulative_cash: List[float] = field(default_factory=list)
    fcf_forecast: List[float] = field(default_factory=list)
    dcf_value: Optional[float] = None
    terminal_value: Optional[float] = None
    discount_rate: float = 0.10
    growth_rates: List[float] = field(default_factory=list)


@dataclass
class TornadoDriver:
    """A single driver's impact on a target metric."""
    variable: str = ""
    low_value: float = 0.0
    high_value: float = 0.0
    base_value: float = 0.0
    spread: float = 0.0  # high_value - low_value


@dataclass
class TornadoResult:
    """Result of tornado/driver ranking analysis."""
    target_metric: str = ""
    base_metric_value: float = 0.0
    drivers: List[TornadoDriver] = field(default_factory=list)
    top_driver: str = ""


@dataclass
class BreakevenResult:
    """Breakeven analysis result."""
    breakeven_revenue: Optional[float] = None
    current_revenue: Optional[float] = None
    margin_of_safety: Optional[float] = None  # (current - breakeven) / current
    fixed_costs: Optional[float] = None
    variable_cost_ratio: Optional[float] = None
    contribution_margin_ratio: Optional[float] = None


@dataclass
class CovenantCheck:
    """Result of a single covenant/KPI check."""
    name: str = ""
    current_value: Optional[float] = None
    threshold: float = 0.0
    direction: str = "above"  # 'above' = good if current >= threshold
    status: str = "unknown"   # 'pass', 'warning', 'breach'
    headroom: Optional[float] = None  # how far above/below threshold


@dataclass
class CovenantMonitorResult:
    """Result of covenant monitoring across all KPIs."""
    checks: List[CovenantCheck] = field(default_factory=list)
    passes: int = 0
    warnings: int = 0
    breaches: int = 0
    summary: str = ""


@dataclass
class WorkingCapitalResult:
    """Working capital efficiency metrics."""
    dso: Optional[float] = None   # Days Sales Outstanding
    dio: Optional[float] = None   # Days Inventory Outstanding
    dpo: Optional[float] = None   # Days Payables Outstanding
    ccc: Optional[float] = None   # Cash Conversion Cycle (DSO + DIO - DPO)
    net_working_capital: Optional[float] = None
    working_capital_ratio: Optional[float] = None
    insights: List[str] = field(default_factory=list)


@dataclass
class NarrativeReport:
    """Auto-generated narrative intelligence from financial analysis."""
    headline: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class RegressionForecast:
    """Result of regression-based trend forecasting."""
    metric_name: str = ""
    historical_values: List[float] = field(default_factory=list)
    forecast_values: List[float] = field(default_factory=list)
    r_squared: Optional[float] = None
    slope: Optional[float] = None
    intercept: Optional[float] = None
    method: str = "linear"  # linear, exponential, polynomial
    confidence_upper: List[float] = field(default_factory=list)
    confidence_lower: List[float] = field(default_factory=list)
    periods_ahead: int = 0


@dataclass
class BenchmarkComparison:
    """Single metric comparison against industry benchmark."""
    metric_name: str = ""
    company_value: Optional[float] = None
    industry_median: Optional[float] = None
    industry_p25: Optional[float] = None
    industry_p75: Optional[float] = None
    percentile_rank: Optional[float] = None  # 0-100, where company falls
    rating: str = ""  # "above average", "average", "below average"


@dataclass
class IndustryBenchmarkResult:
    """Full industry benchmarking result."""
    industry_name: str = ""
    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    overall_percentile: Optional[float] = None
    summary: str = ""


@dataclass
class CustomKPIDefinition:
    """Definition of a user-defined KPI formula."""
    name: str = ""
    formula: str = ""  # e.g. "revenue / total_assets"
    description: str = ""
    target_min: Optional[float] = None
    target_max: Optional[float] = None


@dataclass
class CustomKPIResult:
    """Result of evaluating a custom KPI."""
    name: str = ""
    formula: str = ""
    value: Optional[float] = None
    meets_target: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class CustomKPIReport:
    """Collection of custom KPI evaluation results."""
    results: List[CustomKPIResult] = field(default_factory=list)
    summary: str = ""


@dataclass
class PeerCompanyData:
    """Financial data for a single peer in comparison."""
    name: str = ""
    data: Optional['FinancialData'] = None


@dataclass
class PeerMetricComparison:
    """A single metric compared across peers."""
    metric_name: str = ""
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    best_performer: str = ""
    worst_performer: str = ""
    average: Optional[float] = None
    median: Optional[float] = None


@dataclass
class PeerComparisonReport:
    """Full peer comparison analysis."""
    peer_names: List[str] = field(default_factory=list)
    comparisons: List[PeerMetricComparison] = field(default_factory=list)
    rankings: Dict[str, int] = field(default_factory=dict)
    summary: str = ""


@dataclass
class RatioDecompositionNode:
    """A node in the ratio decomposition tree."""
    name: str = ""
    value: Optional[float] = None
    formula: str = ""
    children: List['RatioDecompositionNode'] = field(default_factory=list)


@dataclass
class RatioDecompositionTree:
    """Full ratio decomposition from ROE down to individual drivers."""
    root: Optional[RatioDecompositionNode] = None
    summary: str = ""


@dataclass
class RatingCategory:
    """A scored category in the financial rating."""
    name: str = ""
    score: float = 0.0
    max_score: float = 10.0
    grade: str = ""
    details: str = ""


@dataclass
class FinancialRating:
    """Comprehensive financial rating scorecard."""
    overall_score: float = 0.0
    overall_grade: str = ""
    categories: List[RatingCategory] = field(default_factory=list)
    summary: str = ""


@dataclass
class WaterfallItem:
    """A single item in a variance waterfall."""
    label: str = ""
    value: float = 0.0
    cumulative: float = 0.0
    item_type: str = "delta"  # "start", "delta", "total"


@dataclass
class VarianceWaterfall:
    """Waterfall breakdown of variance between two periods."""
    start_value: float = 0.0
    end_value: float = 0.0
    items: List[WaterfallItem] = field(default_factory=list)
    total_variance: float = 0.0
    summary: str = ""


@dataclass
class EarningsQualityResult:
    """Earnings quality analysis result."""
    accrual_ratio: Optional[float] = None
    cash_to_income_ratio: Optional[float] = None
    quality_score: float = 0.0
    quality_grade: str = ""
    indicators: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class CapitalEfficiencyResult:
    """Capital efficiency and value creation analysis."""
    roic: Optional[float] = None
    invested_capital: Optional[float] = None
    nopat: Optional[float] = None
    eva: Optional[float] = None
    wacc_estimate: float = 0.10
    capital_turnover: Optional[float] = None
    reinvestment_rate: Optional[float] = None
    summary: str = ""


@dataclass
class LiquidityStressResult:
    """Liquidity stress test result."""
    current_cash: float = 0.0
    monthly_burn: float = 0.0
    months_of_cash: Optional[float] = None
    stressed_quick_ratio: Optional[float] = None
    stress_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = ""  # "Low", "Moderate", "High", "Critical"
    summary: str = ""


@dataclass
class HealthDimension:
    """A single dimension of the composite health score."""
    name: str = ""
    score: float = 0.0
    max_score: float = 100.0
    weight: float = 0.0
    status: str = ""  # "green", "yellow", "red"
    detail: str = ""


@dataclass
class ComprehensiveHealthResult:
    """Comprehensive financial health score aggregating all analyses."""
    overall_score: float = 0.0
    grade: str = ""  # A+ through F
    dimensions: List[HealthDimension] = field(default_factory=list)
    summary: str = ""


@dataclass
class OperatingLeverageResult:
    """Operating leverage and break-even analysis result."""
    degree_of_operating_leverage: Optional[float] = None
    contribution_margin: Optional[float] = None
    contribution_margin_ratio: Optional[float] = None
    estimated_fixed_costs: Optional[float] = None
    estimated_variable_costs: Optional[float] = None
    break_even_revenue: Optional[float] = None
    margin_of_safety: Optional[float] = None
    margin_of_safety_pct: Optional[float] = None
    cost_structure: str = ""  # "High Fixed", "Balanced", "High Variable"
    summary: str = ""


@dataclass
class CashFlowQualityResult:
    """Cash flow quality and free cash flow analysis result."""
    fcf: Optional[float] = None
    fcf_yield: Optional[float] = None
    fcf_margin: Optional[float] = None
    ocf_to_net_income: Optional[float] = None
    capex_to_revenue: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    cash_conversion_efficiency: Optional[float] = None
    quality_grade: str = ""  # "Strong", "Adequate", "Weak", "Poor"
    indicators: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class AssetEfficiencyResult:
    """Asset efficiency and turnover analysis result."""
    total_asset_turnover: Optional[float] = None
    fixed_asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    payables_turnover: Optional[float] = None
    working_capital_turnover: Optional[float] = None
    equity_turnover: Optional[float] = None
    efficiency_score: float = 0.0  # 0-10
    efficiency_grade: str = ""  # "Excellent", "Good", "Average", "Below Average"
    summary: str = ""


@dataclass
class ProfitabilityDecompResult:
    """Profitability decomposition and return analysis result."""
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    invested_capital: Optional[float] = None
    nopat: Optional[float] = None
    spread: Optional[float] = None  # ROIC - cost of capital proxy
    economic_profit: Optional[float] = None  # Spread × Invested Capital
    asset_turnover: Optional[float] = None
    financial_leverage: Optional[float] = None  # TA / Equity
    tax_efficiency: Optional[float] = None  # NI / EBT
    capital_intensity: Optional[float] = None  # TA / Revenue
    profitability_score: float = 0.0  # 0-10
    profitability_grade: str = ""  # "Elite", "Strong", "Adequate", "Poor"
    summary: str = ""


@dataclass
class RiskAdjustedResult:
    """Risk-adjusted performance metrics result."""
    return_on_risk: Optional[float] = None  # ROE / leverage ratio
    risk_adjusted_roe: Optional[float] = None  # ROE adjusted for financial risk
    risk_adjusted_roa: Optional[float] = None  # ROA adjusted for operating risk
    debt_adjusted_return: Optional[float] = None  # NI / (Equity + Debt)
    volatility_proxy: Optional[float] = None  # Margin variability proxy
    downside_risk: Optional[float] = None  # Probability of loss given leverage
    margin_of_safety: Optional[float] = None  # (EBIT - Interest) / Interest
    return_per_unit_risk: Optional[float] = None  # Return / Risk ratio
    risk_score: float = 0.0  # 0-10
    risk_grade: str = ""  # "Superior", "Favorable", "Neutral", "Elevated"
    summary: str = ""


@dataclass
class ValuationIndicatorsResult:
    """Valuation indicators and intrinsic value estimates."""
    earnings_yield: Optional[float] = None  # NI / Total Equity (proxy for E/P)
    book_value_per_share_proxy: Optional[float] = None  # Equity value
    ev_proxy: Optional[float] = None  # Equity + Debt - Cash
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    ev_to_ebit: Optional[float] = None
    price_to_book_proxy: Optional[float] = None  # (Equity + premium) / Book value
    return_on_invested_capital: Optional[float] = None
    graham_number_proxy: Optional[float] = None  # sqrt(22.5 * EPS_proxy * BV_proxy)
    intrinsic_value_proxy: Optional[float] = None  # DCF-lite estimate
    margin_of_safety_pct: Optional[float] = None  # (intrinsic - book) / intrinsic
    valuation_score: float = 0.0  # 0-10
    valuation_grade: str = ""  # "Undervalued", "Fair Value", "Fully Valued", "Overvalued"
    summary: str = ""


@dataclass
class SustainableGrowthResult:
    """Sustainability and growth capacity analysis result."""
    sustainable_growth_rate: Optional[float] = None  # ROE × retention ratio
    internal_growth_rate: Optional[float] = None  # ROA × retention / (1 - ROA × retention)
    retention_ratio: Optional[float] = None  # 1 - payout ratio
    payout_ratio: Optional[float] = None  # Dividends / NI
    plowback_capacity: Optional[float] = None  # Retained earnings / NI
    roe: Optional[float] = None
    roa: Optional[float] = None
    asset_growth_rate: Optional[float] = None  # Growth in TA if prior available
    equity_growth_rate: Optional[float] = None  # Retained earnings / Equity
    reinvestment_rate: Optional[float] = None  # Capex / Depreciation
    growth_profitability_balance: Optional[float] = None  # SGR / cost of equity proxy
    growth_score: float = 0.0  # 0-10
    growth_grade: str = ""  # "High Growth", "Sustainable", "Moderate", "Constrained"
    summary: str = ""


@dataclass
class ConcentrationRiskResult:
    """Concentration and structural risk analysis result."""
    revenue_asset_intensity: Optional[float] = None  # Revenue / TA (asset turnover)
    operating_dependency: Optional[float] = None  # OI / Revenue
    asset_composition_current: Optional[float] = None  # CA / TA
    asset_composition_fixed: Optional[float] = None  # (TA - CA) / TA
    liability_structure_current: Optional[float] = None  # CL / TL
    earnings_retention_ratio: Optional[float] = None  # NI / EBITDA
    working_capital_intensity: Optional[float] = None  # NWC / Revenue
    capex_intensity: Optional[float] = None  # Capex / Revenue
    interest_burden: Optional[float] = None  # Interest / EBIT
    cash_conversion_efficiency: Optional[float] = None  # OCF / NI
    fixed_asset_ratio: Optional[float] = None  # Fixed assets / Equity
    debt_concentration: Optional[float] = None  # Total debt / TL
    concentration_score: float = 0.0  # 0-10
    concentration_grade: str = ""  # "Well Diversified", "Balanced", "Concentrated", "Highly Concentrated"
    summary: str = ""


@dataclass
class MarginOfSafetyResult:
    """Margin of safety and intrinsic value cushion analysis result."""
    earnings_yield: Optional[float] = None  # NI / Market Cap or EPS/Price
    book_value_per_share: Optional[float] = None  # Equity / Shares
    price_to_book: Optional[float] = None  # Price / BV per share
    book_value_discount: Optional[float] = None  # 1 - P/B (positive = trading below BV)
    intrinsic_value_estimate: Optional[float] = None  # NI / discount rate (capitalized earnings)
    market_cap: Optional[float] = None  # Shares * Price
    intrinsic_margin: Optional[float] = None  # (IV - MC) / IV
    tangible_bv: Optional[float] = None  # Equity - Intangibles (approx)
    liquidation_value: Optional[float] = None  # Tangible BV * discount
    net_current_asset_value: Optional[float] = None  # CA - TL (Graham NCAV)
    ncav_per_share: Optional[float] = None  # NCAV / shares
    safety_score: float = 0.0  # 0-10
    safety_grade: str = ""  # "Wide Margin", "Adequate", "Thin", "No Margin"
    summary: str = ""


@dataclass
class FinancialFlexibilityResult:
    """Financial flexibility and adaptive capacity analysis result."""
    cash_to_assets: Optional[float] = None  # Cash / TA
    cash_to_debt: Optional[float] = None  # Cash / Total Debt
    cash_to_revenue: Optional[float] = None  # Cash / Revenue
    fcf_to_revenue: Optional[float] = None  # FCF / Revenue
    fcf_margin: Optional[float] = None  # FCF / Revenue (alias)
    spare_borrowing_capacity: Optional[float] = None  # (TA * 0.50 - Debt) / TA
    unencumbered_assets: Optional[float] = None  # TA - TL
    financial_slack: Optional[float] = None  # (Cash + FCF) / Total Debt
    debt_headroom: Optional[float] = None  # Max additional debt at 3x EBITDA - current debt
    retained_earnings_ratio: Optional[float] = None  # RE / Equity
    flexibility_score: float = 0.0  # 0-10
    flexibility_grade: str = ""  # "Highly Flexible", "Flexible", "Constrained", "Rigid"
    summary: str = ""


@dataclass
class AltmanZScoreResult:
    """Altman Z-Score bankruptcy prediction result."""
    # Component ratios
    working_capital_to_assets: Optional[float] = None  # (CA - CL) / TA
    retained_earnings_to_assets: Optional[float] = None  # RE / TA
    ebit_to_assets: Optional[float] = None  # EBIT / TA
    equity_to_liabilities: Optional[float] = None  # Equity / TL
    revenue_to_assets: Optional[float] = None  # Revenue / TA
    # Weighted components
    x1_weighted: Optional[float] = None  # 1.2 * WC/TA
    x2_weighted: Optional[float] = None  # 1.4 * RE/TA
    x3_weighted: Optional[float] = None  # 3.3 * EBIT/TA
    x4_weighted: Optional[float] = None  # 0.6 * Equity/TL
    x5_weighted: Optional[float] = None  # 1.0 * Rev/TA
    z_score: Optional[float] = None  # Total Z-Score
    z_zone: str = ""  # "Safe", "Gray", "Distress"
    altman_score: float = 0.0  # 0-10 normalized score
    altman_grade: str = ""  # "Strong", "Adequate", "Watch", "Critical"
    summary: str = ""


@dataclass
class PiotroskiFScoreResult:
    """Piotroski F-Score value investing screen (adapted for single-period)."""
    # Profitability signals
    roa_positive: Optional[bool] = None  # NI / TA > 0
    ocf_positive: Optional[bool] = None  # OCF > 0
    accruals_negative: Optional[bool] = None  # OCF > NI (cash > accrual)
    # Leverage / Liquidity signals
    current_ratio_above_1: Optional[bool] = None  # CA / CL > 1
    low_leverage: Optional[bool] = None  # Debt/TA < 0.5
    # Efficiency signals
    gross_margin_healthy: Optional[bool] = None  # Gross Margin > 0.20
    asset_turnover_adequate: Optional[bool] = None  # Rev/TA > 0.5
    # Computed metrics
    roa: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_assets: Optional[float] = None
    gross_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    # Score
    f_score: int = 0  # 0-7 (adapted; original is 0-9 with prior-period deltas)
    f_score_max: int = 7  # Max possible
    piotroski_grade: str = ""  # "Strong Value", "Moderate Value", "Weak", "Avoid"
    summary: str = ""


@dataclass
class InterestCoverageResult:
    """Interest coverage and debt capacity analysis result."""
    ebit_coverage: Optional[float] = None  # EBIT / Interest
    ebitda_coverage: Optional[float] = None  # EBITDA / Interest
    debt_to_ebitda: Optional[float] = None  # Total Debt / EBITDA
    ocf_to_debt: Optional[float] = None  # OCF / Total Debt
    fcf_to_debt: Optional[float] = None  # FCF / Total Debt
    fixed_charge_coverage: Optional[float] = None  # (EBIT + Lease) / (Interest + Lease)
    max_debt_capacity: Optional[float] = None  # 3x EBITDA (theoretical)
    spare_debt_capacity: Optional[float] = None  # Max - Current Debt
    interest_to_revenue: Optional[float] = None  # Interest / Revenue
    debt_to_equity: Optional[float] = None  # Total Debt / Equity
    coverage_score: float = 0.0  # 0-10
    coverage_grade: str = ""  # "Excellent", "Adequate", "Strained", "Critical"
    summary: str = ""


@dataclass
class WACCResult:
    """Weighted Average Cost of Capital estimation from financial statements."""
    cost_of_debt: Optional[float] = None  # Interest / Debt
    after_tax_cost_of_debt: Optional[float] = None  # Rd * (1 - T)
    implied_cost_of_equity: Optional[float] = None  # Estimated from ROE/earnings yield
    effective_tax_rate: Optional[float] = None  # Tax proxy
    debt_weight: Optional[float] = None  # D / (D + E)
    equity_weight: Optional[float] = None  # E / (D + E)
    wacc: Optional[float] = None  # Blended cost of capital
    debt_to_total_capital: Optional[float] = None  # Debt / (Debt + Equity)
    equity_to_total_capital: Optional[float] = None  # Equity / (Debt + Equity)
    total_capital: Optional[float] = None  # Debt + Equity
    wacc_score: float = 0.0  # 0-10
    wacc_grade: str = ""  # "Excellent", "Good", "Fair", "Expensive"
    summary: str = ""


@dataclass
class EVAResult:
    """Economic Value Added analysis."""
    nopat: Optional[float] = None  # Net Operating Profit After Tax
    invested_capital: Optional[float] = None  # Total Assets - Current Liabilities (or Debt + Equity)
    wacc_used: Optional[float] = None  # WACC estimate used
    capital_charge: Optional[float] = None  # Invested Capital * WACC
    eva: Optional[float] = None  # NOPAT - Capital Charge
    eva_margin: Optional[float] = None  # EVA / Revenue
    roic: Optional[float] = None  # NOPAT / Invested Capital
    roic_wacc_spread: Optional[float] = None  # ROIC - WACC
    eva_score: float = 0.0  # 0-10
    eva_grade: str = ""  # "Value Creator", "Adequate", "Marginal", "Value Destroyer"
    summary: str = ""


@dataclass
class FCFYieldResult:
    """Free Cash Flow yield and quality metrics."""
    fcf: Optional[float] = None  # OCF - CapEx
    fcf_margin: Optional[float] = None  # FCF / Revenue
    fcf_to_net_income: Optional[float] = None  # FCF / NI (conversion quality)
    fcf_yield_on_capital: Optional[float] = None  # FCF / Invested Capital
    fcf_yield_on_equity: Optional[float] = None  # FCF / Equity
    fcf_to_debt: Optional[float] = None  # FCF / Total Debt
    capex_to_ocf: Optional[float] = None  # CapEx / OCF (reinvestment rate)
    capex_to_revenue: Optional[float] = None  # CapEx / Revenue (capital intensity)
    fcf_score: float = 0.0  # 0-10
    fcf_grade: str = ""  # "Strong", "Healthy", "Weak", "Negative"
    summary: str = ""


@dataclass
class CashConversionResult:
    """Phase 41: Cash Conversion Efficiency."""
    dso: Optional[float] = None  # Days Sales Outstanding
    dio: Optional[float] = None  # Days Inventory Outstanding
    dpo: Optional[float] = None  # Days Payable Outstanding
    ccc: Optional[float] = None  # Cash Conversion Cycle (DSO+DIO-DPO)
    cash_to_revenue: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    ocf_to_ebitda: Optional[float] = None
    cash_conversion_score: float = 0.0  # 0-10
    cash_conversion_grade: str = ""  # "Excellent", "Good", "Fair", "Poor"
    summary: str = ""


@dataclass
class ProfitRetentionPowerResult:
    """Phase 356: Profit Retention Power Analysis."""
    prp_ratio: Optional[float] = None
    re_to_equity: Optional[float] = None
    re_to_revenue: Optional[float] = None
    re_growth_capacity: Optional[float] = None
    retention_rate: Optional[float] = None
    prp_spread: Optional[float] = None
    prp_score: float = 0.0
    prp_grade: str = ""
    summary: str = ""


@dataclass
class EarningsToDebtResult:
    """Phase 353: Earnings To Debt Analysis."""
    etd_ratio: Optional[float] = None
    ni_to_interest: Optional[float] = None
    ni_to_liabilities: Optional[float] = None
    earnings_yield_on_debt: Optional[float] = None
    debt_years_from_earnings: Optional[float] = None
    etd_spread: Optional[float] = None
    etd_score: float = 0.0
    etd_grade: str = ""
    summary: str = ""


@dataclass
class RevenueGrowthResult:
    """Phase 350: Revenue Growth Capacity Analysis."""
    rg_capacity: Optional[float] = None
    roe: Optional[float] = None
    plowback: Optional[float] = None
    sustainable_growth: Optional[float] = None
    revenue_per_asset: Optional[float] = None
    rg_spread: Optional[float] = None
    rg_score: float = 0.0
    rg_grade: str = ""
    summary: str = ""


@dataclass
class OperatingMarginResult:
    """Phase 349: Operating Margin Analysis."""
    operating_margin: Optional[float] = None
    oi_to_revenue: Optional[float] = None
    ebit_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    margin_trend: Optional[float] = None
    opm_spread: Optional[float] = None
    opm_score: float = 0.0
    opm_grade: str = ""
    summary: str = ""


@dataclass
class DebtToEquityResult:
    """Phase 348: Debt To Equity Analysis."""
    dte_ratio: Optional[float] = None
    td_to_te: Optional[float] = None
    lt_debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    equity_multiplier: Optional[float] = None
    dte_spread: Optional[float] = None
    dte_score: float = 0.0
    dte_grade: str = ""
    summary: str = ""


@dataclass
class CashFlowToDebtResult:
    """Phase 347: Cash Flow To Debt Analysis."""
    cf_to_debt: Optional[float] = None
    ocf_to_td: Optional[float] = None
    fcf_to_td: Optional[float] = None
    debt_payback_years: Optional[float] = None
    ocf_to_interest: Optional[float] = None
    cf_debt_spread: Optional[float] = None
    cfd_score: float = 0.0
    cfd_grade: str = ""
    summary: str = ""


@dataclass
class NetWorthGrowthResult:
    """Phase 346: Net Worth Growth Analysis."""
    nw_growth_ratio: Optional[float] = None
    re_to_equity: Optional[float] = None
    equity_to_assets: Optional[float] = None
    ni_to_equity: Optional[float] = None
    plowback_rate: Optional[float] = None
    nw_spread: Optional[float] = None
    nwg_score: float = 0.0
    nwg_grade: str = ""
    summary: str = ""


@dataclass
class AssetLightnessResult:
    """Phase 341: Asset Lightness Analysis."""
    lightness_ratio: Optional[float] = None
    ca_to_ta: Optional[float] = None
    revenue_to_assets: Optional[float] = None
    intangible_intensity: Optional[float] = None
    fixed_asset_ratio: Optional[float] = None
    lightness_spread: Optional[float] = None
    alt_score: float = 0.0
    alt_grade: str = ""
    summary: str = ""


@dataclass
class InternalGrowthRateResult:
    """Phase 337: Internal Growth Rate Analysis."""
    igr: Optional[float] = None
    roa: Optional[float] = None
    retention_ratio: Optional[float] = None
    roa_times_b: Optional[float] = None
    sustainable_growth: Optional[float] = None
    growth_capacity: Optional[float] = None
    igr_score: float = 0.0
    igr_grade: str = ""
    summary: str = ""


@dataclass
class OperatingExpenseRatioResult:
    """Phase 330: Operating Expense Ratio Analysis."""
    opex_ratio: Optional[float] = None
    opex_per_revenue: Optional[float] = None
    opex_to_gross_profit: Optional[float] = None
    opex_to_ebitda: Optional[float] = None
    opex_coverage: Optional[float] = None
    efficiency_gap: Optional[float] = None
    oer_score: float = 0.0
    oer_grade: str = ""
    summary: str = ""


@dataclass
class NoncurrentAssetRatioResult:
    """Phase 327: Noncurrent Asset Ratio Analysis."""
    nca_ratio: Optional[float] = None
    current_asset_ratio: Optional[float] = None
    nca_to_equity: Optional[float] = None
    nca_to_debt: Optional[float] = None
    asset_structure_spread: Optional[float] = None
    liquidity_complement: Optional[float] = None
    nar_score: float = 0.0
    nar_grade: str = ""
    summary: str = ""


@dataclass
class PayoutResilienceResult:
    """Phase 317: Payout Resilience Analysis."""
    div_to_ni: Optional[float] = None
    div_to_ocf: Optional[float] = None
    div_to_revenue: Optional[float] = None
    div_to_ebitda: Optional[float] = None
    payout_ratio: Optional[float] = None
    resilience_buffer: Optional[float] = None
    prs_score: float = 0.0
    prs_grade: str = ""
    summary: str = ""


@dataclass
class DebtBurdenIndexResult:
    """Phase 314: Debt Burden Index Analysis."""
    debt_to_ebitda: Optional[float] = None
    debt_to_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_revenue: Optional[float] = None
    debt_ratio: Optional[float] = None
    burden_intensity: Optional[float] = None
    dbi_score: float = 0.0
    dbi_grade: str = ""
    summary: str = ""


@dataclass
class InventoryCoverageResult:
    """Phase 309: Inventory Coverage Analysis."""
    inventory_to_revenue: Optional[float] = None
    inventory_to_cogs: Optional[float] = None
    inventory_to_assets: Optional[float] = None
    inventory_to_current_assets: Optional[float] = None
    inventory_days: Optional[float] = None
    inventory_buffer: Optional[float] = None
    icv_score: float = 0.0
    icv_grade: str = ""
    summary: str = ""


@dataclass
class CapexToRevenueResult:
    """Phase 307: CapEx to Revenue Analysis."""
    capex_to_revenue: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    capex_to_ebitda: Optional[float] = None
    capex_to_assets: Optional[float] = None
    investment_intensity: Optional[float] = None
    capex_yield: Optional[float] = None
    ctr_score: float = 0.0
    ctr_grade: str = ""
    summary: str = ""


@dataclass
class InventoryHoldingCostResult:
    """Phase 294: Inventory Holding Cost Analysis."""
    inventory_to_revenue: Optional[float] = None
    inventory_to_current_assets: Optional[float] = None
    inventory_to_total_assets: Optional[float] = None
    inventory_days: Optional[float] = None
    inventory_carrying_cost: Optional[float] = None
    inventory_intensity: Optional[float] = None
    ihc_score: float = 0.0
    ihc_grade: str = ""
    summary: str = ""


@dataclass
class FundingMixBalanceResult:
    """Phase 293: Funding Mix Balance Analysis."""
    equity_to_total_capital: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_total_capital: Optional[float] = None
    equity_multiplier: Optional[float] = None
    leverage_headroom: Optional[float] = None
    funding_stability: Optional[float] = None
    fmb_score: float = 0.0
    fmb_grade: str = ""
    summary: str = ""


@dataclass
class ExpenseRatioDisciplineResult:
    """Phase 292: Expense Ratio Discipline Analysis."""
    opex_to_revenue: Optional[float] = None
    cogs_to_revenue: Optional[float] = None
    sga_to_revenue: Optional[float] = None
    total_expense_ratio: Optional[float] = None
    operating_margin: Optional[float] = None
    expense_efficiency: Optional[float] = None
    erd_score: float = 0.0
    erd_grade: str = ""
    summary: str = ""


@dataclass
class RevenueCashRealizationResult:
    """Phase 291: Revenue Cash Realization Analysis."""
    cash_to_revenue: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    collection_rate: Optional[float] = None
    revenue_cash_gap: Optional[float] = None
    cash_conversion_speed: Optional[float] = None
    revenue_quality_ratio: Optional[float] = None
    rcr_score: float = 0.0
    rcr_grade: str = ""
    summary: str = ""


@dataclass
class NetDebtPositionResult:
    """Phase 286: Net Debt Position Analysis."""
    net_debt: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    net_debt_to_equity: Optional[float] = None
    net_debt_to_assets: Optional[float] = None
    cash_to_debt: Optional[float] = None
    net_debt_to_ocf: Optional[float] = None
    ndp_score: float = 0.0
    ndp_grade: str = ""
    summary: str = ""


@dataclass
class LiabilityCoverageStrengthResult:
    """Phase 281: Liability Coverage Strength Analysis."""
    ocf_to_liabilities: Optional[float] = None
    ebitda_to_liabilities: Optional[float] = None
    assets_to_liabilities: Optional[float] = None
    equity_to_liabilities: Optional[float] = None
    liability_to_revenue: Optional[float] = None
    liability_burden: Optional[float] = None
    lcs_score: float = 0.0
    lcs_grade: str = ""
    summary: str = ""


@dataclass
class CapitalAdequacyResult:
    """Phase 279: Capital Adequacy Analysis."""
    equity_ratio: Optional[float] = None
    equity_to_debt: Optional[float] = None
    retained_to_equity: Optional[float] = None
    equity_to_liabilities: Optional[float] = None
    tangible_equity_ratio: Optional[float] = None
    capital_buffer: Optional[float] = None
    caq_score: float = 0.0
    caq_grade: str = ""
    summary: str = ""


@dataclass
class OperatingIncomeQualityResult:
    """Phase 275: Operating Income Quality Analysis."""
    oi_to_revenue: Optional[float] = None
    oi_to_ebitda: Optional[float] = None
    oi_to_ocf: Optional[float] = None
    oi_to_total_assets: Optional[float] = None
    operating_spread: Optional[float] = None
    oi_cash_backing: Optional[float] = None
    oiq_score: float = 0.0
    oiq_grade: str = ""
    summary: str = ""


@dataclass
class EbitdaToDebtCoverageResult:
    """Phase 274: EBITDA-to-Debt Coverage Analysis."""
    ebitda_to_debt: Optional[float] = None
    ebitda_to_interest: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    ebitda_to_total_liabilities: Optional[float] = None
    debt_service_buffer: Optional[float] = None
    leverage_headroom: Optional[float] = None
    etdc_score: float = 0.0
    etdc_grade: str = ""
    summary: str = ""


@dataclass
class DebtQualityResult:
    """Phase 267: Debt Quality Assessment."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    long_term_debt_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    debt_cost: Optional[float] = None
    dq_score: float = 0.0
    dq_grade: str = ""
    summary: str = ""


@dataclass
class FixedAssetProductivityResult:
    """Phase 263: Fixed Asset Productivity Analysis."""
    fixed_asset_turnover: Optional[float] = None
    revenue_per_fixed_asset: Optional[float] = None
    fixed_to_total_assets: Optional[float] = None
    capex_to_fixed_assets: Optional[float] = None
    depreciation_to_fixed: Optional[float] = None
    net_fixed_asset_intensity: Optional[float] = None
    fap_score: float = 0.0
    fap_grade: str = ""
    summary: str = ""


@dataclass
class DepreciationBurdenResult:
    """Phase 259: Depreciation Burden Analysis."""
    dep_to_revenue: Optional[float] = None
    dep_to_assets: Optional[float] = None
    dep_to_ebitda: Optional[float] = None
    dep_to_gross_profit: Optional[float] = None
    ebitda_to_ebit_spread: Optional[float] = None
    asset_age_proxy: Optional[float] = None
    db_score: float = 0.0
    db_grade: str = ""
    summary: str = ""


@dataclass
class DebtToCapitalResult:
    """Phase 258: Debt-to-Capital Analysis."""
    debt_to_capital: Optional[float] = None
    debt_to_equity: Optional[float] = None
    long_term_debt_to_capital: Optional[float] = None
    equity_ratio: Optional[float] = None
    net_debt_to_capital: Optional[float] = None
    financial_risk_index: Optional[float] = None
    dtc_score: float = 0.0
    dtc_grade: str = ""
    summary: str = ""


@dataclass
class DividendPayoutResult:
    """Phase 251: Dividend Payout Ratio Analysis."""
    div_to_ni: Optional[float] = None
    retention_ratio: Optional[float] = None
    div_to_ocf: Optional[float] = None
    div_to_fcf: Optional[float] = None
    div_to_revenue: Optional[float] = None
    div_coverage: Optional[float] = None
    dpr_score: float = 0.0
    dpr_grade: str = ""
    summary: str = ""


@dataclass
class OperatingCashFlowRatioResult:
    """Phase 249: Operating Cash Flow Ratio Analysis."""
    ocf_to_cl: Optional[float] = None
    ocf_to_tl: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    ocf_to_ni: Optional[float] = None
    ocf_to_debt: Optional[float] = None
    ocf_margin: Optional[float] = None
    ocfr_score: float = 0.0
    ocfr_grade: str = ""
    summary: str = ""


@dataclass
class CashConversionCycleResult:
    """Phase 248: Cash Conversion Cycle Analysis."""
    dso: Optional[float] = None
    dio: Optional[float] = None
    dpo: Optional[float] = None
    ccc: Optional[float] = None
    ccc_to_revenue: Optional[float] = None
    working_cap_days: Optional[float] = None
    ccc_score: float = 0.0
    ccc_grade: str = ""
    summary: str = ""


@dataclass
class InventoryTurnoverResult:
    """Phase 247: Inventory Turnover Analysis."""
    cogs_to_inv: Optional[float] = None
    dio: Optional[float] = None
    inv_to_ca: Optional[float] = None
    inv_to_ta: Optional[float] = None
    inv_to_revenue: Optional[float] = None
    inv_velocity: Optional[float] = None
    ito_score: float = 0.0
    ito_grade: str = ""
    summary: str = ""


@dataclass
class PayablesTurnoverResult:
    """Phase 246: Payables Turnover Analysis."""
    cogs_to_ap: Optional[float] = None
    dpo: Optional[float] = None
    ap_to_cl: Optional[float] = None
    ap_to_tl: Optional[float] = None
    ap_to_cogs: Optional[float] = None
    payment_velocity: Optional[float] = None
    pto_score: float = 0.0
    pto_grade: str = ""
    summary: str = ""


@dataclass
class ReceivablesTurnoverResult:
    """Phase 245: Receivables Turnover Analysis."""
    rev_to_ar: Optional[float] = None
    dso: Optional[float] = None
    ar_to_ca: Optional[float] = None
    ar_to_ta: Optional[float] = None
    ar_to_revenue: Optional[float] = None
    collection_efficiency: Optional[float] = None
    rto_score: float = 0.0
    rto_grade: str = ""
    summary: str = ""


@dataclass
class CashConversionEfficiencyResult:
    """Phase 237: Cash Conversion Efficiency Analysis."""
    ocf_to_oi: Optional[float] = None
    ocf_to_ni: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    ocf_to_ebitda: Optional[float] = None
    fcf_to_oi: Optional[float] = None
    cash_to_oi: Optional[float] = None
    cce_score: float = 0.0
    cce_grade: str = ""
    summary: str = ""


@dataclass
class FixedCostLeverageRatioResult:
    """Phase 236: Fixed Cost Leverage Ratio Analysis."""
    dol: Optional[float] = None
    contribution_margin: Optional[float] = None
    oi_to_revenue: Optional[float] = None
    cogs_to_revenue: Optional[float] = None
    opex_to_revenue: Optional[float] = None
    breakeven_proxy: Optional[float] = None
    fclr_score: float = 0.0
    fclr_grade: str = ""
    summary: str = ""


@dataclass
class RevenueQualityIndexResult:
    """Phase 232: Revenue Quality Index Analysis."""
    ocf_to_revenue: Optional[float] = None
    gross_margin: Optional[float] = None
    ni_to_revenue: Optional[float] = None
    ebitda_to_revenue: Optional[float] = None
    ar_to_revenue: Optional[float] = None
    cash_to_revenue: Optional[float] = None
    rqi_score: float = 0.0
    rqi_grade: str = ""
    summary: str = ""


@dataclass
class FixedAssetUtilizationResult:
    """Phase 221: Fixed Asset Utilization Analysis."""
    fixed_asset_turnover: Optional[float] = None
    depreciation_to_revenue: Optional[float] = None
    capex_to_depreciation: Optional[float] = None
    capex_to_total_assets: Optional[float] = None
    fixed_to_total_assets: Optional[float] = None
    depreciation_to_total_assets: Optional[float] = None
    fau_score: float = 0.0
    fau_grade: str = ""
    summary: str = ""


@dataclass
class CostControlResult:
    """Phase 215: Cost Control Analysis."""
    opex_to_revenue: Optional[float] = None
    cogs_to_revenue: Optional[float] = None
    sga_to_revenue: Optional[float] = None
    operating_margin: Optional[float] = None
    opex_to_gross_profit: Optional[float] = None
    ebitda_margin: Optional[float] = None
    cc_score: float = 0.0
    cc_grade: str = ""
    summary: str = ""


@dataclass
class ValuationSignalResult:
    """Phase 212: Valuation Signal Analysis."""
    ev_to_ebitda: Optional[float] = None
    price_to_earnings: Optional[float] = None
    price_to_book: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    earnings_yield: Optional[float] = None
    fcf_yield: Optional[float] = None
    vsg_score: float = 0.0
    vsg_grade: str = ""
    summary: str = ""


@dataclass
class CapitalDisciplineResult:
    """Phase 211: Capital Discipline Analysis."""
    retained_to_equity: Optional[float] = None
    retained_to_assets: Optional[float] = None
    dividend_payout: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    debt_to_equity: Optional[float] = None
    ocf_to_debt: Optional[float] = None
    cd_score: float = 0.0
    cd_grade: str = ""
    summary: str = ""


@dataclass
class ResourceOptimizationResult:
    """Phase 210: Resource Optimization Analysis."""
    fcf_to_revenue: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    capex_to_revenue: Optional[float] = None
    ocf_to_assets: Optional[float] = None
    fcf_to_assets: Optional[float] = None
    dividend_payout_ratio: Optional[float] = None
    ro_score: float = 0.0
    ro_grade: str = ""
    summary: str = ""


@dataclass
class FinancialProductivityResult:
    """Phase 205: Financial Productivity Analysis."""
    revenue_per_asset: Optional[float] = None
    revenue_per_equity: Optional[float] = None
    ebitda_per_employee_proxy: Optional[float] = None
    operating_income_per_asset: Optional[float] = None
    net_income_per_revenue: Optional[float] = None
    cash_flow_per_asset: Optional[float] = None
    fp_score: float = 0.0
    fp_grade: str = ""
    summary: str = ""


@dataclass
class EquityPreservationResult:
    """Phase 198: Equity Preservation Analysis."""
    equity_to_assets: Optional[float] = None
    retained_to_equity: Optional[float] = None
    equity_growth_capacity: Optional[float] = None
    equity_to_liabilities: Optional[float] = None
    tangible_equity_ratio: Optional[float] = None
    equity_per_revenue: Optional[float] = None
    ep_score: float = 0.0
    ep_grade: str = ""
    summary: str = ""


@dataclass
class DebtManagementResult:
    """Phase 197: Debt Management Analysis."""
    debt_to_operating_income: Optional[float] = None
    debt_to_ocf: Optional[float] = None
    interest_to_revenue: Optional[float] = None
    debt_to_gross_profit: Optional[float] = None
    net_debt_ratio: Optional[float] = None
    debt_coverage_ratio: Optional[float] = None
    dm_score: float = 0.0
    dm_grade: str = ""
    summary: str = ""


@dataclass
class IncomeRetentionResult:
    """Phase 196: Income Retention Analysis."""
    net_to_gross_ratio: Optional[float] = None
    net_to_operating_ratio: Optional[float] = None
    net_to_ebitda_ratio: Optional[float] = None
    retention_rate: Optional[float] = None
    income_to_asset_generation: Optional[float] = None
    after_tax_margin: Optional[float] = None
    ir_score: float = 0.0
    ir_grade: str = ""
    summary: str = ""


@dataclass
class OperationalEfficiencyResult:
    """Phase 195: Operational Efficiency Analysis."""
    oi_margin: Optional[float] = None
    revenue_to_assets: Optional[float] = None
    gross_profit_per_asset: Optional[float] = None
    opex_efficiency: Optional[float] = None
    asset_utilization: Optional[float] = None
    income_per_liability: Optional[float] = None
    oe_score: float = 0.0
    oe_grade: str = ""
    summary: str = ""


@dataclass
class OperatingMomentumResult:
    """Phase 191: Operating Momentum Analysis."""
    ebitda_margin: Optional[float] = None
    ebit_margin: Optional[float] = None
    ocf_margin: Optional[float] = None
    gross_to_operating_conversion: Optional[float] = None
    operating_cash_conversion: Optional[float] = None
    overhead_absorption: Optional[float] = None
    om_score: float = 0.0
    om_grade: str = ""
    summary: str = ""


@dataclass
class PayoutDisciplineResult:
    """Phase 185: Payout Discipline Analysis."""
    cash_dividend_coverage: Optional[float] = None
    payout_ratio: Optional[float] = None
    retention_ratio: Optional[float] = None
    dividend_to_ocf: Optional[float] = None
    capex_priority: Optional[float] = None
    free_cash_after_dividends: Optional[float] = None
    pd_score: float = 0.0
    pd_grade: str = ""
    summary: str = ""


@dataclass
class IncomeResilienceResult:
    """Phase 184: Income Resilience Analysis."""
    operating_income_stability: Optional[float] = None
    ebit_coverage: Optional[float] = None
    net_margin_resilience: Optional[float] = None
    depreciation_buffer: Optional[float] = None
    tax_interest_drag: Optional[float] = None
    ebitda_cushion: Optional[float] = None
    ir_score: float = 0.0
    ir_grade: str = ""
    summary: str = ""


@dataclass
class StructuralStrengthResult:
    """Phase 182: Structural Strength Analysis."""
    equity_multiplier: Optional[float] = None
    debt_to_equity: Optional[float] = None
    liability_composition: Optional[float] = None
    equity_cushion: Optional[float] = None
    fixed_asset_coverage: Optional[float] = None
    financial_leverage_ratio: Optional[float] = None
    ss_score: float = 0.0
    ss_grade: str = ""
    summary: str = ""


@dataclass
class ProfitConversionResult:
    """Phase 179: Profit Conversion Analysis."""
    gross_conversion: Optional[float] = None
    operating_conversion: Optional[float] = None
    net_conversion: Optional[float] = None
    ebitda_conversion: Optional[float] = None
    cash_conversion: Optional[float] = None
    profit_to_cash_ratio: Optional[float] = None
    pc_score: float = 0.0
    pc_grade: str = ""
    summary: str = ""


@dataclass
class AssetDeploymentEfficiencyResult:
    """Phase 172: Asset Deployment Efficiency Analysis."""
    asset_turnover: Optional[float] = None
    fixed_asset_leverage: Optional[float] = None
    asset_income_yield: Optional[float] = None
    asset_cash_yield: Optional[float] = None
    inventory_velocity: Optional[float] = None
    receivables_velocity: Optional[float] = None
    ade_score: float = 0.0
    ade_grade: str = ""
    summary: str = ""


@dataclass
class ProfitSustainabilityResult:
    """Phase 171: Profit Sustainability Analysis."""
    profit_cash_backing: Optional[float] = None
    profit_margin_depth: Optional[float] = None
    profit_reinvestment: Optional[float] = None
    profit_to_asset: Optional[float] = None
    profit_stability_proxy: Optional[float] = None
    profit_leverage: Optional[float] = None
    ps_score: float = 0.0
    ps_grade: str = ""
    summary: str = ""


@dataclass
class DebtDisciplineResult:
    """Phase 170: Debt Discipline Analysis."""
    debt_prudence_ratio: Optional[float] = None
    debt_servicing_power: Optional[float] = None
    debt_coverage_spread: Optional[float] = None
    debt_to_equity_leverage: Optional[float] = None
    interest_absorption: Optional[float] = None
    debt_repayment_capacity: Optional[float] = None
    dd_score: float = 0.0
    dd_grade: str = ""
    summary: str = ""


@dataclass
class CapitalPreservationResult:
    """Phase 168: Capital Preservation Analysis."""
    retained_earnings_power: Optional[float] = None
    capital_erosion_rate: Optional[float] = None
    asset_integrity_ratio: Optional[float] = None
    operating_capital_ratio: Optional[float] = None
    net_worth_growth_proxy: Optional[float] = None
    capital_buffer: Optional[float] = None
    cp_score: float = 0.0
    cp_grade: str = ""
    summary: str = ""


@dataclass
class ObligationCoverageResult:
    """Phase 160: Obligation Coverage Analysis."""
    ebitda_interest_coverage: Optional[float] = None
    cash_interest_coverage: Optional[float] = None
    debt_amortization_capacity: Optional[float] = None
    fixed_charge_coverage: Optional[float] = None
    debt_burden_ratio: Optional[float] = None
    interest_to_revenue: Optional[float] = None
    oc_score: float = 0.0
    oc_grade: str = ""
    summary: str = ""


@dataclass
class InternalGrowthCapacityResult:
    """Phase 159: Internal Growth Capacity Analysis."""
    sustainable_growth_rate: Optional[float] = None
    internal_growth_rate: Optional[float] = None
    plowback_ratio: Optional[float] = None
    reinvestment_rate: Optional[float] = None
    growth_financing_ratio: Optional[float] = None
    equity_growth_rate: Optional[float] = None
    igc_score: float = 0.0
    igc_grade: str = ""
    summary: str = ""


@dataclass
class LiabilityManagementResult:
    """Phase 146: Liability Management Analysis."""
    liability_to_assets: Optional[float] = None
    liability_to_equity: Optional[float] = None
    current_liability_ratio: Optional[float] = None
    liability_coverage: Optional[float] = None
    liability_to_revenue: Optional[float] = None
    net_liability: Optional[float] = None
    lm_score: float = 0.0
    lm_grade: str = ""
    summary: str = ""


@dataclass
class RevenuePredictabilityResult:
    """Phase 142: Revenue Predictability Analysis."""
    revenue_to_assets: Optional[float] = None
    revenue_to_equity: Optional[float] = None
    revenue_to_debt: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    rp_score: float = 0.0
    rp_grade: str = ""
    summary: str = ""


@dataclass
class EquityReinvestmentResult:
    """Phase 139: Equity Reinvestment Analysis."""
    retention_ratio: Optional[float] = None
    reinvestment_rate: Optional[float] = None
    equity_growth_proxy: Optional[float] = None
    plowback_to_assets: Optional[float] = None
    internal_growth_rate: Optional[float] = None
    dividend_coverage: Optional[float] = None
    er_score: float = 0.0
    er_grade: str = ""
    summary: str = ""


@dataclass
class FixedAssetEfficiencyResult:
    """Phase 138: Fixed Asset Efficiency Analysis."""
    fixed_asset_ratio: Optional[float] = None
    fixed_asset_turnover: Optional[float] = None
    fixed_to_equity: Optional[float] = None
    fixed_asset_coverage: Optional[float] = None
    depreciation_to_fixed: Optional[float] = None
    capex_to_fixed: Optional[float] = None
    fae_score: float = 0.0
    fae_grade: str = ""
    summary: str = ""


@dataclass
class IncomeStabilityResult:
    """Phase 134: Income Stability Analysis."""
    net_income_margin: Optional[float] = None
    retained_earnings_ratio: Optional[float] = None
    operating_income_cushion: Optional[float] = None
    net_to_gross_ratio: Optional[float] = None
    ebitda_margin: Optional[float] = None
    income_resilience: Optional[float] = None
    is_score: float = 0.0
    is_grade: str = ""
    summary: str = ""


@dataclass
class DefensivePostureResult:
    """Phase 133: Defensive Posture Analysis."""
    defensive_interval: Optional[float] = None
    cash_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_flow_coverage: Optional[float] = None
    equity_buffer: Optional[float] = None
    debt_shield: Optional[float] = None
    dp_score: float = 0.0
    dp_grade: str = ""
    summary: str = ""


@dataclass
class FundingEfficiencyResult:
    """Phase 131: Funding Efficiency Analysis."""
    debt_to_capitalization: Optional[float] = None
    equity_multiplier: Optional[float] = None
    interest_coverage_ebitda: Optional[float] = None
    cost_of_debt: Optional[float] = None
    weighted_funding_cost: Optional[float] = None
    funding_spread: Optional[float] = None
    fe_score: float = 0.0
    fe_grade: str = ""
    summary: str = ""


@dataclass
class CashFlowStabilityResult:
    """Phase 125: Cash Flow Stability Analysis."""
    ocf_margin: Optional[float] = None
    ocf_to_ebitda: Optional[float] = None
    ocf_to_debt_service: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    dividend_coverage: Optional[float] = None
    cash_flow_sufficiency: Optional[float] = None
    cfs_score: float = 0.0
    cfs_grade: str = ""
    summary: str = ""


@dataclass
class IncomeQualityResult:
    """Phase 124: Income Quality Analysis."""
    ocf_to_net_income: Optional[float] = None
    accruals_ratio: Optional[float] = None
    cash_earnings_ratio: Optional[float] = None
    non_cash_ratio: Optional[float] = None
    earnings_persistence: Optional[float] = None
    operating_income_ratio: Optional[float] = None
    iq_score: float = 0.0
    iq_grade: str = ""
    summary: str = ""


@dataclass
class ReceivablesManagementResult:
    """Phase 114: Receivables Management Analysis."""
    dso: Optional[float] = None
    ar_to_revenue: Optional[float] = None
    ar_to_current_assets: Optional[float] = None
    receivables_turnover: Optional[float] = None
    collection_effectiveness: Optional[float] = None
    ar_concentration: Optional[float] = None
    rm_score: float = 0.0
    rm_grade: str = ""
    summary: str = ""


@dataclass
class SolvencyDepthResult:
    """Phase 109: Solvency Depth Analysis."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    equity_to_assets: Optional[float] = None
    interest_coverage_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    financial_leverage: Optional[float] = None
    sd_score: float = 0.0
    sd_grade: str = ""
    summary: str = ""


@dataclass
class OperationalLeverageDepthResult:
    """Phase 105: Operational Leverage Depth Analysis."""
    fixed_cost_ratio: Optional[float] = None
    variable_cost_ratio: Optional[float] = None
    contribution_margin: Optional[float] = None
    dol_proxy: Optional[float] = None
    breakeven_coverage: Optional[float] = None
    cost_flexibility: Optional[float] = None
    old_score: float = 0.0
    old_grade: str = ""
    summary: str = ""


@dataclass
class ProfitabilityDepthResult:
    """Phase 103: Profitability Depth Analysis."""
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    net_margin: Optional[float] = None
    margin_spread: Optional[float] = None
    profit_retention_ratio: Optional[float] = None
    pd_score: float = 0.0
    pd_grade: str = ""
    summary: str = ""


@dataclass
class RevenueEfficiencyResult:
    """Phase 102: Revenue Efficiency Analysis."""
    revenue_per_asset: Optional[float] = None
    cash_conversion_efficiency: Optional[float] = None
    gross_margin_efficiency: Optional[float] = None
    operating_leverage_ratio: Optional[float] = None
    revenue_to_equity: Optional[float] = None
    net_revenue_retention: Optional[float] = None
    rev_eff_score: float = 0.0
    rev_eff_grade: str = ""
    summary: str = ""


@dataclass
class DebtCompositionResult:
    """Phase 101: Debt Composition Analysis."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    long_term_debt_ratio: Optional[float] = None
    interest_burden: Optional[float] = None
    debt_cost_ratio: Optional[float] = None
    debt_coverage_margin: Optional[float] = None
    dco_score: float = 0.0
    dco_grade: str = ""
    summary: str = ""


@dataclass
class OperationalRiskResult:
    """Phase 92: Operational Risk Analysis."""
    operating_leverage: Optional[float] = None
    cost_rigidity: Optional[float] = None
    breakeven_ratio: Optional[float] = None
    margin_of_safety: Optional[float] = None
    cash_burn_ratio: Optional[float] = None
    risk_buffer: Optional[float] = None
    or_score: float = 0.0
    or_grade: str = ""
    summary: str = ""


@dataclass
class FinancialHealthScoreResult:
    """Phase 90: Financial Health Score Analysis."""
    profitability_score: Optional[float] = None
    liquidity_score: Optional[float] = None
    solvency_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    coverage_score: Optional[float] = None
    composite_score: Optional[float] = None
    fh_score: float = 0.0
    fh_grade: str = ""
    summary: str = ""


@dataclass
class AssetQualityResult:
    """Phase 86: Asset Quality Analysis."""
    tangible_asset_ratio: Optional[float] = None       # (TA - Intangibles) / TA
    fixed_asset_ratio: Optional[float] = None          # Fixed Assets / TA (approx TA - CA)
    current_asset_ratio: Optional[float] = None        # CA / TA
    cash_to_current_assets: Optional[float] = None     # Cash / CA
    receivables_to_assets: Optional[float] = None      # AR / TA
    inventory_to_assets: Optional[float] = None        # Inv / TA
    aq_score: float = 0.0
    aq_grade: str = ""
    summary: str = ""


@dataclass
class FinancialResilienceResult:
    """Phase 82: Financial Resilience Analysis."""
    cash_to_assets: Optional[float] = None              # Cash / Total Assets
    cash_to_debt: Optional[float] = None                # Cash / Total Debt
    operating_cash_coverage: Optional[float] = None     # OCF / Total Liabilities
    interest_coverage_cash: Optional[float] = None      # OCF / Interest Expense
    free_cash_margin: Optional[float] = None            # FCF / Revenue
    resilience_buffer: Optional[float] = None           # (Cash + OCF) / Annual OpEx
    fr_score: float = 0.0
    fr_grade: str = ""
    summary: str = ""


@dataclass
class EquityMultiplierResult:
    """Phase 81: Equity Multiplier Analysis."""
    equity_multiplier: Optional[float] = None        # TA / Equity
    debt_ratio: Optional[float] = None               # TL / TA
    equity_ratio: Optional[float] = None             # Equity / TA
    financial_leverage_index: Optional[float] = None  # ROE / ROA
    dupont_roe: Optional[float] = None               # Margin * Turnover * EM
    leverage_spread: Optional[float] = None          # ROA - Cost of Debt
    em_score: float = 0.0
    em_grade: str = ""
    summary: str = ""


@dataclass
class DefensiveIntervalResult:
    """Phase 80: Defensive Interval Analysis."""
    defensive_interval_days: Optional[float] = None   # Liquid Assets / Daily OpEx
    cash_interval_days: Optional[float] = None        # Cash / Daily OpEx
    liquid_assets_ratio: Optional[float] = None       # Liquid Assets / Total Assets
    days_cash_on_hand: Optional[float] = None         # Cash / (Annual OpEx / 365)
    liquid_reserve_adequacy: Optional[float] = None   # Liquid Assets / CL
    operating_expense_coverage: Optional[float] = None  # Liquid Assets / Annual OpEx
    di_score: float = 0.0
    di_grade: str = ""
    summary: str = ""


@dataclass
class CashBurnResult:
    """Phase 79: Cash Burn Analysis."""
    ocf_margin: Optional[float] = None              # OCF / Revenue
    capex_intensity: Optional[float] = None         # CapEx / Revenue
    fcf_margin: Optional[float] = None              # (OCF - CapEx) / Revenue
    cash_self_sufficiency: Optional[float] = None   # OCF / (CapEx + Div)
    cash_runway_months: Optional[float] = None      # Cash / Monthly Burn (if burning)
    net_cash_position: Optional[float] = None       # Cash - Total Debt
    cb_score: float = 0.0
    cb_grade: str = ""
    summary: str = ""


@dataclass
class ProfitRetentionResult:
    """Phase 78: Profit Retention Analysis."""
    retention_ratio: Optional[float] = None         # (NI - Div) / NI
    payout_ratio: Optional[float] = None            # Div / NI
    re_to_equity: Optional[float] = None            # Retained Earnings / Equity
    sustainable_growth_rate: Optional[float] = None  # ROE * retention
    internal_growth_rate: Optional[float] = None     # ROA * retention / (1 - ROA * retention)
    plowback_amount: Optional[float] = None          # NI - Dividends
    pr_score: float = 0.0
    pr_grade: str = ""
    summary: str = ""


@dataclass
class DebtServiceCoverageResult:
    """Phase 76: Debt Service Coverage Analysis."""
    dscr: Optional[float] = None                   # EBITDA / Total Debt Service
    ocf_to_debt_service: Optional[float] = None    # OCF / Total Debt Service
    ebitda_to_interest: Optional[float] = None     # EBITDA / IE (interest coverage)
    fcf_to_debt_service: Optional[float] = None    # FCF / Total Debt Service
    debt_service_to_revenue: Optional[float] = None  # Total Debt Service / Revenue
    coverage_cushion: Optional[float] = None       # DSCR - 1.0 (excess coverage)
    dsc_score: float = 0.0
    dsc_grade: str = ""
    summary: str = ""


@dataclass
class CapitalAllocationResult:
    """Phase 73: Capital Allocation Analysis."""
    capex_to_revenue: Optional[float] = None          # CapEx / Revenue
    capex_to_ocf: Optional[float] = None              # CapEx / OCF (reinvestment rate)
    shareholder_return_ratio: Optional[float] = None   # (Div+BB) / NI
    reinvestment_rate: Optional[float] = None          # CapEx / Depreciation
    fcf_yield: Optional[float] = None                  # FCF / Revenue
    total_payout_to_fcf: Optional[float] = None        # (Div+BB) / FCF
    ca_score: float = 0.0
    ca_grade: str = ""
    summary: str = ""


@dataclass
class TaxEfficiencyResult:
    """Phase 70: Tax Efficiency Analysis."""
    effective_tax_rate: Optional[float] = None        # TaxExp / PreTaxIncome
    tax_to_revenue: Optional[float] = None            # TaxExp / Revenue
    tax_to_ebitda: Optional[float] = None             # TaxExp / EBITDA
    after_tax_margin: Optional[float] = None          # NI / Revenue
    tax_shield_ratio: Optional[float] = None          # (IE * ETR) / NI
    pretax_to_ebit: Optional[float] = None            # PreTaxIncome / EBIT
    te_score: float = 0.0
    te_grade: str = ""
    summary: str = ""


@dataclass
class ROICResult:
    """Phase 56: Return on Invested Capital Analysis."""
    roic_pct: Optional[float] = None
    nopat: Optional[float] = None
    invested_capital: Optional[float] = None
    roic_roa_spread: Optional[float] = None
    roic_wacc_spread: Optional[float] = None
    capital_efficiency: Optional[float] = None
    roic_score: float = 0.0
    roic_grade: str = ""
    summary: str = ""


@dataclass
class ROAQualityResult:
    """Phase 55: Return on Assets Quality Analysis."""
    roa_pct: Optional[float] = None
    operating_roa_pct: Optional[float] = None
    cash_roa_pct: Optional[float] = None
    asset_turnover: Optional[float] = None
    fixed_asset_turnover: Optional[float] = None
    capital_intensity: Optional[float] = None
    roa_score: float = 0.0
    roa_grade: str = ""
    summary: str = ""


@dataclass
class ROEAnalysisResult:
    """Phase 54: Return on Equity Analysis."""
    roe_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    asset_turnover: Optional[float] = None
    equity_multiplier: Optional[float] = None
    roa_pct: Optional[float] = None
    retention_ratio: Optional[float] = None
    sustainable_growth_rate: Optional[float] = None
    roe_score: float = 0.0
    roe_grade: str = ""
    summary: str = ""


@dataclass
class NetProfitMarginResult:
    """Phase 53: Net Profit Margin Analysis."""
    net_margin_pct: Optional[float] = None          # NI / Revenue
    ebitda_margin_pct: Optional[float] = None       # EBITDA / Revenue
    ebit_margin_pct: Optional[float] = None         # EBIT / Revenue
    tax_burden: Optional[float] = None              # NI / EBT (1 - effective tax rate)
    interest_burden: Optional[float] = None         # EBT / EBIT
    net_to_ebitda: Optional[float] = None           # NI / EBITDA (retention ratio)
    npm_score: float = 0.0                          # 0-10 (higher = better)
    npm_grade: str = ""                             # "Excellent", "Good", "Adequate", "Weak"
    summary: str = ""


@dataclass
class EbitdaMarginQualityResult:
    """Phase 52: EBITDA Margin Quality Analysis."""
    ebitda_margin_pct: Optional[float] = None       # EBITDA / Revenue
    operating_margin_pct: Optional[float] = None    # OI / Revenue
    da_intensity: Optional[float] = None            # D&A / Revenue
    ebitda_oi_spread: Optional[float] = None        # EBITDA margin - OI margin (= D&A/Rev)
    ebitda_to_gp: Optional[float] = None            # EBITDA / Gross Profit
    ebitda_score: float = 0.0                       # 0-10 (higher = better)
    ebitda_grade: str = ""                          # "Excellent", "Good", "Adequate", "Weak"
    summary: str = ""


@dataclass
class GrossMarginStabilityResult:
    """Phase 51: Gross Margin Stability Analysis."""
    gross_margin_pct: Optional[float] = None       # GP / Revenue
    cogs_ratio: Optional[float] = None             # COGS / Revenue
    operating_margin_pct: Optional[float] = None   # Operating Income / Revenue
    margin_spread: Optional[float] = None          # Gross Margin - Operating Margin
    opex_coverage: Optional[float] = None          # GP / Operating Expenses
    margin_buffer: Optional[float] = None          # Distance to breakeven (= gross margin)
    gm_stability_score: float = 0.0                # 0-10 (higher = better)
    gm_stability_grade: str = ""                   # "Excellent", "Good", "Adequate", "Weak"
    summary: str = ""


@dataclass
class BeneishMScoreResult:
    """Phase 43: Beneish M-Score (Earnings Manipulation Detection)."""
    m_score: Optional[float] = None
    dsri: Optional[float] = None  # Days Sales in Receivables Index
    gmi: Optional[float] = None   # Gross Margin Index
    aqi: Optional[float] = None   # Asset Quality Index
    sgi: Optional[float] = None   # Sales Growth Index
    depi: Optional[float] = None  # Depreciation Index
    sgai: Optional[float] = None  # SGA Expense Index
    lvgi: Optional[float] = None  # Leverage Index
    tata: Optional[float] = None  # Total Accruals to Total Assets
    manipulation_score: float = 0.0  # 0-10 (higher = less likely manipulated)
    manipulation_grade: str = ""  # "Unlikely", "Possible", "Likely", "Highly Likely"
    summary: str = ""


@dataclass
class FinancialReport:
    """Structured financial analysis report."""
    sections: Dict[str, str] = field(default_factory=dict)
    executive_summary: str = ""
    generated_at: str = ""


@dataclass
class FinancialData:
    """Container for financial data used in analysis."""
    # Balance Sheet items
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    cash: Optional[float] = None
    inventory: Optional[float] = None
    accounts_receivable: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    accounts_payable: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None

    # Income Statement items
    revenue: Optional[float] = None
    cogs: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    operating_expenses: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    net_income: Optional[float] = None
    ebt: Optional[float] = None  # Earnings Before Tax
    tax_expense: Optional[float] = None  # Income tax expense
    retained_earnings: Optional[float] = None
    depreciation: Optional[float] = None

    # Cash Flow items
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    # Shareholder returns
    dividends_paid: Optional[float] = None
    share_buybacks: Optional[float] = None
    shares_outstanding: Optional[float] = None
    share_price: Optional[float] = None

    # Derived/Previous period
    avg_inventory: Optional[float] = None
    avg_receivables: Optional[float] = None
    avg_payables: Optional[float] = None
    avg_total_assets: Optional[float] = None

    # Startup / SaaS metrics (optional)
    monthly_recurring_revenue: Optional[float] = None
    annual_recurring_revenue: Optional[float] = None
    customer_count: Optional[int] = None
    churned_customers: Optional[int] = None
    monthly_burn_rate: Optional[float] = None
    cash_runway_months: Optional[float] = None
    customer_acquisition_cost: Optional[float] = None
    lifetime_value: Optional[float] = None
    total_funding_raised: Optional[float] = None

    # Time series data
    time_series: Optional[pd.DataFrame] = None
    period: Optional[str] = None


class CharlieAnalyzer:
    """
    CFO-grade financial analysis engine inspired by Charlie Munger's
    analytical framework: focus on fundamentals, cash flow, and
    sustainable competitive advantages.
    """

    def __init__(self, tax_rate: Optional[float] = None):
        from config import settings
        self._tax_rate = tax_rate if tax_rate is not None else settings.default_tax_rate
        self.ratio_definitions = self._load_ratio_definitions()
        self.insight_templates = self._load_insight_templates()

    def _load_ratio_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load standard financial ratio definitions."""
        return {
            'current_ratio': {
                'name': 'Current Ratio',
                'formula': 'Current Assets / Current Liabilities',
                'benchmark': {'good': 1.5, 'warning': 1.0},
                'interpretation': 'Measures ability to pay short-term obligations'
            },
            'quick_ratio': {
                'name': 'Quick Ratio (Acid Test)',
                'formula': '(Current Assets - Inventory) / Current Liabilities',
                'benchmark': {'good': 1.0, 'warning': 0.5},
                'interpretation': 'Measures ability to pay short-term obligations without selling inventory'
            },
            'cash_ratio': {
                'name': 'Cash Ratio',
                'formula': 'Cash / Current Liabilities',
                'benchmark': {'good': 0.5, 'warning': 0.2},
                'interpretation': 'Most conservative liquidity measure'
            },
            'gross_margin': {
                'name': 'Gross Margin',
                'formula': '(Revenue - COGS) / Revenue',
                'benchmark': {'varies': True},
                'interpretation': 'Profitability after direct costs'
            },
            'operating_margin': {
                'name': 'Operating Margin',
                'formula': 'Operating Income / Revenue',
                'benchmark': {'good': 0.15, 'warning': 0.05},
                'interpretation': 'Profitability from core operations'
            },
            'net_margin': {
                'name': 'Net Profit Margin',
                'formula': 'Net Income / Revenue',
                'benchmark': {'good': 0.10, 'warning': 0.02},
                'interpretation': 'Bottom-line profitability'
            },
            'roe': {
                'name': 'Return on Equity',
                'formula': 'Net Income / Shareholders Equity',
                'benchmark': {'good': 0.15, 'warning': 0.08},
                'interpretation': 'Return generated on shareholder investment'
            },
            'roa': {
                'name': 'Return on Assets',
                'formula': 'Net Income / Total Assets',
                'benchmark': {'good': 0.05, 'warning': 0.02},
                'interpretation': 'Efficiency in using assets to generate profit'
            },
            'debt_to_equity': {
                'name': 'Debt-to-Equity Ratio',
                'formula': 'Total Debt / Total Equity',
                'benchmark': {'good': 1.0, 'warning': 2.0},
                'interpretation': 'Financial leverage and risk'
            },
            'interest_coverage': {
                'name': 'Interest Coverage Ratio',
                'formula': 'EBIT / Interest Expense',
                'benchmark': {'good': 3.0, 'warning': 1.5},
                'interpretation': 'Ability to pay interest on debt'
            },
            'asset_turnover': {
                'name': 'Asset Turnover',
                'formula': 'Revenue / Average Total Assets',
                'benchmark': {'varies': True},
                'interpretation': 'Efficiency in using assets to generate sales'
            }
        }

    def _load_insight_templates(self) -> Dict[str, str]:
        """Load insight generation templates."""
        return {
            'high_liquidity': "Strong liquidity position with {ratio_name} of {value:.2f}. The company can comfortably meet short-term obligations.",
            'low_liquidity': "Liquidity concern: {ratio_name} of {value:.2f} is below the recommended threshold of {threshold}. Consider improving working capital management.",
            'improving_margin': "Profitability improving: {metric} increased from {old_value:.1%} to {new_value:.1%}, indicating better operational efficiency.",
            'declining_margin': "Margin pressure: {metric} declined from {old_value:.1%} to {new_value:.1%}. Investigate cost structure and pricing.",
            'high_leverage': "Elevated leverage: Debt-to-equity of {value:.2f} exceeds typical comfort levels. Monitor debt service capacity.",
            'strong_cash_generation': "Robust cash generation: Free cash flow of {value:,.0f} provides strategic flexibility.",
            'cash_conversion_concern': "Cash conversion cycle of {value:.0f} days may tie up working capital. Consider optimizing AR/AP terms.",
            'favorable_variance': "Favorable budget variance in {category}: {percent:.1%} under budget, saving {amount:,.0f}.",
            'unfavorable_variance': "Budget overrun in {category}: {percent:.1%} over budget. Review spending controls."
        }

    # ===== LIQUIDITY RATIOS =====

    def calculate_liquidity_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate liquidity ratios.
        - Current Ratio = Current Assets / Current Liabilities
        - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        - Cash Ratio = Cash & Equivalents / Current Liabilities
        """
        ratios = {}

        ratios['current_ratio'] = safe_divide(data.current_assets, data.current_liabilities)

        quick_numerator = (
            (data.current_assets - data.inventory)
            if data.current_assets is not None and data.inventory is not None
            else None
        )
        ratios['quick_ratio'] = safe_divide(quick_numerator, data.current_liabilities)

        ratios['cash_ratio'] = safe_divide(data.cash, data.current_liabilities)

        return ratios

    # ===== PROFITABILITY RATIOS =====

    def calculate_profitability_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate profitability ratios.
        """
        ratios = {}

        # Gross Margin
        gross_numerator = None
        if data.revenue is not None and data.cogs is not None:
            gross_numerator = data.revenue - data.cogs
        elif data.gross_profit is not None:
            gross_numerator = data.gross_profit
        ratios['gross_margin'] = safe_divide(gross_numerator, data.revenue)

        # Operating Margin
        ratios['operating_margin'] = safe_divide(data.operating_income, data.revenue)

        # Net Margin
        ratios['net_margin'] = safe_divide(data.net_income, data.revenue)

        # ROE
        ratios['roe'] = safe_divide(data.net_income, data.total_equity)

        # ROA
        roa = safe_divide(data.net_income, data.total_assets)
        if roa is None:
            roa = safe_divide(data.net_income, data.avg_total_assets)
        ratios['roa'] = roa

        # ROIC (Return on Invested Capital)
        invested_capital = (
            (data.total_equity + data.total_debt)
            if data.total_equity is not None and data.total_debt is not None
            else None
        )
        nopat = (
            data.operating_income * (1 - self._tax_rate)
            if data.operating_income is not None
            else None
        )
        ratios['roic'] = safe_divide(nopat, invested_capital)

        return ratios

    # ===== LEVERAGE RATIOS =====

    def calculate_leverage_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate leverage/solvency ratios.
        """
        ratios = {}

        ratios['debt_to_equity'] = safe_divide(data.total_debt, data.total_equity)
        ratios['debt_to_assets'] = safe_divide(data.total_debt, data.total_assets)

        interest_cov = safe_divide(data.ebit, data.interest_expense)
        if interest_cov is None:
            interest_cov = safe_divide(data.operating_income, data.interest_expense)
        ratios['interest_coverage'] = interest_cov

        return ratios

    # ===== EFFICIENCY RATIOS =====

    def calculate_efficiency_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate efficiency/activity ratios.
        """
        ratios = {}

        asset_turn = safe_divide(data.revenue, data.avg_total_assets)
        if asset_turn is None:
            asset_turn = safe_divide(data.revenue, data.total_assets)
        ratios['asset_turnover'] = asset_turn

        inv_turn = safe_divide(data.cogs, data.avg_inventory)
        if inv_turn is None:
            inv_turn = safe_divide(data.cogs, data.inventory)
        ratios['inventory_turnover'] = inv_turn

        recv_turn = safe_divide(data.revenue, data.avg_receivables)
        if recv_turn is None:
            recv_turn = safe_divide(data.revenue, data.accounts_receivable)
        ratios['receivables_turnover'] = recv_turn

        pay_turn = safe_divide(data.cogs, data.avg_payables)
        if pay_turn is None:
            pay_turn = safe_divide(data.cogs, data.accounts_payable)
        ratios['payables_turnover'] = pay_turn

        return ratios

    # ===== TREND ANALYSIS =====

    def analyze_trends(self, time_series: pd.DataFrame, metric_column: str,
                      period_column: Optional[str] = None) -> TrendAnalysis:
        """
        Analyze trends in a time series.
        """
        mask = time_series[metric_column].notna()
        values = time_series.loc[mask, metric_column].tolist()

        if period_column and period_column in time_series.columns:
            periods = time_series.loc[mask, period_column].tolist()
        else:
            periods = [f"Period {i+1}" for i in range(len(values))]

        # Calculate growth rates
        yoy_growth = None
        qoq_growth = None
        mom_growth = None

        if len(values) >= 2:
            ratio = safe_divide(values[-1], values[-2])
            mom_growth = (ratio - 1) if ratio is not None else None

        if len(values) >= 4:
            ratio = safe_divide(values[-1], values[-4])
            qoq_growth = (ratio - 1) if ratio is not None else None

        if len(values) >= 12:
            ratio = safe_divide(values[-1], values[-12])
            yoy_growth = (ratio - 1) if ratio is not None else None

        # Calculate CAGR - requires positive start and end values
        cagr = None
        if len(values) >= 2 and values[0] is not None and values[-1] is not None:
            if values[0] > 0 and values[-1] > 0:
                n_periods = len(values) - 1
                cagr = (values[-1] / values[0]) ** (1 / n_periods) - 1

        # Moving averages
        ma_3 = self._calculate_moving_average(values, 3)
        ma_12 = self._calculate_moving_average(values, 12)

        # Trend direction
        if len(values) >= 3:
            recent_avg = np.mean(values[-3:])
            older_avg = np.mean(values[:3])
            if recent_avg > older_avg * 1.05:
                trend_direction = 'up'
            elif recent_avg < older_avg * 0.95:
                trend_direction = 'down'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'insufficient_data'

        # Simple seasonality detection
        seasonality_detected = self._detect_seasonality(values)

        return TrendAnalysis(
            metric_name=metric_column,
            values=values,
            periods=periods,
            yoy_growth=yoy_growth,
            qoq_growth=qoq_growth,
            mom_growth=mom_growth,
            cagr=cagr,
            moving_avg_3=ma_3,
            moving_avg_12=ma_12,
            trend_direction=trend_direction,
            seasonality_detected=seasonality_detected
        )

    def _calculate_moving_average(self, values: List[float], window: int) -> List[float]:
        """Calculate moving average using pandas rolling for O(n) performance."""
        if len(values) < window:
            return list(values)

        s = pd.Series(values)
        return s.rolling(window=window, min_periods=1).mean().tolist()

    def _detect_seasonality(self, values: List[float]) -> bool:
        """Simple seasonality detection."""
        if len(values) < 24:  # Need at least 2 years of monthly data
            return False

        # Check for repeating patterns
        try:
            first_year = values[:12]
            second_year = values[12:24]
            correlation = np.corrcoef(first_year, second_year)[0, 1]
            return correlation > 0.7
        except (ValueError, IndexError, TypeError):
            return False

    def calculate_variance(self, actual: float, budget: float,
                          category: str = "General") -> VarianceResult:
        """
        Calculate budget variance.
        """
        variance = actual - budget
        variance_percent = safe_divide(variance, abs(budget), default=0.0 if variance == 0 else None)

        # For expenses, under budget is favorable
        # For revenue, over budget is favorable
        is_expense = category.lower() in ['expense', 'cost', 'cogs', 'opex', 'sg&a']
        favorable = (variance < 0) if is_expense else (variance > 0)

        return VarianceResult(
            actual=actual,
            budget=budget,
            variance=variance,
            variance_percent=variance_percent,
            favorable=favorable,
            category=category
        )

    def forecast_simple(self, historical: List[float], periods: int = 3,
                       method: str = 'linear') -> Forecast:
        """
        Simple forecasting methods.
        """
        if not historical or len(historical) < 2:
            return Forecast(
                metric_name="unknown",
                historical_values=historical,
                forecasted_values=[],
                forecast_periods=[],
                method=method,
                confidence_interval=(0, 0)
            )

        forecasted = []

        if method == 'linear':
            # Linear trend extrapolation (normalize x to avoid conditioning issues)
            n_hist = len(historical)
            x = np.arange(n_hist, dtype=float)
            # Normalize: shift to zero-mean for better polyfit conditioning
            x_mean = x.mean()
            x_norm = x - x_mean
            coeffs = np.polyfit(x_norm, historical, 1)
            for i in range(periods):
                forecasted.append(coeffs[0] * (n_hist + i - x_mean) + coeffs[1])

        elif method == 'moving_average':
            # Moving average projection
            window = min(3, len(historical))
            ma = np.mean(historical[-window:])
            forecasted = [ma] * periods

        elif method == 'growth_rate':
            # Apply average growth rate
            growth_rates = [
                r - 1 for i in range(1, len(historical))
                if (r := safe_divide(historical[i], historical[i - 1])) is not None
            ]
            if growth_rates:
                avg_growth = np.mean(growth_rates)
                last_value = historical[-1]
                for _ in range(periods):
                    last_value = last_value * (1 + avg_growth)
                    forecasted.append(last_value)
            else:
                forecasted = [historical[-1]] * periods

        # Calculate confidence interval (simple std-based)
        std = np.std(historical) if len(historical) > 1 else 0
        mean_forecast = np.mean(forecasted) if forecasted else 0
        ci = (mean_forecast - 2 * std, mean_forecast + 2 * std)

        forecast_periods = [f"Forecast {i+1}" for i in range(periods)]

        return Forecast(
            metric_name="unknown",
            historical_values=historical,
            forecasted_values=forecasted,
            forecast_periods=forecast_periods,
            method=method,
            confidence_interval=ci
        )

    # ===== BUDGET VS ACTUAL ANALYSIS =====

    def analyze_budget_variance(self, actual_df: pd.DataFrame, budget_df: pd.DataFrame,
                               item_column: str, actual_column: str,
                               budget_column: str) -> BudgetAnalysis:
        """
        Comprehensive budget analysis.
        """
        line_items = []

        # Merge on item column
        merged = actual_df.merge(budget_df, on=item_column, how='outer', suffixes=('_actual', '_budget'))

        unmatched_items: list = []
        for _, row in merged.iterrows():
            category = row[item_column]
            actual = row.get(actual_column, row.get(f'{actual_column}_actual', None))
            budget = row.get(budget_column, row.get(f'{budget_column}_budget', None))

            # Flag line items present in only one dataset
            actual_missing = actual is None or (isinstance(actual, float) and np.isnan(actual))
            budget_missing = budget is None or (isinstance(budget, float) and np.isnan(budget))
            if actual_missing or budget_missing:
                side = "budget" if budget_missing else "actual"
                unmatched_items.append(f"{category} (missing {side})")
                logger.warning("Budget variance: '%s' missing %s value; treating as 0.", category, side)

            actual = 0 if actual_missing else float(actual)
            budget = 0 if budget_missing else float(budget)

            variance_result = self.calculate_variance(actual, budget, category)
            line_items.append(variance_result)

        total_actual = sum(item.actual for item in line_items)
        total_budget = sum(item.budget for item in line_items)
        total_variance = total_actual - total_budget
        total_variance_percent = safe_divide(total_variance, total_budget, default=0.0)

        favorable_items = [item for item in line_items if item.favorable]
        unfavorable_items = [item for item in line_items if not item.favorable]

        # Get largest variances by absolute value
        largest_variances = sorted(line_items, key=lambda x: abs(x.variance), reverse=True)[:5]

        return BudgetAnalysis(
            total_budget=total_budget,
            total_actual=total_actual,
            total_variance=total_variance,
            total_variance_percent=total_variance_percent,
            line_items=line_items,
            favorable_items=favorable_items,
            unfavorable_items=unfavorable_items,
            largest_variances=largest_variances,
            unmatched_items=unmatched_items,
        )

    # ===== CASH FLOW ANALYSIS =====

    def analyze_cash_flow(self, data: FinancialData) -> CashFlowAnalysis:
        """
        Analyze cash flow metrics.
        """
        # Free Cash Flow = Operating CF - CapEx
        fcf = None
        if data.operating_cash_flow is not None and data.capex is not None:
            fcf = data.operating_cash_flow - abs(data.capex)
        elif data.operating_cash_flow is not None:
            fcf = data.operating_cash_flow

        # Days Sales Outstanding (DSO)
        dso_ratio = safe_divide(data.accounts_receivable, data.revenue)
        dso = dso_ratio * 365 if dso_ratio is not None else None

        # Days Inventory Outstanding (DIO)
        dio_ratio = safe_divide(data.inventory, data.cogs)
        dio = dio_ratio * 365 if dio_ratio is not None else None

        # Days Payables Outstanding (DPO)
        dpo_ratio = safe_divide(data.accounts_payable, data.cogs)
        dpo = dpo_ratio * 365 if dpo_ratio is not None else None

        # Cash Conversion Cycle = DSO + DIO - DPO
        ccc = None
        if dso is not None and dio is not None and dpo is not None:
            ccc = dso + dio - dpo

        return CashFlowAnalysis(
            operating_cf=data.operating_cash_flow,
            investing_cf=data.investing_cash_flow,
            financing_cf=data.financing_cash_flow,
            free_cash_flow=fcf,
            dso=dso,
            dio=dio,
            dpo=dpo,
            cash_conversion_cycle=ccc
        )

    # ===== WORKING CAPITAL ANALYSIS =====

    def analyze_working_capital(self, data: FinancialData) -> WorkingCapitalAnalysis:
        """
        Analyze working capital.
        """
        nwc = None
        if data.current_assets is not None and data.current_liabilities is not None:
            nwc = data.current_assets - data.current_liabilities

        wc_ratio = safe_divide(data.current_assets, data.current_liabilities)
        wc_turnover = safe_divide(data.revenue, nwc)

        return WorkingCapitalAnalysis(
            current_assets=data.current_assets,
            current_liabilities=data.current_liabilities,
            net_working_capital=nwc,
            working_capital_ratio=wc_ratio,
            working_capital_turnover=wc_turnover
        )

    # ===== ADVANCED SCORING MODELS =====

    def dupont_analysis(self, data: FinancialData) -> DuPontAnalysis:
        """DuPont decomposition of ROE into component drivers.

        3-factor: ROE = Net Margin × Asset Turnover × Equity Multiplier
        5-factor: ROE = Tax Burden × Interest Burden × EBIT Margin × Asset Turnover × Equity Multiplier
        """
        net_margin = safe_divide(data.net_income, data.revenue)
        asset_turnover = safe_divide(data.revenue, data.total_assets)
        equity_multiplier = safe_divide(data.total_assets, data.total_equity)

        # 3-factor ROE
        roe = None
        if all(v is not None for v in [net_margin, asset_turnover, equity_multiplier]):
            roe = net_margin * asset_turnover * equity_multiplier

        # 5-factor extensions
        ebt = data.ebt
        if ebt is None and data.net_income is not None:
            # Approximate EBT = Net Income / (1 - tax_rate)
            ebt = safe_divide(data.net_income, 1 - self._tax_rate)

        tax_burden = safe_divide(data.net_income, ebt)
        interest_burden = safe_divide(ebt, data.ebit if data.ebit is not None else data.operating_income)

        # Identify primary driver
        primary_driver = None
        if all(v is not None for v in [net_margin, asset_turnover, equity_multiplier]):
            drivers = {
                'net_margin': abs(net_margin) if net_margin else 0,
                'asset_turnover': abs(asset_turnover) if asset_turnover else 0,
                'equity_multiplier': abs(equity_multiplier - 1) if equity_multiplier else 0,
            }
            primary_driver = max(drivers, key=drivers.get)

        # Generate interpretation
        interpretation = None
        if roe is not None:
            parts = []
            if net_margin is not None:
                parts.append(f"net margin {net_margin:.1%}")
            if asset_turnover is not None:
                parts.append(f"asset turnover {asset_turnover:.2f}x")
            if equity_multiplier is not None:
                parts.append(f"equity multiplier {equity_multiplier:.2f}x")
            interpretation = f"ROE of {roe:.1%} driven by {', '.join(parts)}."
            if data.total_equity is not None and data.total_equity < 0:
                interpretation += (
                    " WARNING: Negative equity makes the DuPont decomposition "
                    "misleading — equity multiplier is negative."
                )

        return DuPontAnalysis(
            roe=roe,
            net_margin=net_margin,
            asset_turnover=asset_turnover,
            equity_multiplier=equity_multiplier,
            tax_burden=tax_burden,
            interest_burden=interest_burden,
            primary_driver=primary_driver,
            interpretation=interpretation,
        )

    def altman_z_score(self, data: FinancialData) -> AltmanZScore:
        """Altman Z-Score bankruptcy prediction model.

        Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
        Zones: Safe (>2.99), Grey (1.81-2.99), Distress (<1.81)
        """
        if data.total_assets is None or data.total_assets == 0:
            return AltmanZScore(interpretation="Insufficient data: total assets required.")

        # X1: Working Capital / Total Assets
        working_capital = None
        if data.current_assets is not None and data.current_liabilities is not None:
            working_capital = data.current_assets - data.current_liabilities
        x1 = safe_divide(working_capital, data.total_assets)

        # X2: Retained Earnings / Total Assets
        x2 = safe_divide(data.retained_earnings, data.total_assets)

        # X3: EBIT / Total Assets
        ebit = data.ebit if data.ebit is not None else data.operating_income
        x3 = safe_divide(ebit, data.total_assets)

        # X4: Book Value of Equity / Total Liabilities (proxy for market value)
        x4 = safe_divide(data.total_equity, data.total_liabilities)

        # X5: Sales / Total Assets
        x5 = safe_divide(data.revenue, data.total_assets)

        components = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}

        # Calculate Z-Score only with available components
        available = {k: v for k, v in components.items() if v is not None}
        if not available:
            return AltmanZScore(
                components=components,
                interpretation="Insufficient data for Z-Score calculation."
            )

        weights = {'x1': 1.2, 'x2': 1.4, 'x3': 3.3, 'x4': 0.6, 'x5': 1.0}
        z = sum(weights[k] * v for k, v in available.items())

        # Classify zone
        if len(available) == 5:
            if z > 2.99:
                zone = 'safe'
                interp = f"Z-Score of {z:.2f} indicates low bankruptcy risk (safe zone >2.99)."
            elif z >= 1.81:
                zone = 'grey'
                interp = f"Z-Score of {z:.2f} is in the grey zone (1.81-2.99). Monitor closely."
            else:
                zone = 'distress'
                interp = f"Z-Score of {z:.2f} signals financial distress (<1.81). Immediate attention required."
        else:
            # Partial data — still flag likely distress when score is very low
            if z < 1.81:
                zone = 'partial_distress'
                interp = (
                    f"Partial Z-Score of {z:.2f} (using {len(available)}/5 components) "
                    "suggests possible distress (<1.81). Interpret with caution — "
                    "missing components may raise or lower the score."
                )
            else:
                zone = 'partial'
                interp = (
                    f"Partial Z-Score of {z:.2f} (using {len(available)}/5 components). "
                    "Interpret with caution."
                )

        return AltmanZScore(
            z_score=z,
            zone=zone,
            components=components,
            interpretation=interp,
        )

    def piotroski_f_score(self, data: FinancialData,
                          prior_data: Optional[FinancialData] = None) -> PiotroskiFScore:
        """Piotroski F-Score: 9-criteria financial strength model.

        Profitability (4 pts):
          1. ROA > 0
          2. Operating Cash Flow > 0
          3. Delta ROA > 0 (improving)
          4. Accruals: Operating CF > Net Income (quality of earnings)

        Leverage/Liquidity (3 pts):
          5. Delta Leverage < 0 (decreasing)
          6. Delta Current Ratio > 0 (improving)
          7. No new shares issued (no dilution)

        Operating Efficiency (2 pts):
          8. Delta Gross Margin > 0 (improving)
          9. Delta Asset Turnover > 0 (improving)
        """
        criteria = {}
        score = 0

        # --- Profitability (4 points) ---

        # 1. ROA > 0
        roa = safe_divide(data.net_income, data.total_assets)
        criteria['positive_roa'] = roa is not None and roa > 0
        if criteria['positive_roa']:
            score += 1

        # 2. Operating Cash Flow > 0
        criteria['positive_ocf'] = (data.operating_cash_flow is not None
                                     and data.operating_cash_flow > 0)
        if criteria['positive_ocf']:
            score += 1

        # 3. Delta ROA > 0 (improving profitability)
        if prior_data is not None:
            prior_roa = safe_divide(prior_data.net_income, prior_data.total_assets)
            criteria['improving_roa'] = (roa is not None and prior_roa is not None
                                          and roa > prior_roa)
        else:
            criteria['improving_roa'] = False  # Can't assess without prior data
        if criteria['improving_roa']:
            score += 1

        # 4. Accruals: OCF > Net Income (quality of earnings)
        criteria['quality_earnings'] = (data.operating_cash_flow is not None
                                         and data.net_income is not None
                                         and data.operating_cash_flow > data.net_income)
        if criteria['quality_earnings']:
            score += 1

        # --- Leverage / Liquidity (3 points) ---

        # 5. Delta Leverage < 0 (decreasing debt-to-assets)
        leverage = safe_divide(data.total_debt, data.total_assets)
        if prior_data is not None:
            prior_leverage = safe_divide(prior_data.total_debt, prior_data.total_assets)
            criteria['decreasing_leverage'] = (leverage is not None and prior_leverage is not None
                                                and leverage < prior_leverage)
        else:
            criteria['decreasing_leverage'] = False
        if criteria['decreasing_leverage']:
            score += 1

        # 6. Delta Current Ratio > 0 (improving liquidity)
        current_ratio = safe_divide(data.current_assets, data.current_liabilities)
        if prior_data is not None:
            prior_cr = safe_divide(prior_data.current_assets, prior_data.current_liabilities)
            criteria['improving_liquidity'] = (current_ratio is not None and prior_cr is not None
                                                and current_ratio > prior_cr)
        else:
            criteria['improving_liquidity'] = False
        if criteria['improving_liquidity']:
            score += 1

        # 7. No dilution (no new shares - heuristic: equity didn't jump without income)
        # Without share count, approximate: equity growth < net income suggests no dilution
        if prior_data is not None and data.total_equity is not None and prior_data.total_equity is not None:
            equity_change = data.total_equity - prior_data.total_equity
            criteria['no_dilution'] = (data.net_income is not None
                                        and equity_change <= (data.net_income or 0) * 1.1)
        else:
            criteria['no_dilution'] = True  # Assume no dilution if unknown
        if criteria['no_dilution']:
            score += 1

        # --- Operating Efficiency (2 points) ---

        # 8. Delta Gross Margin > 0
        gross_margin = safe_divide(
            (data.revenue - data.cogs) if data.revenue and data.cogs else data.gross_profit,
            data.revenue
        )
        if prior_data is not None:
            prior_gm = safe_divide(
                (prior_data.revenue - prior_data.cogs) if prior_data.revenue and prior_data.cogs else prior_data.gross_profit,
                prior_data.revenue
            )
            criteria['improving_gross_margin'] = (gross_margin is not None and prior_gm is not None
                                                   and gross_margin > prior_gm)
        else:
            criteria['improving_gross_margin'] = False
        if criteria['improving_gross_margin']:
            score += 1

        # 9. Delta Asset Turnover > 0
        at = safe_divide(data.revenue, data.total_assets)
        if prior_data is not None:
            prior_at = safe_divide(prior_data.revenue, prior_data.total_assets)
            criteria['improving_asset_turnover'] = (at is not None and prior_at is not None
                                                     and at > prior_at)
        else:
            criteria['improving_asset_turnover'] = False
        if criteria['improving_asset_turnover']:
            score += 1

        # Interpretation
        if score >= 8:
            interp = f"F-Score {score}/9: Very strong financial position."
        elif score >= 6:
            interp = f"F-Score {score}/9: Healthy fundamentals."
        elif score >= 4:
            interp = f"F-Score {score}/9: Mixed signals - review weak criteria."
        else:
            interp = f"F-Score {score}/9: Weak financial health. Multiple concerns detected."

        return PiotroskiFScore(
            score=score,
            max_score=9,
            criteria=criteria,
            interpretation=interp,
        )

    # ===== COMPOSITE HEALTH SCORE =====

    def composite_health_score(self, data: FinancialData,
                                prior_data: Optional[FinancialData] = None) -> CompositeHealthScore:
        """Compute a weighted 0-100 composite financial health score with letter grade.

        Components (100 total):
          - Z-Score zone:     25 pts
          - F-Score:          25 pts
          - Profitability:    20 pts
          - Liquidity:        15 pts
          - Leverage:         15 pts
        """
        component_scores: Dict[str, int] = {}

        # 1. Z-Score component (25 points)
        z = self.altman_z_score(data)
        if z.zone == 'safe':
            z_pts = 25
        elif z.zone == 'grey':
            z_pts = 15
        elif z.zone == 'partial':
            z_pts = 10
        elif z.zone == 'distress':
            z_pts = 0
        elif z.zone == 'partial_distress':
            # Partial data with low score — penalize less than confirmed distress
            z_pts = 8
        else:
            z_pts = 5
        component_scores['z_score'] = z_pts

        # 2. F-Score component (25 points)
        f = self.piotroski_f_score(data, prior_data)
        f_pts = round(f.score / f.max_score * 25)
        component_scores['f_score'] = f_pts

        # 3. Profitability component (20 points)
        prof = self.calculate_profitability_ratios(data)
        prof_pts = 0

        nm = prof.get('net_margin')
        if nm is not None:
            if nm > 0.15:
                prof_pts += 8
            elif nm > 0.05:
                prof_pts += 5
            elif nm > 0:
                prof_pts += 3

        roa_val = prof.get('roa')
        if roa_val is not None:
            if roa_val > 0.05:
                prof_pts += 6
            elif roa_val > 0.02:
                prof_pts += 4
            elif roa_val > 0:
                prof_pts += 2

        roe_val = prof.get('roe')
        if roe_val is not None:
            if roe_val > 0.15:
                prof_pts += 6
            elif roe_val > 0.08:
                prof_pts += 4
            elif roe_val > 0:
                prof_pts += 2

        component_scores['profitability'] = prof_pts

        # 4. Liquidity component (15 points)
        liq = self.calculate_liquidity_ratios(data)
        cr = liq.get('current_ratio')
        if cr is not None:
            if cr >= 2.0:
                liq_pts = 15
            elif cr >= 1.5:
                liq_pts = 12
            elif cr >= 1.0:
                liq_pts = 8
            elif cr >= 0.5:
                liq_pts = 4
            else:
                liq_pts = 0
        else:
            liq_pts = 0
        component_scores['liquidity'] = liq_pts

        # 5. Leverage component (15 points)
        lev = self.calculate_leverage_ratios(data)
        de = lev.get('debt_to_equity')
        if de is not None:
            if de <= 0.5:
                lev_pts = 15
            elif de <= 1.0:
                lev_pts = 12
            elif de <= 2.0:
                lev_pts = 8
            elif de <= 3.0:
                lev_pts = 4
            else:
                lev_pts = 0
        else:
            lev_pts = 0
        component_scores['leverage'] = lev_pts

        total = sum(component_scores.values())

        # Letter grade
        if total >= 80:
            grade = 'A'
        elif total >= 65:
            grade = 'B'
        elif total >= 50:
            grade = 'C'
        elif total >= 35:
            grade = 'D'
        else:
            grade = 'F'

        strongest = max(component_scores, key=component_scores.get)
        weakest = min(component_scores, key=component_scores.get)
        interpretation = (
            f"Composite score {total}/100 (Grade {grade}). "
            f"Strongest: {strongest.replace('_', ' ')}. "
            f"Weakest: {weakest.replace('_', ' ')}."
        )

        return CompositeHealthScore(
            score=total,
            grade=grade,
            component_scores=component_scores,
            interpretation=interpretation,
        )

    # ===== MULTI-PERIOD COMPARISON =====

    def compare_periods(self, current: FinancialData,
                         prior: FinancialData) -> PeriodComparison:
        """Compare two periods and compute deltas for all financial metrics."""
        current_ratios: Dict[str, Optional[float]] = {}
        prior_ratios: Dict[str, Optional[float]] = {}

        for category_name, method in [
            ('liquidity', self.calculate_liquidity_ratios),
            ('profitability', self.calculate_profitability_ratios),
            ('leverage', self.calculate_leverage_ratios),
            ('efficiency', self.calculate_efficiency_ratios),
        ]:
            c = method(current)
            p = method(prior)
            for k, v in c.items():
                current_ratios[f"{category_name}_{k}"] = v
            for k, v in p.items():
                prior_ratios[f"{category_name}_{k}"] = v

        # Scoring models
        c_z = self.altman_z_score(current)
        p_z = self.altman_z_score(prior)
        current_ratios['altman_z_score'] = c_z.z_score
        prior_ratios['altman_z_score'] = p_z.z_score

        c_f = self.piotroski_f_score(current, prior)
        p_f = self.piotroski_f_score(prior)
        current_ratios['piotroski_f_score'] = float(c_f.score)
        prior_ratios['piotroski_f_score'] = float(p_f.score)

        # Compute deltas
        deltas: Dict[str, float] = {}
        improvements: List[str] = []
        deteriorations: List[str] = []

        lower_is_better = {'leverage_debt_to_equity', 'leverage_debt_to_assets'}

        for key in current_ratios:
            cv = current_ratios.get(key)
            pv = prior_ratios.get(key)
            if cv is not None and pv is not None:
                delta = cv - pv
                deltas[key] = delta

                if key in lower_is_better:
                    if delta < -1e-6:
                        improvements.append(key)
                    elif delta > 1e-6:
                        deteriorations.append(key)
                else:
                    if delta > 1e-6:
                        improvements.append(key)
                    elif delta < -1e-6:
                        deteriorations.append(key)

        return PeriodComparison(
            current_ratios=current_ratios,
            prior_ratios=prior_ratios,
            deltas=deltas,
            improvements=improvements,
            deteriorations=deteriorations,
        )

    # ===== REPORT GENERATION =====

    def generate_report(self, data: FinancialData,
                         prior_data: Optional[FinancialData] = None) -> FinancialReport:
        """Generate a structured financial analysis report.

        Returns a FinancialReport with sections for executive summary,
        ratio analysis, scoring models, risk assessment, and recommendations.
        """
        sections: Dict[str, str] = {}

        results = self.analyze(data)
        # Reuse health score from analyze() when no prior_data; recompute only
        # when prior_data is supplied (adds period-comparison delta).
        if prior_data is not None:
            health = self.composite_health_score(data, prior_data)
        else:
            health = results['composite_health']

        # --- Executive Summary ---
        lines = [f"Overall Financial Health: {health.grade} ({health.score}/100)"]
        z = results['altman_z_score']
        if z.z_score is not None:
            lines.append(f"Bankruptcy Risk: {z.zone.title()} (Z-Score: {z.z_score:.2f})")
        f_result = results['piotroski_f_score']
        lines.append(f"Financial Strength: {f_result.score}/{f_result.max_score} (Piotroski F-Score)")

        prof = results['profitability_ratios']
        if prof.get('net_margin') is not None:
            lines.append(f"Net Margin: {prof['net_margin']:.1%}")
        if prof.get('roe') is not None:
            lines.append(f"Return on Equity: {prof['roe']:.1%}")

        sections['executive_summary'] = '\n'.join(lines)

        # --- Ratio Analysis ---
        ratio_lines: List[str] = []
        for category, ratios in [
            ('Liquidity', results['liquidity_ratios']),
            ('Profitability', results['profitability_ratios']),
            ('Leverage', results['leverage_ratios']),
            ('Efficiency', results['efficiency_ratios']),
        ]:
            available = {k: v for k, v in ratios.items() if v is not None}
            if available:
                ratio_lines.append(f"\n{category}:")
                for k, v in available.items():
                    label = k.replace('_', ' ').title()
                    if any(x in k for x in ('margin', 'roe', 'roa', 'roic')):
                        ratio_lines.append(f"  {label}: {v:.1%}")
                    else:
                        ratio_lines.append(f"  {label}: {v:.2f}")

        sections['ratio_analysis'] = '\n'.join(ratio_lines) if ratio_lines else 'Insufficient data for ratio analysis.'

        # --- Scoring Models ---
        scoring_lines: List[str] = []
        dupont = results['dupont']
        if dupont.roe is not None:
            scoring_lines.append(f"DuPont ROE: {dupont.roe:.1%}")
            if dupont.net_margin is not None:
                scoring_lines.append(f"  Net Margin: {dupont.net_margin:.1%}")
            if dupont.asset_turnover is not None:
                scoring_lines.append(f"  Asset Turnover: {dupont.asset_turnover:.2f}x")
            if dupont.equity_multiplier is not None:
                scoring_lines.append(f"  Equity Multiplier: {dupont.equity_multiplier:.2f}x")

        if z.z_score is not None:
            scoring_lines.append(f"\nAltman Z-Score: {z.z_score:.2f} ({z.zone})")
        scoring_lines.append(f"Piotroski F-Score: {f_result.score}/{f_result.max_score}")
        scoring_lines.append(f"Composite Health: {health.score}/100 (Grade {health.grade})")

        sections['scoring_models'] = '\n'.join(scoring_lines)

        # --- Risk Assessment ---
        risk_lines: List[str] = []
        insights = results['insights']
        warnings = [i for i in insights if i.severity in ('warning', 'critical')]
        if warnings:
            for w in warnings:
                risk_lines.append(f"[{w.severity.upper()}] {w.message}")
        else:
            risk_lines.append("No significant risk factors identified.")
        sections['risk_assessment'] = '\n'.join(risk_lines)

        # --- Recommendations ---
        recs = [i.recommendation for i in insights if i.recommendation]
        sections['recommendations'] = (
            '\n'.join(f"- {r}" for r in recs)
            if recs
            else 'No specific recommendations at this time.'
        )

        # --- Period Comparison (if prior data available) ---
        if prior_data is not None:
            comparison = self.compare_periods(data, prior_data)
            comp_lines: List[str] = []
            if comparison.improvements:
                comp_lines.append("Improvements:")
                for m in comparison.improvements[:5]:
                    delta = comparison.deltas[m]
                    comp_lines.append(f"  + {m.replace('_', ' ').title()}: {delta:+.4f}")
            if comparison.deteriorations:
                comp_lines.append("Deteriorations:")
                for m in comparison.deteriorations[:5]:
                    delta = comparison.deltas[m]
                    comp_lines.append(f"  - {m.replace('_', ' ').title()}: {delta:+.4f}")
            sections['period_comparison'] = (
                '\n'.join(comp_lines)
                if comp_lines
                else 'No significant changes detected.'
            )

        return FinancialReport(
            sections=sections,
            executive_summary=sections['executive_summary'],
            generated_at=datetime.now().isoformat(),
        )

    # ===== INSIGHT GENERATION =====

    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[Insight]:
        """
        Generate natural language insights using Charlie Munger principles.
        """
        insights = []

        # Liquidity insights
        ratios = analysis_results.get('liquidity_ratios', {})
        if ratios.get('current_ratio') is not None:
            cr = ratios['current_ratio']
            if cr >= 1.5:
                insights.append(Insight(
                    category='liquidity',
                    severity='info',
                    message=f"Strong liquidity: Current ratio of {cr:.2f} indicates healthy ability to meet short-term obligations.",
                    metric_name='current_ratio',
                    metric_value=cr
                ))
            elif cr < 1.0:
                insights.append(Insight(
                    category='liquidity',
                    severity='warning',
                    message=f"Liquidity concern: Current ratio of {cr:.2f} below 1.0 suggests potential difficulty meeting short-term obligations.",
                    metric_name='current_ratio',
                    metric_value=cr,
                    recommendation="Review working capital management and consider improving collection cycles or negotiating extended payment terms."
                ))

        # Profitability insights
        prof_ratios = analysis_results.get('profitability_ratios', {})
        if prof_ratios.get('net_margin') is not None:
            margin = prof_ratios['net_margin']
            if margin > 0.15:
                insights.append(Insight(
                    category='profitability',
                    severity='info',
                    message=f"Strong profitability: Net margin of {margin:.1%} indicates excellent bottom-line performance.",
                    metric_name='net_margin',
                    metric_value=margin
                ))
            elif margin < 0.02:
                insights.append(Insight(
                    category='profitability',
                    severity='warning',
                    message=f"Thin margins: Net margin of {margin:.1%} leaves little room for error.",
                    metric_name='net_margin',
                    metric_value=margin,
                    recommendation="Analyze cost structure and pricing strategy. Consider operational efficiency improvements."
                ))

        # ROE insights (Charlie Munger focus)
        if prof_ratios.get('roe') is not None:
            roe = prof_ratios['roe']
            if roe > 0.20:
                insights.append(Insight(
                    category='profitability',
                    severity='info',
                    message=f"Excellent capital efficiency: ROE of {roe:.1%} suggests strong competitive advantages.",
                    metric_name='roe',
                    metric_value=roe,
                    recommendation="Investigate sources of high returns - sustainable competitive moats or temporary factors?"
                ))

        # Leverage insights
        lev_ratios = analysis_results.get('leverage_ratios', {})
        if lev_ratios.get('debt_to_equity') is not None:
            de = lev_ratios['debt_to_equity']
            if de > 2.0:
                insights.append(Insight(
                    category='risk',
                    severity='warning',
                    message=f"High leverage: Debt-to-equity of {de:.2f} increases financial risk.",
                    metric_name='debt_to_equity',
                    metric_value=de,
                    recommendation="Monitor debt service coverage. Consider deleveraging if cash flows are volatile."
                ))

        # Cash flow insights (Charlie Munger emphasis)
        cf_analysis = analysis_results.get('cash_flow', None)
        if cf_analysis and cf_analysis.free_cash_flow is not None:
            fcf = cf_analysis.free_cash_flow
            if fcf > 0:
                insights.append(Insight(
                    category='cash_flow',
                    severity='info',
                    message=f"Positive free cash flow of {fcf:,.0f} provides strategic flexibility.",
                    metric_name='free_cash_flow',
                    metric_value=fcf,
                    recommendation="Strong cash generation enables reinvestment, debt reduction, or shareholder returns."
                ))
            else:
                insights.append(Insight(
                    category='cash_flow',
                    severity='warning',
                    message=f"Negative free cash flow of {fcf:,.0f} requires external financing or asset monetization.",
                    metric_name='free_cash_flow',
                    metric_value=fcf,
                    recommendation="Investigate cash burn drivers. Assess sustainability of current capital structure."
                ))

        # Cash conversion cycle
        if cf_analysis and cf_analysis.cash_conversion_cycle is not None:
            ccc = cf_analysis.cash_conversion_cycle
            if ccc > 60:
                insights.append(Insight(
                    category='efficiency',
                    severity='warning',
                    message=f"Extended cash conversion cycle of {ccc:.0f} days ties up working capital.",
                    metric_name='cash_conversion_cycle',
                    metric_value=ccc,
                    recommendation="Optimize inventory management, accelerate collections, and negotiate better payment terms."
                ))

        # DuPont decomposition insights
        dupont = analysis_results.get('dupont')
        if dupont and dupont.roe is not None:
            insights.append(Insight(
                category='profitability',
                severity='info',
                message=dupont.interpretation or f"DuPont ROE: {dupont.roe:.1%}",
                metric_name='dupont_roe',
                metric_value=dupont.roe,
                recommendation=(
                    f"Primary ROE driver is {dupont.primary_driver.replace('_', ' ')}. "
                    "Focus improvement efforts there."
                ) if dupont.primary_driver else None
            ))

        # Altman Z-Score insights
        z_result = analysis_results.get('altman_z_score')
        if z_result and z_result.z_score is not None:
            severity = 'info'
            if z_result.zone == 'distress':
                severity = 'critical'
            elif z_result.zone == 'grey':
                severity = 'warning'

            rec = None
            if z_result.zone == 'distress':
                rec = "Urgent: Review capital structure, reduce leverage, and improve profitability immediately."
            elif z_result.zone == 'grey':
                rec = "Monitor closely. Strengthen working capital and profitability to move into the safe zone."

            insights.append(Insight(
                category='risk',
                severity=severity,
                message=z_result.interpretation or f"Altman Z-Score: {z_result.z_score:.2f}",
                metric_name='altman_z_score',
                metric_value=z_result.z_score,
                recommendation=rec
            ))

        # Piotroski F-Score insights
        f_result = analysis_results.get('piotroski_f_score')
        if f_result and f_result.score is not None:
            severity = 'info'
            if f_result.score <= 3:
                severity = 'warning'

            # Find weak criteria
            weak = [k.replace('_', ' ') for k, v in f_result.criteria.items() if not v]
            rec = None
            if weak:
                rec = f"Weak areas: {', '.join(weak[:3])}. Address these to strengthen financial position."

            insights.append(Insight(
                category='risk',
                severity=severity,
                message=f_result.interpretation or f"Piotroski F-Score: {f_result.score}/9",
                metric_name='piotroski_f_score',
                metric_value=float(f_result.score),
                recommendation=rec
            ))

        return insights

    def detect_anomalies(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                         method: str = 'zscore', threshold: float = 2.0) -> List[Anomaly]:
        """Flag unusual patterns using z-score or IQR method.

        Args:
            data: DataFrame containing numeric data to analyze.
            columns: Specific columns to check. Defaults to all numeric columns.
            method: Detection method - 'zscore' (default) or 'iqr'.
            threshold: For zscore: number of std devs (default 2.0).
                       For IQR: multiplier of IQR (default 1.5, but uses threshold param).
        """
        anomalies = []

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            series = data[col].dropna()
            if len(series) < 3:
                continue

            # Use the underlying numpy array directly (avoids copy from to_numpy)
            arr: np.ndarray = series.values.astype(float, copy=False)

            if method == 'iqr':
                q1, q3 = np.percentile(arr, [25, 75])  # single pass
                iqr = q3 - q1
                if iqr == 0:
                    continue
                iqr_multiplier = threshold if threshold != 2.0 else 1.5
                lower = float(q1) - iqr_multiplier * float(iqr)
                upper = float(q3) + iqr_multiplier * float(iqr)

                mean = float(arr.mean())
                # ddof=1 matches pandas Series.std() default; 'or 1.0' avoids div-by-zero
                std = float(np.std(arr, ddof=1)) or 1.0

                mask = (arr < lower) | (arr > upper)
                outlier_vals = arr[mask]
                z_scores = (outlier_vals - mean) / std

                for value, z in zip(outlier_vals.tolist(), z_scores.tolist()):
                    anomalies.append(Anomaly(
                        metric_name=col,
                        value=value,
                        expected_range=(lower, upper),
                        z_score=z,
                        description=(f"IQR anomaly in {col}: {value:.2f} outside "
                                     f"[{lower:.2f}, {upper:.2f}]")
                    ))
            else:
                # Default z-score method
                mean = float(arr.mean())
                std = float(np.std(arr, ddof=1))
                if std == 0:
                    continue

                z_scores = (arr - mean) / std
                mask = np.abs(z_scores) > threshold
                outlier_vals = arr[mask]
                outlier_zs = z_scores[mask]
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

                for value, z_score in zip(outlier_vals.tolist(), outlier_zs.tolist()):
                    anomalies.append(Anomaly(
                        metric_name=col,
                        value=value,
                        expected_range=(lower_bound, upper_bound),
                        z_score=z_score,
                        description=(f"Unusual value in {col}: {value:.2f} is "
                                     f"{abs(z_score):.1f} standard deviations from mean")
                    ))

        return anomalies

    # ===== SCENARIO / WHAT-IF ANALYSIS =====

    def _apply_adjustments(self, data: FinancialData,
                           adjustments: Dict[str, float]) -> FinancialData:
        """Create a copy of FinancialData with percentage adjustments applied.

        Args:
            data: Base financial data.
            adjustments: Dict mapping field names to multipliers.
                         e.g. {'revenue': 1.10} means +10% revenue.

        Returns:
            Adjusted copy.  Sets ``_skipped_adjustments`` attribute on the
            returned object listing field names that were None in base data
            (and therefore could not be adjusted).
        """
        from dataclasses import fields as dc_fields
        # Shallow copy all fields via dict - FinancialData has only scalar/None fields
        field_values = {f.name: getattr(data, f.name) for f in dc_fields(data)}
        skipped: List[str] = []
        for field_name, multiplier in adjustments.items():
            current_val = field_values.get(field_name)
            if current_val is not None:
                field_values[field_name] = current_val * multiplier
            else:
                skipped.append(field_name)
        adjusted = FinancialData(**field_values)
        adjusted._skipped_adjustments = skipped  # type: ignore[attr-defined]
        return adjusted

    def scenario_analysis(self, data: FinancialData,
                          adjustments: Dict[str, float],
                          scenario_name: str = "Custom Scenario") -> ScenarioResult:
        """Run what-if analysis by applying adjustments and comparing all metrics.

        Args:
            data: Base FinancialData.
            adjustments: Field-name-to-multiplier map (e.g. {'revenue': 1.10}).
            scenario_name: Human-readable label for the scenario.

        Returns:
            ScenarioResult comparing base vs scenario across all key metrics.
        """
        adjusted = self._apply_adjustments(data, adjustments)

        # Base metrics
        base_health = self.composite_health_score(data)
        base_z = self.altman_z_score(data)
        base_f = self.piotroski_f_score(data)
        base_liq = self.calculate_liquidity_ratios(data)
        base_prof = self.calculate_profitability_ratios(data)
        base_lev = self.calculate_leverage_ratios(data)

        # Scenario metrics
        scen_health = self.composite_health_score(adjusted)
        scen_z = self.altman_z_score(adjusted)
        scen_f = self.piotroski_f_score(adjusted)
        scen_liq = self.calculate_liquidity_ratios(adjusted)
        scen_prof = self.calculate_profitability_ratios(adjusted)
        scen_lev = self.calculate_leverage_ratios(adjusted)

        # Collect key ratios
        base_ratios: Dict[str, Optional[float]] = {}
        scenario_ratios: Dict[str, Optional[float]] = {}
        for prefix, base_r, scen_r in [
            ('', base_liq, scen_liq),
            ('', base_prof, scen_prof),
            ('', base_lev, scen_lev),
        ]:
            for k, v in base_r.items():
                base_ratios[k] = v
            for k, v in scen_r.items():
                scenario_ratios[k] = v

        base_ratios['health_score'] = float(base_health.score)
        scenario_ratios['health_score'] = float(scen_health.score)
        base_ratios['z_score'] = base_z.z_score
        scenario_ratios['z_score'] = scen_z.z_score

        # Build impact summary
        health_delta = scen_health.score - base_health.score
        z_delta = ((scen_z.z_score or 0) - (base_z.z_score or 0))
        adj_strs = [f"{k} {'+'if v>=1 else ''}{(v-1)*100:+.0f}%"
                    for k, v in adjustments.items()]
        skipped = getattr(adjusted, '_skipped_adjustments', [])
        summary_parts = [
            f"Scenario '{scenario_name}': {', '.join(adj_strs)}.",
            f"Health score: {base_health.score} -> {scen_health.score} ({health_delta:+d}).",
            f"Z-Score: {base_z.z_score:.2f} -> {scen_z.z_score:.2f} ({z_delta:+.2f})." if base_z.z_score and scen_z.z_score else "",
            f"Grade: {base_health.grade} -> {scen_health.grade}.",
        ]
        if skipped:
            summary_parts.append(
                f"WARNING: Adjustments skipped (base value is None): {', '.join(skipped)}."
            )

        return ScenarioResult(
            scenario_name=scenario_name,
            adjustments=adjustments,
            base_health=base_health,
            scenario_health=scen_health,
            base_z_score=base_z.z_score,
            scenario_z_score=scen_z.z_score,
            base_f_score=base_f.score,
            scenario_f_score=scen_f.score,
            base_ratios=base_ratios,
            scenario_ratios=scenario_ratios,
            impact_summary=' '.join(p for p in summary_parts if p),
        )

    def multi_scenario_analysis(
        self,
        data: FinancialData,
        scenarios: Dict[str, Dict[str, float]],
    ) -> List[ScenarioResult]:
        """Run multiple named scenarios in one call.

        Args:
            data: Base FinancialData.
            scenarios: Map of scenario_name -> adjustments dict.
                       e.g. {"Bull": {"revenue": 1.10}, "Bear": {"revenue": 0.90}}

        Returns:
            List of ScenarioResult, one per scenario.
        """
        results = []
        for name, adjustments in scenarios.items():
            results.append(self.scenario_analysis(data, adjustments, scenario_name=name))
        return results

    def probability_weighted_scenarios(
        self,
        data: FinancialData,
        scenario_probs: List[Dict[str, Any]],
    ) -> ProbabilityWeightedResult:
        """Assign probabilities to scenarios and compute expected outcomes.

        Args:
            data: Base FinancialData.
            scenario_probs: List of dicts, each with keys:
                - "name": scenario name
                - "adjustments": field-to-multiplier map
                - "probability": 0.0-1.0 weight

        Returns:
            ProbabilityWeightedResult with expected scores and distress probability.
        """
        if not scenario_probs:
            return ProbabilityWeightedResult(summary="No scenarios provided.")

        # Normalise probabilities to sum to 1.0
        raw_probs = [s.get("probability", 0.0) for s in scenario_probs]
        total_prob = sum(raw_probs) or 1.0
        probs = [p / total_prob for p in raw_probs]

        results: List[ScenarioResult] = []
        names: List[str] = []
        for sp in scenario_probs:
            name = sp.get("name", "Scenario")
            adjustments = sp.get("adjustments", {})
            results.append(self.scenario_analysis(data, adjustments, scenario_name=name))
            names.append(name)

        # Expected values
        expected_health = sum(
            p * (r.scenario_health.score if r.scenario_health else 0)
            for p, r in zip(probs, results)
        )
        expected_z = sum(
            p * (r.scenario_z_score or 0)
            for p, r in zip(probs, results)
        )

        # Distress probability: sum of probs where Z-Score < 1.81
        distress_prob = sum(
            p for p, r in zip(probs, results)
            if r.scenario_z_score is not None and r.scenario_z_score < 1.81
        )

        parts = [f"Prob-weighted across {len(results)} scenarios."]
        parts.append(f"Expected health: {expected_health:.1f}, expected Z: {expected_z:.2f}.")
        if distress_prob > 0.2:
            parts.append(f"WARNING: {distress_prob:.0%} probability of financial distress.")

        return ProbabilityWeightedResult(
            scenarios=results,
            probabilities=probs,
            scenario_names=names,
            expected_health_score=round(expected_health, 1),
            expected_z_score=round(expected_z, 2),
            distress_probability=round(distress_prob, 4),
            summary=" ".join(parts),
        )

    def sensitivity_analysis(self, data: FinancialData, variable: str,
                             pct_range: Optional[List[float]] = None) -> SensitivityResult:
        """Vary a single financial variable and measure impact on key metrics.

        Args:
            data: Base FinancialData.
            variable: Field name to vary (e.g. 'revenue').
            pct_range: List of percentage changes to test.
                       Defaults to [-20, -15, -10, -5, 0, +5, +10, +15, +20].

        Returns:
            SensitivityResult with metric values at each variation point.
        """
        if pct_range is None:
            pct_range = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

        labels = [f"{p:+d}%" for p in pct_range]
        multipliers = [1 + p / 100.0 for p in pct_range]

        metrics: Dict[str, List[Optional[float]]] = {
            'health_score': [],
            'z_score': [],
            'f_score': [],
            'current_ratio': [],
            'net_margin': [],
            'debt_to_equity': [],
            'roe': [],
        }

        for mult in multipliers:
            adjusted = self._apply_adjustments(data, {variable: mult})
            health = self.composite_health_score(adjusted)
            z = self.altman_z_score(adjusted)
            f = self.piotroski_f_score(adjusted)
            liq = self.calculate_liquidity_ratios(adjusted)
            prof = self.calculate_profitability_ratios(adjusted)
            lev = self.calculate_leverage_ratios(adjusted)

            metrics['health_score'].append(float(health.score))
            metrics['z_score'].append(z.z_score)
            metrics['f_score'].append(float(f.score))
            metrics['current_ratio'].append(liq.get('current_ratio'))
            metrics['net_margin'].append(prof.get('net_margin'))
            metrics['debt_to_equity'].append(lev.get('debt_to_equity'))
            metrics['roe'].append(prof.get('roe'))

        return SensitivityResult(
            variable_name=variable,
            variable_labels=labels,
            variable_multipliers=multipliers,
            metric_results=metrics,
        )

    # ===== MONTE CARLO SIMULATION =====

    def monte_carlo_simulation(
        self,
        data: FinancialData,
        assumptions: Optional[Dict[str, Dict[str, float]]] = None,
        n_simulations: int = 1000,
        seed: Optional[int] = 42,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation to model uncertainty in financial outcomes.

        Args:
            data: Base FinancialData.
            assumptions: Dict mapping field names to distribution params.
                         Each entry: {'mean_pct': 0.0, 'std_pct': 10.0}
                         meaning the variable is normally distributed around
                         base value with std of 10% of base.
                         Defaults to revenue/cogs/operating_expenses at 10% std.
            n_simulations: Number of random scenarios to run (default 1000).
            seed: Random seed for reproducibility. Pass None for non-deterministic runs.

        Returns:
            MonteCarloResult with distributions and percentiles.
        """
        n_simulations = min(n_simulations, 10_000)  # cap to prevent thread exhaustion

        if assumptions is None:
            assumptions = {}
            for fld in ['revenue', 'cogs', 'operating_expenses']:
                if getattr(data, fld, None) is not None:
                    assumptions[fld] = {'mean_pct': 0.0, 'std_pct': 10.0}

        if not assumptions:
            return MonteCarloResult(
                n_simulations=0,
                variable_assumptions=assumptions,
                summary="No variables with data to simulate.",
            )

        rng = np.random.default_rng(seed=seed)

        # --- WP-D optimizations (P1-A2) ---
        # 1. Cache dataclasses.fields() once rather than re-calling inside
        #    _apply_adjustments on every iteration.
        from dataclasses import fields as _dc_fields
        _base_field_values: Dict[str, Any] = {
            f.name: getattr(data, f.name) for f in _dc_fields(data)
        }
        # Pre-compute assumption keys and per-field draw parameters (once).
        _assumption_keys: List[str] = list(assumptions.keys())
        _k = len(_assumption_keys)
        _mean_mults = np.array(
            [1.0 + assumptions[fld].get('mean_pct', 0.0) / 100.0
             for fld in _assumption_keys],
            dtype=float,
        )
        _std_mults = np.array(
            [assumptions[fld].get('std_pct', 10.0) / 100.0
             for fld in _assumption_keys],
            dtype=float,
        )
        # 2. Batch-draw all random samples in one call.
        #    size=(n_simulations, k) draws in C (row-major) order:
        #      row=sim_index, col=field_index
        #    This is identical to the original nested-loop draw order
        #    (outer=sim, inner=field), so the same seed produces the same values.
        _all_samples = np.maximum(
            rng.normal(_mean_mults, _std_mults, size=(n_simulations, _k)),
            0.001,  # floor at 0.1% — allows near-total loss modeling
        )

        metrics_collected: Dict[str, List[float]] = {
            'health_score': [],
            'z_score': [],
            'f_score': [],
            'net_margin': [],
            'current_ratio': [],
            'roe': [],
        }

        for _sim_idx in range(n_simulations):
            # Build adjustments dict from pre-drawn row
            adjustments: Dict[str, float] = {
                _assumption_keys[_j]: float(_all_samples[_sim_idx, _j])
                for _j in range(_k)
            }

            # Apply adjustments using the cached field-values dict
            _field_values = _base_field_values.copy()
            _skipped: List[str] = []
            for _fld_name, _multiplier in adjustments.items():
                _cur = _field_values.get(_fld_name)
                if _cur is not None:
                    _field_values[_fld_name] = _cur * _multiplier
                else:
                    _skipped.append(_fld_name)
            adjusted = FinancialData(**_field_values)
            adjusted._skipped_adjustments = _skipped  # type: ignore[attr-defined]

            health = self.composite_health_score(adjusted)
            z = self.altman_z_score(adjusted)
            f = self.piotroski_f_score(adjusted)
            prof = self.calculate_profitability_ratios(adjusted)
            liq = self.calculate_liquidity_ratios(adjusted)

            metrics_collected['health_score'].append(float(health.score))
            metrics_collected['z_score'].append(z.z_score if z.z_score is not None else 0.0)
            metrics_collected['f_score'].append(float(f.score))
            metrics_collected['net_margin'].append(prof.get('net_margin') or 0.0)
            metrics_collected['current_ratio'].append(liq.get('current_ratio') or 0.0)
            metrics_collected['roe'].append(prof.get('roe') or 0.0)

        # Compute percentiles
        percentiles: Dict[str, Dict[str, float]] = {}
        for metric, values in metrics_collected.items():
            arr = np.array(values)
            percentiles[metric] = {
                'p10': float(np.percentile(arr, 10)),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'p90': float(np.percentile(arr, 90)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
            }

        # Summary
        hs = percentiles.get('health_score', {})
        summary = (
            f"Monte Carlo ({n_simulations} simulations): "
            f"Health Score median={hs.get('p50', 0):.0f}, "
            f"range=[{hs.get('p10', 0):.0f}, {hs.get('p90', 0):.0f}] (10th-90th). "
            f"Variables: {', '.join(assumptions.keys())}."
        )

        return MonteCarloResult(
            n_simulations=n_simulations,
            variable_assumptions=assumptions,
            metric_distributions=metrics_collected,
            percentiles=percentiles,
            summary=summary,
        )

    # ===== CASH FLOW FORECASTING =====

    def forecast_cashflow(
        self,
        data: FinancialData,
        periods: int = 12,
        revenue_growth: float = 0.05,
        expense_ratio: Optional[float] = None,
        capex_ratio: float = 0.05,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.02,
    ) -> CashFlowForecast:
        """Forecast cash flows over multiple periods with DCF valuation.

        Args:
            data: Base FinancialData with current financials.
            periods: Number of forecast periods (default 12).
            revenue_growth: Annual revenue growth rate (default 5%).
            expense_ratio: Operating expenses as fraction of revenue.
                          If None, derived from current data.
            capex_ratio: CapEx as fraction of revenue (default 5%).
            discount_rate: WACC / required return (default 10%).
            terminal_growth: Long-term growth for terminal value (default 2%).

        Returns:
            CashFlowForecast with period-by-period projections and DCF value.
        """
        base_revenue = data.revenue or 0.0
        if base_revenue <= 0:
            return CashFlowForecast(periods=[])

        # A discount rate at or below -100% makes (1 + discount_rate) ** i
        # zero or negative, leaving the DCF undefined. Return an empty
        # forecast (dcf_value/terminal_value None) rather than dividing by
        # zero or producing inf/nan.
        if discount_rate <= -1.0:
            return CashFlowForecast(periods=[])

        # Derive expense ratio from current data if not provided
        if expense_ratio is None:
            total_expenses = (data.cogs or 0) + (data.operating_expenses or 0)
            expense_ratio = safe_divide(total_expenses, base_revenue, default=0.70) or 0.70

        period_labels = []
        rev_forecast = []
        exp_forecast = []
        net_cf = []
        cum_cash = []
        fcf_list = []
        growth_rates = []

        current_rev = base_revenue
        cumulative = 0.0
        pv_fcf_sum = 0.0

        for i in range(1, periods + 1):
            current_rev = current_rev * (1 + revenue_growth)
            expenses = current_rev * expense_ratio
            capex = current_rev * capex_ratio
            fcf = current_rev - expenses - capex
            net = current_rev - expenses
            cumulative += net

            # PV of FCF for DCF
            pv_fcf = fcf / ((1 + discount_rate) ** i)
            pv_fcf_sum += pv_fcf

            period_labels.append(f"Period {i}")
            rev_forecast.append(round(current_rev, 2))
            exp_forecast.append(round(expenses, 2))
            net_cf.append(round(net, 2))
            cum_cash.append(round(cumulative, 2))
            fcf_list.append(round(fcf, 2))
            growth_rates.append(round(revenue_growth * 100, 2))

        # Terminal value (Gordon Growth Model)
        # When discount_rate <= terminal_growth, GGM is undefined (infinite
        # value).  We return None to signal the omission rather than a
        # misleading 0.0.
        if discount_rate > terminal_growth and fcf_list:
            terminal_fcf = fcf_list[-1] * (1 + terminal_growth)
            tv = terminal_fcf / (discount_rate - terminal_growth)
            pv_tv = tv / ((1 + discount_rate) ** periods)
        else:
            tv = None
            pv_tv = 0.0

        dcf_value = pv_fcf_sum + pv_tv

        return CashFlowForecast(
            periods=period_labels,
            revenue_forecast=rev_forecast,
            expense_forecast=exp_forecast,
            net_cash_flow=net_cf,
            cumulative_cash=cum_cash,
            fcf_forecast=fcf_list,
            dcf_value=round(dcf_value, 2),
            terminal_value=round(tv, 2) if tv is not None else None,
            discount_rate=discount_rate,
            growth_rates=growth_rates,
        )

    # ===== TORNADO / DRIVER ANALYSIS =====

    def tornado_analysis(
        self,
        data: FinancialData,
        target_metric: str = "health_score",
        variables: Optional[List[str]] = None,
        pct_swing: float = 10.0,
    ) -> TornadoResult:
        """Rank which financial variables have the greatest impact on a target metric.

        For each variable, applies +/- pct_swing% and measures the resulting
        change in the target metric, then ranks by spread (high - low).

        Args:
            data: Base FinancialData.
            target_metric: Metric to measure impact on. One of:
                'health_score', 'z_score', 'f_score', 'net_margin',
                'current_ratio', 'roe', 'debt_to_equity'.
            variables: List of FinancialData field names to vary.
                       If None, auto-detects non-None numeric fields.
            pct_swing: Percentage swing to apply (+/- this value).

        Returns:
            TornadoResult with drivers ranked by spread (descending).
        """
        # Auto-detect variables if not provided
        if variables is None:
            candidate_fields = [
                'revenue', 'cogs', 'operating_expenses', 'net_income',
                'total_assets', 'total_liabilities', 'total_equity',
                'current_assets', 'current_liabilities', 'total_debt',
                'ebit', 'ebitda', 'interest_expense', 'depreciation',
                'inventory', 'accounts_receivable', 'accounts_payable',
                'capex', 'operating_cash_flow', 'retained_earnings',
            ]
            variables = [f for f in candidate_fields
                         if getattr(data, f, None) is not None]

        if not variables:
            return TornadoResult(target_metric=target_metric)

        def _get_metric_value(d: FinancialData) -> float:
            """Extract a single metric value from FinancialData."""
            if target_metric == 'health_score':
                return float(self.composite_health_score(d).score)
            elif target_metric == 'z_score':
                return self.altman_z_score(d).z_score or 0.0
            elif target_metric == 'f_score':
                return float(self.piotroski_f_score(d).score)
            elif target_metric == 'net_margin':
                return self.calculate_profitability_ratios(d).get('net_margin') or 0.0
            elif target_metric == 'current_ratio':
                return self.calculate_liquidity_ratios(d).get('current_ratio') or 0.0
            elif target_metric == 'roe':
                return self.calculate_profitability_ratios(d).get('roe') or 0.0
            elif target_metric == 'debt_to_equity':
                return self.calculate_leverage_ratios(d).get('debt_to_equity') or 0.0
            return 0.0

        base_value = _get_metric_value(data)
        low_mult = 1 - pct_swing / 100.0
        high_mult = 1 + pct_swing / 100.0

        drivers: List[TornadoDriver] = []
        for var in variables:
            low_data = self._apply_adjustments(data, {var: low_mult})
            high_data = self._apply_adjustments(data, {var: high_mult})
            low_val = _get_metric_value(low_data)
            high_val = _get_metric_value(high_data)
            spread = abs(high_val - low_val)
            drivers.append(TornadoDriver(
                variable=var,
                low_value=round(low_val, 4),
                high_value=round(high_val, 4),
                base_value=round(base_value, 4),
                spread=round(spread, 4),
            ))

        # Sort by spread descending (most impactful first)
        drivers.sort(key=lambda d: d.spread, reverse=True)
        top_driver = drivers[0].variable if drivers else ""

        return TornadoResult(
            target_metric=target_metric,
            base_metric_value=round(base_value, 4),
            drivers=drivers,
            top_driver=top_driver,
        )

    # ===== BREAKEVEN ANALYSIS =====

    def breakeven_analysis(self, data: FinancialData) -> BreakevenResult:
        """Calculate breakeven revenue using contribution margin approach.

        Estimates fixed and variable costs from the financial data, then
        computes the revenue needed to cover all fixed costs (zero profit).

        Args:
            data: FinancialData with revenue, cogs, and operating_expenses.

        Returns:
            BreakevenResult with breakeven revenue and margin of safety.
        """
        revenue = data.revenue or 0.0
        if revenue <= 0:
            return BreakevenResult()

        cogs = data.cogs or 0.0
        opex = data.operating_expenses or 0.0

        # Estimate variable costs = COGS (scales with revenue)
        # Estimate fixed costs = operating expenses (relatively fixed)
        variable_cost_ratio = safe_divide(cogs, revenue, default=0.0) or 0.0
        contribution_margin_ratio = 1.0 - variable_cost_ratio

        if contribution_margin_ratio <= 0:
            return BreakevenResult(
                current_revenue=revenue,
                fixed_costs=opex,
                variable_cost_ratio=round(variable_cost_ratio, 4),
                contribution_margin_ratio=0.0,
            )

        # Fixed costs include opex + interest expense
        fixed_costs = opex + (data.interest_expense or 0.0)

        breakeven_rev = safe_divide(fixed_costs, contribution_margin_ratio, default=None)
        margin_of_safety = None
        if breakeven_rev is not None and revenue > 0:
            margin_of_safety = safe_divide(
                revenue - breakeven_rev, revenue, default=None
            )

        return BreakevenResult(
            breakeven_revenue=round(breakeven_rev, 2) if breakeven_rev is not None else None,
            current_revenue=round(revenue, 2),
            margin_of_safety=round(margin_of_safety, 4) if margin_of_safety is not None else None,
            fixed_costs=round(fixed_costs, 2),
            variable_cost_ratio=round(variable_cost_ratio, 4),
            contribution_margin_ratio=round(contribution_margin_ratio, 4),
        )

    # ===== COVENANT / KPI MONITORING =====

    def covenant_monitor(
        self,
        data: FinancialData,
        covenants: Optional[List[Dict[str, Any]]] = None,
    ) -> CovenantMonitorResult:
        """Check financial data against a set of covenant/KPI thresholds.

        Each covenant specifies a metric, a threshold, and a direction
        ('above' means passing if current >= threshold, 'below' if <=).

        A 'warning' is triggered when the value is within 10% of the threshold.

        Args:
            data: FinancialData to evaluate.
            covenants: List of covenant dicts, each with keys:
                - 'name': Display name
                - 'metric': One of 'current_ratio', 'debt_to_equity',
                  'net_margin', 'roe', 'interest_coverage', 'health_score',
                  'dscr' (debt service coverage ratio).
                - 'threshold': Numeric threshold value
                - 'direction': 'above' or 'below' (default 'above')
                If None, uses standard default covenants.

        Returns:
            CovenantMonitorResult with per-covenant checks and summary counts.
        """
        if covenants is None:
            covenants = [
                {'name': 'Current Ratio', 'metric': 'current_ratio',
                 'threshold': 1.5, 'direction': 'above'},
                {'name': 'Debt-to-Equity', 'metric': 'debt_to_equity',
                 'threshold': 2.0, 'direction': 'below'},
                {'name': 'Net Margin', 'metric': 'net_margin',
                 'threshold': 0.05, 'direction': 'above'},
                {'name': 'Interest Coverage', 'metric': 'interest_coverage',
                 'threshold': 3.0, 'direction': 'above'},
                {'name': 'Health Score', 'metric': 'health_score',
                 'threshold': 50.0, 'direction': 'above'},
            ]

        # Gather all metric values
        liq = self.calculate_liquidity_ratios(data)
        prof = self.calculate_profitability_ratios(data)
        lev = self.calculate_leverage_ratios(data)
        health = self.composite_health_score(data)

        # DSCR = EBITDA / (interest_expense + debt repayment)
        # Simplified: EBITDA / interest_expense if no repayment data
        dscr = safe_divide(data.ebitda, data.interest_expense, default=None)

        metric_values = {
            'current_ratio': liq.get('current_ratio'),
            'quick_ratio': liq.get('quick_ratio'),
            'debt_to_equity': lev.get('debt_to_equity'),
            'debt_to_assets': lev.get('debt_to_assets'),
            'net_margin': prof.get('net_margin'),
            'roe': prof.get('roe'),
            'roa': prof.get('roa'),
            'interest_coverage': lev.get('interest_coverage'),
            'health_score': float(health.score),
            'dscr': dscr,
        }

        checks: List[CovenantCheck] = []
        passes = 0
        warnings = 0
        breaches = 0

        for cov in covenants:
            name = cov.get('name', cov.get('metric', 'Unknown'))
            metric_key = cov.get('metric', '')
            threshold = float(cov.get('threshold', 0))
            direction = cov.get('direction', 'above')

            current = metric_values.get(metric_key)

            if current is None:
                checks.append(CovenantCheck(
                    name=name, current_value=None,
                    threshold=threshold, direction=direction,
                    status='unknown', headroom=None,
                ))
                continue

            # Determine pass/warning/breach
            warning_zone = threshold * 0.10  # 10% of threshold

            if direction == 'above':
                headroom = current - threshold
                if current >= threshold:
                    if current < threshold + warning_zone:
                        status = 'warning'
                        warnings += 1
                    else:
                        status = 'pass'
                        passes += 1
                else:
                    status = 'breach'
                    breaches += 1
            else:  # 'below'
                headroom = threshold - current
                if current <= threshold:
                    if current > threshold - warning_zone:
                        status = 'warning'
                        warnings += 1
                    else:
                        status = 'pass'
                        passes += 1
                else:
                    status = 'breach'
                    breaches += 1

            checks.append(CovenantCheck(
                name=name,
                current_value=round(current, 4),
                threshold=threshold,
                direction=direction,
                status=status,
                headroom=round(headroom, 4),
            ))

        # Build summary
        total = len(checks)
        unknown = total - passes - warnings - breaches
        parts = []
        if breaches > 0:
            parts.append(f"{breaches} BREACH(ES)")
        if warnings > 0:
            parts.append(f"{warnings} warning(s)")
        if passes > 0:
            parts.append(f"{passes} passing")
        if unknown > 0:
            parts.append(f"{unknown} unknown")
        summary = f"Covenant Monitor: {', '.join(parts)}." if parts else "No covenants checked."

        return CovenantMonitorResult(
            checks=checks,
            passes=passes,
            warnings=warnings,
            breaches=breaches,
            summary=summary,
        )

    # ===== WORKING CAPITAL ANALYSIS =====

    def working_capital_analysis(self, data: FinancialData) -> WorkingCapitalResult:
        """Analyze working capital efficiency using days metrics and CCC.

        Calculates DSO, DIO, DPO, Cash Conversion Cycle, and generates
        optimization insights based on industry benchmarks.

        Args:
            data: FinancialData with revenue, COGS, AR, inventory, AP.

        Returns:
            WorkingCapitalResult with days metrics and actionable insights.
        """
        revenue = data.revenue or 0.0
        cogs = data.cogs or 0.0
        ar = data.accounts_receivable or data.avg_receivables
        inv = data.inventory or data.avg_inventory
        ap = data.accounts_payable or data.avg_payables
        current_assets = data.current_assets
        current_liabilities = data.current_liabilities

        insights: List[str] = []

        # DSO = Accounts Receivable / Revenue * 365
        dso = None
        if ar is not None and revenue is not None:
            raw_dso = safe_divide(ar, revenue)
            if raw_dso is not None:
                dso = round(raw_dso * 365, 1)
            if dso is not None:
                if dso > 60:
                    insights.append(f"DSO of {dso:.0f} days is high; consider tightening payment terms or improving collections.")
                elif dso > 45:
                    insights.append(f"DSO of {dso:.0f} days is moderate; monitor for deterioration.")
                else:
                    insights.append(f"DSO of {dso:.0f} days is healthy.")

        # DIO = Inventory / COGS * 365
        dio = None
        if inv is not None and cogs is not None:
            dio_raw = safe_divide(inv, cogs)
            dio = round(dio_raw * 365, 1) if dio_raw is not None else None
        if dio is not None:
            if dio > 90:
                insights.append(f"DIO of {dio:.0f} days indicates slow inventory turnover; review SKU performance.")
            elif dio > 60:
                insights.append(f"DIO of {dio:.0f} days is moderate.")
            else:
                insights.append(f"DIO of {dio:.0f} days shows efficient inventory management.")

        # DPO = Accounts Payable / COGS * 365
        dpo = None
        if ap is not None and cogs is not None:
            dpo_raw = safe_divide(ap, cogs)
            dpo = round(dpo_raw * 365, 1) if dpo_raw is not None else None
        if dpo is not None:
            if dpo < 30:
                insights.append(f"DPO of {dpo:.0f} days is low; negotiate longer payment terms with suppliers.")
            elif dpo > 90:
                insights.append(f"DPO of {dpo:.0f} days is very long; verify supplier relationships are healthy.")
            else:
                insights.append(f"DPO of {dpo:.0f} days is within normal range.")

        # CCC = DSO + DIO - DPO
        ccc = None
        if dso is not None and dio is not None and dpo is not None:
            ccc = round(dso + dio - dpo, 1)
            if ccc < 0:
                insights.append(f"CCC of {ccc:.0f} days (negative): company collects cash before paying suppliers -- favorable.")
            elif ccc < 30:
                insights.append(f"CCC of {ccc:.0f} days is efficient.")
            elif ccc < 60:
                insights.append(f"CCC of {ccc:.0f} days is moderate; look for optimization opportunities.")
            else:
                insights.append(f"CCC of {ccc:.0f} days is high; cash is tied up in the operating cycle.")

        # Net Working Capital
        nwc = None
        if current_assets is not None and current_liabilities is not None:
            nwc = round(current_assets - current_liabilities, 2)

        # Working Capital Ratio (same as current ratio but included here for context)
        wc_ratio = safe_divide(current_assets, current_liabilities, default=None)
        if wc_ratio is not None:
            wc_ratio = round(wc_ratio, 4)

        return WorkingCapitalResult(
            dso=dso,
            dio=dio,
            dpo=dpo,
            ccc=ccc,
            net_working_capital=nwc,
            working_capital_ratio=wc_ratio,
            insights=insights,
        )

    # ===== NARRATIVE INTELLIGENCE =====

    def generate_narrative(self, data: FinancialData) -> NarrativeReport:
        """Auto-generate a SWOT-style narrative from financial analysis.

        Synthesizes all major metrics into human-readable strengths, weaknesses,
        opportunities, risks, and a one-line recommendation.

        Args:
            data: FinancialData with sufficient metrics for analysis.

        Returns:
            NarrativeReport with categorized insights and recommendation.
        """
        strengths: List[str] = []
        weaknesses: List[str] = []
        opportunities: List[str] = []
        risks: List[str] = []

        # Health score
        health = self.composite_health_score(data)
        score = health.score
        grade = health.grade

        if score >= 70:
            headline = f"Strong financial position (Score: {score}/100, Grade: {grade})"
            strengths.append(f"Overall health score of {score}/100 ({grade}) indicates robust fundamentals.")
        elif score >= 50:
            headline = f"Moderate financial position (Score: {score}/100, Grade: {grade})"
            opportunities.append("Room to improve financial health through targeted interventions.")
        else:
            headline = f"Weak financial position (Score: {score}/100, Grade: {grade})"
            risks.append(f"Low health score of {score}/100 signals potential financial distress.")

        # Z-Score
        z = self.altman_z_score(data)
        if z.zone == 'safe':
            strengths.append(f"Altman Z-Score of {z.z_score:.2f} places company in the safe zone (low bankruptcy risk).")
        elif z.zone == 'grey':
            risks.append(f"Z-Score of {z.z_score:.2f} is in the grey zone; monitor for deterioration.")
        elif z.zone == 'distress':
            risks.append(f"Z-Score of {z.z_score:.2f} indicates financial distress; immediate attention needed.")

        # Profitability
        prof = self.calculate_profitability_ratios(data)
        nm = prof.get('net_margin')
        if nm is not None:
            if nm > 0.15:
                strengths.append(f"Net margin of {nm*100:.1f}% demonstrates strong profitability.")
            elif nm > 0.05:
                strengths.append(f"Net margin of {nm*100:.1f}% is adequate.")
            elif nm > 0:
                weaknesses.append(f"Thin net margin of {nm*100:.1f}% leaves little room for error.")
            else:
                weaknesses.append(f"Negative net margin of {nm*100:.1f}% indicates the company is unprofitable.")

        roe = prof.get('roe')
        if roe is not None:
            if roe > 0.15:
                strengths.append(f"ROE of {roe*100:.1f}% shows efficient use of shareholder equity.")
            elif roe < 0:
                weaknesses.append(f"Negative ROE of {roe*100:.1f}% means equity is being destroyed.")

        # Liquidity
        liq = self.calculate_liquidity_ratios(data)
        cr = liq.get('current_ratio')
        if cr is not None:
            if cr >= 2.0:
                strengths.append(f"Current ratio of {cr:.2f} provides strong short-term liquidity.")
            elif cr >= 1.0:
                pass  # Acceptable, no specific call-out
            else:
                weaknesses.append(f"Current ratio of {cr:.2f} below 1.0 raises liquidity concerns.")

        # Leverage
        lev = self.calculate_leverage_ratios(data)
        dte = lev.get('debt_to_equity')
        if dte is not None:
            if dte > 2.0:
                risks.append(f"High debt-to-equity of {dte:.2f} increases financial risk and interest burden.")
            elif dte < 0.5:
                strengths.append(f"Conservative leverage (D/E: {dte:.2f}) provides financial flexibility.")

        ic = lev.get('interest_coverage')
        if ic is not None:
            if ic > 5:
                strengths.append(f"Strong interest coverage of {ic:.1f}x comfortably services debt obligations.")
            elif ic < 2:
                risks.append(f"Low interest coverage of {ic:.1f}x; at risk of debt servicing difficulties.")

        # Working Capital
        wc = self.working_capital_analysis(data)
        if wc.ccc is not None:
            if wc.ccc < 0:
                strengths.append(f"Negative cash conversion cycle ({wc.ccc:.0f} days) means the business generates cash quickly.")
            elif wc.ccc > 60:
                opportunities.append(f"CCC of {wc.ccc:.0f} days can be improved through better AR/AP/inventory management.")

        # Breakeven
        be = self.breakeven_analysis(data)
        if be.margin_of_safety is not None:
            if be.margin_of_safety > 0.3:
                strengths.append(f"Margin of safety of {be.margin_of_safety*100:.0f}% above breakeven revenue.")
            elif be.margin_of_safety > 0:
                opportunities.append(f"Operating {be.margin_of_safety*100:.0f}% above breakeven; work to widen this buffer.")
            else:
                risks.append("Operating below breakeven revenue level.")

        # Generate recommendation
        if score >= 70 and len(risks) == 0:
            recommendation = "Company is in strong financial health. Maintain current trajectory and look for growth opportunities."
        elif score >= 50:
            top_weakness = weaknesses[0] if weaknesses else "overall efficiency"
            recommendation = f"Address key weakness: {top_weakness} Focus on improving profitability and reducing leverage."
        else:
            recommendation = "Urgent attention needed. Prioritize cash preservation, debt restructuring, and operational efficiency."

        return NarrativeReport(
            headline=headline,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            risks=risks,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------ #
    #  Phase 9 – Regression Forecasting & Industry Benchmarking           #
    # ------------------------------------------------------------------ #

    def regression_forecast(
        self,
        values: List[float],
        periods_ahead: int = 3,
        method: str = "linear",
        metric_name: str = "metric",
        confidence_level: float = 0.90,
    ) -> RegressionForecast:
        """Fit a regression model and project forward with confidence bands.

        Args:
            values: Historical data points (at least 3).
            periods_ahead: Number of future periods to forecast.
            method: 'linear', 'exponential', or 'polynomial'.
            metric_name: Label for reporting.
            confidence_level: Confidence band width (0-1).

        Returns:
            RegressionForecast with fitted params, projected values, and bands.
        """
        result = RegressionForecast(
            metric_name=metric_name,
            historical_values=list(values),
            periods_ahead=periods_ahead,
            method=method,
        )
        if len(values) < 3:
            return result

        n = len(values)
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)

        if method == "exponential" and np.all(y > 0):
            log_y = np.log(y)
            coeffs = np.polyfit(x, log_y, 1)
            slope_log, intercept_log = coeffs
            fitted = np.exp(intercept_log + slope_log * x)
            residuals = y - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            future_x = np.arange(n, n + periods_ahead, dtype=float)
            forecast = np.exp(intercept_log + slope_log * future_x)
            result.slope = float(slope_log)
            result.intercept = float(intercept_log)
            result.r_squared = float(max(0.0, r2))
            result.forecast_values = forecast.tolist()
            # Confidence bands: use residual std on log scale
            se = np.sqrt(ss_res / max(n - 2, 1))
            z = 1.645 if confidence_level >= 0.90 else 1.28
            result.confidence_upper = np.exp(intercept_log + slope_log * future_x + z * se).tolist()
            result.confidence_lower = np.exp(np.maximum(intercept_log + slope_log * future_x - z * se, -20)).tolist()
        elif method == "polynomial":
            deg = min(2, n - 1)
            coeffs = np.polyfit(x, y, deg)
            poly = np.poly1d(coeffs)
            fitted = poly(x)
            residuals = y - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            future_x = np.arange(n, n + periods_ahead, dtype=float)
            forecast = poly(future_x)
            result.slope = float(coeffs[-2]) if len(coeffs) > 1 else 0.0
            result.intercept = float(coeffs[-1])
            result.r_squared = float(max(0.0, r2))
            result.forecast_values = forecast.tolist()
            se = np.sqrt(ss_res / max(n - deg - 1, 1))
            z = 1.645 if confidence_level >= 0.90 else 1.28
            result.confidence_upper = (forecast + z * se).tolist()
            result.confidence_lower = (forecast - z * se).tolist()
        else:
            # Default linear regression
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            fitted = intercept + slope * x
            residuals = y - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            future_x = np.arange(n, n + periods_ahead, dtype=float)
            forecast = intercept + slope * future_x
            result.slope = float(slope)
            result.intercept = float(intercept)
            result.r_squared = float(max(0.0, r2))
            result.forecast_values = forecast.tolist()
            se = np.sqrt(ss_res / max(n - 2, 1))
            z = 1.645 if confidence_level >= 0.90 else 1.28
            result.confidence_upper = (forecast + z * se).tolist()
            result.confidence_lower = (forecast - z * se).tolist()

        return result

    # Industry benchmark medians (representative defaults)
    INDUSTRY_BENCHMARKS: Dict[str, Dict[str, Dict[str, float]]] = {
        "general": {
            "current_ratio":      {"p25": 1.2, "median": 1.8, "p75": 2.5},
            "quick_ratio":        {"p25": 0.8, "median": 1.2, "p75": 1.8},
            "gross_margin":       {"p25": 0.25, "median": 0.40, "p75": 0.55},
            "net_margin":         {"p25": 0.03, "median": 0.08, "p75": 0.15},
            "roe":                {"p25": 0.06, "median": 0.12, "p75": 0.20},
            "roa":                {"p25": 0.03, "median": 0.06, "p75": 0.12},
            "debt_to_equity":     {"p25": 0.3, "median": 0.8, "p75": 1.5},
            "interest_coverage":  {"p25": 2.0, "median": 5.0, "p75": 10.0},
            "asset_turnover":     {"p25": 0.4, "median": 0.8, "p75": 1.2},
        },
        "technology": {
            "current_ratio":      {"p25": 1.5, "median": 2.5, "p75": 4.0},
            "gross_margin":       {"p25": 0.50, "median": 0.65, "p75": 0.80},
            "net_margin":         {"p25": 0.05, "median": 0.15, "p75": 0.25},
            "roe":                {"p25": 0.08, "median": 0.18, "p75": 0.30},
            "roa":                {"p25": 0.04, "median": 0.10, "p75": 0.18},
            "debt_to_equity":     {"p25": 0.1, "median": 0.3, "p75": 0.7},
        },
        "manufacturing": {
            "current_ratio":      {"p25": 1.0, "median": 1.5, "p75": 2.0},
            "gross_margin":       {"p25": 0.20, "median": 0.30, "p75": 0.40},
            "net_margin":         {"p25": 0.02, "median": 0.05, "p75": 0.10},
            "roe":                {"p25": 0.05, "median": 0.10, "p75": 0.15},
            "debt_to_equity":     {"p25": 0.5, "median": 1.0, "p75": 2.0},
            "asset_turnover":     {"p25": 0.6, "median": 1.0, "p75": 1.5},
        },
        "retail": {
            "current_ratio":      {"p25": 0.8, "median": 1.3, "p75": 1.8},
            "gross_margin":       {"p25": 0.25, "median": 0.35, "p75": 0.50},
            "net_margin":         {"p25": 0.01, "median": 0.04, "p75": 0.08},
            "roe":                {"p25": 0.08, "median": 0.15, "p75": 0.25},
            "debt_to_equity":     {"p25": 0.4, "median": 1.0, "p75": 2.0},
            "asset_turnover":     {"p25": 1.0, "median": 1.8, "p75": 2.5},
        },
        "healthcare": {
            "current_ratio":      {"p25": 1.2, "median": 1.8, "p75": 3.0},
            "gross_margin":       {"p25": 0.30, "median": 0.45, "p75": 0.60},
            "net_margin":         {"p25": 0.02, "median": 0.08, "p75": 0.15},
            "roe":                {"p25": 0.05, "median": 0.12, "p75": 0.20},
            "debt_to_equity":     {"p25": 0.3, "median": 0.7, "p75": 1.5},
        },
    }

    def industry_benchmark(
        self,
        data: FinancialData,
        industry: str = "general",
    ) -> IndustryBenchmarkResult:
        """Compare company metrics against industry percentile benchmarks.

        Args:
            data: Company financial data.
            industry: Industry name (general, technology, manufacturing, retail, healthcare).

        Returns:
            IndustryBenchmarkResult with per-metric comparison and overall percentile.
        """
        benchmarks = self.INDUSTRY_BENCHMARKS.get(
            industry.lower(), self.INDUSTRY_BENCHMARKS["general"]
        )

        # Gather company metrics
        liquidity = self.calculate_liquidity_ratios(data)
        profitability = self.calculate_profitability_ratios(data)
        leverage = self.calculate_leverage_ratios(data)
        efficiency = self.calculate_efficiency_ratios(data)

        company_metrics: Dict[str, Optional[float]] = {}
        company_metrics["current_ratio"] = liquidity.get("current_ratio")
        company_metrics["quick_ratio"] = liquidity.get("quick_ratio")
        company_metrics["gross_margin"] = profitability.get("gross_margin")
        company_metrics["net_margin"] = profitability.get("net_margin")
        company_metrics["roe"] = profitability.get("roe")
        company_metrics["roa"] = profitability.get("roa")
        company_metrics["debt_to_equity"] = leverage.get("debt_to_equity")
        company_metrics["interest_coverage"] = leverage.get("interest_coverage")
        company_metrics["asset_turnover"] = efficiency.get("asset_turnover")

        comparisons: List[BenchmarkComparison] = []
        percentile_ranks: List[float] = []

        for metric_name, bench in benchmarks.items():
            val = company_metrics.get(metric_name)
            if val is None:
                continue

            p25 = bench["p25"]
            median = bench["median"]
            p75 = bench["p75"]

            # Estimate percentile rank (linear interpolation)
            # For debt_to_equity, lower is better (inverted)
            inverted = metric_name in ("debt_to_equity",)

            if inverted:
                if val <= p75:
                    pct = 75 + 25 * (p75 - val) / max(p75 - p25, 1e-9) * (50 / 25)
                    pct = min(pct, 99)
                elif val <= median:
                    pct = 50 + 25 * (median - val) / max(median - p25, 1e-9)
                elif val <= p25:
                    pct = 25 + 25 * (p25 - val) / max(p25, 1e-9)
                else:
                    # Higher than p25 (bad for inverted)
                    pct = max(5, 25 * (1 - (val - p25) / max(p25, 1e-9)))
            else:
                if val >= p75:
                    pct = min(75 + 25 * (val - p75) / max(p75 - median, 1e-9), 99)
                elif val >= median:
                    pct = 50 + 25 * (val - median) / max(p75 - median, 1e-9)
                elif val >= p25:
                    pct = 25 + 25 * (val - p25) / max(median - p25, 1e-9)
                else:
                    pct = max(1, 25 * val / max(p25, 1e-9))

            pct = float(np.clip(pct, 1, 99))

            if pct >= 65:
                rating = "above average"
            elif pct >= 35:
                rating = "average"
            else:
                rating = "below average"

            comparisons.append(BenchmarkComparison(
                metric_name=metric_name,
                company_value=val,
                industry_median=median,
                industry_p25=p25,
                industry_p75=p75,
                percentile_rank=pct,
                rating=rating,
            ))
            percentile_ranks.append(pct)

        overall = float(np.mean(percentile_ranks)) if percentile_ranks else None

        above = sum(1 for c in comparisons if c.rating == "above average")
        below = sum(1 for c in comparisons if c.rating == "below average")
        total = len(comparisons)

        if overall is not None and overall >= 60:
            summary = (f"Company ranks in the {overall:.0f}th percentile overall vs {industry} peers. "
                       f"{above}/{total} metrics above average.")
        elif overall is not None and overall >= 40:
            summary = (f"Company performs near industry median ({overall:.0f}th percentile). "
                       f"{below}/{total} metrics need attention.")
        elif overall is not None:
            summary = (f"Company underperforms {industry} peers ({overall:.0f}th percentile). "
                       f"{below}/{total} metrics below average.")
        else:
            summary = "Insufficient data for industry comparison."

        return IndustryBenchmarkResult(
            industry_name=industry,
            comparisons=comparisons,
            overall_percentile=overall,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    #  Phase 10 – Custom KPI Builder                                      #
    # ------------------------------------------------------------------ #

    # Fields that can be referenced in custom KPI formulas
    _KPI_ALLOWED_FIELDS = frozenset({
        'revenue', 'cogs', 'gross_profit', 'operating_income', 'operating_expenses',
        'ebit', 'ebitda', 'interest_expense', 'net_income', 'ebt',
        'retained_earnings', 'depreciation',
        'total_assets', 'current_assets', 'cash', 'inventory',
        'accounts_receivable', 'total_liabilities', 'current_liabilities',
        'accounts_payable', 'total_debt', 'total_equity',
        'operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow', 'capex',
    })

    _AST_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def _safe_eval_formula(cls, formula: str, namespace: Dict[str, float]) -> float:
        """Evaluate an arithmetic formula using AST walking (no eval).

        Only allows: numeric literals, names in *namespace*, and
        binary/unary +, -, *, /.  Raises ValueError on anything else.
        """
        tree = ast.parse(formula, mode="eval")

        def _walk(node):
            if isinstance(node, ast.Expression):
                return _walk(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id in namespace:
                    return namespace[node.id]
                raise NameError(f"Unknown variable: '{node.id}'")
            if isinstance(node, ast.BinOp):
                op = cls._AST_OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                left_val = _walk(node.left)
                right_val = _walk(node.right)
                if isinstance(node.op, ast.Div) and right_val == 0:
                    raise ValueError("Division by zero in formula")
                return op(left_val, right_val)
            if isinstance(node, ast.UnaryOp):
                op = cls._AST_OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
                return op(_walk(node.operand))
            raise ValueError(f"Unsupported expression: {type(node).__name__}")

        return float(_walk(tree))

    def evaluate_custom_kpis(
        self,
        data: FinancialData,
        kpi_definitions: List[CustomKPIDefinition],
    ) -> CustomKPIReport:
        """Evaluate user-defined KPI formulas against financial data.

        Formulas can reference FinancialData field names and use basic arithmetic:
            ``+``, ``-``, ``*``, ``/``, ``()``, and numeric literals.

        Examples:
            - ``"revenue / total_assets"``  (asset turnover)
            - ``"(revenue - cogs) / revenue"``  (gross margin)
            - ``"ebitda - capex"``  (free cash proxy)

        Args:
            data: Company financial data providing variable values.
            kpi_definitions: List of KPI definitions to evaluate.

        Returns:
            CustomKPIReport with per-KPI results and summary.
        """
        results: List[CustomKPIResult] = []

        # Build namespace from FinancialData fields
        namespace: Dict[str, float] = {}
        for field_name in self._KPI_ALLOWED_FIELDS:
            val = getattr(data, field_name, None)
            if val is not None:
                namespace[field_name] = float(val)

        for kpi in kpi_definitions:
            r = CustomKPIResult(name=kpi.name, formula=kpi.formula)

            if not kpi.formula.strip():
                r.error = "Empty formula"
                results.append(r)
                continue

            # Validate formula tokens (security: only allow field names, numbers, operators)
            sanitized = kpi.formula
            # Remove valid tokens to check for leftovers
            temp = sanitized
            for fname in sorted(self._KPI_ALLOWED_FIELDS, key=len, reverse=True):
                temp = temp.replace(fname, '')
            # Remove numbers, operators, whitespace, parentheses, dots
            temp = re.sub(r'[\d+\-*/().\s]', '', temp)
            if temp:
                r.error = f"Invalid tokens in formula: {temp}"
                results.append(r)
                continue

            try:
                value = self._safe_eval_formula(sanitized, namespace)
                if np.isnan(value) or np.isinf(value):
                    r.error = "Formula produced invalid result"
                else:
                    r.value = value
                    if kpi.target_min is not None and kpi.target_max is not None:
                        r.meets_target = kpi.target_min <= r.value <= kpi.target_max
                    elif kpi.target_min is not None:
                        r.meets_target = r.value >= kpi.target_min
                    elif kpi.target_max is not None:
                        r.meets_target = r.value <= kpi.target_max
            except ZeroDivisionError:
                r.error = "Division by zero"
            except NameError as e:
                r.error = f"Unknown variable: {e}"
            except Exception as e:
                r.error = f"Evaluation error: {e}"

            results.append(r)

        met = sum(1 for r in results if r.meets_target is True)
        total_with_target = sum(1 for r in results if r.meets_target is not None)
        errors = sum(1 for r in results if r.error)

        if errors:
            summary = f"{len(results)} KPIs evaluated, {errors} errors."
        elif total_with_target:
            summary = f"{met}/{total_with_target} KPIs meeting targets."
        else:
            summary = f"{len(results)} KPIs evaluated."

        return CustomKPIReport(results=results, summary=summary)

    # ------------------------------------------------------------------
    # Phase 11: Peer Comparison & Ratio Decomposition
    # ------------------------------------------------------------------

    def peer_comparison(
        self,
        peers: List[PeerCompanyData],
        metrics: Optional[List[str]] = None,
    ) -> PeerComparisonReport:
        """Compare financial metrics across multiple peers.

        Parameters
        ----------
        peers : list of PeerCompanyData
            Each peer has a name and FinancialData.
        metrics : list of str, optional
            Metric names to compare. Defaults to a standard set.

        Returns
        -------
        PeerComparisonReport
        """
        if not peers:
            return PeerComparisonReport(summary="No peers provided.")

        if metrics is None:
            metrics = [
                "current_ratio", "gross_margin", "net_margin", "roe",
                "roa", "debt_to_equity", "asset_turnover",
                "operating_margin", "interest_coverage",
            ]

        peer_names = [p.name or f"Peer {i+1}" for i, p in enumerate(peers)]
        comparisons: List[PeerMetricComparison] = []

        for metric_name in metrics:
            values: Dict[str, Optional[float]] = {}
            for peer in peers:
                name = peer.name or f"Peer {peers.index(peer)+1}"
                val = self._compute_peer_metric(peer.data, metric_name)
                values[name] = val

            valid = {k: v for k, v in values.items() if v is not None}
            if valid:
                avg = sum(valid.values()) / len(valid)
                sorted_v = sorted(valid.items(), key=lambda x: x[1], reverse=True)
                # For debt_to_equity, lower is better
                higher_is_better = metric_name != "debt_to_equity"
                best = sorted_v[0][0] if higher_is_better else sorted_v[-1][0]
                worst = sorted_v[-1][0] if higher_is_better else sorted_v[0][0]
                vals_list = sorted(valid.values())
                median_val = vals_list[len(vals_list) // 2]
            else:
                avg = None
                best = ""
                worst = ""
                median_val = None

            comparisons.append(PeerMetricComparison(
                metric_name=metric_name,
                values=values,
                best_performer=best,
                worst_performer=worst,
                average=avg,
                median=median_val,
            ))

        # Compute overall ranking: count how many metrics each peer is best at
        best_counts: Dict[str, int] = {n: 0 for n in peer_names}
        for comp in comparisons:
            if comp.best_performer and comp.best_performer in best_counts:
                best_counts[comp.best_performer] += 1

        # Rank by best counts (higher = better rank = lower number)
        sorted_peers = sorted(best_counts.items(), key=lambda x: x[1], reverse=True)
        rankings = {name: rank + 1 for rank, (name, _) in enumerate(sorted_peers)}

        top = sorted_peers[0][0] if sorted_peers else ""
        summary = (
            f"Compared {len(peer_names)} peers across {len(metrics)} metrics. "
            f"{top} leads in {best_counts.get(top, 0)} metrics."
        )

        return PeerComparisonReport(
            peer_names=peer_names,
            comparisons=comparisons,
            rankings=rankings,
            summary=summary,
        )

    def _compute_peer_metric(
        self, data: Optional[FinancialData], metric_name: str
    ) -> Optional[float]:
        """Compute a single named metric from FinancialData."""
        if data is None:
            return None
        m = metric_name.lower()
        if m == "current_ratio":
            return safe_divide(data.current_assets, data.current_liabilities)
        elif m == "gross_margin":
            return safe_divide(data.gross_profit, data.revenue)
        elif m == "net_margin":
            return safe_divide(data.net_income, data.revenue)
        elif m == "operating_margin":
            return safe_divide(data.operating_income, data.revenue)
        elif m == "roe":
            return safe_divide(data.net_income, data.total_equity)
        elif m == "roa":
            return safe_divide(data.net_income, data.total_assets)
        elif m == "debt_to_equity":
            return safe_divide(data.total_debt, data.total_equity)
        elif m == "asset_turnover":
            return safe_divide(data.revenue, data.total_assets)
        elif m == "interest_coverage":
            return safe_divide(data.ebit, data.interest_expense)
        elif m == "ebitda_margin":
            return safe_divide(data.ebitda, data.revenue)
        return None

    def ratio_decomposition(self, data: FinancialData) -> RatioDecompositionTree:
        """Build a DuPont-based ratio decomposition tree from ROE.

        Decomposes ROE -> Net Margin × Asset Turnover × Equity Multiplier,
        then further decomposes each factor into its constituent components.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        RatioDecompositionTree
        """
        # Level 0: ROE
        roe_val = safe_divide(data.net_income, data.total_equity)

        # Level 1: DuPont 3-factor
        net_margin = safe_divide(data.net_income, data.revenue)
        asset_turnover = safe_divide(data.revenue, data.total_assets)
        equity_multiplier = safe_divide(data.total_assets, data.total_equity)

        # Level 2: Net Margin decomposition
        gross_margin = safe_divide(data.gross_profit, data.revenue)
        opex_ratio = safe_divide(data.operating_expenses, data.revenue)
        interest_burden = safe_divide(data.interest_expense, data.revenue)
        tax_burden = None
        if data.net_income is not None and data.ebt is not None and data.ebt:
            tax_burden = safe_divide(data.net_income, data.ebt)

        net_margin_children = [
            RatioDecompositionNode(
                name="Gross Margin",
                value=gross_margin,
                formula="gross_profit / revenue",
            ),
            RatioDecompositionNode(
                name="OpEx Ratio",
                value=opex_ratio,
                formula="operating_expenses / revenue",
            ),
            RatioDecompositionNode(
                name="Interest Burden",
                value=interest_burden,
                formula="interest_expense / revenue",
            ),
        ]
        if tax_burden is not None:
            net_margin_children.append(RatioDecompositionNode(
                name="Tax Retention",
                value=tax_burden,
                formula="net_income / ebt",
            ))

        # Level 2: Asset Turnover decomposition
        receivables_turnover = safe_divide(data.revenue, data.accounts_receivable)
        inventory_turnover = safe_divide(data.cogs, data.inventory)
        fixed_asset_ratio = None
        if data.total_assets is not None and data.current_assets is not None:
            fixed_assets = data.total_assets - data.current_assets
            fixed_asset_ratio = safe_divide(data.revenue, fixed_assets if fixed_assets > 0 else None)

        at_children = [
            RatioDecompositionNode(
                name="Receivables Turnover",
                value=receivables_turnover,
                formula="revenue / accounts_receivable",
            ),
            RatioDecompositionNode(
                name="Inventory Turnover",
                value=inventory_turnover,
                formula="cogs / inventory",
            ),
        ]
        if fixed_asset_ratio is not None:
            at_children.append(RatioDecompositionNode(
                name="Fixed Asset Turnover",
                value=fixed_asset_ratio,
                formula="revenue / fixed_assets",
            ))

        # Level 2: Equity Multiplier decomposition
        debt_ratio = safe_divide(data.total_liabilities, data.total_assets)
        debt_to_equity = safe_divide(data.total_debt, data.total_equity)

        em_children = [
            RatioDecompositionNode(
                name="Debt Ratio",
                value=debt_ratio,
                formula="total_liabilities / total_assets",
            ),
            RatioDecompositionNode(
                name="Debt-to-Equity",
                value=debt_to_equity,
                formula="total_debt / total_equity",
            ),
        ]

        # Build tree
        root = RatioDecompositionNode(
            name="ROE",
            value=roe_val,
            formula="net_income / total_equity",
            children=[
                RatioDecompositionNode(
                    name="Net Profit Margin",
                    value=net_margin,
                    formula="net_income / revenue",
                    children=net_margin_children,
                ),
                RatioDecompositionNode(
                    name="Asset Turnover",
                    value=asset_turnover,
                    formula="revenue / total_assets",
                    children=at_children,
                ),
                RatioDecompositionNode(
                    name="Equity Multiplier",
                    value=equity_multiplier,
                    formula="total_assets / total_equity",
                    children=em_children,
                ),
            ],
        )

        # Summary
        parts = []
        if roe_val is not None:
            parts.append(f"ROE: {roe_val:.1%}")
        if net_margin is not None:
            parts.append(f"Net Margin: {net_margin:.1%}")
        if asset_turnover is not None:
            parts.append(f"Asset Turnover: {asset_turnover:.2f}x")
        if equity_multiplier is not None:
            parts.append(f"Equity Multiplier: {equity_multiplier:.2f}x")
        summary = " | ".join(parts) if parts else "Insufficient data for decomposition."

        return RatioDecompositionTree(root=root, summary=summary)

    # ------------------------------------------------------------------
    # Phase 12: Financial Rating Scorecard & Variance Waterfall
    # ------------------------------------------------------------------

    _GRADE_THRESHOLDS = [
        (9.0, "AAA"), (8.0, "AA"), (7.0, "A"),
        (6.0, "BBB"), (5.0, "BB"), (4.0, "B"),
        (3.0, "CCC"), (2.0, "CC"), (0.0, "C"),
    ]

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert a numeric score (0-10) to a letter grade."""
        for threshold, grade in CharlieAnalyzer._GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "C"

    # ------------------------------------------------------------------
    # Generic Scored Analysis Engine
    # ------------------------------------------------------------------

    def _scored_analysis(
        self,
        data: "FinancialData",
        result_class: type,
        ratio_defs: List[Tuple[str, str, str]],
        score_field: str,
        grade_field: str,
        primary: str,
        higher_is_better: bool,
        thresholds: List[Tuple[float, float]],
        adjustments: Optional[List[Tuple]] = None,
        derived: Optional[List[Tuple]] = None,
        label: str = "",
    ):
        """Generic scored ratio analysis engine.

        Replaces the repetitive pattern of: extract ratios -> score ->
        grade -> summarize found in 68+ phase analysis methods.

        Parameters
        ----------
        data : FinancialData
        result_class : dataclass type to instantiate
        ratio_defs : list of (field_name, numerator_attr, denominator_attr)
        score_field : attribute name for the score on result
        grade_field : attribute name for the grade on result
        primary : field_name of the primary ratio used for scoring
        higher_is_better : True = higher ratio -> higher score
        thresholds : list of (threshold, base_score) DESCENDING by threshold
        adjustments : list of (condition_fn(ratios, data) -> bool, delta)
        derived : list of (field_name, compute_fn(ratios, data) -> value)
        label : analysis label for summary string
        """
        result = result_class()
        ratios: Dict[str, Optional[float]] = {}

        # Step 1: Compute ratios via safe_divide
        for field_name, num_attr, denom_attr in ratio_defs:
            val = safe_divide(
                getattr(data, num_attr, None),
                getattr(data, denom_attr, None),
            )
            setattr(result, field_name, val)
            ratios[field_name] = val

        # Step 2: Compute derived fields
        for field_name, compute_fn in (derived or []):
            val = compute_fn(ratios, data)
            setattr(result, field_name, val)
            ratios[field_name] = val

        # Step 3: Check primary ratio
        primary_val = ratios.get(primary)
        if primary_val is None:
            setattr(result, score_field, 0.0)
            result.summary = f"{label}: Insufficient data for analysis."
            return result

        # Step 4: Score from thresholds (callers MUST pass pre-sorted tuples)
        if higher_is_better:
            # Descending: find first threshold where value >= threshold
            base = max(thresholds[-1][1] - 1.0, 0.0)
            for thresh, sc in thresholds:
                if primary_val >= thresh:
                    base = sc
                    break
        else:
            # Ascending: find first threshold where value <= threshold
            sorted_t = thresholds
            base = max(sorted_t[-1][1] - 1.0, 0.0)
            for thresh, sc in sorted_t:
                if primary_val <= thresh:
                    base = sc
                    break

        # Step 5: Apply adjustments
        adj = 0.0
        for cond_fn, delta in (adjustments or []):
            if cond_fn(ratios, data):
                adj += delta

        score = round(min(10.0, max(0.0, base + adj)), 1)
        setattr(result, score_field, score)

        # Step 6: Grade
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        setattr(result, grade_field, grade)

        # Step 7: Summary
        primary_str = f"{primary_val:.4f}" if primary_val is not None else "N/A"
        result.summary = (
            f"{label}: {primary}={primary_str}. "
            f"Score={score}/10 ({grade})."
        )

        return result

    def financial_rating(self, data: FinancialData) -> FinancialRating:
        """Generate a comprehensive letter-grade financial rating.

        Scores 5 categories (0-10 each) and computes a weighted average:
        - Liquidity (20%)
        - Profitability (25%)
        - Leverage (20%)
        - Efficiency (20%)
        - Cash Flow (15%)

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        FinancialRating
        """
        categories = []

        # 1. Liquidity (20%)
        cr = safe_divide(data.current_assets, data.current_liabilities)
        qr = safe_divide(
            (data.current_assets or 0) - (data.inventory or 0),
            data.current_liabilities,
        )
        liq_score = 0.0
        liq_details = []
        if cr is not None:
            if cr >= 2.0:
                liq_score += 5.0
            elif cr >= 1.5:
                liq_score += 4.0
            elif cr >= 1.0:
                liq_score += 2.5
            else:
                liq_score += 1.0
            liq_details.append(f"Current Ratio: {cr:.2f}")
        if qr is not None:
            if qr >= 1.5:
                liq_score += 5.0
            elif qr >= 1.0:
                liq_score += 4.0
            elif qr >= 0.5:
                liq_score += 2.5
            else:
                liq_score += 1.0
            liq_details.append(f"Quick Ratio: {qr:.2f}")
        categories.append(RatingCategory(
            name="Liquidity", score=liq_score, grade=self._score_to_grade(liq_score),
            details="; ".join(liq_details) if liq_details else "Insufficient data",
        ))

        # 2. Profitability (25%)
        gm = safe_divide(data.gross_profit, data.revenue)
        nm = safe_divide(data.net_income, data.revenue)
        roe = safe_divide(data.net_income, data.total_equity)
        prof_score = 0.0
        prof_details = []
        if gm is not None:
            s = min(3.3, max(0, gm * 10))
            prof_score += s
            prof_details.append(f"Gross Margin: {gm:.1%}")
        if nm is not None:
            s = min(3.3, max(0, nm * 20))
            prof_score += s
            prof_details.append(f"Net Margin: {nm:.1%}")
        if roe is not None:
            s = min(3.4, max(0, roe * 20))
            prof_score += s
            prof_details.append(f"ROE: {roe:.1%}")
        categories.append(RatingCategory(
            name="Profitability", score=min(10, prof_score),
            grade=self._score_to_grade(min(10, prof_score)),
            details="; ".join(prof_details) if prof_details else "Insufficient data",
        ))

        # 3. Leverage (20%)
        de = safe_divide(data.total_debt, data.total_equity)
        ic = safe_divide(data.ebit, data.interest_expense)
        lev_score = 0.0
        lev_details = []
        if de is not None:
            # Lower is better
            if de <= 0.5:
                lev_score += 5.0
            elif de <= 1.0:
                lev_score += 4.0
            elif de <= 2.0:
                lev_score += 2.5
            else:
                lev_score += 1.0
            lev_details.append(f"D/E: {de:.2f}")
        if ic is not None:
            if ic >= 8.0:
                lev_score += 5.0
            elif ic >= 4.0:
                lev_score += 4.0
            elif ic >= 2.0:
                lev_score += 2.5
            else:
                lev_score += 1.0
            lev_details.append(f"Interest Coverage: {ic:.1f}x")
        categories.append(RatingCategory(
            name="Leverage", score=lev_score,
            grade=self._score_to_grade(lev_score),
            details="; ".join(lev_details) if lev_details else "Insufficient data",
        ))

        # 4. Efficiency (20%)
        at = safe_divide(data.revenue, data.total_assets)
        inv_t = safe_divide(data.cogs, data.inventory)
        eff_score = 0.0
        eff_details = []
        if at is not None:
            s = min(5.0, max(0, at * 5))
            eff_score += s
            eff_details.append(f"Asset Turnover: {at:.2f}x")
        if inv_t is not None:
            s = min(5.0, max(0, inv_t / 2))
            eff_score += s
            eff_details.append(f"Inventory Turnover: {inv_t:.1f}x")
        categories.append(RatingCategory(
            name="Efficiency", score=min(10, eff_score),
            grade=self._score_to_grade(min(10, eff_score)),
            details="; ".join(eff_details) if eff_details else "Insufficient data",
        ))

        # 5. Cash Flow (15%)
        cf_score = 0.0
        cf_details = []
        ocf = data.operating_cash_flow
        fcf = None
        if ocf is not None and data.capex is not None:
            fcf = ocf - abs(data.capex)
        if ocf is not None and data.revenue:
            cf_ratio = safe_divide(ocf, data.revenue, default=0.0)
            s = min(5.0, max(0, cf_ratio * 20))
            cf_score += s
            cf_details.append(f"OCF/Revenue: {cf_ratio:.1%}")
        if fcf is not None and fcf > 0:
            cf_score += 5.0
            cf_details.append(f"FCF: {fcf:,.0f} (positive)")
        elif fcf is not None:
            cf_score += 1.0
            cf_details.append(f"FCF: {fcf:,.0f} (negative)")
        categories.append(RatingCategory(
            name="Cash Flow", score=min(10, cf_score),
            grade=self._score_to_grade(min(10, cf_score)),
            details="; ".join(cf_details) if cf_details else "Insufficient data",
        ))

        # Weighted average
        weights = [0.20, 0.25, 0.20, 0.20, 0.15]
        overall = sum(c.score * w for c, w in zip(categories, weights))
        overall_grade = self._score_to_grade(overall)

        # Summary
        strengths = [c.name for c in categories if c.score >= 7.0]
        weaknesses = [c.name for c in categories if c.score < 4.0]
        parts = [f"Overall: {overall_grade} ({overall:.1f}/10)"]
        if strengths:
            parts.append(f"Strengths: {', '.join(strengths)}")
        if weaknesses:
            parts.append(f"Weaknesses: {', '.join(weaknesses)}")

        return FinancialRating(
            overall_score=overall,
            overall_grade=overall_grade,
            categories=categories,
            summary=" | ".join(parts),
        )

    def variance_waterfall(
        self,
        current: FinancialData,
        previous: FinancialData,
    ) -> VarianceWaterfall:
        """Build a waterfall showing what drove the change in net income.

        Decomposes net income variance into revenue, COGS, OpEx,
        interest, and other components.

        Parameters
        ----------
        current : FinancialData
        previous : FinancialData

        Returns
        -------
        VarianceWaterfall
        """
        start = previous.net_income or 0
        end = current.net_income or 0
        items: List[WaterfallItem] = []
        cumulative = start

        items.append(WaterfallItem(
            label="Prior Net Income",
            value=start,
            cumulative=start,
            item_type="start",
        ))

        # Revenue change
        rev_delta = (current.revenue or 0) - (previous.revenue or 0)
        cumulative += rev_delta
        items.append(WaterfallItem(
            label="Revenue Change",
            value=rev_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # COGS change (negative = favorable if COGS decreased)
        cogs_delta = -((current.cogs or 0) - (previous.cogs or 0))
        cumulative += cogs_delta
        items.append(WaterfallItem(
            label="COGS Impact",
            value=cogs_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # OpEx change (negative = favorable if OpEx decreased)
        opex_delta = -((current.operating_expenses or 0) - (previous.operating_expenses or 0))
        cumulative += opex_delta
        items.append(WaterfallItem(
            label="OpEx Impact",
            value=opex_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # Interest change (negative = favorable if interest decreased)
        int_delta = -((current.interest_expense or 0) - (previous.interest_expense or 0))
        cumulative += int_delta
        items.append(WaterfallItem(
            label="Interest Impact",
            value=int_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # Other / residual
        residual = end - cumulative
        if abs(residual) > 0.01:
            cumulative += residual
            items.append(WaterfallItem(
                label="Other / Tax",
                value=residual,
                cumulative=cumulative,
                item_type="delta",
            ))

        items.append(WaterfallItem(
            label="Current Net Income",
            value=end,
            cumulative=end,
            item_type="total",
        ))

        total_var = end - start
        pct = (total_var / abs(start) * 100) if start != 0 else 0
        summary = f"Net income changed by {total_var:,.0f} ({pct:+.1f}%) from {start:,.0f} to {end:,.0f}."

        return VarianceWaterfall(
            start_value=start,
            end_value=end,
            items=items,
            total_variance=total_var,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 13 – Earnings Quality & Capital Efficiency
    # ------------------------------------------------------------------

    def earnings_quality(self, data: FinancialData) -> EarningsQualityResult:
        """Assess the quality of reported earnings.

        Evaluates how well net income is supported by actual cash flows
        using the accrual ratio and cash-to-income ratio.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        EarningsQualityResult
        """
        ni = data.net_income
        ocf = data.operating_cash_flow
        ta = data.total_assets

        # Guard: need at least net_income to score
        if ni is None and ocf is None:
            return EarningsQualityResult(
                quality_score=0.0,
                quality_grade="N/A",
                summary="Insufficient data to assess earnings quality.",
            )

        indicators: List[str] = []
        score = 5.0  # start at mid-point

        # --- Accrual ratio ---
        accrual_ratio = None
        if ni is not None and ocf is not None:
            accrual_ratio = safe_divide(ni - ocf, ta)
            if accrual_ratio is not None:
                if accrual_ratio < -0.05:
                    score += 2.0
                    indicators.append("Very low accruals – earnings strongly backed by cash")
                elif accrual_ratio < 0.05:
                    score += 1.0
                    indicators.append("Low accruals – good earnings quality")
                elif accrual_ratio < 0.15:
                    indicators.append("Moderate accruals – monitor working capital changes")
                else:
                    score -= 2.0
                    indicators.append("High accruals – earnings may not be sustainable")

        # --- Cash-to-income ratio ---
        cash_to_income = None
        if ni is not None and ocf is not None:
            cash_to_income = safe_divide(ocf, ni)
            if cash_to_income is not None and ni > 0:
                if cash_to_income >= 1.2:
                    score += 1.5
                    indicators.append("Cash flow exceeds net income – strong cash generation")
                elif cash_to_income >= 0.8:
                    score += 0.5
                    indicators.append("Cash flow approximates net income – adequate quality")
                elif cash_to_income >= 0.5:
                    indicators.append("Cash flow trails net income – some non-cash items")
                else:
                    score -= 1.5
                    indicators.append("Cash flow well below net income – investigate accruals")
            elif ni is not None and ni < 0:
                # Negative NI: if OCF is positive, credit the company
                if ocf > 0:
                    score += 1.0
                    indicators.append("Positive cash flow despite net loss – operationally sound")
                else:
                    score -= 1.0
                    indicators.append("Negative cash flow and net loss – poor quality")

        # --- Operating cash flow positive ---
        if ocf is not None:
            if ocf > 0:
                score += 0.5
                indicators.append("Operating cash flow is positive")
            else:
                score -= 1.0
                indicators.append("Operating cash flow is negative")

        # Clamp score
        score = max(0.0, min(10.0, score))

        # Grade
        if score >= 8.0:
            grade = "High"
        elif score >= 5.0:
            grade = "Moderate"
        else:
            grade = "Low"

        parts = [f"Earnings Quality: {grade} ({score:.1f}/10)"]
        if accrual_ratio is not None:
            parts.append(f"Accrual ratio: {accrual_ratio:.3f}")
        if cash_to_income is not None:
            parts.append(f"Cash-to-income: {cash_to_income:.2f}")

        return EarningsQualityResult(
            accrual_ratio=accrual_ratio,
            cash_to_income_ratio=cash_to_income,
            quality_score=score,
            quality_grade=grade,
            indicators=indicators,
            summary=" | ".join(parts),
        )

    def capital_efficiency(
        self,
        data: FinancialData,
        cost_of_capital: float = 0.10,
    ) -> CapitalEfficiencyResult:
        """Evaluate capital efficiency and economic value creation.

        Computes ROIC, EVA, capital turnover, and reinvestment rate.

        Parameters
        ----------
        data : FinancialData
        cost_of_capital : float
            Weighted average cost of capital (default 10%).

        Returns
        -------
        CapitalEfficiencyResult
        """
        equity = data.total_equity or 0
        debt = data.total_debt or 0
        revenue = data.revenue

        # Invested capital = equity + debt
        invested_capital = equity + debt

        if invested_capital < 1e-12:
            return CapitalEfficiencyResult(
                wacc_estimate=cost_of_capital,
                summary="Insufficient data to compute capital efficiency.",
            )

        # Estimate NOPAT: EBIT * (1 - tax_rate)
        # Estimate tax rate from net_income / ebit if possible
        ebit = data.ebit if data.ebit is not None else data.operating_income
        ni = data.net_income
        tax_rate = 0.25  # default
        if ebit and ni is not None and ebit > 1e-12:
            implied_tax = 1.0 - safe_divide(ni, ebit, default=0.75) if ebit > 0 else 0.25
            if 0.0 <= implied_tax <= 0.60:
                tax_rate = implied_tax

        nopat = None
        if ebit is not None:
            nopat = ebit * (1.0 - tax_rate)

        # ROIC
        roic = safe_divide(nopat, invested_capital)

        # EVA
        eva = None
        if nopat is not None:
            eva = nopat - (cost_of_capital * invested_capital)

        # Capital turnover
        cap_turnover = safe_divide(revenue, invested_capital)

        # Reinvestment rate
        reinvest = None
        capex = data.capex
        if capex is not None and nopat is not None and abs(nopat) > 1e-12:
            reinvest = capex / nopat

        parts = []
        if roic is not None:
            parts.append(f"ROIC: {roic:.1%}")
        if eva is not None:
            parts.append(f"EVA: {eva:,.0f}")
        if cap_turnover is not None:
            parts.append(f"Capital Turnover: {cap_turnover:.2f}x")
        if reinvest is not None:
            parts.append(f"Reinvestment Rate: {reinvest:.1%}")
        summary = " | ".join(parts) if parts else "Insufficient data."

        return CapitalEfficiencyResult(
            roic=roic,
            invested_capital=invested_capital,
            nopat=nopat,
            eva=eva,
            wacc_estimate=cost_of_capital,
            capital_turnover=cap_turnover,
            reinvestment_rate=reinvest,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 14 – Liquidity Stress Test & Debt Service Analysis
    # ------------------------------------------------------------------

    def liquidity_stress_test(
        self,
        data: FinancialData,
        revenue_shocks: Optional[List[float]] = None,
    ) -> LiquidityStressResult:
        """Run a liquidity stress test under various revenue-shock scenarios.

        Parameters
        ----------
        data : FinancialData
        revenue_shocks : list of float, optional
            Revenue decline percentages to simulate (e.g. [0.10, 0.25, 0.50]).
            Defaults to [0.10, 0.25, 0.50].

        Returns
        -------
        LiquidityStressResult
        """
        if revenue_shocks is None:
            revenue_shocks = [0.10, 0.25, 0.50]

        ca = data.current_assets or 0
        cl = data.current_liabilities or 0
        inv = data.inventory or 0
        ocf = data.operating_cash_flow
        revenue = data.revenue or 0
        opex = data.operating_expenses or 0
        cogs = data.cogs or 0

        # Estimate monthly burn from operating data
        monthly_revenue = revenue / 12 if revenue > 0 else 0
        monthly_costs = (opex + cogs) / 12 if (opex + cogs) > 0 else 0

        # If OCF available, use it; otherwise estimate from revenue - costs
        if ocf is not None:
            monthly_net_cf = ocf / 12
        else:
            monthly_net_cf = monthly_revenue - monthly_costs

        # Cash proxy: current_assets - inventory
        cash_proxy = max(ca - inv, 0)

        # Months of cash
        if monthly_costs > 0 and monthly_net_cf < 0:
            months_cash = cash_proxy / abs(monthly_net_cf) if abs(monthly_net_cf) > 1e-12 else None
        elif monthly_costs > 0:
            # Positive OCF => estimate how long cash lasts if revenue stops
            months_cash = cash_proxy / monthly_costs if monthly_costs > 1e-12 else None
        else:
            months_cash = None

        # Stressed quick ratio (remove inventory)
        quick_assets = ca - inv
        stressed_qr = safe_divide(quick_assets, cl) if cl > 0 else None

        # Run stress scenarios
        scenarios = []
        for shock in sorted(revenue_shocks):
            stressed_rev = monthly_revenue * (1.0 - shock)
            stressed_cf = stressed_rev - monthly_costs
            if stressed_cf < 0 and cash_proxy > 0:
                survival_months = cash_proxy / abs(stressed_cf)
            elif stressed_cf >= 0:
                survival_months = float('inf')
            else:
                survival_months = 0.0

            scenarios.append({
                "shock_pct": shock,
                "label": f"{shock:.0%} revenue decline",
                "monthly_cash_flow": round(stressed_cf, 2),
                "survival_months": round(survival_months, 1) if survival_months != float('inf') else None,
                "survives_12m": survival_months >= 12,
            })

        # Risk level
        worst_survival = min(
            (s["survival_months"] for s in scenarios if s["survival_months"] is not None),
            default=None,
        )
        if worst_survival is None:
            risk = "Low"
        elif worst_survival >= 12:
            risk = "Low"
        elif worst_survival >= 6:
            risk = "Moderate"
        elif worst_survival >= 3:
            risk = "High"
        else:
            risk = "Critical"

        parts = [f"Liquidity Risk: {risk}"]
        if months_cash is not None:
            parts.append(f"Months of cash: {months_cash:.1f}")
        if stressed_qr is not None:
            parts.append(f"Quick ratio: {stressed_qr:.2f}")

        return LiquidityStressResult(
            current_cash=cash_proxy,
            monthly_burn=round(monthly_costs, 2),
            months_of_cash=round(months_cash, 1) if months_cash is not None else None,
            stressed_quick_ratio=stressed_qr,
            stress_scenarios=scenarios,
            risk_level=risk,
            summary=" | ".join(parts),
        )

    # ------------------------------------------------------------------
    # Health Score Helpers (restored)
    # ------------------------------------------------------------------

    _HEALTH_GRADES = [
        (90, "A+"), (80, "A"), (70, "B+"), (60, "B"),
        (50, "C+"), (40, "C"), (30, "D"), (0, "F"),
    ]

    @staticmethod
    def _health_grade(score: float) -> str:
        """Map a 0-100 score to a letter grade."""
        for threshold, grade in CharlieAnalyzer._HEALTH_GRADES:
            if score >= threshold:
                return grade
        return "F"

    @staticmethod
    def _traffic_light(score: float) -> str:
        """Map a 0-100 score to a traffic-light status."""
        if score >= 70:
            return "green"
        elif score >= 40:
            return "yellow"
        return "red"

    def comprehensive_health_score(self, data: FinancialData) -> ComprehensiveHealthResult:
        """Compute a comprehensive financial health score (0-100).

        Aggregates 7 dimensions with configurable weights:
        - Profitability (20%)
        - Liquidity (15%)
        - Leverage/Solvency (15%)
        - Efficiency (10%)
        - Cash Flow Quality (15%)
        - Capital Efficiency (10%)
        - Debt Service (15%)

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ComprehensiveHealthResult
        """
        dimensions: List[HealthDimension] = []

        # --- 1. Profitability (20%) ---
        prof_score = 50.0
        gm = safe_divide(data.gross_profit, data.revenue)
        nm = safe_divide(data.net_income, data.revenue)
        roe = safe_divide(data.net_income, data.total_equity)
        if gm is not None:
            if gm >= 0.50:
                prof_score += 20
            elif gm >= 0.30:
                prof_score += 10
            elif gm < 0.10:
                prof_score -= 20
        if nm is not None:
            if nm >= 0.15:
                prof_score += 15
            elif nm >= 0.05:
                prof_score += 5
            elif nm < 0:
                prof_score -= 20
        if roe is not None:
            if roe >= 0.15:
                prof_score += 15
            elif roe >= 0.08:
                prof_score += 5
            elif roe < 0:
                prof_score -= 10
        prof_score = max(0, min(100, prof_score))
        dimensions.append(HealthDimension(
            name="Profitability", score=prof_score, weight=0.20,
            status=self._traffic_light(prof_score),
            detail=f"GM={gm:.1%}" if gm is not None else "N/A",
        ))

        # --- 2. Liquidity (15%) ---
        liq_score = 50.0
        cr = safe_divide(data.current_assets, data.current_liabilities)
        qr = safe_divide(
            (data.current_assets or 0) - (data.inventory or 0),
            data.current_liabilities,
        ) if data.current_liabilities and data.current_liabilities > 0 else None
        if cr is not None:
            if cr >= 2.0:
                liq_score += 25
            elif cr >= 1.5:
                liq_score += 15
            elif cr >= 1.0:
                liq_score += 5
            else:
                liq_score -= 25
        if qr is not None:
            if qr >= 1.5:
                liq_score += 15
            elif qr >= 1.0:
                liq_score += 5
            elif qr < 0.5:
                liq_score -= 15
        liq_score = max(0, min(100, liq_score))
        dimensions.append(HealthDimension(
            name="Liquidity", score=liq_score, weight=0.15,
            status=self._traffic_light(liq_score),
            detail=f"CR={cr:.2f}" if cr is not None else "N/A",
        ))

        # --- 3. Leverage / Solvency (15%) ---
        lev_score = 50.0
        de = safe_divide(data.total_debt, data.total_equity)
        da = safe_divide(data.total_liabilities, data.total_assets)
        if de is not None:
            if de < 0.5:
                lev_score += 25
            elif de < 1.0:
                lev_score += 10
            elif de < 2.0:
                lev_score -= 5
            else:
                lev_score -= 25
        if da is not None:
            if da < 0.40:
                lev_score += 15
            elif da < 0.60:
                lev_score += 5
            elif da > 0.80:
                lev_score -= 15
        lev_score = max(0, min(100, lev_score))
        dimensions.append(HealthDimension(
            name="Leverage", score=lev_score, weight=0.15,
            status=self._traffic_light(lev_score),
            detail=f"D/E={de:.2f}" if de is not None else "N/A",
        ))

        # --- 4. Efficiency (10%) ---
        eff_score = 50.0
        at = safe_divide(data.revenue, data.total_assets)
        if at is not None:
            if at >= 1.0:
                eff_score += 25
            elif at >= 0.5:
                eff_score += 10
            elif at < 0.2:
                eff_score -= 15
        eff_score = max(0, min(100, eff_score))
        dimensions.append(HealthDimension(
            name="Efficiency", score=eff_score, weight=0.10,
            status=self._traffic_light(eff_score),
            detail=f"AT={at:.2f}x" if at is not None else "N/A",
        ))

        # --- 5. Cash Flow Quality (15%) ---
        cf_score = 50.0
        ocf = data.operating_cash_flow
        ni = data.net_income
        if ocf is not None:
            if ocf > 0:
                cf_score += 15
            else:
                cf_score -= 20
        if ocf is not None and ni is not None and abs(ni) > 1e-12:
            ratio = ocf / ni
            if ni > 0:
                if ratio >= 1.2:
                    cf_score += 20
                elif ratio >= 0.8:
                    cf_score += 10
                elif ratio < 0.5:
                    cf_score -= 15
        if data.capex is not None and ocf is not None:
            fcf = ocf - data.capex
            if fcf > 0:
                cf_score += 10
            else:
                cf_score -= 10
        cf_score = max(0, min(100, cf_score))
        dimensions.append(HealthDimension(
            name="Cash Flow", score=cf_score, weight=0.15,
            status=self._traffic_light(cf_score),
            detail=f"OCF={'pos' if ocf and ocf > 0 else 'neg/NA'}",
        ))

        # --- 6. Capital Efficiency (10%) ---
        cap_score = 50.0
        ce = self.capital_efficiency(data)
        if ce.roic is not None:
            if ce.roic >= 0.15:
                cap_score += 25
            elif ce.roic >= 0.10:
                cap_score += 15
            elif ce.roic >= 0.05:
                cap_score += 5
            elif ce.roic < 0:
                cap_score -= 25
        if ce.eva is not None:
            if ce.eva > 0:
                cap_score += 15
            else:
                cap_score -= 10
        cap_score = max(0, min(100, cap_score))
        dimensions.append(HealthDimension(
            name="Capital Efficiency", score=cap_score, weight=0.10,
            status=self._traffic_light(cap_score),
            detail=f"ROIC={ce.roic:.1%}" if ce.roic is not None else "N/A",
        ))

        # --- 7. Debt Service (15%) ---
        ds_score = 50.0
        ebit_ds = data.ebit if data.ebit is not None else data.operating_income
        interest_ds = data.interest_expense or 0
        debt_ds = data.total_debt or 0
        ocf_ds = data.operating_cash_flow
        int_cov = safe_divide(ebit_ds, interest_ds) if interest_ds > 0 else None
        principal_est = debt_ds / 10 if debt_ds > 0 else 0
        annual_svc = interest_ds + principal_est
        if annual_svc > 0 and ocf_ds is not None:
            dscr = ocf_ds / annual_svc
        elif annual_svc > 0 and data.ebitda is not None:
            dscr = data.ebitda / annual_svc
        else:
            dscr = None
        if dscr is not None:
            if dscr >= 2.0:
                ds_score += 30
            elif dscr >= 1.25:
                ds_score += 15
            elif dscr >= 1.0:
                ds_score -= 5
            else:
                ds_score -= 30
        if int_cov is not None:
            if int_cov >= 5.0:
                ds_score += 15
            elif int_cov >= 3.0:
                ds_score += 5
            elif int_cov < 1.5:
                ds_score -= 15
        ds_score = max(0, min(100, ds_score))
        dimensions.append(HealthDimension(
            name="Debt Service", score=ds_score, weight=0.15,
            status=self._traffic_light(ds_score),
            detail=f"DSCR={dscr:.2f}x" if dscr is not None else "N/A",
        ))

        # --- Weighted overall ---
        overall = sum(d.score * d.weight for d in dimensions)
        overall = max(0, min(100, overall))
        grade = self._health_grade(overall)

        green = sum(1 for d in dimensions if d.status == "green")
        yellow = sum(1 for d in dimensions if d.status == "yellow")
        red = sum(1 for d in dimensions if d.status == "red")
        summary = f"Overall: {grade} ({overall:.0f}/100) | {green} green, {yellow} yellow, {red} red"

        return ComprehensiveHealthResult(
            overall_score=overall,
            grade=grade,
            dimensions=dimensions,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 16 – Operating Leverage & Break-Even Analysis
    # ------------------------------------------------------------------

    def operating_leverage_analysis(self, data: FinancialData) -> OperatingLeverageResult:
        """Analyze operating leverage, cost structure, and break-even point.

        Estimates fixed vs variable cost split from COGS/OpEx composition.
        Calculates contribution margin, DOL, break-even revenue, and margin of safety.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        OperatingLeverageResult
        """
        revenue = data.revenue or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        oi = data.operating_income
        depreciation = data.depreciation or 0

        # --- Estimate fixed vs variable costs ---
        # Heuristic: depreciation + portion of opex are fixed; COGS is mostly variable
        # We treat COGS as variable, depreciation as fixed, and split remaining opex 50/50
        variable_costs = cogs
        non_dep_opex = max(0, opex - depreciation)
        fixed_costs = depreciation + non_dep_opex * 0.5
        variable_costs += non_dep_opex * 0.5

        total_costs = fixed_costs + variable_costs

        # --- Contribution margin ---
        contribution_margin = revenue - variable_costs if revenue > 0 else None
        cm_ratio = safe_divide(contribution_margin, revenue) if contribution_margin is not None else None

        # --- Break-even revenue ---
        break_even = safe_divide(fixed_costs, cm_ratio) if cm_ratio and cm_ratio > 0 else None

        # --- Margin of safety ---
        mos = (revenue - break_even) if break_even is not None and revenue > 0 else None
        mos_pct = safe_divide(mos, revenue) if mos is not None else None

        # --- Degree of Operating Leverage ---
        # DOL = Contribution Margin / Operating Income
        dol = None
        if contribution_margin is not None and oi is not None and abs(oi) > 1e-12:
            dol = contribution_margin / oi

        # --- Cost structure classification ---
        fixed_pct = safe_divide(fixed_costs, total_costs) if total_costs > 0 else None
        if fixed_pct is not None:
            if fixed_pct >= 0.60:
                structure = "High Fixed"
            elif fixed_pct >= 0.40:
                structure = "Balanced"
            else:
                structure = "High Variable"
        else:
            structure = "N/A"

        # --- Summary ---
        parts = []
        if dol is not None:
            parts.append(f"DOL={dol:.2f}x")
        if cm_ratio is not None:
            parts.append(f"CM Ratio={cm_ratio:.1%}")
        if break_even is not None:
            parts.append(f"Break-Even=${break_even:,.0f}")
        if mos_pct is not None:
            parts.append(f"Margin of Safety={mos_pct:.1%}")
        parts.append(f"Structure: {structure}")
        summary = "Operating Leverage: " + " | ".join(parts)

        return OperatingLeverageResult(
            degree_of_operating_leverage=dol,
            contribution_margin=contribution_margin,
            contribution_margin_ratio=cm_ratio,
            estimated_fixed_costs=fixed_costs,
            estimated_variable_costs=variable_costs,
            break_even_revenue=break_even,
            margin_of_safety=mos,
            margin_of_safety_pct=mos_pct,
            cost_structure=structure,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 17 – Cash Flow Quality & Free Cash Flow Yield
    # ------------------------------------------------------------------

    def cash_flow_quality(self, data: FinancialData) -> CashFlowQualityResult:
        """Analyze cash flow quality and free cash flow metrics.

        Computes FCF, FCF yield, FCF margin, OCF/NI quality ratio,
        capex intensity, and cash conversion efficiency.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        CashFlowQualityResult
        """
        ocf = data.operating_cash_flow
        ni = data.net_income
        capex = data.capex or 0
        revenue = data.revenue or 0
        ebitda = data.ebitda

        # --- Free Cash Flow ---
        fcf = (ocf - capex) if ocf is not None else None

        # --- FCF Yield (FCF / Enterprise Value proxy: equity + debt) ---
        ev_proxy = (data.total_equity or 0) + (data.total_debt or 0)
        fcf_yield = safe_divide(fcf, ev_proxy) if fcf is not None and ev_proxy > 0 else None

        # --- FCF Margin ---
        fcf_margin = safe_divide(fcf, revenue) if fcf is not None and revenue > 0 else None

        # --- OCF to Net Income ---
        ocf_to_ni = safe_divide(ocf, ni) if ocf is not None and ni is not None and abs(ni) > 1e-12 else None

        # --- Capex intensity ---
        capex_to_rev = safe_divide(capex, revenue) if revenue > 0 else None
        capex_to_ocf = safe_divide(capex, ocf) if ocf is not None and ocf > 0 else None

        # --- Cash Conversion Efficiency (OCF / EBITDA) ---
        cce = safe_divide(ocf, ebitda) if ocf is not None and ebitda is not None and abs(ebitda) > 1e-12 else None

        # --- Quality scoring ---
        score = 5.0  # Start at mid-point
        indicators = []

        # OCF > NI is positive (low accruals)
        if ocf_to_ni is not None:
            if ocf_to_ni >= 1.2:
                score += 2.0
                indicators.append("OCF exceeds NI significantly (strong)")
            elif ocf_to_ni >= 0.8:
                score += 1.0
                indicators.append("OCF roughly matches NI (adequate)")
            elif ocf_to_ni > 0:
                score -= 1.0
                indicators.append("OCF well below NI (weak cash conversion)")
            else:
                score -= 2.0
                indicators.append("Negative OCF/NI ratio (poor)")

        # Positive FCF
        if fcf is not None:
            if fcf > 0:
                score += 1.5
                indicators.append("Positive free cash flow")
            else:
                score -= 1.5
                indicators.append("Negative free cash flow")

        # FCF margin
        if fcf_margin is not None:
            if fcf_margin >= 0.15:
                score += 1.0
                indicators.append("High FCF margin (>=15%)")
            elif fcf_margin >= 0.05:
                score += 0.5
                indicators.append("Moderate FCF margin")
            elif fcf_margin < 0:
                score -= 1.0
                indicators.append("Negative FCF margin")

        # Cash conversion efficiency
        if cce is not None:
            if cce >= 0.80:
                score += 1.0
                indicators.append("Strong cash conversion (>=80% of EBITDA)")
            elif cce >= 0.50:
                score += 0.5
                indicators.append("Adequate cash conversion")
            elif cce < 0.30:
                score -= 1.0
                indicators.append("Weak cash conversion (<30% of EBITDA)")

        # Capex intensity
        if capex_to_rev is not None:
            if capex_to_rev > 0.20:
                score -= 0.5
                indicators.append("High capex intensity (>20% of revenue)")
            elif capex_to_rev < 0.05:
                score += 0.5
                indicators.append("Low capex intensity (<5%)")

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Strong"
        elif score >= 6:
            grade = "Adequate"
        elif score >= 4:
            grade = "Weak"
        else:
            grade = "Poor"

        # Summary
        parts = [f"Cash Flow Quality: {grade} ({score:.1f}/10)"]
        if fcf is not None:
            parts.append(f"FCF=${fcf:,.0f}")
        if fcf_yield is not None:
            parts.append(f"FCF Yield={fcf_yield:.1%}")
        if cce is not None:
            parts.append(f"Cash Conversion={cce:.0%}")
        summary = " | ".join(parts)

        return CashFlowQualityResult(
            fcf=fcf,
            fcf_yield=fcf_yield,
            fcf_margin=fcf_margin,
            ocf_to_net_income=ocf_to_ni,
            capex_to_revenue=capex_to_rev,
            capex_to_ocf=capex_to_ocf,
            cash_conversion_efficiency=cce,
            quality_grade=grade,
            indicators=indicators,
            summary=summary,
        )

    def asset_efficiency_analysis(self, data: FinancialData) -> AssetEfficiencyResult:
        """Analyze asset efficiency and turnover ratios.

        Computes total asset turnover, fixed asset turnover, inventory turnover,
        receivables turnover, payables turnover, working capital turnover,
        equity turnover, and an efficiency score.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        AssetEfficiencyResult
        """
        revenue = data.revenue or 0
        cogs = data.cogs or 0
        ta = data.total_assets
        ca = data.current_assets
        cl = data.current_liabilities
        inv = data.inventory
        ar = data.accounts_receivable
        ap = data.accounts_payable
        equity = data.total_equity

        # Use average values if available, otherwise point-in-time
        avg_ta = data.avg_total_assets or ta
        avg_inv = data.avg_inventory or inv
        avg_ar = data.avg_receivables or ar
        avg_ap = data.avg_payables or ap

        # --- Turnover ratios ---
        total_at = safe_divide(revenue, avg_ta) if avg_ta is not None and avg_ta > 0 else None

        # Fixed assets = Total assets - Current assets
        fixed_assets = (ta - ca) if ta is not None and ca is not None and (ta - ca) > 0 else None
        fixed_at = safe_divide(revenue, fixed_assets) if fixed_assets is not None and revenue > 0 else None

        inv_turnover = safe_divide(cogs, avg_inv) if avg_inv is not None and avg_inv > 0 and cogs > 0 else None

        ar_turnover = safe_divide(revenue, avg_ar) if avg_ar is not None and avg_ar > 0 and revenue > 0 else None

        ap_turnover = safe_divide(cogs, avg_ap) if avg_ap is not None and avg_ap > 0 and cogs > 0 else None

        # Working capital turnover
        wc = (ca - cl) if ca is not None and cl is not None else None
        wc_turnover = safe_divide(revenue, wc) if wc is not None and wc > 0 and revenue > 0 else None

        # Equity turnover
        eq_turnover = safe_divide(revenue, equity) if equity is not None and equity > 0 and revenue > 0 else None

        # --- Efficiency scoring (0-10) ---
        score = 5.0
        count = 0

        if total_at is not None:
            count += 1
            if total_at >= 1.5:
                score += 1.5
            elif total_at >= 0.8:
                score += 0.5
            elif total_at < 0.3:
                score -= 1.5

        if inv_turnover is not None:
            count += 1
            if inv_turnover >= 8:
                score += 1.5
            elif inv_turnover >= 4:
                score += 0.5
            elif inv_turnover < 2:
                score -= 1.0

        if ar_turnover is not None:
            count += 1
            if ar_turnover >= 10:
                score += 1.0
            elif ar_turnover >= 6:
                score += 0.5
            elif ar_turnover < 3:
                score -= 1.0

        if fixed_at is not None:
            count += 1
            if fixed_at >= 3.0:
                score += 1.0
            elif fixed_at >= 1.5:
                score += 0.5
            elif fixed_at < 0.5:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Average"
        else:
            grade = "Below Average"

        # Summary
        parts = [f"Asset Efficiency: {grade} ({score:.1f}/10)"]
        if total_at is not None:
            parts.append(f"Asset Turnover={total_at:.2f}x")
        if inv_turnover is not None:
            parts.append(f"Inventory Turnover={inv_turnover:.1f}x")
        if ar_turnover is not None:
            parts.append(f"AR Turnover={ar_turnover:.1f}x")
        summary = " | ".join(parts)

        return AssetEfficiencyResult(
            total_asset_turnover=total_at,
            fixed_asset_turnover=fixed_at,
            inventory_turnover=inv_turnover,
            receivables_turnover=ar_turnover,
            payables_turnover=ap_turnover,
            working_capital_turnover=wc_turnover,
            equity_turnover=eq_turnover,
            efficiency_score=score,
            efficiency_grade=grade,
            summary=summary,
        )

    def profitability_decomposition(self, data: FinancialData) -> ProfitabilityDecompResult:
        """Decompose profitability into return components.

        Computes ROE, ROA, ROIC, NOPAT, invested capital, spread,
        economic profit, and related leverage/efficiency metrics.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ProfitabilityDecompResult
        """
        revenue = data.revenue or 0
        ni = data.net_income
        ta = data.total_assets
        equity = data.total_equity
        ebit = data.ebit if data.ebit is not None else data.operating_income
        debt = data.total_debt or 0
        interest = data.interest_expense or 0
        cl = data.current_liabilities or 0

        # --- ROE = NI / Equity ---
        roe = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None

        # --- ROA = NI / Total Assets ---
        roa = safe_divide(ni, ta) if ni is not None and ta is not None and ta > 0 else None

        # --- Invested Capital = Total Debt + Equity (or TA - non-interest-bearing CL) ---
        invested_capital = None
        if equity is not None and equity > 0:
            invested_capital = equity + debt
        elif ta is not None and ta > 0:
            invested_capital = ta - cl

        # --- NOPAT = EBIT × (1 - tax_rate) ---
        nopat = None
        if ebit is not None:
            nopat = ebit * (1 - self._tax_rate)

        # --- ROIC = NOPAT / Invested Capital ---
        roic = safe_divide(nopat, invested_capital) if nopat is not None and invested_capital is not None and invested_capital > 0 else None

        # --- Cost of capital proxy (simplified WACC estimate) ---
        # Use interest rate on debt as debt cost, assume 10% equity cost
        cost_of_debt = safe_divide(interest, debt) if debt > 0 else 0.05
        equity_weight = safe_divide(equity, invested_capital) if invested_capital and invested_capital > 0 and equity else 0.5
        debt_weight = 1 - equity_weight
        wacc_proxy = (equity_weight * 0.10) + (debt_weight * (cost_of_debt or 0.05) * (1 - self._tax_rate))

        # --- Spread = ROIC - WACC proxy ---
        spread = (roic - wacc_proxy) if roic is not None else None

        # --- Economic profit = Spread × Invested Capital ---
        economic_profit = None
        if spread is not None and invested_capital is not None and invested_capital > 0:
            economic_profit = spread * invested_capital

        # --- Asset turnover ---
        asset_to = safe_divide(revenue, ta) if ta is not None and ta > 0 and revenue > 0 else None

        # --- Financial leverage = TA / Equity ---
        fin_leverage = safe_divide(ta, equity) if ta is not None and equity is not None and equity > 0 else None

        # --- Tax efficiency = NI / EBT ---
        ebt = data.ebt
        if ebt is None and ni is not None:
            ebt = safe_divide(ni, 1 - self._tax_rate)
        tax_eff = safe_divide(ni, ebt) if ni is not None and ebt is not None and ebt != 0 else None

        # --- Capital intensity = TA / Revenue ---
        cap_intensity = safe_divide(ta, revenue) if ta is not None and revenue > 0 else None

        # --- Scoring (0-10) ---
        score = 5.0

        if roe is not None:
            if roe >= 0.20:
                score += 2.0
            elif roe >= 0.12:
                score += 1.0
            elif roe < 0:
                score -= 2.0

        if roic is not None:
            if roic >= 0.15:
                score += 1.5
            elif roic >= 0.08:
                score += 0.5
            elif roic < 0:
                score -= 1.5

        if spread is not None:
            if spread > 0.05:
                score += 1.0
            elif spread > 0:
                score += 0.5
            elif spread < -0.05:
                score -= 1.0

        if roa is not None:
            if roa >= 0.10:
                score += 0.5
            elif roa < 0:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Elite"
        elif score >= 6:
            grade = "Strong"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Poor"

        # Summary
        parts = [f"Profitability: {grade} ({score:.1f}/10)"]
        if roe is not None:
            parts.append(f"ROE={roe:.1%}")
        if roic is not None:
            parts.append(f"ROIC={roic:.1%}")
        if spread is not None:
            parts.append(f"Spread={spread:.1%}")
        summary = " | ".join(parts)

        return ProfitabilityDecompResult(
            roe=roe,
            roa=roa,
            roic=roic,
            invested_capital=invested_capital,
            nopat=nopat,
            spread=spread,
            economic_profit=economic_profit,
            asset_turnover=asset_to,
            financial_leverage=fin_leverage,
            tax_efficiency=tax_eff,
            capital_intensity=cap_intensity,
            profitability_score=score,
            profitability_grade=grade,
            summary=summary,
        )

    def risk_adjusted_performance(self, data: FinancialData) -> RiskAdjustedResult:
        """Compute risk-adjusted performance metrics.

        Evaluates returns relative to financial and operating risk,
        including leverage-adjusted returns, margin of safety, and
        risk-return trade-off ratios.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        RiskAdjustedResult
        """
        ni = data.net_income
        equity = data.total_equity
        ta = data.total_assets
        debt = data.total_debt or 0
        ebit = data.ebit if data.ebit is not None else data.operating_income
        interest = data.interest_expense or 0
        revenue = data.revenue or 0

        # --- ROE and ROA ---
        roe = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None
        roa = safe_divide(ni, ta) if ni is not None and ta is not None and ta > 0 else None

        # --- Leverage ratio ---
        leverage = safe_divide(ta, equity) if ta is not None and equity is not None and equity > 0 else None

        # --- Return on risk = ROE / leverage_ratio ---
        # Adjusts for the fact that high ROE from leverage isn't truly "better"
        return_on_risk = safe_divide(roe, leverage) if roe is not None and leverage is not None and leverage > 0 else None

        # --- Risk-adjusted ROE = ROE × (Equity / Total Assets) ---
        # Penalizes ROE that comes from high leverage
        risk_adj_roe = None
        if roe is not None and ta is not None and equity is not None and ta > 0:
            equity_ratio = equity / ta
            risk_adj_roe = roe * equity_ratio

        # --- Risk-adjusted ROA = ROA × (1 - Debt/Assets) ---
        risk_adj_roa = None
        if roa is not None and ta is not None and ta > 0:
            debt_ratio = debt / ta
            risk_adj_roa = roa * (1 - debt_ratio)

        # --- Debt-adjusted return = NI / (Equity + Debt) ---
        invested = (equity or 0) + debt
        debt_adj_return = safe_divide(ni, invested) if ni is not None and invested > 0 else None

        # --- Volatility proxy: abs(operating margin - net margin) ---
        op_margin = safe_divide(ebit, revenue) if ebit is not None and revenue > 0 else None
        net_margin = safe_divide(ni, revenue) if ni is not None and revenue > 0 else None
        vol_proxy = None
        if op_margin is not None and net_margin is not None:
            vol_proxy = abs(op_margin - net_margin)

        # --- Downside risk proxy: Debt / EBITDA (higher = more risk) ---
        ebitda = data.ebitda
        downside = safe_divide(debt, ebitda) if ebitda is not None and ebitda > 0 else None

        # --- Margin of safety = (EBIT - Interest) / Interest ---
        margin_safety = None
        if ebit is not None:
            margin_safety = safe_divide(ebit - interest, interest)

        # --- Return per unit risk ---
        # Use ROE / (Debt/Equity ratio), higher = better risk-adjusted
        de_ratio = safe_divide(debt, equity) if equity is not None and equity > 0 else None
        return_per_risk = None
        if roe is not None and de_ratio is not None:
            return_per_risk = safe_divide(roe, de_ratio)

        # --- Scoring (0-10) ---
        score = 5.0

        if return_on_risk is not None:
            if return_on_risk >= 0.10:
                score += 1.5
            elif return_on_risk >= 0.05:
                score += 0.5
            elif return_on_risk < 0:
                score -= 1.5

        if margin_safety is not None:
            if margin_safety >= 5.0:
                score += 2.0
            elif margin_safety >= 2.0:
                score += 1.0
            elif margin_safety < 1.0:
                score -= 1.5

        if downside is not None:
            if downside <= 1.0:
                score += 1.0
            elif downside <= 3.0:
                score += 0.0
            elif downside > 5.0:
                score -= 1.5

        if risk_adj_roe is not None:
            if risk_adj_roe >= 0.12:
                score += 1.0
            elif risk_adj_roe >= 0.06:
                score += 0.5
            elif risk_adj_roe < 0:
                score -= 1.0

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Superior"
        elif score >= 6:
            grade = "Favorable"
        elif score >= 4:
            grade = "Neutral"
        else:
            grade = "Elevated"

        # Summary
        parts = [f"Risk-Adjusted: {grade} ({score:.1f}/10)"]
        if return_on_risk is not None:
            parts.append(f"Return/Risk={return_on_risk:.1%}")
        if margin_safety is not None:
            parts.append(f"Safety Margin={margin_safety:.1f}x")
        summary = " | ".join(parts)

        return RiskAdjustedResult(
            return_on_risk=return_on_risk,
            risk_adjusted_roe=risk_adj_roe,
            risk_adjusted_roa=risk_adj_roa,
            debt_adjusted_return=debt_adj_return,
            volatility_proxy=vol_proxy,
            downside_risk=downside,
            margin_of_safety=margin_safety,
            return_per_unit_risk=return_per_risk,
            risk_score=score,
            risk_grade=grade,
            summary=summary,
        )

    def valuation_indicators(self, data: FinancialData) -> ValuationIndicatorsResult:
        """Compute valuation indicators and intrinsic value proxies.

        Since we don't have market price data, this uses fundamental-only
        proxies: earnings yield, EV/EBITDA, Graham Number, and a simplified
        DCF-lite intrinsic value estimate based on earnings power.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ValuationIndicatorsResult
        """
        ni = data.net_income
        equity = data.total_equity
        ta = data.total_assets
        debt = data.total_debt or 0
        ebit = data.ebit if data.ebit is not None else data.operating_income
        ebitda = data.ebitda
        revenue = data.revenue or 0
        interest = data.interest_expense or 0
        cash = data.cash or (data.current_assets - data.current_liabilities
                             if data.current_assets is not None and data.current_liabilities is not None
                             else None)

        # --- Earnings yield proxy = NI / Equity (like E/P without market price) ---
        earnings_yield = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None

        # --- Book value proxy = Equity ---
        bv_proxy = equity if equity is not None and equity > 0 else None

        # --- Enterprise Value proxy = Equity + Debt - Cash ---
        ev_proxy = None
        if equity is not None:
            cash_val = cash if cash is not None else 0
            ev_proxy = equity + debt - cash_val
            if ev_proxy <= 0:
                ev_proxy = None

        # --- EV / EBITDA ---
        ev_ebitda = safe_divide(ev_proxy, ebitda) if ev_proxy is not None and ebitda is not None and ebitda > 0 else None

        # --- EV / Revenue ---
        ev_revenue = safe_divide(ev_proxy, revenue) if ev_proxy is not None and revenue > 0 else None

        # --- EV / EBIT ---
        ev_ebit = safe_divide(ev_proxy, ebit) if ev_proxy is not None and ebit is not None and ebit > 0 else None

        # --- Price-to-book proxy ---
        # Use (EV / Equity) as proxy since we lack market cap
        ptb_proxy = safe_divide(ev_proxy, equity) if ev_proxy is not None and equity is not None and equity > 0 else None

        # --- ROIC (from profitability decomp, repeated for convenience) ---
        invested = (equity or 0) + debt
        nopat = ebit * (1 - self._tax_rate) if ebit is not None else None
        roic = safe_divide(nopat, invested) if nopat is not None and invested > 0 else None

        # --- Graham Number proxy = sqrt(22.5 × EPS_proxy × BV_proxy) ---
        # Use NI as EPS proxy, Equity as BV proxy (no per-share data)
        graham = None
        if ni is not None and ni > 0 and bv_proxy is not None:
            import math
            graham = math.sqrt(22.5 * ni * bv_proxy)

        # --- DCF-lite intrinsic value proxy ---
        # Capitalized earnings: NI / discount_rate (assumes perpetuity)
        discount_rate = 0.10  # 10% required return
        intrinsic = None
        if ni is not None and ni > 0:
            intrinsic = ni / discount_rate

        # --- Margin of safety % = (intrinsic - book) / intrinsic ---
        mos_pct = None
        if intrinsic is not None and bv_proxy is not None:
            mos_pct = safe_divide(intrinsic - bv_proxy, intrinsic)

        # --- Scoring (0-10) ---
        score = 5.0

        # EV/EBITDA scoring (lower is cheaper)
        if ev_ebitda is not None:
            if ev_ebitda < 6.0:
                score += 1.5
            elif ev_ebitda < 10.0:
                score += 0.5
            elif ev_ebitda > 15.0:
                score -= 1.0

        # Earnings yield scoring (higher is cheaper)
        if earnings_yield is not None:
            if earnings_yield >= 0.15:
                score += 2.0
            elif earnings_yield >= 0.10:
                score += 1.0
            elif earnings_yield >= 0.05:
                score += 0.5
            elif earnings_yield < 0:
                score -= 1.5

        # ROIC scoring (quality of returns)
        if roic is not None:
            if roic >= 0.15:
                score += 1.0
            elif roic >= 0.08:
                score += 0.5
            elif roic < 0:
                score -= 1.0

        # Margin of safety scoring
        if mos_pct is not None:
            if mos_pct >= 0.30:
                score += 0.5
            elif mos_pct < 0:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Undervalued"
        elif score >= 6:
            grade = "Fair Value"
        elif score >= 4:
            grade = "Fully Valued"
        else:
            grade = "Overvalued"

        # Summary
        parts = [f"Valuation: {grade} ({score:.1f}/10)"]
        if ev_ebitda is not None:
            parts.append(f"EV/EBITDA={ev_ebitda:.1f}x")
        if earnings_yield is not None:
            parts.append(f"Earnings Yield={earnings_yield:.1%}")
        summary = " | ".join(parts)

        return ValuationIndicatorsResult(
            earnings_yield=earnings_yield,
            book_value_per_share_proxy=bv_proxy,
            ev_proxy=ev_proxy,
            ev_to_ebitda=ev_ebitda,
            ev_to_revenue=ev_revenue,
            ev_to_ebit=ev_ebit,
            price_to_book_proxy=ptb_proxy,
            return_on_invested_capital=roic,
            graham_number_proxy=graham,
            intrinsic_value_proxy=intrinsic,
            margin_of_safety_pct=mos_pct,
            valuation_score=score,
            valuation_grade=grade,
            summary=summary,
        )

    def sustainable_growth_analysis(self, data: FinancialData) -> SustainableGrowthResult:
        """Analyze sustainable and internal growth capacity.

        Computes the maximum growth rate achievable without external
        financing (internal growth) and with constant capital structure
        (sustainable growth), along with reinvestment and plowback metrics.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        SustainableGrowthResult
        """
        ni = data.net_income
        equity = data.total_equity
        ta = data.total_assets
        dividends = data.dividends_paid or 0
        capex = data.capex or 0
        depreciation = data.depreciation or 0
        retained_earnings = data.retained_earnings

        # --- ROE and ROA ---
        roe = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None
        roa = safe_divide(ni, ta) if ni is not None and ta is not None and ta > 0 else None

        # --- Payout and retention ---
        payout = None
        retention = None
        if ni is not None and ni > 0:
            payout = dividends / ni if dividends > 0 else 0.0
            retention = 1 - payout

        # --- Sustainable growth rate = ROE × retention ---
        sgr = None
        if roe is not None and retention is not None:
            sgr = roe * retention

        # --- Internal growth rate = ROA × b / (1 - ROA × b) ---
        igr = None
        if roa is not None and retention is not None:
            roa_b = roa * retention
            if roa_b < 1.0:  # avoid division by zero / infinity
                igr = roa_b / (1 - roa_b)

        # --- Plowback capacity = retained earnings / NI ---
        plowback = None
        if retained_earnings is not None and ni is not None and ni > 0:
            plowback = safe_divide(retained_earnings, ni)

        # --- Equity growth rate = retained earnings / equity ---
        eq_growth = safe_divide(retained_earnings, equity) if retained_earnings is not None and equity is not None and equity > 0 else None

        # --- Reinvestment rate = Capex / Depreciation ---
        reinvest = safe_divide(capex, depreciation) if depreciation > 0 else None

        # --- Growth-profitability balance = SGR / cost of equity proxy (10%) ---
        gpb = None
        cost_of_equity = 0.10
        if sgr is not None:
            gpb = safe_divide(sgr, cost_of_equity)

        # --- Scoring (0-10) ---
        score = 5.0

        # SGR scoring
        if sgr is not None:
            if sgr >= 0.15:
                score += 2.0
            elif sgr >= 0.08:
                score += 1.0
            elif sgr >= 0.03:
                score += 0.0
            elif sgr < 0:
                score -= 2.0

        # ROE contribution
        if roe is not None:
            if roe >= 0.20:
                score += 1.0
            elif roe >= 0.10:
                score += 0.5
            elif roe < 0:
                score -= 1.5

        # Retention ratio (high retention = reinvesting for growth)
        if retention is not None:
            if retention >= 0.80:
                score += 1.0
            elif retention >= 0.50:
                score += 0.5

        # Reinvestment rate (maintaining/growing assets)
        if reinvest is not None:
            if reinvest >= 1.5:
                score += 0.5
            elif reinvest >= 1.0:
                score += 0.0
            elif reinvest < 0.5:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "High Growth"
        elif score >= 6:
            grade = "Sustainable"
        elif score >= 4:
            grade = "Moderate"
        else:
            grade = "Constrained"

        # Summary
        parts = [f"Growth Capacity: {grade} ({score:.1f}/10)"]
        if sgr is not None:
            parts.append(f"SGR={sgr:.1%}")
        if igr is not None:
            parts.append(f"IGR={igr:.1%}")
        summary = " | ".join(parts)

        return SustainableGrowthResult(
            sustainable_growth_rate=sgr,
            internal_growth_rate=igr,
            retention_ratio=retention,
            payout_ratio=payout,
            plowback_capacity=plowback,
            roe=roe,
            roa=roa,
            equity_growth_rate=eq_growth,
            reinvestment_rate=reinvest,
            growth_profitability_balance=gpb,
            growth_score=score,
            growth_grade=grade,
            summary=summary,
        )

    def concentration_risk_analysis(self, data: FinancialData) -> ConcentrationRiskResult:
        """Analyze concentration and structural risk across financial dimensions.

        Evaluates how diversified or concentrated the company's financial
        structure is across assets, liabilities, revenue, and earnings.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ConcentrationRiskResult
        """
        revenue = data.revenue
        ta = data.total_assets
        ca = data.current_assets
        cl = data.current_liabilities
        tl = data.total_liabilities
        equity = data.total_equity
        oi = data.operating_income
        ni = data.net_income
        ebit = data.ebit
        ebitda = data.ebitda
        interest = data.interest_expense
        ocf = data.operating_cash_flow
        capex = data.capex or 0
        debt = data.total_debt

        # --- Revenue-asset intensity (asset turnover) ---
        rev_asset = safe_divide(revenue, ta) if revenue is not None and ta is not None and ta > 0 else None

        # --- Operating dependency (operating margin) ---
        op_dep = safe_divide(oi, revenue) if oi is not None and revenue is not None and revenue > 0 else None

        # --- Asset composition ---
        asset_current = safe_divide(ca, ta) if ca is not None and ta is not None and ta > 0 else None
        asset_fixed = (1.0 - asset_current) if asset_current is not None else None

        # --- Liability structure ---
        liab_current = safe_divide(cl, tl) if cl is not None and tl is not None and tl > 0 else None

        # --- Earnings retention (NI / EBITDA) ---
        earn_retention = safe_divide(ni, ebitda) if ni is not None and ebitda is not None and ebitda > 0 else None

        # --- Working capital intensity (NWC / Revenue) ---
        nwc = None
        wc_intensity = None
        if ca is not None and cl is not None:
            nwc = ca - cl
            if revenue is not None and revenue > 0:
                wc_intensity = safe_divide(nwc, revenue)

        # --- Capex intensity ---
        capex_int = safe_divide(capex, revenue) if revenue is not None and revenue > 0 and capex > 0 else None

        # --- Interest burden (Interest / EBIT) ---
        int_burden = safe_divide(interest, ebit) if interest is not None and ebit is not None and ebit > 0 else None

        # --- Cash conversion efficiency (OCF / NI) ---
        cash_conv = safe_divide(ocf, ni) if ocf is not None and ni is not None and ni > 0 else None

        # --- Fixed asset ratio (fixed assets / equity) ---
        fixed_assets = (ta - ca) if ta is not None and ca is not None else None
        fa_ratio = safe_divide(fixed_assets, equity) if fixed_assets is not None and equity is not None and equity > 0 else None

        # --- Debt concentration (debt / TL) ---
        debt_conc = safe_divide(debt, tl) if debt is not None and tl is not None and tl > 0 else None

        # --- Scoring (0-10) ---
        score = 5.0

        # Balanced asset composition: not too heavy on either side
        if asset_current is not None:
            if 0.30 <= asset_current <= 0.60:
                score += 1.0  # well balanced
            elif asset_current > 0.80 or asset_current < 0.10:
                score -= 1.0  # too concentrated

        # Operating dependency - moderate margins are healthier than extremes
        if op_dep is not None:
            if 0.10 <= op_dep <= 0.30:
                score += 1.0  # healthy operating margin
            elif op_dep >= 0.05:
                score += 0.5
            elif op_dep < 0:
                score -= 1.5  # operating losses

        # Cash conversion - OCF/NI near or above 1.0 is positive
        if cash_conv is not None:
            if cash_conv >= 1.2:
                score += 1.5
            elif cash_conv >= 0.8:
                score += 0.5
            elif cash_conv < 0.5:
                score -= 1.0

        # Interest burden - low is good
        if int_burden is not None:
            if int_burden <= 0.15:
                score += 0.5
            elif int_burden >= 0.50:
                score -= 1.0

        # Working capital intensity - moderate is ideal
        if wc_intensity is not None:
            if 0.05 <= wc_intensity <= 0.25:
                score += 0.5
            elif wc_intensity < 0:
                score -= 0.5  # negative NWC can be risky

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Well Diversified"
        elif score >= 6:
            grade = "Balanced"
        elif score >= 4:
            grade = "Concentrated"
        else:
            grade = "Highly Concentrated"

        # Summary
        parts = [f"Concentration Risk: {grade} ({score:.1f}/10)"]
        if rev_asset is not None:
            parts.append(f"Asset Turnover={rev_asset:.2f}x")
        if op_dep is not None:
            parts.append(f"Op Margin={op_dep:.1%}")
        summary = " | ".join(parts)

        return ConcentrationRiskResult(
            revenue_asset_intensity=rev_asset,
            operating_dependency=op_dep,
            asset_composition_current=asset_current,
            asset_composition_fixed=asset_fixed,
            liability_structure_current=liab_current,
            earnings_retention_ratio=earn_retention,
            working_capital_intensity=wc_intensity,
            capex_intensity=capex_int,
            interest_burden=int_burden,
            cash_conversion_efficiency=cash_conv,
            fixed_asset_ratio=fa_ratio,
            debt_concentration=debt_conc,
            concentration_score=score,
            concentration_grade=grade,
            summary=summary,
        )

    def margin_of_safety_analysis(self, data: FinancialData) -> MarginOfSafetyResult:
        """Analyze margin of safety using intrinsic value vs market price."""
        ni = data.net_income
        equity = data.total_equity
        shares = data.shares_outstanding
        price = data.share_price
        ca = data.current_assets
        tl = data.total_liabilities
        ta = data.total_assets

        # --- Market Cap ---
        mcap = None
        if shares and shares > 0 and price and price > 0:
            mcap = shares * price

        # --- Earnings Yield ---
        earn_yield = None
        if ni is not None and mcap and mcap > 0:
            earn_yield = safe_divide(ni, mcap)

        # --- Book Value ---
        bv_per_share = None
        if equity is not None and shares and shares > 0:
            bv_per_share = safe_divide(equity, shares)

        # --- Price to Book ---
        ptb = None
        bv_discount = None
        if bv_per_share is not None and bv_per_share > 0 and price and price > 0:
            ptb = safe_divide(price, bv_per_share)
            bv_discount = 1.0 - ptb  # Positive = below book value

        # --- Intrinsic Value (capitalized earnings at 10%) ---
        iv = None
        iv_margin = None
        if ni is not None and ni > 0:
            iv = ni / 0.10  # Capitalize at 10%
            if mcap and mcap > 0:
                iv_margin = safe_divide(iv - mcap, iv)

        # --- Tangible Book Value (approximate: equity as proxy) ---
        tbv = equity  # No intangibles field, use equity as approximation

        # --- Liquidation Value (70% of tangible BV) ---
        liq_val = None
        if tbv is not None and tbv > 0:
            liq_val = tbv * 0.70

        # --- Net Current Asset Value (Graham NCAV) ---
        ncav = None
        ncav_ps = None
        if ca is not None and tl is not None:
            ncav = ca - tl
            if shares and shares > 0:
                ncav_ps = safe_divide(ncav, shares)

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # Earnings yield
        if earn_yield is not None:
            if earn_yield >= 0.15:
                score += 2.0  # Very high yield = cheap
            elif earn_yield >= 0.10:
                score += 1.0
            elif earn_yield >= 0.05:
                score += 0.0
            elif earn_yield < 0:
                score -= 1.5

        # Intrinsic value margin
        if iv_margin is not None:
            if iv_margin >= 0.50:
                score += 1.5  # IV is 2x+ market cap
            elif iv_margin >= 0.25:
                score += 1.0
            elif iv_margin < 0:
                score -= 1.0  # Overvalued vs IV

        # Book value discount
        if bv_discount is not None:
            if bv_discount > 0:
                score += 0.5  # Trading below book
            elif bv_discount < -1.0:
                score -= 0.5  # Trading at 2x+ book

        # NCAV check
        if ncav_ps is not None and price and price > 0:
            if ncav_ps >= price:
                score += 1.0  # Graham's net-net

        score = max(0.0, min(10.0, score))

        # --- Grade ---
        if score >= 8.0:
            grade = "Wide Margin"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Thin"
        else:
            grade = "No Margin"

        # --- Summary ---
        parts = ["Margin of Safety:"]
        if earn_yield is not None:
            parts.append(f"Earnings yield {earn_yield:.1%}.")
        if iv_margin is not None:
            parts.append(f"Intrinsic value margin {iv_margin:.1%}.")
        if ptb is not None:
            parts.append(f"P/B ratio {ptb:.2f}.")
        parts.append(f"Grade: {grade} (score {score:.1f}/10).")
        summary = " ".join(parts)

        return MarginOfSafetyResult(
            earnings_yield=earn_yield,
            book_value_per_share=bv_per_share,
            price_to_book=ptb,
            book_value_discount=bv_discount,
            intrinsic_value_estimate=iv,
            market_cap=mcap,
            intrinsic_margin=iv_margin,
            tangible_bv=tbv,
            liquidation_value=liq_val,
            net_current_asset_value=ncav,
            ncav_per_share=ncav_ps,
            safety_score=score,
            safety_grade=grade,
            summary=summary,
        )

    def earnings_quality_analysis(self, data: FinancialData) -> EarningsQualityResult:
        """Analyze earnings quality and sustainability."""
        ni = data.net_income
        ocf = data.operating_cash_flow
        ta = data.total_assets
        rev = data.revenue
        ebitda = data.ebitda
        dep = data.depreciation
        capex = data.capex

        # --- Accruals Ratio: (NI - OCF) / TA ---
        accruals = None
        if ni is not None and ocf is not None and ta and ta > 0:
            accruals = safe_divide(ni - ocf, ta)

        # --- Cash to Income: OCF / NI ---
        cash_to_ni = None
        if ocf is not None and ni is not None and ni > 0:
            cash_to_ni = safe_divide(ocf, ni)

        # --- Earnings Persistence (net margin proxy): NI / Revenue ---
        persist = None
        if ni is not None and rev and rev > 0:
            persist = safe_divide(ni, rev)

        # --- Cash Return on Assets: OCF / TA ---
        cash_roa = None
        if ocf is not None and ta and ta > 0:
            cash_roa = safe_divide(ocf, ta)

        # --- Operating Cash Quality: OCF / EBITDA ---
        ocf_ebitda = None
        if ocf is not None and ebitda and ebitda > 0:
            ocf_ebitda = safe_divide(ocf, ebitda)

        # --- Revenue Cash Ratio: OCF / Revenue ---
        rev_cash = None
        if ocf is not None and rev and rev > 0:
            rev_cash = safe_divide(ocf, rev)

        # --- Depreciation to Capex ---
        dep_capex = None
        if dep is not None and capex and capex > 0:
            dep_capex = safe_divide(dep, capex)

        # --- Capex to Revenue ---
        capex_rev = None
        if capex is not None and rev and rev > 0:
            capex_rev = safe_divide(capex, rev)

        # --- Reinvestment Rate: Capex / Depreciation ---
        reinvest = None
        if capex is not None and dep and dep > 0:
            reinvest = safe_divide(capex, dep)

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # Cash-to-income quality
        if cash_to_ni is not None:
            if cash_to_ni >= 1.2:
                score += 2.0  # Cash earnings exceed accrual earnings
            elif cash_to_ni >= 1.0:
                score += 1.0
            elif cash_to_ni >= 0.5:
                score += 0.0
            elif cash_to_ni < 0.5:
                score -= 1.5  # Cash generation much lower than reported NI

        # Accruals quality (lower is better - less accrual manipulation)
        if accruals is not None:
            if accruals <= -0.05:
                score += 1.0  # OCF exceeds NI (conservative accounting)
            elif accruals <= 0.05:
                score += 0.5  # Low accruals
            elif accruals > 0.15:
                score -= 1.5  # High accruals = potential manipulation
            elif accruals > 0.10:
                score -= 0.5

        # Operating cash quality
        if ocf_ebitda is not None:
            if ocf_ebitda >= 0.80:
                score += 0.5  # Strong cash conversion
            elif ocf_ebitda < 0.40:
                score -= 0.5  # Weak conversion

        score = max(0.0, min(10.0, score))

        # --- Grade ---
        if score >= 8.0:
            grade = "High"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Questionable"
        else:
            grade = "Poor"

        # --- Summary ---
        parts = ["Earnings Quality:"]
        if cash_to_ni is not None:
            parts.append(f"Cash/Income ratio {cash_to_ni:.2f}x.")
        if accruals is not None:
            parts.append(f"Accruals ratio {accruals:.2%}.")
        if ocf_ebitda is not None:
            parts.append(f"OCF/EBITDA {ocf_ebitda:.1%}.")
        parts.append(f"Grade: {grade} (score {score:.1f}/10).")
        summary = " ".join(parts)

        return EarningsQualityResult(
            accruals_ratio=accruals,
            cash_to_income=cash_to_ni,
            earnings_persistence=persist,
            cash_return_on_assets=cash_roa,
            accrual_return_on_assets=accruals,
            operating_cash_quality=ocf_ebitda,
            revenue_cash_ratio=rev_cash,
            depreciation_to_capex=dep_capex,
            capex_to_revenue=capex_rev,
            reinvestment_rate=reinvest,
            earnings_quality_score=score,
            earnings_quality_grade=grade,
            summary=summary,
        )

    def financial_flexibility_analysis(self, data: FinancialData) -> FinancialFlexibilityResult:
        """Analyze financial flexibility and adaptive capacity."""
        cash = data.cash
        ta = data.total_assets
        tl = data.total_liabilities
        debt = data.total_debt
        rev = data.revenue
        ocf = data.operating_cash_flow
        capex = data.capex
        ebitda = data.ebitda
        equity = data.total_equity
        ret_earn = data.retained_earnings
        ca = data.current_assets
        cl = data.current_liabilities

        # Cash fallback
        if cash is None and ca is not None and cl is not None:
            cash = ca - cl

        # FCF
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf - capex

        # --- Cash / Assets ---
        cash_ta = safe_divide(cash, ta) if cash is not None and ta and ta > 0 else None

        # --- Cash / Debt ---
        cash_debt = safe_divide(cash, debt) if cash is not None and debt and debt > 0 else None

        # --- Cash / Revenue ---
        cash_rev = safe_divide(cash, rev) if cash is not None and rev and rev > 0 else None

        # --- FCF / Revenue ---
        fcf_rev = safe_divide(fcf, rev) if fcf is not None and rev and rev > 0 else None

        # --- Spare borrowing capacity: (TA * 0.50 - Debt) / TA ---
        spare_cap = None
        if ta and ta > 0 and debt is not None:
            spare_cap = safe_divide(ta * 0.50 - debt, ta)

        # --- Unencumbered assets: TA - TL ---
        unencumbered = None
        if ta is not None and tl is not None:
            unencumbered = ta - tl

        # --- Financial slack: (Cash + FCF) / Debt ---
        slack = None
        if cash is not None and fcf is not None and debt and debt > 0:
            slack = safe_divide(cash + fcf, debt)

        # --- Debt headroom: 3x EBITDA - current debt ---
        headroom = None
        if ebitda is not None and ebitda > 0 and debt is not None:
            headroom = ebitda * 3.0 - debt

        # --- Retained earnings ratio: RE / Equity ---
        re_ratio = None
        if ret_earn is not None and equity and equity > 0:
            re_ratio = safe_divide(ret_earn, equity)

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # Cash-to-assets
        if cash_ta is not None:
            if cash_ta >= 0.20:
                score += 1.5  # Substantial cash buffer
            elif cash_ta >= 0.10:
                score += 0.5
            elif cash_ta < 0.03:
                score -= 1.0  # Very thin cash

        # FCF margin
        if fcf_rev is not None:
            if fcf_rev >= 0.15:
                score += 1.5  # Strong free cash generation
            elif fcf_rev >= 0.05:
                score += 0.5
            elif fcf_rev < 0:
                score -= 1.0  # Negative FCF

        # Debt headroom
        if headroom is not None:
            if headroom > 0:
                score += 1.0  # Room to borrow more
            elif headroom < 0:
                score -= 1.0  # Over-leveraged vs 3x EBITDA

        # Financial slack
        if slack is not None:
            if slack >= 1.0:
                score += 0.5  # Can cover debt with cash+FCF
            elif slack < 0.2:
                score -= 0.5

        score = max(0.0, min(10.0, score))

        # --- Grade ---
        if score >= 8.0:
            grade = "Highly Flexible"
        elif score >= 6.0:
            grade = "Flexible"
        elif score >= 4.0:
            grade = "Constrained"
        else:
            grade = "Rigid"

        # --- Summary ---
        parts = ["Financial Flexibility:"]
        if cash_ta is not None:
            parts.append(f"Cash/Assets {cash_ta:.1%}.")
        if fcf_rev is not None:
            parts.append(f"FCF margin {fcf_rev:.1%}.")
        if headroom is not None:
            parts.append(f"Debt headroom ${headroom:,.0f}.")
        parts.append(f"Grade: {grade} (score {score:.1f}/10).")
        summary = " ".join(parts)

        return FinancialFlexibilityResult(
            cash_to_assets=cash_ta,
            cash_to_debt=cash_debt,
            cash_to_revenue=cash_rev,
            fcf_to_revenue=fcf_rev,
            fcf_margin=fcf_rev,
            spare_borrowing_capacity=spare_cap,
            unencumbered_assets=unencumbered,
            financial_slack=slack,
            debt_headroom=headroom,
            retained_earnings_ratio=re_ratio,
            flexibility_score=score,
            flexibility_grade=grade,
            summary=summary,
        )

    def altman_z_score_analysis(self, data: FinancialData) -> AltmanZScoreResult:
        """Altman Z-Score bankruptcy prediction model (original manufacturing formula)."""
        ca = data.current_assets
        cl = data.current_liabilities
        ta = data.total_assets
        ret_earn = data.retained_earnings
        ebit = data.ebit
        eq = data.total_equity
        tl = data.total_liabilities
        rev = data.revenue

        # Component ratios
        wc_ta = None
        if ca is not None and cl is not None:
            wc_ta = safe_divide(ca - cl, ta)

        re_ta = safe_divide(ret_earn, ta)
        ebit_ta = safe_divide(ebit, ta)
        eq_tl = safe_divide(eq, tl)
        rev_ta = safe_divide(rev, ta)

        # Weighted components
        x1 = 1.2 * wc_ta if wc_ta is not None else None
        x2 = 1.4 * re_ta if re_ta is not None else None
        x3 = 3.3 * ebit_ta if ebit_ta is not None else None
        x4 = 0.6 * eq_tl if eq_tl is not None else None
        x5 = 1.0 * rev_ta if rev_ta is not None else None

        # Z-Score (only if all components available)
        z = None
        zone = ""
        if all(v is not None for v in [x1, x2, x3, x4, x5]):
            z = x1 + x2 + x3 + x4 + x5
            if z > 2.99:
                zone = "Safe"
            elif z >= 1.81:
                zone = "Gray"
            else:
                zone = "Distress"

        # --- Scoring (base 5.0) ---
        score = 5.0

        if z is not None:
            if z > 3.0:
                score += 2.5
            elif z > 2.99:
                score += 2.0
            elif z >= 2.5:
                score += 1.0
            elif z >= 1.81:
                score += 0.0
            elif z >= 1.0:
                score -= 1.5
            else:
                score -= 2.5

        # Bonus/penalty for individual components
        if wc_ta is not None:
            if wc_ta > 0.20:
                score += 0.5
            elif wc_ta < 0:
                score -= 0.5

        if ebit_ta is not None:
            if ebit_ta > 0.15:
                score += 0.5
            elif ebit_ta < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            grade = "Strong"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Watch"
        else:
            grade = "Critical"

        parts = [f"Altman Z-Score — Grade: {grade} ({score:.1f}/10)."]
        if z is not None:
            parts.append(f"Z-Score: {z:.2f} ({zone})")
        summary = " | ".join(parts)

        return AltmanZScoreResult(
            working_capital_to_assets=wc_ta,
            retained_earnings_to_assets=re_ta,
            ebit_to_assets=ebit_ta,
            equity_to_liabilities=eq_tl,
            revenue_to_assets=rev_ta,
            x1_weighted=x1,
            x2_weighted=x2,
            x3_weighted=x3,
            x4_weighted=x4,
            x5_weighted=x5,
            z_score=z,
            z_zone=zone,
            altman_score=score,
            altman_grade=grade,
            summary=summary,
        )

    def piotroski_f_score_analysis(self, data: FinancialData) -> PiotroskiFScoreResult:
        """Piotroski F-Score (adapted for single-period: 7 signals instead of 9)."""
        ni = data.net_income
        ta = data.total_assets
        ocf = data.operating_cash_flow
        ca = data.current_assets
        cl = data.current_liabilities
        debt = data.total_debt
        rev = data.revenue
        gp = data.gross_profit

        # Profitability signals
        roa = safe_divide(ni, ta)
        roa_pos = roa > 0 if roa is not None else None
        ocf_pos = ocf > 0 if ocf is not None else None
        accruals_neg = None
        if ocf is not None and ni is not None:
            accruals_neg = ocf > ni  # Cash earnings exceed accrual earnings

        # Leverage / Liquidity signals
        cr = safe_divide(ca, cl)
        cr_above_1 = cr > 1.0 if cr is not None else None
        dta = safe_divide(debt, ta)
        low_lev = dta < 0.5 if dta is not None else None

        # Efficiency signals
        gm = safe_divide(gp, rev)
        gm_healthy = gm > 0.20 if gm is not None else None
        at = safe_divide(rev, ta)
        at_adequate = at > 0.5 if at is not None else None

        # Count score (each True signal = +1)
        signals = [roa_pos, ocf_pos, accruals_neg, cr_above_1, low_lev, gm_healthy, at_adequate]
        f_score = sum(1 for s in signals if s is True)
        testable = sum(1 for s in signals if s is not None)

        if f_score >= 6:
            grade = "Strong Value"
        elif f_score >= 4:
            grade = "Moderate Value"
        elif f_score >= 2:
            grade = "Weak"
        else:
            grade = "Avoid"

        parts = [f"Piotroski F-Score — {grade} ({f_score}/{testable} signals)."]
        if roa is not None:
            parts.append(f"ROA: {roa:.1%}")
        if cr is not None:
            parts.append(f"Current Ratio: {cr:.2f}x")
        summary = " | ".join(parts)

        return PiotroskiFScoreResult(
            roa_positive=roa_pos,
            ocf_positive=ocf_pos,
            accruals_negative=accruals_neg,
            current_ratio_above_1=cr_above_1,
            low_leverage=low_lev,
            gross_margin_healthy=gm_healthy,
            asset_turnover_adequate=at_adequate,
            roa=roa,
            current_ratio=cr,
            debt_to_assets=dta,
            gross_margin=gm,
            asset_turnover=at,
            f_score=f_score,
            f_score_max=testable if testable > 0 else 7,
            piotroski_grade=grade,
            summary=summary,
        )

    def interest_coverage_analysis(self, data: FinancialData) -> InterestCoverageResult:
        """Comprehensive interest coverage and debt capacity analysis."""
        ebit = data.ebit
        ebitda = data.ebitda
        interest = data.interest_expense
        debt = data.total_debt
        ocf = data.operating_cash_flow
        capex = data.capex
        rev = data.revenue
        eq = data.total_equity

        ebit_cov = safe_divide(ebit, interest)
        ebitda_cov = safe_divide(ebitda, interest)
        d_ebitda = safe_divide(debt, ebitda)
        ocf_debt = safe_divide(ocf, debt)

        fcf_debt = None
        if ocf is not None and capex is not None and debt and debt > 0:
            fcf_debt = safe_divide(ocf - capex, debt)

        # Fixed charge coverage: approximate (no lease data, so FCCR ≈ EBIT coverage)
        fcc = ebit_cov  # Simplification when lease data unavailable

        max_cap = None
        spare_cap = None
        if ebitda is not None:
            max_cap = 3.0 * ebitda
            if debt is not None:
                spare_cap = max_cap - debt

        int_rev = safe_divide(interest, rev)
        d_eq = safe_divide(debt, eq)

        # --- Scoring (base 5.0) ---
        score = 5.0

        if ebit_cov is not None:
            if ebit_cov >= 5.0:
                score += 2.0
            elif ebit_cov >= 3.0:
                score += 1.0
            elif ebit_cov >= 1.5:
                score += 0.0
            elif ebit_cov >= 1.0:
                score -= 1.0
            else:
                score -= 2.0

        if d_ebitda is not None:
            if d_ebitda <= 1.5:
                score += 1.0
            elif d_ebitda <= 3.0:
                score += 0.5
            elif d_ebitda > 5.0:
                score -= 1.0
            elif d_ebitda > 4.0:
                score -= 0.5

        if ocf_debt is not None:
            if ocf_debt >= 0.30:
                score += 0.5
            elif ocf_debt < 0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Strained"
        else:
            grade = "Critical"

        parts = [f"Interest Coverage — Grade: {grade} ({score:.1f}/10)."]
        if ebit_cov is not None:
            parts.append(f"EBIT Coverage: {ebit_cov:.1f}x")
        if d_ebitda is not None:
            parts.append(f"Debt/EBITDA: {d_ebitda:.1f}x")
        summary = " | ".join(parts)

        return InterestCoverageResult(
            ebit_coverage=ebit_cov,
            ebitda_coverage=ebitda_cov,
            debt_to_ebitda=d_ebitda,
            ocf_to_debt=ocf_debt,
            fcf_to_debt=fcf_debt,
            fixed_charge_coverage=fcc,
            max_debt_capacity=max_cap,
            spare_debt_capacity=spare_cap,
            interest_to_revenue=int_rev,
            debt_to_equity=d_eq,
            coverage_score=score,
            coverage_grade=grade,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 37: WACC & Cost of Capital
    # ------------------------------------------------------------------
    def wacc_analysis(self, data: FinancialData) -> WACCResult:
        """Estimate WACC from financial statement data.

        Cost of debt = Interest Expense / Total Debt.
        Cost of equity = implied from ROE (capped/floored for realism).
        WACC = Wd * Rd * (1-T) + We * Re.
        """
        result = WACCResult()

        debt = data.total_debt
        equity = data.total_equity

        # Capital structure weights
        if debt is not None and equity is not None and (debt + equity) > 0:
            total_cap = debt + equity
            result.total_capital = total_cap
            result.debt_weight = safe_divide(debt, total_cap)
            result.equity_weight = safe_divide(equity, total_cap)
            result.debt_to_total_capital = safe_divide(debt, total_cap)
            result.equity_to_total_capital = safe_divide(equity, total_cap)
        else:
            # Without both debt and equity, cannot compute WACC
            result.wacc_score = 5.0
            result.wacc_grade = "Fair"
            result.summary = "WACC: Insufficient data to estimate cost of capital."
            return result

        # Cost of debt
        cost_of_debt = safe_divide(data.interest_expense, debt)
        result.cost_of_debt = cost_of_debt

        # Effective tax rate (proxy: 1 - NI/EBT, where EBT = EBIT - Interest)
        eff_tax = None
        ebit = data.ebit
        ie = data.interest_expense
        ni = data.net_income
        if ebit is not None and ie is not None and ni is not None:
            ebt = ebit - ie
            if ebt > 0:
                eff_tax = 1.0 - safe_divide(ni, ebt, default=0.75)
                eff_tax = max(0.0, min(eff_tax, 0.60))  # Clamp 0-60%
        if eff_tax is None:
            eff_tax = 0.25  # Default assumption
        result.effective_tax_rate = eff_tax

        # After-tax cost of debt
        if cost_of_debt is not None:
            result.after_tax_cost_of_debt = cost_of_debt * (1 - eff_tax)

        # Implied cost of equity from ROE (floored at 8%, capped at 30%)
        roe = safe_divide(ni, equity)
        if roe is not None:
            implied_re = max(0.08, min(abs(roe) + 0.03, 0.30))
        else:
            implied_re = 0.12  # Default
        result.implied_cost_of_equity = implied_re

        # WACC calculation
        wd = result.debt_weight
        we = result.equity_weight
        if result.after_tax_cost_of_debt is not None:
            wacc = wd * result.after_tax_cost_of_debt + we * implied_re
        elif cost_of_debt is not None:
            wacc = wd * cost_of_debt * (1 - eff_tax) + we * implied_re
        else:
            # No cost of debt info; use equity cost only weighted by equity
            wacc = we * implied_re
        result.wacc = wacc

        # Scoring (base 5.0)
        score = 5.0

        # Lower WACC is better
        if wacc is not None:
            if wacc < 0.08:
                score += 2.0
            elif wacc < 0.10:
                score += 1.5
            elif wacc < 0.12:
                score += 0.5
            elif wacc > 0.20:
                score -= 2.0
            elif wacc > 0.15:
                score -= 1.0

        # Capital structure balance
        if wd is not None:
            if 0.20 <= wd <= 0.50:
                score += 0.5  # Balanced mix
            elif wd > 0.70:
                score -= 1.0  # Over-levered
            elif wd < 0.05 and debt > 0:
                score -= 0.5  # Under-utilizing tax shield

        # Cost of debt efficiency
        if cost_of_debt is not None:
            if cost_of_debt < 0.04:
                score += 0.5
            elif cost_of_debt > 0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.wacc_score = score

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Fair"
        else:
            grade = "Expensive"
        result.wacc_grade = grade

        wacc_pct = f"{wacc * 100:.1f}%" if wacc is not None else "N/A"
        result.summary = (
            f"WACC Analysis: Estimated WACC of {wacc_pct}. "
            f"Capital structure: {wd * 100:.0f}% debt / {we * 100:.0f}% equity. "
            f"Grade: {grade}."
        )

        return result

    # ------------------------------------------------------------------
    # Phase 38: Economic Value Added (EVA)
    # ------------------------------------------------------------------
    def eva_analysis(self, data: FinancialData) -> EVAResult:
        """Compute Economic Value Added = NOPAT - (Invested Capital * WACC).

        NOPAT = EBIT * (1 - Tax Rate).
        Invested Capital = Total Assets - Current Liabilities.
        Uses WACC from wacc_analysis().
        """
        result = EVAResult()

        ebit = data.ebit
        ta = data.total_assets
        cl = data.current_liabilities

        # Invested Capital = Total Assets - Current Liabilities
        if ta is not None and cl is not None and ta > 0:
            invested_cap = ta - cl
        elif data.total_debt is not None and data.total_equity is not None:
            invested_cap = data.total_debt + data.total_equity
        else:
            result.eva_score = 5.0
            result.eva_grade = "Marginal"
            result.summary = "EVA: Insufficient data to compute invested capital."
            return result

        result.invested_capital = invested_cap

        # Effective tax rate (same logic as WACC)
        eff_tax = None
        ie = data.interest_expense
        ni = data.net_income
        if ebit is not None and ie is not None and ni is not None:
            ebt = ebit - ie
            if ebt > 0:
                eff_tax = 1.0 - safe_divide(ni, ebt, default=0.75)
                eff_tax = max(0.0, min(eff_tax, 0.60))
        if eff_tax is None:
            eff_tax = 0.25

        # NOPAT
        if ebit is not None:
            nopat = ebit * (1 - eff_tax)
            result.nopat = nopat
        else:
            result.eva_score = 5.0
            result.eva_grade = "Marginal"
            result.summary = "EVA: Insufficient data (no EBIT) to compute NOPAT."
            return result

        # Get WACC
        wacc_result = self.wacc_analysis(data)
        wacc = wacc_result.wacc
        if wacc is None:
            wacc = 0.10  # Default fallback
        result.wacc_used = wacc

        # Capital charge & EVA
        capital_charge = invested_cap * wacc
        result.capital_charge = capital_charge
        eva = nopat - capital_charge
        result.eva = eva

        # EVA margin
        if data.revenue and data.revenue > 0:
            result.eva_margin = safe_divide(eva, data.revenue)

        # ROIC
        roic = safe_divide(nopat, invested_cap)
        result.roic = roic

        # ROIC-WACC spread
        if roic is not None:
            result.roic_wacc_spread = roic - wacc

        # Scoring (base 5.0)
        score = 5.0

        # EVA positive/negative
        if eva > 0:
            score += 2.0
        elif eva < 0:
            score -= 2.0

        # ROIC-WACC spread
        spread = result.roic_wacc_spread
        if spread is not None:
            if spread > 0.10:
                score += 1.0
            elif spread > 0.05:
                score += 0.5
            elif spread < -0.05:
                score -= 1.0
            elif spread < 0:
                score -= 0.5

        # EVA margin
        if result.eva_margin is not None:
            if result.eva_margin > 0.10:
                score += 0.5
            elif result.eva_margin < -0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.eva_score = score

        if score >= 8.0:
            grade = "Value Creator"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Marginal"
        else:
            grade = "Value Destroyer"
        result.eva_grade = grade

        eva_str = f"${eva:,.0f}" if eva is not None else "N/A"
        result.summary = (
            f"EVA Analysis: Economic Value Added of {eva_str}. "
            f"ROIC: {roic * 100:.1f}% vs WACC: {wacc * 100:.1f}%. "
            f"Grade: {grade}."
        )

        return result

    # ------------------------------------------------------------------
    # Phase 39: Free Cash Flow Yield
    # ------------------------------------------------------------------
    def fcf_yield_analysis(self, data: FinancialData) -> FCFYieldResult:
        """Compute Free Cash Flow yield and quality metrics.

        FCF = Operating Cash Flow - CapEx.
        """
        result = FCFYieldResult()

        ocf = data.operating_cash_flow
        capex = data.capex or 0

        if ocf is None:
            result.fcf_score = 5.0
            result.fcf_grade = "Weak"
            result.summary = "FCF Yield: No operating cash flow data available."
            return result

        fcf = ocf - capex
        result.fcf = fcf

        # FCF margin
        result.fcf_margin = safe_divide(fcf, data.revenue)

        # FCF conversion (vs net income)
        result.fcf_to_net_income = safe_divide(fcf, data.net_income)

        # FCF yield on invested capital
        ta = data.total_assets
        cl = data.current_liabilities
        if ta is not None and cl is not None and (ta - cl) > 0:
            result.fcf_yield_on_capital = safe_divide(fcf, ta - cl)
        elif data.total_debt is not None and data.total_equity is not None:
            total_cap = data.total_debt + data.total_equity
            if total_cap > 0:
                result.fcf_yield_on_capital = safe_divide(fcf, total_cap)

        # FCF yield on equity
        result.fcf_yield_on_equity = safe_divide(fcf, data.total_equity)

        # FCF to debt
        result.fcf_to_debt = safe_divide(fcf, data.total_debt)

        # CapEx ratios
        if ocf != 0:
            result.capex_to_ocf = safe_divide(capex, ocf)
        result.capex_to_revenue = safe_divide(capex, data.revenue)

        # Scoring (base 5.0)
        score = 5.0

        # FCF positive/negative
        if fcf > 0:
            score += 1.5
        elif fcf < 0:
            score -= 2.0

        # FCF margin
        fm = result.fcf_margin
        if fm is not None:
            if fm > 0.15:
                score += 1.0
            elif fm > 0.08:
                score += 0.5
            elif fm < 0:
                score -= 1.0

        # FCF conversion quality
        conv = result.fcf_to_net_income
        if conv is not None and data.net_income is not None and data.net_income > 0:
            if conv > 1.0:
                score += 0.5  # FCF exceeds NI (high quality)
            elif conv < 0.5:
                score -= 0.5  # Poor conversion

        # Capital intensity
        capex_ratio = result.capex_to_ocf
        if capex_ratio is not None and capex_ratio >= 0:
            if capex_ratio < 0.30:
                score += 0.5  # Low reinvestment needs
            elif capex_ratio > 0.70:
                score -= 0.5  # Heavy capex burden

        score = max(0.0, min(10.0, score))
        result.fcf_score = score

        if score >= 8.0:
            grade = "Strong"
        elif score >= 6.0:
            grade = "Healthy"
        elif score >= 4.0:
            grade = "Weak"
        else:
            grade = "Negative"
        result.fcf_grade = grade

        fcf_str = f"${fcf:,.0f}" if fcf is not None else "N/A"
        margin_str = f"{fm * 100:.1f}%" if fm is not None else "N/A"
        result.summary = (
            f"FCF Yield: Free cash flow of {fcf_str} ({margin_str} margin). "
            f"Grade: {grade}."
        )

        return result

    def cash_conversion_analysis(self, data: FinancialData) -> CashConversionResult:
        """Phase 41: Cash Conversion Efficiency.

        Computes DSO, DIO, DPO, Cash Conversion Cycle (CCC),
        cash/revenue ratio, OCF/revenue, and OCF/EBITDA.
        """
        result = CashConversionResult()

        revenue = data.revenue
        cogs = data.cogs

        # --- DSO: AR / Revenue * 365 ---
        ar = data.accounts_receivable
        dso = None
        if ar is not None and revenue and revenue > 0:
            dso = safe_divide(ar, revenue, default=0.0) * 365
            result.dso = dso

        # --- DIO: Inventory / COGS * 365 ---
        inv = data.inventory
        dio = None
        if inv is not None and cogs and cogs > 0:
            dio = safe_divide(inv, cogs, default=0.0) * 365
            result.dio = dio

        # --- DPO: AP / COGS * 365 ---
        ap = data.accounts_payable
        dpo = None
        if ap is not None and cogs and cogs > 0:
            dpo = safe_divide(ap, cogs, default=0.0) * 365
            result.dpo = dpo

        # --- CCC: DSO + DIO - DPO ---
        if dso is not None and dio is not None and dpo is not None:
            ccc = dso + dio - dpo
            result.ccc = ccc
        elif dso is not None and dio is not None:
            ccc = dso + dio
            result.ccc = ccc
        else:
            ccc = None

        # --- Cash / Revenue ---
        cash = data.cash
        if cash is not None and revenue and revenue > 0:
            result.cash_to_revenue = safe_divide(cash, revenue)

        # --- OCF / Revenue ---
        ocf = data.operating_cash_flow
        if ocf is not None and revenue and revenue > 0:
            result.ocf_to_revenue = safe_divide(ocf, revenue)

        # --- OCF / EBITDA ---
        ebitda = data.ebitda
        if ocf is not None and ebitda and ebitda > 0:
            result.ocf_to_ebitda = safe_divide(ocf, ebitda)

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # CCC scoring: lower is better
        if ccc is not None:
            if ccc < 30:
                score += 2.0
            elif ccc < 60:
                score += 1.0
            elif ccc < 90:
                score += 0.0
            elif ccc < 120:
                score -= 1.0
            else:
                score -= 2.0

        # OCF/Revenue
        ocf_rev = result.ocf_to_revenue
        if ocf_rev is not None:
            if ocf_rev >= 0.20:
                score += 1.0
            elif ocf_rev >= 0.10:
                score += 0.5
            elif ocf_rev < 0:
                score -= 1.0

        # OCF/EBITDA quality
        ocf_ebitda = result.ocf_to_ebitda
        if ocf_ebitda is not None:
            if ocf_ebitda >= 0.80:
                score += 0.5
            elif ocf_ebitda < 0.50:
                score -= 0.5

        # DSO efficiency
        if dso is not None:
            if dso < 30:
                score += 0.5
            elif dso > 90:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cash_conversion_score = score

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Fair"
        else:
            grade = "Poor"
        result.cash_conversion_grade = grade

        ccc_str = f"{ccc:.0f} days" if ccc is not None else "N/A"
        result.summary = (
            f"Cash Conversion: CCC of {ccc_str}. "
            f"Grade: {grade}."
        )

        return result

    def beneish_m_score_analysis(self, data: FinancialData) -> BeneishMScoreResult:
        """Phase 43: Beneish M-Score (Earnings Manipulation Detection).

        Single-period simplified M-Score using available data.
        Without prior-period data, indices default to 1.0 (neutral).
        TATA = (NI - OCF) / TA is the most reliable single-period indicator.

        M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
            + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

        M > -1.78: likely manipulator
        M <= -1.78: unlikely manipulator
        """
        result = BeneishMScoreResult()

        revenue = data.revenue
        ta = data.total_assets
        ni = data.net_income
        ocf = data.operating_cash_flow
        gp = data.gross_profit
        dep = data.depreciation
        tl = data.total_liabilities

        if not revenue or revenue <= 0 or not ta or ta <= 0:
            result.summary = "Insufficient data for Beneish M-Score."
            return result

        # --- DSRI: Days Sales in Receivables Index ---
        # Single-period proxy: AR/Revenue ratio vs benchmark (0.15)
        ar = data.accounts_receivable
        if ar is not None:
            ar_ratio = safe_divide(ar, revenue, default=0.0)
            dsri = safe_divide(ar_ratio, 0.15, default=1.0) if ar_ratio > 0 else 1.0
        else:
            dsri = 1.0
        result.dsri = dsri

        # --- GMI: Gross Margin Index ---
        # Single-period proxy: (1 - GM) where GM = GP/Revenue
        if gp is not None:
            gm = safe_divide(gp, revenue, default=0.0)
            gmi = safe_divide(1.0 - gm, 0.60, default=1.0) if gm < 1.0 else 1.0
        else:
            gmi = 1.0
        result.gmi = gmi

        # --- AQI: Asset Quality Index ---
        # Proxy: (1 - (CA + PP&E)/TA); without PP&E use CA/TA
        ca = data.current_assets
        if ca is not None:
            hard_assets_ratio = safe_divide(ca, ta, default=0.0)
            aqi = 1.0 + (1.0 - hard_assets_ratio) * 0.5
        else:
            aqi = 1.0
        result.aqi = aqi

        # --- SGI: Sales Growth Index ---
        # Without prior revenue, default to 1.0
        sgi = 1.0
        result.sgi = sgi

        # --- DEPI: Depreciation Index ---
        if dep is not None and ta > 0:
            dep_rate = safe_divide(dep, ta, default=0.0)
            depi = safe_divide(0.05, dep_rate, default=1.0) if dep_rate > 0 else 1.0
        else:
            depi = 1.0
        result.depi = depi

        # --- SGAI: SGA Expense Index ---
        opex = data.operating_expenses
        if opex is not None and revenue > 0:
            sga_ratio = safe_divide(opex, revenue, default=0.0)
            sgai = safe_divide(sga_ratio, 0.20, default=1.0) if sga_ratio > 0 else 1.0
        else:
            sgai = 1.0
        result.sgai = sgai

        # --- LVGI: Leverage Index ---
        if tl is not None and ta > 0:
            leverage = safe_divide(tl, ta, default=0.0)
            lvgi = safe_divide(leverage, 0.40, default=1.0) if leverage > 0 else 1.0
        else:
            lvgi = 1.0
        result.lvgi = lvgi

        # --- TATA: Total Accruals to Total Assets ---
        if ni is not None and ocf is not None and ta > 0:
            tata = safe_divide(ni - ocf, ta, default=0.0)
        else:
            tata = 0.0
        result.tata = tata

        # --- Compute M-Score ---
        m = (-4.84
             + 0.920 * dsri
             + 0.528 * gmi
             + 0.404 * aqi
             + 0.892 * sgi
             + 0.115 * depi
             - 0.172 * sgai
             + 4.679 * tata
             - 0.327 * lvgi)
        result.m_score = m

        # --- Scoring (invert: lower M = better, score 0-10) ---
        # M < -2.22: very unlikely => 9-10
        # M < -1.78: unlikely => 7-8
        # M >= -1.78: likely => 2-4
        # M >= -1.0: highly likely => 0-2
        if m < -2.50:
            score = 10.0
        elif m < -2.22:
            score = 9.0
        elif m < -1.78:
            score = 7.0
        elif m < -1.50:
            score = 5.0
        elif m < -1.0:
            score = 3.0
        else:
            score = 1.0

        # Adjust based on TATA
        if tata <= -0.05:
            score += 0.5  # Negative accruals (good)
        elif tata > 0.05:
            score -= 0.5  # High accruals (bad)

        score = max(0.0, min(10.0, score))
        result.manipulation_score = score

        if score >= 8.0:
            grade = "Unlikely"
        elif score >= 6.0:
            grade = "Possible"
        elif score >= 4.0:
            grade = "Likely"
        else:
            grade = "Highly Likely"
        result.manipulation_grade = grade

        result.summary = (
            f"Beneish M-Score: {m:.2f} "
            f"(threshold -1.78). "
            f"Manipulation: {grade}."
        )

        return result

    def profit_retention_power_analysis(self, data: FinancialData) -> ProfitRetentionPowerResult:
        """Phase 356: Profit Retention Power Analysis."""
        result = self._scored_analysis(
            data=data,
            result_class=ProfitRetentionPowerResult,
            ratio_defs=[
                ("prp_ratio", "retained_earnings", "total_assets"),
                ("re_to_equity", "retained_earnings", "total_equity"),
                ("re_to_revenue", "retained_earnings", "revenue"),
            ],
            score_field="prp_score",
            grade_field="prp_grade",
            primary="prp_ratio",
            higher_is_better=True,
            thresholds=[(0.40, 10.0), (0.30, 8.5), (0.20, 7.0), (0.15, 5.5), (0.10, 4.0), (0.05, 2.5)],
            adjustments=[
                (lambda r, d: (
                    d.net_income is not None and d.net_income > 0 and
                    ((d.net_income - (d.dividends_paid or 0)) / d.net_income) >= 0.60
                ), 0.5),
                (lambda r, d: d.retained_earnings is not None and d.retained_earnings > 0 and d.total_assets is not None and d.total_assets > 0, 0.5),
            ],
            derived=[
                ("retention_rate", lambda r, d: (d.net_income - (d.dividends_paid or 0)) / d.net_income if d.net_income is not None and d.net_income > 0 else None),
                ("re_growth_capacity", lambda r, d: safe_divide(d.net_income - (d.dividends_paid or 0), d.retained_earnings) if d.net_income is not None and d.net_income > 0 else None),
                ("prp_spread", lambda r, d: r["prp_ratio"] - 0.20 if r.get("prp_ratio") is not None else None),
            ],
            label="Profit Retention Power",
        )
        return result

    def earnings_to_debt_analysis(self, data: FinancialData) -> EarningsToDebtResult:
        """Phase 353: Earnings To Debt Analysis."""
        result = self._scored_analysis(
            data=data,
            result_class=EarningsToDebtResult,
            ratio_defs=[
                ("etd_ratio", "net_income", "total_debt"),
                ("ni_to_interest", "net_income", "interest_expense"),
                ("ni_to_liabilities", "net_income", "total_liabilities"),
            ],
            score_field="etd_score",
            grade_field="etd_grade",
            primary="etd_ratio",
            higher_is_better=True,
            thresholds=[(0.40, 10.0), (0.30, 8.5), (0.20, 7.0), (0.15, 5.5), (0.10, 4.0), (0.05, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ni_to_interest") is not None and r["ni_to_interest"] >= 3.0, 0.5),
                (lambda r, d: d.net_income is not None and d.net_income > 0 and d.total_debt is not None and d.total_debt > 0, 0.5),
            ],
            derived=[
                ("earnings_yield_on_debt", lambda r, d: r.get("etd_ratio")),
                ("debt_years_from_earnings", lambda r, d: safe_divide(d.total_debt, d.net_income)),
                ("etd_spread", lambda r, d: r["etd_ratio"] - 0.20 if r.get("etd_ratio") is not None else None),
            ],
            label="Earnings To Debt",
        )
        return result

    def revenue_growth_analysis(self, data: FinancialData) -> RevenueGrowthResult:
        """Phase 350: Revenue Growth Capacity Analysis."""
        result = self._scored_analysis(
            data=data,
            result_class=RevenueGrowthResult,
            ratio_defs=[
                ("roe", "net_income", "total_equity"),
                ("revenue_per_asset", "revenue", "total_assets"),
            ],
            score_field="rg_score",
            grade_field="rg_grade",
            primary="sustainable_growth",
            higher_is_better=True,
            thresholds=[(0.15, 10.0), (0.12, 8.5), (0.10, 7.0), (0.07, 5.5), (0.05, 4.0), (0.02, 2.5)],
            adjustments=[
                (lambda r, d: r.get("roe") is not None and r["roe"] >= 0.12, 0.5),
                (lambda r, d: d.net_income is not None and d.net_income > 0 and d.total_equity is not None and d.total_equity > 0, 0.5),
            ],
            derived=[
                ("plowback", lambda r, d: (d.net_income - (d.dividends_paid or 0)) / d.net_income if d.net_income is not None and d.net_income != 0 else None),
                ("sustainable_growth", lambda r, d: r["roe"] * r["plowback"] if r.get("roe") is not None and r.get("plowback") is not None else None),
                ("rg_capacity", lambda r, d: r.get("sustainable_growth")),
                ("rg_spread", lambda r, d: r["sustainable_growth"] - r["roe"] if r.get("sustainable_growth") is not None and r.get("roe") is not None else None),
            ],
            label="Revenue Growth",
        )
        return result

    def operating_margin_analysis(self, data: FinancialData) -> OperatingMarginResult:
        """Phase 349: Operating Margin Analysis."""
        result = self._scored_analysis(
            data=data,
            result_class=OperatingMarginResult,
            ratio_defs=[
                ("oi_to_revenue", "operating_income", "revenue"),
                ("ebit_margin", "ebit", "revenue"),
                ("ebitda_margin", "ebitda", "revenue"),
            ],
            score_field="opm_score",
            grade_field="opm_grade",
            primary="oi_to_revenue",
            higher_is_better=True,
            thresholds=[(0.25, 10.0), (0.20, 8.5), (0.15, 7.0), (0.10, 5.5), (0.05, 4.0), (0.02, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ebitda_margin") is not None and r["ebitda_margin"] >= 0.20, 0.5),
                (lambda r, d: d.operating_income is not None and d.operating_income > 0 and d.revenue is not None and d.revenue > 0, 0.5),
            ],
            derived=[
                ("operating_margin", lambda r, d: r.get("oi_to_revenue")),
                ("opm_spread", lambda r, d: r["ebitda_margin"] - r["oi_to_revenue"] if r.get("oi_to_revenue") is not None and r.get("ebitda_margin") is not None else None),
            ],
            label="Operating Margin",
        )
        return result

    def debt_to_equity_analysis(self, data: FinancialData) -> DebtToEquityResult:
        """Phase 348: Debt To Equity Analysis."""
        result = self._scored_analysis(
            data=data,
            result_class=DebtToEquityResult,
            ratio_defs=[
                ("td_to_te", "total_debt", "total_equity"),
                ("debt_to_assets", "total_debt", "total_assets"),
                ("equity_multiplier", "total_assets", "total_equity"),
            ],
            score_field="dte_score",
            grade_field="dte_grade",
            primary="td_to_te",
            higher_is_better=False,
            thresholds=[(0.30, 10.0), (0.50, 8.5), (0.80, 7.0), (1.00, 5.5), (1.50, 4.0), (2.00, 2.5)],
            adjustments=[
                (lambda r, d: r.get("debt_to_assets") is not None and r["debt_to_assets"] <= 0.50, 0.5),
                (lambda r, d: d.total_debt is not None and d.total_debt > 0 and d.total_equity is not None and d.total_equity > 0, 0.5),
            ],
            derived=[
                ("dte_ratio", lambda r, d: r.get("td_to_te")),
                ("lt_debt_to_equity", lambda r, d: d.total_debt / d.total_equity if d.total_debt is not None and d.total_equity is not None and d.total_equity != 0 else None),
                ("dte_spread", lambda r, d: r["td_to_te"] - r["debt_to_assets"] if r.get("td_to_te") is not None and r.get("debt_to_assets") is not None else None),
            ],
            label="Debt to Equity",
        )
        return result

    def cash_flow_to_debt_analysis(self, data: FinancialData) -> CashFlowToDebtResult:
        """Phase 347: Cash Flow To Debt Analysis."""
        result = self._scored_analysis(
            data=data,
            result_class=CashFlowToDebtResult,
            ratio_defs=[
                ("ocf_to_td", "operating_cash_flow", "total_debt"),
                ("ocf_to_interest", "operating_cash_flow", "interest_expense"),
            ],
            score_field="cfd_score",
            grade_field="cfd_grade",
            primary="ocf_to_td",
            higher_is_better=True,
            thresholds=[(0.50, 10.0), (0.40, 8.5), (0.30, 7.0), (0.20, 5.5), (0.10, 4.0), (0.05, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ocf_to_interest") is not None and r["ocf_to_interest"] >= 3.0, 0.5),
                (lambda r, d: d.operating_cash_flow is not None and d.operating_cash_flow > 0 and d.total_debt is not None and d.total_debt > 0, 0.5),
            ],
            derived=[
                ("cf_to_debt", lambda r, d: r.get("ocf_to_td")),
                ("fcf_to_td", lambda r, d: (d.operating_cash_flow - (d.capex or 0)) / d.total_debt if d.operating_cash_flow is not None and d.capex is not None and d.total_debt is not None and d.total_debt != 0 else None),
                ("debt_payback_years", lambda r, d: d.total_debt / d.operating_cash_flow if d.total_debt is not None and d.operating_cash_flow is not None and d.operating_cash_flow > 0 else None),
                ("cf_debt_spread", lambda r, d: r["ocf_to_td"] - r["fcf_to_td"] if r.get("ocf_to_td") is not None and r.get("fcf_to_td") is not None else None),
            ],
            label="Cash Flow to Debt",
        )
        return result

    def net_worth_growth_analysis(self, data: FinancialData) -> NetWorthGrowthResult:
        """Phase 346: Net Worth Growth Analysis."""
        result = self._scored_analysis(
            data=data,
            result_class=NetWorthGrowthResult,
            ratio_defs=[
                ("re_to_equity", "retained_earnings", "total_equity"),
                ("equity_to_assets", "total_equity", "total_assets"),
                ("ni_to_equity", "net_income", "total_equity"),
            ],
            score_field="nwg_score",
            grade_field="nwg_grade",
            primary="re_to_equity",
            higher_is_better=True,
            thresholds=[(0.70, 10.0), (0.60, 8.5), (0.50, 7.0), (0.40, 5.5), (0.25, 4.0), (0.10, 2.5)],
            adjustments=[
                (lambda r, d: r.get("equity_to_assets") is not None and r["equity_to_assets"] >= 0.40, 0.5),
                (lambda r, d: d.retained_earnings is not None and d.retained_earnings > 0 and d.total_equity is not None and d.total_equity > 0, 0.5),
            ],
            derived=[
                ("plowback_rate", lambda r, d: (d.net_income - (d.dividends_paid or 0)) / d.net_income if d.net_income is not None and d.net_income != 0 else None),
                ("nw_growth_ratio", lambda r, d: r.get("re_to_equity")),
                ("nw_spread", lambda r, d: r["re_to_equity"] - r["equity_to_assets"] if r.get("re_to_equity") is not None and r.get("equity_to_assets") is not None else None),
            ],
            label="Net Worth Growth",
        )
        return result

    def asset_lightness_analysis(self, data: FinancialData) -> AssetLightnessResult:
        """Phase 341: Asset Lightness Analysis.

        Lightness Ratio = CA / TA. Higher means more current/liquid assets
        vs fixed assets, indicating an asset-light business model.
        Complemented by Revenue/TA (asset turnover) for efficiency.
        """
        result = AssetLightnessResult()

        ca = data.current_assets
        ta = data.total_assets

        # Primary: CA / TA
        lightness = safe_divide(ca, ta)
        result.lightness_ratio = lightness
        result.ca_to_ta = lightness

        # Revenue to assets (asset turnover)
        result.revenue_to_assets = safe_divide(data.revenue, ta)

        # Fixed asset ratio = (TA - CA) / TA = 1 - lightness
        if lightness is not None:
            result.fixed_asset_ratio = 1.0 - lightness
        else:
            result.fixed_asset_ratio = None

        # Intangible intensity proxy: (TA - CA - Inventory) / TA
        if ca is not None and ta is not None and ta > 0:
            inv = data.inventory or 0
            tangible_current = ca - inv
            result.intangible_intensity = safe_divide(ta - tangible_current, ta)
        else:
            result.intangible_intensity = None

        # Lightness spread: lightness - 0.50 benchmark
        if lightness is not None:
            result.lightness_spread = lightness - 0.50

        # Scoring
        if lightness is None:
            result.alt_score = 0.0
            result.alt_grade = ""
            result.summary = "Asset Lightness: Insufficient data."
            return result

        # CA/TA ranges: higher = more asset-light
        if lightness >= 0.70:
            score = 10.0
        elif lightness >= 0.60:
            score = 8.5
        elif lightness >= 0.50:
            score = 7.0
        elif lightness >= 0.40:
            score = 5.5
        elif lightness >= 0.30:
            score = 4.0
        elif lightness >= 0.15:
            score = 2.5
        else:
            score = 1.0

        # Adj: revenue/TA >= 0.50 (+0.5) — efficient asset use
        rat = result.revenue_to_assets
        if rat is not None and rat >= 0.50:
            score += 0.5

        # Adj: both > 0 (+0.5)
        if lightness > 0 and ta is not None and ta > 0:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.alt_score = score

        if score >= 8:
            result.alt_grade = "Excellent"
        elif score >= 6:
            result.alt_grade = "Good"
        elif score >= 4:
            result.alt_grade = "Adequate"
        else:
            result.alt_grade = "Weak"

        rat_str = f"{result.revenue_to_assets:.4f}" if result.revenue_to_assets is not None else "N/A"
        result.summary = (
            f"Asset Lightness: CA/TA={lightness:.4f}, "
            f"Revenue/TA={rat_str}, "
            f"Score={score:.1f}/10 ({result.alt_grade})."
        )
        return result

    def internal_growth_rate_analysis(self, data: FinancialData) -> InternalGrowthRateResult:
        """Phase 337: Internal Growth Rate Analysis.

        Primary: IGR = (ROA * b) / (1 - ROA * b), where b = retention ratio.
        Higher IGR means more self-funded growth capacity.
        """
        result = InternalGrowthRateResult()
        ni = getattr(data, 'net_income', None) or 0.0
        ta = getattr(data, 'total_assets', None) or 0.0
        div = getattr(data, 'dividends_paid', None) or 0.0
        te = getattr(data, 'total_equity', None) or 0.0

        if not ni and not ta:
            result.summary = "Internal Growth Rate: Insufficient data."
            return result

        roa = safe_divide(ni, ta)
        result.roa = roa
        b = safe_divide(ni - div, ni) if ni else None
        result.retention_ratio = b

        if roa is not None and b is not None:
            roa_b = roa * b
            result.roa_times_b = roa_b
            result.igr = safe_divide(roa_b, 1.0 - roa_b)
        else:
            result.roa_times_b = None
            result.igr = None

        # Sustainable growth = ROE * b
        roe = safe_divide(ni, te) if te else None
        result.sustainable_growth = (roe * b) if roe is not None and b is not None else None
        result.growth_capacity = result.igr

        igr = result.igr
        if igr is None:
            result.summary = "Internal Growth Rate: Cannot compute."
            return result

        # Scoring: higher IGR is better (as percentage)
        igr_pct = igr * 100.0  # convert to percentage
        if igr_pct >= 15.0:
            base = 10.0
        elif igr_pct >= 12.0:
            base = 8.5
        elif igr_pct >= 8.0:
            base = 7.0
        elif igr_pct >= 5.0:
            base = 5.5
        elif igr_pct >= 3.0:
            base = 4.0
        elif igr_pct >= 0.0:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if b is not None and b >= 0.70:
            adj += 0.5
        if ni > 0 and ta > 0:
            adj += 0.5

        score = min(base + adj, 10.0)
        result.igr_score = score
        if score >= 8:
            result.igr_grade = "Excellent"
        elif score >= 6:
            result.igr_grade = "Good"
        elif score >= 4:
            result.igr_grade = "Adequate"
        else:
            result.igr_grade = "Weak"

        result.summary = (
            f"Internal Growth Rate: IGR={f'{igr_pct:.2f}%' if igr is not None else 'N/A'}, "
            f"ROA={f'{roa:.4f}' if roa is not None else 'N/A'}, "
            f"Retention={f'{b:.4f}' if b is not None else 'N/A'}, "
            f"Score={result.igr_score:.1f}/10 ({result.igr_grade})."
        )
        return result

    def operating_expense_ratio_analysis(self, data: FinancialData) -> OperatingExpenseRatioResult:
        """Phase 330: Operating Expense Ratio Analysis."""
        return self._scored_analysis(
            data=data, result_class=OperatingExpenseRatioResult,
            ratio_defs=[
                ("opex_ratio", "operating_expenses", "revenue"),
                ("opex_to_gross_profit", "operating_expenses", "gross_profit"),
                ("opex_to_ebitda", "operating_expenses", "ebitda"),
            ],
            score_field="oer_score", grade_field="oer_grade",
            primary="opex_ratio", higher_is_better=False,
            thresholds=[(0.10, 10.0), (0.15, 8.5), (0.20, 7.0), (0.25, 5.5), (0.30, 4.0), (0.40, 2.5)],
            adjustments=[
                (lambda r, d: r.get("opex_to_gross_profit") is not None and r["opex_to_gross_profit"] < 1.0, 0.5),
                (lambda r, d: d.operating_expenses is not None and d.operating_expenses > 0 and d.revenue is not None and d.revenue > 0, 0.5),
            ],
            derived=[
                ("opex_per_revenue", lambda r, d: r.get("opex_ratio")),
                ("opex_coverage", lambda r, d: safe_divide(d.revenue, d.operating_expenses)),
                ("efficiency_gap", lambda r, d: (r["opex_ratio"] - 0.20) if r.get("opex_ratio") is not None else None),
            ],
            label="Operating Expense Ratio",
        )

    def noncurrent_asset_ratio_analysis(self, data: FinancialData) -> NoncurrentAssetRatioResult:
        """Phase 327: Noncurrent Asset Ratio Analysis.

        Primary: (Total Assets - Current Assets) / Total Assets.
        Measures long-term asset concentration. Moderate 0.40-0.65 is balanced.
        Very high (>0.80) means illiquid, very low (<0.20) means mostly current assets.
        """
        result = NoncurrentAssetRatioResult()
        ta = getattr(data, 'total_assets', None) or 0.0
        ca = getattr(data, 'current_assets', None) or 0.0
        te = getattr(data, 'total_equity', None) or 0.0
        td = getattr(data, 'total_debt', None) or 0.0

        if not ta:
            result.summary = "Noncurrent Asset Ratio: Insufficient data (need total_assets)."
            return result

        nca = ta - ca
        ratio = safe_divide(nca, ta)
        result.nca_ratio = ratio

        result.current_asset_ratio = safe_divide(ca, ta)
        result.nca_to_equity = safe_divide(nca, te) if te else None
        result.nca_to_debt = safe_divide(nca, td) if td else None
        result.asset_structure_spread = (ratio - 0.50) if ratio is not None else None
        result.liquidity_complement = safe_divide(ca, ta)

        if ratio is None:
            result.summary = "Noncurrent Asset Ratio: Cannot compute."
            return result

        # Scoring: moderate ratio is best (inverted U around 0.40-0.65)
        if 0.40 <= ratio <= 0.65:
            base = 10.0
        elif (0.30 <= ratio < 0.40) or (0.65 < ratio <= 0.75):
            base = 8.5
        elif (0.20 <= ratio < 0.30) or (0.75 < ratio <= 0.85):
            base = 7.0
        elif (0.10 <= ratio < 0.20) or (0.85 < ratio <= 0.90):
            base = 5.5
        elif ratio < 0.10 or ratio > 0.90:
            base = 4.0
        else:
            base = 2.5

        adj = 0.0
        nca_eq = result.nca_to_equity
        if nca_eq is not None and nca_eq <= 1.0:
            adj += 0.5
        if ta > 0 and ca > 0:
            adj += 0.5

        score = min(base + adj, 10.0)
        result.nar_score = score
        if score >= 8:
            result.nar_grade = "Excellent"
        elif score >= 6:
            result.nar_grade = "Good"
        elif score >= 4:
            result.nar_grade = "Adequate"
        else:
            result.nar_grade = "Weak"

        result.summary = (
            f"Noncurrent Asset Ratio: NCA Ratio={f'{ratio:.4f}' if ratio is not None else 'N/A'}, "
            f"NCA/Equity={f'{nca_eq:.4f}' if nca_eq is not None else 'N/A'}, "
            f"Score={result.nar_score:.1f}/10 ({result.nar_grade})."
        )
        return result

    def payout_resilience_analysis(self, data: FinancialData) -> PayoutResilienceResult:
        """Phase 317: Payout Resilience Analysis.

        Measures sustainability of dividend payments from earnings and cash flow.
        Primary metric: Div/NI payout ratio (moderate 0.20-0.50 is ideal).
        """
        result = PayoutResilienceResult()

        div = data.dividends_paid
        ni = data.net_income
        ocf = data.operating_cash_flow
        revenue = data.revenue
        ebitda = data.ebitda

        if not div or div <= 0:
            return result
        if not ni or ni <= 0:
            return result

        # Ratios
        result.div_to_ni = safe_divide(div, ni)
        result.div_to_ocf = safe_divide(div, ocf)
        result.div_to_revenue = safe_divide(div, revenue)
        result.div_to_ebitda = safe_divide(div, ebitda)

        # Primary: Div/NI
        primary = result.div_to_ni
        if primary is None:
            return result

        result.payout_ratio = primary

        # Resilience buffer = 1 - payout ratio (how much earnings retained)
        result.resilience_buffer = 1.0 - primary if primary <= 1.0 else 0.0

        # Scoring: moderate payout [0.20, 0.50] is ideal
        if 0.20 <= primary <= 0.50:
            score = 10.0
        elif (0.10 <= primary < 0.20) or (0.50 < primary <= 0.60):
            score = 8.5
        elif (0.05 <= primary < 0.10) or (0.60 < primary <= 0.70):
            score = 7.0
        elif 0.70 < primary <= 0.80:
            score = 5.5
        elif 0.80 < primary <= 0.90:
            score = 4.0
        elif 0.90 < primary <= 1.0:
            score = 2.5
        elif primary < 0.05:
            score = 5.5
        else:
            score = 1.0

        # Adjustments
        if result.div_to_ocf is not None and result.div_to_ocf <= 0.40:
            score += 0.5
        if div > 0 and ni > 0:
            score += 0.5

        result.prs_score = min(score, 10.0)

        # Grade
        if result.prs_score >= 8:
            result.prs_grade = "Excellent"
        elif result.prs_score >= 6:
            result.prs_grade = "Good"
        elif result.prs_score >= 4:
            result.prs_grade = "Adequate"
        else:
            result.prs_grade = "Weak"

        primary_str = f"{primary:.4f}" if primary is not None else "N/A"
        ocf_str = f"{result.div_to_ocf:.4f}" if result.div_to_ocf is not None else "N/A"
        result.summary = (
            f"Payout Resilience Analysis: Div/NI={primary_str}, "
            f"Div/OCF={ocf_str}, "
            f"Score={result.prs_score:.1f}/10 ({result.prs_grade})"
        )

        return result

    def debt_burden_index_analysis(self, data: FinancialData) -> DebtBurdenIndexResult:
        """Phase 314: Debt Burden Index Analysis.

        Measures total debt burden relative to earnings and assets.
        Primary metric: Debt/EBITDA (lower = less leveraged = better).
        """
        result = DebtBurdenIndexResult()

        debt = data.total_debt
        ebitda = data.ebitda
        assets = data.total_assets
        equity = data.total_equity
        revenue = data.revenue

        if not debt or debt <= 0:
            return result

        # Ratios
        result.debt_to_ebitda = safe_divide(debt, ebitda)
        result.debt_to_assets = safe_divide(debt, assets)
        result.debt_to_equity = safe_divide(debt, equity)
        result.debt_to_revenue = safe_divide(debt, revenue)

        # Primary: Debt/EBITDA
        primary = result.debt_to_ebitda
        if primary is None:
            # Fallback to Debt/Revenue
            primary = result.debt_to_revenue
            if primary is None:
                return result

        result.debt_ratio = primary

        # Burden intensity = Debt/Assets
        result.burden_intensity = result.debt_to_assets

        # Scoring: lower Debt/EBITDA is better
        if primary <= 1.0:
            score = 10.0
        elif primary <= 2.0:
            score = 8.5
        elif primary <= 3.0:
            score = 7.0
        elif primary <= 4.0:
            score = 5.5
        elif primary <= 5.0:
            score = 4.0
        elif primary <= 6.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.debt_to_assets is not None and result.debt_to_assets <= 0.40:
            score += 0.5
        if debt > 0 and ebitda is not None and ebitda > 0:
            score += 0.5

        result.dbi_score = min(score, 10.0)

        # Grade
        if result.dbi_score >= 8:
            result.dbi_grade = "Excellent"
        elif result.dbi_score >= 6:
            result.dbi_grade = "Good"
        elif result.dbi_score >= 4:
            result.dbi_grade = "Adequate"
        else:
            result.dbi_grade = "Weak"

        primary_str = f"{primary:.4f}" if primary is not None else "N/A"
        da_str = f"{result.debt_to_assets:.4f}" if result.debt_to_assets is not None else "N/A"
        result.summary = (
            f"Debt Burden Index Analysis: Debt/EBITDA={primary_str}, "
            f"Debt/Assets={da_str}, "
            f"Score={result.dbi_score:.1f}/10 ({result.dbi_grade})"
        )

        return result

    def inventory_coverage_analysis(self, data: FinancialData) -> InventoryCoverageResult:
        """Phase 309: Inventory Coverage Analysis.

        Evaluates how well inventory levels support revenue operations.
        Primary metric: Inventory/COGS (lower = faster turnover, better coverage).
        """
        result = InventoryCoverageResult()

        inv = data.inventory
        cogs = data.cogs
        rev = data.revenue
        ta = data.total_assets
        ca = data.current_assets

        if inv is None or (cogs is None and rev is None):
            return result

        if inv == 0 and cogs in (None, 0) and rev in (None, 0):
            return result

        # Ratios
        result.inventory_to_cogs = safe_divide(inv, cogs)
        result.inventory_to_revenue = safe_divide(inv, rev)
        result.inventory_to_assets = safe_divide(inv, ta)
        result.inventory_to_current_assets = safe_divide(inv, ca)

        # Inventory days = Inv / (COGS / 365)
        if cogs and cogs > 0:
            result.inventory_days = (inv / cogs) * 365
        # Buffer = Inv / (Rev / 365)
        if rev and rev > 0:
            result.inventory_buffer = (inv / rev) * 365

        # Scoring: based on inventory_to_cogs (lower is generally better — efficient)
        primary = result.inventory_to_cogs
        if primary is None:
            primary = result.inventory_to_revenue

        if primary is None:
            return result

        # Scoring — moderate inventory relative to COGS is ideal
        # Very low (<0.05) means possible stockouts; moderate (0.08-0.20) is ideal; high (>0.40) is excess
        if 0.08 <= primary <= 0.20:
            base = 10.0
        elif 0.05 <= primary < 0.08 or 0.20 < primary <= 0.25:
            base = 8.5
        elif 0.03 <= primary < 0.05 or 0.25 < primary <= 0.30:
            base = 7.0
        elif 0.02 <= primary < 0.03 or 0.30 < primary <= 0.40:
            base = 5.5
        elif primary < 0.02 or 0.40 < primary <= 0.50:
            base = 4.0
        elif 0.50 < primary <= 0.60:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Bonus: inventory days 30-90 (healthy range)
        if result.inventory_days is not None and 30 <= result.inventory_days <= 90:
            adj += 0.5
        # Bonus: inventory > 0 and cogs > 0
        if inv and inv > 0 and cogs and cogs > 0:
            adj += 0.5

        result.icv_score = min(10.0, max(0.0, base + adj))

        if result.icv_score >= 8:
            result.icv_grade = "Excellent"
        elif result.icv_score >= 6:
            result.icv_grade = "Good"
        elif result.icv_score >= 4:
            result.icv_grade = "Adequate"
        else:
            result.icv_grade = "Weak"

        primary_str = f"{primary:.4f}" if primary is not None else "N/A"
        days_str = f"{result.inventory_days:.1f}" if result.inventory_days is not None else "N/A"
        result.summary = (
            f"Inventory Coverage Analysis: Inv/COGS={primary_str}, "
            f"Days={days_str}, "
            f"Score={result.icv_score:.1f}/10 ({result.icv_grade})"
        )

        return result

    def capex_to_revenue_analysis(self, data: FinancialData) -> CapexToRevenueResult:
        """Phase 307: CapEx to Revenue Analysis.

        Measures capital expenditure relative to revenue — investment intensity.
        Moderate CapEx/Revenue indicates balanced investment without over-spending.
        """
        result = CapexToRevenueResult()

        capex = data.capex
        rev = data.revenue
        ocf = data.operating_cash_flow
        ebitda = data.ebitda
        ta = data.total_assets

        if not capex and not rev:
            return result

        # Primary: CapEx/Revenue
        cap_rev = safe_divide(capex, rev)
        result.capex_to_revenue = cap_rev

        # Secondary metrics
        result.capex_to_ocf = safe_divide(capex, ocf)
        result.capex_to_ebitda = safe_divide(capex, ebitda)
        result.capex_to_assets = safe_divide(capex, ta)

        # Investment intensity
        result.investment_intensity = cap_rev

        # CapEx yield: Revenue / CapEx
        result.capex_yield = safe_divide(rev, capex)

        # Scoring: moderate CapEx is best (5-15% of revenue)
        # Too low = underinvesting, too high = over-spending
        if cap_rev is not None:
            if 0.05 <= cap_rev <= 0.15:
                base = 10.0
            elif 0.03 <= cap_rev < 0.05 or 0.15 < cap_rev <= 0.20:
                base = 8.5
            elif 0.02 <= cap_rev < 0.03 or 0.20 < cap_rev <= 0.25:
                base = 7.0
            elif 0.01 <= cap_rev < 0.02 or 0.25 < cap_rev <= 0.30:
                base = 5.5
            elif cap_rev < 0.01 or 0.30 < cap_rev <= 0.40:
                base = 4.0
            elif 0.40 < cap_rev <= 0.50:
                base = 2.5
            else:
                base = 1.0

            adj = 0.0
            # Bonus: OCF covers CapEx (OCF > CapEx)
            if ocf is not None and capex is not None and (ocf or 0) > (capex or 0):
                adj += 0.5
            # Bonus: CapEx > 0 and Rev > 0
            if (capex or 0) > 0 and (rev or 0) > 0:
                adj += 0.5

            result.ctr_score = min(10.0, base + adj)
        else:
            result.ctr_score = 0.0

        # Grade
        if result.ctr_score >= 8:
            result.ctr_grade = "Excellent"
        elif result.ctr_score >= 6:
            result.ctr_grade = "Good"
        elif result.ctr_score >= 4:
            result.ctr_grade = "Adequate"
        else:
            result.ctr_grade = "Weak"

        result.summary = (
            f"CapEx to Revenue: CapEx/Rev={cap_rev:.2f}, "
            f"Score={result.ctr_score:.1f}/10 ({result.ctr_grade})"
            if cap_rev is not None
            else "CapEx to Revenue: Insufficient data"
        )

        return result

    def inventory_holding_cost_analysis(self, data: FinancialData) -> InventoryHoldingCostResult:
        """Phase 294: Inventory Holding Cost Analysis."""
        return self._scored_analysis(
            data=data, result_class=InventoryHoldingCostResult,
            ratio_defs=[
                ("inventory_to_revenue", "inventory", "revenue"),
                ("inventory_to_current_assets", "inventory", "current_assets"),
                ("inventory_to_total_assets", "inventory", "total_assets"),
            ],
            score_field="ihc_score", grade_field="ihc_grade",
            primary="inventory_to_revenue", higher_is_better=False,
            thresholds=[(0.05, 10.0), (0.08, 8.5), (0.12, 7.0), (0.18, 5.5), (0.25, 4.0), (0.35, 2.5)],
            adjustments=[
                (lambda r, d: r.get("inventory_to_current_assets") is not None and r["inventory_to_current_assets"] <= 0.30, 0.5),
                (lambda r, d: d.inventory is not None and d.inventory > 0 and d.revenue is not None and d.revenue > 0, 0.5),
            ],
            derived=[
                ("inventory_days", lambda r, d: safe_divide(d.inventory, d.cogs) * 365 if d.cogs and d.cogs > 0 and d.inventory is not None else None),
                ("inventory_carrying_cost", lambda r, d: r.get("inventory_to_revenue")),
                ("inventory_intensity", lambda r, d: r.get("inventory_to_revenue")),
            ],
            label="Inventory Holding Cost",
        )

    def funding_mix_balance_analysis(self, data: FinancialData) -> FundingMixBalanceResult:
        """Phase 293: Funding Mix Balance Analysis."""
        return self._scored_analysis(
            data=data, result_class=FundingMixBalanceResult,
            ratio_defs=[
                ("debt_to_equity", "total_debt", "total_equity"),
            ],
            score_field="fmb_score", grade_field="fmb_grade",
            primary="equity_to_total_capital", higher_is_better=True,
            thresholds=[(0.80, 10.0), (0.70, 8.5), (0.60, 7.0), (0.50, 5.5), (0.40, 4.0), (0.30, 2.5)],
            adjustments=[
                (lambda r, d: r.get("debt_to_equity") is not None and r["debt_to_equity"] <= 0.50, 0.5),
                (lambda r, d: d.total_equity is not None and d.total_equity > 0 and d.total_debt is not None and d.total_debt >= 0, 0.5),
            ],
            derived=[
                ("equity_to_total_capital", lambda r, d: safe_divide(d.total_equity, (d.total_equity or 0) + (d.total_debt or 0)) if d.total_equity and d.total_equity > 0 and ((d.total_equity or 0) + (d.total_debt or 0)) > 0 else None),
                ("debt_to_total_capital", lambda r, d: safe_divide(d.total_debt, (d.total_equity or 0) + (d.total_debt or 0)) if d.total_equity and ((d.total_equity or 0) + (d.total_debt or 0)) > 0 else None),
                ("equity_multiplier", lambda r, d: safe_divide((d.total_equity or 0) + (d.total_debt or 0), d.total_equity) if d.total_equity and d.total_equity > 0 else None),
                ("leverage_headroom", lambda r, d: 1.0 - (r.get("debt_to_total_capital") or 0)),
                ("funding_stability", lambda r, d: r.get("equity_to_total_capital")),
            ],
            label="Funding Mix Balance",
        )

    def expense_ratio_discipline_analysis(self, data: FinancialData) -> ExpenseRatioDisciplineResult:
        """Phase 292: Expense Ratio Discipline Analysis."""
        return self._scored_analysis(
            data=data, result_class=ExpenseRatioDisciplineResult,
            ratio_defs=[
                ("opex_to_revenue", "operating_expenses", "revenue"),
                ("cogs_to_revenue", "cogs", "revenue"),
                ("operating_margin", "operating_income", "revenue"),
            ],
            score_field="erd_score", grade_field="erd_grade",
            primary="opex_to_revenue", higher_is_better=False,
            thresholds=[(0.30, 10.0), (0.40, 8.5), (0.50, 7.0), (0.60, 5.5), (0.70, 4.0), (0.80, 2.5)],
            adjustments=[
                (lambda r, d: r.get("cogs_to_revenue") is not None and r["cogs_to_revenue"] <= 0.60, 0.5),
                (lambda r, d: d.operating_income is not None and d.operating_income > 0, 0.5),
            ],
            derived=[
                ("total_expense_ratio", lambda r, d: safe_divide((d.operating_expenses or 0) + (d.cogs or 0), d.revenue) if d.revenue else None),
                ("expense_efficiency", lambda r, d: safe_divide(d.revenue, (d.operating_expenses or 0) + (d.cogs or 0)) if ((d.operating_expenses or 0) + (d.cogs or 0)) > 0 else None),
            ],
            label="Expense Ratio Discipline",
        )

    def revenue_cash_realization_analysis(self, data: FinancialData) -> RevenueCashRealizationResult:
        """Phase 291: Revenue Cash Realization Analysis."""
        return self._scored_analysis(
            data=data, result_class=RevenueCashRealizationResult,
            ratio_defs=[
                ("ocf_to_revenue", "operating_cash_flow", "revenue"),
            ],
            score_field="rcr_score", grade_field="rcr_grade",
            primary="ocf_to_revenue", higher_is_better=True,
            thresholds=[(0.30, 10.0), (0.22, 8.5), (0.15, 7.0), (0.10, 5.5), (0.05, 4.0), (0.02, 2.5)],
            adjustments=[
                (lambda r, d: r.get("collection_rate") is not None and r["collection_rate"] >= 0.85, 0.5),
                (lambda r, d: d.operating_cash_flow is not None and d.operating_cash_flow > 0 and d.revenue is not None and d.revenue > 0, 0.5),
            ],
            derived=[
                ("collection_rate", lambda r, d: safe_divide((d.revenue or 0) - (d.accounts_receivable or 0), d.revenue) if d.revenue and d.revenue > 0 else None),
                ("cash_to_revenue", lambda r, d: safe_divide((d.revenue or 0) - (d.accounts_receivable or 0), d.revenue) if d.revenue and d.revenue > 0 and ((d.revenue or 0) - (d.accounts_receivable or 0)) else None),
                ("revenue_cash_gap", lambda r, d: (d.revenue or 0) - (d.operating_cash_flow or 0) if d.revenue else None),
                ("cash_conversion_speed", lambda r, d: r.get("ocf_to_revenue")),
                ("revenue_quality_ratio", lambda r, d: r.get("ocf_to_revenue")),
            ],
            label="Revenue Cash Realization",
        )

    def net_debt_position_analysis(self, data: FinancialData) -> NetDebtPositionResult:
        """Phase 286: Net Debt Position Analysis.

        Measures the company's net indebtedness after accounting for cash,
        indicating how leveraged the balance sheet truly is.
        """
        result = NetDebtPositionResult()

        total_debt = getattr(data, 'total_debt', None) or 0
        cash = getattr(data, 'cash', None) or 0
        ebitda = getattr(data, 'ebitda', None) or 0
        total_equity = getattr(data, 'total_equity', None) or 0
        total_assets = getattr(data, 'total_assets', None) or 0
        ocf = getattr(data, 'operating_cash_flow', None) or 0

        if not total_debt or total_debt <= 0:
            return result

        # Core metrics
        net_debt = total_debt - cash
        result.net_debt = net_debt
        result.net_debt_to_ebitda = safe_divide(net_debt, ebitda)
        result.net_debt_to_equity = safe_divide(net_debt, total_equity)
        result.net_debt_to_assets = safe_divide(net_debt, total_assets)
        result.cash_to_debt = safe_divide(cash, total_debt)
        result.net_debt_to_ocf = safe_divide(net_debt, ocf)

        # Scoring: Net Debt / EBITDA — lower = better (negative = net cash position)
        nd_ebitda = result.net_debt_to_ebitda
        if nd_ebitda is None:
            # No EBITDA — use net_debt sign
            if net_debt <= 0:
                base = 10.0
            else:
                base = 2.0
        elif nd_ebitda <= 0:
            base = 10.0  # Net cash position
        elif nd_ebitda <= 1.0:
            base = 8.5
        elif nd_ebitda <= 2.0:
            base = 7.0
        elif nd_ebitda <= 3.0:
            base = 5.5
        elif nd_ebitda <= 4.0:
            base = 4.0
        elif nd_ebitda <= 6.0:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Bonus: Cash/Debt >= 0.30
        if result.cash_to_debt is not None and result.cash_to_debt >= 0.30:
            adj += 0.5
        # Bonus: total_debt > 0
        if total_debt > 0:
            adj += 0.5

        result.ndp_score = min(10.0, max(0.0, base + adj))

        if result.ndp_score >= 8:
            result.ndp_grade = "Excellent"
        elif result.ndp_score >= 6:
            result.ndp_grade = "Good"
        elif result.ndp_score >= 4:
            result.ndp_grade = "Adequate"
        else:
            result.ndp_grade = "Weak"

        result.summary = (
            f"Net Debt Position: {result.ndp_grade} "
            f"(Score: {result.ndp_score:.1f}/10). "
            f"Net Debt=${net_debt:,.0f}, "
            f"Net Debt/EBITDA={nd_ebitda:.2f}x."
            if nd_ebitda is not None
            else f"Net Debt Position: {result.ndp_grade} "
            f"(Score: {result.ndp_score:.1f}/10). "
            f"Net Debt=${net_debt:,.0f}."
        )

        return result

    def liability_coverage_strength_analysis(self, data: FinancialData) -> LiabilityCoverageStrengthResult:
        """Phase 281: Liability Coverage Strength Analysis."""
        return self._scored_analysis(
            data=data, result_class=LiabilityCoverageStrengthResult,
            ratio_defs=[
                ("ocf_to_liabilities", "operating_cash_flow", "total_liabilities"),
                ("ebitda_to_liabilities", "ebitda", "total_liabilities"),
                ("assets_to_liabilities", "total_assets", "total_liabilities"),
                ("equity_to_liabilities", "total_equity", "total_liabilities"),
                ("liability_to_revenue", "total_liabilities", "revenue"),
                ("liability_burden", "total_liabilities", "total_assets"),
            ],
            score_field="lcs_score", grade_field="lcs_grade",
            primary="ocf_to_liabilities", higher_is_better=True,
            thresholds=[(0.50, 10.0), (0.35, 8.5), (0.25, 7.0), (0.15, 5.5), (0.10, 4.0), (0.05, 2.5)],
            adjustments=[
                (lambda r, d: r.get("assets_to_liabilities") is not None and r["assets_to_liabilities"] >= 2.0, 0.5),
                (lambda r, d: d.operating_cash_flow is not None and d.operating_cash_flow > 0 and d.total_liabilities is not None and d.total_liabilities > 0, 0.5),
            ],
            label="Liability Coverage Strength",
        )

    def capital_adequacy_analysis(self, data: FinancialData) -> CapitalAdequacyResult:
        """Phase 279: Capital Adequacy Analysis."""
        return self._scored_analysis(
            data=data, result_class=CapitalAdequacyResult,
            ratio_defs=[
                ("equity_ratio", "total_equity", "total_assets"),
                ("equity_to_debt", "total_equity", "total_debt"),
                ("retained_to_equity", "retained_earnings", "total_equity"),
                ("equity_to_liabilities", "total_equity", "total_liabilities"),
                ("tangible_equity_ratio", "total_equity", "total_assets"),
            ],
            score_field="caq_score", grade_field="caq_grade",
            primary="equity_ratio", higher_is_better=True,
            thresholds=[(0.60, 10.0), (0.50, 8.5), (0.40, 7.0), (0.30, 5.5), (0.20, 4.0), (0.10, 2.5)],
            adjustments=[
                (lambda r, d: r.get("retained_to_equity") is not None and r["retained_to_equity"] >= 0.50, 0.5),
                (lambda r, d: d.total_equity is not None and d.total_equity > 0 and d.total_assets is not None and d.total_assets > 0, 0.5),
            ],
            derived=[
                ("capital_buffer", lambda r, d: safe_divide((d.total_equity or 0) - (d.total_liabilities or 0), d.total_assets) if d.total_liabilities is not None and d.total_assets is not None and d.total_assets > 0 else None),
            ],
            label="Capital Adequacy",
        )

    def operating_income_quality_analysis(self, data: FinancialData) -> OperatingIncomeQualityResult:
        """Phase 275: Operating Income Quality Analysis."""
        return self._scored_analysis(
            data=data, result_class=OperatingIncomeQualityResult,
            ratio_defs=[
                ("oi_to_revenue", "operating_income", "revenue"),
                ("oi_to_ebitda", "operating_income", "ebitda"),
                ("oi_to_ocf", "operating_income", "operating_cash_flow"),
                ("oi_to_total_assets", "operating_income", "total_assets"),
            ],
            score_field="oiq_score", grade_field="oiq_grade",
            primary="oi_to_revenue", higher_is_better=True,
            thresholds=[(0.30, 10.0), (0.20, 8.5), (0.15, 7.0), (0.10, 5.5), (0.05, 4.0), (0.02, 2.5)],
            adjustments=[
                (lambda r, d: d.operating_cash_flow is not None and d.operating_cash_flow > 0 and d.operating_income is not None and d.operating_cash_flow > d.operating_income, 0.5),
                (lambda r, d: d.operating_income is not None and d.operating_income > 0 and d.revenue is not None and d.revenue > 0, 0.5),
            ],
            derived=[
                ("operating_spread", lambda r, d: r.get("oi_to_revenue")),
                ("oi_cash_backing", lambda r, d: safe_divide(d.operating_cash_flow, d.operating_income) if d.operating_cash_flow else None),
            ],
            label="Operating Income Quality",
        )

    def ebitda_to_debt_coverage_analysis(self, data: FinancialData) -> EbitdaToDebtCoverageResult:
        """Phase 274: EBITDA-to-Debt Coverage Analysis."""
        return self._scored_analysis(
            data=data, result_class=EbitdaToDebtCoverageResult,
            ratio_defs=[
                ("ebitda_to_debt", "ebitda", "total_debt"),
                ("ebitda_to_interest", "ebitda", "interest_expense"),
                ("debt_to_ebitda", "total_debt", "ebitda"),
                ("ebitda_to_total_liabilities", "ebitda", "total_liabilities"),
            ],
            score_field="etdc_score", grade_field="etdc_grade",
            primary="ebitda_to_debt", higher_is_better=True,
            thresholds=[(1.0, 10.0), (0.60, 8.5), (0.40, 7.0), (0.25, 5.5), (0.15, 4.0), (0.08, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ebitda_to_interest") is not None and r["ebitda_to_interest"] >= 3.0, 0.5),
                (lambda r, d: d.net_income is not None and d.net_income > 0 and d.ebitda is not None and d.ebitda > 0, 0.5),
            ],
            derived=[
                ("debt_service_buffer", lambda r, d: safe_divide((d.ebitda or 0) - (d.interest_expense or 0), d.total_debt) if d.interest_expense else r.get("ebitda_to_debt")),
                ("leverage_headroom", lambda r, d: r.get("ebitda_to_debt")),
            ],
            label="EBITDA-to-Debt Coverage",
        )

    def debt_quality_analysis(self, data: FinancialData) -> DebtQualityResult:
        """Phase 267: Debt Quality Assessment."""
        return self._scored_analysis(
            data=data, result_class=DebtQualityResult,
            ratio_defs=[
                ("debt_to_equity", "total_debt", "total_equity"),
                ("debt_to_assets", "total_debt", "total_assets"),
                ("long_term_debt_ratio", "total_debt", "total_liabilities"),
                ("debt_to_ebitda", "total_debt", "ebitda"),
            ],
            score_field="dq_score", grade_field="dq_grade",
            primary="debt_to_equity", higher_is_better=False,
            thresholds=[(0.20, 10.0), (0.50, 8.5), (1.00, 7.0), (1.50, 5.5), (2.00, 4.0), (3.00, 2.5)],
            adjustments=[
                (lambda r, d: safe_divide(d.ebit, d.interest_expense) is not None and safe_divide(d.ebit, d.interest_expense) >= 5.0, 0.5),
                (lambda r, d: r.get("debt_to_ebitda") is not None and r["debt_to_ebitda"] <= 3.0, 0.5),
            ],
            derived=[
                ("interest_coverage", lambda r, d: safe_divide(d.ebit, d.interest_expense)),
                ("debt_cost", lambda r, d: safe_divide(d.interest_expense, d.total_debt)),
            ],
            label="Debt Quality",
        )

    def fixed_asset_productivity_analysis(self, data: FinancialData) -> FixedAssetProductivityResult:
        """Phase 263: Fixed Asset Productivity Analysis.

        Evaluates how effectively the company uses its fixed (non-current) assets
        to generate revenue. Fixed Asset Turnover = Revenue / Fixed Assets
        where Fixed Assets = Total Assets - Current Assets.

        Scoring (FAT-based, higher = better):
            >=8.0  → 10
            >=5.0  → 8.5
            >=3.0  → 7.0
            >=2.0  → 5.5
            >=1.0  → 4.0
            >=0.5  → 2.5
            <0.5   → 1.0
        Adjustments:
            +0.5 if fixed_to_total_assets <= 0.40 (asset-light)
            +0.5 if depreciation_to_fixed <= 0.10 (newer assets)
        """
        result = FixedAssetProductivityResult()

        revenue = getattr(data, 'revenue', None)
        ta = getattr(data, 'total_assets', None)
        ca = getattr(data, 'current_assets', None)

        if not revenue or not ta or revenue <= 0 or ta <= 0:
            result.summary = "Fixed Asset Productivity: Insufficient data."
            return result

        fixed_assets = ta - (ca if ca else 0)
        if fixed_assets <= 0:
            result.summary = "Fixed Asset Productivity: No fixed assets."
            return result

        # Core metrics
        result.fixed_asset_turnover = safe_divide(revenue, fixed_assets)
        result.revenue_per_fixed_asset = result.fixed_asset_turnover
        result.fixed_to_total_assets = safe_divide(fixed_assets, ta)
        result.net_fixed_asset_intensity = safe_divide(fixed_assets, revenue)

        capex = getattr(data, 'capex', None)
        if capex and fixed_assets > 0:
            result.capex_to_fixed_assets = safe_divide(capex, fixed_assets)

        dep = getattr(data, 'depreciation', None)
        if dep and fixed_assets > 0:
            result.depreciation_to_fixed = safe_divide(dep, fixed_assets)

        # Scoring
        fat = result.fixed_asset_turnover
        if fat is None:
            result.summary = "Fixed Asset Productivity: Could not compute turnover."
            return result

        if fat >= 8.0:
            score = 10.0
        elif fat >= 5.0:
            score = 8.5
        elif fat >= 3.0:
            score = 7.0
        elif fat >= 2.0:
            score = 5.5
        elif fat >= 1.0:
            score = 4.0
        elif fat >= 0.5:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.fixed_to_total_assets is not None and result.fixed_to_total_assets <= 0.40:
            score += 0.5
        if result.depreciation_to_fixed is not None and result.depreciation_to_fixed <= 0.10:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.fap_score = score
        if score >= 8:
            result.fap_grade = "Excellent"
        elif score >= 6:
            result.fap_grade = "Good"
        elif score >= 4:
            result.fap_grade = "Adequate"
        else:
            result.fap_grade = "Weak"

        result.summary = f"Fixed Asset Productivity: Turnover {fat:.1f}x — {result.fap_grade} ({score:.1f}/10)."
        return result

    def depreciation_burden_analysis(self, data: FinancialData) -> DepreciationBurdenResult:
        """Phase 259: Depreciation Burden Analysis."""
        return self._scored_analysis(
            data=data, result_class=DepreciationBurdenResult,
            ratio_defs=[
                ("dep_to_revenue", "depreciation", "revenue"),
                ("dep_to_assets", "depreciation", "total_assets"),
                ("dep_to_ebitda", "depreciation", "ebitda"),
                ("dep_to_gross_profit", "depreciation", "gross_profit"),
            ],
            score_field="db_score", grade_field="db_grade",
            primary="dep_to_revenue", higher_is_better=False,
            thresholds=[(0.03, 10.0), (0.05, 8.5), (0.08, 7.0), (0.12, 5.5), (0.18, 4.0), (0.25, 2.5)],
            adjustments=[
                (lambda r, d: r.get("dep_to_ebitda") is not None and r["dep_to_ebitda"] <= 0.20, 0.5),
                (lambda r, d: r.get("dep_to_assets") is not None and r["dep_to_assets"] <= 0.03, 0.5),
            ],
            derived=[
                ("ebitda_to_ebit_spread", lambda r, d: safe_divide(d.ebitda, d.ebit) if d.ebitda and d.ebit and d.ebit > 0 else None),
                ("asset_age_proxy", lambda r, d: r.get("dep_to_assets")),
            ],
            label="Depreciation Burden",
        )

    def debt_to_capital_analysis(self, data: FinancialData) -> DebtToCapitalResult:
        """Phase 258: Debt-to-Capital Analysis.

        Measures the proportion of debt in the capital structure.
        Debt-to-Capital = Total Debt / (Total Debt + Total Equity).
        Lower ratio = less financial risk.
        """
        result = DebtToCapitalResult()

        td = getattr(data, 'total_debt', None)
        te = getattr(data, 'total_equity', None)
        cash = getattr(data, 'cash', None)

        if not td or not te or te <= 0:
            return result

        total_capital = td + te
        if total_capital <= 0:
            return result

        result.debt_to_capital = safe_divide(td, total_capital)
        result.debt_to_equity = safe_divide(td, te)
        result.equity_ratio = safe_divide(te, total_capital)

        # Long-term debt proxy (use total_debt as proxy)
        result.long_term_debt_to_capital = result.debt_to_capital

        # Net debt to capital
        if cash is not None:
            net_debt = td - cash
            result.net_debt_to_capital = safe_divide(max(0, net_debt), total_capital)

        # Financial risk index: D/E * (1 - equity_ratio)
        if result.debt_to_equity is not None and result.equity_ratio is not None:
            result.financial_risk_index = result.debt_to_equity * (1 - result.equity_ratio)

        # Scoring based on debt_to_capital (lower = better)
        dtc = result.debt_to_capital
        if dtc is not None:
            if dtc <= 0.20:
                score = 10.0
            elif dtc <= 0.30:
                score = 8.5
            elif dtc <= 0.40:
                score = 7.0
            elif dtc <= 0.50:
                score = 5.5
            elif dtc <= 0.60:
                score = 4.0
            elif dtc <= 0.75:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: net debt to capital < debt to capital (cash cushion)
            if result.net_debt_to_capital is not None and result.net_debt_to_capital < dtc - 0.05:
                score += 0.5

            # Adjustment: equity ratio >= 0.60
            if result.equity_ratio is not None and result.equity_ratio >= 0.60:
                score += 0.5

            result.dtc_score = min(10.0, max(0.0, score))

            if result.dtc_score >= 8:
                result.dtc_grade = "Excellent"
            elif result.dtc_score >= 6:
                result.dtc_grade = "Good"
            elif result.dtc_score >= 4:
                result.dtc_grade = "Adequate"
            else:
                result.dtc_grade = "Weak"

            result.summary = (
                f"Debt-to-Capital: D/C={dtc:.1%}, D/E={result.debt_to_equity:.2f}, "
                f"Score={result.dtc_score:.1f}/10 ({result.dtc_grade})."
            )

        return result

    def dividend_payout_analysis(self, data: FinancialData) -> DividendPayoutResult:
        """Phase 251: Dividend Payout Ratio Analysis.

        Measures proportion of earnings distributed as dividends.
        Primary metric: Dividends / Net Income (moderate 20-60% is ideal)."""
        result = DividendPayoutResult()

        div = data.dividends_paid
        ni = data.net_income
        ocf = data.operating_cash_flow
        capex = data.capex
        rev = data.revenue

        if div is None or ni is None or ni <= 0:
            result.summary = "Dividend Payout: Insufficient data (need Dividends and Net Income > 0)."
            return result

        # Dividends_paid is often stored as positive; ensure ratio is positive
        div_abs = abs(div) if div else 0

        result.div_to_ni = safe_divide(div_abs, ni)
        result.retention_ratio = 1.0 - result.div_to_ni if result.div_to_ni is not None else None
        result.div_to_ocf = safe_divide(div_abs, ocf) if ocf and ocf > 0 else None
        fcf = (ocf - capex) if ocf is not None and capex is not None else None
        result.div_to_fcf = safe_divide(div_abs, fcf) if fcf and fcf > 0 else None
        result.div_to_revenue = safe_divide(div_abs, rev)
        result.div_coverage = safe_divide(ni, div_abs) if div_abs > 0 else None

        rat = result.div_to_ni
        if rat is None:
            result.summary = "Dividend Payout: Cannot compute Div/NI."
            return result

        # Scoring: Moderate payout (20-60%) is ideal
        if 0.20 <= rat <= 0.60:
            score = 10.0
        elif 0.10 <= rat < 0.20 or 0.60 < rat <= 0.75:
            score = 8.0
        elif 0.05 <= rat < 0.10 or 0.75 < rat <= 0.90:
            score = 6.0
        elif rat < 0.05:
            score = 5.0  # Very low — may not attract income investors
        else:
            score = 3.0  # >90% — unsustainable

        # Adjustments
        if result.div_coverage is not None:
            if result.div_coverage >= 2.0:
                score += 0.5  # Well-covered dividends
            elif result.div_coverage < 1.0:
                score -= 0.5  # Dividend exceeds earnings — unsustainable

        if result.div_to_ocf is not None and result.div_to_ocf <= 0.50:
            score += 0.5  # Dividends well within cash flow

        score = max(0.0, min(10.0, score))
        result.dpr_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.dpr_grade = grade

        result.summary = (
            f"Dividend Payout: Div/NI={rat:.1%}, Retention={result.retention_ratio:.1%}. "
            f"Score={result.dpr_score:.1f}/10 ({grade})."
        )

        return result

    def operating_cash_flow_ratio_analysis(self, data: FinancialData) -> OperatingCashFlowRatioResult:
        """Phase 249: Operating Cash Flow Ratio Analysis.

        Measures ability to cover obligations from operations.
        Primary metric: OCF / Current Liabilities (higher = better)."""
        result = OperatingCashFlowRatioResult()

        ocf = data.operating_cash_flow
        cl = data.current_liabilities
        tl = data.total_liabilities
        rev = data.revenue
        ni = data.net_income
        td = data.total_debt

        if ocf is None or cl is None or cl <= 0:
            result.summary = "Operating Cash Flow Ratio: Insufficient data (need OCF and CL > 0)."
            return result

        result.ocf_to_cl = safe_divide(ocf, cl)
        result.ocf_to_tl = safe_divide(ocf, tl)
        result.ocf_to_revenue = safe_divide(ocf, rev)
        result.ocf_to_ni = safe_divide(ocf, ni) if ni and ni != 0 else None
        result.ocf_to_debt = safe_divide(ocf, td)
        result.ocf_margin = safe_divide(ocf, rev)

        rat = result.ocf_to_cl
        if rat is None:
            result.summary = "Operating Cash Flow Ratio: Cannot compute OCF/CL."
            return result

        # Scoring: Higher OCF/CL = better short-term coverage
        if rat >= 2.0:
            score = 10.0
        elif rat >= 1.5:
            score = 8.5
        elif rat >= 1.0:
            score = 7.0
        elif rat >= 0.75:
            score = 5.5
        elif rat >= 0.50:
            score = 4.0
        elif rat >= 0.25:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.ocf_to_revenue is not None:
            if result.ocf_to_revenue >= 0.20:
                score += 0.5  # Strong OCF margin
            elif result.ocf_to_revenue < 0.05:
                score -= 0.5  # Weak OCF margin

        if result.ocf_to_ni is not None and result.ocf_to_ni >= 1.0:
            score += 0.5  # OCF exceeds net income — quality earnings

        score = max(0.0, min(10.0, score))
        result.ocfr_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ocfr_grade = grade

        result.summary = (
            f"Operating Cash Flow Ratio: OCF/CL={rat:.2f}x. "
            f"Score={result.ocfr_score:.1f}/10 ({grade})."
        )

        return result

    def cash_conversion_cycle_analysis(self, data: FinancialData) -> CashConversionCycleResult:
        """Phase 248: Cash Conversion Cycle Analysis.

        Measures time to convert resource inputs into cash.
        CCC = DSO + DIO - DPO. Lower/negative is better."""
        result = CashConversionCycleResult()

        rev = data.revenue
        cogs = data.cogs
        ar = data.accounts_receivable
        inv = data.inventory
        ap = data.accounts_payable

        # Need at least revenue and COGS to compute any component
        if rev is None or rev <= 0 or cogs is None or cogs <= 0:
            result.summary = "Cash Conversion Cycle: Insufficient data (need Revenue and COGS > 0)."
            return result

        # Compute components (each may be None if numerator is None/0)
        result.dso = safe_divide(ar * 365, rev) if ar and ar > 0 else None
        result.dio = safe_divide(inv * 365, cogs) if inv and inv > 0 else None
        result.dpo = safe_divide(ap * 365, cogs) if ap and ap > 0 else None

        # CCC = DSO + DIO - DPO (need at least DSO or DIO to be meaningful)
        components = [result.dso, result.dio]
        if all(c is None for c in components):
            result.summary = "Cash Conversion Cycle: Insufficient data (need AR or Inventory > 0)."
            return result

        dso_val = result.dso if result.dso is not None else 0.0
        dio_val = result.dio if result.dio is not None else 0.0
        dpo_val = result.dpo if result.dpo is not None else 0.0

        result.ccc = dso_val + dio_val - dpo_val
        result.working_cap_days = dso_val + dio_val
        result.ccc_to_revenue = safe_divide(result.ccc, 365)

        ccc = result.ccc

        # Scoring: Lower CCC is better; negative is excellent
        if ccc <= 0:
            score = 10.0
        elif ccc <= 15:
            score = 9.0
        elif ccc <= 30:
            score = 8.0
        elif ccc <= 45:
            score = 7.0
        elif ccc <= 60:
            score = 6.0
        elif ccc <= 90:
            score = 5.0
        elif ccc <= 120:
            score = 3.5
        else:
            score = 1.5

        # Adjustments
        if result.dpo is not None and result.dpo >= 30:
            score += 0.5  # Good supplier terms
        if result.dso is not None and result.dso <= 30:
            score += 0.5  # Fast collection

        score = max(0.0, min(10.0, score))
        result.ccc_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ccc_grade = grade

        result.summary = (
            f"Cash Conversion Cycle: CCC={ccc:.1f} days "
            f"(DSO={dso_val:.0f} + DIO={dio_val:.0f} - DPO={dpo_val:.0f}). "
            f"Score={result.ccc_score:.1f}/10 ({grade})."
        )

        return result

    def inventory_turnover_analysis(self, data: FinancialData) -> InventoryTurnoverResult:
        """Phase 247: Inventory Turnover Analysis.

        Measures how efficiently inventory is managed.
        Primary metric: COGS / Inventory (higher = faster turns)."""
        result = InventoryTurnoverResult()

        cogs = data.cogs
        inv = data.inventory
        ca = data.current_assets
        ta = data.total_assets
        rev = data.revenue

        if inv is None or inv <= 0 or cogs is None or cogs <= 0:
            result.summary = "Inventory Turnover Analysis: Insufficient data (need COGS and Inventory > 0)."
            return result

        result.cogs_to_inv = safe_divide(cogs, inv)
        result.dio = safe_divide(inv * 365, cogs)
        result.inv_to_ca = safe_divide(inv, ca)
        result.inv_to_ta = safe_divide(inv, ta)
        result.inv_to_revenue = safe_divide(inv, rev)
        result.inv_velocity = safe_divide(cogs, inv * 12)  # Monthly turns

        rat = result.cogs_to_inv
        if rat is None:
            result.summary = "Inventory Turnover Analysis: Cannot compute COGS/Inventory."
            return result

        # Scoring: Higher COGS/Inv = faster turns = better
        if rat >= 12.0:
            score = 10.0
        elif rat >= 8.0:
            score = 8.5
        elif rat >= 6.0:
            score = 7.0
        elif rat >= 4.0:
            score = 5.5
        elif rat >= 2.0:
            score = 4.0
        elif rat >= 1.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.dio is not None:
            if result.dio <= 30:
                score += 0.5  # Very fast inventory movement
            elif result.dio > 120:
                score -= 0.5  # Very slow — obsolescence risk

        if result.inv_to_ca is not None:
            if result.inv_to_ca <= 0.15:
                score += 0.5  # Low inventory relative to CA
            elif result.inv_to_ca > 0.50:
                score -= 0.5  # Inventory dominates CA

        score = max(0.0, min(10.0, score))
        result.ito_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ito_grade = grade

        result.summary = (
            f"Inventory Turnover Analysis: COGS/Inv={rat:.2f}x, DIO={result.dio:.0f} days. "
            f"Score={result.ito_score:.1f}/10 ({grade})."
        )

        return result

    def payables_turnover_analysis(self, data: FinancialData) -> PayablesTurnoverResult:
        """Phase 246: Payables Turnover Analysis.

        Measures how efficiently AP is managed.
        Primary metric: COGS / AP (moderate is ideal — not too fast, not too slow).
        Optimal DPO: 30-60 days (balances cash management with supplier relations)."""
        result = PayablesTurnoverResult()

        cogs = data.cogs
        ap = data.accounts_payable
        cl = data.current_liabilities
        tl = data.total_liabilities

        if ap is None or ap <= 0 or cogs is None or cogs <= 0:
            result.summary = "Payables Turnover Analysis: Insufficient data (need COGS and AP > 0)."
            return result

        result.cogs_to_ap = safe_divide(cogs, ap)
        result.dpo = safe_divide(ap * 365, cogs)
        result.ap_to_cl = safe_divide(ap, cl)
        result.ap_to_tl = safe_divide(ap, tl)
        result.ap_to_cogs = safe_divide(ap, cogs)
        result.payment_velocity = safe_divide(cogs, ap * 12)  # Monthly turns

        rat = result.cogs_to_ap
        if rat is None:
            result.summary = "Payables Turnover Analysis: Cannot compute COGS/AP."
            return result

        dpo = result.dpo if result.dpo is not None else 0

        # Scoring: Moderate COGS/AP is ideal (DPO 30-60 days = COGS/AP 6-12)
        if 6.0 <= rat <= 12.0:
            score = 10.0  # Ideal range
        elif 4.0 <= rat < 6.0 or 12.0 < rat <= 15.0:
            score = 8.5
        elif 3.0 <= rat < 4.0 or 15.0 < rat <= 20.0:
            score = 7.0
        elif 2.0 <= rat < 3.0:
            score = 5.5  # Paying too slowly
        elif rat > 20.0:
            score = 5.5  # Paying too fast
        elif 1.0 <= rat < 2.0:
            score = 4.0
        else:
            score = 1.0

        # Adjustments based on DPO
        if 30 <= dpo <= 60:
            score += 0.5  # Ideal payment window
        elif dpo > 120:
            score -= 0.5  # Very slow — supplier risk

        if result.ap_to_cl is not None:
            if result.ap_to_cl <= 0.30:
                score += 0.5  # AP well within CL capacity
            elif result.ap_to_cl > 0.70:
                score -= 0.5  # AP dominates CL

        score = max(0.0, min(10.0, score))
        result.pto_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.pto_grade = grade

        result.summary = (
            f"Payables Turnover Analysis: COGS/AP={rat:.2f}x, DPO={dpo:.0f} days. "
            f"Score={result.pto_score:.1f}/10 ({grade})."
        )

        return result

    def receivables_turnover_analysis(self, data: FinancialData) -> ReceivablesTurnoverResult:
        """Phase 245: Receivables Turnover Analysis.

        Measures how efficiently AR is collected.
        Primary metric: Revenue / AR (higher = faster collection)."""
        result = ReceivablesTurnoverResult()

        rev = data.revenue
        ar = data.accounts_receivable
        ca = data.current_assets
        ta = data.total_assets

        if ar is None or ar <= 0 or rev is None or rev <= 0:
            result.summary = "Receivables Turnover Analysis: Insufficient data (need Revenue and AR > 0)."
            return result

        result.rev_to_ar = safe_divide(rev, ar)
        result.dso = safe_divide(ar * 365, rev)
        result.ar_to_ca = safe_divide(ar, ca)
        result.ar_to_ta = safe_divide(ar, ta)
        result.ar_to_revenue = safe_divide(ar, rev)
        result.collection_efficiency = safe_divide(rev - ar, rev)

        rat = result.rev_to_ar
        if rat is None:
            result.summary = "Receivables Turnover Analysis: Cannot compute Rev/AR."
            return result

        # Scoring: Higher Rev/AR = better (faster collection)
        if rat >= 15.0:
            score = 10.0
        elif rat >= 10.0:
            score = 8.5
        elif rat >= 8.0:
            score = 7.0
        elif rat >= 6.0:
            score = 5.5
        elif rat >= 4.0:
            score = 4.0
        elif rat >= 2.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.dso is not None:
            if result.dso <= 30:
                score += 0.5  # Very fast collection
            elif result.dso > 90:
                score -= 0.5  # Slow collection

        if result.ar_to_revenue is not None:
            if result.ar_to_revenue <= 0.05:
                score += 0.5  # Low AR relative to revenue
            elif result.ar_to_revenue > 0.25:
                score -= 0.5  # High AR relative to revenue

        score = max(0.0, min(10.0, score))
        result.rto_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.rto_grade = grade

        result.summary = (
            f"Receivables Turnover Analysis: Rev/AR={rat:.2f}x, DSO={result.dso:.0f} days. "
            f"Score={result.rto_score:.1f}/10 ({grade})."
        )

        return result

    def cash_conversion_efficiency_analysis(self, data: FinancialData) -> CashConversionEfficiencyResult:
        """Phase 237: Cash Conversion Efficiency Analysis.

        Measures how effectively operating income converts to cash.
        Primary metric: OCF/OI (higher = better cash conversion quality).
        """
        result = CashConversionEfficiencyResult()

        ocf = data.operating_cash_flow
        oi = data.operating_income
        ni = data.net_income
        rev = data.revenue
        ebitda = data.ebitda
        capex = data.capex
        cash = data.cash

        if ocf is None or oi is None:
            result.summary = "Cash Conversion Efficiency: Insufficient data."
            return result

        ocf_to_oi = safe_divide(ocf, oi)
        result.ocf_to_oi = ocf_to_oi

        if ni is not None and ni != 0:
            result.ocf_to_ni = safe_divide(ocf, ni)
        if rev is not None and rev != 0:
            result.ocf_to_revenue = safe_divide(ocf, rev)
        if ebitda is not None and ebitda != 0:
            result.ocf_to_ebitda = safe_divide(ocf, ebitda)
        if capex is not None:
            fcf = ocf - capex
            result.fcf_to_oi = safe_divide(fcf, oi)
        if cash is not None:
            result.cash_to_oi = safe_divide(cash, oi)

        if ocf_to_oi is None:
            result.summary = "Cash Conversion Efficiency: OCF/OI not computable."
            return result

        # Scoring: OCF/OI
        if ocf_to_oi >= 1.20:
            score = 10.0
        elif ocf_to_oi >= 1.00:
            score = 8.5
        elif ocf_to_oi >= 0.80:
            score = 7.0
        elif ocf_to_oi >= 0.60:
            score = 5.5
        elif ocf_to_oi >= 0.40:
            score = 4.0
        elif ocf_to_oi >= 0.20:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OCF/NI
        if result.ocf_to_ni is not None:
            if result.ocf_to_ni >= 1.20:
                score += 0.5
            elif result.ocf_to_ni < 0.80:
                score -= 0.5

        # Adjustment: OCF/Revenue
        if result.ocf_to_revenue is not None:
            if result.ocf_to_revenue >= 0.20:
                score += 0.5
            elif result.ocf_to_revenue < 0.05:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cce_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.cce_grade = grade

        result.summary = (
            f"Cash Conversion Efficiency: OCF/OI={ocf_to_oi:.4f}. "
            f"Score={score:.1f}/10. Grade={grade}."
        )

        return result

    def fixed_cost_leverage_ratio_analysis(self, data: FinancialData) -> FixedCostLeverageRatioResult:
        """Phase 236: Fixed Cost Leverage Ratio Analysis.

        Measures degree of operating leverage (DOL) as GP/OI.
        Lower DOL means less sensitivity to revenue changes (more stable earnings).
        """
        result = FixedCostLeverageRatioResult()

        gp = data.gross_profit
        oi = data.operating_income
        rev = data.revenue
        cogs = data.cogs
        opex = data.operating_expenses

        if gp is None or oi is None or rev is None:
            result.summary = "Fixed Cost Leverage Ratio: Insufficient data."
            return result

        dol = safe_divide(gp, oi)
        result.dol = dol

        result.contribution_margin = safe_divide(gp, rev)
        result.oi_to_revenue = safe_divide(oi, rev)
        if cogs is not None:
            result.cogs_to_revenue = safe_divide(cogs, rev)
        if opex is not None:
            result.opex_to_revenue = safe_divide(opex, rev)

        # Breakeven proxy: OpEx / GP (how much of GP is consumed by fixed costs)
        if opex is not None:
            result.breakeven_proxy = safe_divide(opex, gp)

        if dol is None:
            result.summary = "Fixed Cost Leverage Ratio: DOL not computable (OI=0)."
            return result

        # Scoring: DOL (lower = more stable = better)
        if dol <= 1.5:
            score = 10.0
        elif dol <= 2.0:
            score = 8.5
        elif dol <= 2.5:
            score = 7.0
        elif dol <= 3.0:
            score = 5.5
        elif dol <= 4.0:
            score = 4.0
        elif dol <= 5.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OI/Revenue (operating margin)
        if result.oi_to_revenue is not None:
            if result.oi_to_revenue >= 0.15:
                score += 0.5
            elif result.oi_to_revenue < 0.05:
                score -= 0.5

        # Adjustment: COGS/Revenue (cost efficiency)
        if result.cogs_to_revenue is not None:
            if result.cogs_to_revenue <= 0.50:
                score += 0.5
            elif result.cogs_to_revenue > 0.80:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.fclr_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.fclr_grade = grade

        result.summary = (
            f"Fixed Cost Leverage Ratio: DOL={dol:.4f}. "
            f"Score={score:.1f}/10. Grade={grade}."
        )

        return result

    def revenue_quality_index_analysis(self, data: FinancialData) -> RevenueQualityIndexResult:
        """Phase 232: Revenue Quality Index Analysis."""
        return self._scored_analysis(
            data=data, result_class=RevenueQualityIndexResult,
            ratio_defs=[
                ("ocf_to_revenue", "operating_cash_flow", "revenue"),
                ("gross_margin", "gross_profit", "revenue"),
                ("ni_to_revenue", "net_income", "revenue"),
                ("ebitda_to_revenue", "ebitda", "revenue"),
                ("ar_to_revenue", "accounts_receivable", "revenue"),
                ("cash_to_revenue", "cash", "revenue"),
            ],
            score_field="rqi_score", grade_field="rqi_grade",
            primary="ocf_to_revenue", higher_is_better=True,
            thresholds=[(0.25, 10.0), (0.20, 8.5), (0.15, 7.0), (0.10, 5.5), (0.05, 4.0), (0.00, 2.5)],
            adjustments=[
                (lambda r, d: r.get("gross_margin") is not None and r["gross_margin"] >= 0.50, 0.5),
                (lambda r, d: r.get("gross_margin") is not None and r["gross_margin"] < 0.20, -0.5),
                (lambda r, d: r.get("ar_to_revenue") is not None and r["ar_to_revenue"] <= 0.10, 0.5),
                (lambda r, d: r.get("ar_to_revenue") is not None and r["ar_to_revenue"] > 0.30, -0.5),
            ],
            label="Revenue Quality Index",
        )

    def fixed_asset_utilization_analysis(self, data: FinancialData) -> FixedAssetUtilizationResult:
        """Phase 221: Fixed Asset Utilization Analysis.

        Measures how effectively fixed assets generate revenue.
        Primary metric: Revenue / Fixed Assets (higher is better).
        Fixed Assets proxy: Total Assets - Current Assets.
        """
        result = FixedAssetUtilizationResult()

        # Fixed assets proxy
        fixed_assets = None
        if data.total_assets is not None and data.current_assets is not None:
            fixed_assets = data.total_assets - data.current_assets
            if fixed_assets <= 0:
                fixed_assets = None

        fa_turnover = safe_divide(data.revenue, fixed_assets) if fixed_assets is not None else None
        result.fixed_asset_turnover = fa_turnover
        result.depreciation_to_revenue = safe_divide(data.depreciation, data.revenue)
        result.capex_to_depreciation = safe_divide(data.capex, data.depreciation)
        result.capex_to_total_assets = safe_divide(data.capex, data.total_assets)

        if fixed_assets is not None and data.total_assets is not None:
            result.fixed_to_total_assets = safe_divide(fixed_assets, data.total_assets)

        result.depreciation_to_total_assets = safe_divide(data.depreciation, data.total_assets)

        if fa_turnover is None:
            result.fau_score = 0.0
            result.fau_grade = "Weak"
            result.summary = "Fixed Asset Utilization: Insufficient data."
            return result

        # Scoring: FA Turnover higher is better
        if fa_turnover >= 3.0:
            score = 10.0
        elif fa_turnover >= 2.5:
            score = 8.5
        elif fa_turnover >= 2.0:
            score = 7.0
        elif fa_turnover >= 1.5:
            score = 5.5
        elif fa_turnover >= 1.0:
            score = 4.0
        elif fa_turnover >= 0.5:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: CapEx/D&A — reinvestment ratio
        capex_da = result.capex_to_depreciation
        if capex_da is not None:
            if capex_da >= 1.5:
                score += 0.5
            elif capex_da < 0.5:
                score -= 0.5

        # Adjustment: D&A/Rev — low means assets are productive
        da_rev = result.depreciation_to_revenue
        if da_rev is not None:
            if da_rev <= 0.03:
                score += 0.5
            elif da_rev > 0.15:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.fau_score = score

        if score >= 8.0:
            result.fau_grade = "Excellent"
        elif score >= 6.0:
            result.fau_grade = "Good"
        elif score >= 4.0:
            result.fau_grade = "Adequate"
        else:
            result.fau_grade = "Weak"

        result.summary = (
            f"Fixed Asset Utilization: FA Turnover={fa_turnover:.4f}. "
            f"Score={score:.1f}/10 ({result.fau_grade})."
        )

        return result

    def cost_control_analysis(self, data: FinancialData) -> CostControlResult:
        """Phase 215: Cost Control Analysis.

        Primary metric: OpEx/Revenue — lower is better.
        Measures how well the company controls operating costs.
        """
        result = CostControlResult()

        # OpEx/Revenue (primary)
        opex_ratio = safe_divide(data.operating_expenses, data.revenue)
        result.opex_to_revenue = opex_ratio

        # COGS/Revenue
        cogs_ratio = safe_divide(data.cogs, data.revenue)
        result.cogs_to_revenue = cogs_ratio

        # SGA proxy: (OpEx - COGS) / Revenue — only if both exist
        if data.operating_expenses is not None and data.cogs is not None:
            sga = data.operating_expenses - data.cogs
            if sga >= 0:
                result.sga_to_revenue = safe_divide(sga, data.revenue)
            else:
                result.sga_to_revenue = None
        else:
            result.sga_to_revenue = None

        # Operating Margin: OI/Revenue
        oi_ratio = safe_divide(data.operating_income, data.revenue)
        result.operating_margin = oi_ratio

        # OpEx/Gross Profit
        result.opex_to_gross_profit = safe_divide(data.operating_expenses, data.gross_profit)

        # EBITDA Margin
        result.ebitda_margin = safe_divide(data.ebitda, data.revenue)

        # Scoring: based on OpEx/Revenue (lower is better)
        if opex_ratio is None:
            result.cc_score = 0.0
            result.cc_grade = "Weak"
            result.summary = "Cost Control: Insufficient data for analysis."
            return result

        if opex_ratio <= 0.15:
            score = 10.0
        elif opex_ratio <= 0.20:
            score = 8.5
        elif opex_ratio <= 0.25:
            score = 7.0
        elif opex_ratio <= 0.30:
            score = 5.5
        elif opex_ratio <= 0.40:
            score = 4.0
        elif opex_ratio <= 0.55:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: Operating Margin
        if oi_ratio is not None:
            if oi_ratio >= 0.25:
                score += 0.5
            elif oi_ratio < 0.05:
                score -= 0.5

        # Adjustment: COGS/Revenue
        if cogs_ratio is not None:
            if cogs_ratio <= 0.40:
                score += 0.5
            elif cogs_ratio > 0.75:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cc_score = score

        if score >= 8.0:
            result.cc_grade = "Excellent"
        elif score >= 6.0:
            result.cc_grade = "Good"
        elif score >= 4.0:
            result.cc_grade = "Adequate"
        else:
            result.cc_grade = "Weak"

        or_str = f"{opex_ratio:.4f}" if opex_ratio is not None else "N/A"
        oi_str = f"{oi_ratio:.4f}" if oi_ratio is not None else "N/A"
        cogs_str = f"{cogs_ratio:.4f}" if cogs_ratio is not None else "N/A"
        result.summary = (
            f"Cost Control: OpEx/Rev={or_str}, "
            f"OI/Rev={oi_str}, COGS/Rev={cogs_str}. "
            f"Score={score:.1f}/10 ({result.cc_grade})."
        )

        return result

    def valuation_signal_analysis(self, data: FinancialData) -> ValuationSignalResult:
        """Phase 212: Valuation Signal Analysis.

        Evaluates valuation attractiveness using proxy multiples.
        Primary metric: EV/EBITDA proxy (lower is better — more undervalued).
        EV proxy = Total Assets + Total Debt - Cash.
        """
        result = ValuationSignalResult()

        # Enterprise Value proxy
        ev = None
        if data.total_assets is not None:
            ev = data.total_assets
            if data.total_debt is not None:
                ev += data.total_debt
            if data.cash is not None:
                ev -= data.cash

        # EV/EBITDA (primary — lower is better)
        ev_ebitda = safe_divide(ev, data.ebitda)
        result.ev_to_ebitda = ev_ebitda

        # P/E proxy (TA/NI)
        result.price_to_earnings = safe_divide(data.total_assets, data.net_income)

        # P/B proxy (TA/TE)
        result.price_to_book = safe_divide(data.total_assets, data.total_equity)

        # EV/Revenue
        result.ev_to_revenue = safe_divide(ev, data.revenue)

        # Earnings Yield (NI/TA)
        result.earnings_yield = safe_divide(data.net_income, data.total_assets)

        # FCF Yield (FCF/TA)
        fcf = None
        if data.operating_cash_flow is not None and data.capex is not None:
            fcf = data.operating_cash_flow - data.capex
        result.fcf_yield = safe_divide(fcf, data.total_assets)

        # --- Scoring: EV/EBITDA-based (lower is better) ---
        if ev_ebitda is None:
            result.vsg_score = 0.0
            result.vsg_grade = "Weak"
            result.summary = "Valuation Signal: Insufficient data for analysis."
            return result

        if ev_ebitda <= 5.0:
            score = 10.0
        elif ev_ebitda <= 8.0:
            score = 8.5
        elif ev_ebitda <= 12.0:
            score = 7.0
        elif ev_ebitda <= 16.0:
            score = 5.5
        elif ev_ebitda <= 20.0:
            score = 4.0
        elif ev_ebitda <= 30.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: Earnings Yield (higher is better)
        ey = result.earnings_yield
        if ey is not None:
            if ey >= 0.10:
                score += 0.5
            elif ey < 0.03:
                score -= 0.5

        # Adjustment: P/B (lower is better)
        pb = result.price_to_book
        if pb is not None:
            if pb <= 1.0:
                score += 0.5
            elif pb > 5.0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.vsg_score = score

        if score >= 8:
            result.vsg_grade = "Excellent"
        elif score >= 6:
            result.vsg_grade = "Good"
        elif score >= 4:
            result.vsg_grade = "Adequate"
        else:
            result.vsg_grade = "Weak"

        eve_s = f"{ev_ebitda:.2f}" if ev_ebitda is not None else "N/A"
        ey_s = f"{ey:.4f}" if ey is not None else "N/A"
        pb_s = f"{pb:.2f}" if pb is not None else "N/A"
        result.summary = (
            f"Valuation Signal: EV/EBITDA={eve_s}, Earnings Yield={ey_s}, "
            f"P/B={pb_s}. Score={score:.1f}/10 ({result.vsg_grade})."
        )

        return result

    def capital_discipline_analysis(self, data: FinancialData) -> CapitalDisciplineResult:
        """Phase 211: Capital Discipline Analysis."""
        return self._scored_analysis(
            data=data, result_class=CapitalDisciplineResult,
            ratio_defs=[
                ("retained_to_equity", "retained_earnings", "total_equity"),
                ("retained_to_assets", "retained_earnings", "total_assets"),
                ("dividend_payout", "dividends_paid", "net_income"),
                ("capex_to_ocf", "capex", "operating_cash_flow"),
                ("debt_to_equity", "total_debt", "total_equity"),
                ("ocf_to_debt", "operating_cash_flow", "total_debt"),
            ],
            score_field="cd_score", grade_field="cd_grade",
            primary="retained_to_equity", higher_is_better=True,
            thresholds=[(0.70, 10.0), (0.60, 8.5), (0.50, 7.0), (0.40, 5.5), (0.30, 4.0), (0.15, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ocf_to_debt") is not None and r["ocf_to_debt"] >= 0.50, 0.5),
                (lambda r, d: r.get("ocf_to_debt") is not None and r["ocf_to_debt"] < 0.15, -0.5),
                (lambda r, d: r.get("capex_to_ocf") is not None and r["capex_to_ocf"] <= 0.30, 0.5),
                (lambda r, d: r.get("capex_to_ocf") is not None and r["capex_to_ocf"] > 0.80, -0.5),
            ],
            label="Capital Discipline",
        )

    def resource_optimization_analysis(self, data: FinancialData) -> ResourceOptimizationResult:
        """Phase 210: Resource Optimization Analysis."""
        return self._scored_analysis(
            data=data, result_class=ResourceOptimizationResult,
            ratio_defs=[
                ("ocf_to_revenue", "operating_cash_flow", "revenue"),
                ("capex_to_revenue", "capex", "revenue"),
                ("ocf_to_assets", "operating_cash_flow", "total_assets"),
                ("dividend_payout_ratio", "dividends_paid", "net_income"),
            ],
            score_field="ro_score", grade_field="ro_grade",
            primary="fcf_to_revenue", higher_is_better=True,
            thresholds=[(0.20, 10.0), (0.15, 8.5), (0.12, 7.0), (0.08, 5.5), (0.05, 4.0), (0.02, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ocf_to_revenue") is not None and r["ocf_to_revenue"] >= 0.20, 0.5),
                (lambda r, d: r.get("ocf_to_revenue") is not None and r["ocf_to_revenue"] < 0.08, -0.5),
                (lambda r, d: r.get("capex_to_revenue") is not None and r["capex_to_revenue"] <= 0.05, 0.5),
                (lambda r, d: r.get("capex_to_revenue") is not None and r["capex_to_revenue"] > 0.20, -0.5),
            ],
            derived=[
                ("fcf_to_revenue", lambda r, d: safe_divide((d.operating_cash_flow or 0) - (d.capex or 0), d.revenue) if d.operating_cash_flow is not None and d.capex is not None and d.revenue and d.revenue > 0 else None),
                ("fcf_to_assets", lambda r, d: safe_divide((d.operating_cash_flow or 0) - (d.capex or 0), d.total_assets) if d.operating_cash_flow is not None and d.capex is not None else None),
            ],
            label="Resource Optimization",
        )

    def financial_productivity_analysis(self, data: FinancialData) -> FinancialProductivityResult:
        """Phase 205: Financial Productivity Analysis."""
        return self._scored_analysis(
            data=data, result_class=FinancialProductivityResult,
            ratio_defs=[
                ("revenue_per_asset", "revenue", "total_assets"),
                ("revenue_per_equity", "revenue", "total_equity"),
                ("ebitda_per_employee_proxy", "ebitda", "operating_expenses"),
                ("operating_income_per_asset", "operating_income", "total_assets"),
                ("net_income_per_revenue", "net_income", "revenue"),
                ("cash_flow_per_asset", "operating_cash_flow", "total_assets"),
            ],
            score_field="fp_score", grade_field="fp_grade",
            primary="revenue_per_asset", higher_is_better=True,
            thresholds=[(2.0, 10.0), (1.5, 8.5), (1.0, 7.0), (0.70, 5.5), (0.40, 4.0), (0.20, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ebitda_per_employee_proxy") is not None and r["ebitda_per_employee_proxy"] >= 1.5, 0.5),
                (lambda r, d: r.get("ebitda_per_employee_proxy") is not None and r["ebitda_per_employee_proxy"] < 0.50, -0.5),
                (lambda r, d: r.get("cash_flow_per_asset") is not None and r["cash_flow_per_asset"] >= 0.12, 0.5),
                (lambda r, d: r.get("cash_flow_per_asset") is not None and r["cash_flow_per_asset"] < 0.05, -0.5),
            ],
            label="Financial Productivity",
        )

    def equity_preservation_analysis(self, data: FinancialData) -> EquityPreservationResult:
        """Phase 198: Equity Preservation Analysis."""
        return self._scored_analysis(
            data=data, result_class=EquityPreservationResult,
            ratio_defs=[
                ("equity_to_assets", "total_equity", "total_assets"),
                ("retained_to_equity", "retained_earnings", "total_equity"),
                ("equity_growth_capacity", "net_income", "total_equity"),
                ("equity_to_liabilities", "total_equity", "total_liabilities"),
                ("tangible_equity_ratio", "total_equity", "total_assets"),
                ("equity_per_revenue", "total_equity", "revenue"),
            ],
            score_field="ep_score", grade_field="ep_grade",
            primary="equity_to_assets", higher_is_better=True,
            thresholds=[(0.60, 10.0), (0.50, 8.5), (0.40, 7.0), (0.30, 5.5), (0.20, 4.0), (0.10, 2.5)],
            adjustments=[
                (lambda r, d: r.get("retained_to_equity") is not None and r["retained_to_equity"] >= 0.60, 0.5),
                (lambda r, d: r.get("retained_to_equity") is not None and r["retained_to_equity"] < 0.20, -0.5),
                (lambda r, d: r.get("equity_growth_capacity") is not None and r["equity_growth_capacity"] >= 0.15, 0.5),
                (lambda r, d: r.get("equity_growth_capacity") is not None and r["equity_growth_capacity"] < 0.05, -0.5),
            ],
            label="Equity Preservation",
        )

    def debt_management_analysis(self, data: FinancialData) -> DebtManagementResult:
        """Phase 197: Debt Management Analysis."""
        return self._scored_analysis(
            data=data, result_class=DebtManagementResult,
            ratio_defs=[
                ("debt_to_operating_income", "total_debt", "operating_income"),
                ("debt_to_ocf", "total_debt", "operating_cash_flow"),
                ("interest_to_revenue", "interest_expense", "revenue"),
                ("debt_to_gross_profit", "total_debt", "gross_profit"),
            ],
            score_field="dm_score", grade_field="dm_grade",
            primary="debt_to_operating_income", higher_is_better=False,
            thresholds=[(1.0, 10.0), (2.0, 8.5), (3.0, 7.0), (4.0, 5.5), (5.0, 4.0), (7.0, 2.5)],
            adjustments=[
                (lambda r, d: r.get("interest_to_revenue") is not None and r["interest_to_revenue"] <= 0.03, 0.5),
                (lambda r, d: r.get("interest_to_revenue") is not None and r["interest_to_revenue"] > 0.10, -0.5),
                (lambda r, d: r.get("debt_coverage_ratio") is not None and r["debt_coverage_ratio"] >= 3.0, 0.5),
                (lambda r, d: r.get("debt_coverage_ratio") is not None and r["debt_coverage_ratio"] < 1.5, -0.5),
            ],
            derived=[
                ("net_debt_ratio", lambda r, d: ((d.total_debt or 0) - (d.cash or 0)) / d.total_assets if d.total_debt is not None and d.cash is not None and d.total_assets and d.total_assets > 0 else None),
                ("debt_coverage_ratio", lambda r, d: d.ebitda / ((d.interest_expense or 0) + (d.total_debt or 0) * 0.1) if d.ebitda is not None and d.interest_expense is not None and d.total_debt is not None and ((d.interest_expense or 0) + (d.total_debt or 0) * 0.1) > 0 else None),
            ],
            label="Debt Management",
        )

    def income_retention_analysis(self, data: FinancialData) -> IncomeRetentionResult:
        """Phase 196: Income Retention Analysis."""
        return self._scored_analysis(
            data=data, result_class=IncomeRetentionResult,
            ratio_defs=[
                ("net_to_gross_ratio", "net_income", "gross_profit"),
                ("net_to_operating_ratio", "net_income", "operating_income"),
                ("net_to_ebitda_ratio", "net_income", "ebitda"),
                ("retention_rate", "retained_earnings", "net_income"),
                ("income_to_asset_generation", "net_income", "total_assets"),
                ("after_tax_margin", "net_income", "revenue"),
            ],
            score_field="ir_score", grade_field="ir_grade",
            primary="net_to_gross_ratio", higher_is_better=True,
            thresholds=[(0.45, 10.0), (0.35, 8.5), (0.25, 7.0), (0.18, 5.5), (0.10, 4.0), (0.05, 2.5)],
            adjustments=[
                (lambda r, d: r.get("net_to_operating_ratio") is not None and r["net_to_operating_ratio"] >= 0.80, 0.5),
                (lambda r, d: r.get("net_to_operating_ratio") is not None and r["net_to_operating_ratio"] < 0.50, -0.5),
                (lambda r, d: r.get("after_tax_margin") is not None and r["after_tax_margin"] >= 0.15, 0.5),
                (lambda r, d: r.get("after_tax_margin") is not None and r["after_tax_margin"] < 0.05, -0.5),
            ],
            label="Income Retention",
        )

    def operational_efficiency_analysis(self, data: FinancialData) -> OperationalEfficiencyResult:
        """Phase 195: Operational Efficiency Analysis."""
        return self._scored_analysis(
            data=data, result_class=OperationalEfficiencyResult,
            ratio_defs=[
                ("oi_margin", "operating_income", "revenue"),
                ("revenue_to_assets", "revenue", "total_assets"),
                ("gross_profit_per_asset", "gross_profit", "total_assets"),
                ("opex_efficiency", "revenue", "operating_expenses"),
                ("asset_utilization", "revenue", "current_assets"),
                ("income_per_liability", "operating_income", "total_liabilities"),
            ],
            score_field="oe_score", grade_field="oe_grade",
            primary="oi_margin", higher_is_better=True,
            thresholds=[(0.25, 10.0), (0.20, 8.5), (0.15, 7.0), (0.10, 5.5), (0.05, 4.0), (0.02, 2.5)],
            adjustments=[
                (lambda r, d: r.get("revenue_to_assets") is not None and r["revenue_to_assets"] >= 0.80, 0.5),
                (lambda r, d: r.get("revenue_to_assets") is not None and r["revenue_to_assets"] < 0.30, -0.5),
                (lambda r, d: r.get("opex_efficiency") is not None and r["opex_efficiency"] >= 5.0, 0.5),
                (lambda r, d: r.get("opex_efficiency") is not None and r["opex_efficiency"] < 2.0, -0.5),
            ],
            label="Operational Efficiency",
        )

    def operating_momentum_analysis(self, data: FinancialData) -> OperatingMomentumResult:
        """Phase 191: Operating Momentum Analysis."""
        return self._scored_analysis(
            data=data, result_class=OperatingMomentumResult,
            ratio_defs=[
                ("ebitda_margin", "ebitda", "revenue"),
                ("ebit_margin", "ebit", "revenue"),
                ("ocf_margin", "operating_cash_flow", "revenue"),
                ("gross_to_operating_conversion", "operating_income", "gross_profit"),
                ("operating_cash_conversion", "operating_cash_flow", "operating_income"),
                ("overhead_absorption", "operating_income", "operating_expenses"),
            ],
            score_field="om_score", grade_field="om_grade",
            primary="ebitda_margin", higher_is_better=True,
            thresholds=[(0.30, 10.0), (0.25, 8.5), (0.20, 7.0), (0.15, 5.5), (0.10, 4.0), (0.05, 2.5)],
            adjustments=[
                (lambda r, d: r.get("gross_to_operating_conversion") is not None and r["gross_to_operating_conversion"] >= 0.50, 0.5),
                (lambda r, d: r.get("gross_to_operating_conversion") is not None and r["gross_to_operating_conversion"] < 0.25, -0.5),
                (lambda r, d: r.get("operating_cash_conversion") is not None and r["operating_cash_conversion"] >= 1.0, 0.5),
                (lambda r, d: r.get("operating_cash_conversion") is not None and r["operating_cash_conversion"] < 0.60, -0.5),
            ],
            label="Operating Momentum",
        )

    def payout_discipline_analysis(self, data: FinancialData) -> PayoutDisciplineResult:
        """Phase 185: Payout Discipline Analysis."""
        return self._scored_analysis(
            data=data, result_class=PayoutDisciplineResult,
            ratio_defs=[
                ("cash_dividend_coverage", "operating_cash_flow", "dividends_paid"),
                ("payout_ratio", "dividends_paid", "net_income"),
                ("dividend_to_ocf", "dividends_paid", "operating_cash_flow"),
            ],
            derived=[
                ("retention_ratio", lambda r, d: (
                    (d.net_income - d.dividends_paid) / d.net_income
                    if d.net_income is not None and d.dividends_paid is not None and d.net_income > 0
                    else None
                )),
                ("capex_priority", lambda r, d: (
                    d.capex / (d.capex + d.dividends_paid)
                    if d.capex is not None and d.dividends_paid is not None
                    and (d.capex + d.dividends_paid) > 0 else None
                )),
                ("free_cash_after_dividends", lambda r, d: (
                    (d.operating_cash_flow - d.capex - d.dividends_paid) / d.revenue
                    if d.operating_cash_flow is not None and d.capex is not None
                    and d.dividends_paid is not None and d.revenue is not None and d.revenue > 0
                    else None
                )),
            ],
            score_field="pd_score", grade_field="pd_grade",
            primary="cash_dividend_coverage", higher_is_better=True,
            thresholds=[(5.0, 10.0), (4.0, 8.5), (3.0, 7.0), (2.0, 5.5), (1.5, 4.0), (1.0, 2.5)],
            adjustments=[
                (lambda r, d: r.get("capex_priority") is not None and r["capex_priority"] >= 0.60, 0.5),
                (lambda r, d: r.get("capex_priority") is not None and r["capex_priority"] < 0.30, -0.5),
                (lambda r, d: r.get("retention_ratio") is not None and r["retention_ratio"] >= 0.60, 0.5),
                (lambda r, d: r.get("retention_ratio") is not None and r["retention_ratio"] < 0.30, -0.5),
            ],
            label="Payout Discipline",
        )

    def income_resilience_analysis(self, data: FinancialData) -> IncomeResilienceResult:
        """Phase 184: Income Resilience Analysis."""
        return self._scored_analysis(
            data=data, result_class=IncomeResilienceResult,
            ratio_defs=[
                ("operating_income_stability", "operating_income", "revenue"),
                ("ebit_coverage", "ebit", "interest_expense"),
                ("net_margin_resilience", "net_income", "operating_income"),
                ("depreciation_buffer", "depreciation", "operating_income"),
                ("ebitda_cushion", "ebitda", "interest_expense"),
            ],
            derived=[
                ("tax_interest_drag", lambda r, d: (
                    (d.operating_income - d.net_income) / d.operating_income
                    if d.operating_income is not None and d.net_income is not None and d.operating_income > 0
                    else None
                )),
            ],
            score_field="ir_score", grade_field="ir_grade",
            primary="operating_income_stability", higher_is_better=True,
            thresholds=[(0.25, 10.0), (0.20, 8.5), (0.15, 7.0), (0.10, 5.5), (0.06, 4.0), (0.03, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ebit_coverage") is not None and r["ebit_coverage"] >= 5.0, 0.5),
                (lambda r, d: r.get("ebit_coverage") is not None and r["ebit_coverage"] < 2.0, -0.5),
                (lambda r, d: r.get("net_margin_resilience") is not None and r["net_margin_resilience"] >= 0.70, 0.5),
                (lambda r, d: r.get("net_margin_resilience") is not None and r["net_margin_resilience"] < 0.40, -0.5),
            ],
            label="Income Resilience",
        )

    def structural_strength_analysis(self, data: FinancialData) -> StructuralStrengthResult:
        """Phase 182: Structural Strength Analysis."""
        return self._scored_analysis(
            data=data, result_class=StructuralStrengthResult,
            ratio_defs=[
                ("equity_multiplier", "total_assets", "total_equity"),
                ("debt_to_equity", "total_debt", "total_equity"),
                ("liability_composition", "current_liabilities", "total_liabilities"),
                ("financial_leverage_ratio", "total_liabilities", "total_equity"),
            ],
            derived=[
                ("equity_cushion", lambda r, d: (
                    (d.total_equity - d.total_debt) / d.total_assets
                    if d.total_equity is not None and d.total_debt is not None
                    and d.total_assets is not None and d.total_assets > 0 else None
                )),
                ("fixed_asset_coverage", lambda r, d: (
                    safe_divide(d.total_equity, d.total_assets - d.current_assets)
                    if d.total_equity is not None and d.total_assets is not None and d.current_assets is not None
                    and (d.total_assets - d.current_assets) > 0 else None
                )),
            ],
            score_field="ss_score", grade_field="ss_grade",
            primary="equity_multiplier", higher_is_better=False,
            thresholds=[(1.25, 10.0), (1.50, 8.5), (1.75, 7.0), (2.00, 5.5), (2.50, 4.0), (3.50, 2.5)],
            adjustments=[
                (lambda r, d: r.get("debt_to_equity") is not None and r["debt_to_equity"] <= 0.50, 0.5),
                (lambda r, d: r.get("debt_to_equity") is not None and r["debt_to_equity"] >= 2.00, -0.5),
                (lambda r, d: r.get("equity_cushion") is not None and r["equity_cushion"] >= 0.30, 0.5),
                (lambda r, d: r.get("equity_cushion") is not None and r["equity_cushion"] < 0.10, -0.5),
            ],
            label="Structural Strength",
        )

    def profit_conversion_analysis(self, data: FinancialData) -> ProfitConversionResult:
        """Phase 179: Profit Conversion Analysis."""
        return self._scored_analysis(
            data=data, result_class=ProfitConversionResult,
            ratio_defs=[
                ("gross_conversion", "gross_profit", "revenue"),
                ("operating_conversion", "operating_income", "revenue"),
                ("net_conversion", "net_income", "revenue"),
                ("ebitda_conversion", "ebitda", "revenue"),
                ("cash_conversion", "operating_cash_flow", "revenue"),
                ("profit_to_cash_ratio", "operating_cash_flow", "net_income"),
            ],
            score_field="pc_score", grade_field="pc_grade",
            primary="gross_conversion", higher_is_better=True,
            thresholds=[(0.60, 10.0), (0.50, 8.5), (0.40, 7.0), (0.30, 5.5), (0.20, 4.0), (0.10, 2.5)],
            adjustments=[
                (lambda r, d: r.get("operating_conversion") is not None and r["operating_conversion"] >= 0.20, 0.5),
                (lambda r, d: r.get("operating_conversion") is not None and r["operating_conversion"] < 0.05, -0.5),
                (lambda r, d: r.get("cash_conversion") is not None and r["cash_conversion"] >= 0.20, 0.5),
                (lambda r, d: r.get("cash_conversion") is not None and r["cash_conversion"] < 0.05, -0.5),
            ],
            label="Profit Conversion",
        )

    def asset_deployment_efficiency_analysis(self, data: FinancialData) -> AssetDeploymentEfficiencyResult:
        """Phase 172: Asset Deployment Efficiency Analysis."""
        return self._scored_analysis(
            data=data, result_class=AssetDeploymentEfficiencyResult,
            ratio_defs=[
                ("asset_turnover", "revenue", "total_assets"),
                ("asset_income_yield", "operating_income", "total_assets"),
                ("asset_cash_yield", "operating_cash_flow", "total_assets"),
                ("inventory_velocity", "revenue", "inventory"),
                ("receivables_velocity", "revenue", "accounts_receivable"),
            ],
            derived=[
                ("fixed_asset_leverage", lambda r, d: (
                    safe_divide(d.revenue, d.total_assets - d.current_assets)
                    if d.revenue is not None and d.total_assets is not None and d.current_assets is not None
                    and (d.total_assets - d.current_assets) > 0 else None
                )),
            ],
            score_field="ade_score", grade_field="ade_grade",
            primary="asset_turnover", higher_is_better=True,
            thresholds=[(1.50, 10.0), (1.20, 8.5), (0.90, 7.0), (0.60, 5.5), (0.40, 4.0), (0.20, 2.5)],
            adjustments=[
                (lambda r, d: r.get("asset_income_yield") is not None and r["asset_income_yield"] >= 0.10, 0.5),
                (lambda r, d: r.get("asset_income_yield") is not None and r["asset_income_yield"] < 0.02, -0.5),
                (lambda r, d: r.get("asset_cash_yield") is not None and r["asset_cash_yield"] >= 0.10, 0.5),
                (lambda r, d: r.get("asset_cash_yield") is not None and r["asset_cash_yield"] < 0.02, -0.5),
            ],
            label="Asset Deployment Efficiency",
        )

    def profit_sustainability_analysis(self, data: FinancialData) -> ProfitSustainabilityResult:
        """Phase 171: Profit Sustainability Analysis."""
        return self._scored_analysis(
            data=data, result_class=ProfitSustainabilityResult,
            ratio_defs=[
                ("profit_cash_backing", "operating_cash_flow", "net_income"),
                ("profit_margin_depth", "net_income", "revenue"),
                ("profit_to_asset", "net_income", "total_assets"),
                ("profit_stability_proxy", "operating_income", "ebitda"),
                ("profit_leverage", "net_income", "operating_income"),
            ],
            derived=[
                ("profit_reinvestment", lambda r, d: (
                    (d.net_income - d.dividends_paid) / d.net_income
                    if d.net_income is not None and d.dividends_paid is not None and d.net_income != 0
                    else None
                )),
            ],
            score_field="ps_score", grade_field="ps_grade",
            primary="profit_cash_backing", higher_is_better=True,
            thresholds=[(1.50, 10.0), (1.20, 8.5), (1.00, 7.0), (0.80, 5.5), (0.50, 4.0), (0.20, 2.5)],
            adjustments=[
                (lambda r, d: r.get("profit_margin_depth") is not None and r["profit_margin_depth"] >= 0.15, 0.5),
                (lambda r, d: r.get("profit_margin_depth") is not None and r["profit_margin_depth"] < 0.05, -0.5),
                (lambda r, d: r.get("profit_reinvestment") is not None and r["profit_reinvestment"] >= 0.70, 0.5),
                (lambda r, d: r.get("profit_reinvestment") is not None and r["profit_reinvestment"] < 0.30, -0.5),
            ],
            label="Profit Sustainability",
        )

    def debt_discipline_analysis(self, data: FinancialData) -> DebtDisciplineResult:
        """Phase 170: Debt Discipline Analysis."""
        return self._scored_analysis(
            data=data, result_class=DebtDisciplineResult,
            ratio_defs=[
                ("debt_prudence_ratio", "total_debt", "total_assets"),
                ("debt_coverage_spread", "operating_cash_flow", "total_debt"),
                ("debt_to_equity_leverage", "total_debt", "total_equity"),
                ("interest_absorption", "interest_expense", "revenue"),
            ],
            derived=[
                ("debt_servicing_power", lambda r, d: (
                    d.ebitda / (d.interest_expense + d.total_debt / 5.0)
                    if d.ebitda is not None and d.interest_expense is not None and d.total_debt is not None
                    and (d.interest_expense + d.total_debt / 5.0) > 0 else None
                )),
                ("debt_repayment_capacity", lambda r, d: (
                    (d.operating_cash_flow - d.capex) / d.total_debt
                    if d.operating_cash_flow is not None and d.capex is not None
                    and d.total_debt is not None and d.total_debt != 0 else None
                )),
            ],
            score_field="dd_score", grade_field="dd_grade",
            primary="debt_prudence_ratio", higher_is_better=False,
            thresholds=[(0.10, 10.0), (0.20, 8.5), (0.30, 7.0), (0.40, 5.5), (0.50, 4.0), (0.70, 2.5)],
            adjustments=[
                (lambda r, d: r.get("debt_coverage_spread") is not None and r["debt_coverage_spread"] >= 0.50, 0.5),
                (lambda r, d: r.get("debt_coverage_spread") is not None and r["debt_coverage_spread"] < 0.10, -0.5),
                (lambda r, d: r.get("debt_to_equity_leverage") is not None and r["debt_to_equity_leverage"] <= 0.30, 0.5),
                (lambda r, d: r.get("debt_to_equity_leverage") is not None and r["debt_to_equity_leverage"] >= 1.50, -0.5),
            ],
            label="Debt Discipline",
        )

    def capital_preservation_analysis(self, data: FinancialData) -> CapitalPreservationResult:
        """Phase 168: Capital Preservation Analysis."""
        return self._scored_analysis(
            data=data, result_class=CapitalPreservationResult,
            ratio_defs=[
                ("retained_earnings_power", "retained_earnings", "total_assets"),
                ("operating_capital_ratio", "operating_cash_flow", "total_debt"),
                ("net_worth_growth_proxy", "net_income", "total_equity"),
            ],
            derived=[
                ("capital_erosion_rate", lambda r, d: (
                    (d.total_liabilities - d.cash) / d.total_equity
                    if d.total_liabilities is not None and d.cash is not None
                    and d.total_equity is not None and d.total_equity != 0 else None
                )),
                ("asset_integrity_ratio", lambda r, d: (
                    (d.total_assets - d.total_liabilities) / d.total_assets
                    if d.total_assets is not None and d.total_liabilities is not None
                    and d.total_assets != 0 else None
                )),
                ("capital_buffer", lambda r, d: (
                    (d.current_assets - d.current_liabilities) / d.total_assets
                    if d.current_assets is not None and d.current_liabilities is not None
                    and d.total_assets is not None and d.total_assets != 0 else None
                )),
            ],
            score_field="cp_score", grade_field="cp_grade",
            primary="retained_earnings_power", higher_is_better=True,
            thresholds=[(0.40, 10.0), (0.35, 8.5), (0.30, 7.0), (0.25, 5.5), (0.20, 4.0), (0.10, 2.5)],
            adjustments=[
                (lambda r, d: r.get("capital_erosion_rate") is not None and r["capital_erosion_rate"] <= 0.50, 0.5),
                (lambda r, d: r.get("capital_erosion_rate") is not None and r["capital_erosion_rate"] >= 1.50, -0.5),
                (lambda r, d: r.get("operating_capital_ratio") is not None and r["operating_capital_ratio"] >= 0.50, 0.5),
                (lambda r, d: r.get("operating_capital_ratio") is not None and r["operating_capital_ratio"] < 0.10, -0.5),
            ],
            label="Capital Preservation",
        )

    def obligation_coverage_analysis(self, data: FinancialData) -> ObligationCoverageResult:
        """Phase 160: Obligation Coverage Analysis."""
        return self._scored_analysis(
            data=data, result_class=ObligationCoverageResult,
            ratio_defs=[
                ("ebitda_interest_coverage", "ebitda", "interest_expense"),
                ("cash_interest_coverage", "operating_cash_flow", "interest_expense"),
                ("debt_burden_ratio", "total_debt", "ebitda"),
                ("interest_to_revenue", "interest_expense", "revenue"),
            ],
            derived=[
                ("debt_amortization_capacity", lambda r, d: (
                    safe_divide(d.operating_cash_flow - d.capex, d.total_debt)
                    if d.operating_cash_flow is not None and d.capex is not None else None
                )),
                ("fixed_charge_coverage", lambda r, d: (
                    d.ebitda / ((d.interest_expense or 0.0) + (d.capex or 0.0))
                    if d.ebitda is not None and ((d.interest_expense or 0.0) + (d.capex or 0.0)) > 0
                    else None
                )),
            ],
            score_field="oc_score", grade_field="oc_grade",
            primary="ebitda_interest_coverage", higher_is_better=True,
            thresholds=[(10.0, 10.0), (7.0, 8.5), (5.0, 7.0), (3.0, 5.5), (2.0, 4.0), (1.0, 2.5)],
            adjustments=[
                (lambda r, d: r.get("debt_burden_ratio") is not None and r["debt_burden_ratio"] <= 2.0, 0.5),
                (lambda r, d: r.get("debt_burden_ratio") is not None and r["debt_burden_ratio"] >= 5.0, -0.5),
                (lambda r, d: r.get("cash_interest_coverage") is not None and r["cash_interest_coverage"] >= 8.0, 0.5),
                (lambda r, d: r.get("cash_interest_coverage") is not None and r["cash_interest_coverage"] < 1.5, -0.5),
            ],
            label="Obligation Coverage",
        )

    def internal_growth_capacity_analysis(self, data: FinancialData) -> InternalGrowthCapacityResult:
        """Phase 159: Internal Growth Capacity Analysis."""
        return self._scored_analysis(
            data=data, result_class=InternalGrowthCapacityResult,
            ratio_defs=[
                ("reinvestment_rate", "capex", "depreciation"),
                ("equity_growth_rate", "retained_earnings", "total_equity"),
            ],
            derived=[
                ("plowback_ratio", lambda r, d: (
                    (d.net_income - (d.dividends_paid if d.dividends_paid is not None else 0.0)) / d.net_income
                    if d.net_income is not None and d.net_income != 0 else None
                )),
                ("sustainable_growth_rate", lambda r, d: (
                    safe_divide(d.net_income, d.total_equity) * r["plowback_ratio"]
                    if safe_divide(d.net_income, d.total_equity) is not None and r.get("plowback_ratio") is not None
                    else None
                )),
                ("internal_growth_rate", lambda r, d: (
                    (lambda roa_b: roa_b / (1.0 - roa_b) if roa_b < 1.0 else None)(
                        safe_divide(d.net_income, d.total_assets) * r["plowback_ratio"]
                    ) if safe_divide(d.net_income, d.total_assets) is not None and r.get("plowback_ratio") is not None
                    else None
                )),
                ("growth_financing_ratio", lambda r, d: (
                    d.operating_cash_flow / ((d.capex or 0.0) + (d.dividends_paid or 0.0))
                    if d.operating_cash_flow is not None
                    and ((d.capex or 0.0) + (d.dividends_paid or 0.0)) > 0
                    else None
                )),
            ],
            score_field="igc_score", grade_field="igc_grade",
            primary="growth_financing_ratio", higher_is_better=True,
            thresholds=[(2.5, 10.0), (2.0, 8.5), (1.5, 7.0), (1.2, 5.5), (1.0, 4.0), (0.5, 2.5)],
            adjustments=[
                (lambda r, d: r.get("plowback_ratio") is not None and r["plowback_ratio"] >= 0.60, 0.5),
                (lambda r, d: r.get("plowback_ratio") is not None and r["plowback_ratio"] < 0.20, -0.5),
                (lambda r, d: r.get("reinvestment_rate") is not None and 1.0 <= r["reinvestment_rate"] <= 2.5, 0.5),
                (lambda r, d: r.get("reinvestment_rate") is not None and r["reinvestment_rate"] > 4.0, -0.5),
            ],
            label="Internal Growth Capacity",
        )

    def liability_management_analysis(self, data: FinancialData) -> LiabilityManagementResult:
        """Phase 146: Liability Management Analysis."""
        return self._scored_analysis(
            data=data, result_class=LiabilityManagementResult,
            ratio_defs=[
                ("liability_to_assets", "total_liabilities", "total_assets"),
                ("liability_to_equity", "total_liabilities", "total_equity"),
                ("current_liability_ratio", "current_liabilities", "total_liabilities"),
                ("liability_coverage", "ebitda", "total_liabilities"),
                ("liability_to_revenue", "total_liabilities", "revenue"),
            ],
            derived=[
                ("net_liability", lambda r, d: (
                    (d.total_liabilities - (d.cash if d.cash is not None else 0.0)) / d.total_assets
                    if d.total_liabilities is not None and d.total_assets is not None and d.total_assets > 0
                    else None
                )),
            ],
            score_field="lm_score", grade_field="lm_grade",
            primary="liability_to_assets", higher_is_better=False,
            thresholds=[(0.30, 10.0), (0.35, 8.5), (0.40, 7.0), (0.50, 5.5), (0.60, 4.0), (0.70, 2.5)],
            adjustments=[
                (lambda r, d: r.get("liability_coverage") is not None and r["liability_coverage"] >= 0.30, 0.5),
                (lambda r, d: r.get("liability_coverage") is not None and r["liability_coverage"] < 0.10, -0.5),
                (lambda r, d: r.get("current_liability_ratio") is not None and r["current_liability_ratio"] <= 0.40, 0.5),
                (lambda r, d: r.get("current_liability_ratio") is not None and r["current_liability_ratio"] > 0.70, -0.5),
            ],
            label="Liability Management",
        )

    def revenue_predictability_analysis(self, data: FinancialData) -> RevenuePredictabilityResult:
        """Phase 142: Revenue Predictability Analysis."""
        return self._scored_analysis(
            data=data, result_class=RevenuePredictabilityResult,
            ratio_defs=[
                ("revenue_to_assets", "revenue", "total_assets"),
                ("revenue_to_equity", "revenue", "total_equity"),
                ("revenue_to_debt", "revenue", "total_debt"),
                ("gross_margin", "gross_profit", "revenue"),
                ("operating_margin", "operating_income", "revenue"),
                ("net_margin", "net_income", "revenue"),
            ],
            score_field="rp_score", grade_field="rp_grade",
            primary="operating_margin", higher_is_better=True,
            thresholds=[(0.30, 10.0), (0.25, 8.5), (0.20, 7.0), (0.15, 5.5), (0.10, 4.0), (0.05, 2.5)],
            adjustments=[
                (lambda r, d: r.get("gross_margin") is not None and r["gross_margin"] >= 0.50, 0.5),
                (lambda r, d: r.get("gross_margin") is not None and r["gross_margin"] < 0.20, -0.5),
                (lambda r, d: r.get("net_margin") is not None and r["net_margin"] >= 0.15, 0.5),
                (lambda r, d: r.get("net_margin") is not None and r["net_margin"] < 0.03, -0.5),
            ],
            label="Revenue Predictability",
        )

    def equity_reinvestment_analysis(self, data: FinancialData) -> EquityReinvestmentResult:
        """Phase 139: Equity Reinvestment Analysis."""
        return self._scored_analysis(
            data=data, result_class=EquityReinvestmentResult,
            ratio_defs=[
                ("reinvestment_rate", "capex", "net_income"),
                ("equity_growth_proxy", "retained_earnings", "total_equity"),
                ("dividend_coverage", "net_income", "dividends_paid"),
            ],
            derived=[
                ("retention_ratio", lambda r, d: (
                    safe_divide(d.net_income - d.dividends_paid, d.net_income)
                    if d.net_income is not None and d.dividends_paid is not None
                    else (safe_divide(d.net_income, d.net_income) if d.net_income is not None else None)
                )),
                ("plowback_to_assets", lambda r, d: (
                    safe_divide(d.net_income - d.dividends_paid, d.total_assets)
                    if d.net_income is not None and d.dividends_paid is not None else None
                )),
                ("internal_growth_rate", lambda r, d: (
                    r["equity_growth_proxy"] * safe_divide(d.net_income, d.total_equity)
                    if r.get("equity_growth_proxy") is not None
                    and safe_divide(d.net_income, d.total_equity) is not None else None
                )),
            ],
            score_field="er_score", grade_field="er_grade",
            primary="retention_ratio", higher_is_better=True,
            thresholds=[(0.90, 10.0), (0.80, 8.5), (0.70, 7.0), (0.60, 5.5), (0.50, 4.0), (0.30, 2.5)],
            adjustments=[
                (lambda r, d: r.get("dividend_coverage") is not None and r["dividend_coverage"] >= 3.0, 0.5),
                (lambda r, d: r.get("dividend_coverage") is not None and r["dividend_coverage"] < 1.0, -0.5),
                (lambda r, d: r.get("equity_growth_proxy") is not None and r["equity_growth_proxy"] >= 0.50, 0.5),
                (lambda r, d: r.get("equity_growth_proxy") is not None and r["equity_growth_proxy"] < 0.20, -0.5),
            ],
            label="Equity Reinvestment",
        )

    def fixed_asset_efficiency_analysis(self, data: FinancialData) -> FixedAssetEfficiencyResult:
        """Phase 138: Fixed Asset Efficiency Analysis."""
        def _fa(d):
            return (d.total_assets - d.current_assets) if d.total_assets is not None and d.current_assets is not None else None

        return self._scored_analysis(
            data=data, result_class=FixedAssetEfficiencyResult,
            ratio_defs=[],
            derived=[
                ("fixed_asset_ratio", lambda r, d: safe_divide(_fa(d), d.total_assets)),
                ("fixed_asset_turnover", lambda r, d: safe_divide(d.revenue, _fa(d))),
                ("fixed_to_equity", lambda r, d: safe_divide(_fa(d), d.total_equity)),
                ("fixed_asset_coverage", lambda r, d: safe_divide(d.total_equity, _fa(d))),
                ("depreciation_to_fixed", lambda r, d: safe_divide(d.depreciation, _fa(d))),
                ("capex_to_fixed", lambda r, d: safe_divide(d.capex, _fa(d))),
            ],
            score_field="fae_score", grade_field="fae_grade",
            primary="fixed_asset_turnover", higher_is_better=True,
            thresholds=[(5.0, 10.0), (3.0, 8.5), (2.0, 7.0), (1.5, 5.5), (1.0, 4.0), (0.5, 2.5)],
            adjustments=[
                (lambda r, d: r.get("capex_to_fixed") is not None and r["capex_to_fixed"] >= 0.10, 0.5),
                (lambda r, d: r.get("capex_to_fixed") is not None and r["capex_to_fixed"] < 0.02, -0.5),
                (lambda r, d: r.get("fixed_asset_coverage") is not None and r["fixed_asset_coverage"] >= 1.0, 0.5),
                (lambda r, d: r.get("fixed_asset_coverage") is not None and r["fixed_asset_coverage"] < 0.3, -0.5),
            ],
            label="Fixed Asset Efficiency",
        )

    def income_stability_analysis(self, data: FinancialData) -> IncomeStabilityResult:
        """Phase 134: Income Stability Analysis."""
        return self._scored_analysis(
            data=data, result_class=IncomeStabilityResult,
            ratio_defs=[
                ("net_income_margin", "net_income", "revenue"),
                ("retained_earnings_ratio", "retained_earnings", "total_assets"),
                ("operating_income_cushion", "operating_income", "interest_expense"),
                ("net_to_gross_ratio", "net_income", "gross_profit"),
                ("ebitda_margin", "ebitda", "revenue"),
                ("income_resilience", "operating_cash_flow", "net_income"),
            ],
            score_field="is_score", grade_field="is_grade",
            primary="operating_income_cushion", higher_is_better=True,
            thresholds=[(10.0, 10.0), (7.0, 8.5), (5.0, 7.0), (3.0, 5.5), (2.0, 4.0), (1.0, 2.5)],
            adjustments=[
                (lambda r, d: r.get("net_income_margin") is not None and r["net_income_margin"] >= 0.15, 0.5),
                (lambda r, d: r.get("net_income_margin") is not None and r["net_income_margin"] < 0.02, -0.5),
                (lambda r, d: r.get("income_resilience") is not None and r["income_resilience"] >= 1.0, 0.5),
                (lambda r, d: r.get("income_resilience") is not None and r["income_resilience"] < 0.5, -0.5),
            ],
            label="Income Stability",
        )

    def defensive_posture_analysis(self, data: FinancialData) -> DefensivePostureResult:
        """Phase 133: Defensive Posture Analysis."""
        return self._scored_analysis(
            data=data, result_class=DefensivePostureResult,
            ratio_defs=[
                ("cash_ratio", "cash", "current_liabilities"),
                ("cash_flow_coverage", "operating_cash_flow", "total_assets"),
                ("equity_buffer", "total_equity", "total_assets"),
                ("debt_shield", "ebitda", "total_debt"),
            ],
            derived=[
                ("defensive_interval", lambda r, d: (
                    safe_divide(d.current_assets, d.operating_expenses / 365)
                    if d.operating_expenses is not None and d.operating_expenses > 0 else None
                )),
                ("quick_ratio", lambda r, d: (
                    (d.current_assets - (d.inventory or 0)) / d.current_liabilities
                    if d.current_assets is not None and d.current_liabilities is not None
                    and d.current_liabilities > 0 else None
                )),
            ],
            score_field="dp_score", grade_field="dp_grade",
            primary="defensive_interval", higher_is_better=True,
            thresholds=[(365, 10.0), (270, 8.5), (180, 7.0), (120, 5.5), (90, 4.0), (30, 2.5)],
            adjustments=[
                (lambda r, d: r.get("cash_ratio") is not None and r["cash_ratio"] >= 0.50, 0.5),
                (lambda r, d: r.get("cash_ratio") is not None and r["cash_ratio"] < 0.10, -0.5),
                (lambda r, d: r.get("debt_shield") is not None and r["debt_shield"] >= 2.0, 0.5),
                (lambda r, d: r.get("debt_shield") is not None and r["debt_shield"] < 0.5, -0.5),
            ],
            label="Defensive Posture",
        )

    def funding_efficiency_analysis(self, data: FinancialData) -> FundingEfficiencyResult:
        """Phase 131: Funding Efficiency Analysis."""
        return self._scored_analysis(
            data=data, result_class=FundingEfficiencyResult,
            ratio_defs=[
                ("equity_multiplier", "total_assets", "total_equity"),
                ("interest_coverage_ebitda", "ebitda", "interest_expense"),
                ("cost_of_debt", "interest_expense", "total_debt"),
            ],
            derived=[
                ("debt_to_capitalization", lambda r, d: (
                    d.total_debt / (d.total_debt + d.total_equity)
                    if d.total_debt is not None and d.total_equity is not None
                    and (d.total_debt + d.total_equity) > 0 else None
                )),
                ("weighted_funding_cost", lambda r, d: (
                    r["cost_of_debt"] * r["debt_to_capitalization"]
                    if r.get("cost_of_debt") is not None and r.get("debt_to_capitalization") is not None
                    else None
                )),
                ("funding_spread", lambda r, d: (
                    safe_divide(d.net_income, d.total_assets) - r["weighted_funding_cost"]
                    if safe_divide(d.net_income, d.total_assets) is not None
                    and r.get("weighted_funding_cost") is not None else None
                )),
            ],
            score_field="fe_score", grade_field="fe_grade",
            primary="interest_coverage_ebitda", higher_is_better=True,
            thresholds=[(10.0, 10.0), (7.0, 8.5), (5.0, 7.0), (3.0, 5.5), (2.0, 4.0), (1.0, 2.5)],
            adjustments=[
                (lambda r, d: r.get("debt_to_capitalization") is not None and r["debt_to_capitalization"] <= 0.30, 0.5),
                (lambda r, d: r.get("debt_to_capitalization") is not None and r["debt_to_capitalization"] > 0.60, -0.5),
                (lambda r, d: r.get("funding_spread") is not None and r["funding_spread"] >= 0.05, 0.5),
                (lambda r, d: r.get("funding_spread") is not None and r["funding_spread"] < 0.0, -0.5),
            ],
            label="Funding Efficiency",
        )

    def cash_flow_stability_analysis(self, data: FinancialData) -> CashFlowStabilityResult:
        """Phase 125: Cash Flow Stability Analysis."""
        return self._scored_analysis(
            data=data, result_class=CashFlowStabilityResult,
            ratio_defs=[
                ("ocf_margin", "operating_cash_flow", "revenue"),
                ("ocf_to_ebitda", "operating_cash_flow", "ebitda"),
                ("ocf_to_debt_service", "operating_cash_flow", "interest_expense"),
                ("capex_to_ocf", "capex", "operating_cash_flow"),
                ("dividend_coverage", "operating_cash_flow", "dividends_paid"),
            ],
            derived=[
                ("cash_flow_sufficiency", lambda r, d: (
                    safe_divide(d.operating_cash_flow,
                                (d.capex or 0.0) + (d.dividends_paid or 0.0) + (d.interest_expense or 0.0))
                    if d.operating_cash_flow is not None and d.operating_cash_flow > 0
                    and ((d.capex or 0.0) + (d.dividends_paid or 0.0) + (d.interest_expense or 0.0)) > 0
                    else None
                )),
            ],
            score_field="cfs_score", grade_field="cfs_grade",
            primary="ocf_margin", higher_is_better=True,
            thresholds=[(0.25, 10.0), (0.20, 8.5), (0.15, 7.0), (0.10, 5.5), (0.05, 4.0), (0.0, 2.5)],
            adjustments=[
                (lambda r, d: r.get("ocf_to_ebitda") is not None and r["ocf_to_ebitda"] >= 0.90, 0.5),
                (lambda r, d: r.get("ocf_to_ebitda") is not None and r["ocf_to_ebitda"] < 0.50, -0.5),
                (lambda r, d: r.get("cash_flow_sufficiency") is not None and r["cash_flow_sufficiency"] >= 1.5, 0.5),
                (lambda r, d: r.get("cash_flow_sufficiency") is not None and r["cash_flow_sufficiency"] < 1.0, -0.5),
            ],
            label="Cash Flow Stability",
        )

    def income_quality_analysis(self, data: FinancialData) -> IncomeQualityResult:
        """Phase 124: Income Quality Analysis."""
        return self._scored_analysis(
            data=data, result_class=IncomeQualityResult,
            ratio_defs=[
                ("ocf_to_net_income", "operating_cash_flow", "net_income"),
                ("cash_earnings_ratio", "operating_cash_flow", "ebitda"),
                ("non_cash_ratio", "depreciation", "net_income"),
                ("earnings_persistence", "operating_income", "revenue"),
                ("operating_income_ratio", "operating_income", "net_income"),
            ],
            derived=[
                ("accruals_ratio", lambda r, d: (
                    safe_divide((d.net_income or 0) - (d.operating_cash_flow or 0), d.total_assets)
                    if d.net_income is not None and d.net_income > 0
                    and d.operating_cash_flow is not None
                    and d.total_assets is not None and d.total_assets > 0 else None
                )),
            ],
            score_field="iq_score", grade_field="iq_grade",
            primary="ocf_to_net_income", higher_is_better=True,
            thresholds=[(1.5, 10.0), (1.2, 8.5), (1.0, 7.0), (0.8, 5.5), (0.5, 4.0), (0.0, 2.5)],
            adjustments=[
                (lambda r, d: r.get("accruals_ratio") is not None and r["accruals_ratio"] <= -0.05, 0.5),
                (lambda r, d: r.get("accruals_ratio") is not None and r["accruals_ratio"] > 0.10, -0.5),
                (lambda r, d: r.get("cash_earnings_ratio") is not None and r["cash_earnings_ratio"] >= 0.90, 0.5),
                (lambda r, d: r.get("cash_earnings_ratio") is not None and r["cash_earnings_ratio"] < 0.50, -0.5),
            ],
            label="Income Quality",
        )

    def receivables_management_analysis(self, data: FinancialData) -> ReceivablesManagementResult:
        """Phase 114: Receivables Management Analysis."""
        result = ReceivablesManagementResult()
        if not data.revenue or not data.accounts_receivable:
            return result
        return self._scored_analysis(
            data=data, result_class=ReceivablesManagementResult,
            ratio_defs=[
                ("ar_to_revenue", "accounts_receivable", "revenue"),
                ("ar_to_current_assets", "accounts_receivable", "current_assets"),
                ("receivables_turnover", "revenue", "accounts_receivable"),
            ],
            derived=[
                ("dso", lambda r, d: safe_divide(d.accounts_receivable * 365, d.revenue)
                    if d.accounts_receivable is not None and d.revenue is not None else None),
                ("collection_effectiveness", lambda r, d: safe_divide(d.revenue - d.accounts_receivable, d.revenue)
                    if d.revenue is not None and d.accounts_receivable is not None else None),
                ("ar_concentration", lambda r, d: (
                    safe_divide(d.accounts_receivable, d.accounts_receivable + (d.cash or 0))
                    if d.accounts_receivable is not None and (d.accounts_receivable + (d.cash or 0)) > 0 else None
                )),
            ],
            score_field="rm_score", grade_field="rm_grade",
            primary="dso", higher_is_better=False,
            thresholds=[(30, 10.0), (45, 8.5), (60, 7.0), (75, 5.5), (90, 4.0), (120, 2.5)],
            adjustments=[
                (lambda r, d: r.get("receivables_turnover") is not None and r["receivables_turnover"] >= 12, 0.5),
                (lambda r, d: r.get("receivables_turnover") is not None and r["receivables_turnover"] < 4, -0.5),
                (lambda r, d: r.get("collection_effectiveness") is not None and r["collection_effectiveness"] >= 0.90, 0.5),
                (lambda r, d: r.get("collection_effectiveness") is not None and r["collection_effectiveness"] < 0.70, -0.5),
            ],
            label="Receivables Management",
        )

    def solvency_depth_analysis(self, data: FinancialData) -> SolvencyDepthResult:
        """Phase 109: Solvency Depth Analysis."""
        result = SolvencyDepthResult()
        td = data.total_debt or 0
        te = data.total_equity or 0
        ta = data.total_assets or 0
        ebit = data.ebit or 0
        ie = data.interest_expense or 0
        ebitda = data.ebitda or 0

        if ta == 0 and te == 0:
            return result

        # Debt-to-Equity = TD / TE
        result.debt_to_equity = safe_divide(td, te) if te > 0 else None

        # Debt-to-Assets = TD / TA
        result.debt_to_assets = safe_divide(td, ta) if ta > 0 else None

        # Equity-to-Assets = TE / TA
        result.equity_to_assets = safe_divide(te, ta) if ta > 0 else None

        # Interest Coverage = EBIT / IE
        result.interest_coverage_ratio = safe_divide(ebit, ie) if ie > 0 else None

        # Debt-to-EBITDA = TD / EBITDA
        result.debt_to_ebitda = safe_divide(td, ebitda) if ebitda > 0 else None

        # Financial Leverage = TA / TE
        result.financial_leverage = safe_divide(ta, te) if te > 0 else None

        # Scoring based on Debt-to-EBITDA
        de = result.debt_to_ebitda
        if de is not None:
            if de <= 1.0:
                score = 10.0
            elif de <= 2.0:
                score = 8.5
            elif de <= 3.0:
                score = 7.0
            elif de <= 4.0:
                score = 5.5
            elif de <= 5.0:
                score = 4.0
            elif de <= 6.0:
                score = 2.5
            else:
                score = 1.0
        else:
            # No EBITDA, fall back to D/A
            da = result.debt_to_assets
            if da is not None:
                if da <= 0.30:
                    score = 8.0
                elif da <= 0.50:
                    score = 5.0
                else:
                    score = 2.0
            else:
                score = 0.0

        # Adjustments
        dte = result.debt_to_equity
        if dte is not None:
            if dte <= 1.0:
                score += 0.5
            elif dte > 3.0:
                score -= 0.5

        ic = result.interest_coverage_ratio
        if ic is not None:
            if ic >= 5.0:
                score += 0.5
            elif ic < 2.0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.sd_score = round(score, 2)

        if score >= 8:
            result.sd_grade = "Excellent"
        elif score >= 6:
            result.sd_grade = "Good"
        elif score >= 4:
            result.sd_grade = "Adequate"
        else:
            result.sd_grade = "Weak"

        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Solvency Depth: {result.sd_grade} ({result.sd_score}/10). "
            f"D/EBITDA={_r2(result.debt_to_ebitda)}, "
            f"D/E={_r2(result.debt_to_equity)}, "
            f"IC={_r2(result.interest_coverage_ratio)}."
        )

        return result

    def operational_leverage_depth_analysis(self, data: FinancialData) -> OperationalLeverageDepthResult:
        """Phase 105: Operational Leverage Depth Analysis."""
        result = OperationalLeverageDepthResult()
        rev = data.revenue or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        oi = data.operating_income or 0
        gp = data.gross_profit or 0
        da = data.depreciation or 0

        if rev <= 0:
            return result

        total_costs = cogs + opex
        if total_costs <= 0:
            return result

        # Fixed Cost Ratio = (OpEx + D&A) / Total Costs (proxy: opex as fixed)
        fixed_proxy = opex + da
        result.fixed_cost_ratio = safe_divide(fixed_proxy, total_costs)

        # Variable Cost Ratio = COGS / Total Costs
        result.variable_cost_ratio = safe_divide(cogs, total_costs)

        # Contribution Margin = (Rev - COGS) / Rev = GP / Rev
        result.contribution_margin = safe_divide(gp, rev) if gp else safe_divide(rev - cogs, rev)

        # DOL Proxy = GP / OI (if both available)
        result.dol_proxy = safe_divide(gp, oi) if oi and oi != 0 else None

        # Breakeven Coverage = Rev / (Rev - OI) = Rev / Total Costs
        result.breakeven_coverage = safe_divide(rev, total_costs) if total_costs > 0 else None

        # Cost Flexibility = Variable Costs / Total Costs (higher = more flexible)
        result.cost_flexibility = safe_divide(cogs, total_costs)

        # Scoring based on contribution margin
        cm = result.contribution_margin
        if cm is not None:
            if cm >= 0.60:
                score = 10.0
            elif cm >= 0.50:
                score = 8.5
            elif cm >= 0.40:
                score = 7.0
            elif cm >= 0.30:
                score = 5.5
            elif cm >= 0.20:
                score = 4.0
            elif cm >= 0.10:
                score = 2.5
            else:
                score = 1.0
        else:
            score = 3.0

        # Adjustments
        bc = result.breakeven_coverage
        if bc is not None:
            if bc >= 1.5:
                score += 0.5
            elif bc < 1.0:
                score -= 0.5

        cf = result.cost_flexibility
        if cf is not None:
            if cf >= 0.60:
                score += 0.5
            elif cf < 0.30:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.old_score = round(score, 2)

        if score >= 8:
            result.old_grade = "Excellent"
        elif score >= 6:
            result.old_grade = "Good"
        elif score >= 4:
            result.old_grade = "Adequate"
        else:
            result.old_grade = "Weak"

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Operational Leverage Depth: {result.old_grade} ({result.old_score}/10). "
            f"Contribution margin={_pct(result.contribution_margin)}, "
            f"Breakeven coverage={_r2(result.breakeven_coverage)}, "
            f"Cost flexibility={_pct(result.cost_flexibility)}."
        )

        return result

    def profitability_depth_analysis(self, data: FinancialData) -> ProfitabilityDepthResult:
        """Phase 103: Profitability Depth Analysis."""
        result = ProfitabilityDepthResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        gp = data.gross_profit or 0
        cogs = data.cogs or 0
        oi = data.operating_income or 0
        ebitda = data.ebitda or 0
        ni = data.net_income or 0

        # 1. Gross Margin
        gp_calc = gp if gp > 0 else (rev - cogs) if cogs > 0 else None
        if gp_calc is not None:
            result.gross_margin = round(safe_divide(gp_calc, rev), 4)

        # 2. Operating Margin
        if oi != 0 or (data.operating_income is not None):
            result.operating_margin = round(safe_divide(oi, rev), 4)

        # 3. EBITDA Margin
        if ebitda != 0 or (data.ebitda is not None):
            result.ebitda_margin = round(safe_divide(ebitda, rev), 4)

        # 4. Net Margin
        result.net_margin = round(safe_divide(ni, rev), 4)

        # 5. Margin Spread = Gross Margin - Net Margin (cost/tax leakage)
        if result.gross_margin is not None and result.net_margin is not None:
            result.margin_spread = round(result.gross_margin - result.net_margin, 4)

        # 6. Profit Retention Ratio = NI / GP (what fraction of gross profit survives)
        if gp_calc is not None and gp_calc > 0:
            result.profit_retention_ratio = round(safe_divide(ni, gp_calc), 4)

        # Scoring: based on Operating Margin (core profitability)
        om = result.operating_margin
        if om is not None:
            if om >= 0.25:
                score = 10.0
            elif om >= 0.20:
                score = 8.5
            elif om >= 0.15:
                score = 7.0
            elif om >= 0.10:
                score = 5.5
            elif om >= 0.05:
                score = 4.0
            elif om >= 0.0:
                score = 2.5
            else:
                score = 1.0
        elif result.net_margin is not None:
            nm = result.net_margin
            if nm >= 0.15:
                score = 8.0
            elif nm >= 0.05:
                score = 5.0
            else:
                score = 2.0
        else:
            return result

        # Adjustment: Gross Margin
        gm = result.gross_margin
        if gm is not None:
            if gm >= 0.50:
                score += 0.5
            elif gm < 0.20:
                score -= 0.5

        # Adjustment: Profit Retention
        prr = result.profit_retention_ratio
        if prr is not None:
            if prr >= 0.40:
                score += 0.5
            elif prr < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.pd_score = round(score, 2)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.pd_grade = grade

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        result.summary = (
            f"Profitability Depth: OM={_pct(result.operating_margin)}, "
            f"NM={_pct(result.net_margin)}, GM={_pct(result.gross_margin)}. "
            f"Grade: {grade} ({result.pd_score}/10)."
        )

        return result

    def revenue_efficiency_analysis(self, data: FinancialData) -> RevenueEfficiencyResult:
        """Phase 102: Revenue Efficiency Analysis."""
        result = RevenueEfficiencyResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        ta = data.total_assets or 0
        te = data.total_equity or 0
        gp = data.gross_profit or 0
        oi = data.operating_income or 0
        ni = data.net_income or 0
        ocf = data.operating_cash_flow or 0
        cogs = data.cogs or 0

        # 1. Revenue per Asset = Rev / TA
        if ta > 0:
            result.revenue_per_asset = round(safe_divide(rev, ta), 4)

        # 2. Cash Conversion Efficiency = OCF / Rev
        result.cash_conversion_efficiency = round(safe_divide(ocf, rev), 4)

        # 3. Gross Margin Efficiency = GP / Rev
        if gp > 0 or cogs > 0:
            gp_calc = gp if gp > 0 else (rev - cogs)
            result.gross_margin_efficiency = round(safe_divide(gp_calc, rev), 4)

        # 4. Operating Leverage Ratio = OI / GP
        if gp > 0:
            result.operating_leverage_ratio = round(safe_divide(oi, gp), 4)
        elif cogs > 0:
            gp_calc = rev - cogs
            if gp_calc > 0:
                result.operating_leverage_ratio = round(safe_divide(oi, gp_calc), 4)

        # 5. Revenue to Equity = Rev / TE
        if te > 0:
            result.revenue_to_equity = round(safe_divide(rev, te), 4)

        # 6. Net Revenue Retention = NI / Rev (net margin as efficiency proxy)
        result.net_revenue_retention = round(safe_divide(ni, rev), 4)

        # Scoring: based on Cash Conversion Efficiency (higher = better)
        cce = result.cash_conversion_efficiency
        if cce is not None:
            if cce >= 0.25:
                score = 10.0
            elif cce >= 0.20:
                score = 8.5
            elif cce >= 0.15:
                score = 7.0
            elif cce >= 0.10:
                score = 5.5
            elif cce >= 0.05:
                score = 4.0
            elif cce >= 0.0:
                score = 2.5
            else:
                score = 1.0
        else:
            return result

        # Adjustment: Gross Margin Efficiency
        gme = result.gross_margin_efficiency
        if gme is not None:
            if gme >= 0.50:
                score += 0.5
            elif gme < 0.20:
                score -= 0.5

        # Adjustment: Operating Leverage
        olr = result.operating_leverage_ratio
        if olr is not None:
            if olr >= 0.50:
                score += 0.5
            elif olr < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.rev_eff_score = round(score, 2)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.rev_eff_grade = grade

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        result.summary = (
            f"Revenue Efficiency: CCE={_pct(result.cash_conversion_efficiency)}, "
            f"GM={_pct(result.gross_margin_efficiency)}. "
            f"Grade: {grade} ({result.rev_eff_score}/10)."
        )

        return result

    def debt_composition_analysis(self, data: FinancialData) -> DebtCompositionResult:
        """Phase 101: Debt Composition Analysis."""
        result = DebtCompositionResult()
        td = data.total_debt or 0
        te = data.total_equity or 0
        ta = data.total_assets or 0
        ie = abs(data.interest_expense or 0)
        ebit = data.ebit or 0
        ni = data.net_income or 0
        ocf = data.operating_cash_flow or 0
        tl = data.total_liabilities or 0

        if td <= 0 and tl <= 0:
            # No debt => perfect score
            result.dco_score = 10.0
            result.dco_grade = "Excellent"
            result.summary = "Debt Composition: No debt — Excellent (10.0/10)."
            return result

        # 1. Debt-to-Equity = TD / TE
        if te > 0:
            result.debt_to_equity = round(safe_divide(td, te), 4)

        # 2. Debt-to-Assets = TD / TA
        if ta > 0:
            result.debt_to_assets = round(safe_divide(td, ta), 4)

        # 3. Long-term Debt Ratio = (TD - CL_portion) / TD approximated as (TL - CL) / TL
        cl = data.current_liabilities or 0
        if tl > 0:
            long_term = max(0, tl - cl)
            result.long_term_debt_ratio = round(safe_divide(long_term, tl), 4)

        # 4. Interest Burden = IE / EBIT
        if ebit > 0 and ie > 0:
            result.interest_burden = round(safe_divide(ie, ebit), 4)

        # 5. Debt Cost Ratio = IE / TD
        if td > 0 and ie > 0:
            result.debt_cost_ratio = round(safe_divide(ie, td), 4)

        # 6. Debt Coverage Margin = (OCF - IE) / TD
        if td > 0:
            result.debt_coverage_margin = round(safe_divide(ocf - ie, td), 4)

        # Scoring: based on Debt-to-Equity (lower = better)
        de = result.debt_to_equity
        if de is not None:
            if de <= 0.30:
                score = 10.0
            elif de <= 0.50:
                score = 8.5
            elif de <= 1.00:
                score = 7.0
            elif de <= 1.50:
                score = 5.5
            elif de <= 2.00:
                score = 4.0
            elif de <= 3.00:
                score = 2.5
            else:
                score = 1.0
        elif result.debt_to_assets is not None:
            da = result.debt_to_assets
            if da <= 0.20:
                score = 10.0
            elif da <= 0.40:
                score = 7.0
            elif da <= 0.60:
                score = 4.0
            else:
                score = 1.0
        else:
            return result

        # Adjustment: Interest Burden
        ib = result.interest_burden
        if ib is not None:
            if ib <= 0.10:
                score += 0.5
            elif ib > 0.50:
                score -= 0.5

        # Adjustment: Debt Coverage Margin
        dcm = result.debt_coverage_margin
        if dcm is not None:
            if dcm >= 0.30:
                score += 0.5
            elif dcm < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.dco_score = round(score, 2)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.dco_grade = grade

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        de_str = f"{result.debt_to_equity:.2f}x" if result.debt_to_equity is not None else "N/A"
        result.summary = (
            f"Debt Composition: D/E={de_str}, "
            f"D/A={_pct(result.debt_to_assets)}. "
            f"Grade: {grade} ({result.dco_score}/10)."
        )

        return result

    def operational_risk_analysis(self, data: FinancialData) -> OperationalRiskResult:
        """Phase 92: Operational Risk Analysis."""
        result = OperationalRiskResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        gp = data.gross_profit or 0
        oi = data.operating_income or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        ocf = data.operating_cash_flow or 0
        ni = data.net_income or 0
        ta = data.total_assets or 0

        # Operating leverage: GP / OI (how much GP translates to OI)
        result.operating_leverage = safe_divide(gp, oi) if oi != 0 else None

        # Cost rigidity: fixed costs (opex) as proportion of total costs
        total_costs = cogs + opex
        result.cost_rigidity = safe_divide(opex, total_costs) if total_costs > 0 else None

        # Breakeven ratio: total costs / revenue
        result.breakeven_ratio = safe_divide(total_costs, rev)

        # Margin of safety: (Rev - breakeven) / Rev ≈ 1 - breakeven_ratio
        be = result.breakeven_ratio
        if be is not None:
            result.margin_of_safety = 1.0 - be

        # Cash burn ratio: OCF / total costs (how well cash covers costs)
        result.cash_burn_ratio = safe_divide(ocf, total_costs) if total_costs > 0 else None

        # Risk buffer: (OCF + cash) / opex
        cash = data.cash or 0
        result.risk_buffer = safe_divide(ocf + cash, opex) if opex > 0 else None

        # Score on margin_of_safety (higher = less risk)
        mos = result.margin_of_safety
        if mos is not None:
            if mos >= 0.30:
                score = 10.0
            elif mos >= 0.25:
                score = 8.5
            elif mos >= 0.20:
                score = 7.0
            elif mos >= 0.15:
                score = 5.5
            elif mos >= 0.10:
                score = 4.0
            elif mos >= 0.05:
                score = 2.5
            else:
                score = 1.0

            rb = result.risk_buffer
            if rb is not None:
                if rb >= 2.0:
                    score += 0.5
                elif rb < 0.5:
                    score -= 0.5

            cr = result.cost_rigidity
            if cr is not None:
                if cr <= 0.20:
                    score += 0.5
                elif cr > 0.50:
                    score -= 0.5

            score = max(0.0, min(10.0, score))
            result.or_score = round(score, 2)

            if score >= 8:
                result.or_grade = "Excellent"
            elif score >= 6:
                result.or_grade = "Good"
            elif score >= 4:
                result.or_grade = "Adequate"
            else:
                result.or_grade = "Weak"

            def _pct(v):
                return f"{v:.1%}" if v is not None else "N/A"

            result.summary = (
                f"Operational Risk: Margin of safety {_pct(mos)}, "
                f"risk buffer {result.risk_buffer:.2f}x — {result.or_grade} ({result.or_score}/10)."
                if result.risk_buffer is not None else
                f"Operational Risk: Margin of safety {_pct(mos)} — {result.or_grade} ({result.or_score}/10)."
            )

        return result

    def financial_health_score_analysis(self, data: FinancialData) -> FinancialHealthScoreResult:
        """Phase 90: Financial Health Score Analysis — composite of 5 pillars."""
        result = FinancialHealthScoreResult()
        rev = data.revenue or 0
        ta = data.total_assets or 0
        if rev <= 0 and ta <= 0:
            return result

        # Profitability: net_margin
        ni = data.net_income or 0
        if rev > 0:
            nm = ni / rev
            if nm >= 0.15:
                result.profitability_score = 10.0
            elif nm >= 0.10:
                result.profitability_score = 8.0
            elif nm >= 0.05:
                result.profitability_score = 6.0
            elif nm >= 0.0:
                result.profitability_score = 4.0
            else:
                result.profitability_score = 2.0

        # Liquidity: current_ratio
        ca = data.current_assets or 0
        cl = data.current_liabilities or 0
        if cl > 0:
            cr = ca / cl
            if cr >= 2.0:
                result.liquidity_score = 10.0
            elif cr >= 1.5:
                result.liquidity_score = 8.0
            elif cr >= 1.0:
                result.liquidity_score = 6.0
            elif cr >= 0.5:
                result.liquidity_score = 4.0
            else:
                result.liquidity_score = 2.0

        # Solvency: debt_to_equity
        te = data.total_equity or 0
        td = data.total_debt or 0
        if te > 0:
            de = td / te
            if de <= 0.3:
                result.solvency_score = 10.0
            elif de <= 0.6:
                result.solvency_score = 8.0
            elif de <= 1.0:
                result.solvency_score = 6.0
            elif de <= 2.0:
                result.solvency_score = 4.0
            else:
                result.solvency_score = 2.0

        # Efficiency: asset_turnover
        if ta > 0 and rev > 0:
            at = safe_divide(rev, ta, default=0.0)
            if at >= 1.5:
                result.efficiency_score = 10.0
            elif at >= 1.0:
                result.efficiency_score = 8.0
            elif at >= 0.5:
                result.efficiency_score = 6.0
            elif at >= 0.2:
                result.efficiency_score = 4.0
            else:
                result.efficiency_score = 2.0

        # Coverage: interest_coverage
        ebit = data.ebit or 0
        ie = data.interest_expense or 0
        if ie > 0:
            ic = safe_divide(ebit, ie, default=0.0)
            if ic >= 8.0:
                result.coverage_score = 10.0
            elif ic >= 5.0:
                result.coverage_score = 8.0
            elif ic >= 3.0:
                result.coverage_score = 6.0
            elif ic >= 1.5:
                result.coverage_score = 4.0
            else:
                result.coverage_score = 2.0

        # Composite: weighted average of available pillars
        pillars = [
            (result.profitability_score, 0.25),
            (result.liquidity_score, 0.20),
            (result.solvency_score, 0.20),
            (result.efficiency_score, 0.15),
            (result.coverage_score, 0.20),
        ]
        available = [(s, w) for s, w in pillars if s is not None]
        if available:
            total_weight = sum(w for _, w in available)
            composite = safe_divide(sum(s * w for s, w in available), total_weight, default=5.0)
            result.composite_score = round(composite, 2)
            result.fh_score = result.composite_score

            if result.fh_score >= 8:
                result.fh_grade = "Excellent"
            elif result.fh_score >= 6:
                result.fh_grade = "Good"
            elif result.fh_score >= 4:
                result.fh_grade = "Adequate"
            else:
                result.fh_grade = "Weak"

            result.summary = (
                f"Financial Health: Composite {result.composite_score}/10 "
                f"across {len(available)} pillars — {result.fh_grade}."
            )

        return result

    def asset_quality_analysis(self, data: FinancialData) -> AssetQualityResult:
        """Phase 86: Asset Quality Analysis.

        Evaluates the composition and quality of the company's asset base,
        focusing on tangibility, liquidity of assets, and concentration risks.
        """
        result = AssetQualityResult()
        ta = data.total_assets or 0
        if ta <= 0:
            return result

        ca = data.current_assets or 0
        cash = data.cash or 0
        ar = data.accounts_receivable or 0
        inv = data.inventory or 0

        # Tangible asset ratio (assume no separate intangibles field; approx as 1.0)
        # Fixed assets approximated as TA - CA
        fixed_assets = max(0, ta - ca)

        result.tangible_asset_ratio = 1.0  # No intangibles data; assume all tangible
        result.fixed_asset_ratio = safe_divide(fixed_assets, ta)
        result.current_asset_ratio = safe_divide(ca, ta)
        result.cash_to_current_assets = safe_divide(cash, ca) if ca > 0 else None
        result.receivables_to_assets = safe_divide(ar, ta)
        result.inventory_to_assets = safe_divide(inv, ta)

        # Scoring: current_asset_ratio based (higher = more liquid asset base)
        car = result.current_asset_ratio
        if car is None:
            return result

        if car >= 0.60:
            score = 10.0
        elif car >= 0.50:
            score = 8.5
        elif car >= 0.40:
            score = 7.0
        elif car >= 0.30:
            score = 5.5
        elif car >= 0.20:
            score = 4.0
        elif car >= 0.10:
            score = 2.5
        else:
            score = 1.0

        # Adj: cash_to_current_assets
        cca = result.cash_to_current_assets
        if cca is not None:
            if cca >= 0.30:
                score += 0.5
            elif cca < 0.05:
                score -= 0.5

        # Adj: receivables concentration
        rta = result.receivables_to_assets
        if rta is not None:
            if rta > 0.40:
                score -= 0.5  # High AR concentration risk
            elif rta <= 0.10:
                score += 0.5  # Low collection risk

        score = max(0.0, min(10.0, score))
        result.aq_score = round(score, 1)

        if score >= 8:
            result.aq_grade = "Excellent"
        elif score >= 6:
            result.aq_grade = "Good"
        elif score >= 4:
            result.aq_grade = "Adequate"
        else:
            result.aq_grade = "Weak"

        result.summary = (
            f"Asset Quality: Current Asset Ratio={car:.1%}, "
            f"Cash/CA={cca:.1%}" if cca is not None else f"Asset Quality: Current Asset Ratio={car:.1%}, Cash/CA=N/A"
        )
        result.summary += (
            f", AR/TA={rta:.1%}" if rta is not None else ", AR/TA=N/A"
        )
        result.summary += f" — {result.aq_grade} ({result.aq_score}/10)"
        return result

    def financial_resilience_analysis(self, data: FinancialData) -> FinancialResilienceResult:
        """Phase 82: Financial Resilience Analysis.

        Measures a company's ability to withstand financial shocks through
        cash reserves, operating cash flow coverage, and buffer metrics.
        """
        result = FinancialResilienceResult()

        cash = data.cash or 0
        ta = data.total_assets or 0
        td = data.total_debt or 0
        tl = data.total_liabilities or 0
        ocf = data.operating_cash_flow or 0
        ie = data.interest_expense or 0
        rev = data.revenue or 0
        capex = data.capex or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0

        if ta <= 0:
            return result

        # Core metrics
        result.cash_to_assets = safe_divide(cash, ta)

        result.cash_to_debt = safe_divide(cash, td) if td and td > 0 else None

        result.operating_cash_coverage = safe_divide(ocf, tl) if tl and tl > 0 else None

        result.interest_coverage_cash = safe_divide(ocf, ie) if ie and ie > 0 else None

        fcf = ocf - capex
        result.free_cash_margin = safe_divide(fcf, rev) if rev and rev > 0 else None

        # Resilience buffer = (Cash + OCF) / Annual OpEx
        annual_opex = cogs + opex
        if annual_opex > 0:
            result.resilience_buffer = safe_divide(cash + ocf, annual_opex)

        # Scoring based on resilience_buffer
        rb = result.resilience_buffer
        if rb is not None:
            if rb >= 1.0:
                base = 10.0
            elif rb >= 0.75:
                base = 8.5
            elif rb >= 0.50:
                base = 7.0
            elif rb >= 0.35:
                base = 5.5
            elif rb >= 0.20:
                base = 4.0
            elif rb >= 0.10:
                base = 2.5
            else:
                base = 1.0
        elif result.cash_to_assets is not None:
            # Fallback scoring on cash_to_assets
            ca = result.cash_to_assets
            if ca >= 0.30:
                base = 8.5
            elif ca >= 0.15:
                base = 5.5
            else:
                base = 2.5
        else:
            return result

        adj = 0.0

        # Cash-to-debt adjustment
        cd = result.cash_to_debt
        if cd is not None:
            if cd >= 1.0:
                adj += 0.5
            elif cd < 0.10:
                adj -= 0.5

        # Interest coverage cash adjustment
        icc = result.interest_coverage_cash
        if icc is not None:
            if icc >= 5.0:
                adj += 0.5
            elif icc < 1.5:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.fr_score = score
        result.fr_grade = (
            "Excellent" if score >= 8.0
            else "Good" if score >= 6.0
            else "Adequate" if score >= 4.0
            else "Weak"
        )

        grade = result.fr_grade
        result.summary = (
            f"Financial Resilience Analysis: "
            f"Resilience Buffer={rb:.2f}x. " if rb is not None else "Financial Resilience Analysis: "
        ) + (
            f"Cash/Assets={result.cash_to_assets:.1%}. "
            f"Score: {result.fr_score:.1f}/10. Grade: {grade}."
        )

        return result

    def equity_multiplier_analysis(self, data: FinancialData) -> EquityMultiplierResult:
        """Phase 81: Equity Multiplier Analysis.

        Measures financial leverage through equity multiplier and related
        leverage decomposition metrics including DuPont ROE.
        """
        result = EquityMultiplierResult()

        ta = data.total_assets or 0
        te = data.total_equity or 0
        tl = data.total_liabilities or 0
        ni = data.net_income or 0
        rev = data.revenue or 0
        ie = data.interest_expense or 0
        td = data.total_debt or 0

        if te <= 0 or ta <= 0:
            return result

        # Core metrics
        em = safe_divide(ta, te)
        result.equity_multiplier = em

        result.debt_ratio = safe_divide(tl, ta)
        result.equity_ratio = safe_divide(te, ta)

        # Financial leverage index = ROE / ROA
        roe = safe_divide(ni, te)
        roa = safe_divide(ni, ta)
        result.financial_leverage_index = safe_divide(roe, roa) if roa and roa > 0 else None

        # DuPont ROE = Net Margin * Asset Turnover * Equity Multiplier
        if rev > 0 and em is not None:
            net_margin = safe_divide(ni, rev)
            asset_turnover = safe_divide(rev, ta)
            if net_margin is not None and asset_turnover is not None:
                result.dupont_roe = net_margin * asset_turnover * em

        # Leverage spread = ROA - Cost of Debt
        if roa is not None and td and td > 0 and ie is not None:
            cost_of_debt = safe_divide(ie, td)
            if cost_of_debt is not None:
                result.leverage_spread = roa - cost_of_debt

        # Scoring based on equity_multiplier
        # Lower EM = less leverage = safer
        if em is not None:
            if em <= 1.5:
                base = 10.0
            elif em <= 2.0:
                base = 8.5
            elif em <= 2.5:
                base = 7.0
            elif em <= 3.0:
                base = 5.5
            elif em <= 4.0:
                base = 4.0
            elif em <= 5.0:
                base = 2.5
            else:
                base = 1.0

            adj = 0.0

            # Debt ratio adjustment
            dr = result.debt_ratio
            if dr is not None:
                if dr <= 0.30:
                    adj += 0.5
                elif dr > 0.70:
                    adj -= 0.5

            # Leverage spread adjustment
            ls = result.leverage_spread
            if ls is not None:
                if ls > 0.05:
                    adj += 0.5
                elif ls < -0.05:
                    adj -= 0.5

            score = max(0.0, min(10.0, base + adj))
            result.em_score = score
            result.em_grade = (
                "Excellent" if score >= 8.0
                else "Good" if score >= 6.0
                else "Adequate" if score >= 4.0
                else "Weak"
            )

        grade = result.em_grade or "N/A"
        result.summary = (
            f"Equity Multiplier Analysis: EM={em if em is not None else 'N/A':.2f}. "
            f"Debt Ratio={result.debt_ratio if result.debt_ratio is not None else 'N/A'}. "
            f"Score: {result.em_score:.1f}/10. Grade: {grade}."
        ) if em is not None else "Equity Multiplier Analysis: Insufficient data."

        return result

    def defensive_interval_analysis(self, data: FinancialData) -> DefensiveIntervalResult:
        """Phase 80: Defensive Interval Analysis.

        Measures how long a company can operate from liquid assets
        without additional revenue.
        """
        result = DefensiveIntervalResult()
        cash = data.cash or 0
        ar = data.accounts_receivable or 0
        ta = data.total_assets or 0
        cl = data.current_liabilities or 0

        # Liquid assets = Cash + AR (conservative; no short-term investments field)
        liquid = cash + ar

        # Daily operating expenses
        # Use (COGS + OpEx) as annual operating expenses
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        annual_opex = cogs + opex
        if annual_opex <= 0:
            return result

        daily_opex = annual_opex / 365.0

        result.defensive_interval_days = round(liquid / daily_opex, 1) if daily_opex > 0 else None
        result.cash_interval_days = round(cash / daily_opex, 1) if daily_opex > 0 else None
        result.liquid_assets_ratio = safe_divide(liquid, ta)
        result.days_cash_on_hand = round(cash / daily_opex, 1) if daily_opex > 0 else None
        result.liquid_reserve_adequacy = safe_divide(liquid, cl)
        result.operating_expense_coverage = safe_divide(liquid, annual_opex)

        # --- scoring based on defensive_interval_days ---
        did = result.defensive_interval_days
        if did is None:
            return result

        if did >= 180:
            score = 10.0
        elif did >= 120:
            score = 8.5
        elif did >= 90:
            score = 7.0
        elif did >= 60:
            score = 5.5
        elif did >= 30:
            score = 4.0
        elif did >= 15:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: liquid reserve adequacy
        lra = result.liquid_reserve_adequacy
        if lra is not None:
            if lra >= 2.0:
                score += 0.5
            elif lra < 0.5:
                score -= 0.5

        # Adjustment: liquid assets ratio
        lar = result.liquid_assets_ratio
        if lar is not None:
            if lar >= 0.30:
                score += 0.5
            elif lar < 0.05:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.di_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.di_grade = grade

        result.summary = (
            f"Defensive Interval: {did:.0f} days of operating coverage. "
            f"Score: {result.di_score}/10. Grade: {grade}."
        )

        return result

    def cash_burn_analysis(self, data: FinancialData) -> CashBurnResult:
        """Phase 79: Cash Burn Analysis.

        Evaluates cash generation vs consumption patterns and
        the company's cash self-sufficiency.
        """
        result = CashBurnResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        ocf = data.operating_cash_flow or 0
        capex = data.capex or 0
        div = data.dividends_paid or 0
        cash = data.cash or 0
        td = data.total_debt or 0

        result.ocf_margin = safe_divide(ocf, rev)
        result.capex_intensity = safe_divide(capex, rev)
        fcf = ocf - capex
        result.fcf_margin = safe_divide(fcf, rev)
        total_outflows = capex + div
        result.cash_self_sufficiency = safe_divide(ocf, total_outflows) if total_outflows > 0 else None
        result.net_cash_position = cash - td

        # Runway: only meaningful if FCF is negative (burning cash)
        if fcf < 0 and cash > 0:
            monthly_burn = abs(fcf) / 12.0
            result.cash_runway_months = round(cash / monthly_burn, 1) if monthly_burn > 0 else None

        # --- scoring based on FCF margin ---
        fm = result.fcf_margin
        if fm is None:
            return result

        if fm >= 0.20:
            score = 10.0
        elif fm >= 0.15:
            score = 8.5
        elif fm >= 0.10:
            score = 7.0
        elif fm >= 0.05:
            score = 5.5
        elif fm >= 0.0:
            score = 4.0
        elif fm >= -0.10:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: cash self-sufficiency
        css = result.cash_self_sufficiency
        if css is not None:
            if css >= 2.0:
                score += 0.5
            elif css < 0.5:
                score -= 0.5

        # Adjustment: net cash position
        ncp = result.net_cash_position
        if ncp is not None:
            if ncp > 0:
                score += 0.5
            elif ncp < -rev * 0.5:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cb_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.cb_grade = grade

        result.summary = (
            f"Cash Burn: FCF margin {fm:.1%}, OCF margin {result.ocf_margin:.1%}. "
            f"Score: {result.cb_score}/10. Grade: {grade}."
        )

        return result

    def profit_retention_analysis(self, data: FinancialData) -> ProfitRetentionResult:
        """Phase 78: Profit Retention Analysis.

        Measures how much profit a company retains versus distributes,
        and whether retained earnings drive sustainable growth.
        """
        result = ProfitRetentionResult()
        ni = data.net_income or 0
        if ni <= 0:
            return result

        div = data.dividends_paid or 0
        te = data.total_equity or 0
        ta = data.total_assets or 0
        ret_earn = data.retained_earnings or 0

        plowback = ni - div
        result.plowback_amount = plowback
        result.retention_ratio = safe_divide(plowback, ni)
        result.payout_ratio = safe_divide(div, ni)
        result.re_to_equity = safe_divide(ret_earn, te)

        # Sustainable growth rate = ROE * retention ratio
        roe = safe_divide(ni, te)
        rr = result.retention_ratio
        if roe is not None and rr is not None:
            result.sustainable_growth_rate = round(roe * rr, 6)

        # Internal growth rate = ROA * retention / (1 - ROA * retention)
        roa = safe_divide(ni, ta)
        if roa is not None and rr is not None:
            roa_rr = roa * rr
            if roa_rr < 1.0:
                result.internal_growth_rate = round(roa_rr / (1.0 - roa_rr), 6)

        # --- scoring based on retention_ratio ---
        if rr is None:
            return result

        if rr >= 0.80:
            score = 10.0
        elif rr >= 0.70:
            score = 8.5
        elif rr >= 0.60:
            score = 7.0
        elif rr >= 0.50:
            score = 5.5
        elif rr >= 0.30:
            score = 4.0
        elif rr >= 0.10:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: RE/Equity
        re_eq = result.re_to_equity
        if re_eq is not None:
            if re_eq >= 0.60:
                score += 0.5
            elif re_eq < 0.10:
                score -= 0.5

        # Adjustment: sustainable growth rate
        sgr = result.sustainable_growth_rate
        if sgr is not None:
            if sgr >= 0.15:
                score += 0.5
            elif sgr < 0.03:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.pr_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.pr_grade = grade

        result.summary = (
            f"Profit Retention: Retention ratio {rr:.1%}, "
            f"payout ratio {result.payout_ratio:.1%}. "
            f"Score: {result.pr_score}/10. Grade: {grade}."
        )

        return result

    def debt_service_coverage_analysis(self, data: FinancialData) -> DebtServiceCoverageResult:
        """Phase 76: Debt Service Coverage Analysis.

        Evaluates the company's ability to meet debt obligations
        from operating income and cash flow.
        """
        result = DebtServiceCoverageResult()

        ie = data.interest_expense or 0
        if ie <= 0:
            return result

        ebitda = data.ebitda or 0
        ocf = data.operating_cash_flow or 0
        rev = data.revenue or 0
        capex = data.capex or 0
        fcf = ocf - capex

        # Total debt service = interest expense (no principal data available)
        tds = ie

        result.dscr = safe_divide(ebitda, tds)
        result.ocf_to_debt_service = safe_divide(ocf, tds)
        result.ebitda_to_interest = safe_divide(ebitda, ie)
        result.fcf_to_debt_service = safe_divide(fcf, tds) if fcf != 0 else safe_divide(fcf, tds)
        result.debt_service_to_revenue = safe_divide(tds, rev)

        dscr = result.dscr
        if dscr is not None:
            result.coverage_cushion = dscr - 1.0

        # Scoring based on DSCR
        if dscr is None:
            return result

        if dscr >= 4.0:
            score = 10.0
        elif dscr >= 3.0:
            score = 8.5
        elif dscr >= 2.5:
            score = 7.0
        elif dscr >= 2.0:
            score = 5.5
        elif dscr >= 1.5:
            score = 4.0
        elif dscr >= 1.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OCF coverage bonus/penalty
        ocd = result.ocf_to_debt_service
        if ocd is not None:
            if ocd >= 5.0:
                score += 0.5
            elif ocd < 1.0:
                score -= 0.5

        # Adjustment: debt service burden on revenue
        dsr = result.debt_service_to_revenue
        if dsr is not None:
            if dsr <= 0.02:
                score += 0.5
            elif dsr >= 0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.dsc_score = round(score, 1)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.dsc_grade = grade

        dscr_str = f"{dscr:.2f}x" if dscr is not None else "N/A"
        result.summary = (
            f"Debt Service Coverage — DSCR: {dscr_str}, "
            f"Score: {score}/10. Grade: {grade}."
        )

        return result

    def capital_allocation_analysis(self, data: FinancialData) -> CapitalAllocationResult:
        """Phase 73: Capital Allocation Analysis.

        Evaluates how effectively the company allocates capital
        across reinvestment, shareholder returns, and growth.
        """
        result = CapitalAllocationResult()

        capex = data.capex or 0
        rev = data.revenue or 0
        ocf = data.operating_cash_flow or 0
        ni = data.net_income or 0
        dep = data.depreciation or 0
        div = data.dividends_paid or 0
        bb = data.share_buybacks or 0

        # Need some capital allocation data
        if capex <= 0 and div <= 0 and bb <= 0:
            return result

        fcf = ocf - capex

        # Compute ratios
        result.capex_to_revenue = safe_divide(capex, rev)
        result.capex_to_ocf = safe_divide(capex, ocf)

        total_payout = div + bb
        result.shareholder_return_ratio = safe_divide(total_payout, ni) if total_payout > 0 else None
        result.reinvestment_rate = safe_divide(capex, dep) if dep > 0 else None
        result.fcf_yield = safe_divide(fcf, rev) if rev > 0 else None
        result.total_payout_to_fcf = safe_divide(total_payout, fcf) if fcf > 0 and total_payout > 0 else None

        # Scoring based on capex_to_ocf (reinvestment efficiency)
        cto = result.capex_to_ocf
        if cto is None:
            # Fall back to capex_to_revenue
            ctr = result.capex_to_revenue
            if ctr is None:
                return result
            if ctr <= 0.05:
                score = 10.0
            elif ctr <= 0.10:
                score = 8.5
            elif ctr <= 0.15:
                score = 7.0
            elif ctr <= 0.20:
                score = 5.5
            elif ctr <= 0.30:
                score = 4.0
            elif ctr <= 0.40:
                score = 2.5
            else:
                score = 1.0
        else:
            # Balanced reinvestment: 20-50% of OCF is healthy
            if cto <= 0.25:
                score = 10.0
            elif cto <= 0.35:
                score = 8.5
            elif cto <= 0.50:
                score = 7.0
            elif cto <= 0.65:
                score = 5.5
            elif cto <= 0.80:
                score = 4.0
            elif cto <= 0.95:
                score = 2.5
            else:
                score = 1.0

        # Adjustment: FCF yield bonus/penalty
        fy = result.fcf_yield
        if fy is not None:
            if fy >= 0.10:
                score += 0.5
            elif fy < 0.0:
                score -= 0.5

        # Adjustment: total payout sustainability
        tpf = result.total_payout_to_fcf
        if tpf is not None:
            if tpf <= 0.50:
                score += 0.5
            elif tpf >= 1.0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.ca_score = round(score, 1)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ca_grade = grade

        cto_str = f"{cto:.1%}" if cto is not None else "N/A"
        result.summary = (
            f"Capital Allocation — CapEx/OCF: {cto_str}, "
            f"Score: {score}/10. Grade: {grade}."
        )

        return result

    def tax_efficiency_analysis(self, data: FinancialData) -> TaxEfficiencyResult:
        """Phase 70: Tax Efficiency Analysis.

        Measures how effectively a company manages its tax obligations.
        Examines effective tax rate, tax burden ratios, and after-tax margins.
        """
        result = TaxEfficiencyResult()

        # Need tax_expense or ability to derive it
        te = data.tax_expense or 0
        ebt_val = data.ebt or 0
        rev = data.revenue or 0
        ni = data.net_income or 0
        ebitda = data.ebitda or 0
        ebit_val = data.ebit or 0
        ie = data.interest_expense or 0

        # If no tax expense, try to derive: EBT - NI = tax
        if not data.tax_expense and data.ebt and data.net_income:
            te = data.ebt - data.net_income

        # If no EBT, try EBIT - IE
        if not ebt_val and ebit_val and data.interest_expense is not None:
            ebt_val = ebit_val - ie

        # Need at least some tax data to proceed
        if te <= 0 and ni <= 0 and rev <= 0:
            return result

        # Compute ratios
        result.effective_tax_rate = safe_divide(te, ebt_val) if ebt_val > 0 else None
        result.tax_to_revenue = safe_divide(te, rev)
        result.tax_to_ebitda = safe_divide(te, ebitda)
        result.after_tax_margin = safe_divide(ni, rev)
        # Tax shield: interest * ETR / NI (benefit of debt tax shield relative to NI)
        if result.effective_tax_rate is not None and ni > 0:
            result.tax_shield_ratio = safe_divide(ie * result.effective_tax_rate, ni)
        # PreTax/EBIT ratio (impact of non-operating items)
        result.pretax_to_ebit = safe_divide(ebt_val, ebit_val)

        # Scoring based on effective tax rate (lower = more efficient)
        etr = result.effective_tax_rate
        if etr is None:
            # Fall back to tax_to_revenue for scoring
            etr_proxy = result.tax_to_revenue
            if etr_proxy is None or etr_proxy <= 0:
                return result
            # Use after_tax_margin as primary scoring instead
            atm = result.after_tax_margin or 0
            if atm >= 0.20:
                score = 10.0
            elif atm >= 0.15:
                score = 8.5
            elif atm >= 0.10:
                score = 7.0
            elif atm >= 0.05:
                score = 5.5
            elif atm >= 0.02:
                score = 4.0
            elif atm >= 0.0:
                score = 2.5
            else:
                score = 1.0
        else:
            # Primary scoring on ETR (lower is better for tax efficiency)
            if etr <= 0.15:
                score = 10.0
            elif etr <= 0.20:
                score = 8.5
            elif etr <= 0.25:
                score = 7.0
            elif etr <= 0.30:
                score = 5.5
            elif etr <= 0.35:
                score = 4.0
            elif etr <= 0.40:
                score = 2.5
            else:
                score = 1.0

        # Adjustment: after-tax margin bonus/penalty
        atm = result.after_tax_margin
        if atm is not None:
            if atm >= 0.15:
                score += 0.5
            elif atm < 0.02:
                score -= 0.5

        # Adjustment: tax-to-revenue ratio bonus/penalty
        ttr = result.tax_to_revenue
        if ttr is not None:
            if ttr <= 0.03:
                score += 0.5
            elif ttr >= 0.15:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.te_score = round(score, 1)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.te_grade = grade

        etr_str = f"{etr:.1%}" if etr is not None else "N/A"
        result.summary = (
            f"Tax Efficiency — ETR: {etr_str}, "
            f"Score: {score}/10. Grade: {grade}."
        )

        return result

    def roic_analysis(self, data: FinancialData) -> ROICResult:
        """Phase 56: Return on Invested Capital Analysis.

        ROIC = NOPAT / Invested Capital.
        Invested Capital = Total Equity + Total Debt - Cash.
        NOPAT = EBIT * (1 - tax_rate), estimated tax_rate from NI/EBT.
        """
        result = ROICResult()
        ebit = data.ebit
        ni = data.net_income
        ie = data.interest_expense or 0
        te = data.total_equity
        td = data.total_debt or 0
        cash = data.cash or 0

        if ebit is None or te is None:
            return result

        # Estimate tax rate from NI/EBT
        ebt = ebit - ie
        if ebt > 0 and ni is not None:
            tax_rate = 1 - safe_divide(ni, ebt, default=0.75)
        else:
            tax_rate = 0.25  # default assumption

        nopat = ebit * (1 - tax_rate)
        result.nopat = round(nopat, 2)

        # Invested capital
        ic = te + td - cash
        if ic <= 0:
            return result
        result.invested_capital = round(ic, 2)

        # ROIC
        roic_val = safe_divide(nopat, ic)
        if roic_val is None:
            return result
        result.roic_pct = round(roic_val * 100, 2)

        # ROA for spread
        ta = data.total_assets
        if ta and ta > 0 and ni is not None:
            roa = safe_divide(ni, ta) * 100
            result.roic_roa_spread = round(result.roic_pct - roa, 2) if roa is not None else None

        # Capital efficiency = Revenue / IC
        rev = data.revenue
        if rev and rev > 0:
            result.capital_efficiency = safe_divide(rev, ic)
            if result.capital_efficiency is not None:
                result.capital_efficiency = round(result.capital_efficiency, 2)

        # Scoring
        roic = result.roic_pct
        if roic >= 20:
            score = 10.0
        elif roic >= 15:
            score = 8.5
        elif roic >= 10:
            score = 7.0
        elif roic >= 6:
            score = 5.5
        elif roic >= 0:
            score = 3.5
        elif roic >= -5:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.capital_efficiency is not None:
            if result.capital_efficiency > 2.0:
                score += 0.5
            elif result.capital_efficiency < 0.5:
                score -= 0.5
        if result.roic_roa_spread is not None:
            if result.roic_roa_spread > 5:
                score += 0.5

        score = max(0.0, min(10.0, score))
        result.roic_score = round(score, 2)

        if score >= 8:
            result.roic_grade = "Excellent"
        elif score >= 6:
            result.roic_grade = "Good"
        elif score >= 4:
            result.roic_grade = "Adequate"
        else:
            result.roic_grade = "Weak"

        result.summary = (
            f"ROIC: {roic:.1f}%. NOPAT: ${nopat:,.0f}. "
            f"Invested Capital: ${ic:,.0f}. "
            f"Grade: {result.roic_grade}."
        )

        return result

    def roa_quality_analysis(self, data: FinancialData) -> ROAQualityResult:
        """Phase 55: Return on Assets Quality Analysis.

        Examines asset productivity: ROA, operating ROA, cash ROA,
        asset turnover, fixed-asset turnover, and capital intensity.
        """
        result = ROAQualityResult()
        rev = data.revenue
        ta = data.total_assets

        if not ta or ta == 0:
            return result

        # Core ROA
        ni = data.net_income
        if ni is not None:
            result.roa_pct = safe_divide(ni, ta) * 100 if safe_divide(ni, ta) is not None else None

        # Operating ROA
        oi = data.operating_income or data.ebit
        if oi is not None:
            result.operating_roa_pct = safe_divide(oi, ta) * 100 if safe_divide(oi, ta) is not None else None

        # Cash ROA
        ocf = data.operating_cash_flow
        if ocf is not None:
            result.cash_roa_pct = safe_divide(ocf, ta) * 100 if safe_divide(ocf, ta) is not None else None

        # Asset turnover
        if rev and rev > 0:
            result.asset_turnover = safe_divide(rev, ta)

        # Fixed asset turnover
        ca = data.current_assets
        if rev and rev > 0 and ca is not None:
            fixed_assets = ta - ca
            if fixed_assets > 0:
                result.fixed_asset_turnover = safe_divide(rev, fixed_assets)

        # Capital intensity = TA / Revenue
        if rev and rev > 0:
            result.capital_intensity = safe_divide(ta, rev)

        # Scoring based on ROA
        if result.roa_pct is None:
            return result

        roa = result.roa_pct
        if roa >= 15:
            score = 10.0
        elif roa >= 10:
            score = 8.5
        elif roa >= 7:
            score = 7.0
        elif roa >= 4:
            score = 5.5
        elif roa >= 0:
            score = 3.5
        elif roa >= -5:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.asset_turnover is not None:
            if result.asset_turnover > 1.5:
                score += 0.5
            elif result.asset_turnover < 0.3:
                score -= 0.5
        if result.cash_roa_pct is not None and result.roa_pct is not None:
            if result.cash_roa_pct > result.roa_pct:
                score += 0.5

        score = max(0.0, min(10.0, score))
        result.roa_score = round(score, 2)

        if score >= 8:
            result.roa_grade = "Excellent"
        elif score >= 6:
            result.roa_grade = "Good"
        elif score >= 4:
            result.roa_grade = "Adequate"
        else:
            result.roa_grade = "Weak"

        result.summary = (
            f"Return on Assets: {roa:.1f}%. "
        )
        if result.operating_roa_pct is not None:
            result.summary += f"Operating ROA: {result.operating_roa_pct:.1f}%. "
        if result.cash_roa_pct is not None:
            result.summary += f"Cash ROA: {result.cash_roa_pct:.1f}%. "
        result.summary += f"Grade: {result.roa_grade}."

        return result

    def roe_analysis(self, data: FinancialData) -> ROEAnalysisResult:
        """Phase 54: Return on Equity Analysis.

        Decomposes ROE via DuPont: Net Margin × Asset Turnover × Equity Multiplier.
        Also computes ROA, retention ratio, and sustainable growth rate.
        """
        result = ROEAnalysisResult()
        rev = data.revenue
        ni = data.net_income
        ta = data.total_assets
        te = data.total_equity

        if not rev or not te or te == 0:
            return result

        # Core ROE
        if ni is not None:
            result.roe_pct = safe_divide(ni, te) * 100 if safe_divide(ni, te) is not None else None

        # DuPont components
        if ni is not None and rev > 0:
            result.net_margin_pct = safe_divide(ni, rev) * 100 if safe_divide(ni, rev) is not None else None
        if ta and ta > 0:
            result.asset_turnover = safe_divide(rev, ta)
            result.equity_multiplier = safe_divide(ta, te)
        if ni is not None and ta and ta > 0:
            result.roa_pct = safe_divide(ni, ta) * 100 if safe_divide(ni, ta) is not None else None

        # Retention & sustainable growth
        ret_earn = data.retained_earnings
        if ni is not None and ni > 0 and ret_earn is not None:
            dividends_paid = ni - ret_earn if ret_earn < ni else 0
            result.retention_ratio = safe_divide(ni - dividends_paid, ni)
            if result.roe_pct is not None and result.retention_ratio is not None:
                result.sustainable_growth_rate = (result.roe_pct / 100) * result.retention_ratio * 100

        # Scoring
        if result.roe_pct is None:
            return result

        roe = result.roe_pct
        if roe >= 25:
            score = 10.0
        elif roe >= 18:
            score = 8.5
        elif roe >= 12:
            score = 7.0
        elif roe >= 8:
            score = 5.5
        elif roe >= 0:
            score = 3.5
        elif roe >= -10:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.asset_turnover is not None:
            if result.asset_turnover > 1.5:
                score += 0.5
            elif result.asset_turnover < 0.3:
                score -= 0.5
        if result.equity_multiplier is not None:
            if result.equity_multiplier > 5.0:
                score -= 0.5
            elif result.equity_multiplier < 2.0:
                score += 0.5

        score = max(0.0, min(10.0, score))
        result.roe_score = round(score, 2)

        if score >= 8:
            result.roe_grade = "Excellent"
        elif score >= 6:
            result.roe_grade = "Good"
        elif score >= 4:
            result.roe_grade = "Adequate"
        else:
            result.roe_grade = "Weak"

        result.summary = (
            f"Return on Equity: {roe:.1f}%. "
            f"DuPont decomposition — Net Margin: "
            f"{result.net_margin_pct:.1f}%" if result.net_margin_pct is not None else "N/A"
        )
        if result.asset_turnover is not None:
            result.summary += f", Asset Turnover: {result.asset_turnover:.2f}x"
        if result.equity_multiplier is not None:
            result.summary += f", Equity Multiplier: {result.equity_multiplier:.2f}x"
        result.summary += f". Grade: {result.roe_grade}."

        return result

    def net_profit_margin_analysis(self, data: FinancialData) -> NetProfitMarginResult:
        """Phase 53: Net Profit Margin Analysis.

        Evaluates bottom-line profitability, tax/interest burden,
        and the path from EBITDA to net income.
        """
        result = NetProfitMarginResult()

        revenue = data.revenue
        ni = data.net_income
        ebitda = data.ebitda
        ebit = data.ebit
        ie = data.interest_expense

        if not revenue or revenue <= 0:
            result.summary = "Insufficient data for Net Profit Margin."
            return result

        # --- Compute ratios ---
        if ni is not None:
            result.net_margin_pct = safe_divide(ni, revenue, 0.0) * 100

        if ebitda is not None:
            result.ebitda_margin_pct = safe_divide(ebitda, revenue, 0.0) * 100

        if ebit is not None:
            result.ebit_margin_pct = safe_divide(ebit, revenue, 0.0) * 100

        # Tax burden = NI / EBT where EBT = EBIT - IE
        if ni is not None and ebit is not None:
            ebt = ebit - (ie or 0)
            if ebt and ebt > 0:
                result.tax_burden = safe_divide(ni, ebt)

        # Interest burden = EBT / EBIT
        if ebit is not None and ebit > 0:
            ebt = ebit - (ie or 0)
            result.interest_burden = safe_divide(ebt, ebit)

        # Net to EBITDA
        if ni is not None and ebitda and ebitda > 0:
            result.net_to_ebitda = safe_divide(ni, ebitda)

        # --- Scoring ---
        nm = result.net_margin_pct
        if nm is None:
            result.summary = "Net income not available."
            return result

        if nm >= 25:
            score = 10.0
        elif nm >= 15:
            score = 8.5
        elif nm >= 10:
            score = 7.0
        elif nm >= 5:
            score = 5.5
        elif nm >= 0:
            score = 3.5
        elif nm >= -10:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.tax_burden is not None:
            if result.tax_burden > 0.80:
                score += 0.5  # Low effective tax rate
            elif result.tax_burden < 0.50:
                score -= 0.5  # High tax burden

        if result.interest_burden is not None:
            if result.interest_burden > 0.90:
                score += 0.5  # Low interest burden
            elif result.interest_burden < 0.60:
                score -= 0.5  # Heavy interest cost

        score = max(0.0, min(10.0, score))
        result.npm_score = score

        # Grading
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.npm_grade = grade

        result.summary = (
            f"Net Profit Margin: {nm:.1f}% margin, "
            f"score {score:.1f}/10. Status: {grade}."
        )

        return result

    def ebitda_margin_quality_analysis(self, data: FinancialData) -> EbitdaMarginQualityResult:
        """Phase 52: EBITDA Margin Quality Analysis.

        Evaluates EBITDA margin level, D&A intensity, and spread between
        EBITDA and operating income as an earnings quality indicator.
        """
        result = EbitdaMarginQualityResult()

        revenue = data.revenue
        ebitda = data.ebitda
        oi = data.operating_income
        dep = data.depreciation
        gp = data.gross_profit

        if not revenue or revenue <= 0:
            result.summary = "Insufficient data for EBITDA Margin Quality."
            return result

        # --- Compute ratios ---
        if ebitda is not None:
            result.ebitda_margin_pct = safe_divide(ebitda, revenue, 0.0) * 100

        if oi is not None:
            result.operating_margin_pct = safe_divide(oi, revenue, 0.0) * 100

        if dep is not None:
            result.da_intensity = safe_divide(dep, revenue, 0.0) * 100

        if result.ebitda_margin_pct is not None and result.operating_margin_pct is not None:
            result.ebitda_oi_spread = result.ebitda_margin_pct - result.operating_margin_pct

        if ebitda is not None and gp and gp > 0:
            result.ebitda_to_gp = safe_divide(ebitda, gp)

        # --- Scoring ---
        em = result.ebitda_margin_pct
        if em is None:
            result.summary = "EBITDA not available."
            return result

        if em >= 40:
            score = 10.0
        elif em >= 30:
            score = 8.5
        elif em >= 20:
            score = 7.0
        elif em >= 15:
            score = 5.5
        elif em >= 10:
            score = 4.0
        elif em >= 0:
            score = 2.5
        else:
            score = 0.5

        # Adjustments
        if result.da_intensity is not None:
            if result.da_intensity > 15:
                score -= 0.5  # Capital-heavy business
            elif result.da_intensity < 3:
                score += 0.5  # Asset-light

        if result.ebitda_to_gp is not None:
            if result.ebitda_to_gp > 0.70:
                score += 0.5  # Strong conversion from GP to EBITDA
            elif result.ebitda_to_gp < 0.30:
                score -= 0.5  # OpEx eating too much

        score = max(0.0, min(10.0, score))
        result.ebitda_score = score

        # Grading
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ebitda_grade = grade

        result.summary = (
            f"EBITDA Margin Quality: {em:.1f}% margin, "
            f"score {score:.1f}/10. Status: {grade}."
        )

        return result

    def gross_margin_stability_analysis(self, data: FinancialData) -> GrossMarginStabilityResult:
        """Phase 51: Gross Margin Stability Analysis.

        Assesses gross margin quality, cost structure efficiency, and
        margin sustainability using single-period data.
        """
        result = GrossMarginStabilityResult()

        revenue = data.revenue
        gp = data.gross_profit
        cogs = data.cogs
        oi = data.operating_income
        opex = data.operating_expenses

        if not revenue or revenue <= 0:
            result.summary = "Insufficient data for Gross Margin Stability."
            return result

        # --- Compute ratios ---
        gm_pct = safe_divide(gp, revenue, 0.0) * 100 if gp is not None else None
        cogs_r = safe_divide(cogs, revenue, None) * 100 if cogs is not None else None
        om_pct = safe_divide(oi, revenue, 0.0) * 100 if oi is not None else None

        result.gross_margin_pct = gm_pct
        result.cogs_ratio = cogs_r

        if gm_pct is not None and om_pct is not None:
            result.operating_margin_pct = om_pct
            result.margin_spread = gm_pct - om_pct

        if gp is not None and opex and opex > 0:
            result.opex_coverage = safe_divide(gp, opex)

        if gm_pct is not None:
            result.margin_buffer = gm_pct  # distance above 0% margin

        # --- Scoring ---
        if gm_pct is None:
            result.summary = "Gross profit not available."
            return result

        if gm_pct >= 60:
            score = 10.0
        elif gm_pct >= 45:
            score = 8.5
        elif gm_pct >= 30:
            score = 7.0
        elif gm_pct >= 20:
            score = 5.5
        elif gm_pct >= 10:
            score = 3.5
        elif gm_pct >= 0:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.opex_coverage is not None:
            if result.opex_coverage > 2.0:
                score += 0.5
            elif result.opex_coverage < 1.0:
                score -= 0.5

        if result.margin_spread is not None:
            if result.margin_spread > 20:
                score += 0.5
            elif result.margin_spread < 5:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.gm_stability_score = score

        # Grading
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.gm_stability_grade = grade

        result.summary = (
            f"Gross Margin Stability: {gm_pct:.1f}% gross margin, "
            f"score {score:.1f}/10. Status: {grade}."
        )

        return result

    def analyze(self, data: Union[FinancialData, pd.DataFrame]) -> AnalysisResults:
        """
        Run comprehensive analysis on financial data.
        Returns an AnalysisResults that supports dict-style access for backward compat.
        """
        if isinstance(data, pd.DataFrame):
            financial_data = self._dataframe_to_financial_data(data)
        else:
            financial_data = data

        liq = self.calculate_liquidity_ratios(financial_data)
        prof = self.calculate_profitability_ratios(financial_data)
        lev = self.calculate_leverage_ratios(financial_data)
        eff = self.calculate_efficiency_ratios(financial_data)

        results = AnalysisResults(
            liquidity_ratios=LiquidityRatios.from_dict(liq),
            profitability_ratios=ProfitabilityRatios.from_dict(prof),
            leverage_ratios=LeverageRatios.from_dict(lev),
            efficiency_ratios=EfficiencyRatios.from_dict(eff),
            cash_flow=self.analyze_cash_flow(financial_data),
            working_capital=self.analyze_working_capital(financial_data),
            dupont=self.dupont_analysis(financial_data),
            altman_z_score=self.altman_z_score(financial_data),
            piotroski_f_score=self.piotroski_f_score(financial_data),
            composite_health=self.composite_health_score(financial_data),
        )

        # Generate insights — build dict directly without seeding the cache so
        # the cache is populated only after insights are assigned.
        results.insights = self.generate_insights(results._build_dict())

        return results

    def _dataframe_to_financial_data(self, df: pd.DataFrame) -> FinancialData:
        """
        Convert a DataFrame to FinancialData by detecting column mappings.
        """
        data = FinancialData()

        # Column name mapping (lowercase patterns to attribute names)
        mappings = {
            'revenue': ['revenue', 'sales', 'net sales', 'total revenue'],
            'cogs': ['cogs', 'cost of goods sold', 'cost of sales', 'cost of revenue'],
            'gross_profit': ['gross profit', 'gross margin'],
            'operating_income': ['operating income', 'operating profit', 'ebit'],
            'net_income': ['net income', 'net profit', 'net earnings', 'profit after tax'],
            'total_assets': ['total assets', 'assets'],
            'current_assets': ['current assets'],
            'cash': ['cash', 'cash and equivalents', 'cash & equivalents'],
            'inventory': ['inventory', 'inventories'],
            'accounts_receivable': ['accounts receivable', 'receivables', 'trade receivables'],
            'total_liabilities': ['total liabilities', 'liabilities'],
            'current_liabilities': ['current liabilities'],
            'accounts_payable': ['accounts payable', 'payables', 'trade payables'],
            'total_debt': ['total debt', 'long term debt', 'debt'],
            'total_equity': ['total equity', 'shareholders equity', 'stockholders equity'],
            'interest_expense': ['interest expense', 'interest'],
            'ebt': ['earnings before tax', 'ebt', 'income before tax', 'profit before tax', 'pre-tax income'],
            'retained_earnings': ['retained earnings', 'accumulated earnings'],
            'depreciation': ['depreciation', 'depreciation and amortization', 'd&a'],
            'operating_cash_flow': ['operating cash flow', 'cash from operations'],
            'capex': ['capex', 'capital expenditure', 'capital expenditures'],
        }

        # Build reverse lookup (prefer longest pattern match to avoid greedy short patterns)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            best_attr = None
            best_len = 0
            for attr, patterns in mappings.items():
                for pattern in patterns:
                    if pattern in col_lower and len(pattern) > best_len:
                        best_attr = attr
                        best_len = len(pattern)
            if best_attr is not None:
                column_map[col] = best_attr

        # Fallback: detect row-based layout (financial terms as row values)
        if not column_map:
            label_col = self._detect_label_column(df, mappings)
            if label_col is not None:
                return self._transpose_financial_df(df, label_col, mappings)

        # Extract values (use first row if single row, or sum/latest)
        for col, attr in column_map.items():
            try:
                if len(df) == 1:
                    value = df[col].iloc[0]
                else:
                    # Use the most recent non-null value
                    value = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None

                if value is not None and not pd.isna(value):
                    setattr(data, attr, float(value))
            except (ValueError, TypeError):
                pass

        return data

    def _detect_label_column(self, df: pd.DataFrame, mappings: dict) -> object:
        """Find the first text column with >= 3 financial term matches in row values."""
        all_patterns = []
        for patterns in mappings.values():
            all_patterns.extend(patterns)
        for col in df.columns:
            if df[col].dtype == object:
                matches = 0
                for val in df[col].dropna():
                    val_lower = str(val).lower().strip()
                    if any(p in val_lower for p in all_patterns):
                        matches += 1
                if matches >= 3:
                    return col
        return None

    def _transpose_financial_df(self, df: pd.DataFrame, label_col: str, mappings: dict) -> 'FinancialData':
        """Map row labels to FinancialData using the rightmost numeric column."""
        data = FinancialData()
        # Find rightmost numeric column (latest period)
        numeric_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return data
        value_col = numeric_cols[-1]
        for _, row in df.iterrows():
            label = str(row[label_col]).lower().strip()
            best_attr = None
            best_len = 0
            for attr, patterns in mappings.items():
                for pattern in patterns:
                    if pattern in label and len(pattern) > best_len:
                        best_attr = attr
                        best_len = len(pattern)
            if best_attr is not None:
                try:
                    val = row[value_col]
                    if val is not None and not pd.isna(val):
                        setattr(data, best_attr, float(val))
                except (ValueError, TypeError):
                    pass
        return data


# Convenience function for quick analysis
def quick_analyze(df: pd.DataFrame) -> AnalysisResults:
    """
    Quick financial analysis of a DataFrame.
    """
    analyzer = CharlieAnalyzer()
    return analyzer.analyze(df)
