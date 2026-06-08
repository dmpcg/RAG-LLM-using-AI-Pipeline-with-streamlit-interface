"""Structured types replacing Dict[str, Any] patterns across the codebase."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RatioCategory(str, Enum):
    """Financial ratio categories."""
    LIQUIDITY = "Liquidity"
    PROFITABILITY = "Profitability"
    LEVERAGE = "Leverage"
    EFFICIENCY = "Efficiency"
    VALUATION = "Valuation"
    OTHER = "Other"


class ReportSectionKey(str, Enum):
    """Known keys for FinancialReport.sections dict."""
    EXECUTIVE_SUMMARY = "executive_summary"
    RATIO_ANALYSIS = "ratio_analysis"
    SCORING_MODELS = "scoring_models"
    CASH_FLOW = "cash_flow"
    WORKING_CAPITAL = "working_capital"
    DUPONT = "dupont"
    TRENDS = "trends"
    ANOMALIES = "anomalies"
    INSIGHTS = "insights"
    RECOMMENDATIONS = "recommendations"
    RISK_ASSESSMENT = "risk_assessment"
    PEER_COMPARISON = "peer_comparison"
    SCENARIO_ANALYSIS = "scenario_analysis"
    VALUATION = "valuation"


# ---------------------------------------------------------------------------
# Ratio dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LiquidityRatios:
    """Typed container for liquidity ratio results."""
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    working_capital: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "current_ratio": self.current_ratio,
            "quick_ratio": self.quick_ratio,
            "cash_ratio": self.cash_ratio,
            "working_capital": self.working_capital,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LiquidityRatios:
        return cls(
            current_ratio=d.get("current_ratio"),
            quick_ratio=d.get("quick_ratio"),
            cash_ratio=d.get("cash_ratio"),
            working_capital=d.get("working_capital"),
        )


@dataclass
class ProfitabilityRatios:
    """Typed container for profitability ratio results."""
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "roe": self.roe,
            "roa": self.roa,
            "roic": self.roic,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProfitabilityRatios:
        return cls(
            gross_margin=d.get("gross_margin"),
            operating_margin=d.get("operating_margin"),
            net_margin=d.get("net_margin"),
            roe=d.get("roe"),
            roa=d.get("roa"),
            roic=d.get("roic"),
        )


@dataclass
class LeverageRatios:
    """Typed container for leverage ratio results."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    debt_ratio: Optional[float] = None
    equity_multiplier: Optional[float] = None
    interest_coverage: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "debt_to_equity": self.debt_to_equity,
            "debt_to_assets": self.debt_to_assets,
            "debt_ratio": self.debt_ratio,
            "equity_multiplier": self.equity_multiplier,
            "interest_coverage": self.interest_coverage,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LeverageRatios:
        return cls(
            debt_to_equity=d.get("debt_to_equity"),
            debt_to_assets=d.get("debt_to_assets"),
            debt_ratio=d.get("debt_ratio"),
            equity_multiplier=d.get("equity_multiplier"),
            interest_coverage=d.get("interest_coverage"),
        )


@dataclass
class EfficiencyRatios:
    """Typed container for efficiency ratio results."""
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    payables_turnover: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "asset_turnover": self.asset_turnover,
            "inventory_turnover": self.inventory_turnover,
            "receivables_turnover": self.receivables_turnover,
            "payables_turnover": self.payables_turnover,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EfficiencyRatios:
        return cls(
            asset_turnover=d.get("asset_turnover"),
            inventory_turnover=d.get("inventory_turnover"),
            receivables_turnover=d.get("receivables_turnover"),
            payables_turnover=d.get("payables_turnover"),
        )


# ---------------------------------------------------------------------------
# Analysis results (replaces Dict[str, Any] from CharlieAnalyzer.analyze())
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResults:
    """Typed container for CharlieAnalyzer.analyze() output."""
    liquidity_ratios: LiquidityRatios = field(default_factory=LiquidityRatios)
    profitability_ratios: ProfitabilityRatios = field(default_factory=ProfitabilityRatios)
    leverage_ratios: LeverageRatios = field(default_factory=LeverageRatios)
    efficiency_ratios: EfficiencyRatios = field(default_factory=EfficiencyRatios)
    cash_flow: Optional[Any] = None
    working_capital: Optional[Any] = None
    dupont: Optional[Any] = None
    altman_z_score: Optional[Any] = None
    piotroski_f_score: Optional[Any] = None
    composite_health: Optional[Any] = None
    insights: List[Any] = field(default_factory=list)
    # Internal cache field -- must come after all data fields so the dataclass
    # __init__ writes it last.  init=False means callers cannot pass it; the
    # generated __init__ sets it to None via __setattr__, which triggers the
    # override below, but the guard (`if name != '_dict_cache'`) prevents
    # the cache from being touched during that write, so no AttributeError.
    _dict_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        # Invalidate the dict cache whenever a real field is mutated so that
        # post-construction mutations are always visible via dict-style access.
        # Guarding on name avoids infinite recursion when we clear the cache
        # itself.
        if name != '_dict_cache':
            object.__setattr__(self, '_dict_cache', None)

    def __post_init__(self) -> None:
        # _dict_cache is already initialised to None by the generated __init__
        # (via the field default above).  Nothing more to do here; the method
        # is retained so subclasses / existing tests that call super().__post_init__
        # continue to work.
        pass

    def _get_dict(self) -> Dict[str, Any]:
        """Return cached dict representation, building it on first access."""
        if self._dict_cache is None:
            object.__setattr__(self, '_dict_cache', self._build_dict())
        return self._dict_cache  # type: ignore[return-value]

    def _build_dict(self) -> Dict[str, Any]:
        """Build the dict representation (uncached)."""
        return {
            "liquidity_ratios": self.liquidity_ratios.to_dict(),
            "profitability_ratios": self.profitability_ratios.to_dict(),
            "leverage_ratios": self.leverage_ratios.to_dict(),
            "efficiency_ratios": self.efficiency_ratios.to_dict(),
            "cash_flow": self.cash_flow,
            "working_capital": self.working_capital,
            "dupont": self.dupont,
            "altman_z_score": self.altman_z_score,
            "piotroski_f_score": self.piotroski_f_score,
            "composite_health": self.composite_health,
            "insights": self.insights,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the legacy Dict[str, Any] format for backward compat."""
        return self._get_dict()

    def __getitem__(self, key: str) -> Any:
        """Support dict-style access for backward compat."""
        return self._get_dict()[key]

    def __contains__(self, key: object) -> bool:
        """Support 'key in results' for backward compat."""
        return key in self._get_dict()

    def __iter__(self):
        """Support iteration over keys for backward compat."""
        return iter(self._get_dict())

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict-style .get() for backward compat."""
        return self._get_dict().get(key, default)

    def items(self):
        """Support .items() for backward compat."""
        return self._get_dict().items()

    def keys(self):
        """Support .keys() for backward compat."""
        return self._get_dict().keys()

    def values(self):
        """Support .values() for backward compat."""
        return self._get_dict().values()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AnalysisResults:
        liq = d.get("liquidity_ratios", {})
        prof = d.get("profitability_ratios", {})
        lev = d.get("leverage_ratios", {})
        eff = d.get("efficiency_ratios", {})
        return cls(
            liquidity_ratios=LiquidityRatios.from_dict(liq) if isinstance(liq, dict) else liq,
            profitability_ratios=ProfitabilityRatios.from_dict(prof) if isinstance(prof, dict) else prof,
            leverage_ratios=LeverageRatios.from_dict(lev) if isinstance(lev, dict) else lev,
            efficiency_ratios=EfficiencyRatios.from_dict(eff) if isinstance(eff, dict) else eff,
            cash_flow=d.get("cash_flow"),
            working_capital=d.get("working_capital"),
            dupont=d.get("dupont"),
            altman_z_score=d.get("altman_z_score"),
            piotroski_f_score=d.get("piotroski_f_score"),
            composite_health=d.get("composite_health"),
            insights=d.get("insights", []),
        )


# ---------------------------------------------------------------------------
# Graph retrieval types
# ---------------------------------------------------------------------------


@dataclass
class GraphChunk:
    """A single chunk from graph-enhanced search."""
    source: str = ""
    content: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "content": self.content, "score": self.score}


@dataclass
class GraphFinancialContext:
    """Financial context connected to a graph chunk."""
    document: str = ""
    period: str = ""
    ratios: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "period": self.period,
            "ratios": self.ratios,
            "scores": self.scores,
        }


@dataclass
class GraphRetrievalResult:
    """Typed result from graph_enhanced_search()."""
    chunks: List[GraphChunk] = field(default_factory=list)
    financial_context: List[GraphFinancialContext] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "financial_context": [fc.to_dict() for fc in self.financial_context],
        }

    def __getitem__(self, key: str) -> Any:
        """Support dict-style access for backward compat."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict-style .get() for backward compat."""
        return self.to_dict().get(key, default)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GraphRetrievalResult:
        chunks = [
            GraphChunk(source=c.get("source", ""), content=c.get("content", ""), score=c.get("score", 0.0))
            for c in d.get("chunks", [])
        ]
        contexts = [
            GraphFinancialContext(
                document=fc.get("document", ""),
                period=fc.get("period", ""),
                ratios=fc.get("ratios", []),
                scores=fc.get("scores", []),
            )
            for fc in d.get("financial_context", [])
        ]
        return cls(chunks=chunks, financial_context=contexts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RATIO_CATEGORY_MAP: Dict[str, RatioCategory] = {
    "current_ratio": RatioCategory.LIQUIDITY,
    "quick_ratio": RatioCategory.LIQUIDITY,
    "cash_ratio": RatioCategory.LIQUIDITY,
    "working_capital": RatioCategory.LIQUIDITY,
    "gross_margin": RatioCategory.PROFITABILITY,
    "operating_margin": RatioCategory.PROFITABILITY,
    "net_margin": RatioCategory.PROFITABILITY,
    "roe": RatioCategory.PROFITABILITY,
    "roa": RatioCategory.PROFITABILITY,
    "roic": RatioCategory.PROFITABILITY,
    "debt_to_equity": RatioCategory.LEVERAGE,
    "debt_to_assets": RatioCategory.LEVERAGE,
    "debt_ratio": RatioCategory.LEVERAGE,
    "equity_multiplier": RatioCategory.LEVERAGE,
    "interest_coverage": RatioCategory.LEVERAGE,
    "asset_turnover": RatioCategory.EFFICIENCY,
    "inventory_turnover": RatioCategory.EFFICIENCY,
    "receivables_turnover": RatioCategory.EFFICIENCY,
    "payables_turnover": RatioCategory.EFFICIENCY,
}


def categorize_ratio(key: str) -> RatioCategory:
    """Return the RatioCategory for a ratio key, defaulting to OTHER."""
    return _RATIO_CATEGORY_MAP.get(key, RatioCategory.OTHER)
