"""Specialized financial analysis agents.

Each agent extends :class:`~agents.base.BaseAgent` and pre-registers a
domain-specific set of :class:`~agents.base.Tool` objects at construction
time.  The tools are thin wrappers around the pure computation functions
defined in :mod:`agents.tools` — no LLM is needed to execute a tool.

Agents accept an optional *llm* that satisfies :class:`~protocols.LLMProvider`
(i.e., exposes a ``generate(prompt: str) -> str`` method).  When ``llm`` is
``None`` a lightweight stub is used, enabling tests and workflows to run
without a live Ollama service.

Domain tool registration:

* :class:`RatioAnalystAgent`  -- ``calculate_ratio``, ``compare_ratios``,
  ``explain_ratio``
* :class:`RiskAssessorAgent`  -- ``assess_distress``, ``check_anomalies``,
  ``evaluate_leverage``
* :class:`TrendForecasterAgent`` -- ``forecast_metric``, ``analyze_trend``,
  ``detect_seasonality``
* :class:`ReportWriterAgent`  -- ``search_documents``, ``analyze_data``,
  ``format_section``
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.base import AgentMemory, BaseAgent, Tool, ToolRegistry
from agents.tools import (
    tool_analyze_trend,
    tool_assess_distress,
    tool_calculate_ratio,
    tool_check_anomalies,
    tool_compare_ratios,
    tool_detect_seasonality,
    tool_evaluate_leverage,
    tool_explain_ratio,
    tool_forecast,
    tool_format_section,
    tool_search_documents,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub LLM for offline / test use
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal LLM stand-in that returns deterministic canned responses."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return "Final Answer: stub response"


# ---------------------------------------------------------------------------
# Helpers for building domain ToolRegistry instances
# ---------------------------------------------------------------------------


def _build_ratio_registry() -> ToolRegistry:
    """Construct a ToolRegistry pre-loaded with ratio analysis tools."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculate_ratio",
            description=(
                "Compute a named financial ratio from field values. "
                "Supports current_ratio, quick_ratio, debt_to_equity, roe, roa, "
                "gross_margin, net_margin, asset_turnover, and more."
            ),
            func=tool_calculate_ratio,
            parameters={
                "ratio_name": "Name of the ratio to compute (e.g. 'current_ratio').",
                "**financial_fields": "Named financial statement fields as keyword arguments.",
            },
        )
    )
    registry.register(
        Tool(
            name="compare_ratios",
            description=(
                "Compare two ratio values across periods or entities and produce a quantitative comparison analysis."
            ),
            func=tool_compare_ratios,
            parameters={
                "ratio_name": "Name of the ratio being compared.",
                "value_a": "First ratio value (float).",
                "value_b": "Second ratio value (float).",
                "label_a": "Label for the first value (default 'Period A').",
                "label_b": "Label for the second value (default 'Period B').",
            },
        )
    )
    registry.register(
        Tool(
            name="explain_ratio",
            description=(
                "Return a comprehensive explanation of a financial ratio, "
                "including its formula, typical good range, and what high/low values indicate."
            ),
            func=tool_explain_ratio,
            parameters={
                "ratio_name": "Name of the ratio to explain (e.g. 'roe').",
            },
        )
    )
    return registry


def _build_risk_registry() -> ToolRegistry:
    """Construct a ToolRegistry pre-loaded with risk assessment tools."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="assess_distress",
            description=(
                "Compute the Altman Z-Score from financial statement fields "
                "and classify the company into Safe, Grey, or Distress zone."
            ),
            func=tool_assess_distress,
            parameters={
                "total_assets": "Total assets (required, must be positive).",
                "current_assets": "Current assets.",
                "current_liabilities": "Current liabilities.",
                "retained_earnings": "Retained earnings (or net_income as proxy).",
                "ebit": "Earnings before interest and taxes (or operating_income).",
                "total_equity": "Total shareholder equity.",
                "total_liabilities": "Total liabilities.",
                "revenue": "Total revenue.",
            },
        )
    )
    registry.register(
        Tool(
            name="check_anomalies",
            description=(
                "Identify statistical outliers in a list of metric values using Z-score analysis (threshold |z| > 2.0)."
            ),
            func=tool_check_anomalies,
            parameters={
                "metric_values": "List of numeric observations.",
                "metric_name": "Display name of the metric (optional).",
            },
        )
    )
    registry.register(
        Tool(
            name="evaluate_leverage",
            description=(
                "Evaluate a company's leverage position using debt-to-equity, "
                "debt ratio, and interest coverage metrics."
            ),
            func=tool_evaluate_leverage,
            parameters={
                "total_debt": "Total debt obligations.",
                "total_equity": "Total shareholder equity.",
                "total_assets": "Total assets.",
                "ebit": "Earnings before interest and taxes.",
                "interest_expense": "Annual interest expense.",
            },
        )
    )
    return registry


def _build_forecaster_registry() -> ToolRegistry:
    """Construct a ToolRegistry pre-loaded with trend/forecast tools."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="forecast_metric",
            description=(
                "Forecast future metric values using an AR(2) model. "
                "Returns point forecasts and 95% prediction intervals."
            ),
            func=tool_forecast,
            parameters={
                "values": "List of historical float values (at least 3 required).",
                "steps": "Number of future periods to forecast (default 4).",
            },
        )
    )
    registry.register(
        Tool(
            name="analyze_trend",
            description=(
                "Analyse trend direction (upward/downward/mixed), strength (R²), "
                "CAGR, and inflection points in a time series."
            ),
            func=tool_analyze_trend,
            parameters={
                "values": "List of sequential numeric observations (at least 2).",
            },
        )
    )
    registry.register(
        Tool(
            name="detect_seasonality",
            description=(
                "Detect seasonal patterns in a time series by comparing within-period deviations to the overall mean."
            ),
            func=tool_detect_seasonality,
            parameters={
                "values": "List of sequential numeric observations.",
                "period": "Assumed seasonal period (default 4 for quarterly data).",
            },
        )
    )
    return registry


def _build_report_writer_registry(rag_instance: Any = None) -> ToolRegistry:
    """Construct a ToolRegistry pre-loaded with report writing tools.

    Args:
        rag_instance: Optional ``SimpleRAG`` instance used by the
            ``search_documents`` tool.
    """
    registry = ToolRegistry()

    _rag = rag_instance  # capture for closure

    def _search(query: str) -> str:
        return tool_search_documents(query, rag_instance=_rag)

    def _analyze_data(analysis_type: str, **data: Any) -> str:
        """Dispatch to the appropriate computation tool."""
        atype = analysis_type.lower().replace(" ", "_")
        if atype in ("ratio", "ratios"):
            ratio_name = data.pop("ratio_name", "current_ratio")
            return tool_calculate_ratio(ratio_name, **data)
        if atype in ("distress", "z_score", "risk"):
            return tool_assess_distress(**data)
        if atype in ("trend",):
            return tool_analyze_trend(list(data.get("values", [])))
        if atype in ("forecast",):
            return tool_forecast(list(data.get("values", [])), steps=int(data.get("steps", 4)))
        if atype in ("leverage",):
            return tool_evaluate_leverage(**data)
        return f"Unknown analysis_type '{analysis_type}'. Choose from: ratio, distress, trend, forecast, leverage."

    registry.register(
        Tool(
            name="search_documents",
            description=("Retrieve relevant document chunks from the RAG store using semantic + keyword search."),
            func=_search,
            parameters={
                "query": "Natural-language search query.",
            },
        )
    )
    registry.register(
        Tool(
            name="analyze_data",
            description=(
                "Dispatch a financial computation (ratio, distress, trend, "
                "forecast, or leverage) with the provided data fields."
            ),
            func=_analyze_data,
            parameters={
                "analysis_type": "Type of analysis: ratio | distress | trend | forecast | leverage.",
                "**data": "Additional keyword arguments forwarded to the analysis function.",
            },
        )
    )
    registry.register(
        Tool(
            name="format_section",
            description=(
                "Format raw content into a structured, headed report section suitable for executive presentation."
            ),
            func=tool_format_section,
            parameters={
                "content": "Raw text content to format.",
                "section_type": (
                    "Section type: executive_summary | analysis | risk | recommendation | data | trend | forecast."
                ),
            },
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Specialized agents
# ---------------------------------------------------------------------------


class RatioAnalystAgent(BaseAgent):
    """Expert financial ratio analyst.

    Pre-registered tools: ``calculate_ratio``, ``compare_ratios``,
    ``explain_ratio``.

    Covers liquidity, profitability, leverage, and efficiency ratios.
    Results are stored in memory under ``"ratio_analysis"``.

    Args:
        llm: LLM provider (must expose ``generate(prompt) -> str``).
            Uses a lightweight stub when ``None``.
        tools: Optional pre-populated tool registry.  When supplied the
            caller's registry completely replaces the domain defaults.
        memory: Optional memory instance.
        max_steps: Step cap forwarded to :class:`BaseAgent`.
    """

    SYSTEM_PROMPT = (
        "You are an expert financial ratio analyst. You calculate, compare, and "
        "explain financial ratios including liquidity, profitability, leverage, and "
        "efficiency metrics. You provide precise numerical analysis with clear "
        "interpretations grounded in financial theory and industry context. "
        "Use the available tools to compute ratios and interpret the results."
    )

    def __init__(
        self,
        llm: Optional[object] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
        max_steps: int = 10,
    ) -> None:
        effective_tools = tools if tools is not None else _build_ratio_registry()
        super().__init__(
            name="ratio_analyst",
            llm=llm or _StubLLM(),  # type: ignore[arg-type]
            tools=effective_tools,
            memory=memory,
            max_steps=max_steps,
        )

    def run(self, query: str) -> str:
        """Run ratio analysis for *query*.

        Delegates to the parent ReAct loop and stores the final result
        under ``"ratio_analysis"`` in memory for downstream workflow steps.

        Args:
            query: Natural-language question about financial ratios.

        Returns:
            Textual analysis of relevant financial ratios.
        """
        result = super().run(query)
        self.memory.store_fact("ratio_analysis", result)
        return result


class RiskAssessorAgent(BaseAgent):
    """Financial risk assessment expert.

    Pre-registered tools: ``assess_distress``, ``check_anomalies``,
    ``evaluate_leverage``.

    Produces a risk summary covering distress prediction, anomaly detection,
    and leverage.  Results are stored in memory under ``"risk_assessment"``.

    Args:
        llm: LLM provider.  Uses a stub when ``None``.
        tools: Optional tool registry.
        memory: Optional memory instance.
        max_steps: Step cap forwarded to :class:`BaseAgent`.
    """

    SYSTEM_PROMPT = (
        "You are a financial risk assessment expert specialising in distress "
        "prediction, anomaly detection, and leverage analysis. You apply the "
        "Altman Z-Score model and statistical outlier detection to identify and "
        "communicate financial risks clearly and actionably. Always quantify "
        "risk levels and provide concrete recommendations."
    )

    def __init__(
        self,
        llm: Optional[object] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
        max_steps: int = 10,
    ) -> None:
        effective_tools = tools if tools is not None else _build_risk_registry()
        super().__init__(
            name="risk_assessor",
            llm=llm or _StubLLM(),  # type: ignore[arg-type]
            tools=effective_tools,
            memory=memory,
            max_steps=max_steps,
        )

    def run(self, query: str) -> str:
        """Assess risk given *query* context.

        Args:
            query: Textual context or question to evaluate for risk.

        Returns:
            Textual risk assessment with severity ratings.
        """
        result = super().run(query)
        self.memory.store_fact("risk_assessment", result)
        return result


class TrendForecasterAgent(BaseAgent):
    """Financial forecasting specialist.

    Pre-registered tools: ``forecast_metric``, ``analyze_trend``,
    ``detect_seasonality``.

    Identifies trends and produces short-term financial forecasts.
    Results are stored in memory under ``"trend_forecast"``.

    Args:
        llm: LLM provider.  Uses a stub when ``None``.
        tools: Optional tool registry.
        memory: Optional memory instance.
        max_steps: Step cap forwarded to :class:`BaseAgent`.
    """

    SYSTEM_PROMPT = (
        "You are a financial forecasting specialist with expertise in time series "
        "analysis, trend identification, and quantitative forecasting. You use "
        "autoregressive models and exponential smoothing to produce reliable "
        "forecasts with uncertainty quantification. Clearly state forecast "
        "assumptions and confidence levels."
    )

    def __init__(
        self,
        llm: Optional[object] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
        max_steps: int = 10,
    ) -> None:
        effective_tools = tools if tools is not None else _build_forecaster_registry()
        super().__init__(
            name="trend_forecaster",
            llm=llm or _StubLLM(),  # type: ignore[arg-type]
            tools=effective_tools,
            memory=memory,
            max_steps=max_steps,
        )

    def run(self, query: str) -> str:
        """Forecast trends for *query*.

        Args:
            query: Textual context or question about future financial trends.

        Returns:
            Textual trend analysis and forecast.
        """
        result = super().run(query)
        self.memory.store_fact("trend_forecast", result)
        return result


class ReportWriterAgent(BaseAgent):
    """Financial report writing expert.

    Pre-registered tools: ``search_documents``, ``analyze_data``,
    ``format_section``.

    Synthesises analysis outputs into a coherent narrative report.
    Results are stored in memory under ``"report"``.

    Args:
        llm: LLM provider.  Uses a stub when ``None``.
        rag_instance: Optional ``SimpleRAG`` instance used by the
            ``search_documents`` tool.  When ``None`` the tool returns an
            informational "no RAG configured" message.
        tools: Optional tool registry (replaces domain defaults when supplied).
        memory: Optional memory instance.
        max_steps: Step cap forwarded to :class:`BaseAgent`.
    """

    SYSTEM_PROMPT = (
        "You are a financial report writing expert. You synthesise quantitative "
        "analysis results, retrieved document context, and financial insights into "
        "clear, structured, and professional financial reports. Use precise language, "
        "cite relevant data, and format outputs for executive audiences. Always "
        "structure reports with clear section headers and supporting evidence."
    )

    def __init__(
        self,
        llm: Optional[object] = None,
        rag_instance: Any = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
        max_steps: int = 10,
    ) -> None:
        effective_tools = tools if tools is not None else _build_report_writer_registry(rag_instance)
        super().__init__(
            name="report_writer",
            llm=llm or _StubLLM(),  # type: ignore[arg-type]
            tools=effective_tools,
            memory=memory,
            max_steps=max_steps,
        )

    def run(self, query: str) -> str:
        """Write a report synthesising the context in *query*.

        Args:
            query: Aggregated context from prior workflow steps.

        Returns:
            Formatted textual report.
        """
        result = super().run(query)
        self.memory.store_fact("report", result)
        return result
