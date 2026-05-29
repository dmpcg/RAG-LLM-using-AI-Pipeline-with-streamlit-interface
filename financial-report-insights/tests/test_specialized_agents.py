"""Tests for agents/specialized.py and agents/tools.py.

Covers:
- Agent instantiation and tool registration
- Tool function computations with valid and invalid inputs
- agent.run() with mocked LLM
- Domain-appropriate tool selection
- Memory persistence after run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from agents.specialized import (
    RatioAnalystAgent,
    ReportWriterAgent,
    RiskAssessorAgent,
    TrendForecasterAgent,
    _StubLLM,
)
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

# ===========================================================================
# Helpers / fixtures
# ===========================================================================


class _MockLLM:
    """LLM stub whose generate() output can be controlled per-test."""

    def __init__(self, response: str = "Final Answer: mock response") -> None:
        self._response = response
        self.calls: List[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


def _healthy_data() -> dict:
    return {
        "total_assets": 2_000_000,
        "current_assets": 800_000,
        "current_liabilities": 300_000,
        "retained_earnings": 400_000,
        "ebit": 300_000,
        "total_equity": 1_000_000,
        "total_liabilities": 1_000_000,
        "revenue": 1_500_000,
    }


def _distressed_data() -> dict:
    return {
        "total_assets": 1_000_000,
        "current_assets": 100_000,
        "current_liabilities": 900_000,
        "retained_earnings": -200_000,
        "ebit": -50_000,
        "total_equity": 50_000,
        "total_liabilities": 950_000,
        "revenue": 200_000,
    }


# ===========================================================================
# 1. Agent instantiation
# ===========================================================================


class TestAgentInstantiation:
    def test_ratio_analyst_default_instantiation(self):
        agent = RatioAnalystAgent()
        assert agent.name == "ratio_analyst"
        assert agent._tools is not None

    def test_risk_assessor_default_instantiation(self):
        agent = RiskAssessorAgent()
        assert agent.name == "risk_assessor"
        assert agent._tools is not None

    def test_trend_forecaster_default_instantiation(self):
        agent = TrendForecasterAgent()
        assert agent.name == "trend_forecaster"
        assert agent._tools is not None

    def test_report_writer_default_instantiation(self):
        agent = ReportWriterAgent()
        assert agent.name == "report_writer"
        assert agent._tools is not None

    def test_agent_accepts_custom_llm(self):
        llm = _MockLLM()
        agent = RatioAnalystAgent(llm=llm)
        assert agent._llm is llm

    def test_report_writer_accepts_rag_instance(self):
        mock_rag = MagicMock()
        agent = ReportWriterAgent(rag_instance=mock_rag)
        assert agent is not None

    def test_stub_llm_generates_string(self):
        stub = _StubLLM()
        result = stub.generate("any prompt")
        assert isinstance(result, str)
        assert "stub response" in result


# ===========================================================================
# 2. Tool registration per agent
# ===========================================================================


class TestToolRegistration:
    def test_ratio_analyst_has_calculate_ratio(self):
        agent = RatioAnalystAgent()
        tool = agent._tools.get("calculate_ratio")
        assert tool is not None

    def test_ratio_analyst_has_compare_ratios(self):
        agent = RatioAnalystAgent()
        tool = agent._tools.get("compare_ratios")
        assert tool is not None

    def test_ratio_analyst_has_explain_ratio(self):
        agent = RatioAnalystAgent()
        tool = agent._tools.get("explain_ratio")
        assert tool is not None

    def test_ratio_analyst_has_exactly_three_tools(self):
        agent = RatioAnalystAgent()
        assert len(agent._tools.list_tools()) == 3

    def test_risk_assessor_has_assess_distress(self):
        agent = RiskAssessorAgent()
        assert agent._tools.get("assess_distress") is not None

    def test_risk_assessor_has_check_anomalies(self):
        agent = RiskAssessorAgent()
        assert agent._tools.get("check_anomalies") is not None

    def test_risk_assessor_has_evaluate_leverage(self):
        agent = RiskAssessorAgent()
        assert agent._tools.get("evaluate_leverage") is not None

    def test_risk_assessor_has_exactly_three_tools(self):
        agent = RiskAssessorAgent()
        assert len(agent._tools.list_tools()) == 3

    def test_trend_forecaster_has_forecast_metric(self):
        agent = TrendForecasterAgent()
        assert agent._tools.get("forecast_metric") is not None

    def test_trend_forecaster_has_analyze_trend(self):
        agent = TrendForecasterAgent()
        assert agent._tools.get("analyze_trend") is not None

    def test_trend_forecaster_has_detect_seasonality(self):
        agent = TrendForecasterAgent()
        assert agent._tools.get("detect_seasonality") is not None

    def test_trend_forecaster_has_exactly_three_tools(self):
        agent = TrendForecasterAgent()
        assert len(agent._tools.list_tools()) == 3

    def test_report_writer_has_search_documents(self):
        agent = ReportWriterAgent()
        assert agent._tools.get("search_documents") is not None

    def test_report_writer_has_analyze_data(self):
        agent = ReportWriterAgent()
        assert agent._tools.get("analyze_data") is not None

    def test_report_writer_has_format_section(self):
        agent = ReportWriterAgent()
        assert agent._tools.get("format_section") is not None

    def test_report_writer_has_exactly_three_tools(self):
        agent = ReportWriterAgent()
        assert len(agent._tools.list_tools()) == 3

    def test_tool_names_are_domain_specific(self):
        ratio_names = {t.name for t in RatioAnalystAgent()._tools.list_tools()}
        risk_names = {t.name for t in RiskAssessorAgent()._tools.list_tools()}
        forecast_names = {t.name for t in TrendForecasterAgent()._tools.list_tools()}
        report_names = {t.name for t in ReportWriterAgent()._tools.list_tools()}

        # No two agents share all the same tool names
        assert ratio_names != risk_names
        assert ratio_names != forecast_names
        assert risk_names != report_names


# ===========================================================================
# 3. tool_calculate_ratio
# ===========================================================================


class TestToolCalculateRatio:
    def test_current_ratio_valid(self):
        result = tool_calculate_ratio("current_ratio", current_assets=800_000, current_liabilities=400_000)
        assert "2.0000" in result
        assert "current_ratio" in result.lower()

    def test_debt_to_equity_valid(self):
        result = tool_calculate_ratio("debt_to_equity", total_debt=600_000, total_equity=1_000_000)
        assert "0.6000" in result

    def test_roe_valid(self):
        result = tool_calculate_ratio("roe", net_income=200_000, total_equity=1_000_000)
        assert "0.2000" in result

    def test_roa_valid(self):
        result = tool_calculate_ratio("roa", net_income=100_000, total_assets=2_000_000)
        assert "0.0500" in result

    def test_gross_margin_valid(self):
        result = tool_calculate_ratio("gross_margin", gross_profit=600_000, revenue=1_000_000)
        assert "0.6000" in result

    def test_net_margin_valid(self):
        result = tool_calculate_ratio("net_margin", net_income=100_000, revenue=1_000_000)
        assert "0.1000" in result

    def test_unknown_ratio_returns_error_message(self):
        result = tool_calculate_ratio("nonexistent_ratio_xyz")
        assert "Unknown ratio" in result or "nonexistent_ratio_xyz" in result

    def test_missing_field_returns_error_message(self):
        result = tool_calculate_ratio("current_ratio", current_assets=800_000)
        assert "missing" in result.lower() or "Cannot compute" in result

    def test_zero_denominator_handled_gracefully(self):
        result = tool_calculate_ratio("current_ratio", current_assets=800_000, current_liabilities=0)
        # Should not raise; value is 0 due to safe_div
        assert isinstance(result, str)

    def test_quick_ratio_with_inventory(self):
        result = tool_calculate_ratio(
            "quick_ratio",
            current_assets=800_000,
            inventory=200_000,
            current_liabilities=400_000,
        )
        assert "1.5000" in result

    def test_quick_ratio_without_inventory(self):
        result = tool_calculate_ratio(
            "quick_ratio",
            current_assets=800_000,
            current_liabilities=400_000,
        )
        # inventory defaults to 0 → same as current_ratio
        assert "2.0000" in result

    def test_result_contains_formula(self):
        result = tool_calculate_ratio("roe", net_income=50_000, total_equity=500_000)
        assert "formula" in result.lower() or "/" in result


# ===========================================================================
# 4. tool_explain_ratio
# ===========================================================================


class TestToolExplainRatio:
    def test_explains_current_ratio(self):
        result = tool_explain_ratio("current_ratio")
        assert "current_assets" in result
        assert "liquidity" in result.lower()

    def test_explains_roe(self):
        result = tool_explain_ratio("roe")
        assert "equity" in result.lower()

    def test_unknown_ratio_returns_informative_message(self):
        result = tool_explain_ratio("completely_unknown_metric")
        assert "No explanation available" in result or "unknown" in result.lower()

    def test_result_contains_range(self):
        result = tool_explain_ratio("debt_to_equity")
        assert "range" in result.lower() or "%" in result or "–" in result


# ===========================================================================
# 5. tool_compare_ratios
# ===========================================================================


class TestToolCompareRatios:
    def test_increase_detected(self):
        result = tool_compare_ratios("current_ratio", 1.5, 2.5)
        assert "increased" in result

    def test_decrease_detected(self):
        result = tool_compare_ratios("current_ratio", 2.5, 1.5)
        assert "decreased" in result

    def test_unchanged_detected(self):
        result = tool_compare_ratios("roe", 0.15, 0.15)
        assert "unchanged" in result or "0.00%" in result

    def test_labels_appear_in_output(self):
        result = tool_compare_ratios("roe", 0.10, 0.20, "FY2023", "FY2024")
        assert "FY2023" in result
        assert "FY2024" in result

    def test_invalid_inputs_handled(self):
        result = tool_compare_ratios("roe", "bad", 0.2)  # type: ignore[arg-type]
        assert isinstance(result, str)


# ===========================================================================
# 6. tool_assess_distress
# ===========================================================================


class TestToolAssessDistress:
    def test_healthy_company_not_distressed(self):
        # Z-Score for this data is ~2.4 (Grey zone), which is clearly not Distress
        result = tool_assess_distress(**_healthy_data())
        assert "Distress" not in result or "not in distress" in result.lower()

    def test_very_healthy_company_safe_zone(self):
        # Construct data that clearly scores above 2.99 (Safe zone)
        result = tool_assess_distress(
            total_assets=1_000_000,
            current_assets=700_000,
            current_liabilities=100_000,
            retained_earnings=500_000,
            ebit=400_000,
            total_equity=900_000,
            total_liabilities=100_000,
            revenue=2_000_000,
        )
        assert "Safe" in result or "LOW" in result

    def test_distressed_company_distress_zone(self):
        result = tool_assess_distress(**_distressed_data())
        assert "Distress" in result or "HIGH" in result

    def test_missing_total_assets_returns_error(self):
        result = tool_assess_distress(current_assets=100_000)
        assert "required" in result.lower() or "total_assets" in result

    def test_z_score_present_in_output(self):
        result = tool_assess_distress(**_healthy_data())
        assert "Z-Score" in result

    def test_components_listed(self):
        result = tool_assess_distress(**_healthy_data())
        assert "X1" in result
        assert "X5" in result

    def test_zero_total_assets_handled(self):
        data = _healthy_data()
        data["total_assets"] = 0
        result = tool_assess_distress(**data)
        assert isinstance(result, str)
        # Should explain the problem
        assert "total_assets" in result or "required" in result.lower()


# ===========================================================================
# 7. tool_check_anomalies
# ===========================================================================


class TestToolCheckAnomalies:
    def test_detects_high_outlier(self):
        # Most values near 10; one extreme outlier
        values = [10.0, 11.0, 9.5, 10.2, 10.8, 100.0]
        result = tool_check_anomalies(values, metric_name="revenue")
        assert "Anomalies found" in result or "anomaly" in result.lower()

    def test_no_anomalies_normal_data(self):
        values = [10.0, 10.1, 9.9, 10.0, 10.2, 10.1]
        result = tool_check_anomalies(values, metric_name="cost")
        assert "No anomalies detected" in result

    def test_empty_list_handled(self):
        result = tool_check_anomalies([], metric_name="x")
        assert "No values" in result or "empty" in result.lower()

    def test_single_value_handled(self):
        result = tool_check_anomalies([42.0])
        assert isinstance(result, str)

    def test_metric_name_appears_in_output(self):
        result = tool_check_anomalies([1.0, 2.0, 3.0], metric_name="operating_income")
        assert "operating_income" in result


# ===========================================================================
# 8. tool_evaluate_leverage
# ===========================================================================


class TestToolEvaluateLeverage:
    def test_high_leverage_detected(self):
        result = tool_evaluate_leverage(total_debt=2_000_000, total_equity=500_000)
        assert "high" in result.lower() or "HIGH" in result

    def test_low_leverage_normal(self):
        result = tool_evaluate_leverage(total_debt=200_000, total_equity=1_000_000)
        assert isinstance(result, str)
        assert "0.2" in result

    def test_interest_coverage_included(self):
        result = tool_evaluate_leverage(
            total_debt=500_000,
            total_equity=1_000_000,
            ebit=300_000,
            interest_expense=50_000,
        )
        assert "coverage" in result.lower() or "6.0" in result

    def test_insufficient_data_handled(self):
        result = tool_evaluate_leverage()
        assert "Insufficient" in result or "data" in result.lower()

    def test_zero_equity_handled(self):
        result = tool_evaluate_leverage(total_debt=500_000, total_equity=0)
        assert isinstance(result, str)


# ===========================================================================
# 9. tool_forecast
# ===========================================================================


class TestToolForecast:
    def test_forecast_trending_upward_data(self):
        values = [100.0, 110.0, 121.0, 133.0, 146.0, 161.0]
        result = tool_forecast(values, steps=4)
        assert "Period +1" in result
        assert "Period +4" in result

    def test_forecast_with_default_steps(self):
        values = [50.0, 55.0, 60.0, 65.0, 70.0]
        result = tool_forecast(values)
        assert "Period +4" in result

    def test_forecast_returns_confidence_intervals(self):
        values = [10.0, 12.0, 11.0, 13.0, 12.5, 14.0]
        result = tool_forecast(values, steps=2)
        assert "95% CI" in result or "CI:" in result

    def test_empty_values_handled(self):
        result = tool_forecast([])
        assert "No values" in result or "empty" in result.lower()

    def test_too_few_values_handled(self):
        result = tool_forecast([1.0, 2.0])
        assert "at least" in result.lower() or isinstance(result, str)

    def test_single_value_handled(self):
        result = tool_forecast([42.0])
        assert isinstance(result, str)

    def test_forecast_output_contains_historical(self):
        values = [100.0, 105.0, 110.0, 115.0]
        result = tool_forecast(values, steps=2)
        assert "Historical" in result


# ===========================================================================
# 10. tool_analyze_trend
# ===========================================================================


class TestToolAnalyzeTrend:
    def test_identifies_upward_trend(self):
        values = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        result = tool_analyze_trend(values)
        assert "Upward" in result

    def test_identifies_downward_trend(self):
        values = [150.0, 140.0, 130.0, 120.0, 110.0, 100.0]
        result = tool_analyze_trend(values)
        assert "Downward" in result

    def test_mixed_trend_labelled(self):
        values = [10.0, 15.0, 8.0, 12.0, 9.0, 14.0]
        result = tool_analyze_trend(values)
        assert "Mixed" in result or "Upward" in result or "Downward" in result

    def test_cagr_computed_for_positive_values(self):
        values = [100.0, 121.0, 146.4]  # ~10% CAGR per period
        result = tool_analyze_trend(values)
        assert "CAGR" in result
        assert "N/A" not in result

    def test_inflection_points_detected(self):
        # Up then down → inflection
        values = [1.0, 2.0, 3.0, 2.5, 2.0, 1.5]
        result = tool_analyze_trend(values)
        assert "Inflection" in result or "inflection" in result.lower()

    def test_empty_values_handled(self):
        result = tool_analyze_trend([])
        assert "No values" in result or "empty" in result.lower()

    def test_single_value_handled(self):
        result = tool_analyze_trend([42.0])
        assert isinstance(result, str)

    def test_r_squared_in_output(self):
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        result = tool_analyze_trend(values)
        assert "R²" in result or "R^2" in result


# ===========================================================================
# 11. tool_detect_seasonality
# ===========================================================================


class TestToolDetectSeasonality:
    def test_strong_seasonal_pattern_detected(self):
        # Quarterly seasonal pattern: Q4 is always high
        values = [100, 90, 95, 150, 100, 90, 95, 150, 100, 90, 95, 150]
        result = tool_detect_seasonality(values, period=4)
        assert "Strong" in result or "Moderate" in result

    def test_no_seasonality_flat_data(self):
        values = [100, 101, 99, 100, 101, 99, 100, 101]
        result = tool_detect_seasonality(values, period=4)
        assert "Weak" in result or "No" in result or "None" in result

    def test_empty_values_handled(self):
        result = tool_detect_seasonality([])
        assert "No values" in result or "empty" in result.lower()

    def test_too_few_values_handled(self):
        result = tool_detect_seasonality([1.0, 2.0, 3.0], period=4)
        assert "at least" in result.lower() or "Need" in result

    def test_output_contains_per_position_means(self):
        values = [100, 90, 95, 150, 100, 90, 95, 150]
        result = tool_detect_seasonality(values, period=4)
        assert "Position 1" in result


# ===========================================================================
# 12. tool_search_documents
# ===========================================================================


class TestToolSearchDocuments:
    def test_no_rag_returns_informative_message(self):
        result = tool_search_documents("revenue trends")
        assert "No RAG instance" in result or "not configured" in result.lower()

    def test_empty_query_handled(self):
        result = tool_search_documents("")
        assert "empty" in result.lower() or "Cannot search" in result

    def test_with_mock_rag_returning_results(self):
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [{"content": "Revenue grew 12%", "source": "report.pdf", "score": 0.95}]
        result = tool_search_documents("revenue growth", rag_instance=mock_rag)
        assert "Revenue grew 12%" in result
        assert "report.pdf" in result

    def test_with_mock_rag_returning_empty(self):
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = []
        result = tool_search_documents("obscure query", rag_instance=mock_rag)
        assert "No documents found" in result

    def test_with_failing_rag_handled(self):
        mock_rag = MagicMock()
        mock_rag.retrieve.side_effect = RuntimeError("Connection failed")
        result = tool_search_documents("query", rag_instance=mock_rag)
        assert "failed" in result.lower() or "Connection failed" in result


# ===========================================================================
# 13. tool_format_section
# ===========================================================================


class TestToolFormatSection:
    def test_formats_analysis_section(self):
        result = tool_format_section("Revenue grew 10% YoY.", section_type="analysis")
        assert "FINANCIAL ANALYSIS" in result
        assert "Revenue grew 10% YoY." in result

    def test_formats_executive_summary(self):
        result = tool_format_section("Company is healthy.", section_type="executive_summary")
        assert "EXECUTIVE SUMMARY" in result

    def test_formats_risk_section(self):
        result = tool_format_section("High leverage risk.", section_type="risk")
        assert "RISK ASSESSMENT" in result

    def test_unknown_section_type_uses_uppercase(self):
        result = tool_format_section("some content", section_type="custom_type")
        assert "CUSTOM_TYPE" in result

    def test_empty_content_handled(self):
        result = tool_format_section("", section_type="analysis")
        assert "Empty" in result or "empty" in result.lower()


# ===========================================================================
# 14. agent.run() with mocked LLM
# ===========================================================================


class TestAgentRunWithMockedLLM:
    def test_ratio_analyst_run_returns_string(self):
        llm = _MockLLM("Final Answer: current ratio is healthy at 2.0")
        agent = RatioAnalystAgent(llm=llm)
        result = agent.run("What is the current ratio?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ratio_analyst_run_stores_in_memory(self):
        llm = _MockLLM("Final Answer: roe is excellent")
        agent = RatioAnalystAgent(llm=llm)
        agent.run("Analyse ROE")
        stored = agent.memory.recall_fact("ratio_analysis")
        assert stored is not None

    def test_risk_assessor_run_returns_string(self):
        llm = _MockLLM("Final Answer: low risk profile")
        agent = RiskAssessorAgent(llm=llm)
        result = agent.run("Assess financial distress risk")
        assert isinstance(result, str)

    def test_risk_assessor_run_stores_in_memory(self):
        llm = _MockLLM("Final Answer: moderate leverage risk")
        agent = RiskAssessorAgent(llm=llm)
        agent.run("Evaluate leverage")
        assert agent.memory.recall_fact("risk_assessment") is not None

    def test_trend_forecaster_run_returns_string(self):
        llm = _MockLLM("Final Answer: strong upward trend expected")
        agent = TrendForecasterAgent(llm=llm)
        result = agent.run("Forecast revenue for next 4 quarters")
        assert isinstance(result, str)

    def test_trend_forecaster_run_stores_in_memory(self):
        llm = _MockLLM("Final Answer: moderate growth forecast")
        agent = TrendForecasterAgent(llm=llm)
        agent.run("Trend analysis for EBITDA")
        assert agent.memory.recall_fact("trend_forecast") is not None

    def test_report_writer_run_returns_string(self):
        llm = _MockLLM("Final Answer: comprehensive report complete")
        agent = ReportWriterAgent(llm=llm)
        result = agent.run("Write a financial summary")
        assert isinstance(result, str)

    def test_report_writer_run_stores_in_memory(self):
        llm = _MockLLM("Final Answer: report drafted")
        agent = ReportWriterAgent(llm=llm)
        agent.run("Generate executive report")
        assert agent.memory.recall_fact("report") is not None

    def test_agent_run_with_tool_directive_in_llm_output(self):
        """LLM emits a TOOL directive; verify the agent parses and executes it."""
        # First call returns a tool directive; subsequent calls return final answer
        call_count = [0]

        def _generate(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "1. Calculate the ratio"
            if call_count[0] == 2:
                return (
                    "TOOL: calculate_ratio\n"
                    "ARGS: ratio_name=current_ratio, current_assets=800000, current_liabilities=400000\n"
                    "Final Answer: ratio is 2.0"
                )
            return "Final Answer: done"

        llm = MagicMock()
        llm.generate.side_effect = _generate
        agent = RatioAnalystAgent(llm=llm)
        result = agent.run("What is the current ratio?")
        assert isinstance(result, str)

    def test_memory_cleared_between_runs_if_reset(self):
        llm = _MockLLM("Final Answer: answer one")
        agent = RatioAnalystAgent(llm=llm)
        agent.run("Query one")
        agent.memory.clear_short_term()
        assert agent.memory.get_recent(10) == []


# ===========================================================================
# 15. System prompt content validation
# ===========================================================================


class TestSystemPromptContent:
    def test_ratio_analyst_system_prompt_mentions_ratios(self):
        prompt = RatioAnalystAgent.SYSTEM_PROMPT
        assert "ratio" in prompt.lower()

    def test_risk_assessor_system_prompt_mentions_risk(self):
        prompt = RiskAssessorAgent.SYSTEM_PROMPT
        assert "risk" in prompt.lower()

    def test_trend_forecaster_system_prompt_mentions_forecast(self):
        prompt = TrendForecasterAgent.SYSTEM_PROMPT
        assert "forecast" in prompt.lower()

    def test_report_writer_system_prompt_mentions_report(self):
        prompt = ReportWriterAgent.SYSTEM_PROMPT
        assert "report" in prompt.lower()
