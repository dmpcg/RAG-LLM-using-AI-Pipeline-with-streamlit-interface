"""Tests for agents/workflows.py -- multi-agent workflow orchestration.

All agents are mocked so the test suite does not require a live LLM service.
The mock strategy creates a MagicMock whose `run()` method returns a
predetermined string, bypassing the real BaseAgent loop entirely.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from agents.base import AgentMemory, AgentMessage
from agents.specialized import (
    RatioAnalystAgent,
    ReportWriterAgent,
    RiskAssessorAgent,
    TrendForecasterAgent,
)
from agents.workflows import (
    ConditionalWorkflow,
    ParallelWorkflow,
    SequentialWorkflow,
    WorkflowResult,
    WorkflowStep,
    _resolve_template,
    create_comprehensive_analysis_workflow,
    create_query_router_workflow,
    create_quick_scan_workflow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, return_value: str = "") -> MagicMock:
    """Return a MagicMock that satisfies the BaseAgent duck-type contract.

    The mock's ``.run()`` returns *return_value* (or ``"[<name>] result"`` when
    empty).  We use a plain ``MagicMock`` rather than ``spec=BaseAgent`` so
    that we can set ``agent.name`` freely without triggering the LLM init.
    """
    agent = MagicMock()
    agent.name = name
    agent.run.return_value = return_value or f"[{name}] result"
    return agent


def _agents_fixture() -> Dict[str, MagicMock]:
    return {
        "ratio_analyst": _make_agent("ratio_analyst", "[ratio] analysis done"),
        "risk_assessor": _make_agent("risk_assessor", "[risk] assessment done"),
        "trend_forecaster": _make_agent("trend_forecaster", "[trend] forecast done"),
        "report_writer": _make_agent("report_writer", "[report] written"),
    }


# ---------------------------------------------------------------------------
# _resolve_template helper
# ---------------------------------------------------------------------------


class TestResolveTemplate:
    def test_no_placeholders(self):
        assert _resolve_template("hello world", {}) == "hello world"

    def test_single_placeholder_resolved(self):
        result = _resolve_template("Hello {name}!", {"name": "Alice"})
        # Context values wrapped in sentinel delimiters (P0-SEC-05 injection defense)
        assert "Alice" in result
        assert "[USER_CONTEXT_START]" in result
        assert "[USER_CONTEXT_END]" in result

    def test_multiple_placeholders(self):
        result = _resolve_template("{a} and {b}", {"a": "X", "b": "Y"})
        assert "X" in result
        assert "Y" in result

    def test_missing_placeholder_left_verbatim(self):
        result = _resolve_template("Value: {missing}", {})
        assert "{missing}" in result

    def test_extra_context_keys_ignored(self):
        result = _resolve_template("hi {name}", {"name": "Bob", "extra": "unused"})
        assert "Bob" in result
        assert "unused" not in result

    def test_numeric_value_converted_to_str(self):
        result = _resolve_template("Count: {n}", {"n": 42})
        assert "42" in result


# ---------------------------------------------------------------------------
# WorkflowStep dataclass
# ---------------------------------------------------------------------------


class TestWorkflowStep:
    def test_default_output_key_equals_agent_name(self):
        step = WorkflowStep(agent_name="ratio_analyst", query_template="q")
        assert step.output_key == "ratio_analyst"

    def test_explicit_output_key(self):
        step = WorkflowStep(agent_name="ratio_analyst", query_template="q", output_key="my_key")
        assert step.output_key == "my_key"

    def test_default_depends_on_empty(self):
        step = WorkflowStep(agent_name="a", query_template="q")
        assert step.depends_on == []

    def test_depends_on_stored(self):
        step = WorkflowStep(agent_name="b", query_template="q", depends_on=["a_result"])
        assert "a_result" in step.depends_on


# ---------------------------------------------------------------------------
# WorkflowResult dataclass
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def test_defaults(self):
        r = WorkflowResult()
        assert r.steps_completed == []
        assert r.final_output == ""
        assert r.total_duration_ms == 0.0
        assert r.success is True
        assert r.errors == []

    def test_success_false_when_errors_present(self):
        r = WorkflowResult(errors=["oops"], success=False)
        assert not r.success

    def test_steps_appended(self):
        r = WorkflowResult()
        r.steps_completed.append({"agent_name": "a", "result": "ok"})
        assert len(r.steps_completed) == 1


# ---------------------------------------------------------------------------
# SequentialWorkflow
# ---------------------------------------------------------------------------


class TestSequentialWorkflow:
    def test_executes_steps_in_order(self):
        call_order: List[str] = []

        def make_recording_agent(name: str) -> MagicMock:
            agent = MagicMock()
            agent.name = name
            agent.run.side_effect = lambda q: call_order.append(name) or f"[{name}]"
            return agent

        agents = {
            "a": make_recording_agent("a"),
            "b": make_recording_agent("b"),
            "c": make_recording_agent("c"),
        }
        steps = [
            WorkflowStep("a", "step a", output_key="r_a"),
            WorkflowStep("b", "step b", output_key="r_b"),
            WorkflowStep("c", "step c", output_key="r_c"),
        ]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()
        assert call_order == ["a", "b", "c"]
        assert result.success

    def test_passes_output_as_context_to_next_step(self):
        agents = {
            "step1": _make_agent("step1", "FIRST_OUTPUT"),
            "step2": _make_agent("step2", "SECOND_OUTPUT"),
        }
        steps = [
            WorkflowStep("step1", "initial query", output_key="first"),
            WorkflowStep("step2", "based on {first}", output_key="second"),
        ]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        wf.run()
        # step2 should receive the resolved template containing FIRST_OUTPUT
        call_args = agents["step2"].run.call_args[0][0]
        assert "FIRST_OUTPUT" in call_args

    def test_initial_context_available_to_first_step(self):
        agent = _make_agent("a", "done")
        agents = {"a": agent}
        steps = [WorkflowStep("a", "company={company}", output_key="r")]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        wf.run(initial_context={"company": "Acme"})
        call_args = agent.run.call_args[0][0]
        assert "Acme" in call_args

    def test_step_failure_does_not_abort_subsequent_steps(self):
        def failing_run(q: str) -> str:
            raise RuntimeError("agent exploded")

        bad_agent = MagicMock()
        bad_agent.name = "bad"
        bad_agent.run.side_effect = failing_run

        good_agent = _make_agent("good", "good result")

        agents = {"bad": bad_agent, "good": good_agent}
        steps = [
            WorkflowStep("bad", "fail here", output_key="bad_out"),
            WorkflowStep("good", "continue here", output_key="good_out"),
        ]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()

        assert len(result.errors) == 1
        assert "agent exploded" in result.errors[0]
        # good agent still ran
        assert good_agent.run.called
        assert len(result.steps_completed) == 2

    def test_result_has_all_required_fields(self):
        agents = {"a": _make_agent("a", "output")}
        steps = [WorkflowStep("a", "q", output_key="r")]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()

        assert isinstance(result, WorkflowResult)
        assert isinstance(result.steps_completed, list)
        assert isinstance(result.final_output, str)
        assert isinstance(result.total_duration_ms, float)
        assert isinstance(result.success, bool)
        assert isinstance(result.errors, list)

    def test_final_output_is_last_successful_step(self):
        agents = {
            "first": _make_agent("first", "output1"),
            "second": _make_agent("second", "output2"),
        }
        steps = [
            WorkflowStep("first", "q1", output_key="r1"),
            WorkflowStep("second", "q2", output_key="r2"),
        ]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()
        assert result.final_output == "output2"

    def test_step_record_contains_timing(self):
        agents = {"a": _make_agent("a", "done")}
        steps = [WorkflowStep("a", "q", output_key="r")]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()
        record = result.steps_completed[0]
        assert "duration_ms" in record
        assert record["duration_ms"] >= 0

    def test_unknown_agent_name_records_error(self):
        agents: Dict[str, Any] = {}
        steps = [WorkflowStep("nonexistent", "q", output_key="r")]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()
        assert not result.success
        assert len(result.errors) == 1
        assert "nonexistent" in result.errors[0]

    def test_empty_steps_returns_empty_result(self):
        wf = SequentialWorkflow(agents={}, steps=[])
        result = wf.run()
        assert result.success
        assert result.steps_completed == []
        assert result.final_output == ""

    def test_step_record_has_agent_name_and_query(self):
        agents = {"a": _make_agent("a", "done")}
        steps = [WorkflowStep("a", "my query", output_key="r")]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()
        record = result.steps_completed[0]
        assert record["agent_name"] == "a"
        assert record["query"] == "my query"

    def test_total_duration_ms_positive(self):
        agents = {"a": _make_agent("a", "done")}
        steps = [WorkflowStep("a", "q", output_key="r")]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()
        assert result.total_duration_ms >= 0

    def test_step_record_error_none_on_success(self):
        agents = {"a": _make_agent("a", "ok")}
        steps = [WorkflowStep("a", "q", output_key="r")]
        wf = SequentialWorkflow(agents=agents, steps=steps)
        result = wf.run()
        assert result.steps_completed[0]["error"] is None


# ---------------------------------------------------------------------------
# ParallelWorkflow
# ---------------------------------------------------------------------------


class TestParallelWorkflow:
    def test_independent_steps_all_run(self):
        agents = _agents_fixture()
        steps = [
            WorkflowStep("ratio_analyst", "q", output_key="r1"),
            WorkflowStep("risk_assessor", "q", output_key="r2"),
            WorkflowStep("trend_forecaster", "q", output_key="r3"),
        ]
        wf = ParallelWorkflow(agents=agents, steps=steps, max_workers=3)
        result = wf.run()
        assert result.success
        assert len(result.steps_completed) == 3

    def test_dependent_step_runs_after_prerequisites(self):
        call_order: List[str] = []

        def make_agent_recording(name: str, out: str) -> MagicMock:
            agent = MagicMock()
            agent.name = name

            def recording_run(q: str) -> str:
                call_order.append(name)
                return out

            agent.run.side_effect = recording_run
            return agent

        agents = {
            "a": make_agent_recording("a", "A_OUT"),
            "b": make_agent_recording("b", "B_OUT"),
            "dep": make_agent_recording("dep", "DEP_OUT"),
        }
        steps = [
            WorkflowStep("a", "q", output_key="a_out"),
            WorkflowStep("b", "q", output_key="b_out"),
            WorkflowStep(
                "dep",
                "from {a_out} and {b_out}",
                depends_on=["a_out", "b_out"],
                output_key="dep_out",
            ),
        ]
        wf = ParallelWorkflow(agents=agents, steps=steps, max_workers=3)
        result = wf.run()
        assert result.success
        # "dep" must appear after both "a" and "b"
        assert call_order.index("dep") > call_order.index("a")
        assert call_order.index("dep") > call_order.index("b")

    def test_dependent_step_receives_prior_outputs_in_query(self):
        agents = {
            "src": _make_agent("src", "SRC_DATA"),
            "consumer": _make_agent("consumer", "CONSUMED"),
        }
        steps = [
            WorkflowStep("src", "produce data", output_key="src_out"),
            WorkflowStep(
                "consumer",
                "consume {src_out}",
                depends_on=["src_out"],
                output_key="con_out",
            ),
        ]
        wf = ParallelWorkflow(agents=agents, steps=steps, max_workers=2)
        wf.run()
        call_arg = agents["consumer"].run.call_args[0][0]
        assert "SRC_DATA" in call_arg

    def test_single_step_works(self):
        agents = {"a": _make_agent("a", "solo")}
        steps = [WorkflowStep("a", "q", output_key="r")]
        wf = ParallelWorkflow(agents=agents, steps=steps, max_workers=1)
        result = wf.run()
        assert result.success
        assert result.final_output == "solo"

    def test_failed_step_recorded_in_errors(self):
        bad = MagicMock()
        bad.name = "bad"
        bad.run.side_effect = ValueError("parallel failure")
        agents = {"bad": bad, "good": _make_agent("good", "ok")}
        steps = [
            WorkflowStep("bad", "q", output_key="bad_out"),
            WorkflowStep("good", "q", output_key="good_out"),
        ]
        wf = ParallelWorkflow(agents=agents, steps=steps, max_workers=2)
        result = wf.run()
        assert not result.success
        assert any("parallel failure" in e for e in result.errors)

    def test_empty_steps_returns_empty_result(self):
        wf = ParallelWorkflow(agents={}, steps=[], max_workers=2)
        result = wf.run()
        assert result.success
        assert result.steps_completed == []

    def test_result_has_timing(self):
        agents = {"a": _make_agent("a", "done")}
        steps = [WorkflowStep("a", "q", output_key="r")]
        wf = ParallelWorkflow(agents=agents, steps=steps)
        result = wf.run()
        assert result.total_duration_ms >= 0

    def test_unknown_agent_in_parallel_records_error(self):
        agents: Dict[str, Any] = {}
        steps = [WorkflowStep("ghost", "q", output_key="r")]
        wf = ParallelWorkflow(agents=agents, steps=steps)
        result = wf.run()
        assert not result.success
        assert "ghost" in result.errors[0]

    def test_initial_context_available_to_independent_steps(self):
        agent = _make_agent("a", "done")
        agents = {"a": agent}
        steps = [WorkflowStep("a", "company={company}", output_key="r")]
        wf = ParallelWorkflow(agents=agents, steps=steps)
        wf.run(initial_context={"company": "BigCo"})
        call_arg = agent.run.call_args[0][0]
        assert "BigCo" in call_arg

    def test_multiple_waves_all_steps_complete(self):
        agents = {
            "a": _make_agent("a", "A"),
            "b": _make_agent("b", "B"),
            "c": _make_agent("c", "C"),
            "d": _make_agent("d", "D"),
        }
        steps = [
            WorkflowStep("a", "q", output_key="a_out"),
            WorkflowStep("b", "q", output_key="b_out"),
            WorkflowStep("c", "{a_out}", depends_on=["a_out"], output_key="c_out"),
            WorkflowStep(
                "d",
                "{b_out} {c_out}",
                depends_on=["b_out", "c_out"],
                output_key="d_out",
            ),
        ]
        wf = ParallelWorkflow(agents=agents, steps=steps, max_workers=4)
        result = wf.run()
        assert result.success
        assert len(result.steps_completed) == 4


# ---------------------------------------------------------------------------
# ConditionalWorkflow
# ---------------------------------------------------------------------------


class TestConditionalWorkflow:
    def test_routes_to_correct_agent(self):
        agents = _agents_fixture()
        router = lambda q: ["ratio_analyst"] if "ratio" in q else ["risk_assessor"]
        wf = ConditionalWorkflow(agents=agents, router_func=router)
        result = wf.run("what are the ratios?")
        assert result.success
        assert agents["ratio_analyst"].run.called
        assert not agents["risk_assessor"].run.called

    def test_unknown_route_returns_empty_when_router_returns_empty(self):
        agents = _agents_fixture()
        router = lambda q: []
        wf = ConditionalWorkflow(agents=agents, router_func=router)
        result = wf.run("something completely unknown")
        assert result.success
        assert result.steps_completed == []

    def test_unknown_agent_name_from_router_records_error(self):
        agents: Dict[str, Any] = {}
        router = lambda q: ["ghost_agent"]
        wf = ConditionalWorkflow(agents=agents, router_func=router)
        result = wf.run("q")
        assert not result.success
        assert any("ghost_agent" in e for e in result.errors)

    def test_router_exception_recorded_as_error(self):
        def broken_router(q: str) -> List[str]:
            raise RuntimeError("router broken")

        wf = ConditionalWorkflow(agents={}, router_func=broken_router)
        result = wf.run("q")
        assert not result.success
        assert any("router broken" in e for e in result.errors)

    def test_multiple_agents_from_router_all_run(self):
        agents = _agents_fixture()
        router = lambda q: ["ratio_analyst", "risk_assessor"]
        wf = ConditionalWorkflow(agents=agents, router_func=router)
        result = wf.run("broad query")
        assert result.success
        assert len(result.steps_completed) == 2
        assert agents["ratio_analyst"].run.called
        assert agents["risk_assessor"].run.called

    def test_final_output_set(self):
        agents = {"a": _make_agent("a", "THE_ANSWER")}
        router = lambda q: ["a"]
        wf = ConditionalWorkflow(agents=agents, router_func=router)
        result = wf.run("q")
        assert result.final_output == "THE_ANSWER"

    def test_agent_failure_in_conditional_recorded(self):
        bad = MagicMock()
        bad.name = "bad"
        bad.run.side_effect = Exception("oops")
        agents = {"bad": bad}
        router = lambda q: ["bad"]
        wf = ConditionalWorkflow(agents=agents, router_func=router)
        result = wf.run("trigger bad")
        assert not result.success
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# create_comprehensive_analysis_workflow
# ---------------------------------------------------------------------------


class TestCreateComprehensiveAnalysisWorkflow:
    def test_returns_sequential_workflow(self):
        agents = _agents_fixture()
        wf = create_comprehensive_analysis_workflow(agents)
        assert isinstance(wf, SequentialWorkflow)

    def test_has_four_steps(self):
        agents = _agents_fixture()
        wf = create_comprehensive_analysis_workflow(agents)
        assert len(wf.steps) == 4

    def test_step_output_keys_correct(self):
        agents = _agents_fixture()
        wf = create_comprehensive_analysis_workflow(agents)
        keys = [s.output_key for s in wf.steps]
        assert "ratio_analysis" in keys
        assert "trend_analysis" in keys
        assert "risk_assessment" in keys
        assert "report" in keys

    def test_runs_successfully_with_mocked_agents(self):
        agents = _agents_fixture()
        wf = create_comprehensive_analysis_workflow(agents)
        result = wf.run(initial_context={"query": "Analyse company XYZ"})
        assert result.success
        assert len(result.steps_completed) == 4
        assert result.final_output != ""

    def test_report_writer_receives_prior_context(self):
        agents = _agents_fixture()
        wf = create_comprehensive_analysis_workflow(agents)
        wf.run(initial_context={"query": "test"})
        # report_writer query should contain outputs from earlier steps
        report_query = agents["report_writer"].run.call_args[0][0]
        assert "[ratio]" in report_query or "analysis done" in report_query


# ---------------------------------------------------------------------------
# create_quick_scan_workflow
# ---------------------------------------------------------------------------


class TestCreateQuickScanWorkflow:
    def test_returns_parallel_workflow(self):
        agents = _agents_fixture()
        wf = create_quick_scan_workflow(agents)
        assert isinstance(wf, ParallelWorkflow)

    def test_has_four_steps(self):
        agents = _agents_fixture()
        wf = create_quick_scan_workflow(agents)
        assert len(wf.steps) == 4

    def test_first_three_steps_have_no_dependencies(self):
        agents = _agents_fixture()
        wf = create_quick_scan_workflow(agents)
        independent = [s for s in wf.steps if not s.depends_on]
        assert len(independent) == 3

    def test_report_step_depends_on_all_three(self):
        agents = _agents_fixture()
        wf = create_quick_scan_workflow(agents)
        report_step = next(s for s in wf.steps if s.output_key == "report")
        assert len(report_step.depends_on) == 3

    def test_runs_successfully(self):
        agents = _agents_fixture()
        wf = create_quick_scan_workflow(agents)
        result = wf.run(initial_context={"query": "quick scan"})
        assert result.success
        assert len(result.steps_completed) == 4


# ---------------------------------------------------------------------------
# create_query_router_workflow
# ---------------------------------------------------------------------------


class TestCreateQueryRouterWorkflow:
    def test_returns_conditional_workflow(self):
        agents = _agents_fixture()
        wf = create_query_router_workflow(agents)
        assert isinstance(wf, ConditionalWorkflow)

    def test_routes_ratio_query(self):
        agents = _agents_fixture()
        wf = create_query_router_workflow(agents)
        result = wf.run("What is the current ratio?")
        assert result.success
        assert agents["ratio_analyst"].run.called

    def test_routes_trend_query(self):
        agents = _agents_fixture()
        wf = create_query_router_workflow(agents)
        result = wf.run("What is the revenue trend?")
        assert result.success
        assert agents["trend_forecaster"].run.called

    def test_routes_risk_query(self):
        agents = _agents_fixture()
        wf = create_query_router_workflow(agents)
        result = wf.run("Assess the risk profile")
        assert result.success
        assert agents["risk_assessor"].run.called

    def test_routes_report_query(self):
        agents = _agents_fixture()
        wf = create_query_router_workflow(agents)
        result = wf.run("Write a summary report")
        assert result.success
        assert agents["report_writer"].run.called

    def test_unknown_query_routes_to_default(self):
        agents = _agents_fixture()
        wf = create_query_router_workflow(agents)
        result = wf.run("completely unrelated question")
        assert result.success
        # Default is ratio_analyst
        assert agents["ratio_analyst"].run.called


# ---------------------------------------------------------------------------
# Integration: all agents mocked end-to-end
# ---------------------------------------------------------------------------


class TestWorkflowWithAllAgentsMocked:
    def test_sequential_full_pipeline(self):
        agents = {
            "ratio_analyst": _make_agent("ratio_analyst", "RATIO_RESULT"),
            "trend_forecaster": _make_agent("trend_forecaster", "TREND_RESULT"),
            "risk_assessor": _make_agent("risk_assessor", "RISK_RESULT"),
            "report_writer": _make_agent("report_writer", "FINAL_REPORT"),
        }
        wf = create_comprehensive_analysis_workflow(agents)
        result = wf.run({"query": "integration test company"})

        assert result.success
        assert result.final_output == "FINAL_REPORT"
        assert result.total_duration_ms >= 0
        assert all(r["error"] is None for r in result.steps_completed)

    def test_parallel_full_pipeline(self):
        agents = {
            "ratio_analyst": _make_agent("ratio_analyst", "RATIO"),
            "risk_assessor": _make_agent("risk_assessor", "RISK"),
            "trend_forecaster": _make_agent("trend_forecaster", "TREND"),
            "report_writer": _make_agent("report_writer", "REPORT"),
        }
        wf = create_quick_scan_workflow(agents)
        result = wf.run({"query": "quick scan integration"})

        assert result.success
        assert len(result.steps_completed) == 4
        assert result.final_output == "REPORT"

    def test_workflow_result_aggregation_success(self):
        agents = _agents_fixture()
        wf = SequentialWorkflow(
            agents=agents,
            steps=[
                WorkflowStep("ratio_analyst", "q1", output_key="r1"),
                WorkflowStep("risk_assessor", "q2", output_key="r2"),
            ],
        )
        result = wf.run()
        assert result.success
        assert len(result.steps_completed) == 2
        assert result.errors == []

    def test_workflow_result_aggregation_with_partial_failure(self):
        bad = MagicMock()
        bad.name = "bad"
        bad.run.side_effect = Exception("partial failure")

        agents = {
            "bad": bad,
            "good": _make_agent("good", "GOOD"),
        }
        wf = SequentialWorkflow(
            agents=agents,
            steps=[
                WorkflowStep("bad", "fail", output_key="bad_r"),
                WorkflowStep("good", "succeed", output_key="good_r"),
            ],
        )
        result = wf.run()
        assert not result.success
        assert len(result.errors) == 1
        assert len(result.steps_completed) == 2
        # good agent still produced output
        assert any(s["result"] == "GOOD" for s in result.steps_completed)

    def test_timing_tracked_per_step(self):
        agents = _agents_fixture()
        wf = SequentialWorkflow(
            agents=agents,
            steps=[
                WorkflowStep("ratio_analyst", "q", output_key="r1"),
                WorkflowStep("risk_assessor", "q", output_key="r2"),
            ],
        )
        result = wf.run()
        for record in result.steps_completed:
            assert "duration_ms" in record
            assert isinstance(record["duration_ms"], float)
            assert record["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# AgentMemory unit tests
# ---------------------------------------------------------------------------


class TestAgentMemory:
    def test_store_and_recall_fact(self):
        mem = AgentMemory()
        mem.store_fact("key", "value")
        assert mem.recall_fact("key") == "value"

    def test_recall_fact_missing_returns_none(self):
        mem = AgentMemory()
        assert mem.recall_fact("nonexistent") is None

    def test_add_message_and_get_recent(self):
        mem = AgentMemory()
        mem.add_message(AgentMessage(role="user", content="hello"))
        mem.add_message(AgentMessage(role="assistant", content="hi"))
        recent = mem.get_recent(10)
        assert len(recent) == 2
        assert recent[0].role == "user"
        assert recent[1].role == "assistant"

    def test_clear_short_term_empties_history(self):
        mem = AgentMemory()
        mem.add_message(AgentMessage(role="user", content="test"))
        mem.clear_short_term()
        assert mem.get_recent(10) == []

    def test_facts_survive_clear_short_term(self):
        mem = AgentMemory()
        mem.store_fact("k", "v")
        mem.clear_short_term()
        assert mem.recall_fact("k") == "v"

    def test_max_messages_respected(self):
        mem = AgentMemory(max_messages=3)
        for i in range(5):
            mem.add_message(AgentMessage(role="user", content=str(i)))
        recent = mem.get_recent(10)
        assert len(recent) == 3

    def test_get_context_summary_empty(self):
        mem = AgentMemory()
        assert mem.get_context_summary() == ""

    def test_get_context_summary_nonempty(self):
        mem = AgentMemory()
        mem.add_message(AgentMessage(role="user", content="hello"))
        summary = mem.get_context_summary()
        assert "user" in summary
        assert "hello" in summary


# ---------------------------------------------------------------------------
# Specialized agent smoke tests (using _StubLLM)
# ---------------------------------------------------------------------------


class TestSpecialisedAgentStubs:
    """Verify that concrete agent stubs respect the BaseAgent interface."""

    def test_ratio_analyst_run_returns_str(self):
        agent = RatioAnalystAgent()
        result = agent.run("What is the current ratio?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ratio_analyst_plan_returns_list(self):
        agent = RatioAnalystAgent()
        steps = agent.plan("ratio query")
        assert isinstance(steps, list)
        assert len(steps) >= 1

    def test_risk_assessor_run_returns_str(self):
        agent = RiskAssessorAgent()
        result = agent.run("Assess risk")
        assert isinstance(result, str)

    def test_trend_forecaster_run_returns_str(self):
        agent = TrendForecasterAgent()
        result = agent.run("Forecast revenue")
        assert isinstance(result, str)

    def test_report_writer_run_returns_str(self):
        agent = ReportWriterAgent()
        result = agent.run("Write report based on analysis")
        assert isinstance(result, str)

    def test_ratio_analyst_stores_fact_in_memory(self):
        agent = RatioAnalystAgent()
        agent.run("test query")
        # The specialized agent stores the result under "ratio_analysis"
        assert agent.memory.recall_fact("ratio_analysis") is not None

    def test_risk_assessor_stores_fact_in_memory(self):
        agent = RiskAssessorAgent()
        agent.run("assess risk test")
        assert agent.memory.recall_fact("risk_assessment") is not None

    def test_trend_forecaster_stores_fact_in_memory(self):
        agent = TrendForecasterAgent()
        agent.run("forecast test")
        assert agent.memory.recall_fact("trend_forecast") is not None

    def test_report_writer_stores_fact_in_memory(self):
        agent = ReportWriterAgent()
        agent.run("write test")
        assert agent.memory.recall_fact("report") is not None

    def test_agent_name_correct(self):
        assert RatioAnalystAgent().name == "ratio_analyst"
        assert RiskAssessorAgent().name == "risk_assessor"
        assert TrendForecasterAgent().name == "trend_forecaster"
        assert ReportWriterAgent().name == "report_writer"
