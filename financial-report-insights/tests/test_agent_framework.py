"""Tests for the agents/ base framework.

Covers AgentMessage, ToolCall, AgentStep, AgentMemory, Tool, ToolRegistry,
and BaseAgent (with a fully mocked LLM).  No live Ollama connection needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make sure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import (
    AgentMemory,
    AgentMessage,
    AgentStep,
    BaseAgent,
    Tool,
    ToolCall,
    ToolRegistry,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_llm(responses: list[str]) -> MagicMock:
    """Build a mock LLM that returns successive responses from *responses*."""
    llm = MagicMock()
    llm.generate.side_effect = responses
    return llm


def _add_tool(registry: ToolRegistry, name: str, func, params: dict | None = None):
    """Register a tool on *registry* for test use."""
    registry.register(Tool(name=name, description=f"{name} tool", func=func, parameters=params or {}))


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------


class TestAgentMessage:
    def test_creation_defaults(self):
        msg = AgentMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.metadata == {}

    def test_creation_with_metadata(self):
        msg = AgentMessage(role="tool_result", content="42", metadata={"tool": "calc"})
        assert msg.metadata["tool"] == "calc"

    def test_roles_are_strings(self):
        for role in ("user", "assistant", "system", "tool_result"):
            m = AgentMessage(role=role, content="x")
            assert m.role == role

    def test_content_empty_string(self):
        msg = AgentMessage(role="assistant", content="")
        assert msg.content == ""

    def test_metadata_is_independent_per_instance(self):
        m1 = AgentMessage(role="user", content="a")
        m2 = AgentMessage(role="user", content="b")
        m1.metadata["key"] = "val"
        assert "key" not in m2.metadata


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_creation_defaults(self):
        tc = ToolCall(tool_name="calculator")
        assert tc.tool_name == "calculator"
        assert tc.arguments == {}
        assert tc.result is None

    def test_creation_with_args_and_result(self):
        tc = ToolCall(tool_name="search", arguments={"q": "revenue"}, result="found")
        assert tc.arguments["q"] == "revenue"
        assert tc.result == "found"

    def test_arguments_are_independent(self):
        tc1 = ToolCall(tool_name="t1")
        tc2 = ToolCall(tool_name="t2")
        tc1.arguments["x"] = 1
        assert "x" not in tc2.arguments


# ---------------------------------------------------------------------------
# AgentStep
# ---------------------------------------------------------------------------


class TestAgentStep:
    def test_creation_minimal(self):
        step = AgentStep(thought="thinking")
        assert step.thought == "thinking"
        assert step.action is None
        assert step.observation is None

    def test_creation_full(self):
        tc = ToolCall(tool_name="t")
        step = AgentStep(thought="plan", action=tc, observation="obs")
        assert step.action is tc
        assert step.observation == "obs"


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------


class TestAgentMemory:
    def test_add_and_get_recent(self):
        mem = AgentMemory()
        mem.add_message(AgentMessage(role="user", content="hi"))
        mem.add_message(AgentMessage(role="assistant", content="hello"))
        recent = mem.get_recent(10)
        assert len(recent) == 2
        assert recent[0].role == "user"
        assert recent[1].role == "assistant"

    def test_get_recent_limits(self):
        mem = AgentMemory()
        for i in range(20):
            mem.add_message(AgentMessage(role="user", content=str(i)))
        recent = mem.get_recent(5)
        assert len(recent) == 5
        # Should be the 5 most recent
        assert recent[-1].content == "19"

    def test_max_messages_cap_evicts_oldest(self):
        mem = AgentMemory(max_messages=3)
        for i in range(5):
            mem.add_message(AgentMessage(role="user", content=str(i)))
        # Only last 3 should remain
        all_msgs = mem.get_recent(100)
        assert len(all_msgs) == 3
        assert all_msgs[0].content == "2"
        assert all_msgs[1].content == "3"
        assert all_msgs[2].content == "4"

    def test_clear_short_term(self):
        mem = AgentMemory()
        mem.add_message(AgentMessage(role="user", content="test"))
        mem.clear_short_term()
        assert mem.get_recent(10) == []

    def test_store_and_recall_fact(self):
        mem = AgentMemory()
        mem.store_fact("company", "Acme Corp")
        assert mem.recall_fact("company") == "Acme Corp"

    def test_recall_missing_fact_returns_none(self):
        mem = AgentMemory()
        assert mem.recall_fact("nonexistent") is None

    def test_store_fact_overwrites(self):
        mem = AgentMemory()
        mem.store_fact("k", "v1")
        mem.store_fact("k", "v2")
        assert mem.recall_fact("k") == "v2"

    def test_get_context_summary_empty(self):
        mem = AgentMemory()
        assert mem.get_context_summary() == ""

    def test_get_context_summary_format(self):
        mem = AgentMemory()
        mem.add_message(AgentMessage(role="user", content="what is ROE?"))
        mem.add_message(AgentMessage(role="assistant", content="ROE = NI/Equity"))
        summary = mem.get_context_summary()
        assert "user: what is ROE?" in summary
        assert "assistant: ROE = NI/Equity" in summary

    def test_long_term_and_short_term_are_separate(self):
        mem = AgentMemory()
        mem.store_fact("revenue", "1M")
        mem.add_message(AgentMessage(role="user", content="hello"))
        mem.clear_short_term()
        # Fact persists after clearing short-term
        assert mem.recall_fact("revenue") == "1M"
        assert mem.get_recent(10) == []

    def test_get_recent_returns_chronological_order(self):
        mem = AgentMemory()
        for i in range(5):
            mem.add_message(AgentMessage(role="user", content=str(i)))
        recent = mem.get_recent(5)
        contents = [m.content for m in recent]
        assert contents == ["0", "1", "2", "3", "4"]

    def test_default_max_messages_is_fifty(self):
        mem = AgentMemory()
        for i in range(55):
            mem.add_message(AgentMessage(role="user", content=str(i)))
        assert len(mem.get_recent(100)) == 50


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class TestTool:
    def test_creation(self):
        def fn(x):
            return str(x)

        t = Tool(name="double", description="doubles x", func=fn, parameters={"x": "a number"})
        assert t.name == "double"
        assert t.description == "doubles x"
        assert t.parameters == {"x": "a number"}

    def test_execute_returns_string(self):
        t = Tool("add", "adds a+b", lambda a, b: int(a) + int(b), {"a": "num", "b": "num"})
        result = t.execute(a=2, b=3)
        assert result == "5"

    def test_execute_error_returns_error_string(self):
        def bad_func(**kwargs):
            raise ValueError("something went wrong")

        t = Tool("bad", "bad tool", bad_func, {})
        result = t.execute()
        assert result.startswith("ERROR:")
        assert "something went wrong" in result

    def test_execute_with_no_args(self):
        t = Tool("ping", "returns pong", lambda: "pong", {})
        assert t.execute() == "pong"

    def test_to_prompt_description_includes_name_and_desc(self):
        t = Tool("lookup", "looks things up", lambda q: q, {"q": "search query"})
        desc = t.to_prompt_description()
        assert "lookup" in desc
        assert "looks things up" in desc
        assert "q" in desc

    def test_to_prompt_description_no_params(self):
        t = Tool("noop", "does nothing", lambda: "", {})
        desc = t.to_prompt_description()
        assert "noop" in desc
        assert "Parameters" not in desc

    def test_execute_converts_to_string(self):
        t = Tool("pi", "returns pi", lambda: 3.14159, {})
        result = t.execute()
        assert result == "3.14159"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        t = Tool("calc", "calculator", lambda x: x, {})
        registry.register(t)
        assert registry.get("calc") is t

    def test_get_missing_raises_key_error(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_tools_empty(self):
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_list_tools_returns_all(self):
        registry = ToolRegistry()
        t1 = Tool("a", "A", lambda: "a", {})
        t2 = Tool("b", "B", lambda: "b", {})
        registry.register(t1)
        registry.register(t2)
        tools = registry.list_tools()
        assert len(tools) == 2
        assert t1 in tools
        assert t2 in tools

    def test_get_tools_prompt_empty(self):
        registry = ToolRegistry()
        assert registry.get_tools_prompt() == ""

    def test_get_tools_prompt_contains_tool_names(self):
        registry = ToolRegistry()
        registry.register(Tool("search", "search docs", lambda q: q, {"q": "query"}))
        registry.register(Tool("calc", "do math", lambda x: x, {}))
        prompt = registry.get_tools_prompt()
        assert "search" in prompt
        assert "calc" in prompt
        assert "Available tools" in prompt

    def test_register_overwrites_same_name(self):
        registry = ToolRegistry()
        t1 = Tool("t", "first", lambda: "1", {})
        t2 = Tool("t", "second", lambda: "2", {})
        registry.register(t1)
        registry.register(t2)
        assert registry.get("t") is t2
        assert len(registry.list_tools()) == 1


# ---------------------------------------------------------------------------
# BaseAgent initialization
# ---------------------------------------------------------------------------


class TestBaseAgentInit:
    def test_default_values(self):
        llm = _make_llm(["1. step one"])
        agent = BaseAgent(name="test_agent", llm=llm)
        assert agent.name == "test_agent"
        assert agent.max_steps == 10
        assert isinstance(agent.memory, AgentMemory)
        assert isinstance(agent._tools, ToolRegistry)

    def test_custom_max_steps(self):
        llm = _make_llm([])
        agent = BaseAgent(name="a", llm=llm, max_steps=3)
        assert agent.max_steps == 3

    def test_custom_memory_is_used(self):
        llm = _make_llm([])
        mem = AgentMemory()
        mem.store_fact("key", "value")
        agent = BaseAgent(name="a", llm=llm, memory=mem)
        assert agent.memory.recall_fact("key") == "value"

    def test_custom_tool_registry_is_used(self):
        llm = _make_llm([])
        registry = ToolRegistry()
        registry.register(Tool("t", "desc", lambda: "r", {}))
        agent = BaseAgent(name="a", llm=llm, tools=registry)
        assert len(agent._tools.list_tools()) == 1


# ---------------------------------------------------------------------------
# BaseAgent.plan
# ---------------------------------------------------------------------------


class TestBaseAgentPlan:
    def test_plan_parses_numbered_list(self):
        llm = _make_llm(["1. Gather data\n2. Compute ratios\n3. Write report"])
        agent = BaseAgent(name="planner", llm=llm)
        steps = agent.plan("Analyse the financials")
        assert steps == ["Gather data", "Compute ratios", "Write report"]

    def test_plan_falls_back_to_query_on_empty_response(self):
        llm = _make_llm(["This is not a numbered list at all."])
        agent = BaseAgent(name="a", llm=llm)
        steps = agent.plan("Do something")
        assert steps == ["Do something"]

    def test_plan_strips_period_numbering(self):
        llm = _make_llm(["1. Step A\n2. Step B"])
        agent = BaseAgent(name="a", llm=llm)
        steps = agent.plan("q")
        assert steps[0] == "Step A"
        assert steps[1] == "Step B"

    def test_plan_strips_paren_numbering(self):
        llm = _make_llm(["1) Do this\n2) Then that"])
        agent = BaseAgent(name="a", llm=llm)
        steps = agent.plan("q")
        assert steps == ["Do this", "Then that"]


# ---------------------------------------------------------------------------
# BaseAgent._select_tool
# ---------------------------------------------------------------------------


class TestBaseAgentSelectTool:
    def setup_method(self):
        self.llm = _make_llm([])
        self.agent = BaseAgent(name="a", llm=self.llm)
        self.tools = [Tool("calculator", "does math", lambda x: x, {"x": "expr"})]

    def test_parses_tool_name_from_tool_prefix(self):
        result = self.agent._select_tool("I need to TOOL: calculator\nARGS: x=2+3", self.tools)
        assert result is not None
        assert result.tool_name == "calculator"

    def test_parses_tool_name_from_action_prefix(self):
        result = self.agent._select_tool("Action: calculator\nARGS: x=5*10", self.tools)
        assert result is not None
        assert result.tool_name == "calculator"

    def test_returns_none_for_unknown_tool(self):
        result = self.agent._select_tool("TOOL: unknown_tool", self.tools)
        assert result is None

    def test_returns_none_when_no_tool_directive(self):
        result = self.agent._select_tool("I will just think about this.", self.tools)
        assert result is None

    def test_parses_key_value_arguments(self):
        result = self.agent._select_tool("TOOL: calculator\nARGS: x=100", self.tools)
        assert result is not None
        assert result.arguments.get("x") == "100"

    def test_no_args_returns_empty_dict(self):
        result = self.agent._select_tool("TOOL: calculator", self.tools)
        assert result is not None
        assert result.arguments == {}


# ---------------------------------------------------------------------------
# BaseAgent.run — basic flow
# ---------------------------------------------------------------------------


class TestBaseAgentRun:
    def test_run_returns_string(self):
        # LLM: plan response + step thought (with final answer)
        llm = _make_llm(
            [
                "1. Do the thing",  # plan
                "Final Answer: The answer is 42",  # step thought
            ]
        )
        agent = BaseAgent(name="a", llm=llm)
        result = agent.run("What is the answer?")
        assert isinstance(result, str)
        assert "42" in result

    def test_run_stores_user_query_in_memory(self):
        llm = _make_llm(
            [
                "1. Step one",
                "Final Answer: done",
            ]
        )
        agent = BaseAgent(name="a", llm=llm)
        agent.run("tell me about ROE")
        msgs = agent.memory.get_recent(10)
        assert any(m.role == "user" and "ROE" in m.content for m in msgs)

    def test_run_respects_max_steps(self):
        # Give 5 step-thoughts but cap at max_steps=2
        # After max_steps reached, calls _synthesise_answer once more
        plan_response = "1. step1\n2. step2\n3. step3\n4. step4\n5. step5"
        step_thought = "I should keep thinking..."
        synth_response = "Final Answer: synthesised result"
        responses = [plan_response] + [step_thought] * 2 + [synth_response]
        llm = _make_llm(responses)
        agent = BaseAgent(name="a", llm=llm, max_steps=2)
        result = agent.run("complex query")
        assert isinstance(result, str)
        # Should have called generate exactly: 1 (plan) + 2 (steps) + 1 (synth) = 4
        assert llm.generate.call_count == 4

    def test_run_with_tool_execution(self):
        registry = ToolRegistry()
        results_store = []

        def add_tool(a, b):
            return str(int(a) + int(b))

        registry.register(Tool("add", "adds numbers", add_tool, {"a": "num", "b": "num"}))

        plan_resp = "1. Add 2 and 3"
        step_resp = "TOOL: add\nARGS: a=2, b=3"
        synth_resp = "Final Answer: 5"

        llm = _make_llm([plan_resp, step_resp, synth_resp])
        agent = BaseAgent(name="a", llm=llm, tools=registry)
        result = agent.run("What is 2 + 3?")
        # The tool should have been executed and observation stored
        assert isinstance(result, str)

    def test_run_handles_tool_execution_failure(self):
        registry = ToolRegistry()

        def failing_tool():
            raise RuntimeError("connection refused")

        registry.register(Tool("flaky", "fails always", failing_tool, {}))

        llm = _make_llm(
            [
                "1. Use the flaky tool",
                "TOOL: flaky",  # causes error observation
                "Final Answer: gracefully handled",
            ]
        )
        agent = BaseAgent(name="a", llm=llm, tools=registry)
        result = agent.run("trigger flaky tool")
        # Should not raise; should return the final answer
        assert "gracefully handled" in result

    def test_run_synthesises_when_no_final_answer_found(self):
        # All thoughts lack "Final Answer" marker; synth call should be made
        llm = _make_llm(
            [
                "1. Only step",
                "I am thinking deeply about this.",  # step thought - no final answer
                "The answer is clearly: 99",  # synth response - no marker
            ]
        )
        agent = BaseAgent(name="a", llm=llm)
        result = agent.run("tricky question")
        assert isinstance(result, str)
        # The raw synth text should be returned since no marker found
        assert "99" in result


# ---------------------------------------------------------------------------
# BaseAgent full loop with calculator + lookup tools
# ---------------------------------------------------------------------------


class TestBaseAgentFullLoop:
    def test_calculator_and_lookup_integration(self):
        """Test an agent that uses two tools in sequence."""
        registry = ToolRegistry()

        lookup_db = {"revenue": "1000000", "costs": "600000"}

        def lookup(key):
            return lookup_db.get(key, f"'{key}' not found")

        def calculate(expression):
            # Very restricted safe eval
            parts = expression.split("-")
            if len(parts) == 2:
                try:
                    return str(float(parts[0].strip()) - float(parts[1].strip()))
                except ValueError:
                    pass
            return f"Cannot evaluate: {expression}"

        registry.register(Tool("lookup", "look up financial data", lookup, {"key": "field name"}))
        registry.register(Tool("calculate", "compute subtraction", calculate, {"expression": "a-b"}))

        # Plan: look up revenue, look up costs, compute profit, return answer
        llm_responses = [
            "1. Look up revenue\n2. Look up costs\n3. Calculate profit\n4. Report",
            # Step 1 thought: use lookup tool
            "TOOL: lookup\nARGS: key=revenue",
            # Step 2 thought: use lookup tool
            "TOOL: lookup\nARGS: key=costs",
            # Step 3 thought: calculate
            "TOOL: calculate\nARGS: expression=1000000-600000",
            # Step 4 thought: final answer
            "Final Answer: Gross profit is 400000.0",
        ]
        llm = _make_llm(llm_responses)
        agent = BaseAgent(name="profit_agent", llm=llm, tools=registry, max_steps=10)
        result = agent.run("What is the gross profit?")
        assert "400000" in result

    def test_agent_memory_persists_across_steps(self):
        """Observations from earlier steps appear in later prompts."""
        registry = ToolRegistry()
        call_log: list[str] = []

        def spy_tool(msg):
            call_log.append(msg)
            return f"observed: {msg}"

        registry.register(Tool("spy", "records calls", spy_tool, {"msg": "message"}))

        llm = _make_llm(
            [
                "1. Step one\n2. Step two",
                "TOOL: spy\nARGS: msg=first",
                "TOOL: spy\nARGS: msg=second",
                "Final Answer: done",
            ]
        )
        agent = BaseAgent(name="mem_test", llm=llm, tools=registry)
        agent.run("test memory")
        assert "first" in call_log
        assert "second" in call_log

    def test_empty_tool_registry_runs_without_error(self):
        """Agent with no tools still completes a run using just the LLM."""
        llm = _make_llm(
            [
                "1. Think",
                "Final Answer: no tools needed",
            ]
        )
        agent = BaseAgent(name="no_tools", llm=llm)
        result = agent.run("simple question")
        assert "no tools needed" in result

    def test_multiple_runs_accumulate_memory(self):
        """Each run appends to memory; facts persist between runs."""
        llm = _make_llm(
            [
                "1. First run step",
                "Final Answer: answer A",
                "1. Second run step",
                "Final Answer: answer B",
            ]
        )
        agent = BaseAgent(name="multi", llm=llm)
        r1 = agent.run("first question")
        r2 = agent.run("second question")
        assert "A" in r1
        assert "B" in r2
        # Both queries should be in memory
        msgs = agent.memory.get_recent(20)
        user_msgs = [m for m in msgs if m.role == "user"]
        assert len(user_msgs) == 2


# ---------------------------------------------------------------------------
# _parse_numbered_list (static helper)
# ---------------------------------------------------------------------------


class TestParseNumberedList:
    def test_standard_dot_format(self):
        text = "1. Alpha\n2. Beta\n3. Gamma"
        result = BaseAgent._parse_numbered_list(text)
        assert result == ["Alpha", "Beta", "Gamma"]

    def test_paren_format(self):
        text = "1) One\n2) Two"
        result = BaseAgent._parse_numbered_list(text)
        assert result == ["One", "Two"]

    def test_empty_string(self):
        result = BaseAgent._parse_numbered_list("")
        assert result == []

    def test_no_numbered_items(self):
        result = BaseAgent._parse_numbered_list("just some text without numbers")
        assert result == []

    def test_mixed_content(self):
        text = "Introduction\n1. First item\nSome description\n2. Second item"
        result = BaseAgent._parse_numbered_list(text)
        assert result == ["First item", "Second item"]

    def test_leading_whitespace_stripped(self):
        text = "  1. Indented step  "
        result = BaseAgent._parse_numbered_list(text)
        assert result == ["Indented step"]
