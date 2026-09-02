"""
Base agent framework: messages, memory, tools, and the core agent loop.

All components are in-process and dependency-injected.  No external agent
frameworks (no langchain, no crewai).  Tool selection uses regex parsing of
LLM output for patterns like "TOOL: tool_name" or "Action: tool_name".
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from protocols import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    """A single message in the agent conversation.

    Args:
        role: One of "user", "assistant", "system", or "tool_result".
        content: The text content of the message.
        metadata: Optional dict with arbitrary extra context (tool name,
            step index, etc.).
    """

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A record of one tool invocation.

    Args:
        tool_name: Name of the tool that was (or should be) called.
        arguments: Keyword arguments passed to the tool.
        result: String result returned by the tool, or None if not yet run.
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None


@dataclass
class AgentStep:
    """One think-act-observe cycle in the agent loop.

    Args:
        thought: The agent's reasoning text for this step.
        action: The tool call the agent decided to make (or None if no tool).
        observation: The result observed after executing the action.
    """

    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[str] = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class AgentMemory:
    """Dual-layer agent memory: short-term conversation history and long-term facts.

    Short-term memory holds a rolling window of ``AgentMessage`` objects
    (oldest entries are evicted once ``max_messages`` is exceeded).
    Long-term memory is a plain dict that persists for the lifetime of the
    instance.

    Args:
        max_messages: Maximum number of messages kept in short-term memory.
    """

    def __init__(self, max_messages: int = 50) -> None:
        self._max_messages = max_messages
        self._history: deque[AgentMessage] = deque(maxlen=max_messages)
        self._facts: dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Short-term (conversation history)
    # ------------------------------------------------------------------

    def add_message(self, msg: AgentMessage) -> None:
        """Append a message to short-term history.

        When the deque is full the oldest message is automatically discarded.

        Args:
            msg: The message to store.
        """
        with self._lock:
            self._history.append(msg)

    def get_recent(self, n: int = 10) -> list[AgentMessage]:
        """Return the *n* most recent messages in chronological order.

        Args:
            n: How many messages to retrieve.

        Returns:
            List of up to *n* messages, oldest first.
        """
        history_list = list(self._history)
        return history_list[-n:] if n < len(history_list) else history_list

    def clear_short_term(self) -> None:
        """Discard all short-term (conversation) history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Long-term (facts / key-value store)
    # ------------------------------------------------------------------

    def store_fact(self, key: str, value: str) -> None:
        """Persist a named fact in long-term memory.

        Args:
            key: Unique identifier for the fact.
            value: String value to associate with *key*.
        """
        with self._lock:
            self._facts[key] = value

    def recall_fact(self, key: str) -> Optional[str]:
        """Look up a fact by key.

        Args:
            key: The fact identifier.

        Returns:
            The stored value, or ``None`` if the key is not present.
        """
        return self._facts.get(key)

    # ------------------------------------------------------------------
    # Summarisation helpers
    # ------------------------------------------------------------------

    def get_context_summary(self) -> str:
        """Build a compact text summary of recent conversation for LLM prompts.

        Returns:
            A multi-line string of ``role: content`` pairs for recent messages,
            or an empty string when history is empty.
        """
        recent = self.get_recent(10)
        if not recent:
            return ""
        lines = [f"{msg.role}: {msg.content}" for msg in recent]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class Tool:
    """A named, typed capability the agent can invoke.

    Args:
        name: Machine-readable tool identifier (no spaces).
        description: Human-readable description used in LLM prompts.
        func: Callable that executes the tool logic and returns a string.
        parameters: Dict describing expected parameters in the form
            ``{param_name: description_string}``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        parameters: dict[str, str],
    ) -> None:
        self.name = name
        self.description = description
        self._func = func
        self.parameters = parameters

    def execute(self, **kwargs: Any) -> str:
        """Invoke the underlying callable and return its string result.

        If the callable raises, the exception message is returned as a
        prefixed error string so the agent can observe what went wrong
        without crashing the loop.

        Args:
            **kwargs: Arguments forwarded to the underlying callable.

        Returns:
            String result from the callable, or an ``"ERROR: ..."`` string on
            failure.
        """
        try:
            result = self._func(**kwargs)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool %r raised: %s", self.name, exc)
            return f"ERROR: {exc}"

    def to_prompt_description(self) -> str:
        """Format the tool signature for inclusion in an LLM prompt.

        Returns:
            A multi-line string describing the tool name, purpose, and
            parameters.
        """
        param_lines = "\n".join(f"    - {pname}: {pdesc}" for pname, pdesc in self.parameters.items())
        params_section = f"\n  Parameters:\n{param_lines}" if param_lines else ""
        return f"- {self.name}: {self.description}{params_section}"


class ToolRegistry:
    """Registry that maps tool names to ``Tool`` instances.

    Tools must be registered before the agent can select them.  Name lookup
    is case-sensitive.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        Args:
            tool: The tool to register.  Overwrites any prior tool with the
                same name.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name.

        Args:
            name: The tool's registered name.

        Returns:
            The corresponding ``Tool`` instance.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        """Return all registered tools in insertion order.

        Returns:
            List of ``Tool`` objects.
        """
        return list(self._tools.values())

    def get_tools_prompt(self) -> str:
        """Format all registered tools for an LLM system prompt.

        Returns:
            Multi-line string listing every tool with its description and
            parameters, or an empty string when no tools are registered.
        """
        if not self._tools:
            return ""
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(tool.to_prompt_description())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regex patterns for LLM output parsing
# ---------------------------------------------------------------------------

_TOOL_PATTERN = re.compile(
    r"(?:TOOL|Action|Use\s+tool|Calling)\s*[:\-]\s*(\w[\w._-]*)",
    re.IGNORECASE,
)

_ARG_PATTERN = re.compile(
    r"(?:ARGS?|Arguments?|Parameters?|Input)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)

_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:Final\s+Answer|Answer|FINAL)\s*[:\-]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent:
    """Minimal ReAct-style agent: plan -> think -> act -> observe -> answer.

    The agent uses an ``LLMProvider`` for all text generation.  Tool
    selection is driven by regex matching on the LLM's output rather than
    structured function-calling, making it compatible with any model that
    follows the prompted format.

    Args:
        name: Human-readable label for logging.
        llm: An object satisfying ``LLMProvider`` (must expose ``generate``).
        tools: Optional ``ToolRegistry`` with available tools.
        memory: Optional ``AgentMemory`` instance.  A fresh one is created
            when not supplied.
        max_steps: Hard cap on think-act-observe iterations per ``run`` call.
    """

    # Patterns that could trick the LLM output parser into invoking tools
    # or emitting a final answer when embedded in user-supplied text.
    _DIRECTIVE_SANITIZE_RE = re.compile(
        r"^[ \t]*(?:TOOL|Tool|Action|ARGS?|Arguments?|Parameters?|Input"
        r"|FINAL_ANSWER|Final\s+Answer|Use\s+tool|Calling)\s*[:\-]",
        re.IGNORECASE | re.MULTILINE,
    )

    @staticmethod
    def _sanitize_user_input(text: str) -> str:
        """Strip tool/action directive patterns from untrusted input.

        This prevents prompt-injection attacks where a user query contains
        lines like ``TOOL: calculate_ratio`` that would be picked up by
        :meth:`_select_tool` when the text is later included in an LLM
        prompt.

        Args:
            text: Raw user or context string.

        Returns:
            Sanitised copy with directive prefixes neutralised.
        """
        sanitized_lines: list[str] = []
        for line in text.splitlines():
            if BaseAgent._DIRECTIVE_SANITIZE_RE.match(line):
                # Prefix with a visible marker so the content is preserved
                # for the LLM to read, but won't match the directive regex.
                sanitized_lines.append(f"[SANITIZED] {line.lstrip()}")
            else:
                sanitized_lines.append(line)
        return "\n".join(sanitized_lines)

    def __init__(
        self,
        name: str,
        llm: LLMProvider,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
        max_steps: int = 10,
    ) -> None:
        self.name = name
        self._llm = llm
        self._tools = tools or ToolRegistry()
        self.memory = memory or AgentMemory()
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, query: str) -> list[str]:
        """Ask the LLM to decompose *query* into an ordered list of steps.

        Args:
            query: The user's request.

        Returns:
            List of step strings (at least one element).  Falls back to a
            single-step list ``[query]`` if the LLM output cannot be parsed.
        """
        safe_query = self._sanitize_user_input(query)
        prompt = (
            "Break the following task into a numbered list of concrete steps.\n"
            "Respond ONLY with a numbered list, one step per line.\n\n"
            f"Task: {safe_query}"
        )
        raw = self._llm.generate(prompt)
        steps = self._parse_numbered_list(raw)
        if not steps:
            logger.debug("plan(): could not parse steps, using query as single step")
            steps = [query]
        logger.debug("Agent %r planned %d step(s) for query", self.name, len(steps))
        return steps

    def execute_step(self, step: str) -> AgentStep:
        """Run one think-act-observe cycle for a single *step*.

        The LLM is asked to reason about the step, optionally select a tool,
        and we execute that tool and record the observation.

        Args:
            step: A natural-language description of the sub-task.

        Returns:
            An ``AgentStep`` with the thought, optional tool call, and
            observation.
        """
        tools_prompt = self._tools.get_tools_prompt()
        context = self.memory.get_context_summary()

        prompt = self._format_step_prompt(step, tools_prompt, context)
        thought_text = self._llm.generate(prompt)

        tool_call = self._select_tool(thought_text, self._tools.list_tools())
        observation: Optional[str] = None

        if tool_call is not None:
            tool = self._tools._tools.get(tool_call.tool_name)
            if tool is not None:
                observation = tool.execute(**tool_call.arguments)
                tool_call.result = observation
                logger.debug(
                    "Agent %r executed tool %r -> %s",
                    self.name,
                    tool_call.tool_name,
                    (observation or "")[:80],
                )
            else:
                observation = f"ERROR: tool '{tool_call.tool_name}' is not registered"
                tool_call.result = observation

        step_result = AgentStep(thought=thought_text, action=tool_call, observation=observation)
        # Record in memory so subsequent steps have context
        mem_content = thought_text
        if observation:
            mem_content += f"\nObservation: {observation}"
        self.memory.add_message(AgentMessage(role="assistant", content=mem_content))
        return step_result

    def run(self, query: str) -> str:
        """Execute the full agent loop for *query* and return a final answer.

        Steps:

        1. Record the user query in memory.
        2. Call ``plan()`` to decompose into sub-steps.
        3. Iterate through steps, calling ``execute_step()`` for each.
        4. Stop early if the LLM produces a "Final Answer: ..." line.
        5. If ``max_steps`` is reached, synthesise the best answer from
           accumulated observations.

        Args:
            query: The user's natural-language request.

        Returns:
            A string containing the agent's final answer.
        """
        safe_query = self._sanitize_user_input(query)
        self.memory.add_message(AgentMessage(role="user", content=safe_query))
        logger.info("Agent %r starting run for: %s", self.name, safe_query[:80])

        steps = self.plan(safe_query)
        completed_steps: list[AgentStep] = []

        for i, step in enumerate(steps):
            if i >= self.max_steps:
                logger.warning(
                    "Agent %r reached max_steps=%d, synthesising answer",
                    self.name,
                    self.max_steps,
                )
                break

            agent_step = self.execute_step(step)
            completed_steps.append(agent_step)

            # Check if the LLM already provided a final answer inside the thought
            final = self._extract_final_answer(agent_step.thought)
            if final:
                self.memory.add_message(AgentMessage(role="assistant", content=final))
                return final

        # Synthesise from all observations when no inline final answer was found
        return self._synthesise_answer(query, completed_steps)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_tool(self, thought: str, available_tools: list[Tool]) -> Optional[ToolCall]:
        """Parse *thought* text for a tool-invocation directive.

        Looks for ``TOOL: tool_name`` / ``Action: tool_name`` patterns
        followed by optional ``ARGS: key=value`` pairs.

        Args:
            thought: The LLM's raw output for a reasoning step.
            available_tools: Tools currently registered (used to validate the
                parsed name).

        Returns:
            A ``ToolCall`` with name and parsed arguments, or ``None`` if no
            valid tool directive is found.
        """
        tool_match = _TOOL_PATTERN.search(thought)
        if tool_match is None:
            return None

        tool_name = tool_match.group(1).strip()
        # Validate against registered tool names
        registered_names = {t.name for t in available_tools}
        if tool_name not in registered_names:
            logger.debug(
                "_select_tool: %r not in registered tools %s",
                tool_name,
                registered_names,
            )
            return None

        arguments = self._parse_tool_arguments(thought)
        return ToolCall(tool_name=tool_name, arguments=arguments)

    def _parse_tool_arguments(self, text: str) -> dict[str, Any]:
        """Extract ``key=value`` pairs from an ARGS/Arguments line in *text*.

        Args:
            text: LLM output that may contain an arguments directive.

        Returns:
            Dict of parsed key-value pairs (values are strings).
        """
        arg_match = _ARG_PATTERN.search(text)
        if arg_match is None:
            return {}

        raw_args = arg_match.group(1).strip()
        # Try JSON first for structured arguments
        try:
            import json

            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback to key=value parsing
        arguments: dict[str, Any] = {}
        for pair in re.split(r"[,\s]+", raw_args):
            if "=" in pair:
                k, _, v = pair.partition("=")
                arguments[k.strip()] = v.strip()
        return arguments

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Return the final answer if *text* contains a recognized marker.

        Args:
            text: LLM output to search.

        Returns:
            Extracted answer string, or ``None``.
        """
        match = _FINAL_ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _format_prompt(self, query: str, steps_so_far: list[AgentStep]) -> str:
        """Build a multi-turn agent prompt with all prior steps and observations.

        Args:
            query: The original user query.
            steps_so_far: Steps completed in the current run.

        Returns:
            Formatted prompt string for the LLM.
        """
        tools_prompt = self._tools.get_tools_prompt()
        parts: list[str] = []

        if tools_prompt:
            parts.append(tools_prompt)
            parts.append(
                "\nTo use a tool, output exactly:\nTOOL: <tool_name>\nARGS: <key=value> pairs separated by commas\n"
            )

        parts.append(f"Question: {query}\n")

        for step in steps_so_far:
            parts.append(f"Thought: {step.thought}")
            if step.action:
                parts.append(f"TOOL: {step.action.tool_name}")
            if step.observation:
                parts.append(f"Observation: {step.observation}")

        parts.append("Provide a Final Answer using the format:\nFinal Answer: <your answer here>")
        return "\n".join(parts)

    def _format_step_prompt(self, step: str, tools_prompt: str, context: str) -> str:
        """Build the per-step reasoning prompt.

        Args:
            step: The specific sub-task to address.
            tools_prompt: Formatted tool listing.
            context: Recent conversation summary from memory.

        Returns:
            Formatted prompt string for the LLM.
        """
        parts: list[str] = []

        if context:
            parts.append(f"Conversation so far:\n{context}\n")

        if tools_prompt:
            parts.append(tools_prompt)
            parts.append(
                "\nTo use a tool write:\n"
                "TOOL: <tool_name>\n"
                "ARGS: key=value, key2=value2\n"
                "If no tool is needed, just write your reasoning and end with:\n"
                "Final Answer: <answer>\n"
            )
        else:
            parts.append("No tools are available. Reason through the task and end with:\nFinal Answer: <answer>\n")

        parts.append(f"Task: {self._sanitize_user_input(step)}")
        return "\n".join(parts)

    def _synthesise_answer(self, query: str, steps: list[AgentStep]) -> str:
        """Ask the LLM to produce a final answer from accumulated observations.

        Args:
            query: The original user query.
            steps: All completed ``AgentStep`` objects.

        Returns:
            Synthesised answer string.
        """
        observations = "\n".join(f"- {s.observation}" for s in steps if s.observation)
        thoughts = "\n".join(f"- {s.thought[:200]}" for s in steps if s.thought)

        prompt = f"You have been working on: {query}\n\nReasoning so far:\n{thoughts}\n\n"
        if observations:
            prompt += f"Tool observations:\n{observations}\n\n"

        prompt += "Based on the above, provide a concise Final Answer."

        raw = self._llm.generate(prompt)
        final = self._extract_final_answer(raw)
        if final:
            return final
        return raw.strip()

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        """Extract items from a numbered list in *text*.

        Args:
            text: LLM output expected to contain ``1. item`` lines.

        Returns:
            List of step strings stripped of leading numbers and whitespace.
        """
        steps: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"^\d+[.)]\s*(.+)", line)
            if m:
                steps.append(m.group(1).strip())
        return steps
