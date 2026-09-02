"""Multi-agent workflow orchestration for financial analysis.

Provides three workflow modes:

* :class:`SequentialWorkflow` -- agents execute one after another, with
  each step able to consume the outputs of all previous steps.
* :class:`ParallelWorkflow` -- steps without mutual dependencies run
  concurrently via :class:`~concurrent.futures.ThreadPoolExecutor`;
  dependent steps execute after all their prerequisites complete.
* :class:`ConditionalWorkflow` -- a router function selects which
  agent(s) handle a given query at runtime.

Pre-built factory functions assemble common workflow configurations:

* :func:`create_comprehensive_analysis_workflow`
* :func:`create_quick_scan_workflow`
* :func:`create_query_router_workflow`
"""

from __future__ import annotations

import logging
import string
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WorkflowStep:
    """Declaration of a single agent invocation within a workflow.

    Attributes:
        agent_name: Key in the ``agents`` dict passed to the workflow.
        query_template: Query string with optional ``{placeholder}``
            tokens.  Placeholders are resolved against the accumulated
            workflow context (which includes ``initial_context`` plus the
            ``output_key`` values of all completed steps).
        depends_on: Names of *output_key* values that must be available
            before this step can execute.  An empty list means the step
            may run as soon as the workflow starts.
        output_key: Key under which the step's result is stored in the
            shared context so that later steps can reference it.
    """

    agent_name: str
    query_template: str
    depends_on: List[str] = field(default_factory=list)
    output_key: str = ""

    def __post_init__(self) -> None:
        if not self.output_key:
            self.output_key = self.agent_name


@dataclass
class WorkflowResult:
    """Aggregated outcome of a completed workflow execution.

    Attributes:
        steps_completed: One dict per step that was attempted, each
            containing ``agent_name``, ``query``, ``result``, and
            ``duration_ms``.
        final_output: The result of the last step to complete
            successfully, or an empty string if no step succeeded.
        total_duration_ms: Wall-clock milliseconds for the full workflow.
        success: ``True`` when *every* step completed without raising.
        errors: Human-readable error descriptions for any failed steps.
    """

    steps_completed: List[Dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    total_duration_ms: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_template(template: str, context: Dict[str, Any]) -> str:
    """Substitute ``{key}`` placeholders with values from *context*.

    Unknown placeholders are left verbatim so that callers can detect
    missing dependencies rather than silently inserting empty strings.

    Args:
        template: String that may contain ``{key}`` tokens.
        context: Mapping of available substitution values.

    Returns:
        Template with all resolvable placeholders filled in.
    """
    # Use Formatter to collect field names without raising on missing keys.
    formatter = string.Formatter()
    result_parts: List[str] = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        result_parts.append(literal_text)
        if field_name is not None:
            # Strip attribute/index access (e.g. "key.attr" -> "key")
            root_key = field_name.split(".")[0].split("[")[0]
            if root_key in context:
                value = context[root_key]
                # Apply format_spec if present
                if format_spec:
                    value = format(value, format_spec)
                else:
                    value = str(value)
                # Wrap in sentinel delimiters so the LLM does not
                # interpret embedded TOOL/Action directives from
                # prior agent outputs as real instructions.
                value = f"[USER_CONTEXT_START]{value}[USER_CONTEXT_END]"
                result_parts.append(value)
            else:
                # Leave placeholder intact
                placeholder = "{" + field_name
                if conversion:
                    placeholder += f"!{conversion}"
                if format_spec:
                    placeholder += f":{format_spec}"
                placeholder += "}"
                result_parts.append(placeholder)
    return "".join(result_parts)


def _run_step(
    step: WorkflowStep,
    agent: BaseAgent,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single workflow step and return a step-record dict.

    Args:
        step: The step definition.
        agent: Agent instance to invoke.
        context: Shared workflow context used for template resolution.

    Returns:
        A dict with keys ``agent_name``, ``output_key``, ``query``,
        ``result``, ``duration_ms``, and ``error`` (``None`` on success).

    Raises:
        Does *not* raise -- exceptions are captured and stored under
        ``error`` in the returned dict.
    """
    query = _resolve_template(step.query_template, context)
    start = time.monotonic()
    error: Optional[str] = None
    result = ""
    try:
        result = agent.run(query)
    except Exception as exc:  # noqa: BLE001
        error = f"Step '{step.output_key}' failed: {exc}"
        logger.warning(error)
    duration_ms = (time.monotonic() - start) * 1000.0
    return {
        "agent_name": step.agent_name,
        "output_key": step.output_key,
        "query": query,
        "result": result,
        "duration_ms": duration_ms,
        "error": error,
    }


# ---------------------------------------------------------------------------
# SequentialWorkflow
# ---------------------------------------------------------------------------


class SequentialWorkflow:
    """Run a fixed list of agent steps one after another.

    Each step receives the accumulated context that includes the
    ``initial_context`` plus every ``output_key`` written by prior steps.
    A failed step is recorded in :attr:`WorkflowResult.errors` but does
    **not** abort subsequent steps.

    Args:
        agents: Mapping of agent-name to :class:`~agents.base.BaseAgent`
            instance.  Step ``agent_name`` values must be keys here.
        steps: Ordered list of :class:`WorkflowStep` definitions.

    Example::

        workflow = SequentialWorkflow(
            agents={"analyst": RatioAnalystAgent(), "writer": ReportWriterAgent()},
            steps=[
                WorkflowStep("analyst", "Analyse ratios", output_key="ratios"),
                WorkflowStep("writer", "Write report from {ratios}", output_key="report"),
            ],
        )
        result = workflow.run({"company": "Acme Corp"})
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        steps: List[WorkflowStep],
    ) -> None:
        self.agents = agents
        self.steps = steps

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute all steps in declaration order.

        Args:
            initial_context: Optional seed values made available to the
                first step's template (and all subsequent steps).

        Returns:
            A :class:`WorkflowResult` with one entry per step.
        """
        context: Dict[str, Any] = dict(initial_context or {})
        workflow_start = time.monotonic()
        result = WorkflowResult()

        for step in self.steps:
            agent = self.agents.get(step.agent_name)
            if agent is None:
                msg = f"No agent registered for name '{step.agent_name}'"
                logger.warning(msg)
                result.errors.append(msg)
                result.steps_completed.append(
                    {
                        "agent_name": step.agent_name,
                        "output_key": step.output_key,
                        "query": "",
                        "result": "",
                        "duration_ms": 0.0,
                        "error": msg,
                    }
                )
                continue

            step_record = _run_step(step, agent, context)
            result.steps_completed.append(step_record)

            if step_record["error"]:
                result.errors.append(step_record["error"])
            else:
                # Publish result so later steps can reference it
                context[step.output_key] = step_record["result"]
                result.final_output = step_record["result"]

        result.total_duration_ms = (time.monotonic() - workflow_start) * 1000.0
        result.success = len(result.errors) == 0
        return result


# ---------------------------------------------------------------------------
# ParallelWorkflow
# ---------------------------------------------------------------------------


class ParallelWorkflow:
    """Run independent agent steps concurrently, respecting dependencies.

    Steps with empty ``depends_on`` are submitted to a thread pool in the
    first wave.  As each wave completes its outputs are added to the
    shared context and the next wave of now-unblocked steps is submitted.

    A step is considered *ready* when every key listed in its
    ``depends_on`` is present in the accumulated context.

    Args:
        agents: Mapping of agent-name to :class:`~agents.base.BaseAgent`.
        steps: List of :class:`WorkflowStep` definitions (order does not
            matter -- dependency graph drives execution order).
        max_workers: Maximum threads in the pool (default ``4``).

    Example::

        workflow = ParallelWorkflow(
            agents={...},
            steps=[
                WorkflowStep("profitability_agent", "Analyse profit", output_key="profit"),
                WorkflowStep("liquidity_agent", "Analyse liquidity", output_key="liquidity"),
                WorkflowStep("writer", "Report on {profit} and {liquidity}",
                             depends_on=["profit", "liquidity"], output_key="report"),
            ],
        )
        result = workflow.run()
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        steps: List[WorkflowStep],
        max_workers: int = 4,
    ) -> None:
        self.agents = agents
        self.steps = steps
        self.max_workers = max_workers

    def _build_dependency_levels(self, context_keys: set[str]) -> List[List[WorkflowStep]]:
        """Group steps into sequential execution waves.

        A step moves to the next wave once all its dependencies appear
        either in the initial context or in the output of a prior wave.

        Args:
            context_keys: Keys already present in the initial context.

        Returns:
            Ordered list of waves; each wave is a list of steps that may
            run concurrently.
        """
        remaining = list(self.steps)
        completed_keys = set(context_keys)
        waves: List[List[WorkflowStep]] = []

        while remaining:
            ready = [s for s in remaining if set(s.depends_on) <= completed_keys]
            if not ready:
                # Circular or unsatisfiable dependency -- force remaining steps
                # into a final wave so the workflow does not hang.
                logger.warning(
                    "Unresolvable dependencies for steps: %s. Running them in a final wave.",
                    [s.output_key for s in remaining],
                )
                waves.append(remaining)
                break
            waves.append(ready)
            completed_keys.update(s.output_key for s in ready)
            remaining = [s for s in remaining if s not in ready]

        return waves

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute steps in dependency-ordered waves, each wave in parallel.

        Args:
            initial_context: Seed values for template resolution.

        Returns:
            A :class:`WorkflowResult` with one entry per step.
        """
        context: Dict[str, Any] = dict(initial_context or {})
        workflow_start = time.monotonic()
        result = WorkflowResult()

        waves = self._build_dependency_levels(set(context.keys()))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for wave in waves:
                future_to_step: Dict[Future[Dict[str, Any]], WorkflowStep] = {}

                for step in wave:
                    agent = self.agents.get(step.agent_name)
                    if agent is None:
                        msg = f"No agent registered for name '{step.agent_name}'"
                        logger.warning(msg)
                        result.errors.append(msg)
                        result.steps_completed.append(
                            {
                                "agent_name": step.agent_name,
                                "output_key": step.output_key,
                                "query": "",
                                "result": "",
                                "duration_ms": 0.0,
                                "error": msg,
                            }
                        )
                        continue

                    # Snapshot context at submission time for thread safety
                    ctx_snapshot = dict(context)
                    future = executor.submit(_run_step, step, agent, ctx_snapshot)
                    future_to_step[future] = step

                _STEP_TIMEOUT_SECONDS = 300  # 5-minute per-step timeout
                try:
                    for future in as_completed(future_to_step, timeout=_STEP_TIMEOUT_SECONDS):
                        step_record = future.result()
                        result.steps_completed.append(step_record)
                        if step_record["error"]:
                            result.errors.append(step_record["error"])
                        else:
                            context[step_record["output_key"]] = step_record["result"]
                            result.final_output = step_record["result"]
                except TimeoutError:
                    timed_out = [s.output_key for s in wave if future_to_step.get(future) and not future.done()]
                    msg = f"Wave timed out after {_STEP_TIMEOUT_SECONDS}s; incomplete steps: {timed_out}"
                    logger.warning(msg)
                    result.errors.append(msg)

        result.total_duration_ms = (time.monotonic() - workflow_start) * 1000.0
        result.success = len(result.errors) == 0
        return result


# ---------------------------------------------------------------------------
# ConditionalWorkflow
# ---------------------------------------------------------------------------


class ConditionalWorkflow:
    """Route queries to different agents based on a router function.

    The *router_func* inspects the query (and optionally the context) and
    returns a list of agent names to invoke.  Selected agents run
    concurrently; their results are concatenated in the
    :class:`WorkflowResult`.

    Args:
        agents: Mapping of agent-name to :class:`~agents.base.BaseAgent`.
        router_func: Callable ``(query: str) -> list[str]`` that returns
            the names of agents to invoke.  May return an empty list to
            produce a no-op result.

    Example::

        def my_router(query: str) -> list[str]:
            if "ratio" in query.lower():
                return ["ratio_analyst"]
            return ["risk_assessor"]

        workflow = ConditionalWorkflow(agents={...}, router_func=my_router)
        result = workflow.run("What are the current ratios?")
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        router_func: Callable[[str], List[str]],
    ) -> None:
        self.agents = agents
        self.router_func = router_func

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Route *query* to the appropriate agents and collect results.

        Args:
            query: The user query to route.
            context: Optional additional context for template resolution
                (not used for routing, only for template substitution if
                individual steps had templates -- here the raw *query* is
                passed directly).

        Returns:
            A :class:`WorkflowResult` with one entry per invoked agent.
        """
        ctx: Dict[str, Any] = dict(context or {})
        workflow_start = time.monotonic()
        result = WorkflowResult()

        try:
            selected_names = self.router_func(query)
        except Exception as exc:  # noqa: BLE001
            msg = f"Router function raised an exception: {exc}"
            logger.warning(msg)
            result.errors.append(msg)
            result.success = False
            result.total_duration_ms = (time.monotonic() - workflow_start) * 1000.0
            return result

        if not selected_names:
            logger.debug("Router returned no agents for query: %r", query)
            result.total_duration_ms = (time.monotonic() - workflow_start) * 1000.0
            # No agents selected is treated as a successful (empty) result
            result.success = True
            return result

        for name in selected_names:
            agent = self.agents.get(name)
            if agent is None:
                msg = f"Router selected unknown agent '{name}'"
                logger.warning(msg)
                result.errors.append(msg)
                result.steps_completed.append(
                    {
                        "agent_name": name,
                        "output_key": name,
                        "query": query,
                        "result": "",
                        "duration_ms": 0.0,
                        "error": msg,
                    }
                )
                continue

            step = WorkflowStep(
                agent_name=name,
                query_template=query,
                output_key=name,
            )
            step_record = _run_step(step, agent, ctx)
            result.steps_completed.append(step_record)
            if step_record["error"]:
                result.errors.append(step_record["error"])
            else:
                ctx[name] = step_record["result"]
                result.final_output = step_record["result"]

        result.total_duration_ms = (time.monotonic() - workflow_start) * 1000.0
        result.success = len(result.errors) == 0
        return result


# ---------------------------------------------------------------------------
# Pre-built workflow factories
# ---------------------------------------------------------------------------


def create_comprehensive_analysis_workflow(
    agents: Dict[str, BaseAgent],
) -> SequentialWorkflow:
    """Build a four-step sequential workflow for deep financial analysis.

    Steps (in order):

    1. ``ratio_analyst`` -- Compute and interpret financial ratios.
    2. ``trend_forecaster`` -- Identify trends using ratio results.
    3. ``risk_assessor`` -- Assess risk using ratios and trends.
    4. ``report_writer`` -- Synthesise all findings into a report.

    Args:
        agents: Must contain keys ``"ratio_analyst"``,
            ``"trend_forecaster"``, ``"risk_assessor"``, and
            ``"report_writer"``.

    Returns:
        A configured :class:`SequentialWorkflow` ready to call ``.run()``.
    """
    steps = [
        WorkflowStep(
            agent_name="ratio_analyst",
            query_template="Perform comprehensive ratio analysis. Context: {query}",
            depends_on=[],
            output_key="ratio_analysis",
        ),
        WorkflowStep(
            agent_name="trend_forecaster",
            query_template=("Identify trends based on the following ratio analysis: {ratio_analysis}"),
            depends_on=["ratio_analysis"],
            output_key="trend_analysis",
        ),
        WorkflowStep(
            agent_name="risk_assessor",
            query_template=("Assess risk given ratios: {ratio_analysis} and trends: {trend_analysis}"),
            depends_on=["ratio_analysis", "trend_analysis"],
            output_key="risk_assessment",
        ),
        WorkflowStep(
            agent_name="report_writer",
            query_template=(
                "Synthesise a comprehensive report from: "
                "Ratios: {ratio_analysis} | "
                "Trends: {trend_analysis} | "
                "Risk: {risk_assessment}"
            ),
            depends_on=["ratio_analysis", "trend_analysis", "risk_assessment"],
            output_key="report",
        ),
    ]
    return SequentialWorkflow(agents=agents, steps=steps)


def create_quick_scan_workflow(
    agents: Dict[str, BaseAgent],
) -> ParallelWorkflow:
    """Build a parallel quick-scan workflow that runs three analyses concurrently.

    Steps:

    * Wave 1 (parallel): ``ratio_analyst``, ``risk_assessor``,
      ``trend_forecaster``.
    * Wave 2 (sequential): ``report_writer`` -- waits for all wave-1 outputs.

    Args:
        agents: Must contain keys ``"ratio_analyst"``, ``"risk_assessor"``,
            ``"trend_forecaster"``, and ``"report_writer"``.

    Returns:
        A configured :class:`ParallelWorkflow` ready to call ``.run()``.
    """
    steps = [
        WorkflowStep(
            agent_name="ratio_analyst",
            query_template="Quick ratio scan. Context: {query}",
            depends_on=[],
            output_key="ratio_analysis",
        ),
        WorkflowStep(
            agent_name="risk_assessor",
            query_template="Quick risk scan. Context: {query}",
            depends_on=[],
            output_key="risk_assessment",
        ),
        WorkflowStep(
            agent_name="trend_forecaster",
            query_template="Quick trend scan. Context: {query}",
            depends_on=[],
            output_key="trend_analysis",
        ),
        WorkflowStep(
            agent_name="report_writer",
            query_template=("Quick report from: {ratio_analysis} | {risk_assessment} | {trend_analysis}"),
            depends_on=["ratio_analysis", "risk_assessment", "trend_analysis"],
            output_key="report",
        ),
    ]
    return ParallelWorkflow(agents=agents, steps=steps, max_workers=4)


def create_query_router_workflow(
    agents: Dict[str, BaseAgent],
) -> ConditionalWorkflow:
    """Build a conditional workflow that routes queries by keyword.

    Routing rules:

    * Contains "ratio" or "metric" or "financial" -> ``ratio_analyst``
    * Contains "trend" or "forecast" or "predict" -> ``trend_forecaster``
    * Contains "risk" or "danger" or "threat" -> ``risk_assessor``
    * Contains "report" or "summary" or "write" -> ``report_writer``
    * Otherwise -> ``ratio_analyst`` (safe default)

    Args:
        agents: Mapping of agent names to instances.

    Returns:
        A configured :class:`ConditionalWorkflow` ready to call ``.run()``.
    """

    def _router(query: str) -> List[str]:
        q_lower = query.lower()
        if any(kw in q_lower for kw in ("ratio", "metric", "financial")):
            return ["ratio_analyst"]
        if any(kw in q_lower for kw in ("trend", "forecast", "predict")):
            return ["trend_forecaster"]
        if any(kw in q_lower for kw in ("risk", "danger", "threat")):
            return ["risk_assessor"]
        if any(kw in q_lower for kw in ("report", "summary", "write")):
            return ["report_writer"]
        # Default route
        return ["ratio_analyst"]

    return ConditionalWorkflow(agents=agents, router_func=_router)
