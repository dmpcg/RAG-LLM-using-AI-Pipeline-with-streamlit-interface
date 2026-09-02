"""Agent package for multi-agent financial analysis workflows."""

from agents.base import AgentMemory, AgentMessage, AgentStep, BaseAgent, Tool, ToolCall, ToolRegistry
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
    create_comprehensive_analysis_workflow,
    create_query_router_workflow,
    create_quick_scan_workflow,
)

__all__ = [
    "AgentMemory",
    "AgentMessage",
    "AgentStep",
    "BaseAgent",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "RatioAnalystAgent",
    "RiskAssessorAgent",
    "TrendForecasterAgent",
    "ReportWriterAgent",
    "WorkflowStep",
    "WorkflowResult",
    "SequentialWorkflow",
    "ParallelWorkflow",
    "ConditionalWorkflow",
    "create_comprehensive_analysis_workflow",
    "create_quick_scan_workflow",
    "create_query_router_workflow",
]
