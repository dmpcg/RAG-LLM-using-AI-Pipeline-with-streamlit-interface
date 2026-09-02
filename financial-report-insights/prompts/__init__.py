"""
Prompt engineering module for the financial-report-insights RAG system.

Provides versioned, query-type-specific prompt templates and formatters
that replace hardcoded prompt strings throughout app_local.py.
"""

from prompts.formatters import (
    build_prompt,
    format_context_with_citations,
    format_few_shot_examples,
    format_json_instruction,
)
from prompts.templates import (
    COMPARISON_PROMPT,
    EXPLANATION_PROMPT,
    GENERAL_PROMPT,
    PROMPT_VERSION,
    RATIO_LOOKUP_PROMPT,
    TREND_ANALYSIS_PROMPT,
    PromptTemplate,
    get_prompt_for_query_type,
)

__all__ = [
    "PROMPT_VERSION",
    "PromptTemplate",
    "RATIO_LOOKUP_PROMPT",
    "TREND_ANALYSIS_PROMPT",
    "COMPARISON_PROMPT",
    "EXPLANATION_PROMPT",
    "GENERAL_PROMPT",
    "get_prompt_for_query_type",
    "format_context_with_citations",
    "format_few_shot_examples",
    "build_prompt",
    "format_json_instruction",
]
