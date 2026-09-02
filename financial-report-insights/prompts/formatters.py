"""
Prompt formatting utilities for the financial-report-insights RAG system.

These functions handle assembly of the final prompt string from a PromptTemplate,
retrieved context documents, and a user query. They are designed to be stateless
and easily testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from prompts.templates import PromptTemplate

# JSON output schema injected for ratio queries when structured output is needed
_RATIO_JSON_SCHEMA = """\
Respond with a valid JSON object using this schema (in addition to your prose answer):
{
  "ratio_name": "<string>",
  "formula": "<string>",
  "inputs": {"<input_name>": <value>, ...},
  "result": <numeric_value>,
  "unit": "<string, e.g. 'x' or '%'>",
  "interpretation": "<one sentence>"
}
"""

# JSON schemas for other query types (lighter weight)
_TREND_JSON_SCHEMA = """\
Optionally include a JSON summary at the end:
{
  "metric": "<string>",
  "start_value": <numeric>,
  "end_value": <numeric>,
  "change_pct": <numeric>,
  "trend_direction": "improving | deteriorating | stable"
}
"""


def format_context_with_citations(documents: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a numbered, citation-ready context block.

    Each document is prefixed with a [N] citation marker that the model is
    instructed to reference in its answer. If the document already carries a
    '_citation' key (added by reranker.py's add_citations), that value is
    incorporated directly; otherwise a sequential number is assigned.

    Args:
        documents: List of document dicts as returned by SimpleRAG.retrieve().
                   Each dict should have at least 'source' and 'content' keys.
                   Optional keys: '_citation', 'type', 'metadata'.

    Returns:
        A formatted multi-line string with numbered citation blocks, or an
        empty string if the document list is empty.
    """
    if not documents:
        return ""

    parts: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        # Prefer pre-assigned citation markers from reranker; fall back to index
        citation = doc.get("_citation", "")
        if citation:
            # Strip surrounding brackets if present so we can re-wrap uniformly
            citation_num = citation.strip("[]")
            marker = f"[{citation_num}]"
        else:
            marker = f"[{idx}]"

        source = doc.get("source", "Unknown source")
        content = doc.get("content", "").strip()

        # Include financial type metadata when available (helps the model cite precisely)
        metadata = doc.get("metadata", {})
        financial_type = metadata.get("financial_type", "") if isinstance(metadata, dict) else ""
        type_suffix = f" | {financial_type}" if financial_type else ""

        header = f"{marker} Source: {source}{type_suffix}"
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def format_few_shot_examples(examples: List[Dict[str, str]]) -> str:
    """Render a list of few-shot Q&A examples into a delimited prompt block.

    The block uses a clear separator so the model can distinguish examples
    from the live question. Returns an empty string when the list is empty
    so callers can safely include it unconditionally.

    Args:
        examples: List of dicts, each containing 'question' and 'answer' keys.

    Returns:
        Formatted string with all examples, ready for inclusion in a prompt,
        or empty string if no examples are provided.
    """
    if not examples:
        return ""

    lines: List[str] = ["--- EXAMPLES (for format reference) ---\n"]
    for i, ex in enumerate(examples, start=1):
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        lines.append(f"Example {i}:\nQ: {question}\nA: {answer}")

    if len(lines) == 1:
        # No valid examples were found after filtering
        return ""

    lines.append("--- END EXAMPLES ---\n\n")
    return "\n\n".join(lines)


def build_prompt(template: "PromptTemplate", query: str, context: str) -> str:
    """Assemble a complete prompt string from a template, query, and context.

    Injects the few-shot examples from the template, substitutes the context
    and query into the user_template, and prepends the system_prompt in a
    clear format compatible with instruction-following models.

    Args:
        template: A PromptTemplate instance (from templates.py).
        query: The user's question string.
        context: The formatted context block (use format_context_with_citations).

    Returns:
        A fully assembled prompt string ready to pass to an LLM.
    """
    few_shot_block = format_few_shot_examples(template.few_shot_examples)

    user_section = template.user_template.format(
        few_shot_block=few_shot_block,
        context=context,
        query=query,
    )

    return f"SYSTEM:\n{template.system_prompt}\n\nUSER:\n{user_section}"


def format_json_instruction(query_type: str) -> str:
    """Return an optional JSON output format instruction for structured queries.

    For ratio_lookup queries the instruction requests a machine-readable JSON
    object alongside the prose answer, enabling downstream parsing.
    For trend_analysis a lighter JSON summary is suggested.
    For other query types an empty string is returned so nothing is appended.

    Args:
        query_type: The query classification string from _classify_query()
                    (e.g., 'ratio_lookup', 'trend_analysis', 'comparison').

    Returns:
        A format instruction string, or empty string if not applicable.
    """
    if query_type == "ratio_lookup":
        return _RATIO_JSON_SCHEMA
    if query_type == "trend_analysis":
        return _TREND_JSON_SCHEMA
    return ""
