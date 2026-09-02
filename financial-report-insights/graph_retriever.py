"""
Graph-augmented financial retrieval.

Extends standard RAG retrieval with graph traversal capabilities:
- Vector search for relevant text chunks (via Neo4j HNSW or numpy)
- Graph traversal for connected financial metrics, ratios, and scores
- Structured context injection alongside text chunks
"""

import logging
import re
from typing import Dict, List, Optional

from structured_types import GraphChunk, GraphFinancialContext, GraphRetrievalResult

logger = logging.getLogger(__name__)


def graph_enhanced_search(
    store,
    query_embedding: List[float],
    top_k: int = 5,
    model_name: str = "mxbai-embed-large",
) -> GraphRetrievalResult:
    """Perform vector search with graph-traversal enrichment.

    Returns both text chunks and structured financial data connected
    to those chunks via the graph.

    Args:
        store: A Neo4jStore instance.
        query_embedding: Embedded query vector.
        top_k: Number of chunks to retrieve.
        model_name: Embedding model name (for index lookup).

    Returns:
        GraphRetrievalResult with typed chunks and financial_context.
    """
    results = store.graph_search(query_embedding, top_k, model_name)

    chunks = []
    financial_context = []

    for r in results:
        chunks.append(
            GraphChunk(
                source=r.get("source", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
        )

        # Collect connected financial data
        ratios = r.get("ratios", [])
        scores = r.get("scores", [])
        if ratios or scores:
            financial_context.append(
                GraphFinancialContext(
                    document=r.get("document", ""),
                    period=r.get("period", ""),
                    ratios=[rt for rt in ratios if rt.get("name")],
                    scores=[sc for sc in scores if sc.get("model")],
                )
            )

    return GraphRetrievalResult(chunks=chunks, financial_context=financial_context)


def format_graph_context(financial_context: list) -> str:
    """Format graph-retrieved financial context as text for LLM prompts.

    Args:
        financial_context: List of GraphFinancialContext or dicts.

    Returns:
        Formatted string ready for prompt injection.
    """
    if not financial_context:
        return ""

    parts = []
    for ctx in financial_context:
        if isinstance(ctx, dict):
            doc = ctx.get("document", "Unknown")
            period = ctx.get("period", "Unknown")
        else:
            doc = ctx.document or "Unknown"
            period = ctx.period or "Unknown"
        header = f"[{doc} / {period}]"

        ratios = ctx.get("ratios", []) if isinstance(ctx, dict) else ctx.ratios
        scores = ctx.get("scores", []) if isinstance(ctx, dict) else ctx.scores

        ratio_lines = []
        for r in ratios:
            name = r.get("name", "")
            value = r.get("value", "N/A")
            category = r.get("category", "")
            ratio_lines.append(f"  - {name}: {value} ({category})")

        score_lines = []
        for s in scores:
            model = s.get("model", "")
            value = s.get("value", "N/A")
            grade = s.get("grade", "")
            score_lines.append(f"  - {model}: {value} (Grade: {grade})")

        section = header
        if ratio_lines:
            section += "\n  Ratios:\n" + "\n".join(ratio_lines)
        if score_lines:
            section += "\n  Scores:\n" + "\n".join(score_lines)

        parts.append(section)

    return "\n\n".join(parts)


def persist_analysis_to_graph(
    store,
    doc_id: str,
    period_label: str,
    report,
) -> None:
    """Persist CharlieAnalyzer report results as graph nodes.

    Extracts ratios and scores from a FinancialReport and stores them
    in Neo4j for later graph-based retrieval.

    Args:
        store: A Neo4jStore instance.
        doc_id: Source document identifier.
        period_label: Fiscal period label (e.g., "FY2024").
        report: FinancialReport instance from CharlieAnalyzer.
    """
    if not store:
        return

    ratios = {}
    scores = {}

    # Extract ratio data from report sections
    ratio_section = report.sections.get("ratio_analysis", "")
    if ratio_section:
        for line in ratio_section.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("="):
                parts = line.split(":", 1)
                # Sanitize name extracted from LLM text to prevent property injection
                name = re.sub(r"[^a-zA-Z0-9_ -]", "", parts[0].strip().lower().replace(" ", "_"))[:100]
                value_str = parts[1].strip()
                try:
                    # Try to parse numeric value
                    value = float(value_str.replace("%", "").replace("x", "").split()[0])
                    ratios[name] = {"value": value, "category": "computed"}
                except (ValueError, IndexError):
                    pass

    # Extract scoring data from report sections
    score_section = report.sections.get("scoring_models", "")
    if score_section:
        for line in score_section.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("="):
                parts = line.split(":", 1)
                # Sanitize model name extracted from LLM text to prevent property injection
                model = re.sub(r"[^a-zA-Z0-9_ -]", "", parts[0].strip().lower().replace(" ", "_"))[:100]
                rest = parts[1].strip()
                try:
                    value = float(rest.split()[0])
                    grade = ""
                    if "(" in rest and ")" in rest:
                        grade = rest[rest.index("(") + 1 : rest.index(")")]
                    scores[model] = {"value": value, "grade": grade, "interpretation": rest}
                except (ValueError, IndexError):
                    pass

    if ratios or scores:
        store.store_financial_data(
            doc_id=doc_id,
            period_label=period_label,
            ratios=ratios,
            scores=scores,
        )
        logger.info("Persisted %d ratios and %d scores for %s/%s", len(ratios), len(scores), doc_id, period_label)


def persist_structured_analysis_to_graph(
    store,
    doc_id: str,
    period_label: str,
    financial_data=None,
    ratio_results: Optional[Dict] = None,
) -> None:
    """Persist structured financial data directly (no text parsing).

    Unlike persist_analysis_to_graph which parses report text, this function
    accepts structured FinancialData and ratio results directly.

    Args:
        store: A Neo4jStore instance.
        doc_id: Source document identifier.
        period_label: Fiscal period label (e.g., "FY2024").
        financial_data: FinancialData instance for line item population.
        ratio_results: Dict from run_all_ratios() or similar structured output.
    """
    if not store:
        return

    import hashlib

    period_id = hashlib.sha256(f"{doc_id}:{period_label}".encode()).hexdigest()

    # Store ratios and scores via existing store_financial_data
    ratios = {}
    scores = {}

    if ratio_results:
        for key, result in ratio_results.items():
            if hasattr(result, "value") and result.value is not None:
                ratios[result.name] = {
                    "value": result.value,
                    "category": getattr(result, "category", "computed"),
                }

    if ratios:
        store.store_financial_data(
            doc_id=doc_id,
            period_label=period_label,
            ratios=ratios,
            scores=scores,
        )

    # Store line items from FinancialData
    if financial_data is not None:
        store.store_line_items(financial_data, period_id)
        store.store_derived_from_edges(period_id)

    logger.info(
        "Persisted structured analysis for %s/%s (%d ratios, line_items=%s)",
        doc_id,
        period_label,
        len(ratios),
        financial_data is not None,
    )
