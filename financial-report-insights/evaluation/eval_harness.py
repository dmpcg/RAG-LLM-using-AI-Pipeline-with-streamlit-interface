"""RAG Evaluation Harness.

Orchestrates retrieval and answer quality evaluation against a golden Q&A
dataset.  Gated by ``settings.enable_evaluation`` (default ``False``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import settings
from evaluation.answer_metrics import (
    completeness_score,
    faithfulness_score,
    relevance_score,
)
from evaluation.golden_qa import GOLDEN_QA_PAIRS, GoldenQA
from evaluation.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Comprehensive evaluation report.

    Attributes:
        retrieval_metrics: Aggregated retrieval quality metrics.
        answer_metrics: Aggregated answer quality metrics.
        per_query_results: Detailed per-query breakdown.
        timestamp: ISO-8601 timestamp of the evaluation run.
        summary: Human-readable summary of results.
    """

    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    answer_metrics: Dict[str, float] = field(default_factory=dict)
    per_query_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    summary: str = ""


class RAGEvalHarness:
    """Evaluation harness for a ``SimpleRAG`` instance.

    Args:
        rag_instance: A ``SimpleRAG`` (or compatible mock) with ``retrieve``
            and ``answer`` methods.
    """

    def __init__(self, rag_instance: Any) -> None:
        self.rag = rag_instance

    # ------------------------------------------------------------------
    # Retrieval evaluation
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        qa_pairs: Optional[List[GoldenQA]] = None,
        k: int = 3,
    ) -> Dict[str, Any]:
        """Run retrieval evaluation over *qa_pairs*.

        Args:
            qa_pairs: Golden Q&A items (defaults to ``GOLDEN_QA_PAIRS``).
            k: Cut-off rank for precision/recall/NDCG.

        Returns:
            Dict with ``"aggregate"`` metrics and ``"per_query"`` details.
        """
        if qa_pairs is None:
            qa_pairs = GOLDEN_QA_PAIRS

        per_query: List[Dict[str, Any]] = []
        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        total_ndcg = 0.0

        for qa in qa_pairs:
            retrieved_docs = self.rag.retrieve(qa.question, top_k=k)
            retrieved_sources = [doc.get("source", "") if isinstance(doc, dict) else "" for doc in retrieved_docs]

            p = precision_at_k(retrieved_sources, qa.expected_sources, k)
            r = recall_at_k(retrieved_sources, qa.expected_sources, k)
            m = mrr(retrieved_sources, qa.expected_sources)
            n = ndcg_at_k(retrieved_sources, qa.expected_sources, k)

            total_precision += p
            total_recall += r
            total_mrr += m
            total_ndcg += n

            per_query.append(
                {
                    "question": qa.question,
                    "query_type": qa.query_type,
                    "difficulty": qa.difficulty,
                    "retrieved_sources": retrieved_sources,
                    "expected_sources": qa.expected_sources,
                    "precision_at_k": p,
                    "recall_at_k": r,
                    "mrr": m,
                    "ndcg_at_k": n,
                }
            )

        n_queries = len(qa_pairs) or 1
        aggregate = {
            "mean_precision_at_k": total_precision / n_queries,
            "mean_recall_at_k": total_recall / n_queries,
            "mean_mrr": total_mrr / n_queries,
            "mean_ndcg_at_k": total_ndcg / n_queries,
            "k": k,
            "n_queries": len(qa_pairs),
        }
        return {"aggregate": aggregate, "per_query": per_query}

    # ------------------------------------------------------------------
    # Answer quality evaluation
    # ------------------------------------------------------------------

    def evaluate_answers(
        self,
        qa_pairs: Optional[List[GoldenQA]] = None,
        k: int = 3,
    ) -> Dict[str, Any]:
        """Run answer quality evaluation over *qa_pairs*.

        Calls ``rag.retrieve`` then ``rag.answer`` for each question.

        Args:
            qa_pairs: Golden Q&A items (defaults to ``GOLDEN_QA_PAIRS``).
            k: Number of documents to retrieve per query.

        Returns:
            Dict with ``"aggregate"`` metrics and ``"per_query"`` details.
        """
        if qa_pairs is None:
            qa_pairs = GOLDEN_QA_PAIRS

        per_query: List[Dict[str, Any]] = []
        total_faith = 0.0
        total_rel = 0.0
        total_comp = 0.0

        for qa in qa_pairs:
            retrieved_docs = self.rag.retrieve(qa.question, top_k=k)
            answer_text = self.rag.answer(qa.question, retrieved_docs=retrieved_docs)
            chunks = [doc.get("content", "") if isinstance(doc, dict) else str(doc) for doc in retrieved_docs]

            faith = faithfulness_score(answer_text, chunks)
            rel = relevance_score(answer_text, qa.question)
            comp = completeness_score(answer_text, qa.expected_answer)

            total_faith += faith
            total_rel += rel
            total_comp += comp

            per_query.append(
                {
                    "question": qa.question,
                    "query_type": qa.query_type,
                    "difficulty": qa.difficulty,
                    "answer": answer_text,
                    "expected_answer": qa.expected_answer,
                    "faithfulness": faith,
                    "relevance": rel,
                    "completeness": comp,
                }
            )

        n_queries = len(qa_pairs) or 1
        aggregate = {
            "mean_faithfulness": total_faith / n_queries,
            "mean_relevance": total_rel / n_queries,
            "mean_completeness": total_comp / n_queries,
            "n_queries": len(qa_pairs),
        }
        return {"aggregate": aggregate, "per_query": per_query}

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def run_full_eval(
        self,
        qa_pairs: Optional[List[GoldenQA]] = None,
        k: int = 3,
    ) -> EvalReport:
        """Run a comprehensive evaluation and return an ``EvalReport``.

        Combines retrieval and answer quality evaluations.  Gated by
        ``settings.enable_evaluation`` -- returns an empty report with a
        warning summary if evaluation is disabled.

        Args:
            qa_pairs: Golden Q&A items (defaults to ``GOLDEN_QA_PAIRS``).
            k: Cut-off rank / retrieval depth.

        Returns:
            ``EvalReport`` with all metrics.
        """
        if not getattr(settings, "enable_evaluation", False):
            logger.warning("Evaluation is disabled. Set RAG_ENABLE_EVALUATION=true to enable.")
            return EvalReport(summary="Evaluation disabled (enable_evaluation=False).")

        if qa_pairs is None:
            qa_pairs = GOLDEN_QA_PAIRS

        retrieval_result = self.evaluate_retrieval(qa_pairs, k=k)
        answer_result = self.evaluate_answers(qa_pairs, k=k)

        # Merge per-query details
        per_query_merged: List[Dict[str, Any]] = []
        for ret_q, ans_q in zip(retrieval_result["per_query"], answer_result["per_query"]):
            merged = {**ret_q, **ans_q}
            per_query_merged.append(merged)

        ret_agg = retrieval_result["aggregate"]
        ans_agg = answer_result["aggregate"]

        summary_lines = [
            f"RAG Evaluation Report ({len(qa_pairs)} queries, k={k})",
            f"  Retrieval: P@{k}={ret_agg['mean_precision_at_k']:.3f}  "
            f"R@{k}={ret_agg['mean_recall_at_k']:.3f}  "
            f"MRR={ret_agg['mean_mrr']:.3f}  "
            f"NDCG@{k}={ret_agg['mean_ndcg_at_k']:.3f}",
            f"  Answers:   Faith={ans_agg['mean_faithfulness']:.3f}  "
            f"Relevance={ans_agg['mean_relevance']:.3f}  "
            f"Completeness={ans_agg['mean_completeness']:.3f}",
        ]

        return EvalReport(
            retrieval_metrics=ret_agg,
            answer_metrics=ans_agg,
            per_query_results=per_query_merged,
            summary="\n".join(summary_lines),
        )
