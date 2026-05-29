"""Tests for the RAG evaluation harness.

Covers retrieval metrics, answer metrics, golden QA dataset validation,
EvalReport construction, and the RAGEvalHarness with mocked RAG instances.
"""

import math
from unittest.mock import MagicMock

import pytest

from evaluation.answer_metrics import (
    completeness_score,
    faithfulness_score,
    relevance_score,
)
from evaluation.eval_harness import EvalReport, RAGEvalHarness
from evaluation.golden_qa import GOLDEN_QA_PAIRS, GoldenQA
from evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

# ======================================================================
# Golden QA dataset
# ======================================================================


class TestGoldenQA:
    """Validate the golden Q&A dataset itself."""

    def test_dataset_has_20_items(self):
        assert len(GOLDEN_QA_PAIRS) == 20

    def test_all_query_types_covered(self):
        types = {qa.query_type for qa in GOLDEN_QA_PAIRS}
        expected = {"ratio_lookup", "trend_analysis", "comparison", "explanation", "general"}
        assert types == expected

    def test_all_difficulties_covered(self):
        difficulties = {qa.difficulty for qa in GOLDEN_QA_PAIRS}
        assert difficulties == {"easy", "medium", "hard"}

    def test_each_query_type_has_at_least_3(self):
        from collections import Counter

        counts = Counter(qa.query_type for qa in GOLDEN_QA_PAIRS)
        for qt, count in counts.items():
            assert count >= 3, f"query_type={qt} only has {count} items"

    def test_golden_qa_fields(self):
        qa = GOLDEN_QA_PAIRS[0]
        assert isinstance(qa.question, str) and len(qa.question) > 0
        assert isinstance(qa.expected_answer, str) and len(qa.expected_answer) > 0
        assert isinstance(qa.expected_sources, list)
        assert qa.query_type in {"ratio_lookup", "trend_analysis", "comparison", "explanation", "general"}
        assert qa.difficulty in {"easy", "medium", "hard"}

    def test_golden_qa_is_frozen(self):
        qa = GOLDEN_QA_PAIRS[0]
        with pytest.raises(AttributeError):
            qa.question = "changed"


# ======================================================================
# Retrieval metrics
# ======================================================================


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.5

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], 3) == 0.0

    def test_empty_relevant(self):
        assert precision_at_k(["a", "b"], [], 2) == 0.0

    def test_k_larger_than_retrieved(self):
        # k=5 but only 2 retrieved -> uses all 2
        assert precision_at_k(["a", "b"], ["a", "b", "c"], 5) == 1.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], ["a", "b"], 2) == 0.5

    def test_k_zero(self):
        assert recall_at_k(["a"], ["a"], 0) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a"], [], 1) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["a"], 1) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a", "c"], ["a"]) == 0.5

    def test_third_position(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], ["a"]) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a"], []) == 0.0

    def test_multiple_relevant_returns_first(self):
        # MRR uses the first relevant hit
        assert mrr(["x", "a", "b"], ["a", "b"]) == 0.5


class TestNDCG:
    def test_perfect_ndcg(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], ["a"], 0) == 0.0

    def test_empty_lists(self):
        assert ndcg_at_k([], [], 3) == 0.0

    def test_single_relevant_at_position_2(self):
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(expected)

    def test_ndcg_rewards_early_ranks(self):
        # Relevant doc at rank 1 should score higher than at rank 3
        score_early = ndcg_at_k(["a", "x", "y"], ["a"], 3)
        score_late = ndcg_at_k(["x", "y", "a"], ["a"], 3)
        assert score_early > score_late


# ======================================================================
# Answer metrics
# ======================================================================


class TestFaithfulness:
    def test_fully_faithful(self):
        answer = "Revenue was 9.8 billion dollars."
        context = ["Revenue was 9.8 billion dollars in FY2025."]
        assert faithfulness_score(answer, context) == pytest.approx(1.0)

    def test_unfaithful(self):
        answer = "The company launched a rocket to Mars."
        context = ["Revenue was 9.8 billion dollars."]
        score = faithfulness_score(answer, context)
        assert score < 0.5

    def test_empty_answer(self):
        assert faithfulness_score("", ["some context"]) == 0.0

    def test_empty_context(self):
        assert faithfulness_score("some answer", []) == 0.0

    def test_empty_both(self):
        assert faithfulness_score("", []) == 0.0

    def test_partial_overlap(self):
        answer = "Revenue grew to 9.8 billion, profits unknown."
        context = ["Revenue reached 9.8 billion in FY2025."]
        score = faithfulness_score(answer, context)
        assert 0.0 < score < 1.0


class TestRelevance:
    def test_fully_relevant(self):
        question = "What is the revenue?"
        answer = "The revenue is 9.8 billion."
        score = relevance_score(answer, question)
        assert score == pytest.approx(1.0)

    def test_irrelevant(self):
        question = "What is the debt ratio?"
        answer = "The sky is blue and the grass is green."
        score = relevance_score(answer, question)
        assert score == 0.0

    def test_empty_answer(self):
        assert relevance_score("", "question") == 0.0

    def test_empty_question(self):
        assert relevance_score("answer", "") == 0.0


class TestCompleteness:
    def test_fully_complete(self):
        expected = "Revenue is 9.8 billion."
        actual = "Revenue is 9.8 billion dollars."
        assert completeness_score(actual, expected) == pytest.approx(1.0)

    def test_zero_completeness(self):
        expected = "Revenue is 9.8 billion."
        actual = "The weather forecast looks sunny."
        assert completeness_score(actual, expected) == 0.0

    def test_partial_completeness(self):
        expected = "Revenue grew 10% to 9.8 billion with strong margins."
        actual = "Revenue reached 9.8 billion."
        score = completeness_score(actual, expected)
        assert 0.0 < score < 1.0

    def test_empty_expected(self):
        assert completeness_score("answer", "") == 0.0

    def test_empty_actual(self):
        assert completeness_score("", "expected answer") == 0.0


# ======================================================================
# EvalReport
# ======================================================================


class TestEvalReport:
    def test_default_construction(self):
        report = EvalReport()
        assert report.retrieval_metrics == {}
        assert report.answer_metrics == {}
        assert report.per_query_results == []
        assert report.summary == ""
        assert len(report.timestamp) > 0

    def test_custom_construction(self):
        report = EvalReport(
            retrieval_metrics={"mean_mrr": 0.75},
            answer_metrics={"mean_faithfulness": 0.9},
            per_query_results=[{"question": "test"}],
            summary="Test summary",
        )
        assert report.retrieval_metrics["mean_mrr"] == 0.75
        assert report.answer_metrics["mean_faithfulness"] == 0.9
        assert len(report.per_query_results) == 1
        assert report.summary == "Test summary"


# ======================================================================
# RAGEvalHarness
# ======================================================================


def _make_mock_rag(docs=None, answer_text="Mock answer"):
    """Create a mock RAG instance with configurable retrieve/answer."""
    rag = MagicMock()
    if docs is None:
        docs = [{"source": "balance_sheet_2025.xlsx", "content": "Revenue 9.8B"}]
    rag.retrieve.return_value = docs
    rag.answer.return_value = answer_text
    return rag


class TestRAGEvalHarness:
    def test_evaluate_retrieval_returns_aggregate(self):
        rag = _make_mock_rag()
        harness = RAGEvalHarness(rag)
        qa = [
            GoldenQA(
                question="What is revenue?",
                expected_answer="9.8B",
                expected_sources=["balance_sheet_2025.xlsx"],
                query_type="general",
                difficulty="easy",
            )
        ]
        result = harness.evaluate_retrieval(qa, k=3)
        assert "aggregate" in result
        assert "per_query" in result
        assert result["aggregate"]["n_queries"] == 1
        # Mock returns the right source, so precision should be > 0
        assert result["aggregate"]["mean_precision_at_k"] > 0

    def test_evaluate_answers_returns_aggregate(self):
        rag = _make_mock_rag(answer_text="Revenue is 9.8 billion.")
        harness = RAGEvalHarness(rag)
        qa = [
            GoldenQA(
                question="What is the revenue?",
                expected_answer="Revenue is 9.8 billion.",
                expected_sources=["balance_sheet_2025.xlsx"],
                query_type="general",
                difficulty="easy",
            )
        ]
        result = harness.evaluate_answers(qa, k=3)
        assert "aggregate" in result
        assert result["aggregate"]["mean_completeness"] > 0

    def test_run_full_eval_disabled(self):
        """When enable_evaluation is False, returns empty report."""
        rag = _make_mock_rag()
        harness = RAGEvalHarness(rag)
        report = harness.run_full_eval()
        assert "disabled" in report.summary.lower()
        assert report.retrieval_metrics == {}

    def test_run_full_eval_enabled(self, monkeypatch):
        """With evaluation enabled, returns populated report."""
        from config import settings as cfg

        monkeypatch.setattr(cfg, "enable_evaluation", True)
        rag = _make_mock_rag(answer_text="Revenue is 9.8 billion dollars.")
        harness = RAGEvalHarness(rag)
        qa = [
            GoldenQA(
                question="What is revenue?",
                expected_answer="Revenue is 9.8 billion.",
                expected_sources=["balance_sheet_2025.xlsx"],
                query_type="general",
                difficulty="easy",
            )
        ]
        report = harness.run_full_eval(qa_pairs=qa, k=3)
        assert report.retrieval_metrics["n_queries"] == 1
        assert report.answer_metrics["n_queries"] == 1
        assert len(report.per_query_results) == 1
        assert "RAG Evaluation Report" in report.summary

    def test_harness_calls_rag_retrieve(self):
        rag = _make_mock_rag()
        harness = RAGEvalHarness(rag)
        qa = [
            GoldenQA(
                question="Test?",
                expected_answer="Test.",
                expected_sources=["a.xlsx"],
            )
        ]
        harness.evaluate_retrieval(qa, k=2)
        rag.retrieve.assert_called_once_with("Test?", top_k=2)

    def test_harness_calls_rag_answer(self):
        docs = [{"source": "a.xlsx", "content": "data"}]
        rag = _make_mock_rag(docs=docs)
        harness = RAGEvalHarness(rag)
        qa = [
            GoldenQA(
                question="Test?",
                expected_answer="Test.",
                expected_sources=["a.xlsx"],
            )
        ]
        harness.evaluate_answers(qa, k=3)
        rag.answer.assert_called_once_with("Test?", retrieved_docs=docs)

    def test_evaluate_retrieval_empty_qa(self):
        rag = _make_mock_rag()
        harness = RAGEvalHarness(rag)
        result = harness.evaluate_retrieval([], k=3)
        assert result["aggregate"]["n_queries"] == 0
        assert result["per_query"] == []

    def test_evaluate_answers_empty_qa(self):
        rag = _make_mock_rag()
        harness = RAGEvalHarness(rag)
        result = harness.evaluate_answers([], k=3)
        assert result["aggregate"]["n_queries"] == 0
