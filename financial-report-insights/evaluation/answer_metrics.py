"""Answer quality metrics for RAG evaluation.

All metrics use lightweight heuristics (word-overlap) rather than LLM calls,
keeping evaluation fast and deterministic.  Scores are in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import re
from typing import List


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenization, stripping punctuation."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?(?:%)?", text.lower())


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "about",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
    }
)


def _content_tokens(text: str) -> set:
    """Return content-bearing tokens (no stop words)."""
    return {t for t in _tokenize(text) if t not in _STOP_WORDS}


def faithfulness_score(answer: str, context_chunks: List[str]) -> float:
    """Fraction of answer content tokens that appear in any context chunk.

    A simple proxy for *faithfulness*: answers that reuse words from the
    retrieved context are more likely grounded in evidence.

    Args:
        answer: Generated answer text.
        context_chunks: List of retrieved text chunks.

    Returns:
        Score in [0.0, 1.0].
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0

    context_tokens: set = set()
    for chunk in context_chunks:
        context_tokens.update(_content_tokens(chunk))

    if not context_tokens:
        return 0.0

    supported = answer_tokens & context_tokens
    return len(supported) / len(answer_tokens)


def relevance_score(answer: str, question: str) -> float:
    """Keyword overlap between answer and question (normalized).

    Measures whether the answer addresses the question's key terms.

    Args:
        answer: Generated answer text.
        question: Original user query.

    Returns:
        Score in [0.0, 1.0].
    """
    question_tokens = _content_tokens(question)
    answer_tokens = _content_tokens(answer)

    if not question_tokens or not answer_tokens:
        return 0.0

    overlap = question_tokens & answer_tokens
    return len(overlap) / len(question_tokens)


def completeness_score(answer: str, expected_answer: str) -> float:
    """Fraction of key terms from expected answer present in actual answer.

    Measures how much of the reference answer's content the generated
    answer covers.

    Args:
        answer: Generated answer text.
        expected_answer: Reference/gold answer text.

    Returns:
        Score in [0.0, 1.0].
    """
    expected_tokens = _content_tokens(expected_answer)
    answer_tokens = _content_tokens(answer)

    if not expected_tokens:
        return 0.0

    covered = expected_tokens & answer_tokens
    return len(covered) / len(expected_tokens)
