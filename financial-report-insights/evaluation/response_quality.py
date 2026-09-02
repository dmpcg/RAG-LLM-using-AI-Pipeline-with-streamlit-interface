"""Response quality evaluation for RAG answers.

Provides hallucination detection, confidence scoring, and answer grounding
using lightweight heuristic overlap methods — no LLM calls required.
All scores are in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Shared tokenisation helpers (mirrors answer_metrics.py conventions)
# ---------------------------------------------------------------------------

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


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenisation, stripping punctuation."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?(?:%)?", text.lower())


def _content_tokens(text: str) -> set:
    """Return content-bearing tokens (stop words excluded)."""
    return {t for t in _tokenize(text) if t not in _STOP_WORDS}


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple punctuation rules.

    Args:
        text: Input text to split.

    Returns:
        List of non-empty sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace or end
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _overlap(sentence: str, chunk: str) -> float:
    """Jaccard-style overlap of content tokens between sentence and chunk.

    Args:
        sentence: A single sentence from the answer.
        chunk: A retrieved context chunk.

    Returns:
        Overlap ratio in [0.0, 1.0].
    """
    s_tokens = _content_tokens(sentence)
    c_tokens = _content_tokens(chunk)
    if not s_tokens or not c_tokens:
        return 0.0
    intersection = s_tokens & c_tokens
    return len(intersection) / len(s_tokens)


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------


class HallucinationDetector:
    """Detect claims in an answer that are not supported by retrieved context.

    Uses token-overlap heuristics: a sentence is considered *grounded* when
    a sufficient fraction of its content tokens appear in at least one
    context chunk.

    Args:
        threshold: Minimum overlap ratio required for a sentence to be
            considered grounded.  Defaults to ``0.3``.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_claims(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> List[Dict[str, Any]]:
        """Check each sentence in *answer* against *context_chunks*.

        Args:
            answer: Generated answer text.
            context_chunks: Retrieved text chunks used to generate the answer.

        Returns:
            List of dicts, one per sentence::

                {
                    "sentence": str,
                    "grounded": bool,
                    "best_chunk_overlap": float,
                    "supporting_chunk_index": int | None,
                }
        """
        sentences = _split_sentences(answer)
        results: List[Dict[str, Any]] = []

        for sentence in sentences:
            best_overlap = 0.0
            best_index: Optional[int] = None

            for idx, chunk in enumerate(context_chunks):
                score = _overlap(sentence, chunk)
                if score > best_overlap:
                    best_overlap = score
                    best_index = idx

            grounded = best_overlap >= self.threshold
            results.append(
                {
                    "sentence": sentence,
                    "grounded": grounded,
                    "best_chunk_overlap": best_overlap,
                    "supporting_chunk_index": best_index if grounded else None,
                }
            )

        return results

    def get_grounding_score(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> float:
        """Fraction of sentences that are grounded in context.

        Args:
            answer: Generated answer text.
            context_chunks: Retrieved text chunks.

        Returns:
            Score in [0.0, 1.0].  Returns 0.0 for an empty answer.
        """
        claims = self.check_claims(answer, context_chunks)
        if not claims:
            return 0.0
        grounded_count = sum(1 for c in claims if c["grounded"])
        return grounded_count / len(claims)

    def get_ungrounded_claims(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> List[str]:
        """Return sentences that are NOT supported by any context chunk.

        Args:
            answer: Generated answer text.
            context_chunks: Retrieved text chunks.

        Returns:
            List of ungrounded sentence strings.
        """
        claims = self.check_claims(answer, context_chunks)
        return [c["sentence"] for c in claims if not c["grounded"]]


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------


class ConfidenceScorer:
    """Score overall confidence in a RAG response.

    Combines retrieval quality signals into a single confidence estimate.
    All component scores are in [0.0, 1.0]; the overall confidence is a
    weighted average with the following default weights:

    * avg_retrieval_score  – 0.50
    * coverage_score       – 0.30
    * source_diversity     – 0.20
    """

    # Weights must sum to 1.0
    _W_RETRIEVAL = 0.50
    _W_COVERAGE = 0.30
    _W_DIVERSITY = 0.20

    def __init__(self) -> None:
        pass

    def score(
        self,
        retrieval_scores: List[float],
        source_coverage: float,
        num_sources: int,
        query_type: str = "general",
    ) -> Dict[str, Any]:
        """Compute a confidence breakdown for a RAG response.

        Args:
            retrieval_scores: Similarity scores for each retrieved chunk,
                each in [0.0, 1.0].
            source_coverage: Fraction of query terms found in retrieved
                chunks, in [0.0, 1.0].
            num_sources: Number of distinct source documents retrieved.
            query_type: Query category (e.g. ``"general"``, ``"financial"``).
                Currently informational; reserved for future routing.

        Returns:
            Dict with keys::

                {
                    "avg_retrieval_score": float,
                    "source_diversity": float,
                    "coverage_score": float,
                    "overall_confidence": float,
                    "confidence_level": "high" | "medium" | "low",
                    "query_type": str,
                }
        """
        total_chunks = len(retrieval_scores)

        # Average retrieval similarity
        avg_retrieval = sum(retrieval_scores) / total_chunks if total_chunks > 0 else 0.0
        avg_retrieval = max(0.0, min(1.0, avg_retrieval))

        # Source diversity: unique sources / total chunks
        # When num_sources > total_chunks (e.g. 1 chunk but 1 source), clamp to 1.
        if total_chunks == 0:
            diversity = 0.0
        else:
            diversity = min(1.0, num_sources / total_chunks)

        # Coverage score is passed in directly (caller computed it)
        coverage = max(0.0, min(1.0, source_coverage))

        # Weighted combination
        overall = self._W_RETRIEVAL * avg_retrieval + self._W_COVERAGE * coverage + self._W_DIVERSITY * diversity
        overall = max(0.0, min(1.0, overall))

        # Confidence level thresholds
        if overall > 0.7:
            level = "high"
        elif overall >= 0.4:
            level = "medium"
        else:
            level = "low"

        return {
            "avg_retrieval_score": round(avg_retrieval, 4),
            "source_diversity": round(diversity, 4),
            "coverage_score": round(coverage, 4),
            "overall_confidence": round(overall, 4),
            "confidence_level": level,
            "query_type": query_type,
        }


# ---------------------------------------------------------------------------
# AnswerGrounder
# ---------------------------------------------------------------------------

# Pattern matching inline citations already present in the answer text
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class AnswerGrounder:
    """Ground an answer by inserting inline citation references.

    Uses keyword overlap to match each sentence to the most relevant
    document and inserts ``[N]`` citation markers inline.
    """

    def ground_answer(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> str:
        """Insert inline citation references into *answer*.

        For each sentence in the answer the document with the highest
        content-token overlap is identified and a ``[N]`` reference is
        appended to that sentence.  Sentences with no overlap are left
        without a citation.

        Args:
            answer: Generated answer text.
            documents: List of document dicts, each containing a ``"content"``
                key and optionally a ``"_citation_id"`` key.

        Returns:
            Answer text with inline ``[N]`` citations inserted.
        """
        if not answer or not documents:
            return answer

        sentences = _split_sentences(answer)
        grounded_sentences: List[str] = []

        for sentence in sentences:
            best_overlap = 0.0
            best_doc_id: Optional[int] = None

            for idx, doc in enumerate(documents):
                content = doc.get("content", "")
                ov = _overlap(sentence, content)
                if ov > best_overlap:
                    best_overlap = ov
                    # Prefer explicit _citation_id; fall back to 1-based index
                    best_doc_id = doc.get("_citation_id", idx + 1)

            if best_overlap > 0.0 and best_doc_id is not None:
                grounded_sentences.append(f"{sentence} [{best_doc_id}]")
            else:
                grounded_sentences.append(sentence)

        return " ".join(grounded_sentences)

    def verify_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify that citations in *answer* reference relevant content.

        Scans the answer for ``[N]`` citation markers.  For each marker, the
        surrounding textual context (the whole answer stripped of citation
        markers) is compared against the referenced document.  A citation is
        *valid* when the surrounding text has a non-zero content-token overlap
        with the referenced document.

        Args:
            answer: Answer text, possibly containing ``[N]`` citation markers.
            documents: List of source document dicts with a ``"content"`` key.

        Returns:
            Dict with keys::

                {
                    "total_citations": int,
                    "valid_citations": int,
                    "invalid_citations": int,
                    "citation_accuracy": float,   # valid / total, or 0.0
                }
        """
        # Build a 1-based lookup for documents
        doc_by_id: Dict[int, Dict[str, Any]] = {}
        for idx, doc in enumerate(documents):
            cid = doc.get("_citation_id", idx + 1)
            doc_by_id[cid] = doc

        # Strip all citation markers to get clean answer text for overlap
        clean_answer = _CITATION_PATTERN.sub("", answer).strip()

        # Find every citation marker across the full answer text
        markers = _CITATION_PATTERN.findall(answer)

        total = len(markers)
        valid = 0

        for marker in markers:
            doc_id = int(marker)
            doc = doc_by_id.get(doc_id)
            if doc is not None:
                content = doc.get("content", "")
                if _overlap(clean_answer, content) > 0.0:
                    valid += 1

        invalid = total - valid
        accuracy = valid / total if total > 0 else 0.0

        return {
            "total_citations": total,
            "valid_citations": valid,
            "invalid_citations": invalid,
            "citation_accuracy": round(accuracy, 4),
        }
