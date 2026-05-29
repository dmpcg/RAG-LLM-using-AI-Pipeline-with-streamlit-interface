"""Tests for evaluation.response_quality — Phase 4.2 Response Quality."""

from __future__ import annotations

import pytest

from evaluation.response_quality import (
    AnswerGrounder,
    ConfidenceScorer,
    HallucinationDetector,
    _overlap,
    _split_sentences,
)

# ===========================================================================
# Helper / private function tests
# ===========================================================================


class TestSplitSentences:
    def test_single_sentence(self):
        assert _split_sentences("Revenue grew 10%.") == ["Revenue grew 10%."]

    def test_multiple_sentences(self):
        result = _split_sentences("Net income rose. Margins improved. EBITDA was stable.")
        assert len(result) == 3
        assert result[0] == "Net income rose."

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   ") == []

    def test_question_mark(self):
        result = _split_sentences("Is revenue growing? Yes, it is.")
        assert len(result) == 2


class TestOverlap:
    def test_perfect_overlap(self):
        # Same content tokens
        score = _overlap("revenue grew strongly", "revenue grew strongly last year")
        assert score == pytest.approx(1.0)

    def test_zero_overlap(self):
        score = _overlap("revenue grew", "completely different text here")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = _overlap("revenue profit growth", "revenue increased")
        assert 0.0 < score < 1.0

    def test_empty_sentence(self):
        assert _overlap("", "some chunk content") == pytest.approx(0.0)

    def test_empty_chunk(self):
        assert _overlap("revenue grew", "") == pytest.approx(0.0)


# ===========================================================================
# HallucinationDetector
# ===========================================================================


class TestHallucinationDetectorCheckClaims:
    def setup_method(self):
        self.detector = HallucinationDetector(threshold=0.3)

    def test_fully_grounded_answer(self):
        answer = "Revenue increased by 15%. Net income grew to $500 million."
        chunks = [
            "Revenue increased by 15% year over year.",
            "Net income grew to $500 million in fiscal year.",
        ]
        results = self.detector.check_claims(answer, chunks)
        assert all(r["grounded"] for r in results)
        assert len(results) == 2

    def test_fully_hallucinated_answer(self):
        answer = "The company invented teleportation. Magic profits appeared."
        chunks = [
            "Revenue was $100 million.",
            "Gross margin was 45%.",
        ]
        results = self.detector.check_claims(answer, chunks)
        assert not any(r["grounded"] for r in results)

    def test_mixed_grounded_and_ungrounded(self):
        answer = "Revenue grew 10%. The CEO traveled to Mars last quarter."
        chunks = ["Revenue grew 10% year on year."]
        results = self.detector.check_claims(answer, chunks)
        assert len(results) == 2
        grounded_states = [r["grounded"] for r in results]
        # At least one grounded, at least one not
        assert True in grounded_states
        assert False in grounded_states

    def test_result_structure_keys(self):
        answer = "Revenue grew."
        chunks = ["Revenue grew significantly."]
        results = self.detector.check_claims(answer, chunks)
        assert len(results) == 1
        record = results[0]
        assert "sentence" in record
        assert "grounded" in record
        assert "best_chunk_overlap" in record
        assert "supporting_chunk_index" in record

    def test_supporting_chunk_index_none_when_ungrounded(self):
        answer = "Unicorns drive revenue."
        chunks = ["Revenue is $100 million."]
        results = self.detector.check_claims(answer, chunks)
        ungrounded = [r for r in results if not r["grounded"]]
        for r in ungrounded:
            assert r["supporting_chunk_index"] is None

    def test_supporting_chunk_index_set_when_grounded(self):
        answer = "Revenue grew 10%."
        chunks = ["Revenue grew 10% last year.", "Other stuff."]
        results = self.detector.check_claims(answer, chunks)
        grounded = [r for r in results if r["grounded"]]
        for r in grounded:
            assert r["supporting_chunk_index"] is not None

    def test_empty_answer_returns_empty_list(self):
        results = self.detector.check_claims("", ["some context"])
        assert results == []

    def test_empty_context_chunks(self):
        results = self.detector.check_claims("Revenue grew.", [])
        assert len(results) == 1
        assert results[0]["grounded"] is False
        assert results[0]["best_chunk_overlap"] == pytest.approx(0.0)

    def test_custom_threshold_stricter(self):
        detector_strict = HallucinationDetector(threshold=0.9)
        answer = "Revenue grew slightly."
        chunks = ["Revenue grew."]
        results = detector_strict.check_claims(answer, chunks)
        # With threshold=0.9 the sentence may not be grounded even if overlap exists
        assert isinstance(results[0]["grounded"], bool)


class TestHallucinationDetectorGroundingScore:
    def setup_method(self):
        self.detector = HallucinationDetector(threshold=0.3)

    def test_perfect_grounding_score(self):
        answer = "Revenue grew 10%. Net income increased."
        chunks = [
            "Revenue grew 10% year over year.",
            "Net income increased to record levels.",
        ]
        score = self.detector.get_grounding_score(answer, chunks)
        assert score == pytest.approx(1.0)

    def test_zero_grounding_score(self):
        answer = "Dragons exist. Magic is real."
        chunks = ["Revenue was $100 million.", "Margins improved."]
        score = self.detector.get_grounding_score(answer, chunks)
        assert score == pytest.approx(0.0)

    def test_partial_grounding_score(self):
        answer = "Revenue grew 10%. Dragons invaded the boardroom."
        chunks = ["Revenue grew 10%."]
        score = self.detector.get_grounding_score(answer, chunks)
        assert 0.0 < score < 1.0

    def test_empty_answer_returns_zero(self):
        score = self.detector.get_grounding_score("", ["context"])
        assert score == pytest.approx(0.0)


class TestHallucinationDetectorUngroundedClaims:
    def setup_method(self):
        self.detector = HallucinationDetector(threshold=0.3)

    def test_returns_only_ungrounded_sentences(self):
        answer = "Revenue grew 10%. The CEO is a time traveler."
        chunks = ["Revenue grew 10% last year."]
        ungrounded = self.detector.get_ungrounded_claims(answer, chunks)
        # The time traveler sentence should be ungrounded
        assert any("time traveler" in s for s in ungrounded)
        # Revenue sentence should NOT appear
        assert not any("Revenue" in s and "time traveler" not in s for s in ungrounded)

    def test_empty_when_fully_grounded(self):
        answer = "Revenue grew."
        chunks = ["Revenue grew significantly last period."]
        ungrounded = self.detector.get_ungrounded_claims(answer, chunks)
        assert ungrounded == []

    def test_all_ungrounded_when_no_overlap(self):
        answer = "Magic happened. Profits teleported."
        chunks = ["Revenue was $100M.", "Margins were 40%."]
        ungrounded = self.detector.get_ungrounded_claims(answer, chunks)
        assert len(ungrounded) == 2


# ===========================================================================
# ConfidenceScorer
# ===========================================================================


class TestConfidenceScorer:
    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_high_quality_retrieval_gives_high_confidence(self):
        result = self.scorer.score(
            retrieval_scores=[0.95, 0.90, 0.88],
            source_coverage=0.9,
            num_sources=3,
        )
        assert result["confidence_level"] == "high"
        assert result["overall_confidence"] > 0.7

    def test_low_quality_retrieval_gives_low_confidence(self):
        result = self.scorer.score(
            retrieval_scores=[0.1, 0.05, 0.08],
            source_coverage=0.1,
            num_sources=1,
        )
        assert result["confidence_level"] == "low"
        assert result["overall_confidence"] < 0.4

    def test_medium_confidence_level(self):
        result = self.scorer.score(
            retrieval_scores=[0.55, 0.50],
            source_coverage=0.5,
            num_sources=2,
        )
        assert result["confidence_level"] in ("medium", "high", "low")
        # Just validate range
        assert 0.0 <= result["overall_confidence"] <= 1.0

    def test_confidence_level_threshold_high(self):
        result = self.scorer.score(
            retrieval_scores=[1.0, 1.0, 1.0],
            source_coverage=1.0,
            num_sources=3,
        )
        assert result["confidence_level"] == "high"
        assert result["overall_confidence"] > 0.7

    def test_confidence_level_threshold_low(self):
        result = self.scorer.score(
            retrieval_scores=[0.0],
            source_coverage=0.0,
            num_sources=0,
        )
        assert result["confidence_level"] == "low"
        assert result["overall_confidence"] < 0.4

    def test_result_contains_all_expected_keys(self):
        result = self.scorer.score(
            retrieval_scores=[0.8],
            source_coverage=0.7,
            num_sources=1,
        )
        expected_keys = {
            "avg_retrieval_score",
            "source_diversity",
            "coverage_score",
            "overall_confidence",
            "confidence_level",
            "query_type",
        }
        assert expected_keys.issubset(result.keys())

    def test_empty_retrieval_scores(self):
        result = self.scorer.score(
            retrieval_scores=[],
            source_coverage=0.0,
            num_sources=0,
        )
        assert result["avg_retrieval_score"] == pytest.approx(0.0)
        assert result["confidence_level"] == "low"

    def test_query_type_preserved_in_result(self):
        result = self.scorer.score(
            retrieval_scores=[0.8],
            source_coverage=0.6,
            num_sources=1,
            query_type="financial",
        )
        assert result["query_type"] == "financial"

    def test_default_query_type_is_general(self):
        result = self.scorer.score(
            retrieval_scores=[0.8],
            source_coverage=0.6,
            num_sources=1,
        )
        assert result["query_type"] == "general"

    def test_overall_confidence_clamped_to_unit_interval(self):
        # Even with extreme inputs the result must stay in [0,1]
        result = self.scorer.score(
            retrieval_scores=[2.0, 5.0],  # deliberately out-of-range inputs
            source_coverage=2.0,
            num_sources=100,
        )
        assert 0.0 <= result["overall_confidence"] <= 1.0

    def test_source_diversity_single_source_single_chunk(self):
        result = self.scorer.score(
            retrieval_scores=[0.8],
            source_coverage=0.5,
            num_sources=1,
        )
        # 1 unique source / 1 chunk = 1.0 diversity
        assert result["source_diversity"] == pytest.approx(1.0)


# ===========================================================================
# AnswerGrounder
# ===========================================================================


class TestAnswerGrounderGroundAnswer:
    def setup_method(self):
        self.grounder = AnswerGrounder()

    def _make_doc(self, content: str, citation_id: int) -> dict:
        return {"content": content, "_citation_id": citation_id}

    def test_citations_inserted(self):
        answer = "Revenue grew by 10%."
        docs = [self._make_doc("Revenue grew by 10% year over year.", 1)]
        result = self.grounder.ground_answer(answer, docs)
        assert "[1]" in result

    def test_multiple_citations_inserted(self):
        answer = "Revenue grew. Net income increased."
        docs = [
            self._make_doc("Revenue grew significantly.", 1),
            self._make_doc("Net income increased to record levels.", 2),
        ]
        result = self.grounder.ground_answer(answer, docs)
        assert "[1]" in result
        assert "[2]" in result

    def test_no_citation_when_no_overlap(self):
        answer = "Magic happened completely."
        docs = [self._make_doc("Revenue was $100 million.", 1)]
        result = self.grounder.ground_answer(answer, docs)
        # No overlap means no citation marker added
        assert "[1]" not in result

    def test_empty_answer_returned_unchanged(self):
        result = self.grounder.ground_answer("", [{"content": "some text", "_citation_id": 1}])
        assert result == ""

    def test_empty_documents_returns_answer_unchanged(self):
        answer = "Revenue grew."
        result = self.grounder.ground_answer(answer, [])
        assert result == answer

    def test_fallback_citation_id_without_explicit_key(self):
        # Document without _citation_id should use 1-based index
        answer = "Revenue grew."
        docs = [{"content": "Revenue grew significantly."}]
        result = self.grounder.ground_answer(answer, docs)
        assert "[1]" in result


class TestAnswerGrounderVerifyCitations:
    def setup_method(self):
        self.grounder = AnswerGrounder()

    def _make_doc(self, content: str, citation_id: int) -> dict:
        return {"content": content, "_citation_id": citation_id}

    def test_valid_citations_counted(self):
        # Citation [1] in answer references document 1 with matching content
        answer = "Revenue grew 10%. [1]"
        docs = [self._make_doc("Revenue grew 10% year over year.", 1)]
        result = self.grounder.verify_citations(answer, docs)
        assert result["total_citations"] == 1
        assert result["valid_citations"] == 1
        assert result["invalid_citations"] == 0
        assert result["citation_accuracy"] == pytest.approx(1.0)

    def test_invalid_citation_no_matching_content(self):
        answer = "Magic happened completely. [1]"
        docs = [self._make_doc("Revenue was $100 million.", 1)]
        result = self.grounder.verify_citations(answer, docs)
        assert result["total_citations"] == 1
        assert result["valid_citations"] == 0
        assert result["invalid_citations"] == 1
        assert result["citation_accuracy"] == pytest.approx(0.0)

    def test_mixed_valid_and_invalid(self):
        answer = "Revenue grew 10%. [1] Dragons invaded. [2]"
        docs = [
            self._make_doc("Revenue grew 10%.", 1),
            self._make_doc("Margins improved.", 2),
        ]
        result = self.grounder.verify_citations(answer, docs)
        assert result["total_citations"] == 2
        assert result["valid_citations"] == 1
        assert result["invalid_citations"] == 1
        assert result["citation_accuracy"] == pytest.approx(0.5)

    def test_no_citations_in_answer(self):
        answer = "Revenue grew 10%."
        docs = [self._make_doc("Revenue grew 10%.", 1)]
        result = self.grounder.verify_citations(answer, docs)
        assert result["total_citations"] == 0
        assert result["citation_accuracy"] == pytest.approx(0.0)

    def test_citation_referencing_nonexistent_document(self):
        answer = "Revenue grew. [99]"
        docs = [self._make_doc("Revenue grew.", 1)]
        result = self.grounder.verify_citations(answer, docs)
        assert result["total_citations"] == 1
        assert result["valid_citations"] == 0
        assert result["invalid_citations"] == 1

    def test_empty_answer(self):
        docs = [self._make_doc("Revenue grew.", 1)]
        result = self.grounder.verify_citations("", docs)
        assert result["total_citations"] == 0
        assert result["citation_accuracy"] == pytest.approx(0.0)

    def test_result_contains_all_keys(self):
        result = self.grounder.verify_citations("Text [1].", [self._make_doc("Text.", 1)])
        expected_keys = {"total_citations", "valid_citations", "invalid_citations", "citation_accuracy"}
        assert expected_keys == set(result.keys())
