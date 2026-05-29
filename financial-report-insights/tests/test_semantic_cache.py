"""
Tests for ml/semantic_cache.py — SemanticCache, ChunkDeduplicator, AdaptiveTopK.

Covers:
- SemanticCache: hit/miss semantics, eviction, stats, thread safety, clear
- ChunkDeduplicator: exact/near-duplicate removal, order preservation, find_duplicates
- AdaptiveTopK: per-type base k, long-query adjustment, boundary clamping
- Edge cases: empty inputs, single item, all duplicates, identical embeddings
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from ml.semantic_cache import (
    AdaptiveTopK,
    ChunkDeduplicator,
    SemanticCache,
    _hamming_similarity,
    _normalise_text,
    _simhash,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, index: int) -> list[float]:
    """Return a unit vector with 1.0 at *index* and 0.0 elsewhere."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _random_vec(dim: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _similar_vec(base: list[float], noise: float = 0.01, seed: int = 42) -> list[float]:
    """Return a vector very close to *base* (cosine similarity ~1 - noise)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(base, dtype=np.float32)
    perturb = rng.standard_normal(len(base)).astype(np.float32) * noise
    arr = arr + perturb
    arr /= np.linalg.norm(arr)
    return arr.tolist()


def _orthogonal_vec(dim: int, index: int) -> list[float]:
    """Return a unit vector orthogonal to the one at *index*."""
    v = [0.0] * dim
    other = (index + 1) % dim
    v[other] = 1.0
    return v


RESULT_A = {"answer": "result_a", "chunks": [1, 2, 3]}
RESULT_B = {"answer": "result_b", "chunks": [4, 5, 6]}

DIM = 64  # small dimension to keep tests fast


# ===========================================================================
# SemanticCache tests
# ===========================================================================


class TestSemanticCacheInit:
    def test_default_construction(self) -> None:
        cache = SemanticCache()
        assert cache.similarity_threshold == 0.95
        assert cache.max_entries == 1000

    def test_custom_construction(self) -> None:
        cache = SemanticCache(similarity_threshold=0.8, max_entries=50)
        assert cache.similarity_threshold == 0.8
        assert cache.max_entries == 50

    def test_invalid_threshold_zero(self) -> None:
        with pytest.raises(ValueError):
            SemanticCache(similarity_threshold=0.0)

    def test_invalid_threshold_negative(self) -> None:
        with pytest.raises(ValueError):
            SemanticCache(similarity_threshold=-0.1)

    def test_invalid_max_entries(self) -> None:
        with pytest.raises(ValueError):
            SemanticCache(max_entries=0)


class TestSemanticCacheHitMiss:
    def test_empty_cache_is_miss(self) -> None:
        cache = SemanticCache()
        result = cache.get(_unit_vec(DIM, 0))
        assert result is None

    def test_hit_with_identical_embedding(self) -> None:
        cache = SemanticCache(similarity_threshold=0.95)
        vec = _unit_vec(DIM, 0)
        cache.put(vec, RESULT_A)
        result = cache.get(vec)
        assert result == RESULT_A

    def test_miss_with_orthogonal_embedding(self) -> None:
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put(_unit_vec(DIM, 0), RESULT_A)
        result = cache.get(_orthogonal_vec(DIM, 0))
        assert result is None

    def test_hit_with_similar_embedding(self) -> None:
        """A small perturbation should still hit at threshold=0.90."""
        cache = SemanticCache(similarity_threshold=0.90)
        base = _random_vec(DIM, seed=1)
        similar = _similar_vec(base, noise=0.005, seed=7)
        cache.put(base, RESULT_A)
        result = cache.get(similar)
        assert result == RESULT_A

    def test_miss_with_dissimilar_embedding(self) -> None:
        """Two random unit vectors in high-d space are nearly orthogonal."""
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put(_random_vec(DIM, seed=0), RESULT_A)
        result = cache.get(_random_vec(DIM, seed=99))
        assert result is None

    def test_returns_correct_result_for_matching_query(self) -> None:
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put(_unit_vec(DIM, 0), RESULT_A)
        cache.put(_unit_vec(DIM, 1), RESULT_B)
        assert cache.get(_unit_vec(DIM, 0)) == RESULT_A
        assert cache.get(_unit_vec(DIM, 1)) == RESULT_B

    def test_threshold_boundary_exact(self) -> None:
        """An entry should be found when similarity is exactly at the threshold."""
        cache = SemanticCache(similarity_threshold=1.0)
        vec = _unit_vec(DIM, 3)
        cache.put(vec, RESULT_A)
        # Identical vector -> cosine similarity = 1.0, threshold = 1.0 -> hit.
        assert cache.get(vec) == RESULT_A


class TestSemanticCacheEviction:
    def test_evicts_oldest_when_at_capacity(self) -> None:
        cache = SemanticCache(similarity_threshold=0.99, max_entries=2)
        vec_a = _unit_vec(DIM, 0)
        vec_b = _unit_vec(DIM, 1)
        vec_c = _unit_vec(DIM, 2)

        cache.put(vec_a, {"r": "a"})
        cache.put(vec_b, {"r": "b"})
        # Inserting a third entry should evict vec_a (oldest).
        cache.put(vec_c, {"r": "c"})

        assert cache.get_stats()["size"] == 2
        # vec_a was evicted.
        assert cache.get(vec_a) is None
        # vec_b and vec_c are still present.
        assert cache.get(vec_b) == {"r": "b"}
        assert cache.get(vec_c) == {"r": "c"}

    def test_capacity_one_always_has_latest(self) -> None:
        cache = SemanticCache(similarity_threshold=0.99, max_entries=1)
        cache.put(_unit_vec(DIM, 0), RESULT_A)
        cache.put(_unit_vec(DIM, 1), RESULT_B)
        assert cache.get(_unit_vec(DIM, 0)) is None
        assert cache.get(_unit_vec(DIM, 1)) == RESULT_B


class TestSemanticCacheStats:
    def test_initial_stats_are_zero(self) -> None:
        cache = SemanticCache()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0

    def test_stats_track_hits_and_misses(self) -> None:
        cache = SemanticCache(similarity_threshold=0.95)
        vec = _unit_vec(DIM, 0)
        cache.get(vec)  # miss (empty cache)
        cache.put(vec, RESULT_A)
        cache.get(vec)  # hit
        cache.get(_unit_vec(DIM, 1))  # miss (different vector)

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert math.isclose(stats["hit_rate"], 1 / 3, rel_tol=1e-6)

    def test_hit_rate_one_when_all_hits(self) -> None:
        cache = SemanticCache(similarity_threshold=0.95)
        vec = _unit_vec(DIM, 0)
        cache.put(vec, RESULT_A)
        cache.get(vec)
        cache.get(vec)
        stats = cache.get_stats()
        assert stats["hit_rate"] == 1.0

    def test_size_reflects_stored_entries(self) -> None:
        cache = SemanticCache()
        assert cache.get_stats()["size"] == 0
        cache.put(_unit_vec(DIM, 0), RESULT_A)
        assert cache.get_stats()["size"] == 1
        cache.put(_unit_vec(DIM, 1), RESULT_B)
        assert cache.get_stats()["size"] == 2


class TestSemanticCacheClear:
    def test_clear_resets_everything(self) -> None:
        cache = SemanticCache()
        cache.put(_unit_vec(DIM, 0), RESULT_A)
        cache.get(_unit_vec(DIM, 0))
        cache.clear()

        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert cache.get(_unit_vec(DIM, 0)) is None


class TestSemanticCacheThreadSafety:
    def test_concurrent_put_and_get_do_not_raise(self) -> None:
        """Multiple threads interleaving put/get must not raise or corrupt."""
        cache = SemanticCache(similarity_threshold=0.95, max_entries=200)
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(10):
                    cache.put(_unit_vec(DIM, (thread_id * 10 + i) % DIM), {"t": thread_id, "i": i})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                for i in range(DIM):
                    cache.get(_unit_vec(DIM, i % DIM))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_stats_are_consistent(self) -> None:
        """hits + misses must equal total operations."""
        cache = SemanticCache(similarity_threshold=0.99, max_entries=500)
        ops = 50
        vec = _unit_vec(DIM, 0)
        cache.put(vec, RESULT_A)

        def do_gets() -> None:
            for _ in range(ops):
                cache.get(vec)

        threads = [threading.Thread(target=do_gets) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = cache.get_stats()
        assert stats["hits"] + stats["misses"] == ops * 4


# ===========================================================================
# ChunkDeduplicator tests
# ===========================================================================


class TestSimhashHelpers:
    def test_normalise_text_lowercases_and_strips(self) -> None:
        assert _normalise_text("  Hello WORLD  ") == "hello world"

    def test_normalise_text_collapses_whitespace(self) -> None:
        assert _normalise_text("a   b\t\tc") == "a b c"

    def test_simhash_is_deterministic(self) -> None:
        h1 = _simhash("hello world")
        h2 = _simhash("hello world")
        assert h1 == h2

    def test_simhash_different_texts_differ(self) -> None:
        h1 = _simhash("apple pie recipe")
        h2 = _simhash("quantum computing theory")
        assert h1 != h2

    def test_hamming_similarity_identical(self) -> None:
        h = _simhash("test")
        assert _hamming_similarity(h, h) == 1.0

    def test_hamming_similarity_all_different(self) -> None:
        # XOR of all-zeros and all-ones is all-ones (64 bits differ).
        assert _hamming_similarity(0, (1 << 64) - 1, num_bits=64) == 0.0

    def test_hamming_similarity_range(self) -> None:
        h1 = _simhash("text one")
        h2 = _simhash("text two")
        sim = _hamming_similarity(h1, h2)
        assert 0.0 <= sim <= 1.0


class TestChunkDeduplicatorInit:
    def test_default_threshold(self) -> None:
        d = ChunkDeduplicator()
        assert d.similarity_threshold == 0.9

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            ChunkDeduplicator(similarity_threshold=0.0)


class TestChunkDeduplicatorDeduplicate:
    def test_empty_input(self) -> None:
        d = ChunkDeduplicator()
        assert d.deduplicate([]) == []

    def test_single_chunk_passthrough(self) -> None:
        d = ChunkDeduplicator()
        assert d.deduplicate(["hello"]) == ["hello"]

    def test_removes_exact_duplicates(self) -> None:
        d = ChunkDeduplicator()
        chunks = ["The revenue was $100M.", "The revenue was $100M.", "Net income grew 10%."]
        result = d.deduplicate(chunks)
        assert "The revenue was $100M." in result
        assert "Net income grew 10%." in result
        # Duplicate should be removed.
        assert result.count("The revenue was $100M.") == 1

    def test_removes_near_duplicates(self) -> None:
        """Texts that differ only in whitespace/casing should be deduplicated."""
        d = ChunkDeduplicator(similarity_threshold=0.85)
        base = "The company reported strong quarterly earnings growth."
        near_dup = "the company reported strong quarterly earnings growth."  # lowercase only
        unique = "Capital expenditures declined sharply in Q3."
        result = d.deduplicate([base, near_dup, unique])
        # near_dup should be considered a duplicate of base.
        assert len(result) == 2
        assert unique in result

    def test_preserves_unique_chunks(self) -> None:
        d = ChunkDeduplicator()
        chunks = [
            "Revenue increased by 15% year-over-year.",
            "Operating expenses declined by 8%.",
            "EBITDA margin expanded to 32%.",
        ]
        result = d.deduplicate(chunks)
        assert len(result) == 3

    def test_preserves_insertion_order(self) -> None:
        d = ChunkDeduplicator()
        # Use long, clearly distinct sentences so SimHash keeps all three.
        chunks = [
            "The company reported record revenue growth for the fiscal quarter.",
            "Operating expenses declined significantly due to cost reduction programs.",
            "Capital expenditure investments increased to support long-term expansion.",
        ]
        result = d.deduplicate(chunks)
        assert result == chunks

    def test_all_duplicates_returns_single(self) -> None:
        text = "Identical financial summary statement repeated verbatim."
        d = ChunkDeduplicator()
        result = d.deduplicate([text, text, text])
        assert result == [text]

    def test_first_occurrence_is_kept(self) -> None:
        d = ChunkDeduplicator()
        chunks = ["first occurrence", "first occurrence", "second unique"]
        result = d.deduplicate(chunks)
        assert result[0] == "first occurrence"


class TestChunkDeduplicatorFindDuplicates:
    def test_empty_returns_empty(self) -> None:
        d = ChunkDeduplicator()
        assert d.find_duplicates([]) == []

    def test_single_chunk_returns_empty(self) -> None:
        d = ChunkDeduplicator()
        assert d.find_duplicates(["only one"]) == []

    def test_exact_duplicate_is_found(self) -> None:
        d = ChunkDeduplicator()
        text = "Gross profit margin for the fiscal year."
        pairs = d.find_duplicates([text, text])
        assert len(pairs) == 1
        idx_a, idx_b, sim = pairs[0]
        assert idx_a == 0
        assert idx_b == 1
        assert sim == 1.0

    def test_no_duplicates_returns_empty(self) -> None:
        d = ChunkDeduplicator()
        pairs = d.find_duplicates(
            [
                "Revenue grew 20% year-over-year in Q4.",
                "Quantum entanglement governs particle spin.",
            ]
        )
        assert pairs == []

    def test_pair_ordering_is_always_idx_a_lt_idx_b(self) -> None:
        d = ChunkDeduplicator()
        text = "duplicate text chunk here"
        pairs = d.find_duplicates([text, text, text])
        for idx_a, idx_b, _ in pairs:
            assert idx_a < idx_b

    def test_similarity_in_valid_range(self) -> None:
        d = ChunkDeduplicator(similarity_threshold=0.5)
        texts = ["alpha bravo charlie", "alpha bravo delta", "zeta eta theta iota"]
        pairs = d.find_duplicates(texts)
        for _, _, sim in pairs:
            assert 0.0 <= sim <= 1.0


# ===========================================================================
# AdaptiveTopK tests
# ===========================================================================


class TestAdaptiveTopKInit:
    def test_default_construction(self) -> None:
        atk = AdaptiveTopK()
        assert atk.min_k == 3
        assert atk.max_k == 15
        assert atk.default_k == 5

    def test_custom_construction(self) -> None:
        atk = AdaptiveTopK(min_k=2, max_k=10, default_k=4)
        assert atk.min_k == 2
        assert atk.max_k == 10
        assert atk.default_k == 4

    def test_invalid_min_k(self) -> None:
        with pytest.raises(ValueError):
            AdaptiveTopK(min_k=0)

    def test_invalid_max_k_less_than_min_k(self) -> None:
        with pytest.raises(ValueError):
            AdaptiveTopK(min_k=5, max_k=3)

    def test_invalid_default_k_out_of_bounds(self) -> None:
        with pytest.raises(ValueError):
            AdaptiveTopK(min_k=3, max_k=10, default_k=1)


class TestAdaptiveTopKComputeK:
    """Short query = <= 15 words; long query = > 15 words."""

    SHORT_QUERY = "What is the gross margin?"
    LONG_QUERY = " ".join(["word"] * 16)  # 16 words -> triggers +2

    def test_ratio_lookup_returns_min_k(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        k = atk.compute_k(self.SHORT_QUERY, "ratio_lookup")
        assert k == atk.min_k

    def test_trend_analysis_returns_default_k(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        k = atk.compute_k(self.SHORT_QUERY, "trend_analysis")
        assert k == atk.default_k

    def test_comparison_returns_default_plus_three(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        k = atk.compute_k(self.SHORT_QUERY, "comparison")
        assert k == min(atk.default_k + 3, atk.max_k)

    def test_explanation_returns_near_max_k(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        k = atk.compute_k(self.SHORT_QUERY, "explanation")
        assert atk.min_k <= k <= atk.max_k
        assert k >= atk.max_k - 2

    def test_general_returns_default_k(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        k = atk.compute_k(self.SHORT_QUERY, "general")
        assert k == atk.default_k

    def test_unknown_type_falls_back_to_default_k(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        k = atk.compute_k(self.SHORT_QUERY, "unknown_type_xyz")
        assert k == atk.default_k

    def test_long_query_adds_two(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        short_k = atk.compute_k(self.SHORT_QUERY, "general")
        long_k = atk.compute_k(self.LONG_QUERY, "general")
        assert long_k == min(short_k + 2, atk.max_k)

    def test_long_ratio_lookup_still_respects_min_k(self) -> None:
        """Even a long ratio_lookup query must not go below min_k."""
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        k = atk.compute_k(self.LONG_QUERY, "ratio_lookup")
        assert k >= atk.min_k

    def test_result_never_exceeds_max_k(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=6, default_k=5)
        k = atk.compute_k(self.LONG_QUERY, "explanation")
        assert k <= atk.max_k

    def test_result_never_below_min_k(self) -> None:
        atk = AdaptiveTopK(min_k=3, max_k=15, default_k=5)
        for qtype in ("ratio_lookup", "trend_analysis", "comparison", "explanation", "general"):
            k = atk.compute_k("", qtype)
            assert k >= atk.min_k, f"k={k} below min_k for query_type={qtype!r}"

    def test_empty_query_handled_gracefully(self) -> None:
        atk = AdaptiveTopK()
        k = atk.compute_k("", "general")
        assert atk.min_k <= k <= atk.max_k

    def test_comparison_long_query_clamped_to_max_k(self) -> None:
        """comparison + long query could exceed max_k; must clamp."""
        atk = AdaptiveTopK(min_k=3, max_k=7, default_k=5)
        k = atk.compute_k(self.LONG_QUERY, "comparison")
        assert k <= atk.max_k

    def test_case_insensitive_query_type(self) -> None:
        atk = AdaptiveTopK()
        k_lower = atk.compute_k(self.SHORT_QUERY, "ratio_lookup")
        k_upper = atk.compute_k(self.SHORT_QUERY, "RATIO_LOOKUP")
        # Both should resolve the same way (via .lower() in _base_k).
        assert k_lower == k_upper
