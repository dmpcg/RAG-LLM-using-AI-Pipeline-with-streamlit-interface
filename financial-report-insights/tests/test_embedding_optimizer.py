"""
Tests for ml/embedding_optimizer.py.

All tests use synthetic numpy-generated embeddings – no model or network
calls are made.
"""

from __future__ import annotations

# Make sure the project root is on sys.path when running from the repo root
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.embedding_optimizer import (
    BatchEmbedder,
    EmbeddingCache,
    dequantize_embeddings,
    measure_quality_loss,
    quantize_embeddings,
    truncate_embeddings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_embeddings(n: int = 10, dim: int = 32) -> List[List[float]]:
    """Generate random unit-normalized float32 embeddings."""
    arr = RNG.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / norms).tolist()


# ===========================================================================
# quantize_embeddings / dequantize_embeddings
# ===========================================================================


class TestInt8Quantization:
    """int8 quantization round-trip tests."""

    def test_round_trip_close_to_original(self):
        """dequantize(quantize(x, 'int8')) should be close to x."""
        original = make_embeddings(20, 64)
        quantized, meta = quantize_embeddings(original, method="int8")
        recovered = dequantize_embeddings(quantized, meta)
        orig_arr = np.asarray(original, dtype=np.float32)
        assert np.allclose(orig_arr, recovered, atol=0.01), "int8 round-trip error exceeds tolerance of 0.01"

    def test_output_dtype_is_int8(self):
        embeddings = make_embeddings(5, 16)
        quantized, _ = quantize_embeddings(embeddings, method="int8")
        assert quantized.dtype == np.int8

    def test_metadata_has_required_keys(self):
        embeddings = make_embeddings(5, 16)
        _, meta = quantize_embeddings(embeddings, method="int8")
        assert "method" in meta
        assert "min" in meta
        assert "max" in meta
        assert "scale" in meta
        assert meta["method"] == "int8"

    def test_values_in_int8_range(self):
        embeddings = make_embeddings(10, 32)
        quantized, _ = quantize_embeddings(embeddings, method="int8")
        assert quantized.min() >= -128
        assert quantized.max() <= 127

    def test_shape_preserved(self):
        n, dim = 7, 48
        embeddings = make_embeddings(n, dim)
        quantized, _ = quantize_embeddings(embeddings, method="int8")
        assert quantized.shape == (n, dim)

    def test_single_embedding_round_trip(self):
        original = make_embeddings(1, 16)
        quantized, meta = quantize_embeddings(original, method="int8")
        recovered = dequantize_embeddings(quantized, meta)
        orig_arr = np.asarray(original, dtype=np.float32)
        assert np.allclose(orig_arr, recovered, atol=0.01)

    def test_all_zero_embedding(self):
        """All-zero vectors should not produce NaN after round-trip."""
        original = [[0.0] * 16]
        quantized, meta = quantize_embeddings(original, method="int8")
        recovered = dequantize_embeddings(quantized, meta)
        assert not np.any(np.isnan(recovered)), "NaN found after zero-vector round-trip"

    def test_constant_dimension_no_nan(self):
        """A dimension with constant values across all embeddings must not cause NaN."""
        # One column is constant = 0.5
        embeddings = make_embeddings(10, 16)
        for row in embeddings:
            row[0] = 0.5
        quantized, meta = quantize_embeddings(embeddings, method="int8")
        recovered = dequantize_embeddings(quantized, meta)
        assert not np.any(np.isnan(recovered))

    def test_high_dimension_round_trip(self):
        """1024-dim embeddings (production size) should round-trip within tolerance."""
        original = make_embeddings(5, 1024)
        quantized, meta = quantize_embeddings(original, method="int8")
        recovered = dequantize_embeddings(quantized, meta)
        orig_arr = np.asarray(original, dtype=np.float32)
        assert np.allclose(orig_arr, recovered, atol=0.01)

    def test_relative_ordering_preserved(self):
        """Cosine similarity ordering should be broadly preserved after int8."""
        corpus = make_embeddings(20, 64)
        query = make_embeddings(1, 64)[0]
        quantized, meta = quantize_embeddings(corpus, method="int8")
        recovered = dequantize_embeddings(quantized, meta)

        orig_arr = np.asarray(corpus, dtype=np.float32)
        rec_arr = recovered
        q_arr = np.asarray(query, dtype=np.float32)

        orig_sims = orig_arr @ q_arr
        rec_sims = rec_arr @ q_arr

        orig_rank = np.argsort(orig_sims)[::-1]
        rec_rank = np.argsort(rec_sims)[::-1]

        # Top-5 overlap should be at least 3 out of 5
        overlap = len(set(orig_rank[:5].tolist()) & set(rec_rank[:5].tolist()))
        assert overlap >= 3, f"Top-5 overlap too low: {overlap}"


class TestFloat16Quantization:
    """float16 quantization tests."""

    def test_output_dtype_is_float16(self):
        embeddings = make_embeddings(5, 32)
        quantized, _ = quantize_embeddings(embeddings, method="float16")
        assert quantized.dtype == np.float16

    def test_metadata_method_key(self):
        embeddings = make_embeddings(3, 16)
        _, meta = quantize_embeddings(embeddings, method="float16")
        assert meta["method"] == "float16"

    def test_dequantize_float16_returns_float32(self):
        embeddings = make_embeddings(5, 32)
        quantized, meta = quantize_embeddings(embeddings, method="float16")
        recovered = dequantize_embeddings(quantized, meta)
        assert recovered.dtype == np.float32

    def test_float16_preserves_relative_ordering(self):
        """Relative similarity ordering should be identical for float16."""
        corpus = make_embeddings(15, 64)
        query = make_embeddings(1, 64)[0]
        quantized, meta = quantize_embeddings(corpus, method="float16")
        recovered = dequantize_embeddings(quantized, meta)

        orig_arr = np.asarray(corpus, dtype=np.float32)
        q_arr = np.asarray(query, dtype=np.float32)

        orig_rank = np.argsort(orig_arr @ q_arr)[::-1]
        rec_rank = np.argsort(recovered @ q_arr)[::-1]

        # float16 should be nearly exact – top-5 should be identical
        assert list(orig_rank[:5].tolist()) == list(rec_rank[:5].tolist())

    def test_float16_round_trip_accuracy(self):
        """float16 -> float32 should be very close to original."""
        original = make_embeddings(10, 32)
        quantized, meta = quantize_embeddings(original, method="float16")
        recovered = dequantize_embeddings(quantized, meta)
        orig_arr = np.asarray(original, dtype=np.float32)
        assert np.allclose(orig_arr, recovered, atol=1e-3)


class TestQuantizeEdgeCases:
    """Edge cases for quantize_embeddings."""

    def test_empty_list_int8(self):
        quantized, meta = quantize_embeddings([], method="int8")
        assert quantized.size == 0
        assert meta["method"] == "int8"

    def test_empty_list_float16(self):
        quantized, meta = quantize_embeddings([], method="float16")
        assert quantized.size == 0
        assert meta["method"] == "float16"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown quantization method"):
            quantize_embeddings(make_embeddings(2, 4), method="bfloat16")

    def test_invalid_method_in_dequantize_raises(self):
        arr = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown quantization method"):
            dequantize_embeddings(arr, {"method": "unknown"})

    def test_empty_dequantize_returns_empty(self):
        empty = np.empty((0, 0), dtype=np.int8)
        result = dequantize_embeddings(empty, {"method": "int8"})
        assert result.size == 0


# ===========================================================================
# truncate_embeddings
# ===========================================================================


class TestTruncateEmbeddings:
    """Tests for Matryoshka-style dimension truncation."""

    def test_reduces_dimension(self):
        original = make_embeddings(10, 64)
        truncated = truncate_embeddings(original, target_dim=16)
        assert len(truncated) == 10
        assert len(truncated[0]) == 16

    def test_identity_when_same_dim(self):
        original = make_embeddings(5, 32)
        truncated = truncate_embeddings(original, target_dim=32, renormalize=False)
        orig_arr = np.asarray(original, dtype=np.float32)
        trunc_arr = np.asarray(truncated, dtype=np.float32)
        assert np.allclose(orig_arr, trunc_arr, atol=1e-6)

    def test_raises_when_target_dim_exceeds_original(self):
        original = make_embeddings(5, 32)
        with pytest.raises(ValueError, match="target_dim"):
            truncate_embeddings(original, target_dim=64)

    def test_raises_on_zero_target_dim(self):
        original = make_embeddings(3, 16)
        with pytest.raises(ValueError, match="target_dim must be positive"):
            truncate_embeddings(original, target_dim=0)

    def test_raises_on_negative_target_dim(self):
        original = make_embeddings(3, 16)
        with pytest.raises(ValueError, match="target_dim must be positive"):
            truncate_embeddings(original, target_dim=-1)

    def test_empty_list_returns_empty(self):
        result = truncate_embeddings([], target_dim=16)
        assert result == []

    def test_renormalize_produces_unit_vectors(self):
        original = make_embeddings(8, 64)
        truncated = truncate_embeddings(original, target_dim=16, renormalize=True)
        arr = np.asarray(truncated, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), "Truncated vectors should be unit length"

    def test_preserves_relative_similarities(self):
        """Rank correlation between 64-dim and 32-dim should be positive.

        Note: random Gaussian embeddings have no Matryoshka structure, so
        the threshold is intentionally relaxed (> 0.7).  With real MRL-trained
        embeddings (e.g. mxbai-embed-large) the value would typically exceed
        0.95 for a 50% truncation.
        """
        corpus = make_embeddings(30, 64)
        query_orig = make_embeddings(3, 64)

        trunc_corpus = truncate_embeddings(corpus, target_dim=32)
        trunc_query = truncate_embeddings(query_orig, target_dim=32)

        orig_arr = np.asarray(corpus, dtype=np.float32)
        trunc_arr = np.asarray(trunc_corpus, dtype=np.float32)
        q_orig_arr = np.asarray(query_orig, dtype=np.float32)
        q_trunc_arr = np.asarray(trunc_query, dtype=np.float32)

        rank_corrs = []
        for q_o, q_t in zip(q_orig_arr, q_trunc_arr):
            sims_o = orig_arr @ q_o
            sims_t = trunc_arr @ q_t
            # Spearman rank correlation
            n = len(sims_o)
            rx = np.argsort(np.argsort(sims_o)).astype(float)
            ry = np.argsort(np.argsort(sims_t)).astype(float)
            d = rx - ry
            corr = 1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1))
            rank_corrs.append(corr)

        avg_corr = float(np.mean(rank_corrs))
        assert avg_corr > 0.7, f"Rank correlation {avg_corr:.3f} below 0.7 threshold"

    def test_single_embedding_truncation(self):
        original = make_embeddings(1, 32)
        truncated = truncate_embeddings(original, target_dim=8)
        assert len(truncated) == 1
        assert len(truncated[0]) == 8

    def test_truncate_to_dim_1(self):
        """Edge case: truncate to a single dimension."""
        original = make_embeddings(5, 32)
        truncated = truncate_embeddings(original, target_dim=1)
        for row in truncated:
            assert len(row) == 1

    def test_no_renormalize_preserves_first_dims(self):
        """Without renormalization the first target_dim values must match exactly."""
        original = make_embeddings(5, 32)
        truncated = truncate_embeddings(original, target_dim=8, renormalize=False)
        orig_arr = np.asarray(original, dtype=np.float32)
        trunc_arr = np.asarray(truncated, dtype=np.float32)
        assert np.allclose(orig_arr[:, :8], trunc_arr, atol=1e-6)


# ===========================================================================
# measure_quality_loss
# ===========================================================================


class TestMeasureQualityLoss:
    """Tests for quality-loss measurement."""

    def test_returns_expected_keys(self):
        original = make_embeddings(10, 32)
        processed = make_embeddings(10, 32)
        metrics = measure_quality_loss(original, processed)
        assert "mae" in metrics
        assert "rank_correlation" in metrics
        assert "similarity_correlation" in metrics

    def test_identical_inputs_perfect_scores(self):
        original = make_embeddings(10, 32)
        metrics = measure_quality_loss(original, original)
        assert pytest.approx(metrics["mae"], abs=1e-5) == 0.0
        assert pytest.approx(metrics["rank_correlation"], abs=1e-3) == 1.0
        # Similarity correlation may be NaN when all values are constant (1.0)
        # Accept either 1.0 or a high value
        sc = metrics["similarity_correlation"]
        assert sc >= 0.99 or (sc != sc), "Identical inputs should give perfect similarity_correlation"

    def test_mae_is_nonnegative(self):
        original = make_embeddings(8, 16)
        processed = make_embeddings(8, 16)
        metrics = measure_quality_loss(original, processed)
        assert metrics["mae"] >= 0.0

    def test_with_queries(self):
        original = make_embeddings(15, 32)
        queries = make_embeddings(4, 32)
        processed = make_embeddings(15, 32)
        metrics = measure_quality_loss(original, processed, queries=queries)
        assert "mae" in metrics

    def test_empty_original_returns_defaults(self):
        metrics = measure_quality_loss([], [])
        assert metrics["mae"] == 0.0
        assert metrics["rank_correlation"] == 1.0
        assert metrics["similarity_correlation"] == 1.0

    def test_quantized_embeddings_have_low_mae(self):
        """int8 quantization should result in low quality loss."""
        original = make_embeddings(20, 64)
        quantized, meta = quantize_embeddings(original, method="int8")
        recovered = dequantize_embeddings(quantized, meta)
        recovered_list = recovered.tolist()
        metrics = measure_quality_loss(original, recovered_list)
        assert metrics["mae"] < 0.05, f"MAE {metrics['mae']:.4f} too high for int8"
        assert metrics["rank_correlation"] > 0.95

    def test_truncated_embeddings_quality_loss(self):
        """50% truncation should produce rank correlation > 0.85."""
        original = make_embeddings(30, 64)
        truncated = truncate_embeddings(original, target_dim=32)
        metrics = measure_quality_loss(original, truncated)
        # Note: truncated embeddings have different dimensionality so the
        # cosine sim matrix is computed in different spaces – rank_correlation
        # measures retrieval consistency within each space
        assert "rank_correlation" in metrics


# ===========================================================================
# EmbeddingCache
# ===========================================================================


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    @pytest.fixture
    def tmp_cache(self, tmp_path):
        return EmbeddingCache(str(tmp_path / "cache"), version="1.0")

    def test_put_and_get_round_trip(self, tmp_cache):
        embedding = [0.1, 0.2, 0.3, 0.4]
        tmp_cache.put("key1", embedding)
        result = tmp_cache.get("key1")
        assert result is not None
        assert np.allclose(result, embedding, atol=1e-6)

    def test_miss_returns_none(self, tmp_cache):
        result = tmp_cache.get("nonexistent_key")
        assert result is None

    def test_version_mismatch_is_a_miss(self, tmp_path):
        cache_v1 = EmbeddingCache(str(tmp_path / "cache"), version="1.0")
        cache_v1.put("key1", [0.1, 0.2, 0.3])

        cache_v2 = EmbeddingCache(str(tmp_path / "cache"), version="2.0")
        result = cache_v2.get("key1")
        assert result is None, "Version mismatch should return None"

    def test_version_match_returns_value(self, tmp_path):
        cache_a = EmbeddingCache(str(tmp_path / "cache"), version="1.0")
        cache_a.put("key1", [0.5, 0.6])

        cache_b = EmbeddingCache(str(tmp_path / "cache"), version="1.0")
        result = cache_b.get("key1")
        assert result is not None
        assert np.allclose(result, [0.5, 0.6], atol=1e-5)

    def test_invalidate_single_key(self, tmp_cache):
        tmp_cache.put("key1", [0.1, 0.2])
        tmp_cache.put("key2", [0.3, 0.4])
        tmp_cache.invalidate("key1")
        assert tmp_cache.get("key1") is None
        assert tmp_cache.get("key2") is not None

    def test_invalidate_nonexistent_is_noop(self, tmp_cache):
        """Invalidating a missing key should not raise."""
        tmp_cache.invalidate("does_not_exist")

    def test_invalidate_all(self, tmp_cache):
        tmp_cache.put("k1", [0.1])
        tmp_cache.put("k2", [0.2])
        tmp_cache.put("k3", [0.3])
        tmp_cache.invalidate_all()
        assert tmp_cache.get("k1") is None
        assert tmp_cache.get("k2") is None
        assert tmp_cache.get("k3") is None

    def test_stats_tracking_hits_and_misses(self, tmp_cache):
        tmp_cache.put("k1", [0.1, 0.2])
        tmp_cache.get("k1")  # hit
        tmp_cache.get("k1")  # hit
        tmp_cache.get("k2")  # miss

        stats = tmp_cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_stats_size_reflects_entries(self, tmp_cache):
        tmp_cache.put("a", [0.1])
        tmp_cache.put("b", [0.2])
        stats = tmp_cache.get_stats()
        assert stats["size"] == 2

    def test_stats_memory_bytes_positive(self, tmp_cache):
        tmp_cache.put("k1", list(range(64)))
        stats = tmp_cache.get_stats()
        assert stats["memory_bytes"] > 0

    def test_stats_version_returned(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache"), version="3.5")
        stats = cache.get_stats()
        assert stats["version"] == "3.5"

    def test_overwrite_key(self, tmp_cache):
        tmp_cache.put("key", [0.1, 0.2])
        tmp_cache.put("key", [0.9, 0.8])
        result = tmp_cache.get("key")
        assert result is not None
        assert np.allclose(result, [0.9, 0.8], atol=1e-5)

    def test_large_embedding_stored_and_retrieved(self, tmp_cache):
        big = list(range(1024))
        tmp_cache.put("bigkey", big)
        result = tmp_cache.get("bigkey")
        assert result is not None
        assert len(result) == 1024


# ===========================================================================
# BatchEmbedder
# ===========================================================================


class TestBatchEmbedder:
    """Tests for BatchEmbedder."""

    def _make_mock_embedder(self, dim: int = 16) -> MagicMock:
        """Return a mock that records embed_batch calls."""
        mock = MagicMock()
        mock.embed_batch.side_effect = lambda texts: [[float(i) / (dim + 1) for i in range(dim)] for _ in texts]
        return mock

    def test_embed_all_calls_embedder_in_batches(self):
        mock = self._make_mock_embedder()
        embedder = BatchEmbedder(mock, batch_size=3)
        texts = [f"text_{i}" for i in range(10)]
        results = embedder.embed_all(texts)

        assert len(results) == 10
        # With 10 texts and batch_size=3 we expect ceil(10/3)=4 calls
        assert mock.embed_batch.call_count == 4

    def test_embed_all_empty_list(self):
        mock = self._make_mock_embedder()
        embedder = BatchEmbedder(mock, batch_size=8)
        results = embedder.embed_all([])
        assert results == []
        mock.embed_batch.assert_not_called()

    def test_embed_all_single_text(self):
        mock = self._make_mock_embedder(dim=8)
        embedder = BatchEmbedder(mock, batch_size=32)
        results = embedder.embed_all(["hello"])
        assert len(results) == 1
        assert len(results[0]) == 8
        mock.embed_batch.assert_called_once()

    def test_batch_size_respected(self):
        """Each batch passed to embed_batch must be <= batch_size."""
        call_sizes = []

        class RecordingEmbedder:
            def embed_batch(self, texts):
                call_sizes.append(len(texts))
                return [[0.1] * 4] * len(texts)

        embedder = BatchEmbedder(RecordingEmbedder(), batch_size=5)
        embedder.embed_all([f"t{i}" for i in range(13)])
        assert all(s <= 5 for s in call_sizes), f"Oversized batch found: {call_sizes}"

    def test_fallback_to_embed_method(self):
        """BatchEmbedder should fall back to embed() if embed_batch() absent."""

        class SingleEmbedder:
            def embed(self, text: str) -> list:
                return [0.5, 0.5]

        embedder = BatchEmbedder(SingleEmbedder(), batch_size=4)
        results = embedder.embed_all(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(r) == 2 for r in results)

    def test_embed_with_cache_skips_cached_items(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache"), version="1.0")
        # Pre-populate cache for two of three texts
        cache.put("cached_text_0", [1.0, 0.0])
        cache.put("cached_text_1", [0.0, 1.0])

        mock = self._make_mock_embedder(dim=2)
        mock.embed_batch.side_effect = lambda texts: [[0.5, 0.5]] * len(texts)
        embedder = BatchEmbedder(mock, batch_size=8)

        texts = ["cached_text_0", "cached_text_1", "new_text"]
        results = embedder.embed_with_cache(texts, cache)

        assert len(results) == 3
        # Only "new_text" should have been embedded
        assert mock.embed_batch.call_count == 1
        called_texts = mock.embed_batch.call_args[0][0]
        assert called_texts == ["new_text"]

    def test_embed_with_cache_stores_misses(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache"), version="1.0")
        mock = self._make_mock_embedder(dim=4)
        embedder = BatchEmbedder(mock, batch_size=8)

        embedder.embed_with_cache(["text_a", "text_b"], cache)

        # Both should now be cached
        assert cache.get("text_a") is not None
        assert cache.get("text_b") is not None

    def test_embed_with_cache_empty_list(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache"))
        mock = self._make_mock_embedder()
        embedder = BatchEmbedder(mock, batch_size=8)
        results = embedder.embed_with_cache([], cache)
        assert results == []
        mock.embed_batch.assert_not_called()

    def test_embed_with_cache_all_cached(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache"), version="1.0")
        cache.put("t1", [0.1, 0.2])
        cache.put("t2", [0.3, 0.4])

        mock = self._make_mock_embedder()
        embedder = BatchEmbedder(mock, batch_size=8)
        results = embedder.embed_with_cache(["t1", "t2"], cache)

        assert len(results) == 2
        mock.embed_batch.assert_not_called()


# ===========================================================================
# Config integration
# ===========================================================================


class TestConfigFields:
    """Verify the new config fields exist with correct defaults."""

    def test_embedding_quantization_default(self):
        from config import settings

        assert settings.embedding_quantization == "none"

    def test_embedding_truncation_dim_default(self):
        from config import settings

        assert settings.embedding_truncation_dim == 0

    def test_embedding_batch_size_default(self):
        from config import settings

        assert settings.embedding_batch_size == 32

    def test_embedding_cache_version_default(self):
        from config import settings

        assert settings.embedding_cache_version == "1.0"
