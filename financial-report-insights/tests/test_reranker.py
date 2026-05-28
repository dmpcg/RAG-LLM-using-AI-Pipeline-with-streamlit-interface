"""Tests for reranker module: EmbeddingReranker, mmr_diversify, add_citations."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from reranker import EmbeddingReranker, mmr_diversify, add_citations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_embedder(query_vec=None, batch_vecs=None):
    """Create a mock embedder with configurable return values."""
    embedder = MagicMock()
    embedder.embed.return_value = query_vec or [0.1, 0.2, 0.3]
    embedder.embed_batch.return_value = batch_vecs or [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    return embedder


def _make_docs(n=3):
    """Create n simple test documents."""
    return [
        {"content": f"Document {i}", "source": f"file{i}.xlsx", "type": "excel"}
        for i in range(n)
    ]


# ===========================================================================
# EmbeddingReranker tests
# ===========================================================================

class TestEmbeddingReranker:

    def test_rerank_returns_top_k_sorted_by_similarity(self):
        """Reranked results should be ordered by descending similarity."""
        embedder = _make_embedder(
            query_vec=[1.0, 0.0, 0.0],
            batch_vecs=[
                [0.0, 1.0, 0.0],   # orthogonal -> low sim
                [1.0, 0.0, 0.0],   # identical  -> high sim
                [0.5, 0.5, 0.0],   # mid sim
            ],
        )
        docs = _make_docs(3)
        reranker = EmbeddingReranker(embedder)

        result = reranker.rerank("test query", docs, top_k=2)

        assert len(result) == 2
        # Highest similarity doc should be first
        assert result[0]["content"] == "Document 1"

    def test_rerank_adds_rerank_score(self):
        """Each reranked doc should have _rerank_score."""
        embedder = _make_embedder()
        docs = _make_docs(3)
        reranker = EmbeddingReranker(embedder)

        result = reranker.rerank("query", docs, top_k=3)

        for doc in result:
            assert "_rerank_score" in doc
            assert isinstance(doc["_rerank_score"], float)

    def test_rerank_empty_documents(self):
        """Reranking empty list returns empty list."""
        embedder = _make_embedder()
        reranker = EmbeddingReranker(embedder)

        result = reranker.rerank("query", [], top_k=5)

        assert result == []

    def test_rerank_handles_embedder_failure(self):
        """When embedder raises, return original docs (truncated to top_k)."""
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("embed failed")
        reranker = EmbeddingReranker(embedder)
        docs = _make_docs(3)

        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        # Should be first 2 in original order
        assert result[0]["content"] == "Document 0"
        assert result[1]["content"] == "Document 1"

    def test_rerank_top_k_larger_than_docs(self):
        """top_k > len(docs) should return all docs."""
        embedder = _make_embedder()
        docs = _make_docs(2)
        embedder.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        reranker = EmbeddingReranker(embedder)

        result = reranker.rerank("query", docs, top_k=10)

        assert len(result) == 2

    def test_rerank_zero_query_norm(self):
        """Zero-norm query vector returns original order."""
        embedder = _make_embedder(query_vec=[0.0, 0.0, 0.0])
        docs = _make_docs(3)
        reranker = EmbeddingReranker(embedder)

        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2

    def test_rerank_does_not_mutate_originals(self):
        """Original documents should not be modified."""
        embedder = _make_embedder()
        docs = _make_docs(2)
        embedder.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        reranker = EmbeddingReranker(embedder)

        reranker.rerank("query", docs, top_k=2)

        for doc in docs:
            assert "_rerank_score" not in doc


# ===========================================================================
# mmr_diversify tests
# ===========================================================================

class TestMMRDiversify:

    def test_selects_diverse_documents(self):
        """MMR should prefer diverse docs over near-duplicates."""
        query_emb = [1.0, 0.0, 0.0]
        docs = _make_docs(4)
        doc_embs = [
            [1.0, 0.0, 0.0],   # doc0: identical to query
            [0.99, 0.01, 0.0],  # doc1: near-duplicate of doc0
            [0.7, 0.7, 0.0],   # doc2: moderate relevance but diverse direction
            [0.5, 0.5, 0.0],   # doc3: moderate
        ]

        # Low lambda emphasises diversity over relevance
        result = mmr_diversify(query_emb, docs, doc_embs, top_k=2, lambda_param=0.3)

        assert len(result) == 2
        # First should be most relevant (doc0)
        assert result[0]["content"] == "Document 0"
        # Second should NOT be the near-duplicate (doc1)
        assert result[1]["content"] != "Document 1"

    def test_lambda_1_pure_relevance(self):
        """lambda=1.0 should rank purely by query similarity."""
        query_emb = [1.0, 0.0, 0.0]
        docs = _make_docs(3)
        doc_embs = [
            [0.0, 1.0, 0.0],   # doc0: low relevance
            [1.0, 0.0, 0.0],   # doc1: highest relevance
            [0.5, 0.5, 0.0],   # doc2: medium relevance
        ]

        result = mmr_diversify(query_emb, docs, doc_embs, top_k=3, lambda_param=1.0)

        assert result[0]["content"] == "Document 1"

    def test_lambda_0_maximizes_diversity(self):
        """lambda=0.0 should maximize diversity from selected docs."""
        query_emb = [1.0, 0.0, 0.0]
        docs = _make_docs(3)
        doc_embs = [
            [1.0, 0.0, 0.0],   # doc0
            [0.99, 0.01, 0.0],  # doc1: near-duplicate of doc0
            [0.0, 1.0, 0.0],   # doc2: very different
        ]

        result = mmr_diversify(query_emb, docs, doc_embs, top_k=2, lambda_param=0.0)

        # First pick: highest relevance (all equal at lambda=0 for first pick)
        # After first pick, second should maximize diversity
        contents = {r["content"] for r in result}
        # doc2 should appear since it's most diverse from any selected doc
        assert "Document 2" in contents

    def test_empty_input(self):
        """Empty documents or embeddings returns empty."""
        assert mmr_diversify([1, 0, 0], [], [], top_k=5) == []
        assert mmr_diversify([1, 0, 0], [], [[1, 0, 0]], top_k=5) == []
        assert mmr_diversify([1, 0, 0], _make_docs(1), [], top_k=5) == []

    def test_single_document(self):
        """Single document should be returned as-is."""
        docs = [{"content": "only doc", "source": "a.txt"}]
        embs = [[0.5, 0.5, 0.0]]

        result = mmr_diversify([1.0, 0.0, 0.0], docs, embs, top_k=1)

        assert len(result) == 1
        assert result[0]["content"] == "only doc"

    def test_zero_query_norm(self):
        """Zero-norm query returns docs in original order."""
        docs = _make_docs(3)
        embs = [[0.1, 0.2, 0.3]] * 3

        result = mmr_diversify([0, 0, 0], docs, embs, top_k=2)

        assert len(result) == 2
        assert result[0]["content"] == "Document 0"

    def test_top_k_clamped(self):
        """top_k larger than docs returns all docs."""
        docs = _make_docs(2)
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        result = mmr_diversify([1, 0, 0], docs, embs, top_k=10)

        assert len(result) == 2


# ===========================================================================
# add_citations tests
# ===========================================================================

class TestAddCitations:

    def test_adds_citation_and_id(self):
        """Each doc should get _citation and _citation_id."""
        docs = _make_docs(3)

        result = add_citations(docs)

        assert len(result) == 3
        for i, doc in enumerate(result):
            assert doc["_citation_id"] == i + 1
            assert doc["_citation"].startswith(f"[{i + 1}]")

    def test_includes_source_info(self):
        """Citation should include the source field."""
        docs = [{"content": "text", "source": "annual_report.xlsx"}]

        result = add_citations(docs)

        assert "annual_report.xlsx" in result[0]["_citation"]

    def test_includes_metadata_when_available(self):
        """Citation should include statement_type, chunk_index, period_columns."""
        docs = [{
            "content": "data",
            "source": "report.xlsx",
            "metadata": {
                "statement_type": "income_statement",
                "chunk_index": 2,
                "period_columns": ["2023", "2024", "2025"],
            },
        }]

        result = add_citations(docs)

        cit = result[0]["_citation"]
        assert "Type: income_statement" in cit
        assert "Section: 3" in cit  # chunk_index + 1
        assert "2023" in cit

    def test_includes_table_structure(self):
        """Citation should include row range from table_structure."""
        docs = [{
            "content": "data",
            "source": "report.xlsx",
            "table_structure": {"row_range": [5, 20]},
        }]

        result = add_citations(docs)

        assert "Rows: 6-20" in result[0]["_citation"]

    def test_includes_cell_references(self):
        """Citation should include cell_references if present."""
        docs = [{
            "content": "data",
            "source": "report.xlsx",
            "cell_references": "A1:D10",
        }]

        result = add_citations(docs)

        assert "Ref: A1:D10" in result[0]["_citation"]

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert add_citations([]) == []

    def test_does_not_mutate_originals(self):
        """Original documents should not be modified."""
        docs = [{"content": "text", "source": "f.txt"}]

        add_citations(docs)

        assert "_citation" not in docs[0]

    def test_skips_unknown_statement_type(self):
        """statement_type='unknown' should not appear in citation."""
        docs = [{
            "content": "data",
            "source": "f.xlsx",
            "metadata": {"statement_type": "unknown"},
        }]

        result = add_citations(docs)

        assert "Type:" not in result[0]["_citation"]

    def test_handles_missing_metadata(self):
        """Documents without metadata should still get citations."""
        docs = [{"content": "data", "source": "f.txt"}]

        result = add_citations(docs)

        assert result[0]["_citation_id"] == 1
        assert "Source: f.txt" in result[0]["_citation"]


# ===========================================================================
# Config defaults
# ===========================================================================

class TestConfigDefaults:

    def test_reranking_defaults(self):
        """Config should have correct retrieval enhancement defaults."""
        from config import Settings

        s = Settings()
        assert s.enable_reranking is False
        assert s.reranking_model == "cross-encoder"
        assert s.rerank_top_n == 20
        assert s.mmr_lambda == pytest.approx(0.7)
        assert s.enable_mmr is True
        assert s.enable_citations is True

    def test_reranker_cache_maxsize_default(self):
        """reranker_cache_maxsize should default to 1024."""
        from config import Settings

        s = Settings()
        assert s.reranker_cache_maxsize == 1024


# ===========================================================================
# P1-A3: Content-hash LRU cache on EmbeddingReranker
# ===========================================================================

def _vec(seed: float, dim: int = 3) -> list:
    """Return a deterministic unit-ish float vector."""
    v = [seed + i * 0.1 for i in range(dim)]
    return v


def _make_embedder_tracked():
    """Return a mock embedder that records every embed_batch call."""
    embedder = MagicMock()
    embedder.embed.return_value = [1.0, 0.0, 0.0]

    call_log: list = []

    def _embed_batch(texts):
        call_log.append(list(texts))
        return [[float(i + 1), 0.0, 0.0] for i in range(len(texts))]

    embedder.embed_batch.side_effect = _embed_batch
    embedder._call_log = call_log
    return embedder


class TestEmbeddingRerankerCache:
    """P1-A3: content-hash LRU cache on EmbeddingReranker instance."""

    def test_second_rerank_embeds_only_new_docs(self):
        """Second rerank() of overlapping docs issues embed_batch only for new docs."""
        embedder = _make_embedder_tracked()
        reranker = EmbeddingReranker(embedder)

        docs_first = [
            {"content": "Alpha text"},
            {"content": "Beta text"},
        ]
        docs_second = [
            {"content": "Alpha text"},   # cache hit
            {"content": "Gamma text"},   # cache miss
        ]

        reranker.rerank("query", docs_first, top_k=2)
        reranker.rerank("query", docs_second, top_k=2)

        # Second call should embed only the new doc
        second_call_texts = embedder._call_log[1]
        assert "Alpha text" not in second_call_texts
        assert "Gamma text" in second_call_texts

    def test_content_collision_distinct_entries(self):
        """Two docs with same initial content but later differing text produce distinct cache entries."""
        embedder = _make_embedder_tracked()
        reranker = EmbeddingReranker(embedder)

        # First rerank: content "A"
        docs_v1 = [{"content": "A"}]
        reranker.rerank("q", docs_v1, top_k=1)

        # Second rerank: same id (no chunk_id) but different content "A_updated"
        docs_v2 = [{"content": "A_updated"}]
        reranker.rerank("q", docs_v2, top_k=1)

        # Both calls should have triggered embed_batch
        assert len(embedder._call_log) == 2
        assert "A_updated" in embedder._call_log[1]

    def test_same_chunk_id_different_content_is_cache_miss(self):
        """Same chunk_id but changed content must NOT reuse the cached embedding."""
        embedder = _make_embedder_tracked()
        reranker = EmbeddingReranker(embedder)

        doc_old = {"content": "Old content", "chunk_id": "c1"}
        doc_new = {"content": "New content", "chunk_id": "c1"}

        reranker.rerank("q", [doc_old], top_k=1)
        reranker.rerank("q", [doc_new], top_k=1)

        # Both calls must embed — stale-content bug guard
        assert len(embedder._call_log) == 2
        assert "New content" in embedder._call_log[1]

    def test_same_chunk_id_same_content_is_cache_hit(self):
        """Same chunk_id AND same content is a cache hit on the second call."""
        embedder = _make_embedder_tracked()
        reranker = EmbeddingReranker(embedder)

        doc = {"content": "Stable content", "chunk_id": "c1"}
        reranker.rerank("q", [doc], top_k=1)
        reranker.rerank("q", [doc], top_k=1)

        # Second call should pass an empty batch (all hits)
        assert len(embedder._call_log) == 1

    def test_cache_size_bounded_by_lru(self):
        """Cache never exceeds reranker_cache_maxsize; oldest entries evicted."""
        small_max = 5
        embedder = _make_embedder_tracked()
        # Inject a tiny cache size directly
        reranker = EmbeddingReranker(embedder, cache_maxsize=small_max)

        # Fill cache beyond the limit with distinct docs
        for i in range(small_max + 3):
            docs = [{"content": f"Unique doc {i}"}]
            reranker.rerank("q", docs, top_k=1)

        assert len(reranker._doc_embed_cache) <= small_max

    def test_cache_not_poisoned_on_embed_failure(self):
        """If embed_batch fails, nothing is inserted into the cache."""
        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]
        embedder.embed_batch.side_effect = RuntimeError("embed fail")

        reranker = EmbeddingReranker(embedder)
        docs = [{"content": "Some doc"}]

        result = reranker.rerank("q", docs, top_k=1)

        # Original order returned (existing behaviour)
        assert result[0]["content"] == "Some doc"
        # Cache must be empty
        assert len(reranker._doc_embed_cache) == 0

    def test_no_embed_batch_call_when_all_cached(self):
        """When all docs are cached, embed_batch is never called on the second rerank."""
        embedder = _make_embedder_tracked()
        reranker = EmbeddingReranker(embedder)

        docs = [{"content": "X"}, {"content": "Y"}]
        reranker.rerank("q", docs, top_k=2)
        reranker.rerank("q", docs, top_k=2)

        # embed_batch called exactly once across both reranks
        assert embedder.embed_batch.call_count == 1


# ===========================================================================
# P1-D6: Integration test - reranker activation via retrieve()
# ===========================================================================

class TestRerankerActivationIntegration:
    """P1-D6: spy-based integration tests for reranker wiring in SimpleRAG.retrieve()."""

    def _make_rag(self):
        """Build a minimal SimpleRAG with mocked embedder and no real disk I/O."""
        from app_local import SimpleRAG

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]
        mock_embedder.embed_batch.side_effect = (
            lambda texts: [[float(i + 1), 0.0, 0.0] for i in range(len(texts))]
        )

        mock_llm = MagicMock()

        rag = SimpleRAG(
            docs_folder=".",
            llm=mock_llm,
            embedder=mock_embedder,
        )

        # Pre-populate in-memory documents and embeddings
        rag.documents = [
            {"content": f"Document {i}", "source": f"file{i}.txt"}
            for i in range(4)
        ]
        rag.embeddings = [[float(i + 1), 0.0, 0.0] for i in range(4)]

        # Build numpy index so _semantic_search uses it directly
        rag._doc_matrix = np.array(rag.embeddings, dtype=np.float32)
        norms = np.linalg.norm(rag._doc_matrix, axis=1)
        rag._doc_norms = np.where(norms == 0, 1.0, norms)

        # Disable BM25 and ANN paths
        rag._bm25_available = False
        rag._bm25_index = None
        rag._vector_index = None

        return rag

    def _settings_patch(self, *, enable_reranking: bool = True):
        """Return a mock settings object with all feature flags set."""
        s = MagicMock()
        s.enable_reranking = enable_reranking
        s.enable_mmr = False
        s.enable_citations = False
        s.enable_parent_expansion = False
        s.enable_tracing = False
        s.enable_chunk_dedup = False
        s.adaptive_top_k = False
        s.enable_hyde = False
        s.enable_query_decomposition = False
        s.top_k = 3
        s.rerank_top_n = 20
        s.max_top_k = 20
        s.bm25_weight = 0.4
        s.semantic_weight = 0.6
        s.rrf_k = 60
        s.max_query_length = 2000
        return s

    def test_retrieve_invokes_reranker_and_uses_output(self):
        """With enable_reranking=True, retrieve() calls rerank() AND its result is used."""
        from reranker import EmbeddingReranker

        rag = self._make_rag()
        sentinel_doc = {"content": "sentinel", "source": "sentinel.txt", "_rerank_score": 0.99}
        sentinel_list = [sentinel_doc]

        mock_settings = self._settings_patch(enable_reranking=True)

        with patch("app_local.settings", mock_settings), \
             patch.object(EmbeddingReranker, "rerank", return_value=sentinel_list) as spy:
            result = rag.retrieve("what is profit?", top_k=3)

        # Reranker must have been invoked
        assert spy.called, "EmbeddingReranker.rerank was NOT called"

        # The reranker output must flow through to retrieve()'s return
        assert result == sentinel_list, (
            "retrieve() did not use the reranker output: "
            f"got {result!r}, expected {sentinel_list!r}"
        )

    def test_retrieve_graceful_when_rerank_raises(self):
        """When rerank() raises, retrieve() still returns results and swallow is logged."""
        from reranker import EmbeddingReranker

        rag = self._make_rag()
        mock_settings = self._settings_patch(enable_reranking=True)

        with patch("app_local.settings", mock_settings), \
             patch.object(EmbeddingReranker, "rerank", side_effect=RuntimeError("boom")) as spy, \
             patch("app_local.logger") as mock_logger:
            result = rag.retrieve("what is revenue?", top_k=3)

        # Must not raise; must return a list
        assert isinstance(result, list)

        # Reranker was attempted
        assert spy.called, "Expected rerank to be called (and raise)"

        # The swallow must be logged at debug level
        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        assert any("Reranking failed" in c or "boom" in c for c in debug_calls), (
            f"Expected debug log about reranking failure; got: {debug_calls}"
        )
