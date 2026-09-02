"""Tests for reranker module: EmbeddingReranker, mmr_diversify, add_citations."""

import numpy as np
import pytest
from unittest.mock import MagicMock

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
        # MMR default flipped to False 2026-06-24: diversity demoted exact-figure
        # matches in this figure-lookup RAG (golden DEEP-TOP1 2->3 with it off);
        # dedup already removed the redundancy MMR guarded against.
        assert s.enable_mmr is False
        assert s.enable_citations is True
