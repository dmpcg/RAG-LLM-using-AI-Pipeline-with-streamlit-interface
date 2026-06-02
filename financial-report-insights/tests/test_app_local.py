"""Tests for app_local.py SimpleRAG core engine."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock providers (satisfy protocols.py LLMProvider / EmbeddingProvider)
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock LLM that satisfies the LLMProvider protocol."""

    def __init__(self, response="Mock answer"):
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response

    def generate_stream(self, prompt: str):
        for word in self._response.split():
            yield word + " "


class MockEmbedder:
    """Mock embedder that satisfies the EmbeddingProvider protocol."""

    def __init__(self, dim=4):
        self._dim = dim
        self._call_count = 0

    def embed(self, text: str) -> list:
        self._call_count += 1
        np.random.seed(abs(hash(text)) % (2**31))
        return np.random.randn(self._dim).tolist()

    def embed_batch(self, texts: list) -> list:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_docs_folder(tmp_path):
    """Create an empty documents folder."""
    docs = tmp_path / "documents"
    docs.mkdir()
    return docs


@pytest.fixture
def docs_folder_with_text(tmp_path):
    """Create a documents folder with a text file."""
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "sample.txt").write_text(
        "Revenue was 1 million dollars. Net income was 200 thousand. "
        "Total assets are 5 million. The company has strong liquidity."
    )
    return docs


@pytest.fixture
def docs_folder_with_xlsx(tmp_path):
    """Create a documents folder with a simple Excel file."""
    import pandas as pd

    docs = tmp_path / "xlsx_documents"
    docs.mkdir()
    df = pd.DataFrame(
        {
            "Line Item": ["Revenue", "COGS", "Net Income"],
            "Amount": [1000000, 600000, 200000],
        }
    )
    df.to_excel(docs / "financials.xlsx", index=False)
    return docs


@pytest.fixture
def rag_empty(empty_docs_folder):
    """SimpleRAG with empty docs folder and mocked providers."""
    from app_local import SimpleRAG

    return SimpleRAG(
        docs_folder=str(empty_docs_folder),
        llm=MockLLM(),
        embedder=MockEmbedder(),
    )


@pytest.fixture
def rag_with_text(docs_folder_with_text):
    """SimpleRAG with a text file loaded."""
    from app_local import SimpleRAG

    return SimpleRAG(
        docs_folder=str(docs_folder_with_text),
        llm=MockLLM(),
        embedder=MockEmbedder(),
    )


@pytest.fixture
def rag_with_docs():
    """SimpleRAG with pre-loaded documents (no disk files)."""
    from app_local import SimpleRAG

    rag = SimpleRAG.__new__(SimpleRAG)
    rag.llm = MockLLM(response="The revenue is 1 million.")
    rag.embedder = MockEmbedder(dim=4)
    rag.documents = [
        {"source": "report.txt", "content": "Revenue was 1 million", "type": "text"},
        {"source": "report.txt", "content": "Net income was 200k", "type": "text"},
        {"source": "data.xlsx", "content": "Total assets 5 million", "type": "excel"},
    ]
    rag.embeddings = [rag.embedder.embed(d["content"]) for d in rag.documents]
    rag._doc_matrix = np.asarray(rag.embeddings, dtype=np.float32)
    norms = np.linalg.norm(rag._doc_matrix, axis=1)
    rag._doc_norms = np.where(norms == 0, 1.0, norms)
    rag._bm25_index = None
    rag._bm25_available = False
    rag._graph_store = None
    rag._lock = threading.Lock()
    rag._embedding_model_name = "test-model"
    rag._excel_processor = None
    rag._charlie_analyzer = None
    rag._financial_analysis_cache = None
    rag._financial_analysis_lock = threading.Lock()
    rag._period_financial_data = {}
    rag.docs_folder = Path("/tmp/docs")
    rag._cache_dir = Path("/tmp/cache")
    return rag


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_di_llm_is_used(self, empty_docs_folder):
        from app_local import SimpleRAG

        mock_llm = MockLLM()
        rag = SimpleRAG(
            docs_folder=str(empty_docs_folder),
            llm=mock_llm,
            embedder=MockEmbedder(),
        )
        assert rag.llm is mock_llm

    def test_di_embedder_is_used(self, empty_docs_folder):
        from app_local import SimpleRAG

        mock_embedder = MockEmbedder()
        rag = SimpleRAG(
            docs_folder=str(empty_docs_folder),
            llm=MockLLM(),
            embedder=mock_embedder,
        )
        assert rag.embedder is mock_embedder

    def test_docs_folder_created(self, tmp_path):
        from app_local import SimpleRAG

        new_folder = tmp_path / "new_docs"
        assert not new_folder.exists()
        rag = SimpleRAG(
            docs_folder=str(new_folder),
            llm=MockLLM(),
            embedder=MockEmbedder(),
        )
        assert new_folder.exists()

    def test_empty_folder_no_documents(self, rag_empty):
        assert rag_empty.documents == []
        assert rag_empty.embeddings == []

    def test_text_file_loaded(self, rag_with_text):
        assert len(rag_with_text.documents) >= 1
        assert len(rag_with_text.embeddings) >= 1
        assert rag_with_text.documents[0]["type"] == "text"
        assert rag_with_text.documents[0]["source"] == "sample.txt"

    def test_graph_store_none_when_no_neo4j(self, empty_docs_folder):
        from app_local import SimpleRAG

        # When Neo4jStore.connect returns None, graph_store should be None
        with patch("graph_store.Neo4jStore") as mock_cls:
            mock_cls.connect.return_value = None
            rag = SimpleRAG(
                docs_folder=str(empty_docs_folder),
                llm=MockLLM(),
                embedder=MockEmbedder(),
            )
            assert getattr(rag, "_graph_store", None) is None

    def test_di_store_used(self, empty_docs_folder):
        from app_local import SimpleRAG

        mock_store = MagicMock()
        rag = SimpleRAG(
            docs_folder=str(empty_docs_folder),
            llm=MockLLM(),
            embedder=MockEmbedder(),
            store=mock_store,
        )
        assert rag._graph_store is mock_store


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_single_chunk(self, rag_empty):
        chunks = rag_empty._chunk_text("hello world", chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert "hello world" in chunks[0]

    def test_long_text_multiple_chunks(self, rag_empty):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = rag_empty._chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1

    def test_overlap_works(self, rag_empty):
        text = " ".join(f"w{i}" for i in range(40))
        chunks = rag_empty._chunk_text(text, chunk_size=20, overlap=5)
        if len(chunks) >= 2:
            # Last words of chunk 0 should appear in chunk 1
            words_0 = set(chunks[0].split()[-5:])
            words_1 = set(chunks[1].split()[:10])
            assert len(words_0 & words_1) > 0

    def test_empty_text_returns_original(self, rag_empty):
        chunks = rag_empty._chunk_text("")
        assert len(chunks) == 0  # empty text returns empty list

    def test_whitespace_only_returns_original(self, rag_empty):
        chunks = rag_empty._chunk_text("   ")
        assert len(chunks) == 0  # whitespace-only returns empty list


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self, rag_empty):
        v = [1.0, 0.0, 0.0, 0.0]
        assert rag_empty._cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self, rag_empty):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert rag_empty._cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self, rag_empty):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert rag_empty._cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self, rag_empty):
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]
        assert rag_empty._cosine_similarity(v1, v2) == 0.0

    def test_both_zero_returns_zero(self, rag_empty):
        v = [0.0, 0.0]
        assert rag_empty._cosine_similarity(v, v) == 0.0


# ---------------------------------------------------------------------------
# _is_financial_query
# ---------------------------------------------------------------------------


class TestIsFinancialQuery:
    def test_ratio_keyword(self, rag_empty):
        assert rag_empty._is_financial_query("What is the current ratio?") is True

    def test_revenue_keyword(self, rag_empty):
        assert rag_empty._is_financial_query("What was the revenue?") is True

    def test_zscore_keyword(self, rag_empty):
        assert rag_empty._is_financial_query("Calculate the z-score") is True

    def test_non_financial(self, rag_empty):
        assert rag_empty._is_financial_query("What color is the sky?") is False

    def test_case_insensitive(self, rag_empty):
        assert rag_empty._is_financial_query("EBITDA margin analysis") is True

    def test_dupont_keyword(self, rag_empty):
        assert rag_empty._is_financial_query("Run dupont analysis") is True

    def test_health_score(self, rag_empty):
        assert rag_empty._is_financial_query("What's the health score?") is True


# ---------------------------------------------------------------------------
# _is_temporal_comparison_query
# ---------------------------------------------------------------------------


class TestIsTemporalComparisonQuery:
    def test_year_over_year(self):
        from app_local import SimpleRAG

        assert SimpleRAG._is_temporal_comparison_query("Year over year growth?") is True

    def test_yoy(self):
        from app_local import SimpleRAG

        assert SimpleRAG._is_temporal_comparison_query("YoY revenue change") is True

    def test_compared_to(self):
        from app_local import SimpleRAG

        assert SimpleRAG._is_temporal_comparison_query("Revenue compared to last year") is True

    def test_trend(self):
        from app_local import SimpleRAG

        assert SimpleRAG._is_temporal_comparison_query("What's the trend?") is True

    def test_non_temporal(self):
        from app_local import SimpleRAG

        assert SimpleRAG._is_temporal_comparison_query("What is the revenue?") is False


# ---------------------------------------------------------------------------
# _decompose_query
# ---------------------------------------------------------------------------


class TestDecomposeQuery:
    def test_always_includes_original(self, rag_empty):
        queries = rag_empty._decompose_query("What is the revenue?")
        assert queries[0] == "What is the revenue?"

    def test_financial_query_expansion(self, rag_empty):
        queries = rag_empty._decompose_query("How profitable is the company?")
        # "profit" keyword matches "profitability" aspect
        assert len(queries) >= 2

    def test_non_financial_adds_variant(self, rag_empty):
        queries = rag_empty._decompose_query("What is the weather?")
        assert len(queries) >= 2

    def test_capped_at_four(self, rag_empty):
        queries = rag_empty._decompose_query("revenue growth cash flow debt ratio leverage")
        assert len(queries) <= 4

    def test_question_mark_variant(self, rag_empty):
        queries = rag_empty._decompose_query("What happened?")
        # Should add a "details?" variant
        assert any("details" in q for q in queries)


# ---------------------------------------------------------------------------
# _build_embedding_index
# ---------------------------------------------------------------------------


class TestBuildEmbeddingIndex:
    def test_builds_matrix(self, rag_with_docs):
        assert rag_with_docs._doc_matrix is not None
        assert rag_with_docs._doc_matrix.shape == (3, 4)

    def test_builds_norms(self, rag_with_docs):
        assert rag_with_docs._doc_norms is not None
        assert len(rag_with_docs._doc_norms) == 3

    def test_empty_embeddings(self, rag_empty):
        rag_empty.embeddings = []
        rag_empty._build_embedding_index()
        assert rag_empty._doc_matrix is None
        assert rag_empty._doc_norms is None

    def test_norms_no_zero(self, rag_with_docs):
        """Zero norms should be replaced with 1.0 to avoid division by zero."""
        # All norms should be positive
        assert (rag_with_docs._doc_norms > 0).all()


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_returns_documents(self, rag_with_docs):
        results = rag_with_docs.retrieve("revenue", top_k=2)
        assert len(results) == 2
        assert all("content" in r for r in results)

    def test_empty_rag_returns_empty(self, rag_empty):
        results = rag_empty.retrieve("anything")
        assert results == []

    def test_top_k_clamped_to_doc_count(self, rag_with_docs):
        results = rag_with_docs.retrieve("revenue", top_k=100)
        assert len(results) == 3  # Only 3 documents

    def test_top_k_minimum_one(self, rag_with_docs):
        results = rag_with_docs.retrieve("revenue", top_k=0)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# _semantic_search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    def test_returns_correct_count(self, rag_with_docs):
        results = rag_with_docs._semantic_search("revenue analysis", top_k=2)
        assert len(results) == 2

    def test_empty_returns_empty(self, rag_empty):
        results = rag_empty._semantic_search("test", top_k=3)
        assert results == []

    def test_zero_query_vector(self, rag_with_docs):
        """If query embedding is all zeros, should return first documents."""
        original_embed = rag_with_docs.embedder.embed
        rag_with_docs.embedder.embed = lambda text: [0.0, 0.0, 0.0, 0.0]
        results = rag_with_docs._semantic_search("test", top_k=2)
        rag_with_docs.embedder.embed = original_embed
        assert len(results) == 2


# ---------------------------------------------------------------------------
# _fuse_results_rrf
# ---------------------------------------------------------------------------


class TestFuseResultsRRF:
    def test_returns_top_k(self, rag_with_docs):
        docs = rag_with_docs.documents
        semantic = [docs[0], docs[1]]
        bm25 = [docs[1], docs[2]]
        fused = rag_with_docs._fuse_results_rrf(semantic, bm25, top_k=2)
        assert len(fused) == 2

    def test_document_in_both_ranked_higher(self, rag_with_docs):
        docs = rag_with_docs.documents
        # docs[1] appears in both lists (rank 0 in both)
        semantic = [docs[1], docs[0]]
        bm25 = [docs[1], docs[2]]
        fused = rag_with_docs._fuse_results_rrf(semantic, bm25, top_k=3)
        # docs[1] should be first (highest combined RRF score)
        assert fused[0] is docs[1]

    def test_empty_lists(self, rag_with_docs):
        fused = rag_with_docs._fuse_results_rrf([], [], top_k=3)
        assert fused == []

    def test_content_identity_fuses_distinct_dict_objects(self, rag_with_docs):
        """AUD0522-04: same logical chunk in both rankers must fuse on CONTENT
        identity even when the two rankers return distinct dict objects (graph
        mode), so the shared chunk gets the combined (boosted) RRF score."""
        shared_content = "Revenue was 1 million"
        # Semantic ranker returns a freshly-built dict (graph mode) that does NOT
        # share id() with the BM25 self.documents[...] object.
        semantic = [
            {"source": "report.txt", "content": shared_content, "type": "text"},
            {"source": "report.txt", "content": "Net income was 200k", "type": "text"},
        ]
        bm25 = [
            rag_with_docs.documents[2],  # data.xlsx — semantic-only would win
            rag_with_docs.documents[0],  # SAME logical chunk as semantic[0]
        ]
        # Object identity differs for the shared chunk.
        assert semantic[0] is not bm25[1]

        fused = rag_with_docs._fuse_results_rrf(semantic, bm25, top_k=3)
        # The shared chunk appears in both rankers -> highest combined RRF score.
        assert fused[0].get("content") == shared_content
        # And it is deduplicated (not emitted twice).
        contents = [d.get("content") for d in fused]
        assert contents.count(shared_content) == 1

    def test_numpy_mode_object_identity_still_fuses(self, rag_with_docs):
        """Numpy-mode behavior is unchanged: when both rankers return the SAME
        self.documents[i] objects, the overlapping doc is still boosted."""
        docs = rag_with_docs.documents
        semantic = [docs[0], docs[1]]
        bm25 = [docs[1], docs[2]]  # docs[1] overlaps
        fused = rag_with_docs._fuse_results_rrf(semantic, bm25, top_k=3)
        # docs[1] is rank 1 in semantic + rank 0 in bm25 -> highest combined.
        assert fused[0] is docs[1]
        # Three distinct logical chunks survive (no spurious dedup).
        assert len(fused) == 3
        assert {d["content"] for d in fused} == {d["content"] for d in docs}

    def test_mocked_graph_path_fuses_with_bm25(self, rag_with_docs):
        """Mocked graph (Neo4j) semantic path must produce dicts that fuse with
        BM25 results: real 'type' (not 'unknown') + matching content identity."""
        # Wire a mocked graph store whose graph_search returns Neo4j-shaped dicts.
        mock_store = MagicMock()
        mock_store.graph_search.return_value = [
            {
                "source": "report.txt",
                "content": "Revenue was 1 million",
                "type": "text",
                "chunk_id": "graph-cid-1",
                "document": "",
                "period": "",
                "ratios": [],
                "scores": [],
            },
        ]
        rag_with_docs._graph_store = mock_store

        semantic = rag_with_docs._semantic_search("revenue", top_k=2)
        # Neo4j path carries a real type and the chunk_id (not 'unknown').
        assert semantic[0]["type"] == "text"
        assert semantic[0]["metadata"]["chunk_id"] == "graph-cid-1"

        # BM25 returns the in-memory dict for the same logical chunk.
        bm25 = [rag_with_docs.documents[0]]  # same content "Revenue was 1 million"
        assert semantic[0] is not bm25[0]

        fused = rag_with_docs._fuse_results_rrf(semantic, bm25, top_k=3)
        # The shared chunk (content identity match) is boosted to the top and
        # only appears once.
        assert fused[0].get("content") == "Revenue was 1 million"
        contents = [d.get("content") for d in fused]
        assert contents.count("Revenue was 1 million") == 1


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_returns_answer(self, rag_with_docs):
        answer = rag_with_docs.answer("What is the revenue?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_no_docs_returns_message(self, rag_empty):
        answer = rag_empty.answer("What is the revenue?")
        assert "No documents loaded" in answer

    def test_query_too_long(self, rag_with_docs):
        long_query = "a" * 50000
        answer = rag_with_docs.answer(long_query)
        assert "too long" in answer.lower()

    def test_uses_pre_retrieved_docs(self, rag_with_docs):
        docs = [{"source": "test.txt", "content": "Revenue is 500k", "type": "text"}]
        answer = rag_with_docs.answer("revenue?", retrieved_docs=docs)
        assert isinstance(answer, str)

    def test_financial_query_uses_financial_prompt(self, rag_with_docs):
        """Financial queries should trigger the financial prompt builder."""
        # The mock LLM just returns "The revenue is 1 million."
        answer = rag_with_docs.answer("What is the EBITDA margin?")
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# answer_stream
# ---------------------------------------------------------------------------


class TestAnswerStream:
    def test_yields_chunks(self, rag_with_docs):
        chunks = list(rag_with_docs.answer_stream("What is the revenue?"))
        assert len(chunks) >= 1
        combined = "".join(chunks)
        assert len(combined) > 0

    def test_no_docs_yields_message(self, rag_empty):
        chunks = list(rag_empty.answer_stream("anything"))
        combined = "".join(chunks)
        assert "No documents loaded" in combined

    def test_query_too_long(self, rag_with_docs):
        chunks = list(rag_with_docs.answer_stream("a" * 50000))
        combined = "".join(chunks)
        assert "too long" in combined.lower()

    def test_with_pre_retrieved_docs(self, rag_with_docs):
        docs = [{"source": "t.txt", "content": "Some content", "type": "text"}]
        chunks = list(rag_with_docs.answer_stream("test", retrieved_docs=docs))
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# retrieve_with_decomposition
# ---------------------------------------------------------------------------


class TestRetrieveWithDecomposition:
    def test_returns_documents(self, rag_with_docs):
        results = rag_with_docs.retrieve_with_decomposition("revenue growth", top_k=2)
        assert len(results) >= 1
        assert len(results) <= 2

    def test_deduplicates(self, rag_with_docs):
        results = rag_with_docs.retrieve_with_decomposition("revenue analysis", top_k=10)
        # Should not have duplicates by source+content
        keys = [(r["source"], r["content"][:200]) for r in results]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# _embedding_cache_key
# ---------------------------------------------------------------------------


class TestEmbeddingCacheKey:
    def test_deterministic(self, rag_empty, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        key1 = rag_empty._embedding_cache_key(f)
        key2 = rag_empty._embedding_cache_key(f)
        assert key1 == key2

    def test_different_content_different_key(self, rag_empty, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert rag_empty._embedding_cache_key(f1) != rag_empty._embedding_cache_key(f2)

    def test_key_is_hex_string(self, rag_empty, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        key = rag_empty._embedding_cache_key(f)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# _load_cached_embeddings / _save_cached_embeddings
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_roundtrip(self, rag_empty, tmp_path):
        rag_empty._cache_dir = tmp_path / "cache"
        rag_empty._cache_dir.mkdir()

        docs = [{"content": "hello", "source": "test.txt"}]
        embs = [[0.1, 0.2, 0.3]]
        rag_empty._save_cached_embeddings("key123", docs, embs)

        loaded = rag_empty._load_cached_embeddings("key123")
        assert loaded is not None
        loaded_docs, loaded_embs = loaded
        assert loaded_docs == docs
        assert loaded_embs == embs

    def test_missing_key_returns_none(self, rag_empty, tmp_path):
        rag_empty._cache_dir = tmp_path / "cache"
        rag_empty._cache_dir.mkdir()
        assert rag_empty._load_cached_embeddings("nonexistent") is None


# ---------------------------------------------------------------------------
# reload_documents
# ---------------------------------------------------------------------------


class TestReloadDocuments:
    def test_clears_state(self, rag_with_docs):
        assert len(rag_with_docs.documents) > 0
        # Mock _load_documents to not actually load
        rag_with_docs._load_documents = lambda: None
        rag_with_docs.reload_documents()
        assert rag_with_docs.documents == []
        assert rag_with_docs.embeddings == []
        assert rag_with_docs._doc_matrix is None
        assert rag_with_docs._doc_norms is None
        assert rag_with_docs._bm25_index is None
        assert rag_with_docs._financial_analysis_cache is None


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_txt_file(self, rag_empty, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        result = rag_empty._extract_text(f)
        assert result == "Hello world"

    def test_md_file(self, rag_empty, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\nContent here")
        result = rag_empty._extract_text(f)
        assert "Title" in result

    def test_empty_txt_returns_none(self, rag_empty, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = rag_empty._extract_text(f)
        assert result is None

    def test_whitespace_only_returns_none(self, rag_empty, tmp_path):
        f = tmp_path / "spaces.txt"
        f.write_text("   \n  \t  ")
        result = rag_empty._extract_text(f)
        assert result is None

    def test_unsupported_extension_returns_none(self, rag_empty, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        result = rag_empty._extract_text(f)
        assert result is None


# ---------------------------------------------------------------------------
# _process_excel_file
# ---------------------------------------------------------------------------


class TestProcessExcelFile:
    def test_processes_xlsx(self, docs_folder_with_xlsx):
        from app_local import SimpleRAG

        xlsx = docs_folder_with_xlsx / "financials.xlsx"
        rag = SimpleRAG(
            docs_folder=str(docs_folder_with_xlsx),
            llm=MockLLM(),
            embedder=MockEmbedder(),
        )
        rag._excel_processor = None  # Force lazy load

        # The excel_processor property should lazy-load and process
        docs = rag._process_excel_file(xlsx)
        assert isinstance(docs, list)
        # Should produce at least one chunk
        assert len(docs) >= 1

    def test_no_processor_returns_empty(self, rag_empty, tmp_path):
        rag_empty._excel_processor = None
        # Patch the property to return None (simulating missing dependency)
        with patch.object(type(rag_empty), "excel_processor", new_callable=lambda: property(lambda self: None)):
            docs = rag_empty._process_excel_file(tmp_path / "fake.xlsx")
            assert docs == []


# ---------------------------------------------------------------------------
# Lazy properties
# ---------------------------------------------------------------------------


class TestLazyProperties:
    def test_excel_processor_lazy_load(self, rag_empty):
        assert rag_empty._excel_processor is None
        proc = rag_empty.excel_processor
        # Should load (excel_processor is available in the project)
        assert proc is not None

    def test_charlie_analyzer_lazy_load(self, rag_empty):
        assert rag_empty._charlie_analyzer is None
        analyzer = rag_empty.charlie_analyzer
        assert analyzer is not None


# ---------------------------------------------------------------------------
# EXCEL_EXTENSIONS constant
# ---------------------------------------------------------------------------


class TestExcelExtensions:
    def test_xlsx_supported(self):
        from app_local import SimpleRAG

        assert ".xlsx" in SimpleRAG.EXCEL_EXTENSIONS

    def test_csv_supported(self):
        from app_local import SimpleRAG

        assert ".csv" in SimpleRAG.EXCEL_EXTENSIONS

    def test_tsv_supported(self):
        from app_local import SimpleRAG

        assert ".tsv" in SimpleRAG.EXCEL_EXTENSIONS


# ---------------------------------------------------------------------------
# _build_financial_prompt
# ---------------------------------------------------------------------------


class TestBuildFinancialPrompt:
    def test_returns_string(self, rag_with_docs):
        prompt = rag_with_docs._build_financial_prompt(
            query="What is the margin?",
            context="Revenue: 1M",
            excel_data=[],
        )
        assert isinstance(prompt, str)
        assert "What is the margin?" in prompt

    def test_contains_framework(self, rag_with_docs):
        prompt = rag_with_docs._build_financial_prompt(
            query="margin?",
            context="data",
            excel_data=[],
        )
        assert "Charlie Munger" in prompt
        assert "margin of safety" in prompt

    def test_includes_context(self, rag_with_docs):
        prompt = rag_with_docs._build_financial_prompt(
            query="q",
            context="Revenue was 5 million dollars",
            excel_data=[],
        )
        assert "Revenue was 5 million dollars" in prompt


# ---------------------------------------------------------------------------
# _get_financial_analysis_context
# ---------------------------------------------------------------------------


class TestGetFinancialAnalysisContext:
    def test_returns_empty_no_analyzer(self, rag_empty):
        result = rag_empty._get_financial_analysis_context()
        assert result == ""

    def test_caches_result(self, rag_empty):
        rag_empty._financial_analysis_cache = "cached value"
        result = rag_empty._get_financial_analysis_context()
        assert result == "cached value"

    def test_thread_safety(self, rag_empty):
        """Multiple threads calling should not crash."""
        results = []

        def call():
            r = rag_empty._get_financial_analysis_context()
            results.append(r)

        threads = [threading.Thread(target=call) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 5
        # All should be the same value (cached)
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# _load_documents with file size limit
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _expand_parent_chunks
# ---------------------------------------------------------------------------


class TestExpandParentChunks:
    def test_child_expanded_to_parent(self, rag_empty):
        """Child chunks should have content replaced with parent_text."""
        docs = [
            {
                "content": "child chunk text",
                "source": "test.pdf",
                "metadata": {
                    "chunk_level": "child",
                    "parent_id": "p1",
                    "parent_text": "full parent text with more context",
                },
            }
        ]
        result = rag_empty._expand_parent_chunks(docs)
        assert len(result) == 1
        assert result[0]["content"] == "full parent text with more context"
        assert result[0]["_child_content"] == "child chunk text"

    def test_parent_chunk_unchanged(self, rag_empty):
        """Parent-level chunks should pass through unchanged."""
        docs = [
            {
                "content": "parent chunk text",
                "source": "test.pdf",
                "metadata": {"chunk_level": "parent"},
            }
        ]
        result = rag_empty._expand_parent_chunks(docs)
        assert len(result) == 1
        assert result[0]["content"] == "parent chunk text"

    def test_dedup_by_parent_id(self, rag_empty):
        """Multiple children from same parent should be deduped."""
        docs = [
            {
                "content": "child 1",
                "source": "test.pdf",
                "metadata": {
                    "chunk_level": "child",
                    "parent_id": "p1",
                    "parent_text": "shared parent",
                },
            },
            {
                "content": "child 2",
                "source": "test.pdf",
                "metadata": {
                    "chunk_level": "child",
                    "parent_id": "p1",
                    "parent_text": "shared parent",
                },
            },
        ]
        result = rag_empty._expand_parent_chunks(docs)
        assert len(result) == 1

    def test_different_parents_kept(self, rag_empty):
        """Children from different parents should both appear."""
        docs = [
            {
                "content": "child 1",
                "source": "test.pdf",
                "metadata": {
                    "chunk_level": "child",
                    "parent_id": "p1",
                    "parent_text": "parent one",
                },
            },
            {
                "content": "child 2",
                "source": "test.pdf",
                "metadata": {
                    "chunk_level": "child",
                    "parent_id": "p2",
                    "parent_text": "parent two",
                },
            },
        ]
        result = rag_empty._expand_parent_chunks(docs)
        assert len(result) == 2

    def test_no_metadata(self, rag_empty):
        """Docs without metadata pass through."""
        docs = [{"content": "plain doc", "source": "test.txt"}]
        result = rag_empty._expand_parent_chunks(docs)
        assert len(result) == 1
        assert result[0]["content"] == "plain doc"

    def test_child_without_parent_text(self, rag_empty):
        """Child without parent_text passes through unchanged."""
        docs = [
            {
                "content": "orphan child",
                "source": "test.pdf",
                "metadata": {"chunk_level": "child", "parent_id": "p1"},
            }
        ]
        result = rag_empty._expand_parent_chunks(docs)
        assert len(result) == 1
        assert result[0]["content"] == "orphan child"

    def test_mixed_children_and_non_pipeline(self, rag_empty):
        """Mix of pipeline children and legacy docs."""
        docs = [
            {"content": "legacy doc", "source": "old.txt", "metadata": {}},
            {
                "content": "child",
                "source": "new.pdf",
                "metadata": {
                    "chunk_level": "child",
                    "parent_id": "p1",
                    "parent_text": "expanded parent",
                },
            },
            {"content": "table chunk", "source": "data.xlsx", "metadata": {"chunk_level": "atomic_table"}},
        ]
        result = rag_empty._expand_parent_chunks(docs)
        assert len(result) == 3
        assert result[0]["content"] == "legacy doc"
        assert result[1]["content"] == "expanded parent"
        assert result[2]["content"] == "table chunk"

    def test_empty_list(self, rag_empty):
        result = rag_empty._expand_parent_chunks([])
        assert result == []


# ---------------------------------------------------------------------------
# WP-B9 -- _PROMPTS_AVAILABLE removed; real prompts always used
# ---------------------------------------------------------------------------


class TestPromptsAvailableRemoved:
    def test_flag_does_not_exist_in_module(self):
        import app_local

        assert not hasattr(app_local, "_PROMPTS_AVAILABLE"), "_PROMPTS_AVAILABLE must be removed from app_local (WP-B9)"

    def test_prompts_functions_importable(self):
        # The real prompts subtree must be importable (permanent dependency).
        from prompts import build_prompt, format_context_with_citations, get_prompt_for_query_type  # noqa: F401

    def test_answer_uses_real_prompt_for_non_financial_query(self, rag_with_docs):
        """Non-financial query (no charlie_analyzer path) must use the real
        prompts-module path and produce a string prompt that the LLM sees."""

        captured_prompts = []

        original_generate = rag_with_docs.llm.generate

        def spy_generate(prompt: str) -> str:
            captured_prompts.append(prompt)
            return original_generate(prompt)

        rag_with_docs.llm.generate = spy_generate
        # charlie_analyzer=None disables the financial path so we fall through
        # to the prompts-module branch.
        rag_with_docs._charlie_analyzer = None

        rag_with_docs.answer("What is the weather like?")

        assert len(captured_prompts) >= 1, "LLM generate should have been called"
        # The real template from prompts/ includes a system/user structure;
        # none of the calls must use the old hardcoded fallback sentinel phrase.
        for p in captured_prompts:
            assert "Use ONLY the information from the context to answer" not in p, (
                "Hardcoded fallback prompt must not be used after WP-B9 removal"
            )

    def test_answer_stream_uses_real_prompt_for_non_financial_query(self, rag_with_docs):
        """answer_stream non-financial path must likewise use the real prompts module.

        Spy on generate_stream (the path taken by MockLLM) to capture the prompt.
        """
        captured_prompts = []

        original_generate_stream = rag_with_docs.llm.generate_stream

        def spy_generate_stream(prompt: str):
            captured_prompts.append(prompt)
            yield from original_generate_stream(prompt)

        rag_with_docs.llm.generate_stream = spy_generate_stream
        rag_with_docs._charlie_analyzer = None

        list(rag_with_docs.answer_stream("What color is the sky?"))

        assert len(captured_prompts) >= 1, "LLM generate_stream should have been called"
        last_prompt = captured_prompts[-1]
        assert "Use ONLY the information from the context to answer" not in last_prompt, (
            "Hardcoded fallback prompt must not be used in answer_stream after WP-B9 removal"
        )

    def test_answer_prompt_contains_query(self, rag_with_docs):
        """The real-prompts path must embed the user query in the generated prompt."""
        captured_prompts = []

        original_generate = rag_with_docs.llm.generate

        def spy_generate(prompt: str) -> str:
            captured_prompts.append(prompt)
            return original_generate(prompt)

        rag_with_docs.llm.generate = spy_generate
        rag_with_docs._charlie_analyzer = None

        query = "Tell me about cash flow projections please"
        rag_with_docs.answer(query)

        assert len(captured_prompts) >= 1, "LLM generate should have been called"
        # All prompt calls should embed the query (real-prompts path).
        assert any(query in p for p in captured_prompts), (
            "The user query must appear in at least one prompt sent to the LLM"
        )


class TestLoadDocumentsFileSizeLimit:
    def test_skips_oversized_file(self, tmp_path):
        from app_local import SimpleRAG

        docs = tmp_path / "documents"
        docs.mkdir()
        # Create a file larger than max_file_size_mb (default 50 MB)
        # We'll patch the setting to 0.001 MB (1 KB) for the test
        big_file = docs / "big.txt"
        big_file.write_text("x" * 2000)  # 2 KB

        with patch("app_local.settings") as mock_settings:
            mock_settings.max_file_size_mb = 0.001  # 1 KB limit
            mock_settings.chunk_size = 500
            mock_settings.chunk_overlap = 50
            mock_settings.embedding_cache_dir = str(tmp_path / "cache")
            mock_settings.llm_timeout_seconds = 30
            mock_settings.llm_max_retries = 3
            mock_settings.circuit_breaker_failure_threshold = 5
            mock_settings.circuit_breaker_recovery_seconds = 60

            rag = SimpleRAG(
                docs_folder=str(docs),
                llm=MockLLM(),
                embedder=MockEmbedder(),
            )
            # The big file should be skipped
            assert rag.documents == []
