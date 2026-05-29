"""Tests for the Neo4j graph store layer (graph_store.py + graph_schema.py)."""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# graph_schema tests
# ---------------------------------------------------------------------------


class TestGraphSchema:
    def test_vector_index_statement_encodes_model_and_dim(self):
        from graph_schema import vector_index_statement

        stmt = vector_index_statement(1024, "mxbai-embed-large")
        assert "chunk_embedding_mxbai_embed_large" in stmt
        assert "1024" in stmt
        assert "cosine" in stmt

    def test_vector_index_statement_sanitises_slashes(self):
        from graph_schema import vector_index_statement

        stmt = vector_index_statement(384, "org/model-name")
        assert "org_model_name" in stmt
        assert "/" not in stmt.split("IF NOT EXISTS")[0]

    def test_vector_index_statement_rejects_injection_chars(self):
        from graph_schema import vector_index_statement

        malicious = "model`; DROP INDEX foo; //"
        stmt = vector_index_statement(512, malicious)
        # Cypher control chars stripped from index name portion
        idx_part = stmt.split("IF NOT EXISTS")[0]
        assert "`" not in idx_part
        assert ";" not in stmt.split("OPTIONS")[0]
        assert "//" not in idx_part
        # Dimension still correct
        assert "512" in stmt

    def test_constraints_are_idempotent(self):
        from graph_schema import CONSTRAINTS

        for c in CONSTRAINTS:
            assert "IF NOT EXISTS" in c

    def test_cypher_templates_exist(self):
        from graph_schema import (
            GRAPH_CONTEXT_FOR_CHUNK,
            GRAPH_CONTEXT_FOR_CHUNKS_BATCH,
            MERGE_CHUNK,
            MERGE_CHUNKS_BATCH,
            MERGE_DOCUMENT,
            MERGE_FISCAL_PERIOD,
            MERGE_LINE_ITEM,
            MERGE_RATIO,
            MERGE_RATIOS_BATCH,
            MERGE_SCORE,
            MERGE_SCORES_BATCH,
            RATIOS_BY_PERIOD,
            SCORES_BY_PERIOD,
            VECTOR_SEARCH,
        )

        # All should be non-empty strings
        for tmpl in [
            MERGE_DOCUMENT,
            MERGE_CHUNK,
            MERGE_FISCAL_PERIOD,
            MERGE_LINE_ITEM,
            MERGE_RATIO,
            MERGE_SCORE,
            MERGE_CHUNKS_BATCH,
            MERGE_RATIOS_BATCH,
            MERGE_SCORES_BATCH,
            VECTOR_SEARCH,
            GRAPH_CONTEXT_FOR_CHUNK,
            GRAPH_CONTEXT_FOR_CHUNKS_BATCH,
            RATIOS_BY_PERIOD,
            SCORES_BY_PERIOD,
        ]:
            assert isinstance(tmpl, str) and len(tmpl) > 10


# ---------------------------------------------------------------------------
# Neo4jStore unit tests (mocked driver)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    # Support both session.run() (auto-commit) and session.begin_transaction() (explicit tx)
    tx = MagicMock()
    session.begin_transaction.return_value.__enter__ = MagicMock(return_value=tx)
    session.begin_transaction.return_value.__exit__ = MagicMock(return_value=False)
    # Wire tx.run to the same mock as session.run so callers can assert on either
    tx.run = session.run
    tx.commit = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture()
def store(mock_driver):
    from graph_store import Neo4jStore

    driver, _ = mock_driver
    return Neo4jStore(driver)


class TestNeo4jStoreConnect:
    def test_returns_none_when_uri_not_set(self):
        from graph_store import Neo4jStore

        with patch.dict("os.environ", {}, clear=True):
            assert Neo4jStore.connect() is None

    def test_returns_none_when_uri_empty(self):
        from graph_store import Neo4jStore

        with patch.dict("os.environ", {"NEO4J_URI": "  "}):
            assert Neo4jStore.connect() is None

    def test_returns_store_on_success(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        with patch.dict("os.environ", {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": "password123"}):
            with patch("neo4j.GraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = mock_driver
                store = Neo4jStore.connect()
                assert store is not None
                mock_driver.verify_connectivity.assert_called_once()

    def test_returns_none_on_connection_failure(self):
        from graph_store import Neo4jStore

        with patch.dict("os.environ", {"NEO4J_URI": "bolt://badhost:9999"}):
            with patch("neo4j.GraphDatabase") as mock_gdb:
                mock_gdb.driver.side_effect = ConnectionError("Connection refused")
                assert Neo4jStore.connect() is None


class TestNeo4jStoreEnsureSchema:
    def test_runs_constraints_and_vector_index(self, store, mock_driver):
        _, session = mock_driver
        store.ensure_schema(1024, "mxbai-embed-large")
        # Should have run constraints + vector index
        from graph_schema import CONSTRAINTS

        assert session.run.call_count >= len(CONSTRAINTS) + 1


class TestNeo4jStoreChunks:
    def test_store_chunks_creates_nodes(self, store, mock_driver):
        _, session = mock_driver
        chunks = [
            {"content": "Revenue was $1M", "source": "report.pdf", "type": "pdf"},
            {"content": "Assets are $5M", "source": "report.pdf", "type": "pdf"},
        ]
        embeddings = [[0.1] * 10, [0.2] * 10]
        result = store.store_chunks(chunks, embeddings, "report.pdf")
        assert result == 2
        # MERGE_DOCUMENT + 1x MERGE_CHUNKS_BATCH (batched UNWIND)
        assert session.run.call_count == 2

    def test_store_chunks_handles_failure(self, store, mock_driver):
        # WS-1 P0-3 (2026-05-07): transient failures must raise
        # Neo4jTransientError (was: silently returned 0).
        from graph_store import Neo4jTransientError

        _, session = mock_driver
        session.run.side_effect = ConnectionError("Neo4j down")
        with pytest.raises(Neo4jTransientError):
            store.store_chunks([{"content": "x"}], [[0.1]], "f.pdf")


class TestNeo4jStoreFinancialData:
    def test_store_ratios_and_scores(self, store, mock_driver):
        _, session = mock_driver
        store.store_financial_data(
            doc_id="report.pdf",
            period_label="FY2024",
            ratios={
                "current_ratio": {"value": 2.1, "category": "liquidity"},
                "debt_to_equity": {"value": 0.5, "category": "leverage"},
            },
            scores={
                "altman_z": {"value": 3.2, "grade": "Safe", "interpretation": "Low risk"},
            },
        )
        # MERGE_FISCAL_PERIOD + 1 MERGE_RATIOS_BATCH + 1 MERGE_SCORES_BATCH
        assert session.run.call_count == 3


class TestNeo4jStoreVectorSearch:
    def test_vector_search_returns_results(self, store, mock_driver):
        _, session = mock_driver
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"chunk_id": "abc", "content": "Revenue data", "source": "r.pdf", "score": 0.95},
                ]
            )
        )
        session.run.return_value = mock_result

        results = store.vector_search([0.1] * 10, top_k=3, model_name="mxbai-embed-large")
        assert len(results) == 1
        assert results[0]["content"] == "Revenue data"

    def test_vector_search_returns_empty_on_failure(self, store, mock_driver):
        _, session = mock_driver
        session.run.side_effect = ConnectionError("index not found")
        results = store.vector_search([0.1] * 10)
        assert results == []


class TestNeo4jStoreGraphSearch:
    def test_graph_search_enriches_results(self, store, mock_driver):
        _, session = mock_driver

        # First call is vector search, second is batched graph context
        vector_result = MagicMock()
        vector_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"chunk_id": "abc", "content": "Revenue data", "source": "r.pdf", "score": 0.95},
                ]
            )
        )
        graph_batch_result = MagicMock()
        graph_batch_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "chunk_id": "abc",
                        "document": "r.pdf",
                        "period": "FY2024",
                        "ratios": [{"name": "current_ratio", "value": 2.0, "category": "liquidity"}],
                        "scores": [{"model": "altman_z", "value": 3.0, "grade": "Safe"}],
                    },
                ]
            )
        )

        session.run.side_effect = [vector_result, graph_batch_result]
        results = store.graph_search([0.1] * 10, top_k=3)
        assert len(results) == 1
        assert results[0]["document"] == "r.pdf"
        assert results[0]["ratios"][0]["name"] == "current_ratio"


class TestNeo4jStoreClose:
    def test_close_closes_driver(self, store, mock_driver):
        driver, _ = mock_driver
        store.close()
        driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration with SimpleRAG (NEO4J_URI unset = in-memory only)
# ---------------------------------------------------------------------------


class TestSimpleRAGGraphIntegration:
    """Verify SimpleRAG still works without Neo4j (backward compat)."""

    def test_graph_store_is_none_without_neo4j(self):
        """When NEO4J_URI is not set, _graph_store should be None."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure NEO4J_URI is not set
            import os

            os.environ.pop("NEO4J_URI", None)

            from graph_store import Neo4jStore

            store = Neo4jStore.connect()
            assert store is None


# ---------------------------------------------------------------------------
# Phase 1: Graph-enhanced retrieval wiring tests
# ---------------------------------------------------------------------------


class TestNeo4jStoreLineItems:
    """Phase 2: FinancialStatement + LineItem node creation."""

    def test_store_line_items_creates_nodes(self, store, mock_driver):
        _, session = mock_driver
        from financial_analyzer import FinancialData

        fd = FinancialData(revenue=1_000_000, net_income=200_000, total_assets=5_000_000)
        period_id = "test_period_hash"
        count = store.store_line_items(fd, period_id)
        assert count >= 3  # revenue, net_income, total_assets at minimum
        assert session.run.call_count >= 2  # at least 1 statement + 1 batch

    def test_store_line_items_skips_none_values(self, store, mock_driver):
        _, session = mock_driver
        from financial_analyzer import FinancialData

        fd = FinancialData(revenue=100)  # Only revenue set, rest None
        count = store.store_line_items(fd, "p1")
        assert count == 1  # Only revenue stored

    def test_store_line_items_handles_failure(self, store, mock_driver):
        # WS-1 P0-3: transient failures now raise Neo4jTransientError.
        from graph_store import Neo4jTransientError

        _, session = mock_driver
        session.run.side_effect = ConnectionError("Neo4j down")
        from financial_analyzer import FinancialData

        fd = FinancialData(revenue=100, total_assets=500)
        with pytest.raises(Neo4jTransientError):
            store.store_line_items(fd, "p1")


class TestNeo4jStoreDerivedFromEdges:
    """Phase 2: DERIVED_FROM edge creation from RATIO_CATALOG."""

    def test_store_derived_from_edges_links_ratios(self, store, mock_driver):
        _, session = mock_driver
        # Use a known period_id that matches the hashing scheme
        period_id = "test_period_id"
        count = store.store_derived_from_edges(period_id)
        # Should create edges for all catalog ratios that have matching fields
        assert count > 0
        session.run.assert_called_once()

    def test_store_derived_from_edges_handles_failure(self, store, mock_driver):
        # WS-1 P0-3: transient failures now raise Neo4jTransientError.
        from graph_store import Neo4jTransientError

        _, session = mock_driver
        session.run.side_effect = ConnectionError("Neo4j down")
        with pytest.raises(Neo4jTransientError):
            store.store_derived_from_edges("p1")


class TestNeo4jStorePeriodLabelReads:
    """Phase 2/3: Period-label-based read methods."""

    def test_ratios_by_period_label(self, store, mock_driver):
        _, session = mock_driver
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"name": "current_ratio", "value": 2.1, "category": "liquidity"},
                ]
            )
        )
        session.run.return_value = mock_result
        results = store.ratios_by_period_label("FY2024")
        assert len(results) == 1
        assert results[0]["name"] == "current_ratio"

    def test_scores_by_period_label(self, store, mock_driver):
        _, session = mock_driver
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"model": "altman_z", "value": 3.2, "grade": "Safe", "interpretation": "Low risk"},
                ]
            )
        )
        session.run.return_value = mock_result
        results = store.scores_by_period_label("FY2024")
        assert len(results) == 1
        assert results[0]["model"] == "altman_z"


class TestSemanticSearchCallsGraphSearch:
    """Verify _semantic_search uses graph_search when store is present."""

    def test_semantic_search_calls_graph_search_when_store_present(self):
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 10
        mock_store = MagicMock()
        mock_store.graph_search.return_value = [
            {
                "chunk_id": "c1",
                "content": "Revenue $1M",
                "source": "report.pdf",
                "score": 0.9,
                "document": "report.pdf",
                "period": "FY2024",
                "ratios": [{"name": "current_ratio", "value": 2.0, "category": "liquidity"}],
                "scores": [],
            }
        ]

        rag = SimpleRAG.__new__(SimpleRAG)
        rag.documents = [{"source": "report.pdf", "content": "Revenue $1M", "type": "pdf"}]
        rag.embeddings = [[0.1] * 10]
        rag._doc_matrix = None
        rag._doc_norms = None
        rag.embedder = mock_embedder
        rag._graph_store = mock_store
        rag._embedding_model_name = "test-model"

        results = rag._semantic_search("What is the revenue?", top_k=3)
        mock_store.graph_search.assert_called_once()
        assert len(results) == 1
        assert results[0]["_graph_context"]["period"] == "FY2024"

    def test_graph_context_in_financial_prompt(self):
        from app_local import SimpleRAG

        rag = SimpleRAG.__new__(SimpleRAG)
        rag._financial_analysis_cache = ""
        rag._charlie_analyzer = MagicMock()

        relevant_docs = [
            {
                "source": "r.pdf",
                "content": "data",
                "type": "unknown",
                "_graph_context": {
                    "document": "r.pdf",
                    "period": "FY2024",
                    "ratios": [{"name": "current_ratio", "value": 2.1, "category": "liquidity"}],
                    "scores": [],
                },
            }
        ]

        prompt = rag._build_financial_prompt("What is the current ratio?", "context text", [], relevant_docs)
        assert "GRAPH-RETRIEVED FINANCIAL METRICS" in prompt
        assert "current_ratio" in prompt
        assert "FY2024" in prompt

    def test_degrades_to_numpy_when_graph_empty(self):
        """When graph_search returns empty, falls back to numpy."""
        import numpy as np

        from app_local import SimpleRAG

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 10
        mock_store = MagicMock()
        mock_store.graph_search.return_value = []

        rag = SimpleRAG.__new__(SimpleRAG)
        rag.documents = [
            {"source": "a.pdf", "content": "Revenue data", "type": "pdf"},
        ]
        rag.embeddings = [[0.1] * 10]
        rag._doc_matrix = np.asarray([[0.1] * 10], dtype=np.float32)
        rag._doc_norms = np.array([np.linalg.norm([0.1] * 10)])
        rag.embedder = mock_embedder
        rag._graph_store = mock_store
        rag._embedding_model_name = "test-model"

        results = rag._semantic_search("test query", top_k=1)
        # Should have fallen through to numpy path
        assert len(results) == 1
        assert results[0]["source"] == "a.pdf"


# ---------------------------------------------------------------------------
# store_portfolio_analysis tests (CRITICAL gap)
# ---------------------------------------------------------------------------


class TestStorePortfolioAnalysis:
    def _make_risk_summary(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            avg_health_score=65.0,
            min_health_score=40,
            max_health_score=90,
            overall_risk_level="moderate",
            distress_count=1,
            risk_flags=["Company X in distress"],
        )

    def test_returns_portfolio_id(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        result = store.store_portfolio_analysis(
            "TestPortfolio",
            ["A", "B"],
            self._make_risk_summary(),
            70,
        )
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_makes_three_run_calls(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        store.store_portfolio_analysis("P", ["A"], self._make_risk_summary())
        assert mock_session.run.call_count == 3

    def test_returns_none_on_failure(self):
        # WS-1 P0-3: transient failures now raise Neo4jTransientError.
        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("connection lost")
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        with pytest.raises(Neo4jTransientError):
            store.store_portfolio_analysis("P", ["A"], self._make_risk_summary())

    def test_id_is_deterministic(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        risk = self._make_risk_summary()
        id1 = store.store_portfolio_analysis("P", ["A", "B"], risk)
        id2 = store.store_portfolio_analysis("P", ["B", "A"], risk)
        # Sorted company names -> same id regardless of input order
        assert id1 == id2

    def test_empty_company_names(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        result = store.store_portfolio_analysis("P", [], self._make_risk_summary())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# store_compliance_report tests (CRITICAL gap)
# ---------------------------------------------------------------------------


class TestStoreComplianceReport:
    def _make_compliance_report(self):
        from types import SimpleNamespace

        sox = SimpleNamespace(overall_risk="low", risk_score=85)
        sec = SimpleNamespace(disclosure_score=75)
        reg = SimpleNamespace(compliance_pct=83.3)
        audit = SimpleNamespace(risk_level="low", score=80, going_concern_risk=False)
        return SimpleNamespace(sox=sox, sec=sec, regulatory=reg, audit_risk=audit)

    def test_returns_compliance_id(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        result = store.store_compliance_report("TestCo", self._make_compliance_report())
        assert isinstance(result, str)
        assert len(result) == 64

    def test_makes_two_run_calls(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        store.store_compliance_report("TestCo", self._make_compliance_report())
        assert mock_session.run.call_count == 2

    def test_returns_none_on_failure(self):
        # WS-1 P0-3: transient failures now raise Neo4jTransientError.
        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("connection lost")
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        with pytest.raises(Neo4jTransientError):
            store.store_compliance_report("TestCo", self._make_compliance_report())

    def test_none_sub_components(self):
        """When sox/sec/regulatory/audit_risk are None, fallbacks work."""
        from types import SimpleNamespace

        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver

        report = SimpleNamespace(sox=None, sec=None, regulatory=None, audit_risk=None)
        result = store.store_compliance_report("TestCo", report)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Read method error paths (CRITICAL gap)
# ---------------------------------------------------------------------------


class TestGraphStoreReadErrorPaths:
    def _make_store_with_failing_session(self):
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run.side_effect = ConnectionError("read failed")
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver
        return store

    def test_ratios_by_period_label_returns_empty_on_failure(self):
        store = self._make_store_with_failing_session()
        result = store.ratios_by_period_label("Q1-2024")
        assert result == []

    def test_scores_by_period_label_returns_empty_on_failure(self):
        store = self._make_store_with_failing_session()
        result = store.scores_by_period_label("Q1-2024")
        assert result == []

    def test_cross_period_ratio_trend_returns_empty_on_failure(self):
        store = self._make_store_with_failing_session()
        result = store.cross_period_ratio_trend(["Q1", "Q2"])
        assert result == []

    def test_cross_period_ratio_trend_empty_period_list(self):
        store = self._make_store_with_failing_session()
        result = store.cross_period_ratio_trend([])
        assert result == []


# ---------------------------------------------------------------------------
# NEO4J_PASSWORD requirement (M-06)
# ---------------------------------------------------------------------------


class TestNeo4jPasswordRequired:
    def test_connect_without_password_returns_none(self):
        """Neo4jStore.connect() must refuse when NEO4J_PASSWORD is empty."""
        from graph_store import Neo4jStore

        env = {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USERNAME": "neo4j"}
        with patch.dict("os.environ", env, clear=False):
            # Ensure NEO4J_PASSWORD is NOT in the env
            import os

            os.environ.pop("NEO4J_PASSWORD", None)
            result = Neo4jStore.connect()
            assert result is None

    def test_connect_with_password_attempts_driver(self):
        """When NEO4J_PASSWORD is set, connect() proceeds to create driver."""
        from graph_store import Neo4jStore

        env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "test-secret",
        }
        mock_neo4j = MagicMock()
        mock_driver = MagicMock()
        mock_neo4j.GraphDatabase.driver.return_value = mock_driver
        with patch.dict("os.environ", env, clear=False):
            with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
                store = Neo4jStore.connect()
                assert store is not None
                mock_neo4j.GraphDatabase.driver.assert_called_once()
                mock_driver.verify_connectivity.assert_called_once()


# ---------------------------------------------------------------------------
# Error path tests: driver.session() throws, empty results, None data
# ---------------------------------------------------------------------------


class TestGraphStoreErrorPaths:
    """Comprehensive error path testing for Neo4jStore.

    Tests scenarios where:
    - driver.session() throws an exception
    - session.run() returns empty results
    - methods receive None or empty data
    - driver is closed/unavailable
    """

    # -----------------------------------------------------------------------
    # driver.session() throws (connection layer failures)
    # -----------------------------------------------------------------------

    def test_ensure_schema_when_session_throws(self):
        """ensure_schema gracefully handles session() failure."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Neo4j unreachable")
        store = Neo4jStore(mock_driver)

        # Should raise because ensure_schema does not catch session() failures
        with pytest.raises(Exception, match="Neo4j unreachable"):
            store.ensure_schema(1024, "mxbai-embed-large")

    def test_store_chunks_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError so the
        caller can choose to retry instead of treating zero stored as
        nothing-to-write (silent data loss)."""
        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Connection refused")
        store = Neo4jStore(mock_driver)

        with pytest.raises(Neo4jTransientError):
            store.store_chunks([{"content": "test", "source": "f.pdf"}], [[0.1] * 10], "f.pdf")

    def test_store_financial_data_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("DB timeout")
        store = Neo4jStore(mock_driver)

        with pytest.raises(Neo4jTransientError):
            store.store_financial_data(
                doc_id="report.pdf",
                period_label="FY2024",
                ratios={"current_ratio": {"value": 2.1, "category": "liquidity"}},
            )

    def test_store_line_items_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from financial_analyzer import FinancialData
        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Network error")
        store = Neo4jStore(mock_driver)

        fd = FinancialData(revenue=1_000_000, total_assets=5_000_000)
        with pytest.raises(Neo4jTransientError):
            store.store_line_items(fd, "period_id_123")

    def test_vector_search_when_session_throws(self):
        """vector_search returns empty list when session() throws."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Session creation failed")
        store = Neo4jStore(mock_driver)

        results = store.vector_search([0.1] * 10, top_k=5)
        assert results == []

    def test_graph_search_when_session_throws_on_context(self):
        """graph_search degrades to vector results when context query fails."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        # Vector search succeeds, context query fails
        vector_result = MagicMock()
        vector_result.__iter__ = MagicMock(
            return_value=iter([{"chunk_id": "abc", "content": "data", "source": "r.pdf", "score": 0.9}])
        )

        # First call to session.run() succeeds (vector), second fails (context)
        mock_session.run.side_effect = [vector_result, Exception("Context query failed")]
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.graph_search([0.1] * 10, top_k=5)
        # Degrades to vector-only results
        assert len(results) == 1
        assert results[0]["chunk_id"] == "abc"

    def test_store_derived_from_edges_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Connection timeout")
        store = Neo4jStore(mock_driver)

        with pytest.raises(Neo4jTransientError):
            store.store_derived_from_edges("period_id_123")

    def test_link_fiscal_periods_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Session unavailable")
        store = Neo4jStore(mock_driver)

        with pytest.raises(Neo4jTransientError):
            store.link_fiscal_periods(
                [
                    {"label": "FY2023", "period_id": "p1"},
                    {"label": "FY2024", "period_id": "p2"},
                ]
            )

    def test_store_credit_assessment_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from types import SimpleNamespace

        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Neo4j down")
        store = Neo4jStore(mock_driver)

        scorecard = SimpleNamespace(
            total_score=75,
            grade="B",
            recommendation="Approve",
            category_scores={"profitability": 80, "leverage": 70},
            strengths=["Growing revenue"],
            weaknesses=["High debt"],
        )
        debt_capacity = SimpleNamespace(max_additional_debt=500000, current_leverage=0.6)

        with pytest.raises(Neo4jTransientError):
            store.store_credit_assessment("TestCo", scorecard, debt_capacity)

    def test_store_covenant_package_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from types import SimpleNamespace

        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Connection lost")
        store = Neo4jStore(mock_driver)

        covenant_pkg = SimpleNamespace(
            covenant_tier="standard",
            financial_covenants={"debt_to_ebitda": 3.5, "interest_coverage": 2.0},
            reporting_requirements=["Monthly financials"],
            events_of_default=["Material breach"],
        )

        with pytest.raises(Neo4jTransientError):
            store.store_covenant_package("assessment_id_123", covenant_pkg)

    def test_store_portfolio_analysis_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from types import SimpleNamespace

        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("DB unavailable")
        store = Neo4jStore(mock_driver)

        risk_summary = SimpleNamespace(
            avg_health_score=65.0,
            min_health_score=40,
            max_health_score=90,
            overall_risk_level="moderate",
            distress_count=1,
            risk_flags=["Company X distressed"],
        )

        with pytest.raises(Neo4jTransientError):
            store.store_portfolio_analysis("Portfolio1", ["CompanyA", "CompanyB"], risk_summary, 70)

    def test_store_compliance_report_when_session_throws(self):
        """WS-1 P0-3: transient failures must raise Neo4jTransientError."""
        from types import SimpleNamespace

        from graph_store import Neo4jStore, Neo4jTransientError

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Neo4j timeout")
        store = Neo4jStore(mock_driver)

        report = SimpleNamespace(
            sox=SimpleNamespace(overall_risk="low", risk_score=85),
            sec=SimpleNamespace(disclosure_score=75),
            regulatory=SimpleNamespace(compliance_pct=90.0),
            audit_risk=SimpleNamespace(risk_level="low", score=80, going_concern_risk=False),
        )

        with pytest.raises(Neo4jTransientError):
            store.store_compliance_report("CompanyX", report)

    # -----------------------------------------------------------------------
    # session.run() returns empty results
    # -----------------------------------------------------------------------

    def test_vector_search_empty_results(self):
        """vector_search returns empty list when index returns no results."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        # Return empty iterator
        empty_result = MagicMock()
        empty_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = empty_result

        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.vector_search([0.1] * 10, top_k=5)
        assert results == []

    def test_ratios_by_period_label_empty_results(self):
        """ratios_by_period_label returns empty list for non-existent period."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        empty_result = MagicMock()
        empty_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = empty_result

        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.ratios_by_period_label("NonExistent2099")
        assert results == []

    def test_scores_by_period_label_empty_results(self):
        """scores_by_period_label returns empty list for non-existent period."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        empty_result = MagicMock()
        empty_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = empty_result

        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.scores_by_period_label("NonExistent2099")
        assert results == []

    def test_cross_period_ratio_trend_empty_results(self):
        """cross_period_ratio_trend returns empty when no data matches."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        empty_result = MagicMock()
        empty_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = empty_result

        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.cross_period_ratio_trend(["FAKE1", "FAKE2"])
        assert results == []

    def test_graph_search_empty_context_results(self):
        """graph_search returns vector results unchanged when context is empty."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        # Vector search returns 1 result
        vector_result = MagicMock()
        vector_result.__iter__ = MagicMock(
            return_value=iter([{"chunk_id": "xyz", "content": "revenue data", "source": "r.pdf", "score": 0.85}])
        )

        # Context query returns empty
        context_result = MagicMock()
        context_result.__iter__ = MagicMock(return_value=iter([]))

        mock_session.run.side_effect = [vector_result, context_result]
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.graph_search([0.1] * 10)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "xyz"
        # No enrichment should have happened
        assert "document" not in results[0]

    # -----------------------------------------------------------------------
    # Methods receive None or empty data
    # -----------------------------------------------------------------------

    def test_store_chunks_with_empty_list(self):
        """store_chunks handles empty chunk list gracefully."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        result = store.store_chunks([], [], "empty.pdf")
        assert result == 0

    def test_store_chunks_with_none_values_in_chunk(self):
        """store_chunks fails gracefully when chunk has explicit None content."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        chunks = [
            {"content": None, "source": "f.pdf", "type": "pdf"},  # explicit None
            {"source": "f.pdf", "type": "pdf"},  # missing content key
        ]
        embeddings = [[0.1] * 10, [0.2] * 10]

        result = store.store_chunks(chunks, embeddings, "f.pdf")
        # None content is now handled gracefully (coerced to ""), both chunks stored
        assert result == 2

    def test_store_financial_data_with_empty_ratios_and_scores(self):
        """store_financial_data handles None ratios and scores."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_tx = MagicMock()
        mock_session.begin_transaction.return_value.__enter__ = MagicMock(return_value=mock_tx)
        mock_session.begin_transaction.return_value.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        # Both ratios and scores are None
        store.store_financial_data(doc_id="report.pdf", period_label="FY2024", ratios=None, scores=None)
        # Should still create FiscalPeriod node via transaction
        mock_tx.run.assert_called()
        assert mock_tx.run.call_count >= 1

    def test_store_line_items_with_all_none_values(self):
        """store_line_items handles FinancialData with all None fields."""
        from financial_analyzer import FinancialData
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        fd = FinancialData()  # All fields None by default
        count = store.store_line_items(fd, "period_id_123")
        # No line items should be created
        assert count == 0

    def test_link_fiscal_periods_with_empty_list(self):
        """link_fiscal_periods returns 0 for empty period list."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore(mock_driver)

        result = store.link_fiscal_periods([])
        assert result == 0

    def test_link_fiscal_periods_with_single_period(self):
        """link_fiscal_periods returns 0 when fewer than 2 periods."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore(mock_driver)

        result = store.link_fiscal_periods([{"label": "FY2024", "period_id": "p1"}])
        assert result == 0

    def test_store_portfolio_analysis_with_empty_company_list(self):
        """store_portfolio_analysis works with empty company names."""
        from types import SimpleNamespace

        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        risk_summary = SimpleNamespace(
            avg_health_score=0.0,
            min_health_score=0,
            max_health_score=0,
            overall_risk_level="unknown",
            distress_count=0,
            risk_flags=[],
        )

        result = store.store_portfolio_analysis("EmptyPortfolio", [], risk_summary, 0)
        assert isinstance(result, str)

    # -----------------------------------------------------------------------
    # Driver is closed/unavailable
    # -----------------------------------------------------------------------

    def test_close_is_idempotent(self):
        """close() can be called multiple times safely."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        store = Neo4jStore(mock_driver)

        # Call close multiple times
        store.close()
        store.close()
        store.close()

        # Should have called driver.close() 3 times
        assert mock_driver.close.call_count == 3

    def test_close_when_driver_throws(self):
        """close() handles driver.close() exceptions silently."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_driver.close.side_effect = ConnectionError("Already closed")
        store = Neo4jStore(mock_driver)

        # Should not raise
        store.close()
        mock_driver.close.assert_called_once()

    def test_operations_after_close_still_fail_gracefully(self):
        """Operations after close() fail gracefully without crashing."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        # After close, session() will fail
        mock_driver.session.side_effect = ConnectionError("Driver is closed")
        store = Neo4jStore(mock_driver)

        store.close()

        # Subsequent operations should fail gracefully
        result = store.vector_search([0.1] * 10)
        assert result == []

    # -----------------------------------------------------------------------
    # Edge cases with run() returning records with None values
    # -----------------------------------------------------------------------

    def test_vector_search_with_none_chunk_id(self):
        """vector_search handles records with None chunk_id."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        # Record with None chunk_id
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([{"chunk_id": None, "content": "data", "source": "r.pdf", "score": 0.9}])
        )
        mock_session.run.return_value = mock_result

        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.vector_search([0.1] * 10)
        # Should still return the record even with None chunk_id
        assert len(results) == 1
        assert results[0]["chunk_id"] is None

    def test_graph_search_with_none_chunk_id_in_context(self):
        """graph_search handles context records with None chunk_id."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()

        # Vector search result
        vector_result = MagicMock()
        vector_result.__iter__ = MagicMock(
            return_value=iter([{"chunk_id": "xyz", "content": "data", "source": "r.pdf", "score": 0.9}])
        )

        # Context with None chunk_id (shouldn't happen but handle it)
        context_result = MagicMock()
        context_result.__iter__ = MagicMock(
            return_value=iter([{"chunk_id": None, "document": "r.pdf", "period": "FY2024", "ratios": [], "scores": []}])
        )

        mock_session.run.side_effect = [vector_result, context_result]
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        results = store.graph_search([0.1] * 10)
        # Vector result should be returned with no enrichment
        assert len(results) == 1
        assert results[0]["chunk_id"] == "xyz"

    def test_store_chunks_with_mismatched_embedding_count(self):
        """store_chunks handles mismatch between chunks and embeddings counts."""
        from graph_store import Neo4jStore

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = Neo4jStore(mock_driver)
        chunks = [
            {"content": "chunk 1", "source": "f.pdf"},
            {"content": "chunk 2", "source": "f.pdf"},
            {"content": "chunk 3", "source": "f.pdf"},
        ]
        embeddings = [[0.1] * 10, [0.2] * 10]  # Only 2, not 3

        # zip() will stop at the shorter sequence
        result = store.store_chunks(chunks, embeddings, "f.pdf")
        assert result == 2  # Only 2 pairs processed by zip()
