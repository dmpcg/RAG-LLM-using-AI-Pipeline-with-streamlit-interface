"""Tests for graph_retriever.py - graph-augmented financial retrieval."""

from unittest.mock import MagicMock

from graph_retriever import (
    format_graph_context,
    graph_enhanced_search,
    persist_analysis_to_graph,
    persist_structured_analysis_to_graph,
)

# ---------------------------------------------------------------------------
# graph_enhanced_search
# ---------------------------------------------------------------------------


class TestGraphEnhancedSearch:
    def test_returns_chunks_and_financial_context(self):
        store = MagicMock()
        store.graph_search.return_value = [
            {
                "chunk_id": "a1",
                "content": "Revenue was $1M",
                "source": "report.pdf",
                "score": 0.95,
                "document": "report.pdf",
                "period": "FY2024",
                "ratios": [{"name": "current_ratio", "value": 2.0, "category": "liquidity"}],
                "scores": [{"model": "altman_z", "value": 3.2, "grade": "Safe"}],
            },
        ]

        result = graph_enhanced_search(store, [0.1] * 10, top_k=3)
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["content"] == "Revenue was $1M"
        assert len(result["financial_context"]) == 1
        assert result["financial_context"][0]["period"] == "FY2024"

    def test_chunks_without_financial_data(self):
        store = MagicMock()
        store.graph_search.return_value = [
            {
                "chunk_id": "b2",
                "content": "General text",
                "source": "notes.pdf",
                "score": 0.8,
            },
        ]

        result = graph_enhanced_search(store, [0.1] * 10)
        assert len(result["chunks"]) == 1
        assert len(result["financial_context"]) == 0

    def test_empty_results(self):
        store = MagicMock()
        store.graph_search.return_value = []

        result = graph_enhanced_search(store, [0.1] * 10)
        assert result["chunks"] == []
        assert result["financial_context"] == []


# ---------------------------------------------------------------------------
# format_graph_context
# ---------------------------------------------------------------------------


class TestFormatGraphContext:
    def test_formats_ratios_and_scores(self):
        context = [
            {
                "document": "report.pdf",
                "period": "FY2024",
                "ratios": [{"name": "current_ratio", "value": 2.1, "category": "liquidity"}],
                "scores": [{"model": "altman_z", "value": 3.2, "grade": "Safe"}],
            }
        ]
        text = format_graph_context(context)
        assert "report.pdf" in text
        assert "FY2024" in text
        assert "current_ratio" in text
        assert "2.1" in text
        assert "altman_z" in text

    def test_empty_context(self):
        assert format_graph_context([]) == ""

    def test_context_without_ratios(self):
        context = [
            {
                "document": "d.pdf",
                "period": "Q1",
                "ratios": [],
                "scores": [{"model": "zscore", "value": 2.5, "grade": "Grey"}],
            }
        ]
        text = format_graph_context(context)
        assert "zscore" in text
        assert "Ratios:" not in text


# ---------------------------------------------------------------------------
# persist_analysis_to_graph
# ---------------------------------------------------------------------------


class TestPersistAnalysisToGraph:
    def test_extracts_and_persists_ratios(self):
        store = MagicMock()
        report = MagicMock()
        report.sections = {
            "ratio_analysis": "Current Ratio: 2.10\nDebt to Equity: 0.45",
            "scoring_models": "Altman Z: 3.20 (Safe zone)",
        }

        persist_analysis_to_graph(store, "doc.pdf", "FY2024", report)
        store.store_financial_data.assert_called_once()
        call_kwargs = store.store_financial_data.call_args
        # Should have extracted ratios
        ratios = call_kwargs.kwargs.get("ratios") or call_kwargs[1].get("ratios", {})
        assert len(ratios) >= 1

    def test_noop_when_store_is_none(self):
        """Should not raise when store is None."""
        report = MagicMock()
        report.sections = {}
        persist_analysis_to_graph(None, "doc.pdf", "FY2024", report)

    def test_handles_unparseable_sections(self):
        store = MagicMock()
        report = MagicMock()
        report.sections = {
            "ratio_analysis": "No ratios available",
            "scoring_models": "No scores",
        }
        persist_analysis_to_graph(store, "doc.pdf", "FY2024", report)
        # Should still call store_financial_data (possibly with empty dicts)
        # or not call if no data extracted


# ---------------------------------------------------------------------------
# persist_structured_analysis_to_graph (Phase 2)
# ---------------------------------------------------------------------------


class TestPersistStructuredAnalysis:
    def test_structured_persist_with_financial_data(self):
        from financial_analyzer import FinancialData

        store = MagicMock()
        fd = FinancialData(revenue=1_000_000, net_income=200_000, total_assets=5_000_000)

        persist_structured_analysis_to_graph(
            store,
            "doc.pdf",
            "FY2024",
            financial_data=fd,
            ratio_results=None,
        )
        # Should call store_line_items and store_derived_from_edges
        store.store_line_items.assert_called_once()
        store.store_derived_from_edges.assert_called_once()

    def test_structured_persist_with_ratio_results(self):
        store = MagicMock()
        from ratio_framework import RatioResult

        results = {
            "roa": RatioResult(name="Return on Assets (ROA)", value=0.04, score=4.0, grade="Weak", summary=""),
        }

        persist_structured_analysis_to_graph(
            store,
            "doc.pdf",
            "FY2024",
            financial_data=None,
            ratio_results=results,
        )
        store.store_financial_data.assert_called_once()
        call_kwargs = store.store_financial_data.call_args
        ratios = call_kwargs.kwargs.get("ratios") or call_kwargs[1].get("ratios", {})
        assert "Return on Assets (ROA)" in ratios

    def test_structured_persist_noop_when_store_is_none(self):
        """Should not raise when store is None."""
        persist_structured_analysis_to_graph(None, "doc.pdf", "FY2024")


# ---------------------------------------------------------------------------
# Neo4j health check (Phase 2D)
# ---------------------------------------------------------------------------


class TestNeo4jHealthCheck:
    def test_returns_ok_when_not_configured(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("NEO4J_URI", None)
            from healthcheck import check_neo4j_connection

            result = check_neo4j_connection()
            assert result["status"] == "ok"
            assert "not configured" in result["detail"]

    def test_returns_ok_when_connected(self):
        from unittest.mock import patch

        mock_store = MagicMock()
        with patch.dict("os.environ", {"NEO4J_URI": "bolt://localhost:7687"}):
            with patch("graph_store.Neo4jStore.connect", return_value=mock_store):
                from healthcheck import check_neo4j_connection

                result = check_neo4j_connection()
                assert result["status"] == "ok"
                mock_store.close.assert_called_once()
