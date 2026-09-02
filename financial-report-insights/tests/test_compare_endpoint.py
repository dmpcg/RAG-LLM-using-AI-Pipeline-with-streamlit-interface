"""Tests for the multi-document comparison endpoint and temporal features."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_rag():
    """Return a mock SimpleRAG with graph store and period data."""
    rag = MagicMock()
    rag.documents = [
        {"source": "fy2023.xlsx", "type": "excel", "content": "FY2023 data"},
        {"source": "fy2024.xlsx", "type": "excel", "content": "FY2024 data"},
    ]
    rag.retrieve.return_value = rag.documents[:1]
    rag.answer.return_value = "Comparison result."
    rag.llm = MagicMock()
    rag.llm.circuit_state = "CLOSED"
    rag.charlie_analyzer = MagicMock()
    rag._graph_store = None
    rag._period_financial_data = {}
    rag._financial_analysis_cache = None
    rag._get_financial_analysis_context = MagicMock(return_value="")
    return rag


@pytest.fixture()
def client(mock_rag):
    import api as api_module

    api_module._rag_instance = mock_rag
    api_module._rate_log.clear()
    from api import app

    with TestClient(app) as c:
        yield c
    api_module._rag_instance = None
    api_module._rate_log.clear()


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


class TestCompareEndpoint:
    def test_compare_with_graph_returns_trend_data(self, client, mock_rag):
        mock_store = MagicMock()
        mock_store.cross_period_ratio_trend.return_value = [
            {"period": "FY2023", "ratio_name": "Current Ratio", "value": 1.5, "category": "liquidity"},
            {"period": "FY2024", "ratio_name": "Current Ratio", "value": 2.1, "category": "liquidity"},
        ]
        mock_rag._graph_store = mock_store

        resp = client.post("/compare", json={"period_labels": ["FY2023", "FY2024"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["periods_compared"] == ["FY2023", "FY2024"]
        assert body["graph_trend_data"] is not None
        assert len(body["deltas"]) == 1
        assert body["deltas"][0]["ratio_name"] == "Current Ratio"
        assert body["deltas"][0]["delta"] == pytest.approx(0.6)
        assert len(body["improvements"]) == 1

    def test_compare_in_memory_fallback(self, client, mock_rag):
        """When no graph, compare uses in-memory cached data."""
        from financial_analyzer import FinancialData

        mock_rag._graph_store = None
        mock_rag._period_financial_data = {
            "FY2023": FinancialData(revenue=100, net_income=10, total_assets=200),
            "FY2024": FinancialData(revenue=120, net_income=15, total_assets=210),
        }

        resp = client.post("/compare", json={"period_labels": ["FY2023", "FY2024"]})
        assert resp.status_code == 200
        body = resp.json()
        assert "Compared 2 periods" in body["summary"]
        # Should have computed some deltas from ratio_framework
        assert len(body["deltas"]) > 0

    def test_compare_degrades_gracefully_without_graph(self, client, mock_rag):
        """No graph, no cached data -> empty but valid response."""
        mock_rag._graph_store = None
        mock_rag._period_financial_data = {}

        resp = client.post("/compare", json={"period_labels": ["FY2023", "FY2024"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["graph_trend_data"] is None
        assert body["periods_compared"] == ["FY2023", "FY2024"]

    def test_compare_rejects_single_period(self, client, mock_rag):
        resp = client.post("/compare", json={"period_labels": ["FY2024"]})
        assert resp.status_code == 422

    def test_compare_deterioration_detection(self, client, mock_rag):
        mock_store = MagicMock()
        mock_store.cross_period_ratio_trend.return_value = [
            {"period": "FY2023", "ratio_name": "ROA", "value": 0.10, "category": "profitability"},
            {"period": "FY2024", "ratio_name": "ROA", "value": 0.04, "category": "profitability"},
        ]
        mock_rag._graph_store = mock_store

        resp = client.post("/compare", json={"period_labels": ["FY2023", "FY2024"]})
        body = resp.json()
        assert len(body["deteriorations"]) == 1
        assert "ROA" in body["deteriorations"][0]


# ---------------------------------------------------------------------------
# Temporal query detection
# ---------------------------------------------------------------------------


class TestTemporalQueryDetection:
    def test_detects_temporal_patterns(self):
        from app_local import SimpleRAG

        assert SimpleRAG._is_temporal_comparison_query("What changed from FY2023 to FY2024?")
        assert SimpleRAG._is_temporal_comparison_query("Year over year revenue growth")
        assert SimpleRAG._is_temporal_comparison_query("How has the trend been?")
        assert SimpleRAG._is_temporal_comparison_query("Compare FY2023 vs FY2024")
        assert SimpleRAG._is_temporal_comparison_query("Has ROA improved?")

    def test_rejects_non_temporal(self):
        from app_local import SimpleRAG

        assert not SimpleRAG._is_temporal_comparison_query("What is the current ratio?")
        assert not SimpleRAG._is_temporal_comparison_query("Show me the balance sheet")
        assert not SimpleRAG._is_temporal_comparison_query("Calculate ROE")


# ---------------------------------------------------------------------------
# Temporal edge creation (graph_store)
# ---------------------------------------------------------------------------


class TestTemporalEdgeCreation:
    def test_link_fiscal_periods(self):
        from graph_store import Neo4jStore

        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        store = Neo4jStore(driver)

        periods = [
            {"label": "FY2022", "period_id": "p1"},
            {"label": "FY2023", "period_id": "p2"},
            {"label": "FY2024", "period_id": "p3"},
        ]
        count = store.link_fiscal_periods(periods)
        assert count == 2
        session.run.assert_called_once()

    def test_link_fiscal_periods_needs_two(self):
        from graph_store import Neo4jStore

        driver = MagicMock()
        store = Neo4jStore(driver)
        assert store.link_fiscal_periods([{"label": "FY2024", "period_id": "p1"}]) == 0


# ---------------------------------------------------------------------------
# Compare endpoint edge cases
# ---------------------------------------------------------------------------


class TestCompareEdgeCases:
    def test_compare_graph_exception_triggers_fallback(self, client, mock_rag):
        """When graph store throws, should fall back to in-memory gracefully."""
        mock_store = MagicMock()
        mock_store.cross_period_ratio_trend.side_effect = RuntimeError("DB down")
        mock_rag._graph_store = mock_store

        resp = client.post("/compare", json={"period_labels": ["FY2023", "FY2024"]})
        assert resp.status_code == 200
        body = resp.json()
        # Graph trend data should be None since graph failed
        assert body["graph_trend_data"] is None
        assert "Compared 2 periods" in body["summary"]

    def test_compare_graph_returns_empty_list(self, client, mock_rag):
        """When graph returns [], should fall back to in-memory."""
        mock_store = MagicMock()
        mock_store.cross_period_ratio_trend.return_value = []
        mock_rag._graph_store = mock_store

        resp = client.post("/compare", json={"period_labels": ["FY2023", "FY2024"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["graph_trend_data"] is None

    def test_compare_max_periods_10(self, client, mock_rag):
        """Max 10 periods should be accepted."""
        labels = [f"FY{2015 + i}" for i in range(10)]
        resp = client.post("/compare", json={"period_labels": labels})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["periods_compared"]) == 10

    def test_compare_rejects_11_periods(self, client, mock_rag):
        """More than 10 periods should be rejected by validation."""
        labels = [f"FY{2010 + i}" for i in range(11)]
        resp = client.post("/compare", json={"period_labels": labels})
        assert resp.status_code == 422

    def test_compare_identical_periods_zero_delta(self, client, mock_rag):
        """Identical period labels should produce zero deltas."""
        mock_store = MagicMock()
        mock_store.cross_period_ratio_trend.return_value = [
            {"period": "FY2024", "ratio_name": "ROA", "value": 0.08, "category": "profitability"},
            {"period": "FY2024", "ratio_name": "ROA", "value": 0.08, "category": "profitability"},
        ]
        mock_rag._graph_store = mock_store

        resp = client.post("/compare", json={"period_labels": ["FY2024", "FY2024"]})
        assert resp.status_code == 200
        body = resp.json()
        for d in body["deltas"]:
            if d["delta"] is not None:
                assert d["delta"] == pytest.approx(0.0)
        # No improvements or deteriorations for zero delta
        assert len(body["improvements"]) == 0
        assert len(body["deteriorations"]) == 0

    def test_compare_graph_row_missing_ratio_name(self, client, mock_rag):
        """Graph row with no ratio_name should be skipped."""
        mock_store = MagicMock()
        mock_store.cross_period_ratio_trend.return_value = [
            {"period": "FY2023", "ratio_name": "", "value": 1.0, "category": ""},
            {"period": "FY2024", "ratio_name": "ROA", "value": 0.08, "category": "profitability"},
        ]
        mock_rag._graph_store = mock_store

        resp = client.post("/compare", json={"period_labels": ["FY2023", "FY2024"]})
        assert resp.status_code == 200
        body = resp.json()
        # Only ROA should appear (blank ratio_name skipped)
        ratio_names = [d["ratio_name"] for d in body["deltas"]]
        assert "" not in ratio_names

    def test_compare_missing_body(self, client, mock_rag):
        """Missing request body should return 422."""
        resp = client.post("/compare")
        assert resp.status_code == 422
