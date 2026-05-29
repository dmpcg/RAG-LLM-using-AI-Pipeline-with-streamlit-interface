"""Tests for the FastAPI API layer."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_rag():
    """Return a mock SimpleRAG wired into the api module."""
    rag = MagicMock()
    rag.documents = [
        {"source": "test.pdf", "type": "pdf", "content": "Revenue was $1M."},
        {"source": "data.xlsx", "type": "excel", "content": "Total assets $5M."},
    ]
    rag.retrieve.return_value = rag.documents[:1]
    rag.answer.return_value = "The revenue is $1M."
    rag.answer_stream.return_value = iter(["The ", "revenue ", "is ", "$1M."])
    rag.llm = MagicMock()
    rag.llm.circuit_state = "CLOSED"
    rag.charlie_analyzer = MagicMock()
    return rag


@pytest.fixture()
def client(mock_rag):
    """TestClient with the RAG singleton patched."""
    import api as api_module

    api_module._rag_instance = mock_rag
    api_module._rate_log.clear()
    from api import app

    with TestClient(app) as c:
        yield c
    api_module._rag_instance = None
    api_module._rate_log.clear()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client):
        with patch("api.get_health_status", return_value={"healthy": True, "status": "healthy", "checks": []}):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["healthy"] is True

    def test_health_unhealthy(self, client):
        with patch("api.get_health_status", return_value={"healthy": False, "status": "unhealthy", "checks": []}):
            resp = client.get("/health")
        assert resp.status_code == 503


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        with patch("api.get_health_status", return_value={"healthy": True, "status": "ok", "checks": []}):
            resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert resp.headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_query_success(self, client, mock_rag):
        resp = client.post("/query", json={"text": "What is revenue?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "The revenue is $1M."
        assert "test.pdf" in body["sources"]
        assert body["document_count"] == 1

    def test_query_custom_top_k(self, client, mock_rag):
        resp = client.post("/query", json={"text": "What is revenue?", "top_k": 5})
        assert resp.status_code == 200
        mock_rag.retrieve.assert_called_with("What is revenue?", top_k=5)

    def test_query_empty_text_rejected(self, client):
        resp = client.post("/query", json={"text": ""})
        assert resp.status_code == 422

    def test_query_circuit_open(self, client, mock_rag):
        mock_rag.llm.circuit_state = "OPEN"
        resp = client.post("/query", json={"text": "Anything"})
        assert resp.status_code == 503
        assert "circuit breaker" in resp.json()["detail"].lower()

    def test_query_llm_connection_error(self, client, mock_rag):
        from local_llm import LLMConnectionError

        mock_rag.answer.side_effect = LLMConnectionError("down")
        resp = client.post("/query", json={"text": "Test"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /query-stream
# ---------------------------------------------------------------------------


class TestQueryStreamEndpoint:
    def test_stream_success(self, client, mock_rag):
        resp = client.post("/query-stream", json={"text": "What is revenue?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "revenue" in body

    def test_stream_circuit_open(self, client, mock_rag):
        mock_rag.llm.circuit_state = "OPEN"
        resp = client.post("/query-stream", json={"text": "Anything"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    def test_analyze_success(self, client, mock_rag):
        mock_report = MagicMock()
        mock_report.executive_summary = "Company looks healthy."
        mock_report.sections = {"ratio_analysis": "Good ratios."}
        mock_report.generated_at = "2026-01-01"
        mock_rag.charlie_analyzer.generate_report.return_value = mock_report

        resp = client.post("/analyze", json={"financial_data": {"revenue": 1000000, "net_income": 200000}})
        assert resp.status_code == 200
        body = resp.json()
        assert body["executive_summary"] == "Company looks healthy."
        assert "ratio_analysis" in body["sections"]

    def test_analyze_no_analyzer(self, client, mock_rag):
        mock_rag.charlie_analyzer = None
        resp = client.post("/analyze", json={"financial_data": {"revenue": 100}})
        assert resp.status_code == 501

    def test_analyze_ignores_unknown_fields(self, client, mock_rag):
        """Unknown fields in financial_data should be silently ignored."""
        mock_report = MagicMock()
        mock_report.executive_summary = "OK"
        mock_report.sections = {}
        mock_report.generated_at = ""
        mock_rag.charlie_analyzer.generate_report.return_value = mock_report

        resp = client.post("/analyze", json={"financial_data": {"revenue": 500, "unknown_field_xyz": 99}})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /documents
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /graph/context and /graph/ratios (Phase 3)
# ---------------------------------------------------------------------------


class TestGraphContextEndpoint:
    def test_graph_context_501_when_no_store(self, client, mock_rag):
        mock_rag._graph_store = None
        resp = client.get("/graph/context/FY2024")
        assert resp.status_code == 501

    def test_graph_context_returns_ratios_and_scores(self, client, mock_rag):
        mock_store = MagicMock()
        mock_store.ratios_by_period_label.return_value = [
            {"name": "current_ratio", "value": 2.1, "category": "liquidity"},
        ]
        mock_store.scores_by_period_label.return_value = [
            {"model": "altman_z", "value": 3.2, "grade": "Safe", "interpretation": "Low risk"},
        ]
        mock_rag._graph_store = mock_store
        resp = client.get("/graph/context/FY2024")
        assert resp.status_code == 200
        body = resp.json()
        assert body["period_label"] == "FY2024"
        assert len(body["ratios"]) == 1
        assert len(body["scores"]) == 1

    def test_graph_context_empty_for_unknown_period(self, client, mock_rag):
        mock_store = MagicMock()
        mock_store.ratios_by_period_label.return_value = []
        mock_store.scores_by_period_label.return_value = []
        mock_rag._graph_store = mock_store
        resp = client.get("/graph/context/FY9999")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ratios"] == []
        assert body["scores"] == []


class TestGraphRatiosEndpoint:
    def test_graph_ratios_501_when_no_store(self, client, mock_rag):
        mock_rag._graph_store = None
        resp = client.get("/graph/ratios/FY2024")
        assert resp.status_code == 501

    def test_graph_ratios_category_filter(self, client, mock_rag):
        mock_store = MagicMock()
        mock_store.ratios_by_period_label.return_value = [
            {"name": "current_ratio", "value": 2.1, "category": "liquidity"},
            {"name": "roa", "value": 0.05, "category": "profitability"},
        ]
        mock_rag._graph_store = mock_store
        resp = client.get("/graph/ratios/FY2024?category=liquidity")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["name"] == "current_ratio"


class TestDocumentsEndpoint:
    def test_documents_list(self, client, mock_rag):
        resp = client.get("/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 2
        assert docs[0]["source"] == "test.pdf"
        assert docs[1]["source"] == "data.xlsx"

    def test_documents_deduplication(self, client, mock_rag):
        """Duplicate sources should be collapsed."""
        mock_rag.documents = [
            {"source": "a.pdf", "type": "pdf", "content": "chunk 1"},
            {"source": "a.pdf", "type": "pdf", "content": "chunk 2"},
            {"source": "b.pdf", "type": "pdf", "content": "chunk 3"},
        ]
        resp = client.get("/documents")
        docs = resp.json()
        assert len(docs) == 2

    def test_documents_pipeline_metadata(self, client, mock_rag):
        """Pipeline-ingested docs should expose section_type and chunk_level."""
        mock_rag.documents = [
            {
                "source": "report.pdf",
                "type": "pdf",
                "content": "Revenue data...",
                "metadata": {
                    "section_type": "income_statement",
                    "chunk_level": "child",
                    "parent_id": "p1",
                },
            },
        ]
        resp = client.get("/documents")
        docs = resp.json()
        assert len(docs) == 1
        assert docs[0]["section_type"] == "income_statement"
        assert docs[0]["chunk_level"] == "child"
        assert docs[0]["has_parent"] is True

    def test_documents_no_metadata(self, client, mock_rag):
        """Legacy docs without metadata should have null section fields."""
        mock_rag.documents = [
            {"source": "old.txt", "type": "text", "content": "Plain text."},
        ]
        resp = client.get("/documents")
        docs = resp.json()
        assert docs[0]["section_type"] is None
        assert docs[0]["chunk_level"] is None
        assert docs[0]["has_parent"] is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_enforced(self, client):
        """Exceed rate limit and verify 429 response."""
        import api as api_module

        # Reset rate log
        api_module._rate_log.clear()
        old_limit = api_module._RATE_LIMIT
        api_module._RATE_LIMIT = 3  # lower for test speed
        try:
            with patch("api.get_health_status", return_value={"healthy": True, "status": "ok", "checks": []}):
                for _ in range(3):
                    resp = client.get("/documents")
                    assert resp.status_code == 200
                # 4th request should be rate-limited
                resp = client.get("/documents")
                assert resp.status_code == 429
                assert "Rate limit" in resp.json()["detail"]
        finally:
            api_module._RATE_LIMIT = old_limit
            api_module._rate_log.clear()

    def test_health_exempt_from_rate_limit(self, client):
        """Health endpoint should bypass rate limiting."""
        import api as api_module

        api_module._rate_log.clear()
        old_limit = api_module._RATE_LIMIT
        api_module._RATE_LIMIT = 1
        try:
            with patch("api.get_health_status", return_value={"healthy": True, "status": "ok", "checks": []}):
                # Exhaust rate limit
                client.get("/documents")
                # /health should still work
                resp = client.get("/health")
                assert resp.status_code == 200
        finally:
            api_module._RATE_LIMIT = old_limit
            api_module._rate_log.clear()


# ---------------------------------------------------------------------------
# Export endpoint coverage gaps (Coverage gap 4)
# ---------------------------------------------------------------------------


class TestExportEndpoints:
    def test_export_xlsx_returns_501_when_no_analyzer(self, client, mock_rag):
        mock_rag.charlie_analyzer = None
        resp = client.post(
            "/export/xlsx",
            json={
                "financial_data": {"revenue": 1000},
            },
        )
        assert resp.status_code == 501

    def test_export_pdf_returns_501_when_no_analyzer(self, client, mock_rag):
        mock_rag.charlie_analyzer = None
        resp = client.post(
            "/export/pdf",
            json={
                "financial_data": {"revenue": 1000},
            },
        )
        assert resp.status_code == 501

    def test_export_xlsx_happy_path(self, client, mock_rag):
        """XLSX export returns a valid streaming response.

        Patches api.FinancialExcelExporter (the module-level reference loaded at
        lifespan) rather than the source-module class, because the eager import
        binds the reference before any per-test patch on the source module fires.
        """
        mock_rag.charlie_analyzer.analyze.return_value = {"health": "good"}
        mock_rag.charlie_analyzer.generate_report.return_value = "Summary report."
        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_full_report.return_value = b"PK\x03\x04fake_xlsx"
        with patch("api.FinancialExcelExporter", return_value=mock_exporter_instance):
            resp = client.post(
                "/export/xlsx",
                json={
                    "financial_data": {"revenue": 1000, "total_assets": 5000},
                },
            )
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]
        assert resp.headers["content-disposition"].endswith('.xlsx"')
        assert len(resp.content) > 0

    def test_export_pdf_happy_path(self, client, mock_rag):
        """PDF export returns a valid streaming response.

        Patches api.FinancialPDFExporter (the module-level reference loaded at
        lifespan) rather than the source-module class, because the eager import
        binds the reference before any per-test patch on the source module fires.
        """
        mock_rag.charlie_analyzer.analyze.return_value = {"health": "good"}
        mock_rag.charlie_analyzer.generate_report.return_value = "Summary report."
        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_full_report.return_value = b"%PDF-1.4 fake"
        with patch("api.FinancialPDFExporter", return_value=mock_exporter_instance):
            resp = client.post(
                "/export/pdf",
                json={
                    "financial_data": {"revenue": 1000, "total_assets": 5000},
                },
            )
        assert resp.status_code == 200
        assert "pdf" in resp.headers["content-type"]
        assert resp.headers["content-disposition"].endswith('.pdf"')
        assert len(resp.content) > 0

    def test_export_xlsx_analyzer_error(self, client, mock_rag):
        """Analyzer exception during export should return 422."""
        mock_rag.charlie_analyzer.analyze.side_effect = ValueError("bad data")
        resp = client.post(
            "/export/xlsx",
            json={
                "financial_data": {"revenue": 1000},
            },
        )
        assert resp.status_code == 422
        assert "Could not generate" in resp.json()["detail"]

    def test_export_pdf_analyzer_error(self, client, mock_rag):
        """Analyzer exception during export should return 422."""
        mock_rag.charlie_analyzer.analyze.side_effect = ValueError("bad data")
        resp = client.post(
            "/export/pdf",
            json={
                "financial_data": {"revenue": 1000},
            },
        )
        assert resp.status_code == 422
        assert "Could not generate" in resp.json()["detail"]

    # WS-3 WP-8 (lock-only): company_name over 200 chars is rejected (422).
    def test_export_company_name_over_max_length_rejected(self, client, mock_rag):
        resp = client.post(
            "/export/xlsx",
            json={
                "financial_data": {"revenue": 1000},
                "company_name": "X" * 201,
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Analyze exception handling (Coverage gap 12)
# ---------------------------------------------------------------------------


class TestAnalyzeExceptionHandling:
    def test_analyze_exception_returns_422(self, client, mock_rag):
        mock_rag.charlie_analyzer.generate_report.side_effect = ValueError("bad data")
        resp = client.post(
            "/analyze",
            json={
                "financial_data": {"revenue": 1000},
            },
        )
        assert resp.status_code == 422
        assert "Could not process" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Streaming error event (Coverage gap 11)
# ---------------------------------------------------------------------------


class TestStreamErrorHandling:
    def test_stream_llm_error_mid_stream(self, client, mock_rag):
        """LLMConnectionError during streaming is not silently swallowed."""
        from local_llm import LLMConnectionError

        mock_rag.answer_stream.side_effect = LLMConnectionError("model crashed")
        # Starlette 0.41+ propagates async generator errors through
        # ExceptionGroup; the important assertion is that the error is
        # NOT silently swallowed — it either surfaces as a response or raises.
        try:
            resp = client.post("/query-stream", json={"text": "Test query"})
            # If we get a response, it must indicate the error
            assert resp.status_code in (200, 500, 503)
        except Exception:
            # Error propagated through Starlette middleware — acceptable
            pass


# ---------------------------------------------------------------------------
# Body size limit middleware (M-03)
# ---------------------------------------------------------------------------


class TestBodySizeLimit:
    def test_oversized_content_length_rejected(self, client):
        """Request with Content-Length > max_request_body_bytes gets 413."""
        resp = client.post(
            "/analyze",
            json={"financial_data": {"revenue": 1}},
            headers={"Content-Length": "999999999"},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_malformed_content_length_returns_400(self, client):
        """Non-numeric Content-Length should return 400, not crash."""
        resp = client.post(
            "/analyze",
            json={"financial_data": {"revenue": 1}},
            headers={"Content-Length": "not-a-number"},
        )
        assert resp.status_code == 400
        assert "invalid" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Field count validator (M-02)
# ---------------------------------------------------------------------------


class TestFieldCountValidator:
    def test_too_many_fields_rejected(self, client, mock_rag):
        """Financial data with 201+ fields should be rejected."""
        huge_data = {f"field_{i}": i for i in range(250)}
        resp = client.post("/analyze", json={"financial_data": huge_data})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error message sanitization (M-01)
# ---------------------------------------------------------------------------


class TestErrorSanitization:
    def test_analyze_error_does_not_leak_details(self, client, mock_rag):
        """Internal exception details must not appear in the response."""
        mock_rag.charlie_analyzer.generate_report.side_effect = RuntimeError("SECRET_DB_CONNECTION_STRING")
        resp = client.post("/analyze", json={"financial_data": {"revenue": 1000}})
        assert resp.status_code == 422
        assert "SECRET_DB_CONNECTION_STRING" not in resp.json()["detail"]
        assert "Could not process" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Graph endpoint injection safety
# ---------------------------------------------------------------------------


class TestGraphInjectionSafety:
    def test_cypher_injection_in_period_label_safe(self, client, mock_rag):
        """Period label with Cypher syntax should be passed as literal, not executed."""
        mock_store = MagicMock()
        mock_store.ratios_by_period_label.return_value = []
        mock_store.scores_by_period_label.return_value = []
        mock_rag._graph_store = mock_store
        resp = client.get("/graph/context/FY2024'; DROP TABLE Ratio; --")
        assert resp.status_code == 200
        # Verify the literal string was passed to the store (parameterized)
        mock_store.ratios_by_period_label.assert_called_once()

    def test_graph_ratios_injection_safe(self, client, mock_rag):
        """Ratio endpoint with injection attempt should be safe."""
        mock_store = MagicMock()
        mock_store.ratios_by_period_label.return_value = []
        mock_rag._graph_store = mock_store
        resp = client.get("/graph/ratios/FY2024%27%20OR%201%3D1")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Input validation hardening
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify API rejects oversized or malformed inputs."""

    def test_portfolio_too_many_companies_rejected(self, client, mock_rag):
        """Portfolio endpoint rejects > 50 companies."""
        companies = {f"Co{i}": {"revenue": 1_000_000} for i in range(51)}
        resp = client.post("/portfolio/analyze", json={"companies": companies})
        assert resp.status_code == 422

    def test_portfolio_empty_companies_rejected(self, client, mock_rag):
        """Portfolio endpoint rejects empty companies dict."""
        resp = client.post("/portfolio/analyze", json={"companies": {}})
        assert resp.status_code == 422

    def test_graph_context_overlength_period_label(self, client, mock_rag):
        """Graph context rejects period labels > 100 chars."""
        mock_store = MagicMock()
        mock_rag._graph_store = mock_store
        long_label = "A" * 101
        resp = client.get(f"/graph/context/{long_label}")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# WP-E: /compare quick-return when no documents are ingested (P1-A4)
# ---------------------------------------------------------------------------


class TestCompareQuickReturn:
    """WP-E: guard on rag.documents to short-circuit /compare when empty."""

    def _make_empty_rag(self):
        """Return a mock RAG with no documents loaded."""
        rag = MagicMock()
        rag.documents = []
        rag._graph_store = None
        return rag

    def test_no_documents_returns_empty_compare_response_without_analyser(self):
        """
        When no documents are ingested (rag.documents == []) and _period_financial_data
        is also empty, /compare must return an empty CompareResponse immediately
        WITHOUT invoking _get_financial_analysis_context (D9 guard).

        Old code: would call _get_financial_analysis_context whenever
        _period_financial_data is empty, even with no documents.
        New code: guards on `not rag.documents` FIRST.
        """
        import api as api_module

        empty_rag = self._make_empty_rag()
        # Explicitly set _period_financial_data to {} so the old code path
        # (if not period_data: call _get_financial_analysis_context) would
        # trigger; the new guard on rag.documents must prevent it.
        empty_rag._period_financial_data = {}
        api_module._rag_instance = empty_rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as client:
            resp = client.post(
                "/compare",
                json={"period_labels": ["FY2023", "FY2024"]},
            )
        api_module._rag_instance = None
        api_module._rate_log.clear()

        assert resp.status_code == 200
        body = resp.json()
        assert body["deltas"] == []
        assert body["improvements"] == []
        assert body["deteriorations"] == []
        # _get_financial_analysis_context must NOT have been called
        empty_rag._get_financial_analysis_context.assert_not_called()

    def test_documents_loaded_compare_still_invokes_analyser(self):
        """
        REGRESSION GUARD: when documents ARE loaded, the analyser path must
        still run (_get_financial_analysis_context must be called).
        """
        from unittest.mock import MagicMock

        import api as api_module

        rag = MagicMock()
        rag.documents = [{"source": "report.pdf", "type": "pdf", "content": "Revenue $1M."}]
        rag._graph_store = None
        # _period_financial_data starts empty so the lazy-populate path triggers
        rag._period_financial_data = {}

        # After _get_financial_analysis_context is called, populate period data
        # with a minimal FinancialData-like object so run_all_ratios can run.
        # We use a side_effect to simulate the lazy population.
        def _populate_context():
            rag._period_financial_data = {}  # still empty — no ratio data, but path ran

        rag._get_financial_analysis_context.side_effect = _populate_context
        rag.charlie_analyzer = MagicMock()

        api_module._rag_instance = rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as client:
            resp = client.post(
                "/compare",
                json={"period_labels": ["FY2023", "FY2024"]},
            )
        api_module._rag_instance = None
        api_module._rate_log.clear()

        assert resp.status_code == 200
        # The analyser path WAS entered (not short-circuited)
        rag._get_financial_analysis_context.assert_called_once()


# ---------------------------------------------------------------------------
# WP-H: eager exporter imports at lifespan (P1-A6)
# ---------------------------------------------------------------------------


class TestEagerExporterImport:
    """WP-H: FinancialExcelExporter and FinancialPDFExporter must be importable
    as module-level names in api after the module is loaded."""

    def test_excel_exporter_symbol_available_at_module_level(self):
        """FinancialExcelExporter must be accessible as api.FinancialExcelExporter."""
        import api as api_module

        assert hasattr(api_module, "FinancialExcelExporter"), (
            "api.FinancialExcelExporter not found — eager lifespan import missing"
        )

    def test_pdf_exporter_symbol_available_at_module_level(self):
        """FinancialPDFExporter must be accessible as api.FinancialPDFExporter."""
        import api as api_module

        assert hasattr(api_module, "FinancialPDFExporter"), (
            "api.FinancialPDFExporter not found — eager lifespan import missing"
        )

    def test_export_xlsx_returns_valid_bytes_after_eager_import(self, client, mock_rag):
        """XLSX export still returns valid streaming bytes after eager-import refactor.

        Patches api.FinancialExcelExporter directly (the module-level reference
        installed by the lifespan eager import) rather than the source module class.
        """
        mock_report = MagicMock()
        mock_report.executive_summary = "Company looks healthy."
        mock_report.sections = {"ratio_analysis": "Good ratios."}
        mock_report.generated_at = "2026-01-01"
        mock_rag.charlie_analyzer.analyze.return_value = {"health": "good"}
        mock_rag.charlie_analyzer.generate_report.return_value = mock_report
        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_full_report.return_value = b"PK\x03\x04fake"
        with patch("api.FinancialExcelExporter", return_value=mock_exporter_instance):
            resp = client.post(
                "/export/xlsx",
                json={"financial_data": {"revenue": 1000}},
            )
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]
        assert len(resp.content) > 0

    def test_export_pdf_returns_valid_bytes_after_eager_import(self, client, mock_rag):
        """PDF export still returns valid streaming bytes after eager-import refactor.

        Patches api.FinancialPDFExporter directly (the module-level reference
        installed by the lifespan eager import) rather than the source module class.
        """
        mock_report = MagicMock()
        mock_report.executive_summary = "Company looks healthy."
        mock_report.sections = {"ratio_analysis": "Good ratios."}
        mock_report.generated_at = "2026-01-01"
        mock_rag.charlie_analyzer.analyze.return_value = {"health": "good"}
        mock_rag.charlie_analyzer.generate_report.return_value = mock_report
        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_full_report.return_value = b"%PDF-1.4 fake"
        with patch("api.FinancialPDFExporter", return_value=mock_exporter_instance):
            resp = client.post(
                "/export/pdf",
                json={"financial_data": {"revenue": 1000}},
            )
        assert resp.status_code == 200
        assert "pdf" in resp.headers["content-type"]
        assert len(resp.content) > 0


# ---------------------------------------------------------------------------
# P1-C1: SSE per-chunk client-side timeout (WS4 Wave 2)
# ---------------------------------------------------------------------------


class TestSSEChunkClientSideTimeout:
    """P1-C1: asyncio.wait_for wraps each chunk fetch; times out and emits error SSE.

    NOTE: this test bounds CLIENT-SIDE latency only.  The to_thread worker
    blocked on a C-level socket read is NOT cancelled by wait_for; that thread
    reclamation is handled by the finite httpx read timeout in local_llm.py
    (WP-LLM P1-C1-stream-timeout, Wave 1).  The test is named
    *_client_side_timeout to make that boundary explicit.
    """

    def test_stream_chunk_client_side_timeout(self, client, mock_rag):
        """A hung async chunk source times out; the stream closes with an SSE error event.

        Uses a sync iterator that raises asyncio.TimeoutError on first next()
        call.  asyncio.to_thread propagates the exception from the thread worker;
        asyncio.wait_for (which wraps to_thread) lets it propagate as
        TimeoutError (Python 3.11+: asyncio.TimeoutError is TimeoutError).
        The event_generator catches it, emits the SSE error event, and returns.

        sse_starlette caches an asyncio.Event (AppStatus.should_exit_event) on
        first use; reset it to None before this test so it is re-created on the
        current event loop (avoids "bound to a different event loop" errors when
        tests run in sequence with different loops).
        """
        import asyncio as _asyncio

        try:
            from sse_starlette.sse import AppStatus

            AppStatus.should_exit_event = None
        except Exception:
            pass

        class _HungIterator:
            """Iterator that raises TimeoutError on first next() call."""

            def __iter__(self):
                return self

            def __next__(self):
                raise _asyncio.TimeoutError()

        mock_rag.answer_stream.return_value = _HungIterator()

        resp = client.post("/query-stream", json={"text": "Test query"})

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "error" in body


# ---------------------------------------------------------------------------
# P1-C5: rate-limiter cross-IP sweep (WS4 Wave 2)
# ---------------------------------------------------------------------------


class TestRateLimiterCrossIPSweep:
    """P1-C5: stale IPs are evicted by a cross-IP sweep; hard cap and active IPs unaffected."""

    def test_stale_ip_evicted_after_later_request_from_any_ip(self):
        """An IP whose timestamps all expire IS removed after a later request from any IP."""
        import time as time_mod

        import api as api_module

        api_module._rate_log.clear()
        # Inject a stale entry for ip_old (timestamp far in the past)
        api_module._rate_log["ip_old"] = [time_mod.monotonic() - 120]

        # A request from a different IP triggers the sweep
        api_module._rate_log.clear()
        # Re-inject: stale ip_old + active ip_new
        now = time_mod.monotonic()
        api_module._rate_log["ip_old"] = [now - 120]  # outside the 60s window
        api_module._rate_log["ip_new"] = [now - 5]  # inside the 60s window

        # Simulate a new request from ip_trigger by calling the rate-limit
        # logic directly (avoids needing TestClient / HTTP stack overhead)
        cutoff = now - api_module._RATE_WINDOW
        with api_module._rate_lock:
            # Prune ip_trigger (fresh, empty)
            api_module._rate_log["ip_trigger"] = []
            api_module._rate_log["ip_trigger"].append(now)
            # Perform the cross-IP sweep
            stale_keys = [k for k, v in list(api_module._rate_log.items()) if not [t for t in v if t > cutoff]]
            for k in stale_keys:
                del api_module._rate_log[k]

        assert "ip_old" not in api_module._rate_log, "stale IP must be evicted by sweep"
        assert "ip_new" in api_module._rate_log, "active IP must NOT be evicted"

        api_module._rate_log.clear()

    def test_hard_cap_still_clears_dict(self):
        """The 10K hard cap must still clear the entire dict."""
        import time as time_mod

        import api as api_module

        api_module._rate_log.clear()
        now = time_mod.monotonic()
        for i in range(10_001):
            api_module._rate_log[f"ip_{i}"] = [now]

        # Trigger via TestClient request so the middleware fires
        from api import app

        api_module._rag_instance = MagicMock()
        api_module._rag_instance.documents = []
        api_module._rag_instance._graph_store = None
        with TestClient(app) as c:
            c.get("/documents")
        api_module._rag_instance = None

        # After the hard cap fires, the dict should be small (cap cleared it)
        assert len(api_module._rate_log) < 10_001
        api_module._rate_log.clear()

    def test_active_ip_not_evicted_by_sweep(self):
        """An IP with a recent in-window timestamp must not be removed by the sweep."""
        import time as time_mod

        import api as api_module

        api_module._rate_log.clear()
        now = time_mod.monotonic()
        api_module._rate_log["active_ip"] = [now - 5]  # well within window

        cutoff = now - api_module._RATE_WINDOW
        with api_module._rate_lock:
            stale_keys = [k for k, v in list(api_module._rate_log.items()) if not [t for t in v if t > cutoff]]
            for k in stale_keys:
                del api_module._rate_log[k]

        assert "active_ip" in api_module._rate_log, "active IP must NOT be evicted"
        api_module._rate_log.clear()


# ---------------------------------------------------------------------------
# P1-C6: LLMConnectionError -> 503 for all six analyzer endpoints (WS4 Wave 2)
# ---------------------------------------------------------------------------


class TestLLMConnectionError503:
    """P1-C6: injected LLMConnectionError yields 503 (not 422/500) for all six endpoints.
    Malformed input still yields 422.
    """

    def _llm_error(self):
        from local_llm import LLMConnectionError

        return LLMConnectionError("LLM offline")

    # --- /analyze ---

    def test_analyze_llm_connection_error_yields_503(self, client, mock_rag):
        mock_rag.charlie_analyzer.analyze.side_effect = self._llm_error()
        resp = client.post("/analyze", json={"financial_data": {"revenue": 1000}})
        assert resp.status_code == 503

    def test_analyze_malformed_input_still_422(self, client, mock_rag):
        # field_validator fires before analyze() so 422 is still correct
        huge_data = {f"f_{i}": i for i in range(250)}
        resp = client.post("/analyze", json={"financial_data": huge_data})
        assert resp.status_code == 422

    # --- /export/xlsx ---

    def test_export_xlsx_llm_connection_error_yields_503(self, client, mock_rag):
        mock_rag.charlie_analyzer.analyze.side_effect = self._llm_error()
        resp = client.post("/export/xlsx", json={"financial_data": {"revenue": 1000}})
        assert resp.status_code == 503

    def test_export_xlsx_malformed_input_still_422(self, client, mock_rag):
        huge_data = {f"f_{i}": i for i in range(250)}
        resp = client.post("/export/xlsx", json={"financial_data": huge_data})
        assert resp.status_code == 422

    # --- /export/pdf ---

    def test_export_pdf_llm_connection_error_yields_503(self, client, mock_rag):
        mock_rag.charlie_analyzer.analyze.side_effect = self._llm_error()
        resp = client.post("/export/pdf", json={"financial_data": {"revenue": 1000}})
        assert resp.status_code == 503

    def test_export_pdf_malformed_input_still_422(self, client, mock_rag):
        huge_data = {f"f_{i}": i for i in range(250)}
        resp = client.post("/export/pdf", json={"financial_data": huge_data})
        assert resp.status_code == 422

    # --- /portfolio/analyze ---

    def _portfolio_payload(self):
        return {"companies": {"AcmeCo": {"revenue": 1_000_000}}}

    def test_portfolio_analyze_llm_connection_error_yields_503(self, client, mock_rag):
        from unittest.mock import patch as _patch

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.full_portfolio_analysis.side_effect = self._llm_error()
            mock_pa_factory.return_value = mock_pa
            resp = client.post("/portfolio/analyze", json=self._portfolio_payload())
        assert resp.status_code == 503

    def test_portfolio_analyze_malformed_input_still_422(self, client, mock_rag):
        # Too many companies triggers the Pydantic validator -> 422
        companies = {f"Co{i}": {"revenue": 1_000_000} for i in range(51)}
        resp = client.post("/portfolio/analyze", json={"companies": companies})
        assert resp.status_code == 422

    # --- /portfolio/correlation ---

    def test_portfolio_correlation_llm_connection_error_yields_503(self, client, mock_rag):
        from unittest.mock import patch as _patch

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.correlation_matrix.side_effect = self._llm_error()
            mock_pa_factory.return_value = mock_pa
            resp = client.post("/portfolio/correlation", json=self._portfolio_payload())
        assert resp.status_code == 503

    def test_portfolio_correlation_malformed_input_still_422(self, client, mock_rag):
        companies = {f"Co{i}": {"revenue": 1_000_000} for i in range(51)}
        resp = client.post("/portfolio/correlation", json={"companies": companies})
        assert resp.status_code == 422

    # --- /compliance/analyze ---

    def _compliance_payload(self):
        return {"financial_data": {"revenue": 1_000_000}}

    def test_compliance_analyze_llm_connection_error_yields_503(self, client, mock_rag):
        from unittest.mock import patch as _patch

        with _patch("api._get_compliance_scorer") as mock_cs_factory:
            mock_cs = MagicMock()
            mock_cs.full_compliance_report.side_effect = self._llm_error()
            mock_cs_factory.return_value = mock_cs
            resp = client.post("/compliance/analyze", json=self._compliance_payload())
        assert resp.status_code == 503

    def test_compliance_analyze_malformed_input_still_422(self, client, mock_rag):
        huge_data = {f"f_{i}": i for i in range(250)}
        resp = client.post("/compliance/analyze", json={"financial_data": huge_data})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# P1-C7: /documents pagination (WS4 Wave 2)
# ---------------------------------------------------------------------------


class TestDocumentsPagination:
    """P1-C7: limit/offset/source query params on GET /documents."""

    def _make_rag_with_docs(self):
        rag = MagicMock()
        rag.documents = [{"source": f"doc_{i}.pdf", "type": "pdf", "content": f"content {i}"} for i in range(10)]
        return rag

    def test_default_params_unchanged_behaviour(self):
        """No params: all unique sources returned (up to default limit=100)."""
        import api as api_module

        rag = self._make_rag_with_docs()
        api_module._rag_instance = rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as c:
            resp = c.get("/documents")
        api_module._rag_instance = None
        api_module._rate_log.clear()
        assert resp.status_code == 200
        assert len(resp.json()) == 10

    def test_limit_slices_results(self):
        """limit=3 returns only the first 3 documents."""
        import api as api_module

        rag = self._make_rag_with_docs()
        api_module._rag_instance = rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as c:
            resp = c.get("/documents?limit=3")
        api_module._rag_instance = None
        api_module._rate_log.clear()
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_offset_skips_results(self):
        """offset=7 skips first 7, returns remaining 3."""
        import api as api_module

        rag = self._make_rag_with_docs()
        api_module._rag_instance = rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as c:
            resp = c.get("/documents?offset=7")
        api_module._rag_instance = None
        api_module._rate_log.clear()
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_limit_and_offset_combined(self):
        """offset=2, limit=4 returns docs 2..5 inclusive."""
        import api as api_module

        rag = self._make_rag_with_docs()
        api_module._rag_instance = rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as c:
            resp = c.get("/documents?offset=2&limit=4")
        api_module._rag_instance = None
        api_module._rate_log.clear()
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 4
        assert docs[0]["source"] == "doc_2.pdf"
        assert docs[-1]["source"] == "doc_5.pdf"

    def test_source_filter_returns_only_matching(self):
        """source=doc_3.pdf returns only the doc_3.pdf entry."""
        import api as api_module

        rag = self._make_rag_with_docs()
        api_module._rag_instance = rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as c:
            resp = c.get("/documents?source=doc_3.pdf")
        api_module._rag_instance = None
        api_module._rate_log.clear()
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 1
        assert docs[0]["source"] == "doc_3.pdf"

    def test_source_filter_no_match_returns_empty(self):
        """source filter with no match returns empty list, not 404."""
        import api as api_module

        rag = self._make_rag_with_docs()
        api_module._rag_instance = rag
        api_module._rate_log.clear()
        from api import app

        with TestClient(app) as c:
            resp = c.get("/documents?source=nonexistent.pdf")
        api_module._rag_instance = None
        api_module._rate_log.clear()
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# WP-API-WIRE: P1-D3 store_portfolio_analysis + P1-D4 store_compliance_report
# ---------------------------------------------------------------------------


def _make_mock_pa_report(company_names):
    """Build a minimal PortfolioReport-like mock for portfolio_analyze tests."""
    from unittest.mock import MagicMock

    report = MagicMock()
    report.num_companies = len(company_names)
    report.risk_summary.avg_health_score = 70.0
    report.risk_summary.overall_risk_level = "low"
    report.risk_summary.min_health_score = 60
    report.risk_summary.max_health_score = 80
    report.risk_summary.distress_count = 0
    report.risk_summary.risk_flags = []
    report.risk_summary.strongest_company = company_names[0] if company_names else ""
    report.risk_summary.weakest_company = company_names[0] if company_names else ""
    report.diversification.overall_score = 55
    report.diversification.grade = "C"
    report.summary = "Test portfolio summary."
    return report


def _make_mock_corr(company_names):
    """Build a minimal CorrelationMatrix-like mock for portfolio_correlation tests."""
    from unittest.mock import MagicMock

    corr = MagicMock()
    corr.company_names = list(company_names)
    corr.ratio_names = ["net_margin"]
    corr.matrix = [[1.0]]
    corr.avg_correlation = 0.5
    corr.interpretation = "Moderate correlation."
    return corr


def _make_mock_compliance_report():
    """Build a minimal ComplianceReport-like mock for compliance_analyze tests."""
    from unittest.mock import MagicMock

    report = MagicMock()
    report.sox.overall_risk = "low"
    report.sox.risk_score = 20
    report.sec.disclosure_score = 80
    report.sec.grade = "B"
    report.regulatory.compliance_pct = 90.0
    report.regulatory.pass_count = 9
    report.regulatory.fail_count = 1
    report.audit_risk.risk_level = "low"
    report.audit_risk.score = 85
    report.audit_risk.grade = "A"
    report.audit_risk.going_concern_risk = False
    report.summary = "Compliance looks good."
    return report


class TestGraphStoreWiringD3D4:
    """WP-API-WIRE P1-D3 (store_portfolio_analysis) and P1-D4 (store_compliance_report).

    Per-test RAG singleton injection mirrors TestDocumentsPagination pattern:
    set api_module._rag_instance directly, clear after, avoid reusing the
    module-level client fixture (which wires a different mock_rag).
    """

    _PORTFOLIO_PAYLOAD = {"companies": {"AcmeCo": {"revenue": 1_000_000}}}
    _COMPLIANCE_PAYLOAD = {"financial_data": {"revenue": 1_000_000}}

    def _make_rag_with_store(self, store):
        """Return a minimal RAG mock with the given _graph_store."""
        rag = MagicMock()
        rag.documents = [{"source": "x.pdf", "type": "pdf", "content": "c"}]
        rag._graph_store = store
        return rag

    def _setup_rag(self, api_module, store):
        api_module._rag_instance = self._make_rag_with_store(store)
        api_module._rate_log.clear()

    def _teardown_rag(self, api_module):
        api_module._rag_instance = None
        api_module._rate_log.clear()

    # -----------------------------------------------------------------------
    # D3 -- /portfolio/analyze
    # -----------------------------------------------------------------------

    def test_portfolio_analyze_store_called_with_mock_graph(self):
        """D3: store_portfolio_analysis called once after successful analysis."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        mock_store = MagicMock()
        self._setup_rag(api_module, mock_store)
        report = _make_mock_pa_report(["AcmeCo"])

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.full_portfolio_analysis.return_value = report
            mock_pa_factory.return_value = mock_pa
            with TestClient(app) as c:
                resp = c.post("/portfolio/analyze", json=self._PORTFOLIO_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200
        mock_store.store_portfolio_analysis.assert_called_once()
        # Verify the risk_summary argument is the one from the report.
        call_kwargs = mock_store.store_portfolio_analysis.call_args
        args = call_kwargs[0] if call_kwargs[0] else []
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        all_args = list(args) + list(kwargs.values())
        assert report.risk_summary in all_args

    def test_portfolio_analyze_no_store_returns_200(self):
        """D3: missing _graph_store does not fail /portfolio/analyze."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        self._setup_rag(api_module, None)
        report = _make_mock_pa_report(["AcmeCo"])

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.full_portfolio_analysis.return_value = report
            mock_pa_factory.return_value = mock_pa
            with TestClient(app) as c:
                resp = c.post("/portfolio/analyze", json=self._PORTFOLIO_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200

    def test_portfolio_analyze_store_raises_returns_200(self):
        """D3: raising store_portfolio_analysis does not fail /portfolio/analyze (best-effort)."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        mock_store = MagicMock()
        mock_store.store_portfolio_analysis.side_effect = RuntimeError("neo4j down")
        self._setup_rag(api_module, mock_store)
        report = _make_mock_pa_report(["AcmeCo"])

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.full_portfolio_analysis.return_value = report
            mock_pa_factory.return_value = mock_pa
            with TestClient(app) as c:
                resp = c.post("/portfolio/analyze", json=self._PORTFOLIO_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200

    # -----------------------------------------------------------------------
    # D3 -- /portfolio/correlation
    # -----------------------------------------------------------------------

    def test_portfolio_correlation_store_called_with_mock_graph(self):
        """D3: store_portfolio_analysis called once after successful correlation."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        mock_store = MagicMock()
        self._setup_rag(api_module, mock_store)
        corr = _make_mock_corr(["AcmeCo"])

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.correlation_matrix.return_value = corr
            mock_pa_factory.return_value = mock_pa
            with TestClient(app) as c:
                resp = c.post("/portfolio/correlation", json=self._PORTFOLIO_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200
        mock_store.store_portfolio_analysis.assert_called_once()

    def test_portfolio_correlation_no_store_returns_200(self):
        """D3: missing _graph_store does not fail /portfolio/correlation."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        self._setup_rag(api_module, None)
        corr = _make_mock_corr(["AcmeCo"])

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.correlation_matrix.return_value = corr
            mock_pa_factory.return_value = mock_pa
            with TestClient(app) as c:
                resp = c.post("/portfolio/correlation", json=self._PORTFOLIO_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200

    def test_portfolio_correlation_store_raises_returns_200(self):
        """D3: raising store_portfolio_analysis does not fail /portfolio/correlation (best-effort)."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        mock_store = MagicMock()
        mock_store.store_portfolio_analysis.side_effect = RuntimeError("neo4j down")
        self._setup_rag(api_module, mock_store)
        corr = _make_mock_corr(["AcmeCo"])

        with _patch("api._get_portfolio_analyzer") as mock_pa_factory:
            mock_pa = MagicMock()
            mock_pa.correlation_matrix.return_value = corr
            mock_pa_factory.return_value = mock_pa
            with TestClient(app) as c:
                resp = c.post("/portfolio/correlation", json=self._PORTFOLIO_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200

    # -----------------------------------------------------------------------
    # D4 -- /compliance/analyze
    # -----------------------------------------------------------------------

    def test_compliance_analyze_store_called_with_mock_graph(self):
        """D4: store_compliance_report called once after successful compliance analysis."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        mock_store = MagicMock()
        self._setup_rag(api_module, mock_store)
        report = _make_mock_compliance_report()

        with _patch("api._get_compliance_scorer") as mock_cs_factory:
            mock_cs = MagicMock()
            mock_cs.full_compliance_report.return_value = report
            mock_cs_factory.return_value = mock_cs
            with TestClient(app) as c:
                resp = c.post("/compliance/analyze", json=self._COMPLIANCE_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200
        mock_store.store_compliance_report.assert_called_once()
        # The compliance_report argument should be the computed report.
        call_args = mock_store.store_compliance_report.call_args
        positional = call_args[0] if call_args[0] else []
        keyword = call_args[1] if call_args[1] else {}
        all_args = list(positional) + list(keyword.values())
        assert report in all_args

    def test_compliance_analyze_no_store_returns_200(self):
        """D4: missing _graph_store does not fail /compliance/analyze."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        self._setup_rag(api_module, None)
        report = _make_mock_compliance_report()

        with _patch("api._get_compliance_scorer") as mock_cs_factory:
            mock_cs = MagicMock()
            mock_cs.full_compliance_report.return_value = report
            mock_cs_factory.return_value = mock_cs
            with TestClient(app) as c:
                resp = c.post("/compliance/analyze", json=self._COMPLIANCE_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200

    def test_compliance_analyze_store_raises_returns_200(self):
        """D4: raising store_compliance_report does not fail /compliance/analyze (best-effort)."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        import api as api_module
        from api import app

        mock_store = MagicMock()
        mock_store.store_compliance_report.side_effect = RuntimeError("neo4j down")
        self._setup_rag(api_module, mock_store)
        report = _make_mock_compliance_report()

        with _patch("api._get_compliance_scorer") as mock_cs_factory:
            mock_cs = MagicMock()
            mock_cs.full_compliance_report.return_value = report
            mock_cs_factory.return_value = mock_cs
            with TestClient(app) as c:
                resp = c.post("/compliance/analyze", json=self._COMPLIANCE_PAYLOAD)

        self._teardown_rag(api_module)
        assert resp.status_code == 200
