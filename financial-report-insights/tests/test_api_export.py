"""Tests for export API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_rag():
    rag = MagicMock()
    rag.documents = [{"source": "test.pdf", "type": "pdf", "content": "Test"}]
    rag.llm = MagicMock()
    rag.llm.circuit_state = "CLOSED"
    analyzer = MagicMock()
    analyzer.analyze.return_value = {"current_ratio": 2.0, "net_margin": 0.15}
    report = MagicMock()
    report.executive_summary = "Strong Q4."
    report.sections = {"Overview": "Good"}
    report.generated_at = "2026-02-20"
    analyzer.generate_report.return_value = report
    rag.charlie_analyzer = analyzer
    return rag


@pytest.fixture()
def client(mock_rag):
    import api as api_module

    api_module._rag_instance = mock_rag
    from api import app

    with TestClient(app) as c:
        yield c
    api_module._rag_instance = None


class TestExportXLSX:
    def test_export_xlsx_returns_xlsx(self, client):
        resp = client.post(
            "/export/xlsx",
            json={
                "financial_data": {
                    "revenue": 1000000,
                    "net_income": 150000,
                    "total_assets": 5000000,
                    "total_equity": 3000000,
                }
            },
        )
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]
        assert resp.content[:2] == b"PK"

    def test_export_xlsx_with_company_name(self, client):
        resp = client.post(
            "/export/xlsx",
            json={
                "financial_data": {"revenue": 500000},
                "company_name": "Test Corp",
            },
        )
        assert resp.status_code == 200


class TestExportPDF:
    def test_export_pdf_returns_pdf(self, client):
        resp = client.post(
            "/export/pdf",
            json={
                "financial_data": {
                    "revenue": 1000000,
                    "net_income": 150000,
                }
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content[:5] == b"%PDF-"

    def test_export_pdf_empty_data(self, client):
        resp = client.post("/export/pdf", json={"financial_data": {}})
        assert resp.status_code == 200
