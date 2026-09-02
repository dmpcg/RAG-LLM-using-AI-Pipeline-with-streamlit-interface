"""
E2E tests for Phase 7 API endpoints: Portfolio Analysis + Regulatory Compliance.

Tests the full request->response cycle through FastAPI's TestClient,
exercising serialization, Pydantic validation, business logic, and
response structure. Each test is independent and deterministic.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test data factories (following E2E pattern: realistic data, deterministic)
# ---------------------------------------------------------------------------


def _strong_company_data():
    """Realistic strong company financial data dict."""
    return {
        "revenue": 10_000_000,
        "net_income": 1_500_000,
        "gross_profit": 6_000_000,
        "operating_income": 2_500_000,
        "ebit": 2_500_000,
        "ebitda": 3_000_000,
        "interest_expense": 200_000,
        "cogs": 4_000_000,
        "total_assets": 20_000_000,
        "current_assets": 8_000_000,
        "current_liabilities": 3_000_000,
        "total_debt": 4_000_000,
        "total_equity": 14_000_000,
        "total_liabilities": 6_000_000,
        "operating_cash_flow": 2_000_000,
        "cash": 3_000_000,
        "accounts_receivable": 2_000_000,
        "inventory": 1_500_000,
        "accounts_payable": 1_000_000,
        "depreciation": 500_000,
        "capex": 800_000,
    }


def _weak_company_data():
    """Realistic weak/distressed company financial data dict."""
    return {
        "revenue": 2_000_000,
        "net_income": -500_000,
        "gross_profit": 200_000,
        "operating_income": -300_000,
        "ebit": -300_000,
        "ebitda": -100_000,
        "interest_expense": 400_000,
        "cogs": 1_800_000,
        "total_assets": 3_000_000,
        "current_assets": 500_000,
        "current_liabilities": 2_000_000,
        "total_debt": 3_500_000,
        "total_equity": -500_000,
        "total_liabilities": 3_500_000,
        "operating_cash_flow": -100_000,
        "cash": 100_000,
        "accounts_receivable": 1_000_000,
    }


def _medium_company_data():
    """Realistic average company financial data dict."""
    return {
        "revenue": 5_000_000,
        "net_income": 250_000,
        "gross_profit": 2_000_000,
        "operating_income": 500_000,
        "ebit": 500_000,
        "ebitda": 800_000,
        "interest_expense": 100_000,
        "total_assets": 10_000_000,
        "current_assets": 4_000_000,
        "current_liabilities": 2_500_000,
        "total_debt": 3_000_000,
        "total_equity": 6_000_000,
        "total_liabilities": 4_000_000,
        "operating_cash_flow": 600_000,
        "cash": 1_000_000,
    }


def _portfolio_payload():
    """Multi-company portfolio request payload."""
    return {
        "companies": {
            "StrongCorp": _strong_company_data(),
            "WeakCo": _weak_company_data(),
            "MediumInc": _medium_company_data(),
        }
    }


# ---------------------------------------------------------------------------
# Fixtures (independent per test, clean teardown)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_rag():
    """Minimal mock RAG to satisfy api module initialization."""
    rag = MagicMock()
    rag.documents = []
    rag.llm = MagicMock()
    rag.llm.circuit_state = "CLOSED"
    rag.charlie_analyzer = MagicMock()
    return rag


@pytest.fixture()
def client(mock_rag):
    """TestClient with RAG singleton patched out."""
    import api as api_module

    api_module._rag_instance = mock_rag
    api_module._rate_log.clear()
    from api import app

    with TestClient(app) as c:
        yield c
    api_module._rag_instance = None
    api_module._rate_log.clear()


# ---------------------------------------------------------------------------
# POST /portfolio/analyze
# ---------------------------------------------------------------------------


class TestPortfolioAnalyze:
    """Test the full portfolio analysis endpoint."""

    def test_three_companies_returns_200(self, client):
        resp = client.post("/portfolio/analyze", json=_portfolio_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_companies"] == 3

    def test_response_has_required_fields(self, client):
        resp = client.post("/portfolio/analyze", json=_portfolio_payload())
        body = resp.json()
        required = [
            "num_companies",
            "avg_health_score",
            "diversification_score",
            "diversification_grade",
            "risk_level",
            "risk_flags",
            "strongest",
            "weakest",
            "summary",
        ]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_health_score_range(self, client):
        resp = client.post("/portfolio/analyze", json=_portfolio_payload())
        body = resp.json()
        assert 0 <= body["avg_health_score"] <= 100

    def test_diversification_score_range(self, client):
        resp = client.post("/portfolio/analyze", json=_portfolio_payload())
        body = resp.json()
        assert 0 <= body["diversification_score"] <= 100
        assert body["diversification_grade"] in ("A", "B", "C", "D", "F")

    def test_risk_level_valid(self, client):
        resp = client.post("/portfolio/analyze", json=_portfolio_payload())
        body = resp.json()
        assert body["risk_level"] in ("low", "moderate", "high", "critical")

    def test_summary_nonempty(self, client):
        resp = client.post("/portfolio/analyze", json=_portfolio_payload())
        body = resp.json()
        assert len(body["summary"]) > 20

    def test_single_company_works(self, client):
        payload = {"companies": {"Only": _strong_company_data()}}
        resp = client.post("/portfolio/analyze", json=payload)
        assert resp.status_code == 200
        assert resp.json()["num_companies"] == 1

    def test_unknown_fields_ignored(self, client):
        """Unknown FinancialData fields should be silently dropped."""
        data = _strong_company_data()
        data["unknown_field"] = 999
        payload = {"companies": {"A": data}}
        resp = client.post("/portfolio/analyze", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /portfolio/correlation
# ---------------------------------------------------------------------------


class TestPortfolioCorrelation:
    """Test the correlation matrix endpoint."""

    def test_correlation_returns_200(self, client):
        resp = client.post("/portfolio/correlation", json=_portfolio_payload())
        assert resp.status_code == 200

    def test_matrix_dimensions(self, client):
        resp = client.post("/portfolio/correlation", json=_portfolio_payload())
        body = resp.json()
        assert len(body["company_names"]) == 3
        assert len(body["matrix"]) == 3
        assert len(body["matrix"][0]) == 3

    def test_diagonal_is_one(self, client):
        resp = client.post("/portfolio/correlation", json=_portfolio_payload())
        body = resp.json()
        for i in range(len(body["matrix"])):
            assert abs(body["matrix"][i][i] - 1.0) < 0.01

    def test_avg_correlation_range(self, client):
        resp = client.post("/portfolio/correlation", json=_portfolio_payload())
        body = resp.json()
        assert -1.0 <= body["avg_correlation"] <= 1.0

    def test_interpretation_nonempty(self, client):
        resp = client.post("/portfolio/correlation", json=_portfolio_payload())
        body = resp.json()
        assert len(body["interpretation"]) > 10


# ---------------------------------------------------------------------------
# POST /compliance/analyze
# ---------------------------------------------------------------------------


class TestComplianceAnalyze:
    """Test the full compliance analysis endpoint."""

    def test_compliant_company_200(self, client):
        resp = client.post(
            "/compliance/analyze",
            json={"financial_data": _strong_company_data()},
        )
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post(
            "/compliance/analyze",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        required = [
            "sox_risk",
            "sox_score",
            "sec_score",
            "sec_grade",
            "regulatory_pct",
            "regulatory_pass",
            "regulatory_fail",
            "audit_risk",
            "audit_score",
            "audit_grade",
            "going_concern",
            "summary",
        ]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_compliant_not_going_concern(self, client):
        resp = client.post(
            "/compliance/analyze",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        assert body["going_concern"] is False

    def test_noncompliant_flagged(self, client):
        resp = client.post(
            "/compliance/analyze",
            json={"financial_data": _weak_company_data()},
        )
        body = resp.json()
        assert body["going_concern"] is True
        assert body["audit_risk"] in ("high", "critical")

    def test_scores_in_range(self, client):
        resp = client.post(
            "/compliance/analyze",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        assert 0 <= body["sox_score"] <= 100
        assert 0 <= body["sec_score"] <= 100
        assert 0 <= body["regulatory_pct"] <= 100
        assert 0 <= body["audit_score"] <= 100

    def test_summary_contains_sox_sec_regulatory(self, client):
        resp = client.post(
            "/compliance/analyze",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        assert "SOX" in body["summary"]
        assert "SEC" in body["summary"]


# ---------------------------------------------------------------------------
# POST /compliance/sox
# ---------------------------------------------------------------------------


class TestComplianceSOX:
    """Test the SOX-only compliance endpoint."""

    def test_sox_returns_200(self, client):
        resp = client.post(
            "/compliance/sox",
            json={"financial_data": _strong_company_data()},
        )
        assert resp.status_code == 200

    def test_sox_response_fields(self, client):
        resp = client.post(
            "/compliance/sox",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        assert "overall_risk" in body
        assert "risk_score" in body
        assert "flags" in body
        assert "checks_performed" in body
        assert "checks_passed" in body

    def test_sox_weak_company_has_flags(self, client):
        resp = client.post(
            "/compliance/sox",
            json={"financial_data": _weak_company_data()},
        )
        body = resp.json()
        assert len(body["flags"]) > 0
        assert body["overall_risk"] in ("moderate", "high")


# ---------------------------------------------------------------------------
# POST /compliance/regulatory
# ---------------------------------------------------------------------------


class TestComplianceRegulatory:
    """Test the regulatory threshold endpoint."""

    def test_regulatory_returns_200(self, client):
        resp = client.post(
            "/compliance/regulatory",
            json={"financial_data": _strong_company_data()},
        )
        assert resp.status_code == 200

    def test_regulatory_response_fields(self, client):
        resp = client.post(
            "/compliance/regulatory",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        assert "pass_count" in body
        assert "fail_count" in body
        assert "compliance_pct" in body
        assert "thresholds" in body

    def test_six_thresholds_checked(self, client):
        resp = client.post(
            "/compliance/regulatory",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        assert len(body["thresholds"]) == 6

    def test_each_threshold_has_fields(self, client):
        resp = client.post(
            "/compliance/regulatory",
            json={"financial_data": _strong_company_data()},
        )
        body = resp.json()
        for t in body["thresholds"]:
            assert "rule_name" in t
            assert "framework" in t
            assert "passes" in t
            assert "severity" in t

    def test_weak_company_has_critical_failures(self, client):
        resp = client.post(
            "/compliance/regulatory",
            json={"financial_data": _weak_company_data()},
        )
        body = resp.json()
        assert len(body["critical_failures"]) > 0
        assert body["fail_count"] > 0


# ---------------------------------------------------------------------------
# Error-path tests for portfolio endpoints
# ---------------------------------------------------------------------------


class TestPortfolioErrorPaths:
    """Validate that portfolio endpoints reject bad input gracefully."""

    def test_portfolio_analyze_empty_companies_rejected(self, client):
        """Empty companies dict should be rejected by Pydantic (min_length=1)."""
        resp = client.post("/portfolio/analyze", json={"companies": {}})
        assert resp.status_code == 422

    def test_portfolio_correlation_empty_companies_rejected(self, client):
        """Empty companies dict should be rejected by Pydantic (min_length=1)."""
        resp = client.post("/portfolio/correlation", json={"companies": {}})
        assert resp.status_code == 422

    def test_portfolio_analyze_missing_body(self, client):
        """Missing request body should return 422."""
        resp = client.post("/portfolio/analyze")
        assert resp.status_code == 422

    def test_portfolio_correlation_missing_body(self, client):
        """Missing request body should return 422."""
        resp = client.post("/portfolio/correlation")
        assert resp.status_code == 422

    def test_portfolio_analyze_single_company_ok(self, client):
        """Single company should still work (min_length=1)."""
        resp = client.post(
            "/portfolio/analyze",
            json={
                "companies": {"Solo": _strong_company_data()},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["num_companies"] == 1


# ---------------------------------------------------------------------------
# Error-path tests for compliance endpoints
# ---------------------------------------------------------------------------


class TestComplianceErrorPaths:
    """Validate that compliance endpoints reject bad input gracefully."""

    def test_compliance_analyze_missing_body(self, client):
        """Missing request body should return 422."""
        resp = client.post("/compliance/analyze")
        assert resp.status_code == 422

    def test_compliance_sox_missing_body(self, client):
        """Missing request body should return 422."""
        resp = client.post("/compliance/sox")
        assert resp.status_code == 422

    def test_compliance_regulatory_missing_body(self, client):
        """Missing request body should return 422."""
        resp = client.post("/compliance/regulatory")
        assert resp.status_code == 422

    def test_compliance_analyze_empty_data_ok(self, client):
        """Empty financial data should still work (graceful defaults)."""
        resp = client.post("/compliance/analyze", json={"financial_data": {}})
        assert resp.status_code == 200
        body = resp.json()
        assert "sox_risk" in body
        assert "going_concern" in body

    def test_compliance_sox_empty_data_ok(self, client):
        """SOX check with empty data should return valid structure."""
        resp = client.post("/compliance/sox", json={"financial_data": {}})
        assert resp.status_code == 200
        body = resp.json()
        assert "overall_risk" in body
        assert "flags" in body

    def test_compliance_regulatory_empty_data_ok(self, client):
        """Regulatory check with empty data should return valid structure."""
        resp = client.post("/compliance/regulatory", json={"financial_data": {}})
        assert resp.status_code == 200
        body = resp.json()
        assert "pass_count" in body
        assert "fail_count" in body
