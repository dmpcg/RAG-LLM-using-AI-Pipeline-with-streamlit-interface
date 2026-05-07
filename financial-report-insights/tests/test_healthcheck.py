"""Tests for healthcheck.py startup validation utilities."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_health_cache():
    """WS-1 P0-2: ``get_health_status`` caches results for 10s.  Reset
    the cache between tests so each test sees its own mocked stages."""
    import healthcheck
    healthcheck._reset_health_cache()
    yield
    healthcheck._reset_health_cache()


# ---------------------------------------------------------------------------
# check_ollama_connection
# ---------------------------------------------------------------------------


class TestCheckOllamaConnection:
    def test_ok_without_host(self):
        from healthcheck import check_ollama_connection

        with patch("ollama.list", return_value={"models": []}):
            result = check_ollama_connection()
        assert result["status"] == "ok"

    def test_ok_with_custom_host(self):
        from healthcheck import check_ollama_connection

        mock_client = MagicMock()
        with patch("ollama.Client", return_value=mock_client):
            result = check_ollama_connection(host="http://localhost:11434")
        assert result["status"] == "ok"
        mock_client.list.assert_called_once()

    def test_error_on_connection_failure(self):
        from healthcheck import check_ollama_connection

        with patch("ollama.list", side_effect=ConnectionError("refused")):
            result = check_ollama_connection()
        assert result["status"] == "error"
        assert "connectivity check failed" in result["detail"]


# ---------------------------------------------------------------------------
# check_model_available
# ---------------------------------------------------------------------------


class TestCheckModelAvailable:
    def test_model_found(self):
        from healthcheck import check_model_available

        with patch("ollama.list", return_value={
            "models": [{"name": "llama3.2:latest"}]
        }):
            result = check_model_available("llama3.2")
        assert result["status"] == "ok"
        assert "available" in result["detail"]

    def test_model_not_found(self):
        from healthcheck import check_model_available

        with patch("ollama.list", return_value={
            "models": [{"name": "mistral:latest"}]
        }):
            result = check_model_available("llama3.2")
        assert result["status"] == "warning"
        assert "not found" in result["detail"]

    def test_error_on_failure(self):
        from healthcheck import check_model_available

        with patch("ollama.list", side_effect=Exception("timeout")):
            result = check_model_available("llama3.2")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# check_documents_folder
# ---------------------------------------------------------------------------


class TestCheckDocumentsFolder:
    def test_existing_writable_folder(self, tmp_path):
        from healthcheck import check_documents_folder

        result = check_documents_folder(str(tmp_path))
        assert result["status"] == "ok"
        assert "files" in result["detail"]

    def test_creates_missing_folder(self, tmp_path):
        from healthcheck import check_documents_folder

        new_folder = tmp_path / "new_docs"
        result = check_documents_folder(str(new_folder))
        assert result["status"] == "ok"
        assert new_folder.exists()

    def test_non_writable_folder(self, tmp_path):
        from healthcheck import check_documents_folder

        # Simulate non-writable by patching os.access
        with patch("healthcheck.os.access", return_value=False):
            result = check_documents_folder(str(tmp_path))
        assert result["status"] == "error"
        assert "not writable" in result["detail"]


# ---------------------------------------------------------------------------
# check_neo4j_connection
# ---------------------------------------------------------------------------


class TestCheckNeo4jConnection:
    def test_not_configured(self):
        from healthcheck import check_neo4j_connection

        with patch.dict("os.environ", {"NEO4J_URI": ""}):
            result = check_neo4j_connection()
        assert result["status"] == "ok"
        assert "not configured" in result["detail"]

    def test_connection_success(self):
        from healthcheck import check_neo4j_connection

        mock_store = MagicMock()
        with (
            patch.dict("os.environ", {"NEO4J_URI": "bolt://localhost:7687"}),
            patch("graph_store.Neo4jStore") as MockStore,
        ):
            MockStore.connect.return_value = mock_store
            result = check_neo4j_connection()
        assert result["status"] == "ok"
        assert "reachable" in result["detail"]
        mock_store.close.assert_called_once()

    def test_connection_failure(self):
        from healthcheck import check_neo4j_connection

        with (
            patch.dict("os.environ", {"NEO4J_URI": "bolt://localhost:7687"}),
            patch("graph_store.Neo4jStore") as MockStore,
        ):
            MockStore.connect.return_value = None
            result = check_neo4j_connection()
        assert result["status"] == "warning"

    def test_connection_exception(self):
        from healthcheck import check_neo4j_connection

        with (
            patch.dict("os.environ", {"NEO4J_URI": "bolt://localhost:7687"}),
            patch("graph_store.Neo4jStore") as MockStore,
        ):
            MockStore.connect.side_effect = Exception("driver error")
            result = check_neo4j_connection()
        assert result["status"] == "warning"
        # Must NOT leak exception details
        assert "driver error" not in result["detail"]


# ---------------------------------------------------------------------------
# check_cache_folders
# ---------------------------------------------------------------------------


class TestCheckCacheFolders:
    def test_cache_folders_created(self, tmp_path):
        from healthcheck import check_cache_folders

        with patch("config.settings") as mock_settings:
            mock_settings.embedding_cache_dir = str(tmp_path / "emb_cache")
            mock_settings.llm_cache_dir = str(tmp_path / "llm_cache")
            result = check_cache_folders()
        assert result["status"] == "ok"

    def test_cache_folder_creation_error(self):
        from healthcheck import check_cache_folders

        with patch("config.settings") as mock_settings:
            # Use paths that trigger OSError (permission denied via mock)
            mock_settings.embedding_cache_dir = "/fake/emb"
            mock_settings.llm_cache_dir = "/fake/llm"
            with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
                result = check_cache_folders()
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# run_preflight_checks & get_health_status
# ---------------------------------------------------------------------------


class TestRunPreflightChecks:
    def test_returns_list_of_check_results(self):
        from healthcheck import run_preflight_checks

        with (
            patch("healthcheck.check_config_valid", return_value={"status": "ok", "detail": "ok"}),
            patch("healthcheck.check_ollama_connection", return_value={"status": "ok", "detail": "ok"}),
            patch("healthcheck.check_model_available", return_value={"status": "ok", "detail": "ok"}),
            patch("healthcheck.check_documents_folder", return_value={"status": "ok", "detail": "ok"}),
            patch("healthcheck.check_cache_folders", return_value={"status": "ok", "detail": "ok"}),
            patch("healthcheck.check_neo4j_connection", return_value={"status": "ok", "detail": "ok"}),
        ):
            results = run_preflight_checks()
        assert len(results) == 6
        assert all(r["status"] == "ok" for r in results)
        assert all("check" in r for r in results)


class TestGetHealthStatus:
    def test_healthy(self):
        from healthcheck import get_health_status

        with patch("healthcheck.run_preflight_checks", return_value=[
            {"status": "ok", "detail": "all good", "check": "test"}
        ]):
            status = get_health_status()
        assert status["healthy"] is True
        assert status["status"] == "healthy"

    def test_degraded(self):
        from healthcheck import get_health_status

        with patch("healthcheck.run_preflight_checks", return_value=[
            {"status": "ok", "detail": "fine", "check": "a"},
            {"status": "warning", "detail": "model missing", "check": "b"},
        ]):
            status = get_health_status()
        assert status["healthy"] is True
        assert status["status"] == "degraded"

    def test_unhealthy(self):
        from healthcheck import get_health_status

        with patch("healthcheck.run_preflight_checks", return_value=[
            {"status": "error", "detail": "cannot connect", "check": "a"},
        ]):
            status = get_health_status()
        assert status["healthy"] is False
        assert status["status"] == "unhealthy"
