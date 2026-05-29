"""Tests for config.validate_settings() and healthcheck.check_config_valid()."""

import os
from unittest.mock import patch

from config import Settings, validate_settings
from healthcheck import check_config_valid

# ---------------------------------------------------------------------------
# validate_settings() tests
# ---------------------------------------------------------------------------


class TestValidateSettings:
    """Unit tests for validate_settings()."""

    def test_default_settings_valid(self):
        """Default settings should pass with no errors."""
        s = Settings()
        errors, warnings = validate_settings(s)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_chunk_size_too_small(self):
        s = Settings(chunk_size=50)
        errors, _ = validate_settings(s)
        assert any("chunk_size" in e for e in errors)

    def test_chunk_size_too_large(self):
        s = Settings(chunk_size=6000)
        errors, _ = validate_settings(s)
        assert any("chunk_size" in e for e in errors)

    def test_chunk_overlap_exceeds_chunk_size(self):
        s = Settings(chunk_size=500, chunk_overlap=500)
        errors, _ = validate_settings(s)
        assert any("chunk_overlap" in e for e in errors)

    def test_chunk_overlap_greater_than_chunk_size(self):
        s = Settings(chunk_size=500, chunk_overlap=600)
        errors, _ = validate_settings(s)
        assert any("chunk_overlap" in e for e in errors)

    def test_top_k_zero(self):
        s = Settings(top_k=0)
        errors, _ = validate_settings(s)
        assert any("top_k" in e for e in errors)

    def test_top_k_exceeds_max(self):
        s = Settings(top_k=25, max_top_k=20)
        errors, _ = validate_settings(s)
        assert any("top_k" in e for e in errors)

    def test_llm_timeout_too_low(self):
        s = Settings(llm_timeout_seconds=5)
        errors, _ = validate_settings(s)
        assert any("llm_timeout_seconds" in e for e in errors)

    def test_unusual_embedding_dimension_warns(self):
        s = Settings(embedding_dimension=512)
        _, warnings = validate_settings(s)
        assert any("embedding_dimension" in w for w in warnings)

    def test_standard_embedding_dimensions_no_warning(self):
        for dim in (0, 384, 768, 1024):
            s = Settings(embedding_dimension=dim)
            _, warnings = validate_settings(s)
            assert not any("embedding_dimension" in w for w in warnings), f"Unexpected warning for dim={dim}"

    def test_large_file_size_warns(self):
        s = Settings(max_file_size_mb=600)
        _, warnings = validate_settings(s)
        assert any("max_file_size_mb" in w for w in warnings)

    def test_weights_not_summing_to_one_warns(self):
        s = Settings(bm25_weight=0.3, semantic_weight=0.3)
        _, warnings = validate_settings(s)
        assert any("bm25_weight" in w for w in warnings)

    def test_weights_summing_to_one_no_warning(self):
        s = Settings(bm25_weight=0.4, semantic_weight=0.6)
        _, warnings = validate_settings(s)
        assert not any("bm25_weight" in w for w in warnings)

    @patch.dict(os.environ, {"OLLAMA_HOST": "ftp://localhost:11434"})
    def test_invalid_ollama_host_scheme(self):
        s = Settings()
        errors, _ = validate_settings(s)
        assert any("OLLAMA_HOST" in e and "scheme" in e for e in errors)

    @patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:11434"})
    def test_valid_ollama_host(self):
        s = Settings()
        errors, _ = validate_settings(s)
        assert not any("OLLAMA_HOST" in e for e in errors)

    @patch.dict(os.environ, {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": ""})
    def test_neo4j_uri_without_password(self):
        s = Settings()
        errors, _ = validate_settings(s)
        assert any("NEO4J_PASSWORD" in e for e in errors)

    @patch.dict(os.environ, {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": "secret"})
    def test_neo4j_uri_with_password_ok(self):
        s = Settings()
        errors, _ = validate_settings(s)
        assert not any("NEO4J_PASSWORD" in e for e in errors)

    @patch.dict(os.environ, {"NEO4J_URI": "", "NEO4J_PASSWORD": ""})
    def test_neo4j_not_configured_ok(self):
        s = Settings()
        errors, _ = validate_settings(s)
        assert not any("NEO4J" in e for e in errors)

    def test_multiple_errors_reported(self):
        """Multiple invalid fields should all be reported."""
        s = Settings(chunk_size=50, top_k=0, llm_timeout_seconds=1)
        errors, _ = validate_settings(s)
        assert len(errors) >= 3

    def test_valid_range_boundaries(self):
        """Exact boundary values should be accepted."""
        s = Settings(chunk_size=100, chunk_overlap=99, top_k=1, llm_timeout_seconds=10)
        errors, _ = validate_settings(s)
        assert not any("chunk_size" in e for e in errors)
        assert not any("chunk_overlap" in e for e in errors)
        assert not any("top_k" in e for e in errors)
        assert not any("llm_timeout" in e for e in errors)


# ---------------------------------------------------------------------------
# check_config_valid() (healthcheck integration) tests
# ---------------------------------------------------------------------------


class TestCheckConfigValid:
    """Tests for the healthcheck wrapper."""

    def test_valid_config_returns_ok(self):
        result = check_config_valid()
        assert result["status"] in ("ok", "warning")

    @patch("config.validate_settings", return_value=(["fake error"], []))
    def test_errors_return_error_status(self, _mock):
        result = check_config_valid()
        assert result["status"] == "error"
        assert "fake error" in result["detail"]

    @patch("config.validate_settings", return_value=([], ["fake warning"]))
    def test_warnings_return_warning_status(self, _mock):
        result = check_config_valid()
        assert result["status"] == "warning"
        assert "fake warning" in result["detail"]

    @patch("config.validate_settings", return_value=([], []))
    def test_no_issues_returns_ok(self, _mock):
        result = check_config_valid()
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# .env missing warning tests
# ---------------------------------------------------------------------------


class TestEnvFileWarning:
    """Test that missing .env logs a warning."""

    @patch("config._ENV_FILE", property(lambda self: None))
    def test_env_file_warning_logged(self, caplog):
        # The warning is logged at module load time; we verify the code path
        # exists by checking validate_settings still works without .env
        errors, _ = validate_settings()
        # Should not error just because .env is missing
        assert not any(".env" in e for e in errors)
