"""Tests for config.py centralized settings."""

import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_chunk_size_default(self):
        from config import Settings

        s = Settings()
        assert s.chunk_size == 500

    def test_chunk_overlap_default(self):
        from config import Settings

        s = Settings()
        assert s.chunk_overlap == 50

    def test_top_k_default(self):
        from config import Settings

        s = Settings()
        assert s.top_k == 3

    def test_max_top_k_default(self):
        from config import Settings

        s = Settings()
        assert s.max_top_k == 20

    def test_bm25_plus_semantic_equals_one(self):
        from config import Settings

        s = Settings()
        assert abs(s.bm25_weight + s.semantic_weight - 1.0) < 1e-9

    def test_max_file_size_positive(self):
        from config import Settings

        s = Settings()
        assert s.max_file_size_mb > 0

    def test_llm_model_default(self):
        from config import Settings

        s = Settings()
        assert s.llm_model == "llama3.2"

    def test_embedding_model_default(self):
        from config import Settings

        s = Settings()
        assert s.embedding_model == "mxbai-embed-large"

    def test_embedding_dimension_default(self):
        from config import Settings

        s = Settings()
        assert s.embedding_dimension == 1024

    def test_default_tax_rate(self):
        from config import Settings

        s = Settings()
        assert 0 < s.default_tax_rate < 1

    def test_api_port_default(self):
        from config import Settings

        s = Settings()
        assert s.api_port == 8504

    def test_circuit_breaker_defaults(self):
        from config import Settings

        s = Settings()
        assert s.circuit_breaker_failure_threshold >= 1
        assert s.circuit_breaker_recovery_seconds >= 1

    def test_llm_cache_maxsize(self):
        from config import Settings

        s = Settings()
        assert s.llm_cache_maxsize > 0

    def test_max_request_body_bytes(self):
        from config import Settings

        s = Settings()
        assert s.max_request_body_bytes > 0

    def test_max_financial_fields(self):
        from config import Settings

        s = Settings()
        assert s.max_financial_fields > 0

    def test_export_max_ratios(self):
        from config import Settings

        s = Settings()
        assert s.export_max_ratios > 0


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_env_prefix_is_rag(self):
        from config import Settings

        assert Settings.model_config["env_prefix"] == "RAG_"

    def test_override_chunk_size(self):
        from config import Settings

        with patch.dict(os.environ, {"RAG_CHUNK_SIZE": "1000"}):
            s = Settings()
            assert s.chunk_size == 1000

    def test_override_top_k(self):
        from config import Settings

        with patch.dict(os.environ, {"RAG_TOP_K": "10"}):
            s = Settings()
            assert s.top_k == 10

    def test_override_llm_model(self):
        from config import Settings

        with patch.dict(os.environ, {"RAG_LLM_MODEL": "mistral"}):
            s = Settings()
            assert s.llm_model == "mistral"

    def test_override_embedding_dimension(self):
        from config import Settings

        with patch.dict(os.environ, {"RAG_EMBEDDING_DIMENSION": "384"}):
            s = Settings()
            assert s.embedding_dimension == 384

    def test_override_api_port(self):
        from config import Settings

        with patch.dict(os.environ, {"RAG_API_PORT": "9000"}):
            s = Settings()
            assert s.api_port == 9000


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_settings_importable(self):
        from config import settings

        assert settings is not None

    def test_settings_is_settings_instance(self):
        from config import Settings, settings

        assert isinstance(settings, Settings)

    def test_singleton_has_expected_defaults(self):
        from config import settings

        # Verify key defaults are populated with sensible values
        assert settings.embedding_dimension > 0
        assert settings.llm_cache_maxsize > 0
        assert settings.api_port > 0
