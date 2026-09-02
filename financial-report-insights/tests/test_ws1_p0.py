"""WS-1 P0 regression tests (2026-05-07).

Coverage:
    P0-2  : healthcheck stage timeout + result cache
    P0-3  : Neo4jTransientError raised by graph_store on transient failure
    P0-4  : docker-compose.yml has kernel hardening directives
    P0-12 : logging_config redacts JWT/AWS/bolt/OpenAI/Anthropic secrets
    P0-15 : pr-review.yml no longer shells out a PR-supplied filename
    P0-16 : config rejects '*' CORS origin when allow_credentials=True
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# P0-2: healthcheck timeout + cache
# ---------------------------------------------------------------------------


class TestHealthcheckTimeout:
    def test_stage_timeout_returns_error_within_budget(self):
        import healthcheck as hc

        def slow_stage():
            time.sleep(5.0)  # Far longer than 3s budget
            return {"status": "ok", "detail": "should not arrive"}

        start = time.monotonic()
        result = hc._run_with_timeout(slow_stage, "slow_test", timeout=0.5)
        elapsed = time.monotonic() - start

        assert result["status"] == "error"
        assert "timed out" in result["detail"].lower()
        # Tight ceiling: must return well under 2s even though stage sleeps 5s
        assert elapsed < 2.0, f"timeout enforcement leaked: {elapsed:.2f}s"

    def test_get_health_status_caches_results(self):
        import healthcheck as hc

        hc._reset_health_cache()

        call_counter = {"n": 0}

        def fake_preflight():
            call_counter["n"] += 1
            return [{"status": "ok", "detail": "fake", "check": "fake"}]

        with patch.object(hc, "run_preflight_checks", side_effect=fake_preflight):
            first = hc.get_health_status()
            second = hc.get_health_status()
            third = hc.get_health_status()

        # Cache TTL is 10s — three rapid calls should hit only once.
        assert call_counter["n"] == 1
        assert first == second == third

    def test_get_health_status_cache_can_be_bypassed(self):
        import healthcheck as hc

        hc._reset_health_cache()

        with patch.object(hc, "run_preflight_checks") as mock:
            mock.return_value = [{"status": "ok", "detail": "x", "check": "x"}]
            hc.get_health_status(use_cache=False)
            hc.get_health_status(use_cache=False)
            assert mock.call_count == 2


# ---------------------------------------------------------------------------
# P0-3: Neo4jTransientError raised on transient failure
# ---------------------------------------------------------------------------


class TestNeo4jTransientError:
    def test_typed_exception_class_exists(self):
        from graph_store import Neo4jTransientError

        assert issubclass(Neo4jTransientError, RuntimeError)
        # Must carry the operation name and original cause
        original = ConnectionError("boom")
        err = Neo4jTransientError("store_chunks", original)
        assert err.operation == "store_chunks"
        assert err.original is original
        assert "store_chunks" in str(err)

    def test_store_chunks_raises_on_transient_failure(self):
        from graph_store import Neo4jStore, Neo4jTransientError

        # Build a Store with a fake driver that raises ConnectionError on
        # session().run().  ConnectionError is in _NEO4J_TRANSIENT.
        fake_session = MagicMock()
        fake_session.run.side_effect = ConnectionError("Neo4j unreachable")
        fake_session.__enter__ = MagicMock(return_value=fake_session)
        fake_session.__exit__ = MagicMock(return_value=False)

        fake_driver = MagicMock()
        fake_driver.session.return_value = fake_session

        store = Neo4jStore(fake_driver)
        with pytest.raises(Neo4jTransientError) as ei:
            store.store_chunks(
                chunks=[{"content": "hi", "source": "doc"}],
                embeddings=[[0.0] * 4],
                doc_id="doc1",
            )
        assert ei.value.operation == "store_chunks"
        assert isinstance(ei.value.original, ConnectionError)

    def test_store_credit_assessment_raises_on_transient_failure(self):
        from graph_store import Neo4jStore, Neo4jTransientError

        fake_session = MagicMock()
        fake_session.run.side_effect = OSError("link reset")
        fake_session.__enter__ = MagicMock(return_value=fake_session)
        fake_session.__exit__ = MagicMock(return_value=False)
        fake_driver = MagicMock()
        fake_driver.session.return_value = fake_session

        scorecard = MagicMock(
            grade="B",
            total_score=72,
            recommendation="approve",
            category_scores={"liquidity": 20},
            strengths=["cash"],
            weaknesses=[],
        )
        debt_capacity = MagicMock(max_additional_debt=1_000_000.0, current_leverage=2.5)

        store = Neo4jStore(fake_driver)
        with pytest.raises(Neo4jTransientError):
            store.store_credit_assessment("Acme", scorecard, debt_capacity)


# ---------------------------------------------------------------------------
# P0-4: compose hardening
# ---------------------------------------------------------------------------


class TestComposeHardening:
    @pytest.fixture
    def compose_text(self) -> str:
        path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
        return path.read_text(encoding="utf-8")

    def test_rag_app_has_no_new_privileges(self, compose_text):
        # Must appear in both rag-app and neo4j blocks
        assert compose_text.count("no-new-privileges:true") >= 2

    def test_rag_app_drops_all_capabilities(self, compose_text):
        assert compose_text.count("cap_drop:") >= 2
        assert "- ALL" in compose_text

    def test_rag_app_root_fs_is_readonly(self, compose_text):
        assert "read_only: true" in compose_text

    def test_rag_app_has_tmpfs_for_tmp(self, compose_text):
        assert "/tmp:size=" in compose_text

    def test_rag_app_has_pids_limit(self, compose_text):
        # Modern compose puts pids under deploy.resources.limits
        assert "pids: 512" in compose_text
        assert "pids: 1024" in compose_text  # neo4j

    def test_rag_app_has_ulimits(self, compose_text):
        assert "ulimits:" in compose_text
        assert "nofile:" in compose_text


# ---------------------------------------------------------------------------
# P0-12: extended logging redaction
# ---------------------------------------------------------------------------


class TestExtendedRedaction:
    def test_redacts_bearer_jwt(self):
        from logging_config import _redact

        msg = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payloadpart.signaturepart"
        out = _redact(msg)
        assert "eyJhbGciOiJIUzI1NiJ9" not in out
        assert "***REDACTED***" in out

    def test_redacts_aws_access_key(self):
        from logging_config import _redact

        out = _redact("found key AKIAIOSFODNN7EXAMPLE in config")
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "AWS_ACCESS_KEY_REDACTED" in out

    def test_redacts_aws_secret_key(self):
        from logging_config import _redact

        out = _redact("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in out
        assert "***REDACTED***" in out

    def test_redacts_bolt_url_password(self):
        from logging_config import _redact

        out = _redact("connecting to bolt://neo4j:s3cret_pw@db.example.com:7687/x")
        assert "s3cret_pw" not in out
        assert "***REDACTED***" in out
        # Hostname must still be visible for diagnostics
        assert "db.example.com" in out

    def test_redacts_openai_api_key_with_assignment(self):
        from logging_config import _redact

        # When the key sits behind an assignment that already matches the
        # generic api_key= pattern, the generic pattern still wipes it.
        out = _redact("OPENAI_API_KEY=sk-proj-abcdef0123456789abcdef0123456789")
        assert "sk-proj-abcdef0123456789abcdef0123456789" not in out
        assert "***REDACTED***" in out

    def test_redacts_openai_api_key_inline(self):
        from logging_config import _redact

        # Free-form mention of an OpenAI key (no assignment) still must redact.
        out = _redact("calling LLM with sk-proj-abcdef0123456789abcdef0123456789 done")
        assert "sk-proj-abcdef0123456789abcdef0123456789" not in out
        assert "sk-***REDACTED***" in out

    def test_redacts_anthropic_api_key(self):
        from logging_config import _redact

        out = _redact("export ANTHROPIC=sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaa")
        assert "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaa" not in out
        assert "sk-ant-***REDACTED***" in out

    def test_normal_text_is_untouched(self):
        from logging_config import _redact

        text = "Revenue grew 12% to $4.2B; net margin 18%"
        assert _redact(text) == text


# ---------------------------------------------------------------------------
# P0-15: pr-review.yml no longer uses execSync on PR filenames
# ---------------------------------------------------------------------------


class TestPrReviewWorkflow:
    @pytest.fixture
    def workflow_text(self) -> str:
        # Repo root is two levels above tests/ (financial-report-insights/tests/..)
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / ".github" / "workflows" / "pr-review.yml"
        return path.read_text(encoding="utf-8")

    def test_no_execsync_on_filename(self, workflow_text):
        # The original injection vector was execSync(`wc -l < "${f.filename}"`).
        # No execSync at all in the workflow now (a comment may still
        # reference `wc -l` to document the migration).
        assert "execSync" not in workflow_text
        assert "child_process" not in workflow_text
        assert "wc -l < " not in workflow_text

    def test_uses_fs_readfilesync(self, workflow_text):
        assert "fs.readFileSync" in workflow_text
        assert "split('\\n').length" in workflow_text or 'split("\\n").length' in workflow_text

    def test_validates_filename_path(self, workflow_text):
        # Path-traversal guard
        assert "..'" in workflow_text or "'..'" in workflow_text or "includes('..')" in workflow_text


# ---------------------------------------------------------------------------
# P0-16: CORS validator rejects '*' with allow_credentials=True
# ---------------------------------------------------------------------------


class TestCorsValidator:
    def test_wildcard_with_credentials_is_error(self):
        from config import Settings, validate_settings

        s = Settings(cors_origins="*", cors_allow_credentials=True)
        errors, _warnings = validate_settings(s)
        assert any("cors_origins" in e and "credentials" in e.lower() for e in errors), (
            f"Expected CORS wildcard+credentials error, got: {errors}"
        )

    def test_wildcard_without_credentials_is_allowed(self):
        from config import Settings, validate_settings

        s = Settings(cors_origins="*", cors_allow_credentials=False)
        errors, _warnings = validate_settings(s)
        assert not any("cors_origins" in e for e in errors)

    def test_explicit_origin_with_credentials_is_allowed(self):
        from config import Settings, validate_settings

        s = Settings(
            cors_origins="https://app.example.com,https://other.example.com",
            cors_allow_credentials=True,
        )
        errors, _warnings = validate_settings(s)
        assert not any("cors_origins" in e for e in errors)
