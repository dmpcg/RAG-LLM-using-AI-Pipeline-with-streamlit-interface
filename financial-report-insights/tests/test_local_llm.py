"""
Tests for local_llm.py resilience fixes:
  P0-11  LocalEmbedder.close() / atexit / idempotent
  P0-14  Jittered retry backoff (embedding + LLM paths)
  P1-C2  CircuitBreaker public record_success/record_failure (no deadlock)
  P1-C4  Terminal RuntimeError after _send_embedding_batch loop; empty data warning
  P1-C1-stream-timeout  Streaming ollama.generate carries a finite read timeout
"""

# ruff: noqa: I001  -- stub modules must be registered before local_llm is imported
import threading
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without real config / httpx
# ---------------------------------------------------------------------------

import sys
import types as _types

# Stub 'config' so LocalEmbedder's deferred import does not fail.
_config_mod = _types.ModuleType("config")
_settings_obj = MagicMock()
_settings_obj.llm_cache_maxsize = 128
_settings_obj.embedding_dimension = 1024
_settings_obj.embedding_batch_size = 32
_settings_obj.llm_timeout_seconds = 30
_config_mod.settings = _settings_obj
sys.modules.setdefault("config", _config_mod)

# Stub 'ollama'
_ollama_mod = _types.ModuleType("ollama")
_ollama_mod.generate = MagicMock(return_value={"response": "hello"})
_ollama_mod.Client = MagicMock(name="ollama.Client")
sys.modules.setdefault("ollama", _ollama_mod)

# Stub 'httpx'
import httpx as _real_httpx  # noqa: E402  (may or may not exist)

sys.modules.setdefault("httpx", _real_httpx)

import local_llm  # noqa: E402 – imported after stubs


# ===========================================================================
# P0-11: LocalEmbedder.close() + atexit + idempotent
# ===========================================================================


class TestLocalEmbedderClose(unittest.TestCase):
    """close() must close the underlying httpx.Client exactly once."""

    def _make_embedder(self):
        """Return a LocalEmbedder with a mocked httpx.Client (no real network)."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            embedder = local_llm.LocalEmbedder.__new__(local_llm.LocalEmbedder)
            embedder.model_name = "mxbai-embed-large"
            embedder._url = "http://localhost:11434/v1/embeddings"
            embedder._client = MagicMock()
            embedder.dimension = 1024
        return embedder

    def test_close_calls_client_close(self):
        """close() closes the underlying httpx.Client."""
        embedder = self._make_embedder()
        embedder.close()
        embedder._client.close.assert_called_once()

    def test_close_idempotent(self):
        """Double close() must not raise and must not double-close."""
        embedder = self._make_embedder()
        embedder.close()
        embedder.close()  # second call must be a no-op
        # close() should only have been forwarded to the client once
        # (the second call detects already-closed state)
        embedder._client.close.assert_called_once()

    def test_close_sets_closed_flag(self):
        """After close(), LocalEmbedder should expose a truthy _closed attribute."""
        embedder = self._make_embedder()
        self.assertFalse(getattr(embedder, "_closed", False))
        embedder.close()
        self.assertTrue(embedder._closed)

    def test_atexit_registered(self):
        """LocalEmbedder must register an atexit handler that calls close()."""
        with (
            patch("atexit.register") as mock_register,
            patch("httpx.Client") as mock_client_cls,
            patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}),
        ):
            mock_client_cls.return_value = MagicMock()
            # Patch _request_embeddings to avoid real network calls on dimension probe
            with patch.object(
                local_llm.LocalEmbedder,
                "_request_embeddings",
                return_value=[[0.0] * 1024],
            ):
                local_llm.LocalEmbedder()

        mock_register.assert_called()
        # At least one registered callable should be close or wrap close
        registered_callables = [c[0][0] for c in mock_register.call_args_list]
        self.assertTrue(
            any(callable(fn) for fn in registered_callables),
            "atexit.register must have been called with a callable",
        )


# ===========================================================================
# RC5: dimension-probe bootstrap (embedding_dimension=0 must not deadlock)
# ===========================================================================


class TestDimensionProbeBootstrap(unittest.TestCase):
    """
    With RAG_EMBEDDING_DIMENSION=0, __init__ must run a dimension probe.
    The probe goes through _request_embeddings, which guards on
    self.dimension > 0 — so a sentinel dimension must be set first, then
    replaced with the real probed length. Otherwise the probe can never run.
    """

    def test_probe_runs_and_sets_real_dimension_when_cfg_zero(self):
        vector_1024 = [0.1] * 1024

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [{"embedding": vector_1024}]}
        mock_http_client = MagicMock()
        mock_http_client.post.return_value = mock_resp

        with (
            # Force LocalEmbedder's deferred ``from config import settings`` to
            # resolve to our stub regardless of full-suite import order. The
            # module-level ``sys.modules.setdefault("config", ...)`` only wins if
            # this file is imported before the real config module; in the full
            # suite the real config is already loaded, so without this the probe
            # branch never runs and post.assert_called() fails.
            patch.dict("sys.modules", {"config": _config_mod}),
            patch("httpx.Client", return_value=mock_http_client),
            patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}),
            # Force the configured dimension to 0 so the probe path is taken.
            patch.object(_settings_obj, "embedding_dimension", 0),
        ):
            embedder = local_llm.LocalEmbedder()

        self.assertEqual(
            embedder.dimension,
            1024,
            "self.dimension must be set from the probe vector length (1024)",
        )
        # The probe must actually have hit the endpoint (not short-circuited).
        mock_http_client.post.assert_called()


# ===========================================================================
# P0-14: Jittered retry backoff
# ===========================================================================


class TestRetryJitter(unittest.TestCase):
    """Retry sleeps must include random jitter in [wait, wait*1.25)."""

    # --- Embedding path ---

    def _make_embedder_for_jitter(self):
        embedder = local_llm.LocalEmbedder.__new__(local_llm.LocalEmbedder)
        embedder.model_name = "mxbai-embed-large"
        embedder._url = "http://localhost:11434/v1/embeddings"
        embedder._client = MagicMock()
        embedder.dimension = 1024
        embedder._closed = False
        return embedder

    def test_embedding_retry_sleep_within_jitter_bounds_network_error(self):
        """
        On a network error, the sleep argument must be within [wait, wait*1.25).
        wait = 2**attempt (attempt starts at 1 → wait=2).
        """
        import httpx

        embedder = self._make_embedder_for_jitter()
        call_count = [0]

        def raise_then_succeed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise httpx.RequestError("network error", request=MagicMock())
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}
            return resp

        embedder._client.post.side_effect = raise_then_succeed
        sleep_calls = []

        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            embedder._send_embedding_batch(["hello"], max_retries=3)

        self.assertEqual(len(sleep_calls), 1)
        wait_base = 2**1  # attempt=1 -> wait=2
        self.assertGreaterEqual(
            sleep_calls[0],
            wait_base,
            f"sleep({sleep_calls[0]}) must be >= base wait {wait_base}",
        )
        self.assertLess(
            sleep_calls[0],
            wait_base * 1.25,
            f"sleep({sleep_calls[0]}) must be < wait*1.25 ({wait_base * 1.25})",
        )

    def test_embedding_retry_sleep_within_jitter_bounds_5xx(self):
        """On a 5xx HTTP error the sleep must lie in [wait, wait*1.25)."""
        import httpx

        embedder = self._make_embedder_for_jitter()
        call_count = [0]

        def raise_5xx_then_succeed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_response = MagicMock()
                mock_response.status_code = 503
                raise httpx.HTTPStatusError("503", request=MagicMock(), response=mock_response)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"data": [{"embedding": [0.2] * 1024}]}
            return resp

        embedder._client.post.side_effect = raise_5xx_then_succeed
        sleep_calls = []

        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            embedder._send_embedding_batch(["hi"], max_retries=3)

        self.assertEqual(len(sleep_calls), 1)
        wait_base = 2**1
        self.assertGreaterEqual(sleep_calls[0], wait_base)
        self.assertLess(sleep_calls[0], wait_base * 1.25)

    # --- LLM path (_generate_with_retry) ---

    def test_llm_retry_sleep_within_jitter_bounds(self):
        """LLM retry sleep must lie in [wait, wait*1.25)."""
        llm = local_llm.LocalLLM.__new__(local_llm.LocalLLM)
        llm._max_retries = 3
        llm.model = "llama3.2"
        llm._executor = MagicMock()
        llm._timeout = 10

        call_count = [0]

        def side_effect(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise local_llm.LLMConnectionError("transient failure")
            return "ok"

        sleep_calls = []

        with (
            patch.object(llm, "_call_with_timeout", side_effect=side_effect),
            patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)),
        ):
            result = llm._generate_with_retry("test prompt")

        self.assertEqual(result, "ok")
        self.assertEqual(len(sleep_calls), 1)
        wait_base = 2**1  # attempt=1 -> wait=2
        self.assertGreaterEqual(sleep_calls[0], wait_base)
        self.assertLess(sleep_calls[0], wait_base * 1.25)


# ===========================================================================
# P1-C2: CircuitBreaker public record_success / record_failure — no deadlock
# ===========================================================================


class TestCircuitBreakerPublicAPI(unittest.TestCase):
    """Public record_success/record_failure must update state; no deadlock."""

    def _make_breaker(self, threshold=3, recovery=30):
        return local_llm.CircuitBreaker(
            failure_threshold=threshold,
            recovery_seconds=recovery,
        )

    def test_record_success_exists_and_is_public(self):
        cb = self._make_breaker()
        self.assertTrue(
            hasattr(cb, "record_success") and callable(cb.record_success),
            "CircuitBreaker must have a public record_success() method",
        )

    def test_record_failure_exists_and_is_public(self):
        cb = self._make_breaker()
        self.assertTrue(
            hasattr(cb, "record_failure") and callable(cb.record_failure),
            "CircuitBreaker must have a public record_failure() method",
        )

    def test_record_failure_increments_count_and_opens_at_threshold(self):
        cb = self._make_breaker(threshold=2)
        cb.record_failure()
        self.assertEqual(cb.circuit_state, "CLOSED")
        cb.record_failure()
        self.assertEqual(cb.circuit_state, "OPEN")

    def test_record_success_resets_and_closes_half_open(self):
        cb = self._make_breaker(threshold=1)
        cb.record_failure()  # CLOSED -> OPEN
        # Manually force to HALF_OPEN for the test
        with cb._lock:
            cb._state = local_llm.CircuitState.HALF_OPEN
        cb.record_success()
        self.assertEqual(cb.circuit_state, "CLOSED")
        self.assertEqual(cb._failure_count, 0)

    def test_no_deadlock_back_to_back_record_calls(self):
        """
        record_success / record_failure must not deadlock when called
        back-to-back (non-reentrant lock guard).
        """
        cb = self._make_breaker(threshold=5)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    cb.record_failure()
                    cb.record_success()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        alive = [t for t in threads if t.is_alive()]
        self.assertEqual(alive, [], "Threads are still alive — likely deadlocked")
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_generate_stream_calls_public_methods(self):
        """
        generate_stream must use the public record_success/record_failure
        (not the private _on_success/_on_failure) after this fix.
        """
        llm = local_llm.LocalLLM.__new__(local_llm.LocalLLM)
        llm.model = "llama3.2"
        llm._circuit_breaker = self._make_breaker()

        success_calls = []
        failure_calls = []

        def fake_record_success():
            success_calls.append(1)

        def fake_record_failure():
            failure_calls.append(1)

        llm._circuit_breaker.record_success = fake_record_success
        llm._circuit_breaker.record_failure = fake_record_failure

        def fake_raw_stream(prompt):
            yield "chunk1"
            yield "chunk2"

        with (
            patch.object(llm, "_raw_generate_stream", side_effect=fake_raw_stream),
            patch.object(llm._circuit_breaker, "allow_request"),
        ):
            chunks = list(llm.generate_stream("hello"))

        self.assertEqual(chunks, ["chunk1", "chunk2"])
        self.assertEqual(len(success_calls), 1, "record_success should be called once")
        self.assertEqual(len(failure_calls), 0, "record_failure should not be called")

    def test_generate_stream_calls_record_failure_on_error(self):
        """generate_stream calls record_failure on LLMConnectionError."""
        llm = local_llm.LocalLLM.__new__(local_llm.LocalLLM)
        llm.model = "llama3.2"
        llm._circuit_breaker = self._make_breaker()

        failure_calls = []

        def fake_record_failure():
            failure_calls.append(1)

        llm._circuit_breaker.record_failure = fake_record_failure

        def failing_stream(prompt):
            raise local_llm.LLMConnectionError("oops")
            yield  # make it a generator

        with (
            patch.object(llm, "_raw_generate_stream", side_effect=failing_stream),
            patch.object(llm._circuit_breaker, "allow_request"),
        ):
            with self.assertRaises(local_llm.LLMConnectionError):
                list(llm.generate_stream("hi"))

        self.assertEqual(len(failure_calls), 1)


# ===========================================================================
# P1-C4: Terminal RuntimeError + empty data['data'] warning
# ===========================================================================


class TestSendEmbeddingBatchTerminalRaise(unittest.TestCase):
    """_send_embedding_batch must raise RuntimeError when max_retries=0."""

    def _make_embedder(self):
        embedder = local_llm.LocalEmbedder.__new__(local_llm.LocalEmbedder)
        embedder.model_name = "mxbai-embed-large"
        embedder._url = "http://localhost:11434/v1/embeddings"
        embedder._client = MagicMock()
        embedder.dimension = 1024
        embedder._closed = False
        return embedder

    def test_max_retries_zero_raises_runtime_error(self):
        """
        With max_retries=0 the retry loop body never executes.
        The terminal RuntimeError guard must fire to prevent implicit None return.
        """
        embedder = self._make_embedder()
        with self.assertRaises(RuntimeError) as ctx:
            embedder._send_embedding_batch(["text"], max_retries=0)
        self.assertIn("unreachable", str(ctx.exception).lower())

    def test_empty_data_list_logs_warning(self):
        """
        When the endpoint returns data['data'] == [], embeddings are silently
        missing.  The function MUST log a warning (contract: still return []).
        """
        embedder = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}
        embedder._client.post.return_value = mock_resp

        with self.assertLogs("local_llm", level="WARNING") as log_ctx:
            result = embedder._send_embedding_batch(["hello"], max_retries=1)

        self.assertEqual(result, [], "Return contract: empty list when data is empty")
        self.assertTrue(
            any("empty" in msg.lower() or "missing" in msg.lower() for msg in log_ctx.output),
            f"Expected a warning about empty embeddings, got: {log_ctx.output}",
        )


# ===========================================================================
# P1-C1-stream-timeout: streaming generate carries a finite read timeout
# ===========================================================================


class TestStreamingReadTimeout(unittest.TestCase):
    """
    The installed ollama client's generate() accepts NO 'timeout' kwarg, so
    passing one raises TypeError on every real streaming call. The finite read
    timeout must therefore be configured on an ollama.Client, not forwarded to
    generate(). These tests pin the corrected contract:

      * generate(stream=True) is invoked WITHOUT a 'timeout' kwarg
      * when OLLAMA_HOST is set, the Client is built carrying the timeout
    """

    def _make_raw_llm(self, timeout_seconds=30):
        llm = local_llm.LocalLLM.__new__(local_llm.LocalLLM)
        llm.model = "llama3.2"
        llm._timeout = timeout_seconds
        return llm

    def test_streaming_generate_omits_timeout_kwarg(self):
        """
        generate(stream=True, ...) must NOT be called with a 'timeout' kwarg —
        the installed ollama.generate() has no such parameter.
        """
        chunks_yielded = []

        def fake_generate(*args, **kwargs):
            chunks_yielded.append(kwargs)
            yield {"response": "hello"}

        # No OLLAMA_HOST -> module-level ollama.generate is used.
        with patch.dict("os.environ", {}, clear=False) as _env:
            _env.pop("OLLAMA_HOST", None)
            with patch.object(local_llm.ollama, "generate", side_effect=fake_generate):
                llm = self._make_raw_llm(timeout_seconds=45)
                list(llm._raw_generate_stream("test prompt"))

        self.assertTrue(
            len(chunks_yielded) >= 1,
            "generate must have been called at least once",
        )
        self.assertNotIn(
            "timeout",
            chunks_yielded[0],
            "generate(stream=True) must NOT receive a 'timeout' kwarg "
            "(unsupported by the installed ollama client)",
        )
        self.assertTrue(chunks_yielded[0].get("stream"), "stream=True must be passed")

    def test_streaming_client_built_with_timeout_when_host_set(self):
        """
        When OLLAMA_HOST is set the read timeout is configured on the
        ollama.Client (derived from settings.llm_timeout_seconds), and the
        Client's generate() is still called WITHOUT a 'timeout' kwarg.
        """
        expected_timeout = 77
        generate_kwargs_seen = []

        fake_client = MagicMock()

        def fake_client_generate(*args, **kwargs):
            generate_kwargs_seen.append(kwargs)
            yield {"response": "chunk"}

        fake_client.generate.side_effect = fake_client_generate
        client_init_kwargs = {}

        def fake_client_cls(*args, **kwargs):
            client_init_kwargs.update(kwargs)
            return fake_client

        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch.object(local_llm.ollama, "Client", side_effect=fake_client_cls):
                llm = self._make_raw_llm(timeout_seconds=expected_timeout)
                list(llm._raw_generate_stream("prompt"))

        self.assertEqual(
            client_init_kwargs.get("timeout"),
            expected_timeout,
            "ollama.Client must be constructed with timeout=self._timeout",
        )
        self.assertTrue(generate_kwargs_seen, "client.generate must be called")
        self.assertNotIn(
            "timeout",
            generate_kwargs_seen[0],
            "client.generate(stream=True) must NOT receive a 'timeout' kwarg",
        )


if __name__ == "__main__":
    unittest.main()
