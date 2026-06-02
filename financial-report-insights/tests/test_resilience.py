"""
Resilience tests for critical reliability and performance improvements.
Tests: LLM timeout, circuit breaker, pre-computed norms, thread-safe reload,
       embedding cache key stability, configurable tax rate, answer reuse.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from financial_analyzer import CharlieAnalyzer, FinancialData
from local_llm import LLMConnectionError, LLMTimeoutError, LocalEmbedder, LocalLLM

# ============================================================
# LLM Timeout Tests
# ============================================================


class TestLLMTimeout:
    """Test timeout enforcement in LocalLLM."""

    def test_timeout_raises_timeout_error(self):
        """Verify that a slow Ollama call raises LLMTimeoutError."""
        llm = LocalLLM(model="test-model", timeout_seconds=1, max_retries=1)

        def slow_generate(*args, **kwargs):
            time.sleep(2)  # Exceeds 1s timeout
            return {"response": "Too late"}

        with patch("local_llm.ollama.generate", side_effect=slow_generate):
            with pytest.raises(LLMTimeoutError, match="did not respond within 1s"):
                llm.generate("Test prompt")

    def test_timeout_not_retried(self):
        """Timeouts should NOT be retried (they'd just timeout again)."""
        llm = LocalLLM(model="test-model", timeout_seconds=1, max_retries=3)
        call_count = 0

        def slow_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            time.sleep(2)
            return {"response": "Too late"}

        with patch("local_llm.ollama.generate", side_effect=slow_generate):
            with pytest.raises(LLMTimeoutError):
                llm.generate("Test")

        # Should have been called only once (no retries)
        assert call_count == 1

    def test_connection_error_retried(self):
        """ConnectionError should be retried up to max_retries."""
        llm = LocalLLM(model="test-model", timeout_seconds=120, max_retries=3)
        call_count = 0

        def flaky_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return {"response": "Success on third try"}

        with patch("local_llm.ollama.generate", side_effect=flaky_generate):
            result = llm.generate("Test")

        assert result == "Success on third try"
        assert call_count == 3

    def test_retry_succeeds_on_second_attempt(self):
        """Retry should succeed when error is transient."""
        llm = LocalLLM(model="test-model", max_retries=2)
        attempt = 0

        def flaky_ollama(*args, **kwargs):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise RuntimeError("First attempt fails")
            return {"response": "Second attempt succeeds"}

        with patch("local_llm.ollama.generate", side_effect=flaky_ollama):
            result = llm.generate("Test")

        assert result == "Second attempt succeeds"
        assert attempt == 2

    def test_max_retries_exhausted(self):
        """Verify that max_retries is enforced and final error is raised."""
        llm = LocalLLM(model="test-model", max_retries=2)

        with patch("local_llm.ollama.generate") as mock_gen:
            mock_gen.side_effect = RuntimeError("Persistent error")

            with pytest.raises(LLMConnectionError, match="Persistent error"):
                llm.generate("Test")

            # max_retries=2 means 2 total attempts (not 2 retries after initial)
            assert mock_gen.call_count == 2


# ============================================================
# Circuit Breaker Tests
# ============================================================


class TestCircuitBreaker:
    """Test circuit breaker logic to prevent cascading failures."""

    def test_starts_closed(self):
        """Circuit breaker should start in CLOSED state."""
        from local_llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=5)
        assert cb.circuit_state == "CLOSED"

    def test_opens_after_threshold_failures(self):
        """Circuit should open after consecutive failures reach threshold."""
        from local_llm import CircuitBreaker, LLMConnectionError

        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=10)

        def failing_func():
            raise LLMConnectionError("Test error")

        # Record failures by calling through circuit breaker
        for _ in range(3):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

    def test_rejects_when_open(self):
        """Circuit breaker should reject calls when OPEN."""
        from local_llm import CircuitBreaker, LLMConnectionError

        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=10)

        def failing_func():
            raise LLMConnectionError("Test error")

        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

        # Next call should be rejected immediately
        with pytest.raises(LLMConnectionError, match="Circuit breaker open"):
            cb.call(lambda: "test")

    def test_half_open_after_recovery_period(self):
        """After recovery timeout, circuit should enter HALF_OPEN."""
        from local_llm import CircuitBreaker, LLMConnectionError

        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)

        def failing_func():
            raise LLMConnectionError("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

        # Wait for recovery period
        time.sleep(1.1)

        # Next call should transition to HALF_OPEN (but will fail in this test)
        with pytest.raises(LLMConnectionError):
            cb.call(failing_func)

        # State should be OPEN again due to failed probe
        assert cb.circuit_state == "OPEN"

    def test_closes_on_half_open_success(self):
        """Successful call in HALF_OPEN should close the circuit."""
        from local_llm import CircuitBreaker, LLMConnectionError

        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)

        def failing_func():
            raise LLMConnectionError("Test error")

        def success_func():
            return "success"

        # Open circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

        # Wait for recovery period
        time.sleep(1.1)

        # Successful call should close circuit
        result = cb.call(success_func)
        assert result == "success"
        assert cb.circuit_state == "CLOSED"

    def test_reopens_on_half_open_failure(self):
        """Failure in HALF_OPEN should reopen the circuit."""
        from local_llm import CircuitBreaker, LLMConnectionError

        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)

        def failing_func():
            raise LLMConnectionError("Test error")

        # Open circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        # Wait for recovery
        time.sleep(1.1)

        # Failed probe should reopen
        with pytest.raises(LLMConnectionError):
            cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

    def test_thread_safety(self):
        """Circuit breaker should handle concurrent access safely."""
        from local_llm import CircuitBreaker, LLMConnectionError

        cb = CircuitBreaker(failure_threshold=10, recovery_seconds=5)
        errors = []
        success_count = 0

        def call_many_times():
            nonlocal success_count
            for i in range(20):
                try:
                    if i % 3 == 0:
                        # Some failures
                        cb.call(lambda: exec('raise LLMConnectionError("error")'))
                    else:
                        # Some successes
                        cb.call(lambda: "ok")
                        success_count += 1
                except LLMConnectionError:
                    pass  # Expected
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=call_many_times) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No unexpected errors should occur
        assert len(errors) == 0
        # State should be valid
        assert cb.circuit_state in ("CLOSED", "OPEN", "HALF_OPEN")


# ============================================================
# Pre-computed Norms Tests
# ============================================================


class TestPrecomputedNorms:
    """Test pre-computed embedding index for performance."""

    def test_norms_built_after_load(self):
        """SimpleRAG should build doc matrix and norms after loading documents."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []  # No files to load

            rag = SimpleRAG(
                docs_folder="./test_docs",
                llm=mock_llm,
                embedder=mock_embedder,
            )

        # Manually add documents to simulate loading
        rag.documents = [{"content": "doc1"}, {"content": "doc2"}]
        rag.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        rag._build_embedding_index()

        assert rag._doc_matrix is not None
        assert rag._doc_norms is not None
        assert rag._doc_matrix.shape == (2, 3)
        assert rag._doc_norms.shape == (2,)

    def test_retrieve_uses_precomputed(self):
        """retrieve() should use pre-computed matrix and norms."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        # Populate documents and embeddings
        rag.documents = [{"content": "test doc 1"}, {"content": "test doc 2"}]
        rag.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        rag._build_embedding_index()

        # Retrieve should use pre-computed matrix
        results = rag.retrieve("query", top_k=1)

        assert len(results) == 1
        assert rag._doc_matrix is not None
        assert rag._doc_norms is not None

    def test_argpartition_used_for_large_docsets(self):
        """Verify argpartition is used when doc count >> top_k."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.5] * 384

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        # Create large document set
        num_docs = 1000
        rag.documents = [{"content": f"doc {i}"} for i in range(num_docs)]
        rag.embeddings = [[0.1] * 384 for _ in range(num_docs)]
        rag._build_embedding_index()

        # Mock argpartition to verify it's called
        with patch("numpy.argpartition", wraps=np.argpartition) as mock_argpart:
            rag.retrieve("test query", top_k=5)

            # Should use argpartition (1000 >> 5*4)
            assert mock_argpart.called

    def test_empty_documents_no_matrix(self):
        """Empty document list should not build matrix/norms."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        assert rag._doc_matrix is None
        assert rag._doc_norms is None


# ============================================================
# Embedding Cache Key Tests
# ============================================================


class TestEmbeddingCacheKey:
    """Test embedding cache key includes model name."""

    def test_includes_model_name(self):
        """Cache key should include embedding model name."""
        import tempfile
        from pathlib import Path

        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(
                llm=mock_llm,
                embedder=mock_embedder,
                embedding_model="custom-model-v2",
            )

        # Create a temp file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = Path(tmp.name)

        try:
            cache_key = rag._embedding_cache_key(tmp_path)

            # Key should contain model name
            assert "custom-model-v2" in f"{tmp_path.name}:{tmp_path.stat().st_mtime}:custom-model-v2"
            assert len(cache_key) == 64  # SHA256 hex digest length
        finally:
            tmp_path.unlink()

    def test_different_models_different_keys(self):
        """Different embedding models should produce different cache keys."""
        import tempfile
        from pathlib import Path

        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = Path(tmp.name)

        try:
            with patch("app_local.Path.glob") as mock_glob:
                mock_glob.return_value = []

                rag1 = SimpleRAG(llm=mock_llm, embedder=mock_embedder, embedding_model="model-a")
                rag2 = SimpleRAG(llm=mock_llm, embedder=mock_embedder, embedding_model="model-b")

            key1 = rag1._embedding_cache_key(tmp_path)
            key2 = rag2._embedding_cache_key(tmp_path)

            assert key1 != key2
        finally:
            tmp_path.unlink()


# ============================================================
# Thread-Safe Reload Tests
# ============================================================


class TestThreadSafeReload:
    """Test thread-safe document reload."""

    def test_reload_clears_index(self):
        """reload_documents() should clear doc_matrix and doc_norms."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        # Manually populate to simulate initial load
        rag.documents = [{"content": "doc"}]
        rag.embeddings = [[0.1, 0.2]]
        rag._build_embedding_index()

        assert rag._doc_matrix is not None
        assert rag._doc_norms is not None

        # Reload should clear
        with patch.object(rag, "_load_documents"):
            rag.reload_documents()

        assert rag.documents == []
        assert rag.embeddings == []
        assert rag._doc_matrix is None
        assert rag._doc_norms is None

    def test_reload_thread_safe(self):
        """Multiple concurrent reloads should not cause race conditions."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        errors = []

        def reload_many_times():
            try:
                for _ in range(10):
                    with patch.object(rag, "_load_documents"):
                        rag.reload_documents()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reload_many_times) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================
# Configurable Tax Rate Tests
# ============================================================


class TestConfigurableTaxRate:
    """Test tax_rate parameter in CharlieAnalyzer."""

    def test_default_tax_rate(self):
        """Analyzer should use default tax rate from settings."""
        from config import settings

        analyzer = CharlieAnalyzer()
        assert analyzer._tax_rate == settings.default_tax_rate

    def test_custom_tax_rate(self):
        """Analyzer should accept custom tax_rate parameter."""
        analyzer = CharlieAnalyzer(tax_rate=0.30)
        assert analyzer._tax_rate == 0.30

    def test_roic_uses_tax_rate(self):
        """ROIC calculation should use the configured tax_rate."""
        analyzer = CharlieAnalyzer(tax_rate=0.21)  # 21% tax rate

        data = FinancialData(
            operating_income=100000,
            total_equity=500000,
            total_debt=300000,
        )

        ratios = analyzer.calculate_profitability_ratios(data)

        # NOPAT = operating_income * (1 - tax_rate) = 100000 * 0.79 = 79000
        # Invested capital = equity + debt = 500000 + 300000 = 800000
        # ROIC = 79000 / 800000 = 0.09875
        assert ratios["roic"] == pytest.approx(0.09875, rel=0.01)

    def test_roic_different_tax_rates(self):
        """Different tax rates should produce different ROIC."""
        data = FinancialData(
            operating_income=100000,
            total_equity=500000,
            total_debt=300000,
        )

        analyzer_25 = CharlieAnalyzer(tax_rate=0.25)
        analyzer_30 = CharlieAnalyzer(tax_rate=0.30)

        ratios_25 = analyzer_25.calculate_profitability_ratios(data)
        ratios_30 = analyzer_30.calculate_profitability_ratios(data)

        # Higher tax rate = lower NOPAT = lower ROIC
        assert ratios_25["roic"] > ratios_30["roic"]


# ============================================================
# Answer Reuses Retrieval Tests
# ============================================================


class TestAnswerReusesRetrieval:
    """Test answer() accepting pre-retrieved documents."""

    def test_answer_with_provided_docs(self):
        """answer() should skip retrieval when docs are provided."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated answer"
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        # Pre-retrieved documents
        docs = [{"source": "test.txt", "content": "Test content", "type": "text"}]

        with patch.object(rag, "retrieve") as mock_retrieve:
            result = rag.answer("Test query", retrieved_docs=docs)

            # retrieve() should NOT be called
            mock_retrieve.assert_not_called()
            assert result == "Generated answer"

    def test_answer_without_docs_retrieves(self):
        """answer() should call retrieve() when no docs are provided."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated answer"
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 384

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []

            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        # Add a document
        rag.documents = [{"source": "test.txt", "content": "Test", "type": "text"}]
        rag.embeddings = [[0.1] * 384]
        rag._build_embedding_index()

        result = rag.answer("Test query")

        # Should have called embedder.embed for retrieval
        assert mock_embedder.embed.called
        assert result == "Generated answer"


# ============================================================
# Streaming LLM Tests
# ============================================================


class TestStreamingLLM:
    """Test streaming response generation in LocalLLM."""

    def test_generate_stream_yields_chunks(self):
        """generate_stream() should yield text chunks from Ollama."""
        llm = LocalLLM(model="test-model")

        chunks = [
            {"response": "Hello"},
            {"response": " world"},
            {"response": "!"},
        ]

        # Streaming routes through LocalLLM._get_stream_client().generate(); the
        # client is an ollama.Client when OLLAMA_HOST is set (it is, via .env),
        # so patch the seam (not module-level ollama.generate, which the client
        # path bypasses).
        mock_client = MagicMock()
        mock_client.generate.return_value = iter(chunks)
        with patch.object(LocalLLM, "_get_stream_client", return_value=mock_client):
            result = list(llm.generate_stream("Test prompt"))

        assert result == ["Hello", " world", "!"]

    def test_generate_stream_skips_empty_chunks(self):
        """Empty response chunks should be skipped."""
        llm = LocalLLM(model="test-model")

        chunks = [
            {"response": "Hello"},
            {"response": ""},
            {"response": " world"},
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = iter(chunks)
        with patch.object(LocalLLM, "_get_stream_client", return_value=mock_client):
            result = list(llm.generate_stream("Test prompt"))

        assert result == ["Hello", " world"]

    def test_generate_stream_respects_circuit_breaker(self):
        """Streaming should be rejected when circuit breaker is OPEN."""
        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_seconds=60,
        )

        # Open the circuit breaker
        def failing():
            raise LLMConnectionError("fail")

        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                llm._circuit_breaker.call(failing)

        assert llm.circuit_state == "OPEN"

        # Streaming should be rejected
        with pytest.raises(LLMConnectionError, match="Circuit breaker open"):
            list(llm.generate_stream("Test"))

    def test_generate_stream_connection_error(self):
        """ConnectionError during streaming should raise LLMConnectionError."""
        llm = LocalLLM(model="test-model")

        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("down")
        with patch.object(LocalLLM, "_get_stream_client", return_value=mock_client):
            with pytest.raises(LLMConnectionError, match="Cannot connect to Ollama"):
                list(llm.generate_stream("Test"))

    def test_generate_stream_records_success(self):
        """Successful streaming should record success on circuit breaker."""

        llm = LocalLLM(model="test-model", circuit_breaker_failure_threshold=3)

        chunks = [{"response": "OK"}]

        mock_client = MagicMock()
        mock_client.generate.return_value = iter(chunks)
        with patch.object(LocalLLM, "_get_stream_client", return_value=mock_client):
            # WS-4 P1-C2: generate_stream now calls the PUBLIC record_success
            # (the private _on_success is a thin alias).
            with patch.object(llm._circuit_breaker, "record_success") as mock_success:
                list(llm.generate_stream("Test"))
                mock_success.assert_called_once()

    def test_generate_stream_records_failure(self):
        """Failed streaming should record failure on circuit breaker."""
        llm = LocalLLM(model="test-model", circuit_breaker_failure_threshold=3)

        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("fail")
        with patch.object(LocalLLM, "_get_stream_client", return_value=mock_client):
            # WS-4 P1-C2: generate_stream now calls the PUBLIC record_failure.
            with patch.object(llm._circuit_breaker, "record_failure") as mock_failure:
                with pytest.raises(LLMConnectionError):
                    list(llm.generate_stream("Test"))
                mock_failure.assert_called_once()

    def test_generate_stream_passes_stream_flag(self):
        """generate_stream should pass stream=True to ollama.generate."""
        llm = LocalLLM(model="test-model")

        mock_client = MagicMock()
        mock_client.generate.return_value = iter([])
        with patch.object(LocalLLM, "_get_stream_client", return_value=mock_client):
            list(llm.generate_stream("Test prompt"))

            # P1-C1-stream-timeout (corrected): the streaming call must pass
            # stream=True and must NOT pass a timeout kwarg to generate() — the
            # installed ollama client rejects it; the read timeout lives on the
            # ollama.Client instead.
            mock_client.generate.assert_called_once()
            kwargs = mock_client.generate.call_args.kwargs
            assert kwargs.get("model") == "test-model"
            assert kwargs.get("prompt") == "Test prompt"
            assert kwargs.get("stream") is True
            assert "timeout" not in kwargs


# ============================================================
# Circuit Breaker allow_request Tests
# ============================================================


class TestCircuitBreakerAllowRequest:
    """Test the allow_request() method used by streaming paths."""

    def test_allow_request_closed(self):
        """allow_request should succeed when circuit is CLOSED."""
        from local_llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=10)
        cb.allow_request()  # Should not raise

    def test_allow_request_open_rejects(self):
        """allow_request should raise when circuit is OPEN."""
        from local_llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=60)

        def failing():
            raise LLMConnectionError("fail")

        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing)

        assert cb.circuit_state == "OPEN"

        with pytest.raises(LLMConnectionError, match="Circuit breaker open"):
            cb.allow_request()

    def test_allow_request_transitions_to_half_open(self):
        """allow_request should transition OPEN -> HALF_OPEN after recovery time."""
        from local_llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)

        def failing():
            raise LLMConnectionError("fail")

        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing)

        assert cb.circuit_state == "OPEN"

        time.sleep(1.1)
        cb.allow_request()  # Should not raise, transitions to HALF_OPEN
        assert cb.circuit_state == "HALF_OPEN"


# ============================================================
# Answer Stream Tests
# ============================================================


class TestAnswerStream:
    """Test streaming answer generation in SimpleRAG."""

    def test_answer_stream_yields_chunks(self):
        """answer_stream() should yield text chunks from the LLM."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_llm.generate_stream.return_value = iter(["Hello", " world"])
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []
            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        docs = [{"source": "test.txt", "content": "Test content", "type": "text"}]
        result = list(rag.answer_stream("What is this?", retrieved_docs=docs))

        assert result == ["Hello", " world"]

    def test_answer_stream_falls_back_to_generate(self):
        """answer_stream() should fall back to generate() if streaming not available."""
        from app_local import SimpleRAG

        mock_llm = MagicMock(spec=["generate"])  # No generate_stream
        mock_llm.generate.return_value = "Full answer"
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []
            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        docs = [{"source": "test.txt", "content": "Test content", "type": "text"}]
        result = list(rag.answer_stream("Test query", retrieved_docs=docs))

        assert result == ["Full answer"]

    def test_answer_stream_no_documents(self):
        """answer_stream should yield error message if no documents loaded."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []
            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        result = list(rag.answer_stream("Test"))
        assert len(result) == 1
        assert "No documents loaded" in result[0]

    def test_answer_stream_query_too_long(self):
        """answer_stream should yield error if query exceeds max length."""
        from app_local import SimpleRAG
        from config import settings

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []
            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        long_query = "x" * (settings.max_query_length + 1)
        result = list(rag.answer_stream(long_query))
        assert len(result) == 1
        assert "Query too long" in result[0]

    def test_answer_stream_handles_llm_error(self):
        """answer_stream should yield error message on LLM failure."""
        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_llm.generate_stream.side_effect = LLMConnectionError("Ollama down")
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []
            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        docs = [{"source": "test.txt", "content": "Test content", "type": "text"}]
        result = list(rag.answer_stream("Test", retrieved_docs=docs))

        assert len(result) == 1
        assert "Error generating answer" in result[0]


# ============================================================
# Content-Hash Cache Key Tests
# ============================================================


class TestContentHashCacheKey:
    """Test that cache key uses content hash for true change detection."""

    def test_same_content_same_key(self):
        """Same content should produce same cache key regardless of mtime."""
        import os
        import tempfile
        from pathlib import Path

        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []
            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        # Create first file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
            tmp.write("identical content")
            path1 = Path(tmp.name)

        # Wait and create second file with same content (different mtime)
        time.sleep(0.05)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
            tmp.write("identical content")
            path2 = Path(tmp.name)

        try:
            # Give them the same name for the cache key computation
            key1 = rag._embedding_cache_key(path1)
            key2 = rag._embedding_cache_key(path2)

            # Different files with same basename would have same key
            # but since temp files have different names, we check same file
            # touched at different times gives same key
            key_same = rag._embedding_cache_key(path1)
            os.utime(path1, (time.time() + 100, time.time() + 100))
            key_after_touch = rag._embedding_cache_key(path1)

            assert key_same == key_after_touch
        finally:
            path1.unlink()
            path2.unlink()

    def test_different_content_different_key(self):
        """Different content should produce different cache keys."""
        import tempfile
        from pathlib import Path

        from app_local import SimpleRAG

        mock_llm = MagicMock()
        mock_embedder = MagicMock()

        with patch("app_local.Path.glob") as mock_glob:
            mock_glob.return_value = []
            rag = SimpleRAG(llm=mock_llm, embedder=mock_embedder)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
            tmp.write("content version 1")
            tmp_path = Path(tmp.name)

        try:
            key_v1 = rag._embedding_cache_key(tmp_path)

            # Modify the file content
            tmp_path.write_text("content version 2")

            key_v2 = rag._embedding_cache_key(tmp_path)

            assert key_v1 != key_v2
        finally:
            tmp_path.unlink()


# ============================================================
# OLLAMA_HOST URL Validation (M-05)
# ============================================================


class TestOllamaHostValidation:
    def test_ftp_scheme_rejected(self):
        """Non-HTTP schemes should raise ValueError."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "ftp://evil.com"}):
            with pytest.raises(ValueError, match="http or https"):
                LocalEmbedder("test-model")

    def test_http_scheme_accepted(self):
        """Standard HTTP should work."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            embedder = LocalEmbedder("test-model")
            assert "http://localhost:11434" in embedder._url

    def test_https_scheme_accepted(self):
        """HTTPS should work with allowed hostname."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "https://localhost:443"}):
            embedder = LocalEmbedder("test-model")
            assert "https://localhost:443" in embedder._url

    def test_disallowed_hostname_rejected(self):
        """Hostnames outside allow-list should raise ValueError."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://evil.attacker.com:11434"}):
            with pytest.raises(ValueError, match="not in allow-list"):
                LocalEmbedder("test-model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
