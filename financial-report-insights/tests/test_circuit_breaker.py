"""
Tests for circuit breaker pattern in LocalLLM.
"""

import time
from unittest.mock import patch

import pytest

from local_llm import CircuitBreaker, LLMConnectionError, LLMTimeoutError, LocalLLM


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=30)
        assert cb.circuit_state == "CLOSED"

    def test_successful_calls_keep_circuit_closed(self):
        """Successful calls should keep the circuit CLOSED."""
        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=30)

        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb.circuit_state == "CLOSED"

        # Multiple successes
        for _ in range(5):
            cb.call(success_func)
        assert cb.circuit_state == "CLOSED"

    def test_circuit_opens_after_threshold_failures(self):
        """Circuit should open after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=30)

        def failing_func():
            raise LLMConnectionError("Service down")

        # First two failures - circuit stays CLOSED
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)
            assert cb.circuit_state == "CLOSED"

        # Third failure - circuit opens
        with pytest.raises(LLMConnectionError):
            cb.call(failing_func)
        assert cb.circuit_state == "OPEN"

    def test_open_circuit_rejects_immediately(self):
        """OPEN circuit should reject calls without executing function."""
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=30)

        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise LLMConnectionError("Service down")

        # Trigger circuit opening
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"
        assert call_count == 2

        # Now function should not be called
        with pytest.raises(LLMConnectionError, match="Circuit breaker open"):
            cb.call(failing_func)
        assert call_count == 2  # No new calls

    def test_circuit_transitions_to_half_open_after_recovery_time(self):
        """Circuit should transition to HALF_OPEN after recovery seconds."""
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)  # 1 second for testing

        def failing_func():
            raise LLMConnectionError("Service down")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

        # Wait for recovery period
        time.sleep(1.1)

        # Next call should transition to HALF_OPEN (then fail)
        with pytest.raises(LLMConnectionError):
            cb.call(failing_func)

        # After the failed probe, should be back to OPEN
        assert cb.circuit_state == "OPEN"

    def test_half_open_success_closes_circuit(self):
        """Successful call in HALF_OPEN should close the circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)

        def failing_func():
            raise LLMConnectionError("Service down")

        def success_func():
            return "recovered"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

        # Wait for recovery period
        time.sleep(1.1)

        # Successful probe should close circuit
        result = cb.call(success_func)
        assert result == "recovered"
        assert cb.circuit_state == "CLOSED"

        # Subsequent calls should work normally
        result = cb.call(success_func)
        assert result == "recovered"
        assert cb.circuit_state == "CLOSED"

    def test_half_open_failure_reopens_circuit(self):
        """Failed call in HALF_OPEN should reopen the circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)

        def failing_func():
            raise LLMConnectionError("Service down")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

        # Wait for recovery period
        time.sleep(1.1)

        # Failed probe should reopen circuit
        with pytest.raises(LLMConnectionError):
            cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

    def test_timeout_errors_trigger_circuit_breaker(self):
        """LLMTimeoutError should also trigger circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=30)

        def timeout_func():
            raise LLMTimeoutError("Request timed out")

        # Timeouts should count as failures
        for _ in range(2):
            with pytest.raises(LLMTimeoutError):
                cb.call(timeout_func)

        assert cb.circuit_state == "OPEN"

    def test_success_resets_failure_count(self):
        """Success should reset failure count, preventing circuit from opening."""
        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=30)

        def failing_func():
            raise LLMConnectionError("Service down")

        def success_func():
            return "success"

        # Two failures
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "CLOSED"

        # Success resets count
        cb.call(success_func)
        assert cb.circuit_state == "CLOSED"

        # Two more failures (should not open since count was reset)
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                cb.call(failing_func)

        assert cb.circuit_state == "CLOSED"

        # One more failure opens circuit
        with pytest.raises(LLMConnectionError):
            cb.call(failing_func)

        assert cb.circuit_state == "OPEN"

    def test_thread_safety(self):
        """Circuit breaker should be thread-safe."""
        import threading

        cb = CircuitBreaker(failure_threshold=5, recovery_seconds=30)
        success_count = 0
        error_count = 0

        def worker():
            nonlocal success_count, error_count
            try:
                cb.call(lambda: "success")
                success_count += 1
            except LLMConnectionError:
                error_count += 1

        # Run multiple threads concurrently
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (circuit is CLOSED)
        assert success_count == 10
        assert error_count == 0
        assert cb.circuit_state == "CLOSED"


class TestLocalLLMWithCircuitBreaker:
    """Test suite for LocalLLM integration with circuit breaker."""

    def test_llm_has_circuit_state_property(self):
        """LocalLLM should expose circuit_state property."""
        llm = LocalLLM(model="test-model")
        assert llm.circuit_state == "CLOSED"

    @patch("local_llm.ollama.generate")
    def test_successful_generation_keeps_circuit_closed(self, mock_generate):
        """Successful LLM calls should keep circuit CLOSED."""
        mock_generate.return_value = {"response": "test response"}

        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_seconds=30,
        )

        result = llm.generate("test prompt")
        assert result == "test response"
        assert llm.circuit_state == "CLOSED"

    @patch("local_llm.ollama.generate")
    def test_repeated_failures_open_circuit(self, mock_generate):
        """Repeated LLM failures should open the circuit."""
        mock_generate.side_effect = ConnectionError("Ollama down")

        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_seconds=30,
            max_retries=1,  # No retries for faster test
        )

        # First failure
        with pytest.raises(LLMConnectionError):
            llm.generate("prompt1")
        assert llm.circuit_state == "CLOSED"

        # Second failure opens circuit
        with pytest.raises(LLMConnectionError):
            llm.generate("prompt2")
        assert llm.circuit_state == "OPEN"

    @patch("local_llm.ollama.generate")
    def test_open_circuit_rejects_without_calling_ollama(self, mock_generate):
        """OPEN circuit should reject calls without calling Ollama."""
        mock_generate.side_effect = ConnectionError("Ollama down")

        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_seconds=30,
            max_retries=1,
        )

        # Open the circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                llm.generate("test prompt")

        assert llm.circuit_state == "OPEN"
        call_count = mock_generate.call_count

        # This should be rejected without calling Ollama
        with pytest.raises(LLMConnectionError, match="Circuit breaker open"):
            llm.generate("another prompt")

        assert mock_generate.call_count == call_count  # No new calls

    @patch("local_llm.ollama.generate")
    def test_circuit_recovery_after_timeout(self, mock_generate):
        """Circuit should allow probe after recovery timeout."""
        # First two calls fail
        mock_generate.side_effect = [
            ConnectionError("Ollama down"),
            ConnectionError("Ollama down"),
            {"response": "recovered"},  # Recovery probe succeeds
        ]

        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_seconds=1,  # 1 second for testing
            max_retries=1,
        )

        # Open the circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                llm.generate("test prompt")

        assert llm.circuit_state == "OPEN"

        # Wait for recovery period
        time.sleep(1.1)

        # This should transition to HALF_OPEN and succeed
        result = llm.generate("recovery probe")
        assert result == "recovered"
        assert llm.circuit_state == "CLOSED"

    @patch("local_llm.ollama.generate")
    def test_cache_bypasses_circuit_breaker(self, mock_generate):
        """Cache hits should bypass circuit breaker entirely."""
        mock_generate.side_effect = [
            {"response": "cached response"},
            ConnectionError("Ollama down"),
            ConnectionError("Ollama down"),
        ]

        llm = LocalLLM(
            model="test-model",
            enable_cache=True,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_seconds=30,
            max_retries=1,
        )

        # First call populates cache
        result = llm.generate("test prompt")
        assert result == "cached response"

        # Open the circuit with other prompts
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                llm.generate("different prompt")

        assert llm.circuit_state == "OPEN"

        # Cache hit should still work even with open circuit
        result = llm.generate("test prompt")
        assert result == "cached response"

    @patch("local_llm.ollama.generate")
    def test_retries_happen_within_circuit_breaker(self, mock_generate):
        """Retries should happen within circuit breaker (all count as one attempt)."""
        # First attempt: retry twice then succeed
        mock_generate.side_effect = [
            ConnectionError("Transient error"),
            {"response": "success after retry"},
        ]

        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_seconds=30,
            max_retries=2,
        )

        # Should succeed after retry
        result = llm.generate("test prompt")
        assert result == "success after retry"
        assert llm.circuit_state == "CLOSED"

        # Only 2 calls to Ollama (initial + 1 retry)
        assert mock_generate.call_count == 2

    @patch("local_llm.ollama.generate")
    def test_circuit_opens_after_all_retries_exhausted(self, mock_generate):
        """Circuit should only count as failure after all retries are exhausted."""
        mock_generate.side_effect = ConnectionError("Persistent error")

        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_seconds=30,
            max_retries=2,
        )

        # First call: 1 initial + 1 retry = 2 calls to Ollama, counts as 1 failure
        with pytest.raises(LLMConnectionError):
            llm.generate("prompt1")
        assert llm.circuit_state == "CLOSED"
        assert mock_generate.call_count == 2

        # Second call: another 2 calls to Ollama, counts as 2nd failure, opens circuit
        with pytest.raises(LLMConnectionError):
            llm.generate("prompt2")
        assert llm.circuit_state == "OPEN"
        assert mock_generate.call_count == 4

    @patch("local_llm.ollama.generate")
    def test_circuit_breaker_error_message_includes_retry_time(self, mock_generate):
        """Circuit breaker error should tell user when to retry."""
        mock_generate.side_effect = ConnectionError("Ollama down")

        llm = LocalLLM(
            model="test-model",
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_seconds=30,
            max_retries=1,
        )

        # Open the circuit
        for _ in range(2):
            with pytest.raises(LLMConnectionError):
                llm.generate("test prompt")

        assert llm.circuit_state == "OPEN"

        # Error message should mention retry time
        with pytest.raises(LLMConnectionError) as exc_info:
            llm.generate("another prompt")

        error_msg = str(exc_info.value)
        assert "Circuit breaker open" in error_msg
        assert "Retry in" in error_msg
        assert "30s" in error_msg or "29s" in error_msg  # Allow for timing variance
