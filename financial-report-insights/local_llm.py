"""
Local LLM wrapper using Ollama.
Replaces Claude Sonnet with a free, local model.
"""

import atexit
import hashlib
import logging
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from enum import Enum

import ollama

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject immediately
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern for resilient LLM calls.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing recovery, allow one probe request

    Transitions:
    - CLOSED -> OPEN: After failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: After recovery_seconds have passed
    - HALF_OPEN -> CLOSED: On successful probe request
    - HALF_OPEN -> OPEN: On failed probe request
    """

    def __init__(self, failure_threshold: int, recovery_seconds: int):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
            recovery_seconds: Seconds to wait before attempting recovery
        """
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    @property
    def circuit_state(self) -> str:
        """Return the current circuit state as a string."""
        with self._lock:
            return self._state.value

    def allow_request(self):
        """Check if the circuit allows a request to proceed.

        Handles OPEN -> HALF_OPEN transition based on recovery time.
        Used by streaming paths that cannot wrap the entire call in
        circuit_breaker.call().

        Note: TOCTOU race exists between allow_request() and the actual call.
        With max_workers=2 this is acceptable — worst case is one extra request
        during HALF_OPEN -> OPEN transition.

        Raises:
            LLMConnectionError: If circuit is OPEN and not ready for recovery.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is not None:
                    time_since_failure = time.monotonic() - self._last_failure_time
                    if time_since_failure >= self._recovery_seconds:
                        logger.info("Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                        self._state = CircuitState.HALF_OPEN
                    else:
                        remaining = self._recovery_seconds - time_since_failure
                        raise LLMConnectionError(
                            f"Circuit breaker open: LLM service unavailable. Retry in {remaining:.0f}s."
                        )
                # Still OPEN after check (no last_failure_time)
                if self._state == CircuitState.OPEN:
                    raise LLMConnectionError("Circuit breaker open: LLM service unavailable.")

    def call(self, func, *args, **kwargs):
        """
        Execute a function through the circuit breaker.

        Args:
            func: The function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            The result of func(*args, **kwargs)

        Raises:
            LLMConnectionError: If circuit is open
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is not None:
                    time_since_failure = time.monotonic() - self._last_failure_time
                    if time_since_failure >= self._recovery_seconds:
                        logger.info("Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                        self._state = CircuitState.HALF_OPEN
                    else:
                        remaining = self._recovery_seconds - time_since_failure
                        raise LLMConnectionError(
                            f"Circuit breaker open: LLM service unavailable. Retry in {remaining:.0f}s."
                        )

            # If OPEN and not ready for recovery, reject
            if self._state == CircuitState.OPEN:
                remaining = self._recovery_seconds
                if self._last_failure_time is not None:
                    remaining = self._recovery_seconds - (time.monotonic() - self._last_failure_time)
                raise LLMConnectionError(f"Circuit breaker open: LLM service unavailable. Retry in {remaining:.0f}s.")

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except (LLMConnectionError, LLMTimeoutError):
            self._on_failure()
            raise

    def record_success(self):
        """Public: record a successful call and update circuit state.

        Note: self._lock is a plain non-reentrant threading.Lock; this method
        acquires it internally.  Do NOT call it while already holding self._lock
        (e.g. from within call()'s with-block) — that would deadlock.
        call() releases the lock before reaching this call site.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning to CLOSED (recovery successful)")
                self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def record_failure(self):
        """Public: record a failed call and update circuit state.

        Same lock-ownership note as record_success: acquires self._lock
        internally; must not be called while the caller holds self._lock.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning to OPEN (recovery failed)")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    logger.error(
                        f"Circuit breaker transitioning to OPEN (failure threshold {self._failure_threshold} reached)"
                    )
                    self._state = CircuitState.OPEN

    # Thin aliases kept for internal callers that pre-date the public API.
    # External code should prefer record_success / record_failure.
    def _on_success(self):
        """Alias for record_success (internal callers)."""
        self.record_success()

    def _on_failure(self):
        """Alias for record_failure (internal callers)."""
        self.record_failure()


class LLMConnectionError(Exception):
    """Raised when the Ollama backend is unreachable or returns an error."""


class LLMTimeoutError(LLMConnectionError):
    """Raised when the Ollama backend does not respond within the timeout."""


class LocalLLM:
    """Wrapper for Ollama local LLM with optional response caching and timeout."""

    def __init__(
        self,
        model: str = "llama3.2",
        enable_cache: bool = True,
        timeout_seconds: int = 120,
        max_retries: int = 2,
        circuit_breaker_failure_threshold: int = 3,
        circuit_breaker_recovery_seconds: int = 30,
    ):
        """
        Initialize the local LLM.

        Args:
            model: Ollama model name (e.g., "llama3.2", "mistral", "phi3")
            enable_cache: Whether to cache responses in memory
            timeout_seconds: Max seconds to wait for a response
            max_retries: Number of retries on transient failures
            circuit_breaker_failure_threshold: Failures before opening circuit
            circuit_breaker_recovery_seconds: Seconds before testing recovery
        """
        self.model = model
        self._cache: OrderedDict[str, str] | None = OrderedDict() if enable_cache else None
        self._cache_lock = threading.Lock()
        try:
            from config import settings as _cfg

            self._cache_maxsize = _cfg.llm_cache_maxsize
        except (ImportError, AttributeError):
            self._cache_maxsize = 128
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=2)
        atexit.register(self._shutdown_executor)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_failure_threshold,
            recovery_seconds=circuit_breaker_recovery_seconds,
        )

    def _shutdown_executor(self) -> None:
        """Shutdown the thread pool executor on process exit."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    @property
    def circuit_state(self) -> str:
        """Return the current circuit breaker state."""
        return self._circuit_breaker.circuit_state

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the local Ollama model.

        Args:
            prompt: The input prompt

        Returns:
            Generated text response

        Raises:
            LLMTimeoutError: If Ollama does not respond within the timeout.
            LLMConnectionError: If Ollama is unreachable, circuit is open, or returns an error.
        """
        # Check cache first
        if self._cache is not None:
            cache_key = self._cache_key(prompt)
            with self._cache_lock:
                if cache_key in self._cache:
                    logger.debug("LLM cache hit")
                    return self._cache[cache_key]

        # Execute through circuit breaker (wraps retry logic)
        result = self._circuit_breaker.call(self._generate_with_retry, prompt)

        # Store in cache (bounded LRU)
        if self._cache is not None:
            cache_key = self._cache_key(prompt)
            with self._cache_lock:
                self._cache[cache_key] = result
                if len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)

        return result

    def generate_stream(self, prompt: str):
        """Generate a streaming response using Ollama.

        Yields text chunks as they arrive. Streaming bypasses the response
        cache but respects the circuit breaker.

        Args:
            prompt: The input prompt

        Yields:
            str: Text chunks as they arrive

        Raises:
            LLMConnectionError: If Ollama is unreachable or circuit is open.
        """
        self._circuit_breaker.allow_request()

        try:
            yield from self._raw_generate_stream(prompt)
            self._circuit_breaker.record_success()
        except (LLMConnectionError, LLMTimeoutError):
            self._circuit_breaker.record_failure()
            raise

    def _get_stream_client(self):
        """Return an ollama.Client configured with a finite read timeout.

        The installed ollama client's ``generate()`` accepts no ``timeout``
        kwarg, so the read timeout must be configured on the Client instead
        (it is forwarded to the underlying httpx client). When OLLAMA_HOST is
        set we build a host-bound Client carrying ``timeout=self._timeout``;
        otherwise we fall back to the module-level ``ollama`` functions which
        use the default host. The client is cached on the instance.
        """
        import os

        client = getattr(self, "_stream_client", None)
        if client is not None:
            return client
        host = os.environ.get("OLLAMA_HOST")
        if host:
            client = ollama.Client(host=host, timeout=self._timeout)
        else:
            # No explicit host: use module-level functions (default host).
            client = ollama
        self._stream_client = client
        return client

    def _raw_generate_stream(self, prompt: str):
        """Raw streaming Ollama call without circuit breaker wrappers.

        A finite read timeout is configured on the ollama.Client so the
        underlying httpx read timeout is bounded.  This is required for
        thread/connection reclamation: without it, the blocked C-level socket
        read in the streaming path keeps the to_thread worker alive
        indefinitely even after the caller abandons the generator (see
        P1-C1-stream-timeout). The installed ollama.generate() has no timeout
        kwarg, so the timeout lives on the Client, not the generate() call.
        """
        try:
            last_chunk = {}
            client = self._get_stream_client()
            for chunk in client.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
            ):
                last_chunk = chunk
                text = chunk.get("response", "")
                if text:
                    yield text
            # Record token counts from the final chunk if trace is active
            try:
                from observability.tracing import get_current_trace

                trace = get_current_trace()
                if trace is not None and last_chunk:
                    prompt_tokens = last_chunk.get("prompt_eval_count", 0) or 0
                    completion_tokens = last_chunk.get("eval_count", 0) or 0
                    trace.add_token_counts(prompt_tokens, completion_tokens)
            except ImportError:
                pass
        except ConnectionError as e:
            raise LLMConnectionError("Cannot connect to Ollama. Is it running? (`ollama serve`)") from e
        except Exception as e:
            raise LLMConnectionError(f"Ollama error: {e}") from e

    def _generate_with_retry(self, prompt: str) -> str:
        """
        Generate with retry logic.

        This method is called by the circuit breaker and contains the retry logic.

        Args:
            prompt: The input prompt

        Returns:
            Generated text response

        Raises:
            LLMTimeoutError: If Ollama does not respond within the timeout.
            LLMConnectionError: If Ollama is unreachable or returns an error.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._call_with_timeout(prompt)
                return result
            except LLMTimeoutError:
                raise  # Don't retry timeouts – they'd just timeout again
            except LLMConnectionError as e:
                last_error = e
                if attempt < self._max_retries:
                    wait = 2**attempt
                    jitter = random.uniform(0, wait * 0.25)
                    sleep_secs = wait + jitter
                    logger.warning(
                        "LLM attempt %d/%d failed: %s (retry in %.2fs)",
                        attempt,
                        self._max_retries,
                        e,
                        sleep_secs,
                    )
                    time.sleep(sleep_secs)
                    continue
                raise
        # This should be unreachable but satisfies type checker
        raise last_error  # type: ignore[misc]

    def _call_with_timeout(self, prompt: str) -> str:
        """Execute the Ollama call with a timeout guard."""
        future = self._executor.submit(self._raw_generate, prompt)
        try:
            return future.result(timeout=self._timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise LLMTimeoutError(
                f"Ollama did not respond within {self._timeout}s. "
                "The model may be loading or the prompt may be too long."
            ) from None
        except (LLMConnectionError, LLMTimeoutError):
            raise
        except Exception as exc:
            raise LLMConnectionError(f"Unexpected LLM error: {exc}") from exc

    def _raw_generate(self, prompt: str) -> str:
        """Raw Ollama call without timeout/retry wrappers."""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
            )
            # Record token counts if trace is active
            try:
                from observability.tracing import get_current_trace

                trace = get_current_trace()
                if trace is not None:
                    prompt_tokens = response.get("prompt_eval_count", 0) or 0
                    completion_tokens = response.get("eval_count", 0) or 0
                    trace.add_token_counts(prompt_tokens, completion_tokens)
            except ImportError:
                pass
            return response.get("response", "")
        except ConnectionError as e:
            raise LLMConnectionError("Cannot connect to Ollama. Is it running? (`ollama serve`)") from e
        except Exception as e:
            raise LLMConnectionError(f"Ollama error: {e}") from e

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}:{prompt}".encode()).hexdigest()

    def __call__(self, prompt: str) -> str:
        """Allow calling the instance directly."""
        return self.generate(prompt)


class LocalEmbedder:
    """Local embeddings via OpenAI-compatible API (Docker Model Runner)."""

    def __init__(self, model_name: str = "mxbai-embed-large"):
        """
        Initialize the embedder using the OpenAI-compatible /v1/embeddings endpoint.

        DMR does not support the Ollama /api/embed endpoint, so we use
        the OpenAI-compatible /v1/embeddings API via httpx instead.

        Args:
            model_name: Model name for embeddings
        """
        import os

        import httpx

        self.model_name = model_name
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        # Validate URL scheme and hostname to prevent SSRF
        from urllib.parse import urlparse

        parsed = urlparse(host)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"OLLAMA_HOST must use http or https scheme, got: {parsed.scheme!r}")
        _ALLOWED_HOSTS = {
            "localhost",
            "127.0.0.1",
            "::1",
            "model-runner.docker.internal",
            "host.docker.internal",
            "ollama",  # Docker service name
        }
        hostname = (parsed.hostname or "").lower()
        if hostname not in _ALLOWED_HOSTS:
            raise ValueError(f"OLLAMA_HOST hostname {hostname!r} not in allow-list. Allowed: {sorted(_ALLOWED_HOSTS)}")
        self._url = f"{host.rstrip('/')}/v1/embeddings"
        self._client = httpx.Client(timeout=60.0)
        self._closed = False
        # Register atexit to close the httpx.Client on interpreter shutdown.
        # Guard against already-closed state is inside close() itself.
        atexit.register(self.close)
        # Use configured dimension to avoid probe HTTP request
        try:
            from config import settings as _cfg

            cfg_dim = _cfg.embedding_dimension
        except (ImportError, AttributeError):
            cfg_dim = 0
        if cfg_dim > 0:
            self.dimension = cfg_dim
        else:
            # Sentinel dimension lets the probe pass the self.dimension>0 guard
            # in _request_embeddings; the real dimension is set from the result.
            self.dimension = 1
            probe = self._request_embeddings(["dimension probe"])
            self.dimension = len(probe[0])

    def close(self) -> None:
        """Close the underlying httpx.Client and release its connection pool.

        Idempotent: calling close() more than once is safe and does nothing
        after the first call.  This method is registered as an atexit handler
        so that the OS-level socket is released even when the interpreter
        shuts down without an explicit close.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 – best-effort; do not let atexit raise
            logger.debug("LocalEmbedder.close(): ignoring error during client close", exc_info=True)

    def __enter__(self):
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the client when exiting the context manager."""
        self.close()
        return False

    def _request_embeddings(self, texts: list) -> list:
        """Call the OpenAI-compatible embeddings endpoint."""
        if self.dimension <= 0:
            raise RuntimeError(
                "Embedder dimension not initialized. Ensure embedding service is "
                "available and wait_for_embedding_service() has completed."
            )
        # mxbai-embed-large via DMR: ~512 token context per text, ~4K total batch tokens
        max_chars = 2500
        max_batch_chars = 3000
        # Item-count cap from config (default 32) prevents unbounded batch sizes
        try:
            from config import settings as _cfg

            max_batch_items = _cfg.embedding_batch_size
        except (ImportError, AttributeError):
            max_batch_items = 32
        safe_texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]
        # Sanitize text: replace NUL bytes and non-UTF8 that crash DMR tokenizer
        safe_texts = [t.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8") for t in safe_texts]
        # Skip empty/whitespace-only texts; return zero vectors for them later
        original_indices = list(range(len(safe_texts)))
        non_empty = [(i, t) for i, t in enumerate(safe_texts) if t.strip()]
        if not non_empty:
            return [[0.0] * self.dimension for _ in texts]
        filtered_indices, filtered_texts = zip(*non_empty)
        filtered_indices = list(filtered_indices)
        # Use extended retry budget for large ingestion jobs (100+ chunks)
        large_batch = len(texts) >= 100
        # Adaptive batching: group texts until char or item limit reached
        filtered_embeddings: list = []
        batch: list = []
        batch_chars = 0
        for t in filtered_texts:
            t_len = len(t)
            if batch and (batch_chars + t_len > max_batch_chars or len(batch) >= max_batch_items):
                filtered_embeddings.extend(self._send_embedding_batch(batch, large_batch=large_batch))
                batch = []
                batch_chars = 0
            batch.append(t)
            batch_chars += t_len
        if batch:
            filtered_embeddings.extend(self._send_embedding_batch(batch, large_batch=large_batch))
        # Map filtered results back, inserting zero vectors for empty texts
        zero_vec = [0.0] * self.dimension
        all_embeddings = [zero_vec] * len(safe_texts)
        for idx, emb in zip(filtered_indices, filtered_embeddings):
            all_embeddings[idx] = emb
        return all_embeddings

    def _send_embedding_batch(
        self,
        texts: list,
        max_retries: int = 3,
        large_batch: bool = False,
    ) -> list:
        """Send a batch of texts to the embedding endpoint with retry on 5xx.

        When large_batch is True, uses 5 retries with longer backoff (2s, 4s,
        8s, 16s) to tolerate DMR GPU warm-up latency during cold start.
        """
        import time

        import httpx

        if large_batch:
            max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.post(
                    self._url,
                    json={
                        "model": self.model_name,
                        "input": texts,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                try:
                    embeddings = [item["embedding"] for item in data["data"]]
                except (KeyError, TypeError, IndexError) as exc:
                    raise ValueError("Malformed embedding response: missing 'data' key or invalid structure") from exc
                if not embeddings:
                    logger.warning(
                        "Embedding response returned empty data list for batch of "
                        "%d texts; embeddings are missing for this batch.",
                        len(texts),
                    )
                return embeddings
            except httpx.RequestError as e:
                if attempt < max_retries:
                    wait = 2**attempt
                    jitter = random.uniform(0, wait * 0.25)
                    sleep_secs = wait + jitter
                    logger.warning(
                        "Embedding network error (attempt %d/%d, batch=%d texts), retrying in %.2fs: %s",
                        attempt,
                        max_retries,
                        len(texts),
                        sleep_secs,
                        e,
                    )
                    time.sleep(sleep_secs)
                    continue
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < max_retries:
                    wait = 2**attempt  # 2s, 4s, 8s, 16s
                    jitter = random.uniform(0, wait * 0.25)
                    sleep_secs = wait + jitter
                    logger.warning(
                        "Embedding 5xx error (attempt %d/%d, batch=%d texts), retrying in %.2fs: %s",
                        attempt,
                        max_retries,
                        len(texts),
                        sleep_secs,
                        e,
                    )
                    time.sleep(sleep_secs)
                    continue
                raise
        # Guard: this line is unreachable with any positive max_retries value
        # because every last-attempt branch either returns or raises.  It exists
        # as a future-edit safeguard: if a future change to the loop logic were
        # to drop a branch, the function would fall through to here rather than
        # silently returning None (which would cause callers to misinterpret
        # missing embeddings as success).
        raise RuntimeError(  # pragma: no branch
            f"unreachable: _send_embedding_batch retry loop exhausted without "
            f"returning or raising (max_retries={max_retries})"
        )

    def wait_for_embedding_service(self, timeout: int = 120) -> bool:
        """Wait for embedding service with graduated warm-up.

        Sends progressively larger batches (1, 5, 20 texts) to prime the
        model's GPU/memory before document ingestion begins.  This reduces
        cold-start 5xx failures when large Excel batches follow startup.

        Returns True if warm-up succeeds, False if timeout exceeded.
        """
        import time

        warmup_stages = [
            (1, "stage 1/3: single text"),
            (5, "stage 2/3: 5 texts"),
            (20, "stage 3/3: 20 texts"),
        ]

        start = time.monotonic()
        for count, label in warmup_stages:
            stage_texts = [f"warmup financial data row {i}" for i in range(count)]
            stage_ok = False
            while time.monotonic() - start < timeout:
                try:
                    self._send_embedding_batch(stage_texts, max_retries=1)
                    logger.info("Embedding warm-up %s succeeded", label)
                    stage_ok = True
                    break
                except Exception:
                    logger.info("Waiting for embedding service (%s)...", label)
                    time.sleep(2)
            if not stage_ok:
                logger.error(
                    "Embedding service not ready after %ds (failed at %s)",
                    timeout,
                    label,
                )
                return False

        elapsed = time.monotonic() - start
        logger.info("Embedding service fully warmed up in %.1fs", elapsed)
        return True

    def embed(self, text: str) -> list:
        """
        Generate embeddings for text.

        Args:
            text: Input text to embed

        Returns:
            List of floats (embedding vector)
        """
        return self._request_embeddings([text])[0]

    def embed_batch(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return self._request_embeddings(texts)
