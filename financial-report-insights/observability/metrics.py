"""RAG pipeline quality metrics collector.

Provides thread-safe in-memory collection of retrieval and LLM metrics for
observability dashboards. Metrics are stored in rolling windows and never
raise exceptions that would interrupt the main RAG pipeline.

Also exposes Prometheus HTTP instrumentation:
- ``_HTTP_REQUESTS``: Counter labeled by method/route/status
- ``_REQUEST_LATENCY``: Histogram labeled by method/route
- ``MetricsMiddleware``: Starlette BaseHTTPMiddleware that records both
- ``render_latest()``: Returns (bytes, content_type) for the /metrics endpoint
"""
import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import prometheus_client
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Default rolling window size (matches config default)
_DEFAULT_WINDOW_SIZE = 10_000


class MetricsCollector:
    """Thread-safe singleton that collects RAG pipeline metrics.

    All ``record_*`` methods are safe to call from concurrent threads and are
    designed to never raise exceptions – they log warnings on unexpected errors
    and return silently.

    Attributes:
        _window_size: Maximum number of entries per rolling window before the
            oldest entry is dropped.
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE) -> None:
        """Initialise the collector with empty rolling windows.

        Args:
            window_size: Maximum entries kept in each rolling window.
        """
        self._window_size = window_size
        self._lock = threading.Lock()

        # Rolling windows (deque with maxlen enforces the rolling limit)
        self._queries: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self._retrievals: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self._llm_calls: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self._cache_events: Deque[Dict[str, Any]] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------

    def record_query(self, trace_summary: Dict[str, Any]) -> None:
        """Record metrics derived from a completed trace summary.

        Args:
            trace_summary: Dict returned by ``Trace.summary()``.  Expected keys
                are ``total_duration_ms``, ``prompt_tokens``,
                ``completion_tokens``, and ``spans``.
        """
        try:
            entry: Dict[str, Any] = {
                "timestamp": time.time(),
                "total_duration_ms": float(trace_summary.get("total_duration_ms", 0.0)),
                "prompt_tokens": int(trace_summary.get("prompt_tokens", 0)),
                "completion_tokens": int(trace_summary.get("completion_tokens", 0)),
                "spans": dict(trace_summary.get("spans", {})),
            }
            with self._lock:
                self._queries.append(entry)
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.record_query failed: %s", exc)

    def record_retrieval(
        self,
        query_type: str,
        num_results: int,
        avg_similarity: float,
        search_time_ms: float,
    ) -> None:
        """Record a retrieval event.

        Args:
            query_type: One of ``"semantic"``, ``"bm25"``, ``"hybrid"``, etc.
            num_results: Number of documents returned.
            avg_similarity: Mean cosine similarity of the returned documents.
            search_time_ms: Wall-clock search time in milliseconds.
        """
        try:
            entry: Dict[str, Any] = {
                "timestamp": time.time(),
                "query_type": str(query_type),
                "num_results": int(num_results),
                "avg_similarity": float(avg_similarity),
                "search_time_ms": float(search_time_ms),
            }
            with self._lock:
                self._retrievals.append(entry)
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.record_retrieval failed: %s", exc)

    def record_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cache_hit: bool,
    ) -> None:
        """Record an LLM generation event.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.
            latency_ms: End-to-end LLM latency in milliseconds.
            cache_hit: Whether the response was served from cache.
        """
        try:
            entry: Dict[str, Any] = {
                "timestamp": time.time(),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "latency_ms": float(latency_ms),
                "cache_hit": bool(cache_hit),
            }
            with self._lock:
                self._llm_calls.append(entry)
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.record_llm_call failed: %s", exc)

    def record_cache_event(self, hit: bool) -> None:
        """Record an embedding or LLM cache hit/miss event.

        Args:
            hit: ``True`` for a cache hit, ``False`` for a miss.
        """
        try:
            entry: Dict[str, Any] = {
                "timestamp": time.time(),
                "hit": bool(hit),
            }
            with self._lock:
                self._cache_events.append(entry)
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.record_cache_event failed: %s", exc)

    # ------------------------------------------------------------------
    # Stat accessors (return copies; never raise)
    # ------------------------------------------------------------------

    def get_query_stats(self) -> Dict[str, Any]:
        """Return aggregated query statistics.

        Returns:
            Dict with keys:
            - ``total_queries`` (int)
            - ``avg_latency_ms`` (float)
            - ``avg_prompt_tokens`` (float)
            - ``avg_completion_tokens`` (float)
        """
        try:
            with self._lock:
                queries = list(self._queries)

            total = len(queries)
            if total == 0:
                return {
                    "total_queries": 0,
                    "avg_latency_ms": 0.0,
                    "avg_prompt_tokens": 0.0,
                    "avg_completion_tokens": 0.0,
                }

            avg_latency = sum(q["total_duration_ms"] for q in queries) / total
            avg_prompt = sum(q["prompt_tokens"] for q in queries) / total
            avg_completion = sum(q["completion_tokens"] for q in queries) / total

            return {
                "total_queries": total,
                "avg_latency_ms": round(avg_latency, 3),
                "avg_prompt_tokens": round(avg_prompt, 3),
                "avg_completion_tokens": round(avg_completion, 3),
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.get_query_stats failed: %s", exc)
            return {"total_queries": 0, "avg_latency_ms": 0.0,
                    "avg_prompt_tokens": 0.0, "avg_completion_tokens": 0.0}

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Return aggregated retrieval statistics.

        Returns:
            Dict with keys:
            - ``total_retrievals`` (int)
            - ``avg_similarity`` (float)
            - ``avg_search_time_ms`` (float)
            - ``avg_num_results`` (float)
            - ``search_type_distribution`` (dict mapping type -> count)
        """
        try:
            with self._lock:
                retrievals = list(self._retrievals)

            total = len(retrievals)
            if total == 0:
                return {
                    "total_retrievals": 0,
                    "avg_similarity": 0.0,
                    "avg_search_time_ms": 0.0,
                    "avg_num_results": 0.0,
                    "search_type_distribution": {},
                }

            avg_sim = sum(r["avg_similarity"] for r in retrievals) / total
            avg_time = sum(r["search_time_ms"] for r in retrievals) / total
            avg_results = sum(r["num_results"] for r in retrievals) / total

            distribution: Dict[str, int] = {}
            for r in retrievals:
                qt = r["query_type"]
                distribution[qt] = distribution.get(qt, 0) + 1

            return {
                "total_retrievals": total,
                "avg_similarity": round(avg_sim, 4),
                "avg_search_time_ms": round(avg_time, 3),
                "avg_num_results": round(avg_results, 3),
                "search_type_distribution": distribution,
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.get_retrieval_stats failed: %s", exc)
            return {
                "total_retrievals": 0,
                "avg_similarity": 0.0,
                "avg_search_time_ms": 0.0,
                "avg_num_results": 0.0,
                "search_type_distribution": {},
            }

    def get_llm_stats(self) -> Dict[str, Any]:
        """Return aggregated LLM call statistics.

        Returns:
            Dict with keys:
            - ``total_calls`` (int)
            - ``total_prompt_tokens`` (int)
            - ``total_completion_tokens`` (int)
            - ``total_tokens`` (int)
            - ``avg_latency_ms`` (float)
            - ``cache_hit_rate`` (float, 0-1)
            - ``avg_tokens_per_query`` (float)
        """
        try:
            with self._lock:
                calls = list(self._llm_calls)

            total = len(calls)
            if total == 0:
                return {
                    "total_calls": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "avg_latency_ms": 0.0,
                    "cache_hit_rate": 0.0,
                    "avg_tokens_per_query": 0.0,
                }

            total_prompt = sum(c["prompt_tokens"] for c in calls)
            total_completion = sum(c["completion_tokens"] for c in calls)
            total_tokens = total_prompt + total_completion
            avg_latency = sum(c["latency_ms"] for c in calls) / total
            hits = sum(1 for c in calls if c["cache_hit"])

            return {
                "total_calls": total,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "avg_latency_ms": round(avg_latency, 3),
                "cache_hit_rate": round(hits / total, 4),
                "avg_tokens_per_query": round(total_tokens / total, 3),
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.get_llm_stats failed: %s", exc)
            return {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0.0,
                "cache_hit_rate": 0.0,
                "avg_tokens_per_query": 0.0,
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return aggregated cache hit/miss statistics.

        Returns:
            Dict with keys:
            - ``total_events`` (int)
            - ``total_hits`` (int)
            - ``total_misses`` (int)
            - ``hit_rate`` (float, 0-1)
        """
        try:
            with self._lock:
                events = list(self._cache_events)

            total = len(events)
            if total == 0:
                return {
                    "total_events": 0,
                    "total_hits": 0,
                    "total_misses": 0,
                    "hit_rate": 0.0,
                }

            hits = sum(1 for e in events if e["hit"])
            misses = total - hits

            return {
                "total_events": total,
                "total_hits": hits,
                "total_misses": misses,
                "hit_rate": round(hits / total, 4),
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("MetricsCollector.get_cache_stats failed: %s", exc)
            return {"total_events": 0, "total_hits": 0, "total_misses": 0, "hit_rate": 0.0}

    def get_summary(self) -> Dict[str, Any]:
        """Return a combined summary of all metric categories.

        Returns:
            Dict with top-level keys ``queries``, ``retrievals``, ``llm``,
            ``cache``.
        """
        return {
            "queries": self.get_query_stats(),
            "retrievals": self.get_retrieval_stats(),
            "llm": self.get_llm_stats(),
            "cache": self.get_cache_stats(),
        }

    def reset(self) -> None:
        """Clear all collected metrics from all rolling windows."""
        with self._lock:
            self._queries.clear()
            self._retrievals.clear()
            self._llm_calls.clear()
            self._cache_events.clear()

    # ------------------------------------------------------------------
    # Internal raw accessors (used by dashboard_data)
    # ------------------------------------------------------------------

    def _get_raw_queries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._queries)

    def _get_raw_retrievals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._retrievals)

    def _get_raw_llm_calls(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._llm_calls)

    def _get_raw_cache_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._cache_events)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector_lock = threading.Lock()
_collector_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Return the process-wide singleton ``MetricsCollector``.

    The singleton is lazy-initialised on first call and the window size is
    read from ``config.settings.metrics_window_size`` when available.

    Returns:
        The shared ``MetricsCollector`` instance.
    """
    global _collector_instance
    if _collector_instance is None:
        with _collector_lock:
            if _collector_instance is None:
                window_size = _DEFAULT_WINDOW_SIZE
                try:
                    from config import settings
                    window_size = int(getattr(settings, "metrics_window_size", _DEFAULT_WINDOW_SIZE))
                except Exception:
                    pass
                _collector_instance = MetricsCollector(window_size=window_size)
    return _collector_instance


# ---------------------------------------------------------------------------
# Prometheus HTTP instrumentation
# ---------------------------------------------------------------------------

# Use the default process-wide registry so standard Go-runtime metrics are
# included alongside application metrics. Tests may swap these out via
# module-attribute replacement to use isolated CollectorRegistry instances.

_REGISTRY: CollectorRegistry = prometheus_client.REGISTRY

_HTTP_REQUESTS: Counter = Counter(
    "http_requests_total",
    "Total HTTP requests handled by this service",
    ["method", "route", "status"],
)

_REQUEST_LATENCY: Histogram = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "route"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Starlette/FastAPI middleware that records per-route request count and latency.

    Labels are derived from the matched route path template (e.g. '/graph/context/{period_label}')
    rather than the raw URL path to avoid high-cardinality label explosions.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        method = request.method
        status = str(response.status_code)

        # Prefer the route template if FastAPI resolved one
        route = request.url.path
        scope_route = request.scope.get("route")
        if scope_route is not None:
            path_attr = getattr(scope_route, "path", None)
            if path_attr:
                route = path_attr

        _HTTP_REQUESTS.labels(method=method, route=route, status=status).inc()
        _REQUEST_LATENCY.labels(method=method, route=route).observe(duration)

        return response


def render_latest() -> Tuple[bytes, str]:
    """Return the current Prometheus exposition in text format.

    Returns:
        A tuple of (exposition_bytes, content_type_string) suitable for
        building a plain-text HTTP response.
    """
    return generate_latest(_REGISTRY), CONTENT_TYPE_LATEST
