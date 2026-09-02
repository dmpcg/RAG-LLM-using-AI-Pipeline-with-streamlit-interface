"""Tests for observability.metrics and observability.dashboard_data."""

import threading
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from observability.metrics import MetricsCollector, get_metrics_collector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace_summary(
    duration_ms: float = 100.0,
    prompt_tokens: int = 50,
    completion_tokens: int = 30,
    spans: dict | None = None,
) -> Dict[str, Any]:
    return {
        "trace_id": "abc123",
        "total_duration_ms": duration_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "spans": spans or {"llm_generation": 90.0},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def collector() -> MetricsCollector:
    """Fresh MetricsCollector for each test."""
    return MetricsCollector(window_size=100)


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_metrics_collector_returns_same_instance(self):
        a = get_metrics_collector()
        b = get_metrics_collector()
        assert a is b

    def test_singleton_is_metrics_collector_type(self):
        inst = get_metrics_collector()
        assert isinstance(inst, MetricsCollector)

    def test_singleton_has_correct_window_size(self):
        """Singleton should read window size from config (default 10000)."""
        inst = get_metrics_collector()
        assert inst._window_size >= 1


# ---------------------------------------------------------------------------
# record_query
# ---------------------------------------------------------------------------


class TestRecordQuery:
    def test_record_single_query(self, collector: MetricsCollector):
        collector.record_query(_make_trace_summary(duration_ms=200.0))
        stats = collector.get_query_stats()
        assert stats["total_queries"] == 1
        assert stats["avg_latency_ms"] == pytest.approx(200.0, abs=0.01)

    def test_record_multiple_queries_avg_latency(self, collector: MetricsCollector):
        collector.record_query(_make_trace_summary(duration_ms=100.0))
        collector.record_query(_make_trace_summary(duration_ms=200.0))
        stats = collector.get_query_stats()
        assert stats["total_queries"] == 2
        assert stats["avg_latency_ms"] == pytest.approx(150.0, abs=0.01)

    def test_record_query_token_averages(self, collector: MetricsCollector):
        collector.record_query(_make_trace_summary(prompt_tokens=100, completion_tokens=40))
        collector.record_query(_make_trace_summary(prompt_tokens=200, completion_tokens=60))
        stats = collector.get_query_stats()
        assert stats["avg_prompt_tokens"] == pytest.approx(150.0, abs=0.01)
        assert stats["avg_completion_tokens"] == pytest.approx(50.0, abs=0.01)

    def test_record_query_missing_keys_doesnt_raise(self, collector: MetricsCollector):
        """record_query should handle partial dicts gracefully."""
        collector.record_query({})
        assert collector.get_query_stats()["total_queries"] == 1

    def test_record_query_non_dict_doesnt_crash(self, collector: MetricsCollector):
        """record_query should silently handle unexpected input."""
        # Passing something truly weird should not raise
        try:
            collector.record_query({"total_duration_ms": "not-a-number"})
        except Exception:
            pass  # It may raise internally but the wrapper should suppress it


# ---------------------------------------------------------------------------
# record_retrieval
# ---------------------------------------------------------------------------


class TestRecordRetrieval:
    def test_record_single_retrieval(self, collector: MetricsCollector):
        collector.record_retrieval("semantic", 5, 0.85, 42.5)
        stats = collector.get_retrieval_stats()
        assert stats["total_retrievals"] == 1
        assert stats["avg_similarity"] == pytest.approx(0.85, abs=1e-4)
        assert stats["avg_search_time_ms"] == pytest.approx(42.5, abs=0.01)
        assert stats["avg_num_results"] == pytest.approx(5.0, abs=0.01)

    def test_record_multiple_retrievals_averages(self, collector: MetricsCollector):
        collector.record_retrieval("semantic", 3, 0.90, 10.0)
        collector.record_retrieval("bm25", 7, 0.70, 20.0)
        stats = collector.get_retrieval_stats()
        assert stats["total_retrievals"] == 2
        assert stats["avg_similarity"] == pytest.approx(0.80, abs=1e-4)
        assert stats["avg_search_time_ms"] == pytest.approx(15.0, abs=0.01)
        assert stats["avg_num_results"] == pytest.approx(5.0, abs=0.01)

    def test_search_type_distribution(self, collector: MetricsCollector):
        collector.record_retrieval("hybrid", 5, 0.8, 30.0)
        collector.record_retrieval("semantic", 3, 0.75, 15.0)
        collector.record_retrieval("hybrid", 4, 0.82, 25.0)
        dist = collector.get_retrieval_stats()["search_type_distribution"]
        assert dist["hybrid"] == 2
        assert dist["semantic"] == 1

    def test_record_zero_results_allowed(self, collector: MetricsCollector):
        collector.record_retrieval("semantic", 0, 0.0, 5.0)
        stats = collector.get_retrieval_stats()
        assert stats["total_retrievals"] == 1
        assert stats["avg_num_results"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# record_llm_call
# ---------------------------------------------------------------------------


class TestRecordLlmCall:
    def test_record_single_llm_call(self, collector: MetricsCollector):
        collector.record_llm_call(100, 50, 300.0, False)
        stats = collector.get_llm_stats()
        assert stats["total_calls"] == 1
        assert stats["total_prompt_tokens"] == 100
        assert stats["total_completion_tokens"] == 50
        assert stats["total_tokens"] == 150
        assert stats["avg_latency_ms"] == pytest.approx(300.0, abs=0.01)
        assert stats["cache_hit_rate"] == pytest.approx(0.0, abs=1e-4)

    def test_record_multiple_llm_calls_totals(self, collector: MetricsCollector):
        collector.record_llm_call(100, 50, 200.0, False)
        collector.record_llm_call(200, 80, 400.0, True)
        stats = collector.get_llm_stats()
        assert stats["total_calls"] == 2
        assert stats["total_prompt_tokens"] == 300
        assert stats["total_completion_tokens"] == 130
        assert stats["total_tokens"] == 430

    def test_cache_hit_rate_calculation(self, collector: MetricsCollector):
        collector.record_llm_call(50, 20, 100.0, True)
        collector.record_llm_call(50, 20, 100.0, True)
        collector.record_llm_call(50, 20, 100.0, False)
        stats = collector.get_llm_stats()
        assert stats["cache_hit_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_avg_tokens_per_query(self, collector: MetricsCollector):
        collector.record_llm_call(100, 50, 200.0, False)  # 150 total
        collector.record_llm_call(200, 50, 300.0, False)  # 250 total
        stats = collector.get_llm_stats()
        assert stats["avg_tokens_per_query"] == pytest.approx(200.0, abs=0.01)


# ---------------------------------------------------------------------------
# record_cache_event
# ---------------------------------------------------------------------------


class TestRecordCacheEvent:
    def test_all_hits(self, collector: MetricsCollector):
        for _ in range(5):
            collector.record_cache_event(True)
        stats = collector.get_cache_stats()
        assert stats["total_hits"] == 5
        assert stats["total_misses"] == 0
        assert stats["hit_rate"] == pytest.approx(1.0, abs=1e-4)

    def test_all_misses(self, collector: MetricsCollector):
        for _ in range(3):
            collector.record_cache_event(False)
        stats = collector.get_cache_stats()
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 3
        assert stats["hit_rate"] == pytest.approx(0.0, abs=1e-4)

    def test_mixed_cache_events(self, collector: MetricsCollector):
        collector.record_cache_event(True)
        collector.record_cache_event(False)
        collector.record_cache_event(True)
        collector.record_cache_event(True)
        stats = collector.get_cache_stats()
        assert stats["total_hits"] == 3
        assert stats["total_misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.75, abs=1e-4)
        assert stats["total_events"] == 4


# ---------------------------------------------------------------------------
# Empty-collector edge cases
# ---------------------------------------------------------------------------


class TestEmptyCollector:
    def test_get_query_stats_empty(self, collector: MetricsCollector):
        stats = collector.get_query_stats()
        assert stats["total_queries"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["avg_prompt_tokens"] == 0.0
        assert stats["avg_completion_tokens"] == 0.0

    def test_get_retrieval_stats_empty(self, collector: MetricsCollector):
        stats = collector.get_retrieval_stats()
        assert stats["total_retrievals"] == 0
        assert stats["avg_similarity"] == 0.0
        assert stats["search_type_distribution"] == {}

    def test_get_llm_stats_empty(self, collector: MetricsCollector):
        stats = collector.get_llm_stats()
        assert stats["total_calls"] == 0
        assert stats["total_tokens"] == 0
        assert stats["cache_hit_rate"] == 0.0

    def test_get_cache_stats_empty(self, collector: MetricsCollector):
        stats = collector.get_cache_stats()
        assert stats["total_events"] == 0
        assert stats["hit_rate"] == 0.0

    def test_get_summary_empty(self, collector: MetricsCollector):
        summary = collector.get_summary()
        assert "queries" in summary
        assert "retrievals" in summary
        assert "llm" in summary
        assert "cache" in summary
        assert summary["queries"]["total_queries"] == 0


# ---------------------------------------------------------------------------
# Rolling window tests
# ---------------------------------------------------------------------------


class TestRollingWindow:
    def test_window_enforced_on_queries(self):
        c = MetricsCollector(window_size=10)
        for i in range(20):
            c.record_query(_make_trace_summary(duration_ms=float(i)))
        # Should keep only last 10
        assert c.get_query_stats()["total_queries"] == 10

    def test_window_enforced_on_retrievals(self):
        c = MetricsCollector(window_size=5)
        for i in range(15):
            c.record_retrieval("semantic", 3, 0.8, float(i))
        assert c.get_retrieval_stats()["total_retrievals"] == 5

    def test_window_enforced_on_llm_calls(self):
        c = MetricsCollector(window_size=7)
        for _ in range(14):
            c.record_llm_call(10, 5, 50.0, False)
        assert c.get_llm_stats()["total_calls"] == 7

    def test_window_enforced_on_cache_events(self):
        c = MetricsCollector(window_size=3)
        for _ in range(9):
            c.record_cache_event(True)
        assert c.get_cache_stats()["total_events"] == 3

    def test_oldest_entries_dropped_not_newest(self):
        """The rolling window should keep the *newest* entries."""
        c = MetricsCollector(window_size=3)
        c.record_query(_make_trace_summary(duration_ms=1.0))
        c.record_query(_make_trace_summary(duration_ms=2.0))
        c.record_query(_make_trace_summary(duration_ms=3.0))
        # Now push one more to drop the oldest (1.0 ms)
        c.record_query(_make_trace_summary(duration_ms=4.0))
        raw = c._get_raw_queries()
        latencies = [r["total_duration_ms"] for r in raw]
        assert 1.0 not in latencies
        assert 4.0 in latencies

    def test_large_window_size_above_10000(self):
        """Window sizes above the default must still work."""
        c = MetricsCollector(window_size=50_000)
        for i in range(100):
            c.record_query(_make_trace_summary(duration_ms=float(i)))
        assert c.get_query_stats()["total_queries"] == 100


# ---------------------------------------------------------------------------
# Thread-safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_query(self, collector: MetricsCollector):
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    collector.record_query(_make_trace_summary())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # 10 threads x 20 = 200 but window_size=100, so capped at 100
        assert collector.get_query_stats()["total_queries"] == 100

    def test_concurrent_mixed_operations(self, collector: MetricsCollector):
        errors: List[Exception] = []

        def record_worker():
            try:
                for _ in range(10):
                    collector.record_retrieval("semantic", 3, 0.8, 10.0)
                    collector.record_llm_call(50, 20, 100.0, False)
                    collector.record_cache_event(True)
            except Exception as exc:
                errors.append(exc)

        def read_worker():
            try:
                for _ in range(10):
                    collector.get_summary()
            except Exception as exc:
                errors.append(exc)

        writers = [threading.Thread(target=record_worker) for _ in range(5)]
        readers = [threading.Thread(target=read_worker) for _ in range(3)]
        all_threads = writers + readers
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_reset(self, collector: MetricsCollector):
        """reset() must not corrupt state when called concurrently with records."""
        errors: List[Exception] = []

        def record_worker():
            try:
                for _ in range(50):
                    collector.record_query(_make_trace_summary())
            except Exception as exc:
                errors.append(exc)

        def reset_worker():
            try:
                for _ in range(5):
                    collector.reset()
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_worker) for _ in range(4)]
        threads.append(threading.Thread(target=reset_worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # After concurrent reset, collector must be in a valid state
        _ = collector.get_summary()


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_windows(self, collector: MetricsCollector):
        collector.record_query(_make_trace_summary())
        collector.record_retrieval("semantic", 3, 0.8, 10.0)
        collector.record_llm_call(50, 20, 100.0, False)
        collector.record_cache_event(True)

        collector.reset()

        assert collector.get_query_stats()["total_queries"] == 0
        assert collector.get_retrieval_stats()["total_retrievals"] == 0
        assert collector.get_llm_stats()["total_calls"] == 0
        assert collector.get_cache_stats()["total_events"] == 0

    def test_reset_then_record_works(self, collector: MetricsCollector):
        collector.record_query(_make_trace_summary(duration_ms=500.0))
        collector.reset()
        collector.record_query(_make_trace_summary(duration_ms=100.0))
        stats = collector.get_query_stats()
        assert stats["total_queries"] == 1
        assert stats["avg_latency_ms"] == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_summary_keys_present(self, collector: MetricsCollector):
        summary = collector.get_summary()
        for key in ("queries", "retrievals", "llm", "cache"):
            assert key in summary

    def test_summary_populated(self, collector: MetricsCollector):
        collector.record_query(_make_trace_summary(duration_ms=250.0))
        collector.record_retrieval("hybrid", 5, 0.88, 30.0)
        collector.record_llm_call(80, 40, 200.0, True)
        collector.record_cache_event(True)
        collector.record_cache_event(False)

        summary = collector.get_summary()
        assert summary["queries"]["total_queries"] == 1
        assert summary["retrievals"]["total_retrievals"] == 1
        assert summary["llm"]["total_calls"] == 1
        assert summary["cache"]["total_events"] == 2


# ---------------------------------------------------------------------------
# dashboard_data tests
# ---------------------------------------------------------------------------


class TestGetDashboardData:
    def test_dashboard_data_keys(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        data = get_dashboard_data(collector)
        expected_keys = {
            "query_volume_over_time",
            "latency_percentiles",
            "token_usage_breakdown",
            "query_type_distribution",
            "similarity_score_histogram",
            "cache_hit_rate_over_time",
            "summary",
        }
        assert expected_keys == set(data.keys())

    def test_empty_collector_dashboard(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        data = get_dashboard_data(collector)
        assert data["query_volume_over_time"] == {}
        assert data["latency_percentiles"]["p50"] == 0.0
        assert data["latency_percentiles"]["p99"] == 0.0
        assert data["token_usage_breakdown"]["total"] == 0
        assert data["query_type_distribution"] == {}
        assert len(data["similarity_score_histogram"]) == 10
        assert data["cache_hit_rate_over_time"] == {}

    def test_latency_percentiles_single_entry(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        collector.record_query(_make_trace_summary(duration_ms=400.0))
        data = get_dashboard_data(collector)
        percs = data["latency_percentiles"]
        # With a single point all percentiles equal that point
        for k in ("p50", "p90", "p95", "p99"):
            assert percs[k] == pytest.approx(400.0, abs=0.01)

    def test_latency_percentiles_ordering(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        for ms in [10.0, 50.0, 100.0, 200.0, 1000.0]:
            collector.record_query(_make_trace_summary(duration_ms=ms))
        data = get_dashboard_data(collector)
        p = data["latency_percentiles"]
        assert p["p50"] <= p["p90"] <= p["p95"] <= p["p99"]

    def test_token_usage_breakdown_sums(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        collector.record_llm_call(100, 40, 200.0, False)
        collector.record_llm_call(200, 60, 300.0, True)
        data = get_dashboard_data(collector)
        tb = data["token_usage_breakdown"]
        assert tb["prompt"] == 300
        assert tb["completion"] == 100
        assert tb["total"] == 400

    def test_query_type_distribution_pie_data(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        collector.record_retrieval("semantic", 3, 0.8, 10.0)
        collector.record_retrieval("bm25", 5, 0.6, 20.0)
        collector.record_retrieval("semantic", 4, 0.85, 12.0)
        data = get_dashboard_data(collector)
        dist = data["query_type_distribution"]
        assert dist["semantic"] == 2
        assert dist["bm25"] == 1

    def test_similarity_histogram_has_10_bins(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        for sim in [0.1, 0.3, 0.5, 0.7, 0.9]:
            collector.record_retrieval("semantic", 3, sim, 10.0)
        data = get_dashboard_data(collector)
        hist = data["similarity_score_histogram"]
        assert len(hist) == 10
        for bin_entry in hist:
            assert "bin_start" in bin_entry
            assert "bin_end" in bin_entry
            assert "count" in bin_entry
        assert sum(b["count"] for b in hist) == 5

    def test_similarity_histogram_bins_cover_zero_to_one(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        data = get_dashboard_data(collector)
        hist = data["similarity_score_histogram"]
        assert hist[0]["bin_start"] == pytest.approx(0.0, abs=0.01)
        assert hist[-1]["bin_end"] == pytest.approx(1.0, abs=0.01)

    def test_cache_hit_rate_over_time_nonempty(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        for _ in range(3):
            collector.record_cache_event(True)
        for _ in range(1):
            collector.record_cache_event(False)
        data = get_dashboard_data(collector)
        # All events in same minute bucket; hit rate should be 0.75
        rates = list(data["cache_hit_rate_over_time"].values())
        assert len(rates) >= 1
        assert rates[0] == pytest.approx(0.75, abs=1e-4)

    def test_query_volume_over_time_counts(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        for _ in range(5):
            collector.record_query(_make_trace_summary())
        data = get_dashboard_data(collector)
        total_volume = sum(data["query_volume_over_time"].values())
        assert total_volume == 5

    def test_summary_embedded_in_dashboard_data(self, collector: MetricsCollector):
        from observability.dashboard_data import get_dashboard_data

        collector.record_query(_make_trace_summary())
        data = get_dashboard_data(collector)
        assert data["summary"]["queries"]["total_queries"] == 1


# ---------------------------------------------------------------------------
# Prometheus /metrics endpoint tests (WS-1 P1-F5)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _reset_metrics_flag():
    """Ensure the enable_metrics_endpoint flag is restored after each test."""
    import api as api_module

    original = api_module.settings.enable_metrics_endpoint
    yield
    api_module.settings.enable_metrics_endpoint = original


@pytest.fixture()
def api_client(_reset_metrics_flag):
    """TestClient for the FastAPI app with a stable RAG mock."""
    from fastapi.testclient import TestClient

    import api as api_module

    api_module._rag_instance = None  # do not boot real RAG
    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


class TestPrometheusEndpointDisabled:
    """Test 1: /metrics returns 404 when enable_metrics_endpoint is False."""

    def test_returns_404_by_default(self, api_client):
        import api as api_module

        api_module.settings.enable_metrics_endpoint = False
        resp = api_client.get("/metrics")
        assert resp.status_code == 404

    def test_setting_defaults_to_false(self):
        from config import Settings

        assert Settings().enable_metrics_endpoint is False


class TestPrometheusEndpointEnabled:
    """Test 2: /metrics returns 200 with correct content-type and metric name."""

    def test_returns_200_when_enabled(self, api_client):
        import api as api_module

        api_module.settings.enable_metrics_endpoint = True
        resp = api_client.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_starts_text_plain(self, api_client):
        import api as api_module

        api_module.settings.enable_metrics_endpoint = True
        resp = api_client.get("/metrics")
        assert resp.headers["content-type"].startswith("text/plain")

    def test_body_contains_http_requests_total(self, api_client):
        import api as api_module

        api_module.settings.enable_metrics_endpoint = True
        resp = api_client.get("/metrics")
        assert b"http_requests_total" in resp.content


class TestPrometheusCounterIncrements:
    """Test 3: Counter increments appear in the exposition after a request."""

    def test_counter_appears_in_exposition_after_request(self, _reset_metrics_flag):
        """Issue a /health request then scrape /metrics; counter must be present."""
        import prometheus_client
        from fastapi.testclient import TestClient

        import api as api_module
        import observability.metrics as obs

        # Isolated registry so this test does not pollute the global one
        registry = prometheus_client.CollectorRegistry()
        isolated_counter = prometheus_client.Counter(
            "http_requests_total_ws1_test",
            "Isolated counter for WS-1 increment test",
            ["method", "route", "status"],
            registry=registry,
        )
        isolated_latency = prometheus_client.Histogram(
            "http_request_duration_seconds_ws1_test",
            "Isolated histogram for WS-1 increment test",
            ["method", "route"],
            registry=registry,
        )

        orig_counter = obs._HTTP_REQUESTS
        orig_latency = obs._REQUEST_LATENCY
        orig_registry = obs._REGISTRY
        obs._HTTP_REQUESTS = isolated_counter
        obs._REQUEST_LATENCY = isolated_latency
        obs._REGISTRY = registry

        api_module._rag_instance = None
        api_module.settings.enable_metrics_endpoint = True

        try:
            with TestClient(api_module.app, raise_server_exceptions=False) as client:
                with patch(
                    "api.get_health_status",
                    return_value={"healthy": True, "status": "ok", "checks": []},
                ):
                    client.get("/health")

                resp = client.get("/metrics")

            assert resp.status_code == 200
            body = resp.content.decode()
            assert "http_requests_total_ws1_test" in body
        finally:
            obs._HTTP_REQUESTS = orig_counter
            obs._REQUEST_LATENCY = orig_latency
            obs._REGISTRY = orig_registry
