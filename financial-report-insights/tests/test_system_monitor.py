"""Tests for observability.system_monitor (SystemMonitor + PerformanceBaseline)."""

import json
import threading
import time

import pytest

from observability.metrics import MetricsCollector
from observability.system_monitor import (
    _ALERT_LATENCY_CRITICAL_MS,
    _ALERT_LATENCY_WARNING_MS,
    PerformanceBaseline,
    SystemMonitor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def monitor() -> SystemMonitor:
    """Fresh SystemMonitor with no MetricsCollector."""
    return SystemMonitor()


@pytest.fixture()
def monitor_with_collector() -> SystemMonitor:
    """Fresh SystemMonitor backed by a real MetricsCollector."""
    collector = MetricsCollector(window_size=500)
    return SystemMonitor(metrics_collector=collector)


@pytest.fixture()
def baseline() -> PerformanceBaseline:
    """Fresh PerformanceBaseline."""
    return PerformanceBaseline()


# ===========================================================================
# SystemMonitor – basic recording
# ===========================================================================


class TestRecordError:
    def test_record_error_increments_count(self, monitor: SystemMonitor) -> None:
        monitor.record_error("timeout", "connection timed out", "llm")
        with monitor._lock:
            assert len(monitor._errors) == 1

    def test_record_error_stores_all_fields(self, monitor: SystemMonitor) -> None:
        monitor.record_error("io_error", "disk full", "embedder")
        ts, etype, msg, comp = monitor._errors[0]
        assert etype == "io_error"
        assert msg == "disk full"
        assert comp == "embedder"
        assert ts == pytest.approx(time.time(), abs=2.0)

    def test_record_multiple_errors(self, monitor: SystemMonitor) -> None:
        for i in range(5):
            monitor.record_error("err", f"msg {i}", f"comp_{i}")
        with monitor._lock:
            assert len(monitor._errors) == 5


class TestRecordLatency:
    def test_record_latency_stores_values(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("retrieval", 123.4)
        with monitor._lock:
            assert len(monitor._latencies) == 1

    def test_record_latency_all_fields(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("llm", 999.0)
        ts, comp, lat = monitor._latencies[0]
        assert comp == "llm"
        assert lat == pytest.approx(999.0)

    def test_record_latency_multiple_components(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("retrieval", 50.0)
        monitor.record_latency("llm", 200.0)
        monitor.record_latency("retrieval", 75.0)
        with monitor._lock:
            comps = [e[1] for e in monitor._latencies]
        assert comps.count("retrieval") == 2
        assert comps.count("llm") == 1


# ===========================================================================
# SystemMonitor – get_health_status structure and semantics
# ===========================================================================


class TestGetHealthStatus:
    def test_returns_correct_keys(self, monitor: SystemMonitor) -> None:
        status = monitor.get_health_status()
        expected_keys = {
            "overall_status",
            "component_status",
            "error_rate",
            "avg_latency_ms",
            "uptime_seconds",
        }
        assert expected_keys == set(status.keys())

    def test_healthy_when_no_errors(self, monitor: SystemMonitor) -> None:
        status = monitor.get_health_status()
        assert status["overall_status"] == "healthy"

    def test_healthy_error_rate_zero(self, monitor: SystemMonitor) -> None:
        status = monitor.get_health_status()
        assert status["error_rate"] == 0.0

    def test_healthy_latency_zero(self, monitor: SystemMonitor) -> None:
        status = monitor.get_health_status()
        assert status["avg_latency_ms"] == 0.0

    def test_uptime_positive(self, monitor: SystemMonitor) -> None:
        status = monitor.get_health_status()
        assert status["uptime_seconds"] >= 0.0

    def test_component_status_empty_when_no_data(self, monitor: SystemMonitor) -> None:
        status = monitor.get_health_status()
        assert status["component_status"] == {}

    def test_degraded_with_some_errors(self, monitor: SystemMonitor) -> None:
        # error_rate between _DEGRADED_ERROR_RATE (2/min) and _UNHEALTHY_ERROR_RATE (10/min)
        # 5-min window, so 5 errors * 1/5 = 1 err/min -> not enough;
        # 15 errors -> 15/5 = 3/min -> degraded
        for i in range(15):
            monitor.record_error("err", "msg", "service")
        status = monitor.get_health_status()
        assert status["overall_status"] in ("degraded", "unhealthy")

    def test_unhealthy_with_many_errors(self, monitor: SystemMonitor) -> None:
        # Need > 10 errors/min over 5-min window => > 50 errors
        for i in range(60):
            monitor.record_error("err", "msg", "service")
        status = monitor.get_health_status()
        assert status["overall_status"] == "unhealthy"

    def test_degraded_with_high_latency(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("llm", 7000.0)  # above warning threshold
        status = monitor.get_health_status()
        assert status["overall_status"] in ("degraded", "unhealthy")

    def test_unhealthy_with_critical_latency(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("llm", 15000.0)  # above critical threshold
        status = monitor.get_health_status()
        assert status["overall_status"] == "unhealthy"

    def test_component_status_present_after_error(self, monitor: SystemMonitor) -> None:
        monitor.record_error("err", "msg", "retrieval")
        status = monitor.get_health_status()
        assert "retrieval" in status["component_status"]

    def test_component_status_present_after_latency(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("embedder", 100.0)
        status = monitor.get_health_status()
        assert "embedder" in status["component_status"]


# ===========================================================================
# SystemMonitor – alert rules
# ===========================================================================


class TestGetAlerts:
    def test_no_alerts_when_healthy(self, monitor: SystemMonitor) -> None:
        alerts = monitor.get_alerts()
        assert alerts == []

    def test_get_alerts_delegates_to_check_alert_rules(self, monitor: SystemMonitor) -> None:
        """get_alerts() must return the same result as check_alert_rules()."""
        assert monitor.get_alerts() == monitor.check_alert_rules()


class TestCheckAlertRules:
    def test_no_alerts_when_no_data(self, monitor: SystemMonitor) -> None:
        assert monitor.check_alert_rules() == []

    def test_critical_alert_on_high_error_rate(self, monitor: SystemMonitor) -> None:
        # 60 errors in the 5-min window = 12/min > 10/min threshold
        for _ in range(60):
            monitor.record_error("err", "msg", "svc")
        alerts = monitor.check_alert_rules()
        levels = [a["level"] for a in alerts]
        assert "critical" in levels

    def test_critical_alert_component_system(self, monitor: SystemMonitor) -> None:
        for _ in range(60):
            monitor.record_error("err", "msg", "svc")
        alerts = monitor.check_alert_rules()
        critical = [a for a in alerts if a["level"] == "critical"]
        assert any(a["component"] == "system" for a in critical)

    def test_warning_alert_on_warning_latency(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("llm", _ALERT_LATENCY_WARNING_MS + 1.0)
        alerts = monitor.check_alert_rules()
        assert any(a["level"] == "warning" for a in alerts)

    def test_critical_alert_on_critical_latency(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("llm", _ALERT_LATENCY_CRITICAL_MS + 1.0)
        alerts = monitor.check_alert_rules()
        assert any(a["level"] == "critical" for a in alerts)

    def test_no_warning_latency_alert_when_below_threshold(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("llm", 100.0)
        alerts = monitor.check_alert_rules()
        assert all(a.get("component") != "system" or a["level"] != "warning" for a in alerts)

    def test_alert_dict_has_required_keys(self, monitor: SystemMonitor) -> None:
        for _ in range(60):
            monitor.record_error("err", "msg", "svc")
        alerts = monitor.check_alert_rules()
        for alert in alerts:
            assert "level" in alert
            assert "component" in alert
            assert "message" in alert
            assert "timestamp" in alert

    def test_cache_hit_rate_warning(self) -> None:
        collector = MetricsCollector(window_size=500)
        # Record 101 misses -> hit_rate = 0 < 0.1 with > 100 events
        for _ in range(101):
            collector.record_cache_event(hit=False)
        mon = SystemMonitor(metrics_collector=collector)
        alerts = mon.check_alert_rules()
        cache_alerts = [a for a in alerts if a["component"] == "cache"]
        assert len(cache_alerts) == 1
        assert cache_alerts[0]["level"] == "warning"

    def test_no_cache_alert_when_below_query_threshold(self) -> None:
        collector = MetricsCollector(window_size=500)
        # Only 50 misses – below the 100-query threshold
        for _ in range(50):
            collector.record_cache_event(hit=False)
        mon = SystemMonitor(metrics_collector=collector)
        alerts = mon.check_alert_rules()
        assert all(a["component"] != "cache" for a in alerts)

    def test_no_cache_alert_when_hit_rate_ok(self) -> None:
        collector = MetricsCollector(window_size=500)
        # 101 hits -> hit_rate = 1.0, no alert
        for _ in range(101):
            collector.record_cache_event(hit=True)
        mon = SystemMonitor(metrics_collector=collector)
        alerts = mon.check_alert_rules()
        assert all(a["component"] != "cache" for a in alerts)

    def test_no_cache_alert_without_collector(self, monitor: SystemMonitor) -> None:
        """Monitor without collector must not raise or produce cache alerts."""
        alerts = monitor.check_alert_rules()
        assert all(a["component"] != "cache" for a in alerts)


# ===========================================================================
# SystemMonitor – thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_record_error(self, monitor: SystemMonitor) -> None:
        n_threads = 20
        errors_per_thread = 50

        def worker():
            for i in range(errors_per_thread):
                monitor.record_error("err", f"msg {i}", "svc")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with monitor._lock:
            assert len(monitor._errors) == n_threads * errors_per_thread

    def test_concurrent_record_latency(self, monitor: SystemMonitor) -> None:
        n_threads = 10
        records_per_thread = 30

        def worker():
            for i in range(records_per_thread):
                monitor.record_latency("svc", float(i))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with monitor._lock:
            assert len(monitor._latencies) == n_threads * records_per_thread

    def test_concurrent_get_health_status(self, monitor: SystemMonitor) -> None:
        """get_health_status must not raise under concurrent writes."""
        results = []
        errors_raised = []

        def writer():
            for _ in range(20):
                monitor.record_error("err", "msg", "svc")
                monitor.record_latency("svc", 100.0)

        def reader():
            try:
                for _ in range(10):
                    results.append(monitor.get_health_status())
            except Exception as exc:
                errors_raised.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors_raised == []


# ===========================================================================
# SystemMonitor – edge cases
# ===========================================================================


class TestEdgeCases:
    def test_no_data_returns_healthy(self, monitor: SystemMonitor) -> None:
        status = monitor.get_health_status()
        assert status["overall_status"] == "healthy"

    def test_single_error_below_degraded_threshold(self, monitor: SystemMonitor) -> None:
        # 1 error in 5-min window = 0.2/min < 2/min degraded threshold
        monitor.record_error("err", "msg", "svc")
        status = monitor.get_health_status()
        # single error might still appear in component_status as degraded,
        # but overall should reflect rate, not raw count
        assert status["error_rate"] == pytest.approx(0.2, abs=0.05)

    def test_single_latency_record(self, monitor: SystemMonitor) -> None:
        monitor.record_latency("retrieval", 42.0)
        status = monitor.get_health_status()
        assert status["avg_latency_ms"] == pytest.approx(42.0)

    def test_uptime_increases_over_time(self, monitor: SystemMonitor) -> None:
        t0 = monitor.get_health_status()["uptime_seconds"]
        time.sleep(0.05)
        t1 = monitor.get_health_status()["uptime_seconds"]
        assert t1 > t0


# ===========================================================================
# PerformanceBaseline – record and get
# ===========================================================================


class TestPerformanceBaselineRecord:
    def test_record_and_get_single_value(self, baseline: PerformanceBaseline) -> None:
        baseline.record_baseline("latency_ms", 100.0)
        stats = baseline.get_baseline("latency_ms")
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(100.0)

    def test_get_unknown_metric_returns_zeros(self, baseline: PerformanceBaseline) -> None:
        stats = baseline.get_baseline("nonexistent")
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["p50"] == 0.0
        assert stats["p95"] == 0.0

    def test_mean_computed_correctly(self, baseline: PerformanceBaseline) -> None:
        for v in [10.0, 20.0, 30.0]:
            baseline.record_baseline("m", v)
        stats = baseline.get_baseline("m")
        assert stats["mean"] == pytest.approx(20.0)

    def test_std_computed_correctly(self, baseline: PerformanceBaseline) -> None:
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            baseline.record_baseline("m", v)
        stats = baseline.get_baseline("m")
        # population std of that sequence is 2.0
        assert stats["std"] == pytest.approx(2.0, abs=0.01)

    def test_p50_median(self, baseline: PerformanceBaseline) -> None:
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            baseline.record_baseline("m", v)
        stats = baseline.get_baseline("m")
        assert stats["p50"] == pytest.approx(3.0)

    def test_p95_for_known_distribution(self, baseline: PerformanceBaseline) -> None:
        # 100 values 1..100; p95 should be near 95
        for v in range(1, 101):
            baseline.record_baseline("m", float(v))
        stats = baseline.get_baseline("m")
        assert 94.0 <= stats["p95"] <= 96.0

    def test_single_value_std_zero(self, baseline: PerformanceBaseline) -> None:
        baseline.record_baseline("m", 42.0)
        stats = baseline.get_baseline("m")
        assert stats["std"] == 0.0

    def test_multiple_metrics_independent(self, baseline: PerformanceBaseline) -> None:
        baseline.record_baseline("alpha", 1.0)
        baseline.record_baseline("beta", 1000.0)
        assert baseline.get_baseline("alpha")["mean"] == pytest.approx(1.0)
        assert baseline.get_baseline("beta")["mean"] == pytest.approx(1000.0)


# ===========================================================================
# PerformanceBaseline – check_regression
# ===========================================================================


class TestCheckRegression:
    def test_no_baseline_data_not_regressed(self, baseline: PerformanceBaseline) -> None:
        result = baseline.check_regression("missing_metric", 999.0)
        assert result["regressed"] is False

    def test_no_baseline_data_returns_zero_scores(self, baseline: PerformanceBaseline) -> None:
        result = baseline.check_regression("missing_metric", 999.0)
        assert result["z_score"] == 0.0
        assert result["deviation_pct"] == 0.0

    def test_normal_value_not_regressed(self, baseline: PerformanceBaseline) -> None:
        for v in [100.0] * 50:
            baseline.record_baseline("m", v)
        result = baseline.check_regression("m", 101.0)
        assert result["regressed"] is False

    def test_outlier_value_regressed(self, baseline: PerformanceBaseline) -> None:
        # Establish tight baseline
        for v in [100.0] * 50:
            baseline.record_baseline("m", v)
        # Value 3 std above mean triggers regression (z > 2.0)
        result = baseline.check_regression("m", 10_000.0)
        assert result["regressed"] is True

    def test_result_has_required_keys(self, baseline: PerformanceBaseline) -> None:
        baseline.record_baseline("m", 1.0)
        result = baseline.check_regression("m", 1.0)
        assert set(result.keys()) == {
            "regressed",
            "baseline_mean",
            "current",
            "deviation_pct",
            "z_score",
        }

    def test_current_value_preserved(self, baseline: PerformanceBaseline) -> None:
        baseline.record_baseline("m", 50.0)
        result = baseline.check_regression("m", 77.5)
        assert result["current"] == pytest.approx(77.5)

    def test_baseline_mean_preserved(self, baseline: PerformanceBaseline) -> None:
        for v in [10.0, 20.0]:
            baseline.record_baseline("m", v)
        result = baseline.check_regression("m", 15.0)
        assert result["baseline_mean"] == pytest.approx(15.0)

    def test_deviation_pct_above_mean(self, baseline: PerformanceBaseline) -> None:
        for v in [100.0] * 10:
            baseline.record_baseline("m", v)
        result = baseline.check_regression("m", 150.0)
        assert result["deviation_pct"] == pytest.approx(50.0, abs=0.1)

    def test_deviation_pct_below_mean(self, baseline: PerformanceBaseline) -> None:
        for v in [100.0] * 10:
            baseline.record_baseline("m", v)
        result = baseline.check_regression("m", 80.0)
        assert result["deviation_pct"] == pytest.approx(-20.0, abs=0.1)

    def test_zero_std_high_deviation_regressed(self, baseline: PerformanceBaseline) -> None:
        # All identical values -> std=0; current is 50% above mean (>20% threshold)
        for v in [100.0] * 10:
            baseline.record_baseline("m", v)
        result = baseline.check_regression("m", 150.0)
        # deviation_pct = 50, z_score = 50/20 = 2.5 > 2.0 -> regressed
        assert result["regressed"] is True

    def test_zero_std_low_deviation_not_regressed(self, baseline: PerformanceBaseline) -> None:
        for v in [100.0] * 10:
            baseline.record_baseline("m", v)
        result = baseline.check_regression("m", 105.0)
        # deviation_pct = 5, z_score = 5/20 = 0.25 <= 2.0 -> not regressed
        assert result["regressed"] is False


# ===========================================================================
# PerformanceBaseline – save/load round-trip
# ===========================================================================


class TestBaselinePersistence:
    def test_save_and_load_round_trip(self, baseline: PerformanceBaseline, tmp_path) -> None:
        baseline.record_baseline("latency_ms", 10.0)
        baseline.record_baseline("latency_ms", 20.0)
        baseline.record_baseline("tokens", 500.0)

        path = str(tmp_path / "baselines.json")
        baseline.save_baselines(path)

        fresh = PerformanceBaseline()
        fresh.load_baselines(path)

        assert fresh.get_baseline("latency_ms")["mean"] == pytest.approx(15.0)
        assert fresh.get_baseline("tokens")["mean"] == pytest.approx(500.0)

    def test_save_creates_valid_json(self, baseline: PerformanceBaseline, tmp_path) -> None:
        baseline.record_baseline("m", 1.0)
        path = str(tmp_path / "baselines.json")
        baseline.save_baselines(path)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "m" in data
        assert data["m"] == [1.0]

    def test_load_merges_existing_data(self, baseline: PerformanceBaseline, tmp_path) -> None:
        baseline.record_baseline("existing", 99.0)

        other = PerformanceBaseline()
        other.record_baseline("new_metric", 42.0)
        path = str(tmp_path / "baselines.json")
        other.save_baselines(path)

        baseline.load_baselines(path)
        # Both metrics should be present
        assert baseline.get_baseline("existing")["count"] == 1
        assert baseline.get_baseline("new_metric")["count"] == 1

    def test_load_raises_on_missing_file(self, baseline: PerformanceBaseline) -> None:
        with pytest.raises(OSError):
            baseline.load_baselines("/nonexistent/path/baselines.json")

    def test_load_raises_on_invalid_json(self, baseline: PerformanceBaseline, tmp_path) -> None:
        path = str(tmp_path / "bad.json")
        with open(path, "w") as fh:
            fh.write("not valid json{{{")
        with pytest.raises(Exception):
            baseline.load_baselines(path)

    def test_load_raises_on_wrong_schema(self, baseline: PerformanceBaseline, tmp_path) -> None:
        path = str(tmp_path / "bad_schema.json")
        with open(path, "w") as fh:
            json.dump([1, 2, 3], fh)  # top-level list instead of dict
        with pytest.raises(ValueError):
            baseline.load_baselines(path)

    def test_empty_baseline_saves_empty_json(self, baseline: PerformanceBaseline, tmp_path) -> None:
        path = str(tmp_path / "empty.json")
        baseline.save_baselines(path)
        with open(path, "r") as fh:
            data = json.load(fh)
        assert data == {}


# ===========================================================================
# PerformanceBaseline – thread safety
# ===========================================================================


class TestBaselineThreadSafety:
    def test_concurrent_record_no_data_loss(self, baseline: PerformanceBaseline) -> None:
        n_threads = 10
        records_per_thread = 100

        def worker():
            for i in range(records_per_thread):
                baseline.record_baseline("m", float(i))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = baseline.get_baseline("m")
        assert stats["count"] == n_threads * records_per_thread
