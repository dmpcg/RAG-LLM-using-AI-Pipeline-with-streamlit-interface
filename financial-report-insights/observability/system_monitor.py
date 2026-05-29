"""System-wide health monitoring and performance baseline tracking.

Provides thread-safe aggregation of error rates, per-component latency, alert
rule evaluation, and rolling performance baselines with JSON persistence.
"""

import json
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ERROR_WINDOW_SECONDS = 300  # 5-minute sliding window for error-rate calc
_ALERT_ERROR_RATE_CRITICAL = 10.0  # errors/min
_ALERT_LATENCY_WARNING_MS = 5_000.0
_ALERT_LATENCY_CRITICAL_MS = 10_000.0
_ALERT_CACHE_HIT_RATE_MIN_QUERIES = 100
_ALERT_CACHE_HIT_RATE_WARNING = 0.1

# Thresholds for overall_status
_DEGRADED_ERROR_RATE = 2.0  # errors/min
_UNHEALTHY_ERROR_RATE = 10.0  # errors/min
_DEGRADED_LATENCY_MS = 5_000.0
_UNHEALTHY_LATENCY_MS = 10_000.0

# Baseline regression detection
_REGRESSION_Z_SCORE_THRESHOLD = 2.0  # z-scores above this => regressed


class SystemMonitor:
    """Aggregate system-wide health metrics with thread-safe access.

    Tracks errors and per-component latency, computes health status, and
    evaluates configurable alert rules against a rolling 5-minute window.

    Attributes:
        _start_time: POSIX timestamp when the monitor was created.
    """

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None) -> None:
        """Initialise the monitor.

        Args:
            metrics_collector: Optional ``MetricsCollector`` to pull cache
                hit-rate data from when evaluating alert rules.  If *None* the
                module-level singleton is used only when explicitly needed.
        """
        self._lock = threading.Lock()
        self._metrics_collector = metrics_collector

        # Rolling error log: (timestamp, error_type, message, component)
        self._errors: Deque[Tuple[float, str, str, str]] = deque(maxlen=30_000)

        # Per-component latency log: (timestamp, component, latency_ms)
        self._latencies: Deque[Tuple[float, str, float]] = deque(maxlen=30_000)

        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_error(self, error_type: str, message: str, component: str) -> None:
        """Record an application error.

        Args:
            error_type: Short category string, e.g. ``"timeout"``.
            message: Human-readable error message.
            component: Logical component that raised the error, e.g. ``"llm"``.
        """
        try:
            entry = (time.time(), str(error_type), str(message), str(component))
            with self._lock:
                self._errors.append(entry)
        except Exception as exc:  # pragma: no cover
            logger.warning("SystemMonitor.record_error failed: %s", exc)

    def record_latency(self, component: str, latency_ms: float) -> None:
        """Record a latency observation for a specific component.

        Args:
            component: Logical component name, e.g. ``"retrieval"``.
            latency_ms: Observed latency in milliseconds.
        """
        try:
            entry = (time.time(), str(component), float(latency_ms))
            with self._lock:
                self._latencies.append(entry)
        except Exception as exc:  # pragma: no cover
            logger.warning("SystemMonitor.record_latency failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _errors_in_window(self, now: float) -> List[Tuple[float, str, str, str]]:
        """Return errors within the 5-minute sliding window.

        Must be called with self._lock held, or with a pre-copied snapshot.
        """
        cutoff = now - _ERROR_WINDOW_SECONDS
        return [e for e in self._errors if e[0] >= cutoff]

    def _error_rate_per_min(self, now: float) -> float:
        """Compute errors per minute over the last 5-minute window.

        Caller must hold ``self._lock`` — this method does not acquire it.
        """
        recent = self._errors_in_window(now)
        if not recent:
            return 0.0
        window_min = _ERROR_WINDOW_SECONDS / 60.0
        return len(recent) / window_min

    def _avg_latency_ms(self) -> float:
        """Return mean latency across all recorded observations."""
        with self._lock:
            latencies = list(self._latencies)
        if not latencies:
            return 0.0
        return sum(e[2] for e in latencies) / len(latencies)

    def _component_statuses(self, now: float) -> Dict[str, str]:
        """Derive per-component health status from recent errors and latency."""
        with self._lock:
            recent_errors = self._errors_in_window(now)
            latencies = list(self._latencies)

        # Count errors per component in window
        error_counts: Dict[str, int] = {}
        for _, _, _, comp in recent_errors:
            error_counts[comp] = error_counts.get(comp, 0) + 1

        # Compute avg latency per component
        comp_latency_sum: Dict[str, float] = {}
        comp_latency_cnt: Dict[str, int] = {}
        for _, comp, lat in latencies:
            comp_latency_sum[comp] = comp_latency_sum.get(comp, 0.0) + lat
            comp_latency_cnt[comp] = comp_latency_cnt.get(comp, 0) + 1

        all_components = set(error_counts) | set(comp_latency_sum)
        statuses: Dict[str, str] = {}
        for comp in all_components:
            err_cnt = error_counts.get(comp, 0)
            avg_lat = comp_latency_sum[comp] / comp_latency_cnt[comp] if comp in comp_latency_sum else 0.0
            if err_cnt >= 5 or avg_lat >= _UNHEALTHY_LATENCY_MS:
                statuses[comp] = "unhealthy"
            elif err_cnt >= 1 or avg_lat >= _DEGRADED_LATENCY_MS:
                statuses[comp] = "degraded"
            else:
                statuses[comp] = "healthy"

        return statuses

    # ------------------------------------------------------------------
    # Public status API
    # ------------------------------------------------------------------

    def get_health_status(self) -> Dict[str, Any]:
        """Return a snapshot of overall system health.

        Returns:
            Dict with keys:

            - ``overall_status`` (str): ``"healthy"``, ``"degraded"``, or
              ``"unhealthy"``.
            - ``component_status`` (dict): Per-component status string.
            - ``error_rate`` (float): Errors per minute over the last 5 min.
            - ``avg_latency_ms`` (float): Mean latency across all observations.
            - ``uptime_seconds`` (float): Seconds since monitor was created.
        """
        now = time.time()
        with self._lock:
            error_rate = self._error_rate_per_min(now)

        avg_lat = self._avg_latency_ms()
        component_status = self._component_statuses(now)

        if error_rate >= _UNHEALTHY_ERROR_RATE or avg_lat >= _UNHEALTHY_LATENCY_MS:
            overall = "unhealthy"
        elif error_rate >= _DEGRADED_ERROR_RATE or avg_lat >= _DEGRADED_LATENCY_MS:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "overall_status": overall,
            "component_status": component_status,
            "error_rate": round(error_rate, 4),
            "avg_latency_ms": round(avg_lat, 3),
            "uptime_seconds": round(time.time() - self._start_time, 3),
        }

    # ------------------------------------------------------------------
    # Alert rules
    # ------------------------------------------------------------------

    def check_alert_rules(self) -> List[Dict[str, Any]]:
        """Evaluate all alert rules and return active alerts.

        Rules evaluated:

        1. ``error_rate > 10/min`` -> critical
        2. ``avg_latency > 5000 ms`` -> warning
        3. ``avg_latency > 10000 ms`` -> critical
        4. ``cache_hit_rate < 0.1`` when the collector has seen > 100
           cache events -> warning

        Returns:
            List of alert dicts, each with keys ``level``, ``component``,
            ``message``, and ``timestamp``.
        """
        alerts: List[Dict[str, Any]] = []
        now = time.time()

        with self._lock:
            error_rate = self._error_rate_per_min(now)

        avg_lat = self._avg_latency_ms()

        if error_rate > _ALERT_ERROR_RATE_CRITICAL:
            alerts.append(
                {
                    "level": "critical",
                    "component": "system",
                    "message": (
                        f"Error rate {error_rate:.2f}/min exceeds critical "
                        f"threshold of {_ALERT_ERROR_RATE_CRITICAL}/min"
                    ),
                    "timestamp": now,
                }
            )

        if avg_lat > _ALERT_LATENCY_CRITICAL_MS:
            alerts.append(
                {
                    "level": "critical",
                    "component": "system",
                    "message": (
                        f"Average latency {avg_lat:.1f}ms exceeds critical "
                        f"threshold of {_ALERT_LATENCY_CRITICAL_MS:.0f}ms"
                    ),
                    "timestamp": now,
                }
            )
        elif avg_lat > _ALERT_LATENCY_WARNING_MS:
            alerts.append(
                {
                    "level": "warning",
                    "component": "system",
                    "message": (
                        f"Average latency {avg_lat:.1f}ms exceeds warning "
                        f"threshold of {_ALERT_LATENCY_WARNING_MS:.0f}ms"
                    ),
                    "timestamp": now,
                }
            )

        # Cache hit-rate check (requires a MetricsCollector)
        if self._metrics_collector is not None:
            try:
                cache_stats = self._metrics_collector.get_cache_stats()
                total_events = cache_stats.get("total_events", 0)
                hit_rate = cache_stats.get("hit_rate", 1.0)
                if total_events > _ALERT_CACHE_HIT_RATE_MIN_QUERIES and hit_rate < _ALERT_CACHE_HIT_RATE_WARNING:
                    alerts.append(
                        {
                            "level": "warning",
                            "component": "cache",
                            "message": (
                                f"Cache hit rate {hit_rate:.2%} is below "
                                f"warning threshold of "
                                f"{_ALERT_CACHE_HIT_RATE_WARNING:.0%}"
                            ),
                            "timestamp": now,
                        }
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("SystemMonitor: cache hit-rate alert check failed: %s", exc)

        return alerts

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Return currently active alerts (delegates to ``check_alert_rules``).

        Returns:
            Same structure as :meth:`check_alert_rules`.
        """
        return self.check_alert_rules()


# ---------------------------------------------------------------------------
# PerformanceBaseline
# ---------------------------------------------------------------------------


class PerformanceBaseline:
    """Track rolling performance baselines and detect regressions.

    Baselines are maintained as lists of observed values per metric name.
    Statistical summaries (mean, std, p50, p95) are computed on demand.
    Baselines can be persisted to and restored from JSON files.
    """

    def __init__(self) -> None:
        """Initialise with empty baselines."""
        self._lock = threading.Lock()
        # metric_name -> list of float values
        self._data: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_baseline(self, metric_name: str, value: float) -> None:
        """Add a value to the rolling baseline for *metric_name*.

        Args:
            metric_name: Logical name of the metric, e.g. ``"retrieval_ms"``.
            value: Observed numeric value.
        """
        with self._lock:
            if metric_name not in self._data:
                self._data[metric_name] = []
            self._data[metric_name].append(float(value))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_baseline(self, metric_name: str) -> Dict[str, Any]:
        """Return statistical summary for *metric_name*.

        Args:
            metric_name: The metric to summarise.

        Returns:
            Dict with keys ``mean``, ``std``, ``p50``, ``p95``, and
            ``count``.  All floats are ``0.0`` and count is ``0`` when no
            data has been recorded.
        """
        with self._lock:
            values = list(self._data.get(metric_name, []))

        if not values:
            return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "count": 0}

        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        std = math.sqrt(variance)
        sorted_vals = sorted(values)
        p50 = _percentile(sorted_vals, 50)
        p95 = _percentile(sorted_vals, 95)

        return {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "p50": round(p50, 6),
            "p95": round(p95, 6),
            "count": n,
        }

    def check_regression(self, metric_name: str, current_value: float) -> Dict[str, Any]:
        """Compare *current_value* against the baseline for *metric_name*.

        A regression is flagged when the z-score of *current_value* relative
        to the baseline distribution exceeds ``2.0`` (i.e. the value is more
        than 2 standard deviations above the mean).  When the baseline has
        zero standard deviation the comparison falls back to a 20 % relative
        deviation threshold.

        Args:
            metric_name: The metric to check.
            current_value: The observation to compare against the baseline.

        Returns:
            Dict with keys:

            - ``regressed`` (bool): Whether a regression was detected.
            - ``baseline_mean`` (float): Mean of the baseline distribution.
            - ``current`` (float): The supplied *current_value*.
            - ``deviation_pct`` (float): Relative deviation from mean as a
              percentage (positive = above mean).
            - ``z_score`` (float): Standard-score of *current_value*.
        """
        stats = self.get_baseline(metric_name)
        baseline_mean = stats["mean"]
        baseline_std = stats["std"]

        if stats["count"] == 0:
            # No baseline data – cannot detect regression
            return {
                "regressed": False,
                "baseline_mean": 0.0,
                "current": float(current_value),
                "deviation_pct": 0.0,
                "z_score": 0.0,
            }

        if baseline_mean != 0.0:
            deviation_pct = ((current_value - baseline_mean) / abs(baseline_mean)) * 100.0
        else:
            deviation_pct = 0.0

        if baseline_std > 0.0:
            z_score = (current_value - baseline_mean) / baseline_std
        else:
            # Zero std – use simple relative threshold (>20% above mean)
            z_score = deviation_pct / 20.0 if baseline_mean != 0.0 else 0.0

        regressed = z_score > _REGRESSION_Z_SCORE_THRESHOLD

        return {
            "regressed": regressed,
            "baseline_mean": round(baseline_mean, 6),
            "current": float(current_value),
            "deviation_pct": round(deviation_pct, 4),
            "z_score": round(z_score, 6),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_baselines(self, path: str) -> None:
        """Persist all baselines to a JSON file at *path*.

        Args:
            path: Filesystem path to write.  Parent directories must exist.

        Raises:
            OSError: If the file cannot be written.
        """
        with self._lock:
            data_copy = {k: list(v) for k, v in self._data.items()}

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data_copy, fh, indent=2)

    def load_baselines(self, path: str) -> None:
        """Load baselines from a JSON file at *path*, merging with existing data.

        Existing baseline values are replaced metric-by-metric by the loaded
        data (not merged element-wise).

        Args:
            path: Filesystem path to read.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the file contains invalid JSON or an unexpected
                schema.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a JSON object at the top level, got {type(raw).__name__}")

        loaded: Dict[str, List[float]] = {}
        for key, values in raw.items():
            if not isinstance(values, list):
                raise ValueError(f"Expected list for metric '{key}', got {type(values).__name__}")
            loaded[str(key)] = [float(v) for v in values]

        with self._lock:
            self._data.update(loaded)


# ---------------------------------------------------------------------------
# Internal percentile helper (mirrors dashboard_data._percentile)
# ---------------------------------------------------------------------------


def _percentile(sorted_values: List[float], p: float) -> float:
    """Compute a percentile from a pre-sorted list.

    Args:
        sorted_values: Ascending-sorted list of floats.
        p: Percentile in the range [0, 100].

    Returns:
        Interpolated percentile value, or 0.0 for empty input.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
