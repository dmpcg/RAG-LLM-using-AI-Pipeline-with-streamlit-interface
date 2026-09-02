"""Prepare metrics data for dashboard display.

All functions in this module return plain Python dicts/lists with no
Streamlit dependency.  The caller is responsible for rendering.
"""

from typing import Any, Dict, List

from observability.metrics import MetricsCollector


def _bucket_by_minute(entries: List[Dict[str, Any]], ts_key: str = "timestamp") -> Dict[str, int]:
    """Group entries by UTC minute bucket (ISO-8601 ``HH:MM``).

    Args:
        entries: List of dicts each containing a POSIX timestamp under *ts_key*.
        ts_key: The key holding the POSIX timestamp.

    Returns:
        Dict mapping ``"YYYY-MM-DD HH:MM"`` bucket string to event count,
        ordered by time.
    """
    import datetime

    buckets: Dict[str, int] = {}
    for e in entries:
        ts = e.get(ts_key, 0.0)
        dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).replace(tzinfo=None)
        bucket = dt.strftime("%Y-%m-%d %H:%M")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return dict(sorted(buckets.items()))


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


def _similarity_histogram(similarities: List[float], bins: int = 10) -> List[Dict[str, Any]]:
    """Build a histogram of similarity scores in [0, 1].

    Args:
        similarities: List of cosine similarity floats.
        bins: Number of equal-width bins across [0, 1].

    Returns:
        List of dicts, each with ``bin_start``, ``bin_end``, and ``count``.
    """
    bin_size = 1.0 / bins
    counts = [0] * bins
    for s in similarities:
        idx = min(int(s / bin_size), bins - 1)
        counts[idx] += 1
    result = []
    for i in range(bins):
        result.append(
            {
                "bin_start": round(i * bin_size, 2),
                "bin_end": round((i + 1) * bin_size, 2),
                "count": counts[i],
            }
        )
    return result


def get_dashboard_data(collector: MetricsCollector) -> Dict[str, Any]:
    """Prepare all dashboard-ready data from a ``MetricsCollector``.

    No Streamlit dependency – returns plain dicts/lists suitable for any
    rendering layer.

    Args:
        collector: The ``MetricsCollector`` instance to read from.

    Returns:
        Dict with the following keys:

        - ``query_volume_over_time``: ``{minute_bucket: count}`` for queries.
        - ``latency_percentiles``: ``{p50, p90, p95, p99}`` of
          ``total_duration_ms`` across all recorded queries.
        - ``token_usage_breakdown``: ``{prompt, completion, total}`` token
          sums across all LLM calls.
        - ``query_type_distribution``: ``{query_type: count}`` from retrieval
          records – suitable for a pie chart.
        - ``similarity_score_histogram``: List of 10 histogram bin dicts
          (``bin_start``, ``bin_end``, ``count``).
        - ``cache_hit_rate_over_time``: ``{minute_bucket: hit_rate}`` rolling
          hit rates bucketed by minute.
        - ``summary``: Full summary dict from ``MetricsCollector.get_summary()``.
    """
    queries = collector._get_raw_queries()
    retrievals = collector._get_raw_retrievals()
    llm_calls = collector._get_raw_llm_calls()
    cache_events = collector._get_raw_cache_events()

    # --- Query volume over time ---
    query_volume_over_time = _bucket_by_minute(queries)

    # --- Latency percentiles ---
    latencies = sorted(q["total_duration_ms"] for q in queries)
    latency_percentiles: Dict[str, float] = {
        "p50": round(_percentile(latencies, 50), 2),
        "p90": round(_percentile(latencies, 90), 2),
        "p95": round(_percentile(latencies, 95), 2),
        "p99": round(_percentile(latencies, 99), 2),
    }

    # --- Token usage breakdown ---
    total_prompt = sum(c["prompt_tokens"] for c in llm_calls)
    total_completion = sum(c["completion_tokens"] for c in llm_calls)
    token_usage_breakdown: Dict[str, int] = {
        "prompt": total_prompt,
        "completion": total_completion,
        "total": total_prompt + total_completion,
    }

    # --- Query type distribution ---
    query_type_distribution: Dict[str, int] = {}
    for r in retrievals:
        qt = r["query_type"]
        query_type_distribution[qt] = query_type_distribution.get(qt, 0) + 1

    # --- Similarity score histogram (10 bins) ---
    similarities = [r["avg_similarity"] for r in retrievals]
    similarity_score_histogram = _similarity_histogram(similarities, bins=10)

    # --- Cache hit rate over time (by minute) ---
    import datetime

    minute_hits: Dict[str, int] = {}
    minute_totals: Dict[str, int] = {}
    for e in cache_events:
        ts = e.get("timestamp", 0.0)
        dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).replace(tzinfo=None)
        bucket = dt.strftime("%Y-%m-%d %H:%M")
        minute_totals[bucket] = minute_totals.get(bucket, 0) + 1
        if e["hit"]:
            minute_hits[bucket] = minute_hits.get(bucket, 0) + 1

    cache_hit_rate_over_time: Dict[str, float] = {
        bucket: round(minute_hits.get(bucket, 0) / total, 4) for bucket, total in sorted(minute_totals.items())
    }

    return {
        "query_volume_over_time": query_volume_over_time,
        "latency_percentiles": latency_percentiles,
        "token_usage_breakdown": token_usage_breakdown,
        "query_type_distribution": query_type_distribution,
        "similarity_score_histogram": similarity_score_histogram,
        "cache_hit_rate_over_time": cache_hit_rate_over_time,
        "summary": collector.get_summary(),
    }
