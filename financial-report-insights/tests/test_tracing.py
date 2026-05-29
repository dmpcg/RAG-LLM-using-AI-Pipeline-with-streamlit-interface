"""Tests for the observability tracing module."""

import threading
import time

import pytest

from observability.tracing import Span, Trace, get_current_trace, start_trace


class TestSpan:
    """Tests for the Span dataclass."""

    def test_span_creation(self):
        s = Span(name="test_op")
        assert s.name == "test_op"
        assert s.end_time is None
        assert s.metadata == {}

    def test_span_duration_while_running(self):
        s = Span(name="running")
        time.sleep(0.01)
        assert s.duration_ms > 0

    def test_span_duration_after_finish(self):
        s = Span(name="finished")
        time.sleep(0.01)
        s.finish()
        d = s.duration_ms
        assert d > 0
        # Duration should be stable after finish
        time.sleep(0.01)
        assert s.duration_ms == d

    def test_span_metadata(self):
        s = Span(name="with_meta", metadata={"key": "value"})
        assert s.metadata["key"] == "value"


class TestTrace:
    """Tests for the Trace dataclass."""

    def test_trace_auto_id(self):
        t = Trace()
        assert isinstance(t.trace_id, str)
        assert len(t.trace_id) == 16

    def test_trace_unique_ids(self):
        t1 = Trace()
        t2 = Trace()
        assert t1.trace_id != t2.trace_id

    def test_total_duration(self):
        t = Trace()
        time.sleep(0.01)
        assert t.total_duration_ms > 0

    def test_span_context_manager(self):
        t = Trace()
        with t.span("operation") as s:
            time.sleep(0.01)
        assert len(t.spans) == 1
        assert t.spans[0].name == "operation"
        assert t.spans[0].end_time is not None
        assert t.spans[0].duration_ms > 0

    def test_span_context_manager_with_metadata(self):
        t = Trace()
        with t.span("op", model="llama3", temperature=0.7) as s:
            pass
        assert s.metadata["model"] == "llama3"
        assert s.metadata["temperature"] == 0.7

    def test_multiple_spans(self):
        t = Trace()
        with t.span("first"):
            pass
        with t.span("second"):
            pass
        assert len(t.spans) == 2
        assert t.spans[0].name == "first"
        assert t.spans[1].name == "second"

    def test_span_finishes_on_exception(self):
        t = Trace()
        with pytest.raises(ValueError):
            with t.span("failing"):
                raise ValueError("boom")
        assert t.spans[0].end_time is not None

    def test_add_token_counts(self):
        t = Trace()
        t.add_token_counts(100, 50)
        assert t.metadata["prompt_tokens"] == 100
        assert t.metadata["completion_tokens"] == 50
        assert t.metadata["total_tokens"] == 150

    def test_add_token_counts_accumulates(self):
        t = Trace()
        t.add_token_counts(100, 50)
        t.add_token_counts(200, 80)
        assert t.metadata["prompt_tokens"] == 300
        assert t.metadata["completion_tokens"] == 130
        assert t.metadata["total_tokens"] == 430

    def test_summary_format(self):
        t = Trace()
        with t.span("embed"):
            pass
        t.add_token_counts(10, 5)
        s = t.summary()
        assert "trace_id" in s
        assert "total_duration_ms" in s
        assert "spans" in s
        assert "embed" in s["spans"]
        assert s["prompt_tokens"] == 10
        assert s["completion_tokens"] == 5
        assert s["total_tokens"] == 15

    def test_summary_no_tokens(self):
        t = Trace()
        s = t.summary()
        assert s["prompt_tokens"] == 0
        assert s["completion_tokens"] == 0
        assert s["total_tokens"] == 0


class TestThreadLocal:
    """Tests for thread-local trace management."""

    def test_no_trace_by_default(self):
        assert get_current_trace() is None

    def test_start_trace_sets_current(self):
        with start_trace() as t:
            assert get_current_trace() is t

    def test_start_trace_clears_after(self):
        with start_trace():
            pass
        assert get_current_trace() is None

    def test_start_trace_clears_on_exception(self):
        with pytest.raises(RuntimeError):
            with start_trace():
                raise RuntimeError("fail")
        assert get_current_trace() is None

    def test_start_trace_with_metadata(self):
        with start_trace(user="test", query="hello") as t:
            assert t.metadata["user"] == "test"
            assert t.metadata["query"] == "hello"

    def test_thread_isolation(self):
        """Traces in different threads should be independent."""
        results = {}
        barrier = threading.Barrier(2)

        def worker(name):
            with start_trace() as t:
                t.metadata["thread"] = name
                barrier.wait(timeout=5)
                current = get_current_trace()
                results[name] = current.metadata["thread"] if current else None

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert results["A"] == "A"
        assert results["B"] == "B"

    def test_no_cross_thread_leakage(self):
        """A trace started in one thread should not be visible in another."""
        seen_in_child = []

        def child():
            seen_in_child.append(get_current_trace())

        with start_trace():
            t = threading.Thread(target=child)
            t.start()
            t.join(timeout=5)

        assert seen_in_child[0] is None

    def test_trace_logging(self, caplog):
        """start_trace should log the summary on completion (at DEBUG level)."""
        with caplog.at_level("DEBUG", logger="observability.tracing"):
            with start_trace():
                pass
        assert "Trace complete" in caplog.text
