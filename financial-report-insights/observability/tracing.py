"""Request tracing for RAG pipeline observability."""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A timed operation within a trace."""

    name: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.monotonic() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def finish(self):
        self.end_time = time.monotonic()


@dataclass
class Trace:
    """A complete request trace with multiple spans."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def total_duration_ms(self) -> float:
        return (time.monotonic() - self._start_time) * 1000

    @contextmanager
    def span(self, name: str, **metadata):
        """Context manager for timing a span."""
        s = Span(name=name, metadata=metadata)
        self.spans.append(s)
        try:
            yield s
        finally:
            s.finish()

    def add_token_counts(self, prompt_tokens: int, completion_tokens: int):
        """Record token counts from LLM response."""
        self.metadata["prompt_tokens"] = self.metadata.get("prompt_tokens", 0) + prompt_tokens
        self.metadata["completion_tokens"] = self.metadata.get("completion_tokens", 0) + completion_tokens
        self.metadata["total_tokens"] = self.metadata.get("prompt_tokens", 0) + self.metadata.get(
            "completion_tokens", 0
        )

    def summary(self) -> Dict:
        """Return a summary dict for logging/metrics."""
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "spans": {s.name: round(s.duration_ms, 2) for s in self.spans},
            "prompt_tokens": self.metadata.get("prompt_tokens", 0),
            "completion_tokens": self.metadata.get("completion_tokens", 0),
            "total_tokens": self.metadata.get("total_tokens", 0),
        }


# Thread-local storage for current trace
_trace_local = threading.local()


def get_current_trace() -> Optional[Trace]:
    """Get the trace for the current thread, if any."""
    return getattr(_trace_local, "trace", None)


@contextmanager
def start_trace(**metadata):
    """Start a new trace for the current request."""
    trace = Trace(metadata=metadata)
    _trace_local.trace = trace
    try:
        yield trace
    finally:
        logger.debug("Trace complete: %s", trace.summary())
        _trace_local.trace = None
