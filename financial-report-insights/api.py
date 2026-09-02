"""
FastAPI REST + SSE API layer for the RAG-LLM Financial Insights engine.
Wraps SimpleRAG, CharlieAnalyzer, and health checks as HTTP endpoints.
"""

import asyncio
import hmac
import io
import logging
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from config import settings
from healthcheck import get_health_status
from local_llm import LLMConnectionError
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.max_query_length)
    top_k: int = Field(default=settings.top_k, ge=1, le=settings.max_top_k)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    document_count: int


class AnalyzeRequest(BaseModel):
    financial_data: Dict[str, Any]

    @field_validator("financial_data")
    @classmethod
    def validate_field_count(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if len(v) > settings.max_financial_fields:
            raise ValueError(f"Too many fields ({len(v)}); max {settings.max_financial_fields}.")
        return v


class AnalyzeResponse(BaseModel):
    """Typed response for /analyze endpoint."""

    executive_summary: str
    sections: Dict[str, str]
    generated_at: str
    liquidity_ratios: Optional[Dict[str, Optional[float]]] = None
    profitability_ratios: Optional[Dict[str, Optional[float]]] = None
    leverage_ratios: Optional[Dict[str, Optional[float]]] = None
    efficiency_ratios: Optional[Dict[str, Optional[float]]] = None
    composite_health_score: Optional[float] = None
    composite_health_grade: Optional[str] = None


class DocumentInfo(BaseModel):
    source: str
    type: str
    content_preview: str
    section_type: Optional[str] = None
    chunk_level: Optional[str] = None
    has_parent: bool = False


# ---------------------------------------------------------------------------
# Application lifespan – lazy-init RAG on first real request
# ---------------------------------------------------------------------------

_rag_instance = None
_rag_lock = threading.Lock()

# Module-level exporter references — populated eagerly at lifespan startup.
# Using None sentinel so tests can detect missing eager import.
FinancialExcelExporter = None  # type: ignore[assignment]
FinancialPDFExporter = None  # type: ignore[assignment]


def _get_rag():
    """Lazy-initialise the RAG singleton (avoids slow startup when importing)."""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                import os

                from app_local import SimpleRAG

                _rag_instance = SimpleRAG(
                    docs_folder="./documents",
                    llm_model=os.getenv("OLLAMA_MODEL", settings.llm_model),
                    embedding_model=os.getenv("EMBEDDING_MODEL", settings.embedding_model),
                )
                logger.info("RAG engine initialised (%d chunks)", len(_rag_instance.documents))
    return _rag_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    global FinancialExcelExporter, FinancialPDFExporter
    logger.info("FastAPI starting up")
    from config import validate_settings

    errors, warnings = validate_settings()
    for w in warnings:
        logger.warning("[config] %s", w)
    if errors:
        for e in errors:
            logger.error("[config] %s", e)
        raise RuntimeError(f"Configuration validation failed: {'; '.join(errors)}")
    # Eagerly import exporters at startup so the first export request pays no
    # per-request import overhead and import errors surface immediately (P1-A6).
    from export_pdf import FinancialPDFExporter as _FPE
    from export_xlsx import FinancialExcelExporter as _FXE

    FinancialExcelExporter = _FXE
    FinancialPDFExporter = _FPE
    logger.info("Exporter classes loaded: %s, %s", _FXE.__name__, _FPE.__name__)
    yield
    logger.info("FastAPI shutting down")


# ---------------------------------------------------------------------------
# Optional API-key authentication
# ---------------------------------------------------------------------------

# Routes that never require an API key (monitoring / liveness).
_AUTH_EXEMPT_PATHS = {"/health", "/metrics"}


def require_api_key(request: Request) -> None:
    """Enforce optional X-API-Key auth (disabled when settings.api_key is empty).

    Backward-compatible: when no key is configured the dependency is a no-op so
    all existing callers and tests pass.  When a key IS set, every route except
    /health and /metrics must present a matching X-API-Key header.  Comparison
    uses hmac.compare_digest to avoid timing leaks.
    """
    expected = settings.api_key
    if not expected or request.url.path in _AUTH_EXEMPT_PATHS:
        return
    provided = request.headers.get("X-API-Key", "")
    if not (provided and hmac.compare_digest(provided, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial Report Insights API",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Prometheus metrics middleware - always added so it is available when the
# endpoint flag is enabled; it records nothing sensitive and is low-overhead.
from observability.metrics import MetricsMiddleware  # noqa: E402

app.add_middleware(MetricsMiddleware)


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (per-IP, sliding window)
# ---------------------------------------------------------------------------

_RATE_WINDOW = 60  # seconds
_RATE_LIMIT = 60  # max requests per window
_rate_log: Dict[str, List[float]] = defaultdict(list)
_rate_lock = threading.Lock()


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add standard security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next):
    """Reject request bodies exceeding the configured size limit.

    Checks both Content-Length header (fast reject) and actual body bytes
    (defense against chunked transfer encoding bypass).
    """
    max_bytes = settings.max_request_body_bytes
    content_length = request.headers.get("content-length")
    try:
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid Content-Length header."},
        )

    # Guard against chunked TE that omits Content-Length
    if request.method in ("POST", "PUT", "PATCH") and not content_length:
        body = await request.body()
        if len(body) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )

    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-IP request rate limit. Skips /health for monitoring."""
    if request.url.path == "/health":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        cutoff = now - _RATE_WINDOW
        _rate_log[client_ip] = [t for t in _rate_log[client_ip] if t > cutoff]
        if len(_rate_log[client_ip]) >= _RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        _rate_log[client_ip].append(now)
        # Cross-IP sweep: evict every IP whose entire timestamp list has expired.
        # Runs on each request (inside the lock) but iterates a snapshot of keys
        # so it is safe to mutate the dict.  This is the only path that evicts
        # an IP that has stopped sending -- the single-IP prune above cannot do
        # it (the defaultdict re-creates the key on touch).
        # Sweep is bounded: iterating N keys is O(N) but N is capped by the
        # 10K hard cap below, so worst case is O(10K) per request.
        stale = [ip for ip, ts in list(_rate_log.items()) if not [t for t in ts if t > cutoff]]
        for ip in stale:
            del _rate_log[ip]
        # Hard cap: if dict exceeds 10K IPs, clear entirely (DoS defense)
        if len(_rate_log) > 10_000:
            _rate_log.clear()
    return await call_next(request)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Service health check."""
    status = await asyncio.to_thread(get_health_status)
    code = 200 if status["healthy"] else 503
    if code == 503:
        raise HTTPException(status_code=503, detail=status)
    return status


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus exposition endpoint.

    Returns HTTP 404 when ``settings.enable_metrics_endpoint`` is False (the
    default) so the endpoint is hidden in environments that do not opt-in.
    The flag is checked at request time so tests can toggle it without
    rebuilding the app.
    """
    if not settings.enable_metrics_endpoint:
        raise HTTPException(status_code=404, detail="Not found.")
    from fastapi.responses import Response as _Response

    from observability.metrics import render_latest

    data, content_type = render_latest()
    return _Response(content=data, media_type=content_type)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Synchronous RAG question answering."""
    rag = _get_rag()

    # Check circuit breaker early
    if hasattr(rag.llm, "circuit_state") and rag.llm.circuit_state == "OPEN":
        raise HTTPException(
            status_code=503,
            detail="LLM circuit breaker is OPEN – service temporarily unavailable.",
        )

    try:
        relevant_docs = await asyncio.to_thread(rag.retrieve, req.text, top_k=req.top_k)
        answer = await asyncio.to_thread(rag.answer, req.text, relevant_docs)
        sources = list({doc.get("source", "unknown") for doc in relevant_docs})
        return QueryResponse(
            answer=answer,
            sources=sources,
            document_count=len(relevant_docs),
        )
    except LLMConnectionError as exc:
        logger.warning("LLM query failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable.",
        ) from exc


@app.post("/query-stream")
async def query_stream(req: QueryRequest):
    """SSE streaming RAG question answering."""
    rag = _get_rag()

    if hasattr(rag.llm, "circuit_state") and rag.llm.circuit_state == "OPEN":
        raise HTTPException(
            status_code=503,
            detail="LLM circuit breaker is OPEN – service temporarily unavailable.",
        )

    relevant_docs = await asyncio.to_thread(rag.retrieve, req.text, top_k=req.top_k)

    async def event_generator():
        """Yield SSE events without blocking the event loop.

        The sync ``answer_stream`` generator performs blocking I/O per chunk,
        so we call ``next()`` in a thread for each iteration.

        Each chunk fetch is wrapped in asyncio.wait_for to bound CLIENT-SIDE
        latency (P1-C1).  NOTE: wait_for cancels only the awaiting coroutine;
        the to_thread worker blocked on a C-level socket read is NOT cancelled
        here -- that reclamation is handled by the finite httpx read timeout in
        local_llm.py (WP-LLM P1-C1-stream-timeout).
        """
        done_sentinel = object()
        it = iter(rag.answer_stream(req.text, relevant_docs))
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        asyncio.to_thread(next, it, done_sentinel),
                        timeout=settings.llm_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning("SSE chunk timed out after %ss", settings.llm_timeout_seconds)
                    yield {"event": "error", "data": "Stream timed out."}
                    return
                if chunk is done_sentinel:
                    break
                yield {"data": chunk}
            yield {"event": "done", "data": ""}
        except LLMConnectionError as exc:
            logger.warning("LLM streaming failed: %s", exc)
            yield {"event": "error", "data": "LLM service temporarily unavailable."}

    return EventSourceResponse(event_generator())


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Run CharlieAnalyzer on provided financial data."""
    rag = _get_rag()

    if not rag.charlie_analyzer:
        raise HTTPException(status_code=501, detail="Financial analyzer not available.")

    try:
        data = _parse_financial_data(req.financial_data)
        analysis = await asyncio.to_thread(rag.charlie_analyzer.analyze, data)
        report = await asyncio.to_thread(rag.charlie_analyzer.generate_report, data)

        # Extract ratio data — handle both AnalysisResults and legacy dict
        def _to_ratio_dict(obj):
            if isinstance(obj, dict):
                return obj or None
            if obj is not None and hasattr(obj, "to_dict") and callable(obj.to_dict):
                try:
                    d = obj.to_dict()
                    return d if isinstance(d, dict) else None
                except Exception:
                    return None
            return None

        liq = _to_ratio_dict(analysis.get("liquidity_ratios"))
        prof = _to_ratio_dict(analysis.get("profitability_ratios"))
        lev = _to_ratio_dict(analysis.get("leverage_ratios"))
        eff = _to_ratio_dict(analysis.get("efficiency_ratios"))
        health = analysis.get("composite_health")
        h_score = getattr(health, "overall_score", None)
        h_grade = getattr(health, "grade", None)

        return AnalyzeResponse(
            executive_summary=report.executive_summary,
            sections=report.sections,
            generated_at=report.generated_at,
            liquidity_ratios=liq,
            profitability_ratios=prof,
            leverage_ratios=lev,
            efficiency_ratios=eff,
            composite_health_score=h_score if isinstance(h_score, (int, float)) else None,
            composite_health_grade=h_grade if isinstance(h_grade, str) else None,
        )
    except LLMConnectionError as exc:
        logger.warning("LLM unavailable during analyze: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.warning("Analyze request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=422,
            detail="Could not process the provided financial data.",
        ) from exc


# ---------------------------------------------------------------------------
# Graph-specific endpoints (Phase 3)
# ---------------------------------------------------------------------------


class RatioData(BaseModel):
    """A single ratio from the graph store."""

    name: str
    value: Optional[float] = None
    category: str = ""


class ScoreData(BaseModel):
    """A single scoring model result from the graph store."""

    model: str
    value: Optional[float] = None
    grade: str = ""


class PeriodContext(BaseModel):
    period_label: str
    ratios: List[RatioData]
    scores: List[ScoreData]


class RatioEntry(BaseModel):
    name: str
    value: Optional[float]
    category: str


def _require_graph_store():
    """Return the graph store or raise 501."""
    rag = _get_rag()
    store = getattr(rag, "_graph_store", None)
    if store is None:
        raise HTTPException(status_code=501, detail="Graph store not configured (NEO4J_URI not set).")
    return store


@app.get("/graph/context/{period_label}", response_model=PeriodContext)
async def graph_context(period_label: str = Path(..., max_length=100)):
    """Return all ratios and scores for a fiscal period."""
    store = _require_graph_store()
    raw_ratios = await asyncio.to_thread(store.ratios_by_period_label, period_label)
    raw_scores = await asyncio.to_thread(store.scores_by_period_label, period_label)
    return PeriodContext(
        period_label=period_label,
        ratios=[
            RatioData(name=r.get("name", ""), value=r.get("value"), category=r.get("category", "")) for r in raw_ratios
        ],
        scores=[
            ScoreData(model=s.get("model", ""), value=s.get("value"), grade=s.get("grade", "")) for s in raw_scores
        ],
    )


@app.get("/graph/ratios/{period_label}", response_model=List[RatioEntry])
async def graph_ratios(
    period_label: str = Path(..., max_length=100), category: Optional[str] = Query(default=None, max_length=100)
):
    """Return ratios for a fiscal period with optional category filter."""
    store = _require_graph_store()
    ratios = await asyncio.to_thread(store.ratios_by_period_label, period_label)
    if category:
        ratios = [r for r in ratios if r.get("category", "").lower() == category.lower()]
    return [RatioEntry(name=r["name"], value=r.get("value"), category=r.get("category", "")) for r in ratios]


# ---------------------------------------------------------------------------
# Multi-document comparison (Phase 4)
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    period_labels: List[str] = Field(..., min_length=2, max_length=10)

    @field_validator("period_labels")
    @classmethod
    def validate_label_lengths(cls, v: List[str]) -> List[str]:
        for label in v:
            if len(label) > 100:
                raise ValueError(f"Period label too long ({len(label)} chars); max 100.")
        return v


class PeriodDelta(BaseModel):
    ratio_name: str
    periods: Dict[str, Optional[float]]
    delta: Optional[float] = None


class TrendDataPoint(BaseModel):
    """A single ratio value at a specific period."""

    ratio_name: str
    period: str
    value: Optional[float] = None


class CompareResponse(BaseModel):
    periods_compared: List[str]
    improvements: List[str]
    deteriorations: List[str]
    deltas: List[PeriodDelta]
    graph_trend_data: Optional[List[TrendDataPoint]] = None
    summary: str


@app.post("/compare", response_model=CompareResponse)
async def compare_periods(req: CompareRequest):
    """Compare financial metrics across multiple fiscal periods."""
    rag = _get_rag()
    store = getattr(rag, "_graph_store", None)

    graph_trend_data = None
    deltas: List[PeriodDelta] = []
    improvements: List[str] = []
    deteriorations: List[str] = []

    # Graph path: query cross-period trends
    if store is not None:
        try:
            raw_trends = await asyncio.to_thread(store.cross_period_ratio_trend, req.period_labels)
            if raw_trends:
                graph_trend_data = [
                    TrendDataPoint(
                        ratio_name=row.get("ratio_name", ""),
                        period=row.get("period", ""),
                        value=row.get("value"),
                    )
                    for row in raw_trends
                ]
                # Build deltas from graph data
                ratio_periods: Dict[str, Dict[str, Optional[float]]] = {}
                for row in raw_trends:
                    rname = row.get("ratio_name", "")
                    if not rname:
                        continue
                    ratio_periods.setdefault(rname, {})
                    ratio_periods[rname][row.get("period", "")] = row.get("value")

                for rname, periods in ratio_periods.items():
                    values = [periods.get(p) for p in req.period_labels]
                    first_val = next((v for v in values if v is not None), None)
                    last_val = next((v for v in reversed(values) if v is not None), None)
                    delta = (last_val - first_val) if first_val is not None and last_val is not None else None
                    deltas.append(PeriodDelta(ratio_name=rname, periods=periods, delta=delta))
                    if delta is not None:
                        if delta > 0:
                            improvements.append(f"{rname}: +{delta:.4f}")
                        elif delta < 0:
                            deteriorations.append(f"{rname}: {delta:.4f}")
        except Exception as exc:
            # Intentional graceful degradation: graph is optional, fall back to in-memory
            logger.debug("Graph comparison failed, using in-memory: %s", exc)

    # In-memory fallback: use cached FinancialData
    if not deltas:
        # Quick-return guard (P1-A4 / D9): when no documents have been ingested
        # there is nothing to compare; skip the expensive context scan entirely.
        # NOTE: guard is on rag.documents (the ingestion list), NOT on
        # _period_financial_data which is lazily populated BY the very scan we
        # want to skip (bailing on empty period_data would regress document-
        # loaded analyzers whose period cache is cold on the first request).
        if not getattr(rag, "documents", None):
            return CompareResponse(
                periods_compared=req.period_labels,
                improvements=[],
                deteriorations=[],
                deltas=[],
                graph_trend_data=graph_trend_data,
                summary=(f"Compared {len(req.period_labels)} periods: 0 improvements, 0 deteriorations."),
            )

        period_data = getattr(rag, "_period_financial_data", {})
        # Ensure financial analysis context is computed
        if not period_data:
            await asyncio.to_thread(rag._get_financial_analysis_context)
            period_data = getattr(rag, "_period_financial_data", {})

        if period_data and rag.charlie_analyzer:
            try:
                from ratio_framework import run_all_ratios

                for label in req.period_labels:
                    fd = period_data.get(label)
                    if fd:
                        results = run_all_ratios(fd)
                        for key, result in results.items():
                            if result.value is not None:
                                found = False
                                for d in deltas:
                                    if d.ratio_name == result.name:
                                        d.periods[label] = result.value
                                        found = True
                                        break
                                if not found:
                                    deltas.append(
                                        PeriodDelta(
                                            ratio_name=result.name,
                                            periods={label: result.value},
                                        )
                                    )

                # Compute deltas between first and last period
                for d in deltas:
                    values = [d.periods.get(p) for p in req.period_labels]
                    first_val = next((v for v in values if v is not None), None)
                    last_val = next((v for v in reversed(values) if v is not None), None)
                    if first_val is not None and last_val is not None:
                        d.delta = last_val - first_val
                        if d.delta > 0:
                            improvements.append(f"{d.ratio_name}: +{d.delta:.4f}")
                        elif d.delta < 0:
                            deteriorations.append(f"{d.ratio_name}: {d.delta:.4f}")
            except Exception as exc:
                # Intentional graceful degradation: return empty deltas rather than 500
                logger.debug("In-memory comparison failed: %s", exc)

    n_improvements = len(improvements)
    n_deteriorations = len(deteriorations)
    summary = (
        f"Compared {len(req.period_labels)} periods: {n_improvements} improvements, {n_deteriorations} deteriorations."
    )

    return CompareResponse(
        periods_compared=req.period_labels,
        improvements=improvements,
        deteriorations=deteriorations,
        deltas=deltas,
        graph_trend_data=graph_trend_data,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------


class ExportRequest(BaseModel):
    financial_data: Dict[str, Any]
    company_name: str = Field(default="", max_length=200)

    @field_validator("financial_data")
    @classmethod
    def validate_field_count(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if len(v) > settings.max_financial_fields:
            raise ValueError(f"Too many fields ({len(v)}); max {settings.max_financial_fields}.")
        return v


@app.post("/export/xlsx")
async def export_xlsx(req: ExportRequest):
    """Export financial analysis as Excel workbook."""
    data = _parse_financial_data(req.financial_data)

    rag = _get_rag()
    if not rag.charlie_analyzer:
        raise HTTPException(status_code=501, detail="Financial analyzer not available.")

    try:
        analysis = await asyncio.to_thread(rag.charlie_analyzer.analyze, data)
        report = await asyncio.to_thread(rag.charlie_analyzer.generate_report, data)

        exporter = FinancialExcelExporter()
        xlsx_bytes = exporter.export_full_report(data, analysis, report=report)
    except LLMConnectionError as exc:
        logger.warning("LLM unavailable during XLSX export: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.warning("XLSX export failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Could not generate XLSX export.",
        ) from exc

    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="financial_report.xlsx"'},
    )


@app.post("/export/pdf")
async def export_pdf(req: ExportRequest):
    """Export financial analysis as PDF report."""
    data = _parse_financial_data(req.financial_data)

    rag = _get_rag()
    if not rag.charlie_analyzer:
        raise HTTPException(status_code=501, detail="Financial analyzer not available.")

    try:
        analysis = await asyncio.to_thread(rag.charlie_analyzer.analyze, data)
        report = await asyncio.to_thread(rag.charlie_analyzer.generate_report, data)

        exporter = FinancialPDFExporter()
        pdf_bytes = exporter.export_full_report(data, analysis, report=report)
    except LLMConnectionError as exc:
        logger.warning("LLM unavailable during PDF export: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.warning("PDF export failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Could not generate PDF export.",
        ) from exc

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="financial_report.pdf"'},
    )


# ---------------------------------------------------------------------------
# Portfolio endpoints (Phase 7)
# ---------------------------------------------------------------------------


_MAX_PORTFOLIO_COMPANIES = 50


class PortfolioRequest(BaseModel):
    companies: Dict[str, Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Map of company_name -> financial_data dict (at least 1)",
    )

    @field_validator("companies")
    @classmethod
    def limit_company_count(cls, v):
        if len(v) > _MAX_PORTFOLIO_COMPANIES:
            raise ValueError(f"Too many companies ({len(v)}). Maximum is {_MAX_PORTFOLIO_COMPANIES}.")
        # Validate field count per company
        for name, data in v.items():
            if len(data) > settings.max_financial_fields:
                raise ValueError(
                    f"Company '{name}' has too many fields ({len(data)}); max {settings.max_financial_fields}."
                )
        return v


class PortfolioResponse(BaseModel):
    num_companies: int
    avg_health_score: float
    diversification_score: int
    diversification_grade: str
    risk_level: str
    risk_flags: List[str]
    strongest: str
    weakest: str
    summary: str


class CorrelationResponse(BaseModel):
    company_names: List[str]
    ratio_names: List[str]
    matrix: List[List[float]]
    avg_correlation: float
    interpretation: str


_portfolio_lock = threading.Lock()


def _get_portfolio_analyzer():
    """Return a module-level singleton PortfolioAnalyzer (thread-safe)."""
    if not hasattr(_get_portfolio_analyzer, "_inst"):
        with _portfolio_lock:
            if not hasattr(_get_portfolio_analyzer, "_inst"):
                from portfolio_analyzer import PortfolioAnalyzer

                _get_portfolio_analyzer._inst = PortfolioAnalyzer()
    return _get_portfolio_analyzer._inst


_compliance_lock = threading.Lock()


def _get_compliance_scorer():
    """Return a module-level singleton ComplianceScorer (thread-safe)."""
    if not hasattr(_get_compliance_scorer, "_inst"):
        with _compliance_lock:
            if not hasattr(_get_compliance_scorer, "_inst"):
                from compliance_scorer import ComplianceScorer

                _get_compliance_scorer._inst = ComplianceScorer()
    return _get_compliance_scorer._inst


def _parse_financial_data(raw: Dict[str, Any]) -> "FinancialData":  # noqa: F821 — forward ref; FinancialData imported in-function
    """Parse a raw dict into FinancialData, filtering unknown fields."""
    from financial_analyzer import FinancialData  # local import: forward ref for type checker

    if not hasattr(_parse_financial_data, "_fields"):
        _parse_financial_data._fields = frozenset(FinancialData.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in _parse_financial_data._fields}
    return FinancialData(**filtered)


@app.post("/portfolio/analyze", response_model=PortfolioResponse)
async def portfolio_analyze(req: PortfolioRequest):
    """Run full portfolio analysis across multiple companies."""
    try:
        companies = {name: _parse_financial_data(raw) for name, raw in req.companies.items()}
    except Exception as exc:
        logger.warning("Invalid financial data in request: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Invalid financial data format.",
        ) from exc

    pa = _get_portfolio_analyzer()
    try:
        report = await asyncio.to_thread(pa.full_portfolio_analysis, companies)
    except LLMConnectionError as exc:
        logger.warning("LLM unavailable during portfolio analyze: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable.",
        ) from exc

    # D3: best-effort graph-store persist (never fails the request).
    rag = _get_rag()
    store = getattr(rag, "_graph_store", None)
    if store is not None:
        try:
            portfolio_name = ", ".join(sorted(companies.keys()))
            await asyncio.to_thread(
                store.store_portfolio_analysis,
                portfolio_name,
                list(companies.keys()),
                report.risk_summary,
                report.diversification.overall_score,
            )
        except Exception as _exc:
            logger.debug("store_portfolio_analysis failed (best-effort): %s", _exc)

    return PortfolioResponse(
        num_companies=report.num_companies,
        avg_health_score=report.risk_summary.avg_health_score,
        diversification_score=report.diversification.overall_score,
        diversification_grade=report.diversification.grade,
        risk_level=report.risk_summary.overall_risk_level,
        risk_flags=report.risk_summary.risk_flags,
        strongest=report.risk_summary.strongest_company,
        weakest=report.risk_summary.weakest_company,
        summary=report.summary,
    )


@app.post("/portfolio/correlation", response_model=CorrelationResponse)
async def portfolio_correlation(req: PortfolioRequest):
    """Compute correlation matrix across portfolio companies."""
    try:
        companies = {name: _parse_financial_data(raw) for name, raw in req.companies.items()}
    except Exception as exc:
        logger.warning("Invalid financial data in request: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Invalid financial data format.",
        ) from exc

    pa = _get_portfolio_analyzer()
    try:
        corr = await asyncio.to_thread(pa.correlation_matrix, companies)
    except LLMConnectionError as exc:
        logger.warning("LLM unavailable during portfolio correlation: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable.",
        ) from exc

    # D3: best-effort graph-store persist (never fails the request).
    rag = _get_rag()
    store = getattr(rag, "_graph_store", None)
    if store is not None:
        try:
            from portfolio_analyzer import PortfolioRiskSummary

            minimal_risk = PortfolioRiskSummary(
                num_companies=len(corr.company_names),
                overall_risk_level="unknown",
            )
            portfolio_name = ", ".join(sorted(companies.keys()))
            await asyncio.to_thread(
                store.store_portfolio_analysis,
                portfolio_name,
                list(corr.company_names),
                minimal_risk,
                0,
            )
        except Exception as _exc:
            logger.debug("store_portfolio_analysis (correlation) failed (best-effort): %s", _exc)

    return CorrelationResponse(
        company_names=corr.company_names,
        ratio_names=corr.ratio_names,
        matrix=corr.matrix,
        avg_correlation=corr.avg_correlation,
        interpretation=corr.interpretation,
    )


# ---------------------------------------------------------------------------
# Compliance endpoints (Phase 7)
# ---------------------------------------------------------------------------


class ComplianceResponse(BaseModel):
    sox_risk: str
    sox_score: int
    sec_score: int
    sec_grade: str
    regulatory_pct: Optional[float]
    regulatory_pass: int
    regulatory_fail: int
    audit_risk: str
    audit_score: int
    audit_grade: str
    going_concern: bool
    summary: str


@app.post("/compliance/analyze", response_model=ComplianceResponse)
async def compliance_analyze(req: AnalyzeRequest):
    """Run full compliance analysis on financial data."""
    try:
        data = _parse_financial_data(req.financial_data)
    except Exception as exc:
        logger.warning("Invalid financial data in request: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Invalid financial data format.",
        ) from exc

    cs = _get_compliance_scorer()
    try:
        report = await asyncio.to_thread(cs.full_compliance_report, data)
    except LLMConnectionError as exc:
        logger.warning("LLM unavailable during compliance analyze: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable.",
        ) from exc

    # D4: best-effort graph-store persist (never fails the request).
    rag = _get_rag()
    store = getattr(rag, "_graph_store", None)
    if store is not None:
        try:
            company_name = str(req.financial_data.get("company_name", "unknown"))
            await asyncio.to_thread(
                store.store_compliance_report,
                company_name,
                report,
            )
        except Exception as _exc:
            logger.debug("store_compliance_report failed (best-effort): %s", _exc)

    return ComplianceResponse(
        sox_risk=report.sox.overall_risk,
        sox_score=report.sox.risk_score,
        sec_score=report.sec.disclosure_score,
        sec_grade=report.sec.grade,
        regulatory_pct=report.regulatory.compliance_pct,
        regulatory_pass=report.regulatory.pass_count,
        regulatory_fail=report.regulatory.fail_count,
        audit_risk=report.audit_risk.risk_level,
        audit_score=report.audit_risk.score,
        audit_grade=report.audit_risk.grade,
        going_concern=report.audit_risk.going_concern_risk,
        summary=report.summary,
    )


@app.post("/compliance/sox")
async def compliance_sox(req: AnalyzeRequest):
    """Run SOX compliance check only."""
    try:
        data = _parse_financial_data(req.financial_data)
    except Exception as exc:
        logger.warning("Invalid financial data in request: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Invalid financial data format.",
        ) from exc

    cs = _get_compliance_scorer()
    sox = await asyncio.to_thread(cs.sox_compliance, data)
    return {
        "overall_risk": sox.overall_risk,
        "risk_score": sox.risk_score,
        "flags": sox.flags,
        "material_weakness_indicators": sox.material_weakness_indicators,
        "significant_deficiency_indicators": sox.significant_deficiency_indicators,
        "checks_performed": sox.checks_performed,
        "checks_passed": sox.checks_passed,
    }


@app.post("/compliance/regulatory")
async def compliance_regulatory(req: AnalyzeRequest):
    """Run regulatory threshold check only."""
    try:
        data = _parse_financial_data(req.financial_data)
    except Exception as exc:
        logger.warning("Invalid financial data in request: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Invalid financial data format.",
        ) from exc

    cs = _get_compliance_scorer()
    reg = await asyncio.to_thread(cs.regulatory_ratios, data)
    return {
        "pass_count": reg.pass_count,
        "fail_count": reg.fail_count,
        "compliance_pct": reg.compliance_pct,
        "critical_failures": reg.critical_failures,
        "thresholds": [
            {
                "rule_name": t.rule_name,
                "framework": t.framework,
                "metric_name": t.metric_name,
                "current_value": t.current_value,
                "threshold_value": t.threshold_value,
                "passes": t.passes,
                "severity": t.severity,
            }
            for t in reg.thresholds_checked
        ],
    }


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    limit: int = Query(100, ge=1, description="Maximum number of results to return."),
    offset: int = Query(0, ge=0, description="Number of results to skip."),
    source: str | None = Query(None, description="Filter by exact source filename."),
):
    """List indexed document chunks with pipeline metadata.

    Supports optional pagination (limit/offset) and source filtering.
    """
    rag = _get_rag()
    results = []
    seen = set()
    for doc in rag.documents:
        doc_source = doc.get("source", "unknown")
        if doc_source in seen:
            continue
        seen.add(doc_source)
        if source is not None and doc_source != source:
            continue
        meta = doc.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        results.append(
            DocumentInfo(
                source=doc_source,
                type=doc.get("type", "unknown"),
                content_preview=doc.get("content", "")[:200],
                section_type=meta.get("section_type"),
                chunk_level=meta.get("chunk_level"),
                has_parent=bool(meta.get("parent_id")),
            )
        )
    return results[offset : offset + limit]
