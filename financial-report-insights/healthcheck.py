"""
Startup validation and health check utilities.
Validates Ollama connectivity, model availability, and filesystem readiness.

WS-1 P0-2 (2026-05-07): Each preflight stage is wrapped in a 3-second
concurrent.futures timeout, and aggregated `get_health_status()` results are
cached for 10 seconds, so a Neo4j or Ollama partition cannot block /health
beyond the per-stage budget.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


# Per-stage timeout for each preflight check (P0-2).  Health probes must
# return within 20s end-to-end even under partition; with 6 stages at 3s each
# we cap at 18s + small overhead.
_STAGE_TIMEOUT_SECONDS = 3.0

# Cache aggregated health results for this many seconds to avoid hammering
# Neo4j/Ollama on rapid-fire health probes from container orchestrators.
_HEALTH_CACHE_TTL_SECONDS = 10.0

_health_cache_lock = threading.Lock()
_health_cache: Dict[str, object] = {"value": None, "ts": 0.0}


def _run_with_timeout(
    fn: Callable[[], Dict[str, str]],
    name: str,
    timeout: float = _STAGE_TIMEOUT_SECONDS,
) -> Dict[str, str]:
    """Execute a healthcheck stage with a hard wall-clock timeout.

    Returns a dict with status="error" and a timeout detail if the stage
    does not complete within `timeout` seconds.  This prevents a single
    slow dependency (Neo4j unreachable, Ollama hanging) from blocking the
    entire /health endpoint.

    Uses a *daemon* thread so we do not block the calling request thread
    waiting for the stage to drain.  ``ThreadPoolExecutor`` cannot be
    used here because its ``__exit__`` joins workers — defeating the
    timeout for I/O-bound checks that ignore cancellation.
    """
    result_box: Dict[str, Dict[str, str]] = {}
    exc_box: Dict[str, BaseException] = {}

    def _runner():
        try:
            result_box["v"] = fn()
        except BaseException as exc:  # noqa: BLE001 - logged below
            exc_box["e"] = exc

    t = threading.Thread(target=_runner, name=f"hc-{name}", daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logger.warning("Healthcheck stage %s timed out after %.1fs", name, timeout)
        return {
            "status": "error",
            "detail": f"{name} check timed out after {timeout:.1f}s",
        }
    if "e" in exc_box:
        logger.warning("Healthcheck stage %s raised: %s", name, exc_box["e"])
        return {"status": "error", "detail": f"{name} check raised: {exc_box['e']!r}"}
    return result_box.get("v", {"status": "error", "detail": f"{name} returned no result"})


def check_ollama_connection(host: str | None = None) -> Dict[str, str]:
    """Check if Ollama is reachable and return status."""
    try:
        import ollama

        if host:
            client = ollama.Client(host=host)
            client.list()
        else:
            ollama.list()
        return {"status": "ok", "detail": "Ollama is reachable"}
    except Exception as e:
        logger.warning("Healthcheck Ollama connection failed: %s", e)
        return {"status": "error", "detail": "Ollama connectivity check failed"}


def check_model_available(model_name: str, host: str | None = None) -> Dict[str, str]:
    """Check if a specific Ollama model is pulled."""
    try:
        import ollama

        if host:
            client = ollama.Client(host=host)
            models = client.list()
        else:
            models = ollama.list()

        # ollama client returns Model objects with .model attr (not dicts)
        raw_models = getattr(models, "models", None) or models.get("models", [])
        model_names = [getattr(m, "model", None) or m.get("name", "") for m in raw_models]
        # Match with or without tag
        if any(model_name in name for name in model_names):
            return {"status": "ok", "detail": f"Model '{model_name}' is available"}
        return {
            "status": "warning",
            "detail": f"Model '{model_name}' not found. Available: {model_names}. Run: ollama pull {model_name}",
        }
    except Exception as e:
        logger.warning("Healthcheck model check failed: %s", e)
        return {"status": "error", "detail": "Ollama model check failed"}


def check_documents_folder(docs_path: str = "./documents") -> Dict[str, str]:
    """Check if documents folder exists and is writable."""
    path = Path(docs_path)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            return {"status": "ok", "detail": f"Created documents folder: {path}"}
        except OSError as e:
            logger.warning("Healthcheck documents folder creation failed: %s", e)
            return {"status": "error", "detail": "Cannot create documents folder"}

    if not os.access(path, os.W_OK):
        return {"status": "error", "detail": f"Documents folder not writable: {path}"}

    file_count = sum(1 for _ in path.glob("*") if _.is_file())
    return {"status": "ok", "detail": f"Documents folder ready ({file_count} files)"}


def check_neo4j_connection() -> Dict[str, str]:
    """Check Neo4j connectivity when configured.

    Reuses the singleton driver returned by Neo4jStore.connect() and only
    pings ``verify_connectivity`` once; the surrounding ``_run_with_timeout``
    wrapper enforces the 3s budget so a stalled Neo4j cannot wedge /health.
    """
    uri = os.environ.get("NEO4J_URI", "").strip()
    if not uri:
        return {"status": "ok", "detail": "Neo4j not configured (optional)"}
    try:
        from graph_store import Neo4jStore

        store = Neo4jStore.connect()
        if store:
            store.close()
            return {"status": "ok", "detail": "Neo4j reachable"}
        return {"status": "warning", "detail": "Neo4j configured but connection failed"}
    except Exception:
        return {"status": "warning", "detail": "Neo4j check error"}


def check_cache_folders() -> Dict[str, str]:
    """Check if cache directories exist and are writable."""
    from config import settings

    cache_dirs = [settings.embedding_cache_dir, settings.llm_cache_dir]
    issues = []
    for d in cache_dirs:
        path = Path(d)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as e:
            issues.append(f"{d}: {e}")

    if issues:
        return {"status": "error", "detail": f"Cache folder issues: {'; '.join(issues)}"}
    return {"status": "ok", "detail": "Cache folders ready"}


def check_config_valid() -> Dict[str, str]:
    """Validate application configuration and return status."""
    from config import validate_settings

    errors, warnings = validate_settings()
    if errors:
        return {
            "status": "error",
            "detail": f"Config errors: {'; '.join(errors)}",
        }
    if warnings:
        return {
            "status": "warning",
            "detail": f"Config warnings: {'; '.join(warnings)}",
        }
    return {"status": "ok", "detail": "Configuration valid"}


def run_preflight_checks() -> List[Dict[str, str]]:
    """Run all startup validation checks. Returns list of check results.

    Each check is wrapped in a 3-second timeout so a hung dependency cannot
    stall the aggregate health probe.
    """
    from config import settings

    ollama_host = os.environ.get("OLLAMA_HOST")

    stages = [
        ("config_valid", lambda: check_config_valid()),
        ("ollama_connection", lambda: check_ollama_connection(ollama_host)),
        ("model_available", lambda: check_model_available(settings.llm_model, ollama_host)),
        ("documents_folder", lambda: check_documents_folder()),
        ("cache_folders", lambda: check_cache_folders()),
        ("neo4j_connection", lambda: check_neo4j_connection()),
    ]

    results = []
    for name, fn in stages:
        result = _run_with_timeout(fn, name)
        result["check"] = name
        level = {"ok": "INFO", "warning": "WARNING", "error": "ERROR"}.get(result["status"], "INFO")
        getattr(logger, level.lower())(f"[{name}] {result['detail']}")
        results.append(result)

    return results


def get_health_status(use_cache: bool = True) -> Dict:
    """Return aggregated health status for health check endpoints.

    Results are cached for ``_HEALTH_CACHE_TTL_SECONDS`` (10s) to insulate
    upstream dependencies from rapid orchestrator-driven probes.  Set
    ``use_cache=False`` to force a fresh probe (used by tests).
    """
    if use_cache:
        with _health_cache_lock:
            cached = _health_cache.get("value")
            cached_ts = _health_cache.get("ts", 0.0)
            if cached and (time.monotonic() - float(cached_ts)) < _HEALTH_CACHE_TTL_SECONDS:
                return cached  # type: ignore[return-value]

    results = run_preflight_checks()
    errors = [r for r in results if r["status"] == "error"]
    warnings = [r for r in results if r["status"] == "warning"]

    if errors:
        status = {"healthy": False, "status": "unhealthy", "checks": results}
    elif warnings:
        status = {"healthy": True, "status": "degraded", "checks": results}
    else:
        status = {"healthy": True, "status": "healthy", "checks": results}

    with _health_cache_lock:
        _health_cache["value"] = status
        _health_cache["ts"] = time.monotonic()
    return status


def _reset_health_cache() -> None:
    """Test helper: clear the cached health-status response."""
    with _health_cache_lock:
        _health_cache["value"] = None
        _health_cache["ts"] = 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_preflight_checks()
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n{len(errors)} preflight check(s) failed. App may not work correctly.")
        raise SystemExit(1)
    print("\nAll preflight checks passed.")
