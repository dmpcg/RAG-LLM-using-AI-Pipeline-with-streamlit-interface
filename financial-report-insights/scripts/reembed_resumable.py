"""Resumable, hang-resistant bulk re-embed for the financial RAG index.

Ollama's CPU embedding runner can deadlock under sustained load: the
/v1/embeddings endpoint stops responding (HTTP 000) while /api/ps still
answers, and the in-process embedder blocks forever. Because SimpleRAG only
writes its embedding cache once the WHOLE file finishes, a single hang loses
all progress.

This script embeds each workbook's chunks in small batches, checkpoints every
batch to disk, and on a hang restarts Ollama and resumes from the checkpoint --
so the job always completes. When done it writes the SimpleRAG embedding cache
(.cache/embeddings/<key>.json) so the app loads the index instantly with no
re-embedding.

It iterates EVERY supported file in documents/ (smallest first, to warm the
embedder on light payloads), skips files whose content-hash cache already
exists, and skips unsupported formats (e.g. .msg) with a clear notice. After
this finishes, restart the Streamlit app so SimpleRAG rebuilds
data/vector_index.npz from the full set of caches.

Run from the financial-report-insights dir:
    ./.venv/Scripts/python.exe scripts/reembed_resumable.py
Optionally pass one or more filenames (within documents/) to limit the run:
    ./.venv/Scripts/python.exe scripts/reembed_resumable.py "SC MI Farm Inventory (1).xlsx"
"""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Run from anywhere: put the app dir (parent of scripts/) on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from ingestion_pipeline import chunks_to_documents, ingest_file  # noqa: E402

OLLAMA = "http://localhost:11434"
URL = f"{OLLAMA}/v1/embeddings"
MODEL = settings.embedding_model
BATCH = 4  # small batches reduce Ollama CPU deadlock risk
REQ_TIMEOUT = 45.0  # per-request; shorter than a true hang
MAX_CHARS = 2500  # match LocalEmbedder truncation
CACHE_DIR = Path(settings.embedding_cache_dir)
# Extensions the ingestion pipeline can actually parse + embed.
SUPPORTED = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".xlsx",
    ".xlsm",
    ".xls",
    ".csv",
    ".tsv",
}
# Resolve the shell to a full path (avoids PATH-hijack; satisfies ruff S607).
_PWSH = shutil.which("powershell") or shutil.which("pwsh") or "powershell"


def _cache_key(file_path: Path) -> str:
    """SimpleRAG cache key: sha256(name:content_hash:model)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    key_str = f"{file_path.name}:{h.hexdigest()}:{MODEL}"
    return hashlib.sha256(key_str.encode()).hexdigest()


def _progress_path(cache_key: str) -> Path:
    return CACHE_DIR / f"_reembed_progress_{cache_key[:16]}.json"


def _sanitize(text: str) -> str:
    text = text[:MAX_CHARS]
    return text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def _restart_ollama():
    print("  ! restarting Ollama (cleared wedged embed runner)...", flush=True)
    # All args are hardcoded constants (no untrusted input) -> S603 suppressed.
    subprocess.run(  # noqa: S603
        [_PWSH, "-NoProfile", "-Command", "Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True,
    )
    time.sleep(3)
    subprocess.Popen(  # noqa: S603
        [_PWSH, "-NoProfile", "-Command", "Start-Process ollama -ArgumentList serve -WindowStyle Hidden"],
    )
    for _ in range(20):
        if _ollama_up():
            break
        time.sleep(2)
    # warm the model back up
    try:
        httpx.post(URL, json={"model": MODEL, "input": ["warm"]}, timeout=60)
    except Exception as exc:
        print(f"  (post-restart warm-up failed, continuing: {exc})", flush=True)
    time.sleep(1)


def _embed_one_batch(texts):
    """Embed a small batch; raises on timeout/non-200 so caller can recover."""
    with httpx.Client(timeout=REQ_TIMEOUT) as c:
        r = c.post(URL, json={"model": MODEL, "input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]


def reembed_file(file_path: Path) -> int:
    """Embed one file's chunks resumably and write its SimpleRAG cache.

    Returns 0 on success (or skip), 1 on fatal embedding failure.
    """
    cache_key = _cache_key(file_path)
    out = CACHE_DIR / f"{cache_key}.json"
    if out.exists():
        print(f"SKIP  {file_path.name}  (already cached {cache_key[:12]})", flush=True)
        return 0

    docs = chunks_to_documents(ingest_file(file_path))
    n = len(docs)
    if n == 0:
        print(f"WARN  {file_path.name}  produced 0 chunks -- nothing to embed", flush=True)
        return 0
    texts = [_sanitize(d["content"]) for d in docs]
    print(f"FILE  {file_path.name}  chunks={n}  cache_key={cache_key[:12]}", flush=True)

    progress = _progress_path(cache_key)
    embeddings = [None] * n
    done = 0
    if progress.exists():
        try:
            saved = json.loads(progress.read_text())
            if saved.get("n") == n and saved.get("cache_key") == cache_key:
                for i, e in enumerate(saved["embeddings"]):
                    embeddings[i] = e
                done = sum(1 for e in embeddings if e is not None)
                print(f"  resumed from checkpoint: {done}/{n} already embedded", flush=True)
        except Exception as exc:
            print(f"  (ignoring unreadable checkpoint: {exc})", flush=True)

    if not _ollama_up():
        _restart_ollama()

    t0 = time.time()
    i = 0
    while i < n:
        if embeddings[i] is not None:
            i += 1
            continue
        batch_idx = []
        j = i
        while j < n and len(batch_idx) < BATCH:
            if embeddings[j] is None:
                batch_idx.append(j)
            j += 1
        batch_texts = [texts[k] if texts[k].strip() else " " for k in batch_idx]

        ok = False
        for attempt in range(1, 5):
            try:
                vecs = _embed_one_batch(batch_texts)
                for k, v in zip(batch_idx, vecs):
                    embeddings[k] = v
                ok = True
                break
            except Exception as exc:
                print(f"  batch at {batch_idx[0]} failed (attempt {attempt}): {type(exc).__name__}", flush=True)
                _restart_ollama()
        if not ok:
            print(f"FATAL: could not embed batch at {batch_idx[0]} after retries", flush=True)
            return 1

        done += len(batch_idx)
        i = batch_idx[-1] + 1
        # Checkpoint every ~20 chunks (BATCH*5) so an external kill loses little.
        if done % (BATCH * 5) < BATCH or done >= n:
            progress.write_text(json.dumps({"n": n, "cache_key": cache_key, "embeddings": embeddings}))
            rate = done / max(time.time() - t0, 1e-6)
            eta = (n - done) / max(rate, 1e-6) / 60
            print(f"  {done}/{n}  ({rate:.1f}/s, ETA {eta:.0f} min)", flush=True)

    dim = len(next(e for e in embeddings if e))
    embeddings = [e if e is not None else [0.0] * dim for e in embeddings]
    out.write_text(json.dumps([docs, embeddings]))
    progress.unlink(missing_ok=True)
    print(
        f"DONE  {file_path.name}  wrote {out.name}  ({n} embeddings, dim={dim}) in {time.time() - t0:.0f}s", flush=True
    )
    return 0


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("documents")

    if len(sys.argv) > 1:
        targets = [docs_dir / name for name in sys.argv[1:]]
    else:
        targets = sorted(
            (f for f in docs_dir.rglob("*") if f.is_file()),
            key=lambda p: p.stat().st_size,
        )

    supported = [f for f in targets if f.suffix.lower() in SUPPORTED]
    skipped = [f for f in targets if f.suffix.lower() not in SUPPORTED]
    for f in skipped:
        print(
            f"UNSUPPORTED  {f.name}  ({f.suffix}) -- pipeline cannot ingest; convert to xlsx/pdf/txt first", flush=True
        )

    print(f"=== re-embed: {len(supported)} supported file(s) ===", flush=True)
    rc = 0
    for f in supported:
        if not f.is_file():
            print(f"MISSING  {f}  (not found)", flush=True)
            continue
        if reembed_file(f) != 0:
            rc = 1
            print(f"  -> stopping run due to fatal error on {f.name}", flush=True)
            break

    print(
        "=== re-embed run complete. Restart Streamlit so SimpleRAG rebuilds data/vector_index.npz from all caches. ===",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
