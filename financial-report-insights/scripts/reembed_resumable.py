"""Resumable, hang-resistant bulk re-embed for the financial RAG index.

Ollama's CPU embedding runner can deadlock under sustained load: the
/v1/embeddings endpoint stops responding (HTTP 000) while /api/ps still
answers, and the in-process embedder blocks forever. Because SimpleRAG only
writes its embedding cache once the WHOLE file finishes, a single hang loses
all progress.

This script embeds the workbook's chunks in small batches, checkpoints every
batch to disk, and on a hang restarts Ollama and resumes from the checkpoint --
so the job always completes. When done it writes the SimpleRAG embedding cache
(.cache/embeddings/<key>.json) so the app loads the index instantly with no
re-embedding.

Run from the financial-report-insights dir:
    ./.venv/Scripts/python.exe scripts/reembed_resumable.py
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
BATCH = 4                    # small batches reduce Ollama CPU deadlock risk
REQ_TIMEOUT = 45.0          # per-request; shorter than a true hang
MAX_CHARS = 2500            # match LocalEmbedder truncation
CACHE_DIR = Path(settings.embedding_cache_dir)
PROGRESS = CACHE_DIR / "_reembed_progress.json"
# Resolve the shell to a full path (avoids PATH-hijack; satisfies ruff S607).
_PWSH = shutil.which("powershell") or shutil.which("pwsh") or "powershell"


def _docs_for_workbook():
    f = next(Path("documents").glob("*.xlsx"))
    docs = chunks_to_documents(ingest_file(f))
    # SimpleRAG cache key: sha256(name:content_hash:model)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    key_str = f"{f.name}:{h.hexdigest()}:{MODEL}"
    cache_key = hashlib.sha256(key_str.encode()).hexdigest()
    return f, docs, cache_key


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
        [_PWSH, "-NoProfile", "-Command",
         "Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True,
    )
    time.sleep(3)
    subprocess.Popen(  # noqa: S603
        [_PWSH, "-NoProfile", "-Command",
         "Start-Process ollama -ArgumentList serve -WindowStyle Hidden"],
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


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f, docs, cache_key = _docs_for_workbook()
    n = len(docs)
    texts = [_sanitize(d["content"]) for d in docs]
    print(f"workbook={f.name}  chunks={n}  cache_key={cache_key[:12]}", flush=True)

    # Resume from checkpoint if present.
    embeddings = [None] * n
    done = 0
    if PROGRESS.exists():
        try:
            saved = json.loads(PROGRESS.read_text())
            if saved.get("n") == n and saved.get("cache_key") == cache_key:
                for i, e in enumerate(saved["embeddings"]):
                    embeddings[i] = e
                done = sum(1 for e in embeddings if e is not None)
                print(f"resumed from checkpoint: {done}/{n} already embedded", flush=True)
        except Exception as exc:
            print(f"(ignoring unreadable checkpoint: {exc})", flush=True)

    if not _ollama_up():
        _restart_ollama()

    t0 = time.time()
    i = 0
    while i < n:
        if embeddings[i] is not None:
            i += 1
            continue
        # build next batch of not-yet-done items
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
                print(f"  batch at {batch_idx[0]} failed (attempt {attempt}): "
                      f"{type(exc).__name__}", flush=True)
                _restart_ollama()
        if not ok:
            print(f"FATAL: could not embed batch at {batch_idx[0]} after retries", flush=True)
            return 1

        done += len(batch_idx)
        i = batch_idx[-1] + 1
        # checkpoint every ~20 batches
        if done % (BATCH * 20) < BATCH or done >= n:
            PROGRESS.write_text(json.dumps(
                {"n": n, "cache_key": cache_key, "embeddings": embeddings}))
            rate = done / max(time.time() - t0, 1e-6)
            eta = (n - done) / max(rate, 1e-6) / 60
            print(f"  {done}/{n}  ({rate:.1f}/s, ETA {eta:.0f} min)", flush=True)

    # Fill any empty-text slots with zero vectors (consistent w/ app behavior).
    dim = len(next(e for e in embeddings if e))
    embeddings = [e if e is not None else [0.0] * dim for e in embeddings]

    # Write the SimpleRAG cache: [documents, embeddings].
    out = CACHE_DIR / f"{cache_key}.json"
    out.write_text(json.dumps([docs, embeddings]))
    PROGRESS.unlink(missing_ok=True)
    print(f"DONE wrote {out.name}  ({n} embeddings, dim={dim}) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
