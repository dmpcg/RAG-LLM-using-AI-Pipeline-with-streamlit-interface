"""
Embedding optimization utilities for the RAG-LLM financial system.

Provides quantization (int8/float16), Matryoshka-style dimension truncation,
quality-loss measurement, a versioned cache, and an optimal batch embedder.

All heavy lifting uses NumPy only – no special ML dependencies required.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization / dequantization
# ---------------------------------------------------------------------------


def quantize_embeddings(
    embeddings: list[list[float]],
    method: str = "int8",
) -> tuple[np.ndarray, dict]:
    """Quantize float32 embeddings to a lower-precision representation.

    Args:
        embeddings: List of float vectors (e.g. 1024-dim from mxbai-embed-large).
        method: One of ``"int8"`` (4x memory reduction) or ``"float16"`` (2x
            memory reduction).

    Returns:
        A 2-tuple of ``(quantized_array, metadata)``.

        For ``"int8"``:
          - ``quantized_array``: ``np.ndarray`` of dtype ``int8``, shape
            ``(n, dim)``.
          - ``metadata``: dict with keys ``"method"``, ``"min"`` (per-dim
            float64 array), ``"max"`` (per-dim float64 array), and
            ``"scale"`` (per-dim float64 array).

        For ``"float16"``:
          - ``quantized_array``: ``np.ndarray`` of dtype ``float16``.
          - ``metadata``: dict with keys ``"method"`` only.

    Raises:
        ValueError: If *method* is not ``"int8"`` or ``"float16"``.
    """
    if not embeddings:
        empty = np.empty((0, 0), dtype=np.int8 if method == "int8" else np.float16)
        return empty, {"method": method}

    method = method.lower()
    arr = np.asarray(embeddings, dtype=np.float64)  # work in float64 for precision

    if method == "float16":
        return arr.astype(np.float16), {"method": "float16"}

    if method == "int8":
        # Per-dimension min/max calibration
        dim_min = arr.min(axis=0)  # shape (dim,)
        dim_max = arr.max(axis=0)  # shape (dim,)
        # Avoid division by zero for constant dimensions
        span = dim_max - dim_min
        span = np.where(span == 0.0, 1.0, span)
        # Scale to [-128, 127]
        scale = 255.0 / span  # values per unit
        quantized = np.round((arr - dim_min) * scale - 128.0).clip(-128, 127).astype(np.int8)
        metadata = {
            "method": "int8",
            "min": dim_min,
            "max": dim_max,
            "scale": scale,
        }
        return quantized, metadata

    raise ValueError(f"Unknown quantization method {method!r}. Expected 'int8' or 'float16'.")


def dequantize_embeddings(quantized: np.ndarray, metadata: dict) -> np.ndarray:
    """Reverse the quantization produced by :func:`quantize_embeddings`.

    Args:
        quantized: Quantized array as returned by :func:`quantize_embeddings`.
        metadata: Calibration dict as returned by :func:`quantize_embeddings`.

    Returns:
        Approximate float32 reconstruction of the original embeddings.

    Raises:
        ValueError: If *metadata* contains an unknown ``"method"`` value.
    """
    if quantized.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    method = metadata.get("method", "")

    if method == "float16":
        return quantized.astype(np.float32)

    if method == "int8":
        dim_min = metadata["min"]
        scale = metadata["scale"]
        # Reverse: (q + 128) / scale + min
        reconstructed = (quantized.astype(np.float64) + 128.0) / scale + dim_min
        return reconstructed.astype(np.float32)

    raise ValueError(f"Unknown quantization method {method!r} in metadata.")


# ---------------------------------------------------------------------------
# Dimension truncation (Matryoshka-style)
# ---------------------------------------------------------------------------


def truncate_embeddings(
    embeddings: list[list[float]],
    target_dim: int,
    *,
    renormalize: bool = True,
) -> list[list[float]]:
    """Reduce embedding dimensionality by keeping only the first *target_dim* values.

    This is compatible with Matryoshka Representation Learning (MRL) models
    whose earlier dimensions carry the most semantic signal.

    Args:
        embeddings: List of float vectors.
        target_dim: Number of leading dimensions to retain.  Must be > 0.
        renormalize: If ``True`` (default), re-normalize each truncated vector
            to unit L2 length so cosine similarity remains well-defined.

    Returns:
        List of truncated (and optionally re-normalized) float vectors.

    Raises:
        ValueError: If *target_dim* is <= 0 or greater than the original
            dimensionality.
    """
    if not embeddings:
        return []

    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}.")

    original_dim = len(embeddings[0])
    if target_dim > original_dim:
        raise ValueError(f"target_dim ({target_dim}) exceeds original dimension ({original_dim}).")

    arr = np.asarray(embeddings, dtype=np.float32)[:, :target_dim]

    if renormalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        arr = arr / safe_norms

    return arr.tolist()


# ---------------------------------------------------------------------------
# Quality-loss measurement
# ---------------------------------------------------------------------------


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities between rows of *a* and *b*.

    Args:
        a: Shape ``(m, d)`` float32 array (L2-normalized).
        b: Shape ``(n, d)`` float32 array (L2-normalized).

    Returns:
        Shape ``(m, n)`` float32 similarity matrix.
    """
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = np.where(a_norms == 0.0, 1.0, a_norms)
    b_safe = np.where(b_norms == 0.0, 1.0, b_norms)
    a_norm = a / a_safe
    b_norm = b / b_safe
    return (a_norm @ b_norm.T).astype(np.float32)


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1-D arrays.

    Args:
        x: First array.
        y: Second array.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    n = len(x)
    if n < 2:
        return 1.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    d = rx - ry
    return float(1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1)))


def measure_quality_loss(
    original: list[list[float]],
    processed: list[list[float]],
    queries: Optional[list[list[float]]] = None,
) -> dict:
    """Quantify the information lost after quantization or truncation.

    Computes three metrics:

    1. **mae**: Mean absolute error of pairwise cosine similarities when
       queries are provided, else similarity of each vector to itself.
    2. **rank_correlation**: Spearman correlation of similarity rankings
       between original and processed spaces (averaged over queries).
    3. **similarity_correlation**: Pearson correlation of raw similarity
       scores between the two spaces.

    Args:
        original: Original float32 embeddings.
        processed: Processed (quantized / truncated) embeddings – must have
            the same number of rows as *original*.
        queries: Optional list of query embeddings used to compute retrieval
            quality.  When omitted, the corpus embeddings themselves are used
            as queries (self-similarity task).

    Returns:
        Dict with keys:
          - ``"mae"`` (float): Mean absolute error of similarity scores.
          - ``"rank_correlation"`` (float): Average Spearman rank correlation.
          - ``"similarity_correlation"`` (float): Pearson correlation of scores.
    """
    if not original:
        return {"mae": 0.0, "rank_correlation": 1.0, "similarity_correlation": 1.0}

    orig_arr = np.asarray(original, dtype=np.float32)

    # Processed may have different dtype / shape (e.g. truncated dim)
    proc_arr = np.asarray(processed, dtype=np.float32)

    q_orig = orig_arr if queries is None else np.asarray(queries, dtype=np.float32)
    q_proc = proc_arr if queries is None else np.asarray(queries, dtype=np.float32)

    sim_orig = _cosine_sim_matrix(q_orig, orig_arr)  # (q, n)
    sim_proc = _cosine_sim_matrix(q_proc, proc_arr)  # (q, n)

    mae = float(np.mean(np.abs(sim_orig - sim_proc)))

    # Pearson on flattened scores
    flat_orig = sim_orig.ravel()
    flat_proc = sim_proc.ravel()
    if flat_orig.std() < 1e-12 and flat_proc.std() < 1e-12:
        sim_corr = 1.0
    elif flat_orig.std() < 1e-12 or flat_proc.std() < 1e-12:
        sim_corr = 0.0
    else:
        corr_mat = np.corrcoef(flat_orig, flat_proc)
        sim_corr = float(corr_mat[0, 1])

    # Spearman rank correlation per query row
    rank_corrs: list[float] = []
    for row_o, row_p in zip(sim_orig, sim_proc):
        rank_corrs.append(_spearman_correlation(row_o, row_p))
    rank_corr = float(np.mean(rank_corrs))

    return {
        "mae": mae,
        "rank_correlation": rank_corr,
        "similarity_correlation": sim_corr,
    }


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Versioned on-disk embedding cache using NumPy .npz files.

    Each entry is stored as a separate ``.npz`` file under *cache_dir*.
    A version string is embedded in the metadata; cache entries from a
    different version are treated as misses and overwritten on the next
    ``put``.

    Args:
        cache_dir: Directory where cache files are stored.
        version: Version string.  Change this to invalidate the entire cache.
    """

    def __init__(self, cache_dir: str, version: str = "1.0", max_entries: int = 10_000) -> None:
        self._cache_dir = Path(cache_dir)
        self._version = version
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _entry_path(self, key: str) -> Path:
        """Return the file path for a given cache *key*."""
        # Sanitize the key so it is safe as a filename
        safe_key = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in key)
        return self._cache_dir / f"{safe_key}.npz"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[list[float]]:
        """Retrieve a cached embedding.

        Args:
            key: Cache key (e.g. ``"filename:mtime"``).

        Returns:
            The cached embedding as a list of floats, or ``None`` on a miss.
        """
        path = self._entry_path(key)
        with self._lock:
            if not path.exists():
                self._misses += 1
                return None
            try:
                data = np.load(path, allow_pickle=False)
                if str(data.get("version", np.array([""]))[0]) != self._version:
                    # Version mismatch – stale entry
                    self._misses += 1
                    return None
                self._hits += 1
                return data["embedding"].tolist()
            except Exception:
                self._misses += 1
                return None

    def put(self, key: str, embedding: list[float]) -> None:
        """Store an embedding in the cache.

        Evicts the oldest entry (by modification time) when the cache
        exceeds ``max_entries``.

        Args:
            key: Cache key.
            embedding: Embedding vector to store.
        """
        path = self._entry_path(key)
        arr = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            # Evict oldest entry if at capacity
            npz_files = sorted(
                self._cache_dir.glob("*.npz"),
                key=lambda p: p.stat().st_mtime,
            )
            if len(npz_files) >= self._max_entries:
                try:
                    npz_files[0].unlink()
                except OSError:
                    pass

            np.savez_compressed(
                path,
                embedding=arr,
                version=np.array([self._version]),
            )

    def invalidate(self, key: str) -> None:
        """Remove a single cache entry.

        Args:
            key: Cache key to invalidate.  No-op if not present.
        """
        path = self._entry_path(key)
        with self._lock:
            if path.exists():
                path.unlink()

    def invalidate_all(self) -> None:
        """Remove all cache entries in the cache directory."""
        with self._lock:
            for p in self._cache_dir.glob("*.npz"):
                try:
                    p.unlink()
                except OSError:
                    pass
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with keys:
              - ``"hits"`` (int): Number of successful cache lookups.
              - ``"misses"`` (int): Number of failed cache lookups.
              - ``"size"`` (int): Number of ``.npz`` files in the cache directory.
              - ``"memory_bytes"`` (int): Total byte size of all cache files.
              - ``"version"`` (str): Current cache version string.
        """
        with self._lock:
            npz_files = list(self._cache_dir.glob("*.npz"))
            total_bytes = sum(p.stat().st_size for p in npz_files if p.exists())
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(npz_files),
                "memory_bytes": total_bytes,
                "version": self._version,
            }


# ---------------------------------------------------------------------------
# BatchEmbedder
# ---------------------------------------------------------------------------


class BatchEmbedder:
    """Wraps an embedder with optimal batching and optional cache integration.

    Args:
        embedder: An object with ``embed_batch(texts: list[str]) -> list[list[float]]``
            or ``embed(text: str) -> list[float]`` methods (e.g. ``LocalEmbedder``).
        batch_size: Maximum number of texts per API call.
    """

    def __init__(self, embedder, batch_size: int = 32) -> None:
        self._embedder = embedder
        self._batch_size = max(1, batch_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_embedder(self, texts: list[str]) -> list[list[float]]:
        """Dispatch to batch or single embed depending on what the embedder exposes."""
        if hasattr(self._embedder, "embed_batch"):
            return self._embedder.embed_batch(texts)
        # Fallback: call embed() for each text
        return [self._embedder.embed(t) for t in texts]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_all(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts in optimal batches.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors in the same order as *texts*.
        """
        if not texts:
            return []

        results: list[list[float]] = []
        total = len(texts)
        processed = 0

        for start in range(0, total, self._batch_size):
            batch = texts[start : start + self._batch_size]
            batch_embeddings = self._call_embedder(batch)
            results.extend(batch_embeddings)
            processed += len(batch)
            logger.debug("BatchEmbedder: embedded %d/%d texts", processed, total)

        return results

    def embed_with_cache(
        self,
        texts: list[str],
        cache: EmbeddingCache,
    ) -> list[list[float]]:
        """Embed texts, returning cached results where available.

        For any text whose embedding is already in *cache*, the cached value
        is used directly.  Remaining texts are embedded in batches and stored
        back to the cache.

        Args:
            texts: List of input strings.
            cache: :class:`EmbeddingCache` instance to check and populate.

        Returns:
            List of embedding vectors in the same order as *texts*.
        """
        if not texts:
            return []

        results: list[Optional[list[float]]] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        # First pass: check cache
        for i, text in enumerate(texts):
            cached = cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        # Embed misses in batches
        if miss_texts:
            miss_embeddings = self.embed_all(miss_texts)
            for idx, text, emb in zip(miss_indices, miss_texts, miss_embeddings):
                results[idx] = emb
                cache.put(text, emb)

        # All slots should be filled
        if any(r is None for r in results):
            logger.warning(
                "Embedding batch returned %d None values",
                sum(1 for r in results if r is None),
            )
        return [r for r in results if r is not None]  # type: ignore[misc]
