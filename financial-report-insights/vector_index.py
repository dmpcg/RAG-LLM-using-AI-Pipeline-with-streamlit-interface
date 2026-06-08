"""
Vector index abstraction for approximate nearest-neighbour search.

Provides a uniform Protocol-based interface over three backends:
  - NumpyFlatIndex  : brute-force cosine similarity (always available)
  - FAISSIndex      : Facebook AI Similarity Search (optional)
  - HNSWIndex       : hnswlib hierarchical NSW (optional)

Use ``create_index()`` to obtain the best available backend automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol definition
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorIndex(Protocol):
    """Abstract interface for approximate nearest-neighbour vector indices.

    All distances/scores returned by ``search`` are cosine similarities in
    the range [-1, 1], higher is better.
    """

    def add(self, embeddings: List[List[float]], ids: List[int]) -> None:
        """Add embedding vectors with associated integer IDs.

        Args:
            embeddings: List of float vectors, each of length *dimension*.
            ids: Integer identifiers corresponding 1-to-1 with *embeddings*.
        """
        ...

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[int, float]]:
        """Return the *top_k* most similar entries to *query_embedding*.

        Args:
            query_embedding: A single float vector.
            top_k: Number of results to return.

        Returns:
            List of ``(id, score)`` tuples sorted by descending score.
        """
        ...

    def save(self, path: str) -> None:
        """Persist the index to *path* (prefix; implementation appends extensions).

        Args:
            path: File-system path prefix for saved files.
        """
        ...

    def load(self, path: str) -> None:
        """Restore the index from *path*.

        Args:
            path: File-system path prefix previously used with ``save``.
        """
        ...

    def __len__(self) -> int:
        """Return the number of vectors currently stored."""
        ...


# ---------------------------------------------------------------------------
# NumpyFlatIndex
# ---------------------------------------------------------------------------


class NumpyFlatIndex:
    """Brute-force cosine similarity index backed by a NumPy matrix.

    This is the always-available fallback.  It has O(n) search complexity
    but requires no additional dependencies.

    Attributes:
        dimension: Expected dimensionality of every vector.
    """

    def __init__(self, dimension: int) -> None:
        """Initialise an empty index.

        Args:
            dimension: Vector dimensionality (e.g. 1024 for mxbai-embed-large).
        """
        self.dimension = dimension
        self._matrix: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._ids: List[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """Return a unit-length copy of *v* (safe against zero vectors)."""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        safe_norm = np.where(norm == 0.0, 1.0, norm)
        return v / safe_norm

    # ------------------------------------------------------------------
    # VectorIndex interface
    # ------------------------------------------------------------------

    def add(self, embeddings: List[List[float]], ids: List[int]) -> None:
        """Add vectors to the index.

        Args:
            embeddings: Float vectors to index.
            ids: Integer identifiers, one per vector.

        Raises:
            ValueError: If *embeddings* and *ids* have different lengths.
        """
        if len(embeddings) != len(ids):
            raise ValueError(
                f"embeddings and ids must have the same length "
                f"(got {len(embeddings)} vs {len(ids)})"
            )
        if not embeddings:
            return

        new_matrix = np.asarray(embeddings, dtype=np.float32)
        if new_matrix.ndim != 2 or new_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors of dimension {self.dimension}, "
                f"got shape {new_matrix.shape}"
            )

        # Pre-normalize at add time so search is a pure dot product
        new_matrix = self._l2_normalize(new_matrix)

        if self._matrix.shape[0] == 0:
            self._matrix = new_matrix
        else:
            self._matrix = np.vstack([self._matrix, new_matrix])

        self._ids.extend(ids)

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[int, float]]:
        """Return the *top_k* closest vectors by cosine similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.

        Returns:
            List of ``(id, cosine_similarity)`` sorted descending.
        """
        n = self._matrix.shape[0]
        if n == 0:
            return []

        top_k = max(1, min(top_k, n))

        q = np.asarray(query_embedding, dtype=np.float32)
        q = self._l2_normalize(q.reshape(1, -1)).reshape(-1)

        # Stored vectors are already L2-normalized at add() time
        similarities = self._matrix @ q

        if n > top_k * 4:
            # O(n) partial sort via argpartition for large collections
            partitioned = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = partitioned[np.argsort(similarities[partitioned])[::-1]]
        else:
            top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(self._ids[i], float(similarities[i])) for i in top_indices]

    def save(self, path: str) -> None:
        """Save index to *<path>.npz*.

        Args:
            path: Path prefix (extension ``.npz`` is appended automatically).
        """
        save_path = Path(str(path) + ".npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            matrix=self._matrix,
            ids=np.asarray(self._ids, dtype=np.int64),
            dimension=np.array([self.dimension], dtype=np.int64),
        )
        logger.debug("NumpyFlatIndex saved to %s", save_path)

    def load(self, path: str) -> None:
        """Load index from *<path>.npz*.

        Args:
            path: Path prefix previously used with ``save``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        load_path = Path(str(path) + ".npz")
        if not load_path.exists():
            raise FileNotFoundError(f"Index file not found: {load_path}")

        data = np.load(load_path)
        self.dimension = int(data["dimension"][0])
        self._matrix = data["matrix"].astype(np.float32)
        self._ids = data["ids"].tolist()
        logger.debug("NumpyFlatIndex loaded from %s (%d vectors)", load_path, len(self._ids))

    def __len__(self) -> int:
        return self._matrix.shape[0]


# ---------------------------------------------------------------------------
# FAISSIndex
# ---------------------------------------------------------------------------


class FAISSIndex:
    """FAISS-backed approximate nearest-neighbour index.

    Uses ``IndexFlatIP`` (exact inner-product on L2-normalised vectors, which
    equals cosine similarity) for collections <= 10 000 vectors, and
    ``IndexIVFFlat`` for larger collections.

    Requires ``faiss-cpu`` or ``faiss-gpu`` to be installed.

    Args:
        dimension: Vector dimensionality.
        nprobe: Number of IVF cells to search (IVF mode only).
    """

    _IVF_THRESHOLD = 10_000

    def __init__(self, dimension: int, nprobe: int = 10) -> None:
        import faiss  # noqa: F401 – validated at construction time

        self.dimension = dimension
        self.nprobe = nprobe
        self._ids: List[int] = []
        self._raw_embeddings: List[List[float]] = []  # kept for IVF rebuild
        self._index = None  # built lazily on first add/search
        self._faiss = faiss  # keep module reference
        # Tracks the total size at which the next IVF retrain is scheduled.
        # Set to 0 initially; updated to the next power-of-two after each
        # IVF build (see add()).
        self._next_retrain_n: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        """Return the smallest power of two that is strictly greater than *n*.

        Args:
            n: A non-negative integer.

        Returns:
            The next power-of-two above *n* (always >= 1).

        Examples:
            _next_power_of_two(0)  -> 1
            _next_power_of_two(16) -> 32
            _next_power_of_two(17) -> 32
            _next_power_of_two(31) -> 32
            _next_power_of_two(32) -> 64
        """
        if n < 1:
            return 1
        return 1 << n.bit_length()

    def _build_index(self, matrix: np.ndarray) -> None:
        """(Re)build the FAISS index from *matrix* (already L2-normalised)."""
        faiss = self._faiss
        n = matrix.shape[0]

        if n <= self._IVF_THRESHOLD:
            index = faiss.IndexFlatIP(self.dimension)
        else:
            n_cells = max(1, int(np.sqrt(n)))
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, n_cells, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.nprobe
            if not index.is_trained:
                index.train(matrix)

        index.add(matrix)
        self._index = index

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """Return L2-normalised rows of *matrix*."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / safe_norms).astype(np.float32)

    # ------------------------------------------------------------------
    # VectorIndex interface
    # ------------------------------------------------------------------

    def add(self, embeddings: List[List[float]], ids: List[int]) -> None:
        """Add vectors to the index.

        For the flat index (<=_IVF_THRESHOLD vectors), new vectors are added
        incrementally.  For larger collections the IVF index is rebuilt only
        when the new total crosses a power-of-two boundary; all other adds
        insert directly into the existing IVF index without retraining.

        This amortises the O(n) rebuild cost: between boundaries vectors are
        assigned to existing centroids, which is fast but slightly approximate.
        A full retrain at each power-of-two restores centroid quality before
        the gap to the next retrain doubles in size.

        Args:
            embeddings: Float vectors.
            ids: Integer identifiers.

        Raises:
            ValueError: On length mismatch.
        """
        if len(embeddings) != len(ids):
            raise ValueError(
                f"embeddings and ids must match in length (got {len(embeddings)} vs {len(ids)})"
            )
        if not embeddings:
            return

        self._raw_embeddings.extend(embeddings)
        self._ids.extend(ids)

        new_matrix = self._normalize(np.asarray(embeddings, dtype=np.float32))
        total = len(self._ids)

        if self._index is None:
            # First add: build index from scratch (flat or IVF depending on size)
            self._build_index(new_matrix)
            if total > self._IVF_THRESHOLD:
                self._next_retrain_n = self._next_power_of_two(total)
        elif total <= self._IVF_THRESHOLD:
            # Flat index: incremental add (O(batch) not O(total))
            self._index.add(new_matrix)
        elif total >= getattr(self, "_next_retrain_n", 0):
            # IVF index: total crossed a power-of-two boundary — full rebuild
            # to retrain centroids on the enlarged corpus.
            all_matrix = self._normalize(
                np.asarray(self._raw_embeddings, dtype=np.float32)
            )
            self._build_index(all_matrix)
            self._next_retrain_n = self._next_power_of_two(total)
            logger.debug(
                "FAISSIndex: IVF retrained at n=%d; next retrain at n=%d",
                total,
                self._next_retrain_n,
            )
        else:
            # IVF index between power-of-two boundaries: add without retraining.
            # New vectors are assigned to the nearest existing centroids, which
            # is approximate but avoids the O(n) retrain cost.
            self._index.add(new_matrix)

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[int, float]]:
        """Search for the *top_k* nearest neighbours.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.

        Returns:
            List of ``(id, score)`` sorted descending.
        """
        if self._index is None or len(self._ids) == 0:
            return []

        top_k = max(1, min(top_k, len(self._ids)))
        q = np.asarray([query_embedding], dtype=np.float32)
        q = self._normalize(q)

        scores, positions = self._index.search(q, top_k)
        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos < 0:  # FAISS sentinel for missing results
                continue
            results.append((self._ids[pos], float(score)))
        return results

    def save(self, path: str) -> None:
        """Save the FAISS index and ID mapping.

        Args:
            path: Path prefix; files *<path>.faiss* and *<path>.ids.npy* are written.
        """
        faiss = self._faiss
        save_path = Path(str(path) + ".faiss")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(save_path))
        np.save(str(Path(str(path) + ".ids.npy")), np.asarray(self._ids, dtype=np.int64))
        logger.debug("FAISSIndex saved to %s", save_path)

    def load(self, path: str) -> None:
        """Load a previously saved FAISS index.

        Args:
            path: Path prefix previously used with ``save``.

        Raises:
            FileNotFoundError: If index files are missing.
        """
        faiss = self._faiss
        index_path = Path(str(path) + ".faiss")
        ids_path = Path(str(path) + ".ids.npy")

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"FAISS IDs file not found: {ids_path}")

        self._index = faiss.read_index(str(index_path))
        self._ids = np.load(str(ids_path)).tolist()
        logger.debug("FAISSIndex loaded from %s (%d vectors)", index_path, len(self._ids))

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# HNSWIndex
# ---------------------------------------------------------------------------


class HNSWIndex:
    """hnswlib-backed Hierarchical Navigable Small World index.

    Provides fast approximate nearest-neighbour search in cosine space.

    Requires the ``hnswlib`` package to be installed.

    Args:
        dimension: Vector dimensionality.
        m: HNSW connectivity parameter (edges per node).
        ef_construction: Build-time quality/speed trade-off.
        ef: Search-time candidate pool size.
    """

    def __init__(
        self,
        dimension: int,
        m: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
    ) -> None:
        import hnswlib  # noqa: F401 – validated at construction time

        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef = ef
        self._ids: List[int] = []
        self._hnswlib = hnswlib
        self._index = None  # built lazily

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_index(self, max_elements: int) -> None:
        """Initialise a fresh hnswlib index."""
        index = self._hnswlib.Index(space="cosine", dim=self.dimension)
        index.init_index(
            max_elements=max(max_elements, 1),
            M=self.m,
            ef_construction=self.ef_construction,
            random_seed=42,
        )
        index.set_ef(self.ef)
        self._index = index

    # ------------------------------------------------------------------
    # VectorIndex interface
    # ------------------------------------------------------------------

    def add(self, embeddings: List[List[float]], ids: List[int]) -> None:
        """Add vectors to the index.

        hnswlib requires ``max_elements`` to be set at construction, so the
        index is rebuilt from all accumulated embeddings when capacity is
        exceeded.

        Args:
            embeddings: Float vectors.
            ids: Integer identifiers.

        Raises:
            ValueError: On length mismatch.
        """
        if len(embeddings) != len(ids):
            raise ValueError(
                f"embeddings and ids must match in length (got {len(embeddings)} vs {len(ids)})"
            )
        if not embeddings:
            return

        new_matrix = np.asarray(embeddings, dtype=np.float32)

        # Accumulate raw embeddings for rebuild capability
        if not hasattr(self, "_raw_embeddings"):
            self._raw_embeddings: List[List[float]] = []
        self._raw_embeddings.extend(embeddings)
        self._ids.extend(ids)

        total = len(self._ids)

        if self._index is None:
            # First add: build with 2x headroom to avoid frequent rebuilds
            self._init_index(max(total * 2, 1024))
            all_matrix = np.asarray(self._raw_embeddings, dtype=np.float32)
            self._index.add_items(all_matrix, self._ids)
        elif total > self._index.get_max_elements():
            # Capacity exceeded: rebuild with 2x headroom from all data
            self._init_index(total * 2)
            all_matrix = np.asarray(self._raw_embeddings, dtype=np.float32)
            self._index.add_items(all_matrix, self._ids)
        else:
            # Incremental add: capacity is sufficient
            self._index.add_items(new_matrix, ids)

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[int, float]]:
        """Search for nearest neighbours.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.

        Returns:
            List of ``(id, score)`` where score = 1 - cosine_distance, sorted descending.
        """
        if self._index is None or len(self._ids) == 0:
            return []

        top_k = max(1, min(top_k, len(self._ids)))
        q = np.asarray([query_embedding], dtype=np.float32)

        labels, distances = self._index.knn_query(q, k=top_k)
        # hnswlib cosine space returns 1 - cosine_similarity as distance
        results = [
            (int(lbl), float(1.0 - dist))
            for lbl, dist in zip(labels[0], distances[0])
        ]
        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save(self, path: str) -> None:
        """Save the hnswlib index to disk.

        Args:
            path: Path prefix; file *<path>.hnsw* is written.
        """
        save_path = Path(str(path) + ".hnsw")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._index.save_index(str(save_path))
        np.save(str(Path(str(path) + ".hnsw.ids.npy")), np.asarray(self._ids, dtype=np.int64))
        logger.debug("HNSWIndex saved to %s", save_path)

    def load(self, path: str) -> None:
        """Load a previously saved hnswlib index.

        Args:
            path: Path prefix previously used with ``save``.

        Raises:
            FileNotFoundError: If index files are missing.
        """
        index_path = Path(str(path) + ".hnsw")
        ids_path = Path(str(path) + ".hnsw.ids.npy")

        if not index_path.exists():
            raise FileNotFoundError(f"HNSW index file not found: {index_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"HNSW IDs file not found: {ids_path}")

        self._ids = np.load(str(ids_path)).tolist()
        self._init_index(len(self._ids))
        self._index.load_index(str(index_path), max_elements=len(self._ids))
        self._index.set_ef(self.ef)
        logger.debug("HNSWIndex loaded from %s (%d vectors)", index_path, len(self._ids))

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_FAISS_AVAILABLE: bool | None = None
_HNSW_AVAILABLE: bool | None = None


def _check_faiss() -> bool:
    global _FAISS_AVAILABLE
    if _FAISS_AVAILABLE is None:
        try:
            import faiss  # noqa: F401
            _FAISS_AVAILABLE = True
        except ImportError:
            _FAISS_AVAILABLE = False
    return _FAISS_AVAILABLE


def _check_hnswlib() -> bool:
    global _HNSW_AVAILABLE
    if _HNSW_AVAILABLE is None:
        try:
            import hnswlib  # noqa: F401
            _HNSW_AVAILABLE = True
        except ImportError:
            _HNSW_AVAILABLE = False
    return _HNSW_AVAILABLE


def create_index(
    dimension: int,
    backend: str = "auto",
    *,
    nprobe: int = 10,
    hnsw_m: int = 16,
    hnsw_ef: int = 50,
) -> VectorIndex:
    """Create and return a ``VectorIndex`` using the requested backend.

    Args:
        dimension: Dimensionality of the embedding vectors.
        backend: One of ``"auto"``, ``"faiss"``, ``"hnswlib"``, or ``"numpy"``.
            ``"auto"`` tries FAISS first, then hnswlib, then falls back to
            the pure-NumPy implementation.
        nprobe: FAISS IVF search parameter (only used when backend includes FAISS).
        hnsw_m: hnswlib connectivity parameter ``M``.
        hnsw_ef: hnswlib search-time ``ef`` parameter.

    Returns:
        A ``VectorIndex`` instance.

    Raises:
        ValueError: If *backend* is not one of the recognised values, or if
            the requested backend is unavailable.
    """
    backend = backend.lower()

    if backend not in ("auto", "faiss", "hnswlib", "numpy"):
        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Expected one of: 'auto', 'faiss', 'hnswlib', 'numpy'."
        )

    if backend == "faiss":
        if not _check_faiss():
            raise ImportError("faiss is not installed. Install faiss-cpu or faiss-gpu.")
        logger.info("VectorIndex: using FAISS backend (dimension=%d)", dimension)
        return FAISSIndex(dimension, nprobe=nprobe)

    if backend == "hnswlib":
        if not _check_hnswlib():
            raise ImportError("hnswlib is not installed. Run: pip install hnswlib")
        logger.info("VectorIndex: using hnswlib backend (dimension=%d)", dimension)
        return HNSWIndex(dimension, m=hnsw_m, ef=hnsw_ef)

    if backend == "numpy":
        logger.info("VectorIndex: using NumpyFlatIndex backend (dimension=%d)", dimension)
        return NumpyFlatIndex(dimension)

    # backend == "auto"
    if _check_faiss():
        logger.info("VectorIndex (auto): selected FAISS backend (dimension=%d)", dimension)
        return FAISSIndex(dimension, nprobe=nprobe)

    if _check_hnswlib():
        logger.info("VectorIndex (auto): selected hnswlib backend (dimension=%d)", dimension)
        return HNSWIndex(dimension, m=hnsw_m, ef=hnsw_ef)

    logger.info(
        "VectorIndex (auto): FAISS and hnswlib not available, "
        "falling back to NumpyFlatIndex (dimension=%d)",
        dimension,
    )
    return NumpyFlatIndex(dimension)
