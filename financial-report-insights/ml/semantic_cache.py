"""
Semantic caching and deduplication utilities for RAG query optimization.

Provides three components:
  - SemanticCache    : Query-level cache keyed by embedding similarity.
  - ChunkDeduplicator: Remove near-duplicate text chunks before indexing.
  - AdaptiveTopK     : Dynamically choose retrieval top-k by query complexity.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Query-level semantic cache backed by cosine-similarity lookup.

    When a new query embedding arrives, the cache checks whether any stored
    query embedding exceeds *similarity_threshold*.  On a hit the previously
    computed result is returned immediately, bypassing the LLM.

    The cache is capped at *max_entries* entries; the oldest entry is evicted
    when capacity is exceeded (FIFO eviction on top of the ordered store).

    All public methods are thread-safe via an internal ``threading.Lock``.

    Attributes:
        similarity_threshold: Minimum cosine similarity to count as a hit.
        max_entries: Maximum number of (embedding, result) pairs to retain.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
    ) -> None:
        """Initialise an empty cache.

        Args:
            similarity_threshold: Cosine similarity above which a stored query
                is considered equivalent to the incoming query.
            max_entries: Maximum cache capacity; oldest entry is evicted first.
        """
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in (0, 1], got {similarity_threshold}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")

        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        # OrderedDict preserves insertion order for FIFO eviction.
        # Values are (embedding_array, result_dict).
        self._store: OrderedDict[int, tuple[np.ndarray, dict]] = OrderedDict()
        self._key_list: list[int] = []  # parallel list for O(1) index lookup
        self._next_key: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.Lock()
        # Cached embedding matrix; rebuilt only when the store changes.
        self._matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Return an L2-normalised copy of *v* (safe against zero vectors)."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0.0 else v

    def _build_matrix(self) -> Optional[np.ndarray]:
        """Return stored embeddings stacked into a (n, dim) matrix, or None."""
        if not self._store:
            return None
        return np.vstack([emb for emb, _ in self._store.values()])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query_embedding: list[float]) -> Optional[dict]:
        """Look up a cached result for the given query embedding.

        Args:
            query_embedding: Float vector representing the query.

        Returns:
            Cached result dict on a hit, or ``None`` on a miss.
        """
        with self._lock:
            if not self._store:
                self._misses += 1
                return None

            q = self._normalize(np.asarray(query_embedding, dtype=np.float32))
            if self._matrix is None:
                self._matrix = self._build_matrix()
            matrix = self._matrix
            # matrix rows are already stored normalised (see put())
            similarities = matrix @ q  # shape: (n,)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self.similarity_threshold:
                # Return the result from the matching entry.
                key = self._key_list[best_idx]
                _, result = self._store[key]
                self._hits += 1
                logger.debug("SemanticCache hit (similarity=%.4f)", best_sim)
                return result

            self._misses += 1
            return None

    def put(self, query_embedding: list[float], result: dict) -> None:
        """Store a (query_embedding, result) pair.

        Evicts the oldest entry when the cache is at capacity.

        Args:
            query_embedding: Float vector representing the query.
            result: The result dict produced for this query.
        """
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Evict oldest (first) entry.
                self._store.popitem(last=False)
                self._key_list.pop(0)

            embedding = self._normalize(np.asarray(query_embedding, dtype=np.float32))
            self._store[self._next_key] = (embedding, result)
            self._key_list.append(self._next_key)
            self._next_key += 1
            # Invalidate cached matrix so it's rebuilt on the next get().
            self._matrix = None

    def get_stats(self) -> dict:
        """Return cache performance statistics.

        Returns:
            Dict with keys ``hits``, ``misses``, ``hit_rate``, ``size``.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._store),
            }

    def clear(self) -> None:
        """Reset the cache and all counters."""
        with self._lock:
            self._store.clear()
            self._key_list.clear()
            self._next_key = 0
            self._hits = 0
            self._misses = 0
            self._matrix = None


# ---------------------------------------------------------------------------
# ChunkDeduplicator
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_text(text: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return _whitespace_RE.sub(" ", text.lower().strip())


# Work around the forward-slash in the name pattern for pre-compiled regex.
_whitespace_RE = _WHITESPACE_RE


def _simhash(text: str, num_bits: int = 64) -> int:
    """Compute a locality-sensitive hash of *text* using 3-gram shingles.

    The resulting integer has *num_bits* bits.  Two texts with similar content
    will produce hashes with a small Hamming distance.

    Args:
        text: Input text (should already be normalised).
        num_bits: Bit-width of the output hash (must be a positive multiple
            of 8 that fits in a Python int).

    Returns:
        Integer SimHash value.
    """
    vector = [0] * num_bits
    # Build 3-gram character shingles.
    for i in range(len(text) - 2):
        shingle = text[i : i + 3]
        digest = hashlib.md5(shingle.encode(), usedforsecurity=False).digest()
        # Interpret first *num_bits* bits of the digest.
        h = int.from_bytes(digest[: num_bits // 8], "big")
        for bit in range(num_bits):
            if h & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1

    result = 0
    for bit in range(num_bits):
        if vector[bit] > 0:
            result |= 1 << bit
    return result


def _hamming_similarity(a: int, b: int, num_bits: int = 64) -> float:
    """Return normalised Hamming similarity in [0, 1].

    Args:
        a: First hash value.
        b: Second hash value.
        num_bits: Bit-width used when computing both hashes.

    Returns:
        1.0 means identical hashes; 0.0 means all bits differ.
    """
    xor = a ^ b
    differing = bin(xor).count("1")
    return 1.0 - differing / num_bits


class ChunkDeduplicator:
    """Remove near-duplicate text chunks using SimHash bit-similarity.

    SimHash maps text to a fixed-width integer such that texts with similar
    n-gram distributions produce hashes with a small Hamming distance.  Two
    chunks are considered duplicates when their normalised Hamming similarity
    exceeds *similarity_threshold*.

    Attributes:
        similarity_threshold: Minimum SimHash similarity to classify as a
            duplicate (0.9 means at most 10% of bits differ).
    """

    _NUM_BITS = 64

    def __init__(self, similarity_threshold: float = 0.9) -> None:
        """Initialise the deduplicator.

        Args:
            similarity_threshold: Chunks are duplicates when their SimHash
                similarity is at or above this value.
        """
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in (0, 1], got {similarity_threshold}")
        self.similarity_threshold = similarity_threshold

    def _hash_chunk(self, chunk: str) -> int:
        """Return the SimHash of a (pre-normalised) chunk."""
        return _simhash(_normalise_text(chunk), self.NUM_BITS)

    # Expose _NUM_BITS as a property-like class attribute alias.
    @property
    def NUM_BITS(self) -> int:  # noqa: N802  (upper-case intentional)
        """Bit-width used for SimHash computation."""
        return self._NUM_BITS

    def deduplicate(self, chunks: list[str]) -> list[str]:
        """Return a deduplicated copy of *chunks*, preserving insertion order.

        The first occurrence of each unique chunk is retained; subsequent
        near-duplicates are dropped.

        Args:
            chunks: Ordered list of text chunks.

        Returns:
            Deduplicated list in original order.
        """
        if not chunks:
            return []
        # Early exit: single-element input is trivially unique.
        if len(chunks) == 1:
            return list(chunks)

        if len(chunks) > 1000:
            logger.warning(
                "Large chunk set (%d), dedup may be slow — consider LSH for O(n) dedup",
                len(chunks),
            )

        hashes: list[int] = [_simhash(_normalise_text(c), self._NUM_BITS) for c in chunks]
        kept_indices: list[int] = []

        for i, h_i in enumerate(hashes):
            is_dup = False
            for j in kept_indices:
                sim = _hamming_similarity(h_i, hashes[j], self._NUM_BITS)
                if sim >= self.similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept_indices.append(i)

        return [chunks[i] for i in kept_indices]

    def find_duplicates(self, chunks: list[str]) -> list[tuple[int, int, float]]:
        """Identify all near-duplicate pairs among *chunks*.

        Args:
            chunks: List of text chunks to compare pairwise.

        Returns:
            List of ``(idx_a, idx_b, similarity)`` triples where
            ``similarity >= self.similarity_threshold``.  Each pair appears
            at most once (``idx_a < idx_b``).
        """
        if len(chunks) < 2:
            return []

        hashes = [_simhash(_normalise_text(c), self._NUM_BITS) for c in chunks]
        pairs: list[tuple[int, int, float]] = []

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                sim = _hamming_similarity(hashes[i], hashes[j], self._NUM_BITS)
                if sim >= self.similarity_threshold:
                    pairs.append((i, j, sim))

        return pairs


# ---------------------------------------------------------------------------
# AdaptiveTopK
# ---------------------------------------------------------------------------

# Words that strongly signal query complexity.
_COMPARISON_WORDS = frozenset({"compare", "versus", "vs", "difference", "against", "between"})
_EXPLANATION_WORDS = frozenset({"explain", "why", "how", "describe", "elaborate", "detail"})
_TREND_WORDS = frozenset({"trend", "over time", "change", "growth", "decline", "historic"})
_RATIO_WORDS = frozenset({"ratio", "margin", "return", "yield", "rate", "multiple"})

_QUERY_TYPE_MAP: dict[str, str] = {
    "ratio_lookup": "ratio_lookup",
    "trend_analysis": "trend_analysis",
    "comparison": "comparison",
    "explanation": "explanation",
    "general": "general",
}


class AdaptiveTopK:
    """Dynamically choose the retrieval top-k based on query complexity.

    Simple queries (e.g. looking up a single financial ratio) need fewer
    retrieved chunks.  Complex queries (e.g. multi-company comparisons or
    detailed explanations) benefit from more context.

    The class uses a rule-based heuristic:
      - Start from a base *k* determined by *query_type*.
      - Add extra *k* for long queries (>15 words).
      - Clip the result to [min_k, max_k].

    Attributes:
        min_k: Minimum value returned by ``compute_k``.
        max_k: Maximum value returned by ``compute_k``.
        default_k: Base *k* for general queries.
    """

    def __init__(
        self,
        min_k: int = 3,
        max_k: int = 15,
        default_k: int = 5,
    ) -> None:
        """Initialise the adaptive top-k selector.

        Args:
            min_k: Hard lower bound on returned *k*.
            max_k: Hard upper bound on returned *k*.
            default_k: Starting *k* for the ``general`` query type.

        Raises:
            ValueError: If bounds are inconsistent.
        """
        if min_k < 1:
            raise ValueError(f"min_k must be >= 1, got {min_k}")
        if max_k < min_k:
            raise ValueError(f"max_k ({max_k}) must be >= min_k ({min_k})")
        if not (min_k <= default_k <= max_k):
            raise ValueError(f"default_k ({default_k}) must be within [min_k, max_k] = [{min_k}, {max_k}]")

        self.min_k = min_k
        self.max_k = max_k
        self.default_k = default_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_k(self, query_type: str) -> int:
        """Return the base top-k for a given *query_type*.

        Args:
            query_type: One of ``ratio_lookup``, ``trend_analysis``,
                ``comparison``, ``explanation``, ``general`` (or any
                unknown string which falls back to ``general``).

        Returns:
            An integer base value before length adjustment.
        """
        qtype = query_type.lower() if query_type else "general"
        if qtype == "ratio_lookup":
            return self.min_k  # minimal context needed
        if qtype == "trend_analysis":
            return self.default_k
        if qtype == "comparison":
            return min(self.default_k + 3, self.max_k)
        if qtype == "explanation":
            # Range: max_k-2 to max_k; use max_k-2 as the base.
            return max(self.min_k, self.max_k - 2)
        # "general" and any unknown type.
        return self.default_k

    @staticmethod
    def _word_count(query: str) -> int:
        """Return the number of whitespace-separated words in *query*."""
        return len(query.split()) if query.strip() else 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_k(self, query: str, query_type: str = "general") -> int:
        """Return the recommended top-k for *query*.

        Rules applied in order:
          1. Start with the base *k* for *query_type*.
          2. If the query has > 15 words, add 2.
          3. Clip to [min_k, max_k].

        Args:
            query: The raw query string.
            query_type: Classifier label for the query (see ``_base_k``).

        Returns:
            Integer top-k in [min_k, max_k].
        """
        base = self._base_k(query_type)

        words = self._word_count(query)
        if words > 15:
            base += 2

        return max(self.min_k, min(self.max_k, base))
