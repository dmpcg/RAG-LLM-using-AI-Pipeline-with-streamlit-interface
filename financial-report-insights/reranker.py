"""Cross-encoder style reranking for RAG retrieval.

Provides reranking using embedding similarity rescoring.
When a real cross-encoder model is available (e.g., via DMR),
it can be swapped in by implementing the RerankerProtocol.
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Protocol

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_MAXSIZE = 1024


def _content_key(doc: dict[str, Any]) -> str:
    """Return a cache key for a document.

    Key = sha256(content) when no chunk_id is present.
    Key = (chunk_id, sha256(content)) when chunk_id is present so that a
    content change under the same id is always a cache miss.
    """
    content = doc.get("content", "")
    content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    chunk_id = doc.get("chunk_id")
    if chunk_id is not None:
        return f"{chunk_id}:{content_hash}"
    return content_hash


class RerankerProtocol(Protocol):
    """Protocol for document rerankers."""

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query."""
        ...


class EmbeddingReranker:
    """Reranker using embedding similarity rescoring.

    Computes query-document similarity using embeddings and rescores
    documents. This is lighter than a true cross-encoder but provides
    a reranking signal when combined with the initial retrieval scores.

    Can be replaced with a cross-encoder model for better quality.

    P1-A3: Doc embeddings are cached on the instance keyed by content-hash
    (plus chunk_id when present) using a bounded LRU OrderedDict.  Only
    cache-miss documents are embedded on each rerank() call.  The cache is
    never poisoned by a partial or full embed failure.
    """

    def __init__(self, embedder, cache_maxsize: int | None = None):
        """Initialize with an embedding provider.

        Args:
            embedder: An EmbeddingProvider instance (e.g., LocalEmbedder)
            cache_maxsize: Max number of doc-embedding entries in the LRU
                cache.  Defaults to settings.reranker_cache_maxsize (1024).
        """
        self._embedder = embedder

        if cache_maxsize is None:
            try:
                from config import settings as _settings

                cache_maxsize = _settings.reranker_cache_maxsize
            except Exception:
                cache_maxsize = _DEFAULT_CACHE_MAXSIZE

        self._cache_maxsize: int = max(1, cache_maxsize)
        # OrderedDict used as an LRU: most-recently-used entry is moved to the
        # end on access; oldest entry (front) is evicted when full.
        self._doc_embed_cache: OrderedDict = OrderedDict()

    def _cache_get(self, key: str) -> list[float] | None:
        """Return the cached embedding for key, updating LRU order."""
        if key not in self._doc_embed_cache:
            return None
        # Move to end (most recently used)
        self._doc_embed_cache.move_to_end(key)
        return self._doc_embed_cache[key]

    def _cache_put(self, key: str, embedding: list[float]) -> None:
        """Insert an embedding into the cache, evicting the LRU entry if full."""
        if key in self._doc_embed_cache:
            self._doc_embed_cache.move_to_end(key)
        else:
            if len(self._doc_embed_cache) >= self._cache_maxsize:
                # Evict least-recently-used (front of OrderedDict)
                self._doc_embed_cache.popitem(last=False)
            self._doc_embed_cache[key] = embedding

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank documents by embedding similarity to query.

        Only embeds cache-miss documents.  Cache is keyed on content-hash
        (plus chunk_id when present) so a content change under the same id
        is always a miss.  On any embed failure the cache is not updated and
        the original document order is returned.

        Args:
            query: The user's query
            documents: List of document dicts with 'content' key
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents (top_k), each with '_rerank_score' added
        """
        if not documents:
            return []

        top_k = min(top_k, len(documents))

        try:
            # Embed query
            query_vec = np.asarray(self._embedder.embed(query), dtype=np.float32)

            # Resolve cache hits and collect miss indices
            keys = [_content_key(d) for d in documents]
            hit_vecs: dict[str, list[float]] = {}
            miss_indices: list[int] = []

            for i, key in enumerate(keys):
                cached = self._cache_get(key)
                if cached is not None:
                    hit_vecs[key] = cached
                else:
                    miss_indices.append(i)

            # Embed only cache misses
            if miss_indices:
                miss_texts = [documents[i].get("content", "") for i in miss_indices]
                miss_vecs = self._embedder.embed_batch(miss_texts)
                # Only populate cache on success (no partial insert on raise)
                for idx, vec in zip(miss_indices, miss_vecs):
                    self._cache_put(keys[idx], vec)
                    hit_vecs[keys[idx]] = vec

            # Assemble full doc_vecs in original document order
            doc_vecs = np.asarray([hit_vecs[k] for k in keys], dtype=np.float32)

            # Compute cosine similarities
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.linalg.norm(doc_vecs, axis=1)
            safe_norms = np.where(doc_norms == 0, 1.0, doc_norms)

            if query_norm == 0:
                return documents[:top_k]

            similarities = doc_vecs @ query_vec / (safe_norms * query_norm)

            # Sort by similarity score
            ranked_indices = np.argsort(similarities)[::-1][:top_k]

            result = []
            for idx in ranked_indices:
                doc = documents[idx].copy()
                doc["_rerank_score"] = float(similarities[idx])
                result.append(doc)

            return result

        except Exception as e:
            logger.warning("Reranking failed, returning original order: %s", e)
            return documents[:top_k]


def mmr_diversify(
    query_embedding: List[float],
    documents: List[Dict[str, Any]],
    doc_embeddings: List[List[float]],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance for result diversification.

    Selects documents that are both relevant to the query AND diverse
    from each other. Prevents returning multiple near-duplicate chunks.

    MMR formula: argmax[lambda * sim(q, d) - (1-lambda) * max(sim(d, d_selected))]

    Args:
        query_embedding: Query vector
        documents: List of candidate documents
        doc_embeddings: Parallel list of document embedding vectors
        top_k: Number of documents to select
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)

    Returns:
        Selected diverse documents
    """
    if not documents or not doc_embeddings:
        return []

    top_k = min(top_k, len(documents))

    query_vec = np.asarray(query_embedding, dtype=np.float32)
    doc_matrix = np.asarray(doc_embeddings, dtype=np.float32)

    # Compute query-document similarities
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return documents[:top_k]

    d_norms = np.linalg.norm(doc_matrix, axis=1)
    safe_d_norms = np.where(d_norms == 0, 1.0, d_norms)
    query_sims = doc_matrix @ query_vec / (safe_d_norms * q_norm)

    # Greedy MMR selection
    selected_indices: List[int] = []
    remaining = set(range(len(documents)))

    for _ in range(top_k):
        if not remaining:
            break

        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = float(query_sims[idx])

            # Max similarity to already-selected documents
            if selected_indices:
                selected_vecs = doc_matrix[selected_indices]
                idx_vec = doc_matrix[idx]
                idx_norm = float(d_norms[idx]) if d_norms[idx] != 0 else 1.0
                sel_norms = np.linalg.norm(selected_vecs, axis=1)
                safe_sel_norms = np.where(sel_norms == 0, 1.0, sel_norms)
                sims_to_selected = selected_vecs @ idx_vec / (safe_sel_norms * idx_norm)
                max_sim_to_selected = float(np.max(sims_to_selected))
            else:
                max_sim_to_selected = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.discard(best_idx)

    return [documents[i] for i in selected_indices]


def add_citations(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add citation metadata to retrieved documents.

    Enriches each document with a citation string that can be referenced
    in the LLM response.

    Args:
        documents: List of document dicts

    Returns:
        Documents with '_citation' key added
    """
    cited_docs = []
    for i, doc in enumerate(documents):
        doc_copy = doc.copy()
        source = doc.get("source", "unknown")

        # Build citation parts
        parts = [f"Source: {source}"]

        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            if metadata.get("statement_type") and metadata["statement_type"] != "unknown":
                parts.append(f"Type: {metadata['statement_type']}")
            if "chunk_index" in metadata:
                parts.append(f"Section: {metadata['chunk_index'] + 1}")
            if metadata.get("period_columns"):
                parts.append(f"Periods: {', '.join(metadata['period_columns'][:3])}")

        # Table structure info
        table_struct = doc.get("table_structure", {})
        if isinstance(table_struct, dict) and "row_range" in table_struct:
            row_range = table_struct["row_range"]
            parts.append(f"Rows: {row_range[0] + 1}-{row_range[1]}")

        # Cell references
        cell_ref = doc.get("cell_references")
        if cell_ref:
            parts.append(f"Ref: {cell_ref}")

        doc_copy["_citation"] = f"[{i + 1}] {' | '.join(parts)}"
        doc_copy["_citation_id"] = i + 1
        cited_docs.append(doc_copy)

    return cited_docs
