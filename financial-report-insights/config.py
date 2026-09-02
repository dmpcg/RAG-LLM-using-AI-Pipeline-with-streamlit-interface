"""
Centralized configuration for the RAG-LLM Financial system.
All hardcoded values are collected here with environment variable overrides.
"""

import logging
import os
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Load ALL env vars from .env (including OLLAMA_HOST for the ollama client)
_ENV_FILE = Path(__file__).parent / ".env"
if not _ENV_FILE.exists():
    logger.warning(
        ".env file not found at %s — using defaults. "
        "Copy .env.example to .env for custom configuration.",
        _ENV_FILE,
    )
load_dotenv(_ENV_FILE)


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 3
    max_top_k: int = 20

    # Hybrid search (BM25 + semantic)
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    rrf_k: int = 60
    # Per-system candidate pool before RRF fusion = top_k * this. A wider pool
    # lets a relevant chunk that ranks mid-list in one system still survive
    # fusion + MMR + parent-dedup down to top_k. Was hardcoded at 2.
    fusion_candidate_multiplier: int = 5
    # Drop fused results whose RRF score is below this floor (0.0 = no floor).
    min_rrf_score: float = 0.0

    # File upload limits
    max_file_size_mb: int = 200
    max_query_length: int = 2000

    # LLM
    llm_model: str = "llama3.2"
    embedding_model: str = "mxbai-embed-large"
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 2

    # Circuit breaker
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_seconds: int = 30

    # Embedding cache
    embedding_cache_dir: str = ".cache/embeddings"

    # LLM response cache
    llm_cache_dir: str = ".cache/llm_responses"
    llm_cache_size_limit_mb: int = 1000
    llm_cache_maxsize: int = 128  # In-memory LRU cache entries

    # Embedding
    embedding_dimension: int = 1024  # mxbai-embed-large; 0 = auto-probe
    embedding_quantization: str = "none"  # "none" | "int8" | "float16"
    embedding_truncation_dim: int = 0  # 0 = no truncation, else target dim
    embedding_batch_size: int = 32  # texts per batch for BatchEmbedder
    embedding_cache_version: str = "1.0"  # bump to invalidate all cached embeddings

    # Excel processing
    max_workbook_rows: int = 500_000

    # Financial analysis
    default_tax_rate: float = 0.25

    # API
    api_port: int = 8504
    cors_origins: str = "http://localhost:8501"  # Comma-separated allowed origins
    max_request_body_bytes: int = 1_048_576  # 1 MB max request body
    max_financial_fields: int = 200  # Max fields in a financial_data dict

    # Export
    export_max_ratios: int = 200  # Max ratios per export
    export_company_name: str = ""  # Default company name for exports

    # Query enhancement
    enable_hyde: bool = False  # HyDE query expansion (see note below)
    # HyDE default OFF (2026-06-24): HyDE generates a hypothetical answer via the
    # LLM and embeds THAT — nondeterministic (fresh draw per query) and it drifts
    # on specific figure-lookup queries, causing ±1 TOP1 swings at the gate
    # threshold. Raw-query embedding is deterministic and at least as accurate
    # here. Re-enable for broad/exploratory corpora where query<->doc vocabulary
    # mismatch is large.
    enable_query_decomposition: bool = True  # LLM-based query decomposition
    max_sub_queries: int = 4  # Max sub-queries for decomposition

    # Retrieval enhancement
    enable_reranking: bool = False  # Cross-encoder reranking (default OFF)
    reranking_model: str = "cross-encoder"  # Placeholder model name
    rerank_top_n: int = 20  # Candidates to rerank from initial retrieval
    mmr_lambda: float = 0.7  # MMR diversity parameter (1.0 = pure relevance, 0.0 = pure diversity)
    # MMR default OFF: this is a figure-lookup RAG where the single most
    # relevant chunk matters more than result diversity. A 2026-06-24 config
    # sweep on the golden set showed MMR demoting exact-figure matches
    # (DEEP-TOP1 1/4 -> 2/4 with MMR off) at no TOP3 cost; the dedup pass
    # already removed the near-duplicate redundancy MMR was guarding against.
    enable_mmr: bool = False  # Maximal Marginal Relevance diversification
    enable_citations: bool = True  # Citation tracking in responses
    enable_parent_expansion: bool = True  # Expand child chunks to parent text for LLM context

    # NOTE: the Phase 4.3 semantic-cache settings (semantic_cache_threshold,
    # semantic_cache_max_entries, enable_semantic_cache, enable_chunk_dedup,
    # adaptive_top_k) were removed alongside ml/semantic_cache.py and
    # ml/embedding_optimizer.py. They had no readers left, and the two that
    # defaulted True advertised behaviour that no longer exists.

    # Evaluation
    enable_evaluation: bool = False  # RAG evaluation harness

    # Prompt engineering (Phase 2.3)
    prompt_version: str = "2.0"  # Active prompt template version (for A/B testing)

    # Observability
    enable_tracing: bool = True  # Enable request tracing
    metrics_window_size: int = 10000  # Max metrics entries per rolling window

    # Vector index
    vector_backend: str = "auto"  # "auto" | "faiss" | "hnswlib" | "numpy"
    vector_index_path: str = "./data/vector_index"  # Persistence path prefix
    faiss_nprobe: int = 10  # FAISS IVF search parameter
    hnsw_m: int = 16  # HNSW connectivity parameter
    hnsw_ef: int = 50  # HNSW search-time ef parameter

    model_config = {"env_prefix": "RAG_"}


settings = Settings()


def validate_settings(s: Settings | None = None) -> Tuple[list[str], list[str]]:
    """Validate settings and return (errors, warnings). Empty errors = valid."""
    if s is None:
        s = settings
    errors: list[str] = []
    warnings: list[str] = []

    # --- OLLAMA_HOST URL validation (mirrors LocalEmbedder check) ---
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    parsed = urlparse(ollama_host)
    if parsed.scheme not in ("http", "https"):
        errors.append(
            f"OLLAMA_HOST scheme must be http or https, got: {parsed.scheme!r}"
        )

    # --- Neo4j consistency ---
    neo4j_uri = os.environ.get("NEO4J_URI", "").strip()
    neo4j_pass = os.environ.get("NEO4J_PASSWORD", "").strip()
    if neo4j_uri and not neo4j_pass:
        errors.append("NEO4J_URI is set but NEO4J_PASSWORD is empty")

    # --- Range validation ---
    if not (100 <= s.chunk_size <= 5000):
        errors.append(f"chunk_size must be 100-5000, got {s.chunk_size}")
    if s.chunk_overlap >= s.chunk_size:
        errors.append(
            f"chunk_overlap ({s.chunk_overlap}) must be < chunk_size ({s.chunk_size})"
        )
    if not (1 <= s.top_k <= s.max_top_k):
        errors.append(f"top_k must be 1-{s.max_top_k}, got {s.top_k}")
    if s.llm_timeout_seconds < 10:
        errors.append(
            f"llm_timeout_seconds too low: {s.llm_timeout_seconds} (min 10)"
        )
    if s.embedding_dimension not in (0, 384, 768, 1024):
        warnings.append(f"Unusual embedding_dimension: {s.embedding_dimension}")
    if s.max_file_size_mb > 500:
        warnings.append(f"max_file_size_mb is very large: {s.max_file_size_mb}")
    if s.bm25_weight + s.semantic_weight != 1.0:
        warnings.append(
            f"bm25_weight + semantic_weight = {s.bm25_weight + s.semantic_weight} "
            f"(expected 1.0)"
        )

    return errors, warnings
