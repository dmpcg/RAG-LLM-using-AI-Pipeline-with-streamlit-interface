# Research Findings - AI/ML Enhancement

## Date: 2026-03-09
## Source: 5 parallel research agents (RAG pipeline, embeddings, LLM integration, analysis/ML, test/quality)

---

## Current RAG Pipeline State

### Chunking
- **Text**: Word-based split, 500 words, 50 overlap (`app_local.py:474-484`) - NAIVE, loses sentence boundaries
- **Excel**: Row-based, 10+ rows/chunk, markdown table format (`excel_processor.py:649-738`)
- **Detection**: `_detect_statement_type()` classifies income/balance/cash_flow/budget by keyword scoring
- **Gap**: No recursive/hierarchical chunking, no header propagation across table chunks

### Search & Retrieval
- **Hybrid**: BM25 (weight 0.4) + Semantic (weight 0.6) with RRF fusion (k=60)
- **BM25**: `rank-bm25` library, whitespace tokenization only - NO stemming, NO stop-words
- **Semantic**: Neo4j HNSW (optional) + numpy brute-force (fallback)
- **Fusion**: RRF at `app_local.py:632-685`, static weights from `config.py:28-30`
- **Graph**: 1-hop traversal only (`graph_retriever.py:18-61`)
- **Gap**: No reranking, no HyDE, no MMR, no citation tracking, no context compression

### Embedding
- **Model**: `mxbai-embed-large` (1024-dim) via Docker Model Runner
- **Cache**: JSON per-file, keyed by SHA256(filename:content_hash:model_name) (`app_local.py:260-313`)
- **Performance**: Numpy vectorized cosine sim, `argpartition` for O(n) partial sort
- **Gap**: No FAISS/hnswlib local ANN, no quantization, no Matryoshka truncation

### LLM Integration
- **Provider**: Ollama (llama3.2), configurable via `OLLAMA_MODEL`
- **Resilience**: Circuit breaker (3-state, 3 failures threshold, 30s recovery)
- **Cache**: OrderedDict LRU, 128 entries, SHA256(model:prompt) key
- **Streaming**: SSE via `sse-starlette`, `answer_stream()` with async wrapper
- **Prompts**: Standard RAG + Charlie Munger financial framework (`app_local.py:900-920`)
- **Gap**: No token counting, no cost tracking, no prompt versioning

### Analysis & ML
- **Ratio methods**: 180+ on CharlieAnalyzer (13,832 LOC)
- **Frameworks**: DuPont, Altman Z-Score, Piotroski F-Score, Composite Health
- **Stochastic**: Monte Carlo (capped 10K sims), DCF with Gordon Growth Model
- **Scoring**: ComplianceScorer (SOX), UnderwritingAnalyzer (credit 0-100), PortfolioAnalyzer
- **Anomaly**: Z-score + IQR statistical methods only
- **Gap**: No sklearn, no model training, no time-series models, no clustering, no SHAP

### Testing & Quality
- **Tests**: 3,662 passing, 151 files
- **Linting**: Ruff with bandit security (S rules enabled)
- **CI**: 5-stage GitHub Actions pipeline
- **Security**: 35+ dedicated security tests (injection, rate-limit, sanitization)
- **Gap**: No mypy, thin Streamlit UI coverage, no RAG evaluation harness

### Observability
- **Logging**: Structured JSON/text, secrets redaction regex
- **Security**: SSRF hostname allowlist, rate limiting (60/min), body size limit (1MB)
- **Gap**: No tracing, no LLM metrics, no RAG quality dashboard, no cost tracking

---

## Key File Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| Text chunking | `app_local.py` | 474-484 |
| Excel chunking | `excel_processor.py` | 649-738 |
| Embedder | `local_llm.py` | 364-448 |
| Vector search | `app_local.py` | 529-598 |
| BM25 search | `app_local.py` | 600-630 |
| RRF fusion | `app_local.py` | 632-685 |
| Graph enrichment | `graph_retriever.py` | 18-61 |
| LLM generation | `local_llm.py` | 220-254 |
| Circuit breaker | `local_llm.py` | 19-165 |
| LLM cache | `local_llm.py` | 199-251 |
| Embedding cache | `app_local.py` | 260-313 |
| Financial prompt | `app_local.py` | 900-920 |
| Query decomp | `app_local.py` | 752-833 |
| Config | `config.py` | 1-76 |
| API endpoints | `api.py` | 191-258 |
| Logging | `logging_config.py` | 1-80 |
| Anomaly detect | `financial_analyzer.py` | 3497 |
| Monte Carlo | `financial_analyzer.py` | 3810 |
| DCF forecast | `financial_analyzer.py` | 3914 |
| Ratio framework | `ratio_framework.py` | full file |

---

## Brainstorming Review Findings

### Skeptic Objections (All Accepted)
1. **Scope explosion** - 21 phases / 50-65 days needs MVP gates every 10-14 days
2. **FAISS on Windows** - Poor support; hnswlib fallback required
3. **Reranker latency** - 200-500ms added; must be config-gated
4. **No training data for ML** - Use Z-score/F-score as pseudo-labels
5. **Agent layer premature** - Deferred to Tier 5, only after WS1-4 validated
6. **Eval ground truth** - Start with 20 curated pairs, expand incrementally

### Constraint Guardian (All Accepted)
1. **15K LOC guardrail** - All ML in `ml/` subdirectory, never in `financial_analyzer.py`
2. **Dependency weight** - Separate `requirements-ml.txt`
3. **Memory footprint** - Budget config + FAISS IVF + int8 quantization
4. **Test runtime** - `@pytest.mark.slow` for ML, fast suite < 2 min
5. **Single-user arch** - Agents in-process, no distributed state
6. **DMR reranker** - Try DMR first, fallback to lightweight sentence-transformers

### User Advocate (All Accepted)
1. **Latency vs quality** - Show progress indicators with span timing
2. **Citation trust** - Citations are WS1 MVP gate, not optional
3. **ML explainability** - SHAP required for ALL user-facing predictions
4. **Eval visibility** - Surface confidence scores to users
5. **Agent transparency** - Show agent activity as progress steps

---

## RAG Pipeline Audit Findings (2026-06-17, 4 read-only agents + adversarial swarm)

Verified against source during the "A-prime" remediation (see progress.md; PR #48).

**P0 (fixed):**
- Excel `chunk_excel_sheet` emitted ~1200-token atomic blocks, never child chunks → ~11k-char chunks overflowing mxbai's 512-tok window, truncated to ~18% coverage.
- `_count_tokens_approx` (`words*1.3`) under-counts markdown tables ~3x → "within budget" blocks overflow.
- `_df_to_markdown` via pandas `to_markdown` → 82% empty-cell/alignment padding (embeds whitespace; ~5x volume).
- Parent/child expansion (`_expand_parent_chunks`) was dead code — Excel chunks never set `chunk_level='child'`+`parent_id`+`parent_text`.
- `.env OLLAMA_HOST=12434` (DMR, not running) → "0 chunks indexed"; live backend is Ollama 11434 (CPU-only, no GPU).

**P1 (deferred):**
- Eval harness non-functional as a gate: `enable_evaluation=False`, not in CI, synthetic golden Q&A referencing files absent from repo, can't detect truncation.
- Vector index silently on O(n) `NumpyFlatIndex` (faiss/hnswlib not installed); RRF fusion keyed on fragile `id(doc)`; no similarity threshold; result `score` never set (avg_similarity always 0); index `.load()` doesn't validate dimension; BM25 over-aggressive stemming.

**Infra:** Ollama's CPU embedding runner deadlocks under sustained load (`/v1/embeddings` hangs, `/api/ps` OK) → resumable embedder required.

**Accepted risk:** if numeric recall stays weak after chunking fix, hybrid BM25/keyword retrieval is the #1 follow-up (general-purpose embedders may not distinguish e.g. 1,240 vs 1,420).
