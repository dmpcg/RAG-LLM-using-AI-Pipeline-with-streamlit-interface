# AI/ML Enhancement Master Plan

## Goal
Exhaustive enhancement of the RAG-LLM financial-report-insights system across 6 workstreams: RAG quality, embedding optimization, LLM evaluation, ML pipeline, AI agent layer, and observability.

## Status: IMPLEMENTED (Tier 1, 2.1/2.2/2.3, 4.2, Tier 5 DONE; remaining items PARTIAL — module built but not wired into live api.py/app_local.py, or simplified algorithm)

> Reality check (2026-06-02): `ml/`, `observability/`, `evaluation/`, `prompts/`, and `agents/` are all implemented with passing tests. Per-phase markers below reflect implemented reality. "PARTIAL" means the core module exists and is tested, but it is either not yet wired into the live request path or uses a simplified algorithm vs. the original plan.

## Architecture Diagram

```
                    +---------------------+
                    |  WS1: RAG Quality   | <-- Foundation
                    |  (Reranking, HyDE,  |
                    |   chunking, MMR)    |
                    +--------+------------+
                             |
              +--------------+--------------+
              v              v              v
     +------------+  +------------+  +------------+
     | WS2: Vector|  | WS3: LLM   |  |WS6: Observe|
     | Optimization|  | Evaluation |  | & Metrics  |
     | (FAISS,    |  | (RAG eval, |  | (tracing,  |
     |  quant)    |  |  prompts)  |  |  costs)    |
     +-----+------+  +-----+------+  +-----+------+
           |               |               |
           +-------+-------+               |
                   v                       |
          +----------------+               |
          | WS4: ML Pipeline|<-------------+
          | (classification,|  (needs metrics)
          |  time-series,  |
          |  clustering)   |
          +-------+--------+
                  v
          +----------------+
          | WS5: AI Agent  | <-- Capstone
          | Layer (multi-  |
          |  agent finance)|
          +----------------+
```

---

## TIER 1: Foundation (~10-14 days)
**Gate**: Measure retrieval P@5, MRR before/after. Latency acceptable?

### Phase 1.1 - LLM Tracing [status: DONE]
- [ ] Add trace_id generation per user query
- [ ] Span tracking: embedding, search, reranking, LLM generation, total
- [ ] Token counting from Ollama response metadata
- [ ] Cost estimation per query (compute-equivalent)
- **Files**: `logging_config.py`, new `observability/tracing.py`
- **Est**: 2-3 days

### Phase 1.2 - Advanced Chunking [status: DONE]
- [ ] Sentence-aware text chunking (replace word-split at `app_local.py:474-484`)
- [ ] Excel column-header propagation in each chunk (`excel_processor.py:714-718`)
- [ ] Chunk metadata enrichment (source_page, statement_type, period, section_title)
- [ ] Chunk quality scoring (filter low-info chunks before embedding)
- **Files**: `app_local.py`, `excel_processor.py`
- **Est**: 2-3 days

### Phase 1.3 - Query Enhancement [status: DONE]
- [ ] HyDE (Hypothetical Document Embeddings) - LLM generates hypothetical answer, embed that
- [ ] LLM-based query decomposition (replace keyword expansion at `app_local.py:783-833`)
- [ ] Query classification (ratio lookup vs trend vs comparison vs explanation)
- [ ] BM25 tokenization upgrade (stop-words, stemming via nltk)
- **Files**: `app_local.py`, `local_llm.py`
- **Est**: 2-3 days

### Phase 1.4 - Retrieval Enhancement [status: PARTIAL]
<!-- Reranker (reranker.py), MMR, contextual compression, citation tracking DONE; multi-hop graph traversal (2-hop) not yet implemented in graph_retriever.py (single-hop only). -->

- [ ] Cross-encoder reranking (config-gated, default OFF for <50 chunks)
- [ ] MMR (Maximal Marginal Relevance) for result diversification
- [ ] Contextual compression (extract relevant sentences from chunks)
- [ ] Multi-hop graph traversal (2-hop in `graph_retriever.py:18-61`)
- [ ] Citation/provenance tracking (chunk IDs, source file, page/row numbers)
- **Files**: new `reranker.py`, `app_local.py`, `graph_retriever.py`
- **Est**: 3-4 days

### TIER 1 GATE: Human review required before proceeding

---

## TIER 2: Measurement & Optimization (~10-14 days)
**Gate**: Eval harness green? Index performing? Dashboard showing improvements?

### Phase 2.1 - RAG Eval Harness [status: DONE]
- [ ] Create 20 golden Q&A pairs from real financial reports
- [ ] Implement retrieval metrics: Precision@k, Recall@k, MRR, NDCG
- [ ] Answer quality metrics: faithfulness, relevance, completeness
- [ ] LLM-as-judge scoring (1-5 scale)
- [ ] CI integration: fail if metrics drop >5%
- **Files**: new `evaluation/` directory
- **Est**: 3-4 days

### Phase 2.2 - FAISS/hnswlib Index [status: DONE]
- [ ] Test `faiss-cpu` install on Windows (fallback: `hnswlib`)
- [ ] Replace numpy brute-force for >1K chunks
- [ ] Index persistence to disk (.faiss files)
- [ ] Auto-rebuild on new document addition
- **Files**: new `vector_index.py`, `app_local.py`
- **Est**: 2-3 days

### Phase 2.3 - Prompt Engineering [status: DONE]
- [ ] Prompt versioning in `prompts/` directory
- [ ] Few-shot examples (2-3 exemplar Q&A) in financial prompt
- [ ] JSON-mode structured output for ratio queries
- [ ] Chain-of-thought for complex analytical queries
- [ ] Prompt A/B testing framework
- **Files**: new `prompts/`, `app_local.py:900-920`
- **Est**: 2-3 days

### Phase 2.4 - RAG Quality Metrics Dashboard [status: PARTIAL]
<!-- Metrics collection (observability/metrics.py) + dashboard_data.py implemented and tested; the Streamlit metrics dashboard page is not yet built. -->

- [ ] Average similarity scores, RRF weight effectiveness, cache hit rates
- [ ] Query analytics: types distribution, avg retrieval time, popular topics
- [ ] Answer quality tracking over time
- [ ] Embedding drift detection
- **Files**: new `observability/metrics.py`, Streamlit dashboard page
- **Est**: 2-3 days

### TIER 2 GATE: Human review required before proceeding

---

## TIER 3: Intelligence (~12-16 days)
**Gate**: ML predictions trustworthy? Forecasts reasonable? Memory budget met?

### Phase 3.1 - Feature Engineering [status: PARTIAL]
<!-- ml/feature_engineering.py implemented and tested, but not wired into the live api.py/app_local.py request path. -->

- [ ] Ratio-to-feature-vector extraction (180+ ratios -> standardized vector)
- [ ] Temporal features (deltas, growth rates, moving averages)
- [ ] Category encoding (industry, company size, report type)
- [ ] Feature scaling (StandardScaler/RobustScaler pipeline)
- [ ] Feature selection (mutual information, correlation filtering)
- **Files**: new `ml/feature_engineering.py`
- **Est**: 2-3 days

### Phase 3.2 - Classification Models [status: PARTIAL]
<!-- ml/classifiers.py (sklearn LogisticRegression/RandomForest) implemented and tested; explainability is a SHAP-like approximation, not the real `shap` library (xgboost not used). -->

- [ ] Financial distress prediction (Z-score < 1.81 = distressed pseudo-label)
- [ ] Models: LogisticRegression, RandomForest, XGBoost
- [ ] Cross-validation, confusion matrix, ROC-AUC, precision-recall
- [ ] SHAP explainability (REQUIRED for all user-facing predictions)
- **Files**: new `ml/classifiers.py`
- **Est**: 3-4 days

### Phase 3.3 - Time Series Forecasting [status: PARTIAL]
<!-- ml/forecasting.py implements SimpleARModel (AR(p) via OLS) + exponential smoothing with prediction intervals; true ARIMA/SARIMA and Prophet are not integrated. -->

- [ ] ARIMA/SARIMA (replace linear regression at `financial_analyzer.py`)
- [ ] Prophet integration for revenue/margin forecasting
- [ ] Ensemble forecasting (weighted averaging)
- [ ] Proper prediction intervals (not just point estimates)
- [ ] Walk-forward backtesting validation
- **Files**: new `ml/forecasting.py`
- **Est**: 3-4 days

### Phase 3.4 - Embedding Optimization [status: PARTIAL]
<!-- ml/embedding_optimizer.py (quantization, Matryoshka truncation) implemented and tested, but not wired into the live embedding path in local_llm.py. -->

- [ ] int8 quantization (4x memory reduction)
- [ ] Matryoshka truncation test (512-dim, 256-dim)
- [ ] Batch size profiling for DMR API
- [ ] Embedding versioning with auto-invalidation
- **Files**: `local_llm.py`, `vector_index.py`
- **Est**: 2-3 days

### TIER 3 GATE: Human review required before proceeding

---

## TIER 4: Polish & Scale (~8-12 days)
**Gate**: System production-ready? All metrics green?

### Phase 4.1 - Clustering & Peer Analysis [status: PARTIAL]
<!-- ml/clustering.py (K-means/DBSCAN/PCA) implemented and tested, but not wired into the live api.py/app_local.py request path. -->

- [ ] K-means peer grouping by financial profile
- [ ] DBSCAN outlier detection
- [ ] PCA/UMAP 2D visualization
- [ ] Cluster profiling (auto-descriptions)
- **Files**: new `ml/clustering.py`
- **Est**: 2-3 days

### Phase 4.2 - Response Quality [status: DONE]
- [ ] Hallucination detection (cross-ref LLM claims vs retrieved chunks)
- [ ] Confidence scoring (retrieval scores + source coverage)
- [ ] Answer grounding (inline citations required)
- **Files**: `app_local.py`, `evaluation/`
- **Est**: 1-2 days

### Phase 4.3 - Caching & Performance [status: PARTIAL]
<!-- ml/semantic_cache.py (embedding-similarity cache) implemented and tested, but not wired into the live query path in app_local.py. -->

- [ ] Semantic cache (query embedding similarity > 0.95 = cache hit)
- [ ] Chunk deduplication (MinHash/SimHash)
- [ ] Adaptive top-k based on query complexity
- **Files**: `app_local.py`
- **Est**: 1-2 days

### Phase 4.4 - System Dashboard [status: pending]
- [ ] Prometheus-style counters/gauges/histograms
- [ ] Streamlit monitoring page
- [ ] Alerting rules (circuit breaker, latency, error rate)
- [ ] Performance baselines
- **Files**: `observability/`, Streamlit page
- **Est**: 1-2 days

### Phase 4.5 - Model Infrastructure [status: PARTIAL]
<!-- ml/registry.py (joblib model registry + metadata) implemented and tested; prediction API endpoints in api.py are not yet exposed. -->

- [ ] Model registry (joblib + metadata JSON)
- [ ] Training pipeline CLI
- [ ] Prediction API endpoints
- [ ] Model/data/feature drift monitoring
- **Files**: `ml/registry.py`, `api.py`
- **Est**: 2 days

### TIER 4 GATE: Human review required before proceeding

---

## TIER 5: Capstone - AI Agents (Future, ~10-14 days)
**Gate**: Agents add value beyond single-query RAG?

### Phase 5.1 - Agent Framework [status: DONE]
- [ ] Agent base class with tool-calling, memory, planning loop
- [ ] Tool registry (all analysis methods as callable tools)
- [ ] Short-term + long-term memory system
- [ ] Orchestrator for complex query routing
- **Files**: new `agents/` directory
- **Est**: 3-4 days

### Phase 5.2 - Specialized Agents [status: DONE]
- [ ] Ratio Analyst agent
- [ ] Risk Assessor agent
- [ ] Trend Forecaster agent
- [ ] Portfolio Manager agent
- [ ] Due Diligence agent
- [ ] Report Writer agent
- **Est**: 3-4 days

### Phase 5.3 - Multi-Agent Workflows [status: DONE]
- [ ] Sequential chains (ratio -> anomaly -> root cause)
- [ ] Parallel fan-out (profitability + liquidity + leverage + efficiency)
- [ ] Human-in-the-loop confirmation gates
- [ ] Agent collaboration via structured memory
- **Est**: 2-3 days

### TIER 5 GATE: Human review

---

## Constraints & Rules

| Constraint | Rule |
|-----------|------|
| **15K LOC** | All ML code in `ml/` subdirectory. ZERO additions to `financial_analyzer.py` |
| **Dependencies** | Separate `requirements-ml.txt`. Core RAG works without ML deps |
| **Memory** | Add memory budget config. Use FAISS IVF + int8 quantization |
| **Test runtime** | `@pytest.mark.slow` for ML tests. Fast suite < 2 min |
| **Architecture** | Agents are in-process, not distributed services |
| **Reranker** | Config-gated (`settings.enable_reranking`), default OFF for <50 chunks |
| **ML labels** | Use Z-score/F-score thresholds as pseudo-labels. Document limitations |
| **Explainability** | SHAP required for ALL user-facing ML predictions |
| **Citations** | Required in every RAG response (part of WS1 MVP) |
| **FAISS fallback** | Try `faiss-cpu` first, fallback to `hnswlib` on Windows |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |

## New Dependencies (by tier)

### Tier 1
- `nltk` (sentence tokenization, stemming)
- `sentence-transformers` (cross-encoder reranking, lightweight)

### Tier 2
- `faiss-cpu` or `hnswlib` (ANN index)
- `ragas` or custom eval (RAG evaluation)

### Tier 3
- `scikit-learn` (classifiers, clustering, scaling)
- `xgboost` (gradient boosting)
- `prophet` (time-series forecasting)
- `shap` (model explainability)

### Tier 4
- `datasketch` (MinHash deduplication)
- `prometheus-client` (metrics export, optional)

### Tier 5
- No new deps (uses existing LLM + tool registry)
