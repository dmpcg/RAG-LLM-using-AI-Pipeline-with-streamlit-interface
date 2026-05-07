# LOKI-PRD — Full-Stack Remediation, Optimization & Completion Plan

**Date generated:** 2026-05-07
**Audit base commit:** `f943cc8` (P0+P1 closed, 4,966 tests passing, working tree clean)
**Author:** Claude Code multi-agent reconnaissance (6 parallel audit agents)
**Target consumer:** `/loki-mode` orchestrator
**Output destination of work:** new branch `loki/2026-05-07-full-stack-remediation`

---

## 0. EXECUTIVE SUMMARY

A 6-agent parallel audit examined **every** module, test, workflow, infra file, and doc in `RAG-LLM-project`. Findings: **52 actionable items** across 5 severity tiers + a **156-item P2/P3 backlog** plus **15-item README drift list** plus **MEMORY.md split** plus **untracked-doc disposition**.

**Top correctness/security risks (must ship first):**
1. **9 unpatched dependency CVEs** — pydantic, jinja2, gitpython, urllib3, protobuf, idna, pymdown-extensions, aiohttp, starlette (sub-pin)
2. **`/health` endpoint DoS** — synchronous Neo4j `verify_connectivity` on every probe with default 60s timeout
3. **Silent data loss** — `_NEO4J_TRANSIENT` swallows transient writes returning `0` indistinguishable from "nothing to write"
4. **Compose lacks kernel hardening** — no `no-new-privileges`, `read_only`, `cap_drop`, `pids_limit`
5. **`AnalysisResults._dict_cache` invalidation bug** — fragile contract that will break on future caller order changes
6. **Domain formula bugs** — DSCR ignores existing debt service, NRR is actually GRR, CAC payback uses gross ARPU not GM-adjusted, debt capacity returns 0 for negative-EBITDA borrowers (overstates capacity), `current_ratio` rendered as percentage in exports (1.5x → 150%)

**Top performance wins (highest ROI):**
1. **Vectorize `detect_anomalies`** — 50–100× speedup, S effort
2. **Migrate 73 inline Plotly blocks to `viz_utils`** — `insights_page.py` shrinks ~25%, single biggest LOC win
3. **Stop re-instantiating `CharlieAnalyzer()` 80+ times per Streamlit rerun** — fixes UI lag on every interaction
4. **Vectorize Monte Carlo** — at 10k cap goes from 30–100s → seconds
5. **Cache reranker doc embeddings** — halves DMR round-trips per query when reranking is on

**Top completion gaps (features advertised but unwired):**
1. **Graph storage methods exist but never called** from app paths — `store_credit_assessment`, `store_covenant_package`, `store_portfolio_analysis`, `store_compliance_report`. Tests cover them; production never invokes them.
2. **`viz_utils.FinancialVizUtils`** has 12 well-tested helpers — not called anywhere in production. `self.viz` attribute is dead.
3. **Exports cover only ~10% of phases** — Monte Carlo, DCF, scenario, sensitivity, DuPont, anomaly, comparison, underwriting, startup, portfolio, compliance not in `export_full_report`.
4. **Reranker** wired but no integration test confirms activation; config-name typo silently disables.
5. **README drift** — claims sentence-transformers/all-MiniLM-L6-v2; reality is DMR/mxbai-embed-large 1024-dim. 15 specific lines wrong.

**Estimated total effort:** ~210 engineer-hours (P0 22h, P1 60h, P2 75h, P3 35h, docs/cleanup 18h).

---

## 1. SCOPE & SOURCES

### 1.1 Audit reports (all 6 agents returned)

| # | Domain | Owner agent | Findings (C/H/Cl) | Archive |
|---|--------|-------------|---|---------|
| 01 | Core analytics — `financial_analyzer.py`, `ratio_framework.py`, `structured_types.py`, `agents/`, `ml/`, `evaluation/` | code-review-expert | 5 / 8 / 6 | `.audit-2026-05-07/01-core-analytics.md` |
| 02 | Data + API — api/llm/vector/chunkers/parsers/graph/reranker/ingestion/protocols | code-review-expert | 4 / 9 / 9 | `.audit-2026-05-07/02-data-api.md` |
| 03 | UI — `streamlit_app_local.py`, `insights_page.py`, `viz_utils.py`, `app_local.py`, `prompts/` | code-review-expert | 4 / 8 / 7 | `.audit-2026-05-07/03-ui.md` |
| 04 | Domain + exports — `underwriting`, `startup_model`, `portfolio_analyzer`, `compliance_scorer`, `export_*` | code-review-expert | 6 / 10 / 7 | `.audit-2026-05-07/04-domain-exports.md` |
| 05 | Infra/security — Dockerfile, compose, CI workflows, healthcheck, observability, config, logging, scripts, security docs | security-auditor | 5 / 10 / 7 | `.audit-2026-05-07/05-infra-security.md` |
| 06 | Tests + backlog rollup + README drift + MEMORY split + findings.md | qa-expert | — / coverage gaps / 84 P2 + 72 P3 | `.audit-2026-05-07/06-tests-backlog.md` |

(Reports captured below in `Appendix A` with exact file:line citations.)

### 1.2 Already-closed (do NOT re-audit)
- All P0 (31 items) and P1 (59 items) from `.audit-2026-03-16/` — committed in `de08666`, `74fd953`, `f943cc8`.
- 3 CVEs already patched (fastapi/starlette/uvicorn upgrade chain, PyMuPDF, aiohttp).
- AST-based safe formula evaluator, Cypher regex sanitization, CORS hardening, rate-limiter, body-size limit, AnalysisResults type module, `_scored_analysis` pattern (40 phases converted).

---

## 2. SEVERITY-RANKED ACTION ITEMS

### TIER P0 — CORRECTNESS, SECURITY, DATA-LOSS RISKS (ship first, ~22 h)

| ID | Title | Files | Effort |
|----|-------|-------|--------|
| **P0-1** | Patch 9 unpatched dependency CVEs | `requirements.txt`, `requirements.lock` | 2h |
| **P0-2** | `/health` Neo4j-blocking DoS — wrap each preflight stage in `concurrent.futures` 3s timeout, cache result for 10s, reuse singleton driver | `healthcheck.py`, `graph_store.py` | 3h |
| **P0-3** | `_NEO4J_TRANSIENT` silent data loss — raise typed `Neo4jTransientError` (or return `(stored, transient)`) so caller can retry | `graph_store.py:140,295,354,427,482,708,767` | 2h |
| **P0-4** | Compose kernel hardening — `security_opt: [no-new-privileges:true]`, `cap_drop: [ALL]`, `read_only: true` + `tmpfs: /tmp`, `pids_limit`, `ulimits` for `rag-app` AND `neo4j` | `docker-compose.yml` | 1h |
| **P0-5** | `AnalysisResults._dict_cache` invalidation bug — invalidate on `__setattr__` OR rebuild every call OR remove the round-trip in `analyze()` (preferred, also kills the bug + test gap) | `structured_types.py:180-218`, `financial_analyzer.py:13758,13787` | 2h |
| **P0-6** | DSCR ignores existing debt service — add `existing_debt_service` to `LoanStructure` or estimate via `interest_expense + total_debt/weighted_term`; underwriting decisions today are wrong for any leveraged borrower | `underwriting.py:299-304` | 2h |
| **P0-7** | Debt-capacity returns 0 for negative-EBITDA — short-circuit to `max_additional_debt=None` with explicit assessment string | `underwriting.py:282-339` | 1h |
| **P0-8** | `_PERCENT_KEYWORDS` matches "ratio" — `current_ratio = 1.5` rendered as `150.00%` in XLSX/PDF exports | `export_utils.py:42`, `export_xlsx.py:188`, `export_pdf.py:269-270` | 1h |
| **P0-9** | NRR mislabeled — currently `1 - gross_churn` (i.e. GRR). Either rename the field to `gross_revenue_retention` or compute properly with multi-period expansion data | `startup_model.py:121-124` | 1h |
| **P0-10** | CAC payback uses gross ARPU — multiply by `gross_margin`; LTV/CAC uses raw LTV — switch to GM-adjusted LTV | `startup_model.py:182,193,198` | 1h |
| **P0-11** | `LocalEmbedder._client` connection leak — add `close()`, register atexit | `local_llm.py:443` | 1h |
| **P0-12** | Logging redaction misses Bearer JWT, AWS keys, `bolt://user:pw@host`, OpenAI/Anthropic SDK keys | `logging_config.py:14-17` | 2h |
| **P0-13** | `internal_growth_rate_analysis` divide-by-near-zero — replace manual `if abs(denom) > 1e-9` with `safe_divide(roa_b, 1 - roa_b)` | `financial_analyzer.py:8848-8852` | 0.5h |
| **P0-14** | Embedding/LLM retries lack jitter — thundering herd on DMR cold start; add `random.uniform(0, wait*0.25)` | `local_llm.py:340-353,541,552` | 0.5h |
| **P0-15** | Workflow `pr-review.yml` `execSync('wc -l')` injection on PR-supplied filename — use `fs.readFileSync` + `split('\n').length` | `.github/workflows/pr-review.yml:96-100` | 0.5h |
| **P0-16** | CORS validator must reject `*` when `allow_credentials=true` | `config.py:81`, `validate_settings()` | 0.5h |

**Parallelization:** P0-1, P0-2, P0-3, P0-4, P0-15 can run in parallel by `infra/security` agent; P0-5–P0-13 by `analytics/domain` agents; P0-11, P0-14 by `data-api` agent. **Single sprint, ~1 calendar day with the right swarm.**

---

### TIER P1 — CORRECTNESS / PERFORMANCE / RELIABILITY (~60 h)

#### P1.A — Performance hotpaths (15 h)
- **P1-A1**: Vectorize `detect_anomalies` (Python loop → numpy mask) — 50–100× speedup. `financial_analyzer.py:3500-3564`. **3h**
- **P1-A2**: Vectorize Monte Carlo — cache `dataclasses.fields()` once, perturb numpy arrays not per-iteration full re-analysis. `financial_analyzer.py:3568,3863-3885`. **5h**
- **P1-A3**: Cache reranker doc embeddings keyed by `chunk_id` — halves DMR round-trips. `reranker.py:69-71`. **2h**
- **P1-A4**: `/compare` quick-return when no period data instead of forcing full analyser pass per period. `api.py:466-501,508`. **1h**
- **P1-A5**: FAISS IVF — only retrain on power-of-two boundaries; today retrains every add past 10k. `vector_index.py:262-274`. **2h**
- **P1-A6**: Eager-import exporters at FastAPI lifespan instead of per-request lazy imports. `api.py:586,617,704,717,724`. **1h**
- **P1-A7**: `wait_for_embedding_service` per-stage budget — single `start` reference, break outer loop on cumulative timeout. `local_llm.py:583`. **1h**

#### P1.B — UI performance + completeness (12 h)
- **P1-B1**: Replace 80+ local `CharlieAnalyzer()` instantiations with `self.analyzer` (use `_get_analyzer()` or delete the dead shim). `insights_page.py`. **2h**
- **P1-B2**: Add `@st.cache_data` on per-tab analyser methods (`financial_rating`, `asset_efficiency_analysis`, `profitability_decomposition`, ~70 callers). **3h**
- **P1-B3**: Cache key uses `pd.util.hash_pandas_object(df).sum()` or `(path,mtime,sheet)` instead of `hash(str(df.to_dict()))`. `insights_page.py:258,1201`. **1h**
- **P1-B4**: DataFrame/WorkbookData out of `st.session_state` into `@st.cache_data`. **2h**
- **P1-B5**: Add empty-state fallbacks to all `try/except` chart blocks (`st.info("Insufficient data")` instead of silent skip). `insights_page.py` 13+ blocks. **2h**
- **P1-B6**: `st.spinner` / `st.status` around `analyzer.analyze()` call. `insights_page.py:260`. **0.5h**
- **P1-B7**: "Refresh Analysis" must clear all `analysis_*`, `_wb_*`, `export_cache_*` keys, not just three. `insights_page.py:336-340`. **0.5h**
- **P1-B8**: Delete dead UI: `submit_query` session-state read with no writer, `ollama_model` selectbox no-op. `streamlit_app_local.py:103-107,260`. **0.5h**
- **P1-B9**: Delete obsolete `_PROMPTS_AVAILABLE` flag + duplicated fallback prompt strings (Phase 2.3 prompts/ subtree is permanent). `app_local.py:1256-1265,1567-1576`. **0.5h**

#### P1.C — Resilience / async correctness (10 h)
- **P1-C1**: SSE per-chunk timeout via `asyncio.wait_for(..., timeout=settings.ollama_timeout_seconds)`. `api.py:285-294`. **2h**
- **P1-C2**: Promote circuit breaker `_on_success`/`_on_failure` → public `record_success`/`record_failure`; serialise streaming-failure recording behind the same lock as `call()`. `local_llm.py:287,289`. **2h**
- **P1-C3**: `parse_pdf` catches `fitz.FileDataError` / `RuntimeError`, returns empty `ParsedDocument`; `ingestion_pipeline.py` ingest_* logs with `exc_info=True`. **2h**
- **P1-C4**: `_send_embedding_batch` add explicit `raise RuntimeError("unreachable")` after retry loop (signature contract). `local_llm.py:511-560`. **0.5h**
- **P1-C5**: Rate-limiter eviction order — prune empty deques *before* append, not in unreachable post-append branch. `api.py:212-213`. **0.5h**
- **P1-C6**: `/analyze`, `/export/xlsx`, `/export/pdf` distinguish `LLMConnectionError` (503) from other `Exception` (422). `api.py:343-348,600,631`. **1h**
- **P1-C7**: `/documents` add `limit`/`offset`/`source` query params (default limit 100). `api.py:894-916`. **2h**

#### P1.D — Wire dead production code (10 h)
- **P1-D1**: Wire `store_credit_assessment` from `/analyze` underwriting branch. `api.py`, `graph_store.py:361`. **2h**
- **P1-D2**: Wire `store_covenant_package` from `/analyze` covenants branch. `graph_store.py:434`. **1h**
- **P1-D3**: Wire `store_portfolio_analysis` from `/portfolio/analyze` and `/portfolio/correlation`. `api.py:732,760`, `graph_store.py:645`. **2h**
- **P1-D4**: Wire `store_compliance_report` from `/compliance/analyze`. `api.py:804`, `graph_store.py:715`. **2h**
- **P1-D5**: `link_fiscal_periods` from compare endpoint. `graph_store.py:489`. **1h**
- **P1-D6**: Add integration test asserting reranker actually activates when `settings.enable_reranking=True` (today no test catches a config typo). **2h**

#### P1.E — Domain correctness round 2 (8 h)
- **P1-E1**: Diversification correlation — fix `int(40 * max(0, 1 - avg_corr))` (today 0 corr → only 20/40 points). `portfolio_analyzer.py:361`. **0.5h**
- **P1-E2**: Conditional formatting threshold — use 65 not 50 to match `score_to_grade`. `export_xlsx.py:455`. **0.5h**
- **P1-E3**: PDF unicode crash — add `_sanitize_text()` or embed DejaVuSans TTF via `pdf.add_font(uni=True)`. `export_pdf.py:205,210,244`. **2h**
- **P1-E4**: PDF table row-pagination — explicit page-break guard with header redraw. `export_pdf.py:221-258`. **1h**
- **P1-E5**: Scenario export memory cap — per-sheet entry caps + `constant_memory=True`. `export_xlsx.py:144,196-279`. **1h**
- **P1-E6**: Funding scenarios `raise_amount = max(0.0, …)` to block negative-dilution. `startup_model.py:311-312`. **0.5h**
- **P1-E7**: SOX score includes balance-sheet imbalance signal (today only MW + SD). `compliance_scorer.py:349-352`. **1h**
- **P1-E8**: Regulatory thresholds — gate Basel III citations by industry, or relabel as "general solvency floor". `compliance_scorer.py:110-171`. **1.5h**

#### P1.F — Observability + supply chain (5 h)
- **P1-F1**: Pin all GitHub Actions to commit SHAs (40-char), per-action comments. **2h**
- **P1-F2**: Add `permissions: contents: read` at workflow level in `ci.yml`, per-job elevation. **0.5h**
- **P1-F3**: Pin Docker base images to digest (`python:3.12.7-slim@sha256:…`, `neo4j:5-community@sha256:…`); add Renovate. **1h**
- **P1-F4**: `pip-audit -r requirements.lock --strict` becomes a hard CI gate. **0.5h**
- **P1-F5**: Starlette middleware exposing per-route Prometheus metrics + ingestion stage latencies; behind `RAG_ENABLE_METRICS_ENDPOINT`. `observability/metrics.py`. **1h**

---

### TIER P2 — STRUCTURAL IMPROVEMENTS (~75 h, drawn from .audit-2026-03-16 + new findings)

Grouped, with batch strategy and effort. Full rollup in `Appendix B`.

| Group | Items | Files | Effort |
|-------|-------|-------|--------|
| **Security headers / CORS** | CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, COEP middleware | `api.py` | 2h |
| **Financial edge cases** | NRR cleanup follow-on, CCC label, DCF/DSCR validation, `or 0` underwriting x4, ratio threshold docs | analyzer + underwriting + startup + ratio_framework | 5h |
| **Ingestion refinements** | chunk-id collision, streaming row constant, ingest_text size guard, PDF page limit, path validation | ingestion + parsers | 3h |
| **RAG pipeline** | pre-normalize at add, circuit breaker TOCTOU, LLMConnectionError wrapping, property injection sanitize | vector_index + local_llm | 3h |
| **Graph DB** | VECTOR_SEARCH LIMIT, bounded collect, dead Cypher cleanup, DDL transaction wrap | graph_schema + graph_store | 3h |
| **ML pipeline** | embed_with_cache assertion, EmbeddingCache size limit, AR float drift, SemanticCache O(n) → matrix cache, walk_forward public | ml/ sub-package | 5h |
| **Observability** | healthcheck exception text, trace log level, deque bounds, dashboard staleness | observability + healthcheck | 2h |
| **Export** | score_to_grade None guard, 500-row cap, cover text truncate, ExportRequest.company_name constraint | export_*.py | 2h |
| **Agent framework** | step timeout, AgentMemory lock, tool failure logging, eval_harness guard | agents/ | 4h |
| **Dependencies** | `pip-compile --upgrade --generate-hashes`, validate, ship lockfile | requirements.* | 2h |
| **Infrastructure** | repo-health.yml perms, Dockerfile per-service healthcheck timeout, docker-compose Neo4j CLI password file, CI coverage gate hard-fail | .github/ + docker | 2h |
| **Performance** | NumpyFlatIndex pre-normalize, SemanticCache matrix cache, EnsembleForecaster cache, ChunkDeduplicator LSH | analyzer + index + ml | 5h |
| **viz_utils migration (NEW)** | Migrate 73 inline Plotly blocks in insights_page.py to `FinancialVizUtils` helpers; estimated ~25% LOC shrink | insights_page + viz_utils | 18h |
| **Three composite-score unification (NEW)** | `financial_rating`, `comprehensive_health_score`, `financial_health_score_analysis` — pick canonical, deprecate others, document migration | financial_analyzer | 6h |
| **`_scored_analysis` framework upgrades (NEW)** | Add `derived_primary` + `band` modes; convert `internal_growth_rate_analysis`, `asset_lightness_analysis`, `payout_resilience_analysis`. Update MEMORY ("22 unconverted" is stale → real number is 6, of which 4 convertible) | financial_analyzer + ratio_framework | 8h |
| **Test debt** | session-scoped mutable fixture → function, `_make_embedder` to conftest, parametrize the ~122 phase test files (file count 122 → ~30), test_security imports real sanitizer, mock `time.sleep` in 3 slow files | tests/ | 5h |

---

### TIER P3 — POLISH (~35 h)

Aggregated from `.audit-2026-03-16/` P3 (72 items). Group-batchable:

- **Dead code cleanup** (22 items, ~5h): `ruff --fix` + manual review pass for orphaned methods, dead Cypher constants, unused imports in 8 modules
- **Naming & docs** (10 items, ~3h): NRR misnomer, CCC mislabel, X-XSS-Protection legacy header removal, ExportRequest.company_name doc note, request correlation IDs
- **Defensive coding** (15 items, ~5h): `AnalysisResults.__iter__` fresh dict, `_detect_seasonality` stub, regex `+` ambiguity, off-by-one grade boundary, ROA adjustment dollar comparison
- **Graph housekeeping** (5 items, ~2h): module-level `_portfolio_analyzer` singleton, dead READ Cypher constants
- **Logging** (7 items, ~2h): trace INFO → DEBUG sweep finish, missing context fields
- **Test quality** (5 items, ~3h): retry-comment accuracy, FAISS/HNSW score-value assertions, mtime sleep removal in test_model_registry, `pytest.raises(Exception)` narrowing
- **Build** (9 items, ~2h): remove `huggingface-cache` volume (DMR migration leftover), pin minor versions
- **Infrastructure** (4 items, ~1h): doc-review.yml perms, metrics-collector scope, bootstrap_labels.sh hardcoded org refs
- **Performance** (7 items, ~3h): document_chunker O(n²) whitespace normalization, BM25 index rebuild frequency

---

### TIER P4 — DOCS / MEMORY / ROOT-CRUFT (~18 h)

- **P4-1** README rewrite — 15 specific drift points (sentence-transformers→DMR/mxbai, port 8504, graph profile, test count 4,966, 14 missing modules, Python 3.13, `ollama-setup` profile gone, `huggingface-cache` volume gone, troubleshooting copy). **3h**
- **P4-2** `MEMORY.md` split into 6 topic files (state-commits, patterns, architecture, audit-history, features-history, shell-notes); thin index <80 lines. **2h**
- **P4-3** Commit `findings.md` with rename → `docs/excel-model-research.md` + 1-line header. Drives future `excel_processor.py`/`line_item_mapper.py` work. **0.5h**
- **P4-4** Consolidate 5 `SECURITY_*.md/.txt` files at module root into single `SECURITY.md` at repo root + `docs/security/` long-form. **2h**
- **P4-5** Root cruft: delete `LEARNING_INFRASTRUCTURE_SETUP_COMPLETE.txt`, move/delete `generate_readme_pdf.py` (→ `scripts/`), confirm `.swarm/state.json` not tracked, verify `README.pdf` not tracked. **1h**
- **P4-6** Move `scripts/consolidate.py` and `scripts/consolidate_insights.py` to `.audit-2026-03-16/` (one-shot tools from Feb 2026 refactor). **0.5h**
- **P4-7** Add coverage tests: `test_insights_page.py`, `test_local_llm.py`, end-to-end ingestion-to-retrieval integration, reranker activation, store_*_to_graph wiring. **8h**
- **P4-8** Save the 6 audit reports verbatim to `.audit-2026-05-07/` for historical traceability. **1h**

---

## 3. WORKSTREAMS — AGENT/SWARM ASSIGNMENTS

`/loki-mode` should orchestrate via `claude-flow swarm init --topology hierarchical-mesh --max-agents 6 --strategy specialized`. Each workstream = one feature branch (rebased onto trunk between merges) with one coordinator + 2–4 worker agents.

### WS-1 — Security & supply-chain hardening (P0-1, P0-2, P0-3, P0-4, P0-15, P1-F*, P0-12, P0-16)
- **Skills:** `security-review`, `dependency-management-deps-audit`, `pci-compliance`, `cicd-automation-workflow-automate`, `dotnet-or-python-security` patterns
- **Agents (4):** `coordinator`, `security-architect`, `security-auditor`, `devops-engineer`
- **Acceptance:** `pip-audit --strict` clean; CI workflows pinned to SHAs with minimal permissions; `/health` returns within 20s under Neo4j partition; compose has hardening directives; logging redaction covers JWT/AWS/bolt-URL/SDK keys.
- **Effort:** ~10 h
- **Branch:** `loki/ws1-security-supply-chain`

### WS-2 — Core analytics correctness + performance (P0-5, P0-13, P1-A1, P1-A2, P1-A4, P1-A5, P1-A6, P1-A7, P2 framework upgrades, three-composite unification)
- **Skills:** `python-pro`, `quant-analyst`, `performance-engineer`, `code-refactoring-refactor-clean`
- **Agents (4):** `coordinator`, `architect`, `coder`, `perf-engineer`
- **Acceptance:** `detect_anomalies` ≥50× faster on a 10k-row dataset benchmark; Monte Carlo 10k sims under 5s; `AnalysisResults` cache invalidates correctly with new test asserting post-construction mutation visibility; one canonical composite score path documented; new framework modes (`derived_primary`, `band`) unit-tested.
- **Effort:** ~22 h
- **Branch:** `loki/ws2-analytics-perf`

### WS-3 — Domain & exports correctness (P0-6, P0-7, P0-8, P0-9, P0-10, P1-E*, P2 financial edge cases, P2 export hardening)
- **Skills:** `quant-analyst`, `financial-modeling`, `risk-manager`, `code-review`
- **Agents (3):** `coordinator`, `domain-architect` (financial-modeler), `coder`
- **Acceptance:** New tests assert: DSCR uses existing debt service; debt capacity returns None for ≤0 EBITDA; current_ratio renders as `1.50x` not `150.00%`; CAC payback is GM-adjusted; `gross_revenue_retention` field name; PDF exporter handles em-dash, `µ`, smart-quotes; XLSX scenario export caps memory at 10k rows.
- **Effort:** ~14 h
- **Branch:** `loki/ws3-domain-exports`

### WS-4 — Data/API resilience + dead-code wiring (P0-11, P0-14, P1-A3, P1-C*, P1-D*)
- **Skills:** `api-design-principles`, `python-fastapi-development`, `error-handling-patterns`, `observability-engineer`
- **Agents (4):** `coordinator`, `backend-developer`, `tester`, `reliability-engineer` (sre-engineer)
- **Acceptance:** SSE streams cancel cleanly on per-chunk timeout; `/documents` paginates; circuit breaker public API used in `generate_stream`; `/analyze`/`/export/*` return 503 on LLM errors not 422; corrupt PDF returns empty doc not 500; portfolio/compliance/credit/covenant/period writes appear in Neo4j after `/analyze`/`/portfolio/*`/`/compliance/*` calls; reranker integration test catches activation regressions.
- **Effort:** ~18 h
- **Branch:** `loki/ws4-data-api-resilience`

### WS-5 — UI completeness, performance & accessibility (P1-B*, P2 viz_utils migration, P3 grade-color consolidation)
- **Skills:** `frontend-developer`, `ui-ux-designer`, `accessibility-compliance-accessibility-audit`, `react-or-streamlit-patterns`
- **Agents (3):** `coordinator`, `frontend-developer`, `ui-visual-validator`
- **Acceptance:** `insights_page.py` shrinks ≥20% LOC after viz_utils migration; no `CharlieAnalyzer()` outside `__init__`; `pd.util.hash_pandas_object` used for cache keys; every `try/except` chart block emits `st.info`; refresh button clears all dependent session keys; widget keys universally present; new `test_insights_page.py` covers the 130-tab routing and the most-used render methods.
- **Effort:** ~30 h
- **Branch:** `loki/ws5-ui`

### WS-6 — Tests, docs, MEMORY, and cleanup (P4-*, P3 cleanup, test debt)
- **Skills:** `test-driven-development`, `qa-expert`, `documentation-engineer`, `technical-writer`, `agent-memory-systems`
- **Agents (3):** `coordinator`, `qa-expert`, `documentation-engineer`
- **Acceptance:** README has 0 drift points (re-run audit-06 punch list); `MEMORY.md` ≤80 lines and split into 6 topic files; `findings.md` committed as `excel-model-research.md`; `SECURITY.md` consolidated; root cruft cleared; CI `pip-audit --strict` green; new coverage test files merged; `time.sleep` no longer real in `test_circuit_breaker.py` / `test_resilience.py` / `test_tracing.py`; `pytest -n auto` wall time drops ≥15 s.
- **Effort:** ~22 h
- **Branch:** `loki/ws6-docs-tests-cleanup`

---

## 4. EXECUTION PLAN

### 4.1 Phase ordering & dependencies

```
Phase 0: Setup           [WS-0  pre-flight]
   ├── create branch tree, archive audit reports, init swarm
Phase 1: P0 sweep        [WS-1 + WS-3 P0 only, parallel]   ← BLOCKS phase 2
Phase 2: P1 hot path     [WS-2, WS-4, WS-5 P1, parallel]
Phase 3: P2 sweep        [all WS in parallel, smaller items first]
Phase 4: P3 + cleanup    [WS-6 leads, others assist]
Phase 5: Docs & memory   [WS-6]
Phase 6: Final QA gate   [full test run, integration smoke, deploy dry-run]
```

Hard dependencies:
- WS-1 P0-1 (CVE patch) MUST land before any other branch starts (avoid stale lockfiles)
- WS-2 framework upgrades (`_scored_analysis` band-mode) MUST land before P2 conversion of methods 3, 4, 5
- WS-5 viz_utils migration MUST happen AFTER WS-3 export changes (export & UI both consume viz primitives)
- WS-6 README rewrite MUST happen LAST so the doc reflects committed reality

### 4.2 Concurrent-execution rules (per CLAUDE.md)

1. Spawn ALL workstream coordinators in ONE Task message with `run_in_background: true`
2. Each coordinator spawns its workers in ONE Task message
3. ZERO polling — `/loki-mode` synthesises when notifications arrive
4. ALL related operations in ONE message at every level
5. Memory: search `npx claude-flow memory search --query "[ws-name]" --namespace patterns` BEFORE starting a workstream; store outcome AFTER

### 4.3 Quality gates

Each branch merges to trunk only when:
- `python -m pytest tests/ -v` passes (4,966 + new tests)
- `pip-audit -r requirements.lock --strict` clean
- `ruff check .` clean
- New coverage ≥ existing coverage on touched modules
- For P0 items: regression test merged in same PR
- `docker compose --profile graph up` healthy probe under 20s
- README/MEMORY rewrite branch additionally: every README claim must have a passing source-of-truth check

### 4.4 Memory protocol

After each WS:
```bash
npx claude-flow memory store \
  --namespace patterns \
  --key "loki-2026-05-07-${WS_NAME}" \
  --value "[outcome summary, key files touched, test deltas]"
```

After all WS complete:
```bash
npx claude-flow memory store \
  --namespace patterns \
  --key "loki-2026-05-07-final" \
  --value "[total LOC delta, test count delta, CVE count, breaking changes]"
```

---

## 5. /LOKI-MODE INVOCATION CHEAT-SHEET

### 5.1 Initialise swarm

```bash
npx @claude-flow/cli@latest swarm init \
  --topology hierarchical-mesh \
  --max-agents 6 \
  --strategy specialized \
  --consensus raft \
  --memory hybrid

npx claude-flow memory search \
  --query "loki audit remediation" \
  --namespace patterns \
  --limit 10
```

### 5.2 Workstream-launch template (one message, all coordinators)

For each WS, spawn a coordinator Task with `run_in_background: true` and prompt template:

> "You are the WS-{N} coordinator. Read `financial-report-insights/docs/LOKI-PRD.md` section §3 WS-{N} and §2 Tier rows assigned to you. Spawn {workers} sub-agents in a single message with `run_in_background:true`, each owning a subset of P0/P1/P2 items. Synthesise their results into a single PR on branch `loki/ws{N}-{slug}`. Acceptance criteria are listed under WS-{N}. Run the test gate before declaring done. Update memory namespace `patterns` with key `loki-2026-05-07-ws{N}` on completion."

### 5.3 Skills to invoke (matched to workstreams)

| WS | Skills (Skill tool) |
|----|---------------------|
| WS-1 | `security-review`, `dependency-management-deps-audit`, `cicd-automation-workflow-automate`, `pci-compliance` |
| WS-2 | `python-pro`, `python-performance-optimization`, `code-refactoring-refactor-clean`, `quant-analyst` |
| WS-3 | `financial-modeling`, `risk-manager`, `excel-financial-modeling`, `code-review` |
| WS-4 | `python-fastapi-development`, `api-design-principles`, `error-handling-patterns`, `observability-engineer` |
| WS-5 | `frontend-developer`, `accessibility-compliance-accessibility-audit`, `ui-ux-designer`, `code-review-excellence` |
| WS-6 | `documentation-engineer`, `tdd-workflows-tdd-cycle`, `agent-memory-systems`, `technical-writer` |

### 5.4 Agent types (Task tool subagent_type)

WS-1: `security-engineer`, `security-auditor`, `cloud-architect`, `devops-engineer`
WS-2: `architect`, `python-pro`, `performance-engineer`, `quant-analyst`
WS-3: `financial-modeler`, `risk-manager`, `code-reviewer`
WS-4: `backend-developer`, `sre-engineer`, `qa-expert`, `error-detective`
WS-5: `frontend-developer`, `ui-designer`, `accessibility-tester`
WS-6: `qa-expert`, `documentation-engineer`, `context-manager`

### 5.5 Stop conditions

`/loki-mode` halts and surfaces to user when ANY of:
- A P0 fix introduces a new test regression that the responsible agent cannot resolve in 2 attempts
- Dependency upgrade breaks API contract (e.g. pydantic v2.8 minor breakage)
- A workstream takes >2× its budgeted hours
- More than 3 PRs need rebasing simultaneously (likely conflict pile-up)
- Trunk fails CI on the canonical branch

---

## 6. SUCCESS METRICS

| Metric | Baseline (today) | Target | Gate |
|--------|------------------|--------|------|
| Test count | 4,966 | ≥5,150 (+184 from new coverage modules) | pass |
| `pip-audit --strict` findings | 9 | 0 | hard |
| `insights_page.py` LOC | 7,987 | ≤6,200 | soft |
| `financial_analyzer.py` LOC | 13,907 | ≤13,500 (after composite unification + framework upgrades) | soft |
| README drift items | 15 | 0 | hard |
| `MEMORY.md` lines | 235 | ≤80 (index only) | hard |
| Monte Carlo 10k sims wall-time | 30–100s | ≤5s | hard |
| `detect_anomalies` 10k×20 | baseline | ≥50× faster | hard |
| `/health` worst-case latency under Neo4j partition | 60s | ≤3s | hard |
| Streamlit interaction lag (median) | re-instantiates 80 analyzers | 0 | hard |
| Graph-store writers wired | 1 of 5 | 5 of 5 | hard |
| Compose security_opt directives | 0 | 4+ | hard |
| CI actions pinned to SHA | 0 | 100% | hard |
| Logging redaction patterns | 5 | ≥10 (incl. JWT, AWS, bolt-URL, OpenAI/Anthropic SDK) | hard |

---

## 7. RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pydantic 2.7→2.8 breakage on `Settings` model | Medium | Medium | Pin compatible range first, run full test suite, fix call-sites |
| `_scored_analysis` framework upgrade breaks 40 already-converted methods | Low | High | New modes are additive (`derived_primary`/`band` opt-in); existing conversions unaffected; comprehensive existing test suite catches regressions |
| viz_utils migration introduces visual regressions across 130 tabs | Medium | Medium | Migrate by category tab; UI agent runs `ui-visual-validator` before merge; keep old path behind feature flag for one release |
| Graph-store wiring causes write contention / deadlocks under load | Low | Medium | Use UNWIND batching (already pattern); add `/health` Neo4j metrics; load-test via `k6-load-testing` skill |
| README rewrite drifts again before next release | High | Low | Add doc-review GitHub workflow (already exists at `.github/workflows/doc-review.yml`) — promote it to a hard CI gate |
| 22-month-old lockfile upgrade cascades into incompatible deps | Medium | High | One-step `pip-compile --upgrade --generate-hashes`, full test gate, rollback plan: revert lockfile commit |

---

## 8. APPENDIX A — FULL AUDIT FINDINGS (verbatim)

### A.1 Core Analytics (Agent 01)

[Complete report archived to `.audit-2026-05-07/01-core-analytics.md`. Key citations preserved in Tier P0/P1/P2 above. Headlines: 5 critical (cache invalidation, IGR divide, anomaly Python loop, MC re-runs, _scored_analysis derived-primary), 6 phase candidates (not 22), three competing composite models, four grade mappers.]

### A.2 Data + API (Agent 02)

[Complete report archived to `.audit-2026-05-07/02-data-api.md`. Headlines: /health blocks on Neo4j; embedder client leaks; embedding-batch may return None; retries lack jitter; rate-limiter eviction unreachable; 5 graph-store writers never called; parse_pdf misses fitz.FileDataError; circuit breaker private API misuse; reranker re-embeds; /compare double-loops; SSE no per-chunk timeout; /documents no pagination; transient Neo4j errors silent.]

### A.3 UI (Agent 03)

[Complete report archived to `.audit-2026-05-07/03-ui.md`. Headlines: charts crash silently; `submit_query`/`ollama_model` dead UI; CharlieAnalyzer re-instantiated 80+ times; `viz_utils.FinancialVizUtils` entirely unused (73 inline plotly blocks); cache key serialises whole DataFrame; whole DataFrame in session_state; refresh leaks state; missing widget keys; no spinner on long analyse; color-only encoding; obsolete `_PROMPTS_AVAILABLE` flag.]

### A.4 Domain + Exports (Agent 04)

[Complete report archived to `.audit-2026-05-07/04-domain-exports.md`. Headlines: debt capacity ignores negative EBITDA; DSCR ignores existing service; CAC payback uses gross ARPU; LTV missing GM; NRR is wrong (= GRR); `_PERCENT_KEYWORDS` matches "ratio"; exports cover ~10% of phases; fpdf2 unicode crash; PDF table no row-pagination; XLSX memory blow-up; conditional formatting threshold 50 vs grade 65; diversification asymmetric; SOX missing balance-sheet imbalance; Basel III applied to non-banks.]

### A.5 Infra + Security (Agent 05)

[Complete report archived to `.audit-2026-05-07/05-infra-security.md`. Headlines: 9 unpatched CVEs (pydantic, jinja2, gitpython, urllib3, protobuf, idna, pymdown-extensions, aiohttp, starlette sub-pin); Dockerfile/compose unpinned base + missing kernel hardening; healthcheck stage timeouts missing; CORS wildcard accepted; pre-commit not cross-platform; CI actions not SHA-pinned; permissions not minimal; pr-review.yml execSync injection risk; metrics-collector lacks --require-hashes; logging misses Bearer/AWS/bolt/JWT/SDK keys; validate_settings gaps; Dockerfile COPY chown waste; observability gaps (no per-route metrics, no W3C trace propagation); 5 SECURITY_*.md duplicates.]

### A.6 Tests + Backlog (Agent 06)

[Complete report archived to `.audit-2026-05-07/06-tests-backlog.md`. Headlines: 177 test files / ~4,966 tests grouped; coverage gaps `insights_page.py` `local_llm.py` `graph_schema.py` `protocols.py`; ~16s real `time.sleep` in 4 test files; session-scoped mutable fixture; `_make_embedder` duplicated 4×; `parametrize` chronically underused (~122 phase files → ~30); test_security tests local copy; 84 P2 + 72 P3 backlog rolled up by category with effort; 15-line README drift punch list; 6-file MEMORY.md split plan; findings.md commit-with-rename recommendation.]

---

## 9. APPENDIX B — P2 + P3 BACKLOG ROLLUP

(Full inventory: see `.audit-2026-05-07/06-tests-backlog.md` §5 and §6. Total P2 = 84 items, ~35 hr; total P3 = 72 items, ~25 hr; absorbed into Tiers P2 and P3 of this PRD.)

---

## 10. APPENDIX C — FILE INDEX

**Primary code (financial-report-insights/):** api.py, app_local.py, compliance_scorer.py, config.py, document_chunker.py, excel_processor.py, export_pdf.py, export_utils.py, export_xlsx.py, financial_analyzer.py, graph_retriever.py, graph_schema.py, graph_store.py, healthcheck.py, ingestion_pipeline.py, insights_page.py, line_item_mapper.py, local_llm.py, logging_config.py, pdf_parser.py, portfolio_analyzer.py, protocols.py, ratio_framework.py, reranker.py, startup_model.py, streamlit_app_local.py, structured_types.py, underwriting.py, vector_index.py, viz_utils.py
**Sub-packages:** agents/, ml/, evaluation/, observability/, prompts/, scripts/
**Tests:** financial-report-insights/tests/ (177 files, 4,966 tests)
**Infra:** Dockerfile, docker-compose.yml, requirements.txt, requirements.lock, ruff.toml, .githooks/pre-commit, .github/workflows/ (9 workflows)
**Audit archives:** .audit-2026-03-16/ (closed P0+P1), .audit-2026-05-07/ (this audit)
**Memory:** ~/.claude/projects/C--Users-dtmcg-RAG-LLM-project/memory/MEMORY.md (to split per P4-2)

---

**END PRD.** Hand to `/loki-mode` for execution.
