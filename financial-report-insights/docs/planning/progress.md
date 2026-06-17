# Progress Log - AI/ML Enhancement

## Session: 2026-03-09

### Planning Phase [COMPLETE]
- [x] Launched 5 parallel research agents (RAG, embeddings, LLM, analysis/ML, test/quality)
- [x] All 5 agents returned comprehensive findings
- [x] Multi-agent brainstorming protocol executed (5 roles)
- [x] Phase 1: Primary Designer created master plan (6 workstreams, 21 phases)
- [x] Phase 2: Skeptic (6 objections), Constraint Guardian (6 constraints), User Advocate (5 UX concerns)
- [x] Phase 3: Integrator resolved all 17 objections, produced 10 decisions
- [x] Arbiter verdict: APPROVED
- [x] Plan saved to `docs/planning/task_plan.md`
- [x] Findings saved to `docs/planning/findings.md`

### Current Status
- **Tier 1**: NOT STARTED (awaiting user go-ahead)
- **Next action**: Begin Phase 1.1 (LLM Tracing) or user adjusts priorities

### Files Created This Session
| File | Purpose |
|------|---------|
| `docs/planning/task_plan.md` | Master plan with 5 tiers, 21 phases, MVP gates |
| `docs/planning/findings.md` | Research results from 5 agents + brainstorming review |
| `docs/planning/progress.md` | This file - session log |

### Decisions Made
| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Start with observability before RAG | Measure first, optimize second |
| D2 | FAISS with hnswlib fallback | FAISS has Windows issues |
| D3 | Reranking config-gated, default OFF | Latency concern for small doc sets |
| D4 | ML code in `ml/` subdirectory only | 15K LOC guardrail |
| D5 | Separate `requirements-ml.txt` | Docker image size |
| D6 | Pseudo-labels from Z/F-score thresholds | No labeled training data |
| D7 | Agent layer deferred to Tier 5 | Premature without WS1-4 |
| D8 | 20 curated Q&A for eval MVP | Quality over quantity |
| D9 | SHAP required for user-facing ML | Trust requires transparency |
| D10 | Citations required in every response | Core UX need |

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none) | | |

---

## Session: 2026-06-17 — RAG Chunking Remediation "A-prime" [COMPLETE, PR #48]

### Trigger
User reported launcher failing, then "0 chunks indexed" in the web UI.

### Diagnosis (root causes)
1. **Launcher:** killed `*streamlit*` by process name, but the server runs as `python.exe` → stale instance squatted port 8501. Fixed: kill the port OWNER by PID.
2. **"0 chunks":** `.env OLLAMA_HOST=12434` (Docker Model Runner, not running) → every embed refused. Fixed → `11434` (Ollama, live backend on this machine).
3. **Root retrieval defect (4-agent audit + adversarial swarm):** `chunk_excel_sheet` emitted ~1200-token atomic blocks (~11k chars) that overflowed mxbai's 512-tok window and were silently truncated to ~18% coverage; 82% of each chunk was empty-cell/alignment padding from `to_markdown`. Parent/child expansion was dead code.

### Decision
Expert swarm (5 lenses + 4 adversarial critics) → **"A-prime"**: fix chunking root cause, keep Ollama/mxbai, defer faiss + full eval-harness rebuild.

### Implementation
- `ingestion_pipeline._df_to_markdown` → dense rows (non-empty cells, no padding): 6.86M → 1.40M embed chars.
- `document_chunker.chunk_excel_sheet` → small parent-child CHILD chunks sized to the embed window; `_count_tokens_approx` table-aware.
- `local_llm` → warn-not-truncate, `max_batch_chars` 3000→12000, capped retry backoff.
- Tests: row-conservation + child-invariant guards (61/61 pass).
- `scripts/reembed_resumable.py` — checkpointed, hang-recovering bulk re-embed (Ollama CPU embed-runner deadlocks under load).
- `evaluation/run_golden_qa.py` + `golden_financial_qa.py` (git-ignored, confidential) — real-content gate.

### Result
609 oversized chunks → **1346 children** (max 1689 ch, 0 over window), rows conserved 4377==4377. **Gate PASS: TOP1 10/12, TOP3 12/12, DEEP-TOP1 4/4.** App live on 8501. Commits c821c92 + 2259da8 → PR #48 (base = integration branch).

### Errors Encountered
| Error | Resolution |
|-------|------------|
| Ollama `/v1/embeddings` deadlock under sustained CPU load | resumable embedder: checkpoint + auto-restart Ollama + resume |
| Re-embed est. 10-25 min was really ~5h | measured volume/throughput first; dense rendering cut content ~5x → ~50 min |

### Deferred follow-ups
- Hybrid BM25/keyword retrieval if numeric recall proves weak (pre-agreed #1).
- Eval-harness rebuild (currently gated off / synthetic goldens / not in CI); RRF `id(doc)` fragility; similarity threshold; optional faiss-cpu.
- `/clarity-gate` (pre-ingestion epistemic-quality gate) — parked.
