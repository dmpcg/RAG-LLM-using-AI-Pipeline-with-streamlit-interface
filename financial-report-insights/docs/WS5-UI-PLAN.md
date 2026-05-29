# WS-5 UI Completeness, Performance & Accessibility — Execution Plan

**Branch:** `loki/ws5-ui` (from `main` @ ff96369 — all of WS-1..4 merged; no stacking)
**Author (Primary Designer):** Claude Code
**Date:** 2026-05-29
**Source of truth:** Phase-2 recon + re-anchored on current main (2026-05-29)
**Baseline:** to be PINNED before Wave 1 — `python3.13 -m pytest tests/ -q` on `loki/ws5-ui` HEAD (expected ~5199); record exact count.
**Status:** DRAFT — pending brainstorming + clarity review
**3-swarm model:** Swarm-1 (this plan + brainstorming + clarity gates) → Swarm-2 (execution waves) → Swarm-3 (validate/test/iterate to green + acceptance).

---

## 1. Objective
Close WS-5: kill the per-rerun `CharlieAnalyzer()` churn, add caching/spinners/empty-states/refresh-correctness, delete dead UI, migrate inline Plotly to `FinancialVizUtils` where a helper exists, and add `test_insights_page.py`. DoD: all WP acceptance met; full suite green with no regression vs pinned baseline; `ruff` no new errors; insights_page.py LOC reduced (target trend toward ≤6,200, soft).

## 2. Verified current state (re-anchored on main, 2026-05-29)
| Item | State | Evidence |
|------|-------|----------|
| P1-B1 CharlieAnalyzer churn | not_done | `insights_page.py` has 80 `CharlieAnalyzer()` calls; 1 is `__init__` (`self.analyzer`), 79 are per-render-method re-instantiations rebuilt every Streamlit rerun. |
| P1-B2 per-tab caching | not_done | only 1 `@st.cache_data` in file (a static helper); ~70 per-tab analyzer calls uncached. |
| P1-B3 cache key | not_done | `:258` `hash(str(df.to_dict()))`; `:1199` `hash(str(financial_data))`. Should use `pd.util.hash_pandas_object`. |
| P1-B4 DataFrame in session_state | not_done | raw df/workbook stored in `st.session_state`. |
| P1-B5 silent chart blocks | partial | ~12 `except Exception: logger.debug(...)` chart blocks with no `st.info` empty-state. |
| P1-B6 spinner | not_done | no `st.spinner`/`st.status` around `self.analyzer.analyze(df)` (`:260`). |
| P1-B7 refresh leaks state | not_done | `:334` Refresh button clears only 3 fixed keys (`current_df`/`current_workbook`/`analysis_results`), not the dynamic `analysis_*` (per-tab `:258` AND export `:1199` — both build `analysis_<hash>`) or `_wb_*` (`:305`) keys. NOTE: there is NO `export_cache_*` prefix in source — the export cache key is `f"analysis_{hash(str(financial_data))}"` (`:1199`). |
| P1-B8 dead UI | not_done | `streamlit_app_local.py` `ollama_model` selectbox (no reader) + `submit_query` (no writer). |
| P1-B9 obsolete flag | not_done | `app_local.py:27/29/1248/1559` `_PROMPTS_AVAILABLE` + duplicated fallback prompt strings (prompts/ is now permanent). |
| P2 viz migration | partial | `FinancialVizUtils` IS wired (overview tabs); ~102 `go.Figure`/`st.plotly_chart`/`px.` inline blocks remain. |
| test_insights_page.py | ABSENT | acceptance gap — no UI render/routing test. |

## 3. Work packages
> **Shared-file constraint:** WP-B1..B7 + WP-VIZ all edit `insights_page.py` (7993 LOC) → they MUST serialize. `streamlit_app_local.py` (WP-B8), `app_local.py` (WP-B9), and `tests/test_insights_page.py` (WP-TEST) are parallelizable. Watch the 15K LOC guard (insights_page is well under, but VIZ should REDUCE it).

### WP-TEST — add tests/test_insights_page.py FIRST  *(M, ~7h — re-estimated)*  [enables safe refactor]
The safety net needs THREE distinct test layers because a bare `MagicMock` st cannot exercise the behaviors B1/B2/B4/VIZ change (see Decision Log D4–D7 and objections OBJ-1, OBJ-2, UA-1, UA-2). Build:

1. **A purpose-built `FakeStreamlit` stub** (NOT a bare `MagicMock`) with an explicit contract:
   - `columns(n)` / `tabs(list)` return correctly-sized lists of context-manager objects (each supports `__enter__`/`__exit__` and `.metric`/`.write`/...); `c1,c2,c3,c4 = st.columns(4)` must unpack. (UA-2: 173 `columns(...)`/`tabs(...)` sites + `with tab:` usage.)
   - `spinner`/`status` are context managers; `session_state` is a real dict-like supporting `in`, `del`, item get/set.
   - **Widget returns are configurable, not Mocks**: `selectbox`/`slider`/`checkbox` return scripted values per `key` (default to the first option / supplied default) so methods take their REAL branch, not the `Mock`-coerced fallback (OBJ-2, UA-1). The stub records calls (spy) for primitive assertions.
2. **Rich fixtures that drive methods PAST their data guards** (UA-1): a fully-populated `FinancialData`/`df` fixture so methods like `_render_net_profit_margin` (early-returns when `result.net_margin_pct is None`, `:7371`) and `_render_trend_forecast` (`base_val and base_val > 0`, `:2103`) reach the chart/Plotly path. Acceptance asserts the chart path executed (e.g. `st.plotly_chart` was called / a `go.Figure` was built), not merely "no exception."
3. **Real `st.cache_data` semantics tests for B2/B4** (OBJ-1): the caching/memoization assertions in WP-B2 do NOT use the FakeStreamlit stub — they import the REAL `streamlit` and call the actual module-level `@st.cache_data` fns, asserting memoization via a call-count spy on the wrapped pure function. (Under a mocked `st`, `st.cache_data` is a no-op and cannot validate caching.) These live in WP-B2's own tests, gated separately.

- **Acceptance:**
  - Stub-based: routing map (CATEGORY_TABS) exercised; ≥10 render methods run on the RICH fixture and the assertion confirms each reached its chart/analyzer path (guard-passed), not just "no raise".
  - Real-st: at least the B2 cached fns memoize (call-count spy) under genuine `streamlit`.
  - All green.

### WP-B1 — replace 79 local CharlieAnalyzer() with self.analyzer  *(M, ~2.5h)*
- In `insights_page.py`, replace every local `CharlieAnalyzer()` (outside `__init__`) with `self.analyzer`. (Verified: 80 `CharlieAnalyzer()` total — 1 is `self.analyzer` in `__init__`, 79 are per-render re-instantiations.)
- **`self`-in-scope check (OBJ-5):** before replacing, verify EACH of the 79 sites is inside an instance method of `FinancialInsightsPage` where `self` is bound — NOT a `@staticmethod`/`@classmethod`/module-level helper. (A `@staticmethod` already exists at `:7958` — `_extract_key_metrics` — proving the structural risk is real; it is NOT a `CharlieAnalyzer()` site but confirms the file mixes staticmethods in.) For any site lacking `self`, leave it bare (or refactor the method signature) — do not blindly substitute `self.analyzer`.
- **Local-import cleanup (UA-6):** each such method also has a co-located `from financial_analyzer import CharlieAnalyzer, <ResultType>` (e.g. `:7366`). After removing the local `CharlieAnalyzer()`, drop `CharlieAnalyzer` from that import (keep the result type if still referenced); if the import line becomes fully unused, delete it. This keeps the LOC-reduction and `ruff` (F401) gates coherent.
- **Test:** assert (via `inspect.getsource`) only ONE `CharlieAnalyzer(` outside `__init__`; render methods reuse `self.analyzer`; `ruff` reports no new F401.
- **Risk:** a local instance configured differently — grep-verify all are bare `CharlieAnalyzer()` (done: all 79 are bare; `CharlieAnalyzer` holds no mutable per-call state). **Acceptance:** ≤1 `CharlieAnalyzer()` call site; no per-rerun re-instantiation; no orphaned `CharlieAnalyzer` imports; all replaced sites had `self` in scope.

### WP-B3 — robust cache keys  *(S, ~1h)*
- Replace `hash(str(df.to_dict()))` (`:258`) and `hash(str(financial_data))` (`:1199`) with `pd.util.hash_pandas_object(df).sum()` (df) / a stable hash of the financial-data tuple. **Test:** same df → same key; mutated df → different key; key stable across runs.

### WP-B6 — spinner around analyze()  *(XS, ~0.5h)*
- Wrap the `self.analyzer.analyze(df)` call (`:260`) in `st.spinner("Analyzing…")`. **Test:** spinner context entered around analyze (mock st).

### WP-B7 — refresh clears ALL dependent keys  *(S, ~0.5h)*
- **Corrected prefixes (CG-1, UA-3):** there is NO `export_cache_*` prefix in source. The ACTUAL session-state keys are: the fixed names `current_df`/`current_workbook`/`analysis_results` (`:336-338`), the dynamic `analysis_<hash>` produced at BOTH the per-tab path (`:258`) and the export path (`:1199`), and `_wb_<file>` (`:305`). The Refresh button (`:334`) must iterate `st.session_state` and delete every key matching prefix `analysis_` or `_wb_`, plus the three fixed names (the `analysis_results` fixed name is covered by the `analysis_` prefix sweep, but keep it explicit for clarity). **Do NOT key on `export_cache_` — it would clear nothing.**
- **Plus the cache layer (see WP-B2/B4 / UA-4):** if B2/B4 move analysis behind module-level `@st.cache_data`, Refresh must ALSO call `st.cache_data.clear()` (or the scoped `.clear()` on each cached fn), because `@st.cache_data` lives OUTSIDE `session_state` and the key sweep alone won't bust it.
- **Test:** after refresh, no `analysis_*` or `_wb_*` session keys remain (assert against the REAL prefixes, with the dict pre-seeded with an `analysis_<hash>` and a `_wb_<file>` key so the assertion is non-vacuous); and the `@st.cache_data` layer is cleared (call-count spy shows recompute after Refresh).

### WP-B5 — empty-state fallbacks  *(S, ~2h)*
- Each silent `except Exception: logger.debug(...)` chart block emits `st.info("Insufficient data to render this chart")` (keep the debug log). **Test:** a render method whose chart raises shows an st.info (mock st).

### WP-B2 + WP-B4 — caching + DataFrame out of session_state  *(M, ~6h)*  [behavior-sensitive — gate scrutiny]
- B2: add `@st.cache_data` to the heavy per-tab analyzer computations where safe (module-level cached fns keyed by a hashable df digest, since `self` is unhashable). B4: keep the df out of `session_state`, behind `@st.cache_data`. **These two are the riskiest** (Streamlit cache semantics); the gate should confirm scope.
- **Mutation-after-retrieval audit (OBJ-4) — DO THIS BEFORE migrating:** `@st.cache_data` pickles return values, so each retrieval is a COPY, whereas today the SAME `AnalysisResults` object is reused across tabs via `session_state[cache_key]` (`:260-261`). `AnalysisResults` has a custom `__setattr__` that invalidates `_dict_cache` on every field write (`structured_types.py:186-193`) — i.e. post-retrieval mutation is an explicitly supported use case. **Before caching `analyze()` output, grep/audit whether ANY render method mutates the returned `AnalysisResults`/analysis dict after retrieval.** If any do, caching changes behavior (mutations hit a per-call copy, not a shared object) → either (a) cache only pure inputs and keep the shared object in session_state, or (b) make the cached fn return immutable/frozen data and route mutations elsewhere. Record the audit result in the gate.
- **Refresh contract (UA-4):** Today Refresh (`:336`) deletes session keys to force recompute. Once heavy analysis moves behind module-level `@st.cache_data`, deleting session keys NO LONGER invalidates it. WP-B7 is updated to also call `st.cache_data.clear()` so Refresh still busts the cache — otherwise users press Refresh and see stale numbers. This is a hard requirement, not optional.
- **Test:** (real `streamlit`, per WP-TEST layer 3) cached fn returns same result for same digest and is NOT recomputed on rerun (call-count spy on the wrapped pure fn); Refresh → cache cleared → recompute occurs (spy); and a regression test asserting no audited render method relies on mutating a shared post-`analyze()` object (or, if one does, that the chosen mitigation preserves the visible result).

### WP-B8 — delete dead UI  *(XS, ~0.5h)*
- `streamlit_app_local.py`: remove the `ollama_model` selectbox (no reader) and the `submit_query` session read (no writer). **Test:** symbols gone; app module imports + a smoke render works.

### WP-B9 — delete obsolete _PROMPTS_AVAILABLE  *(S, ~0.5h)*
- `app_local.py`: remove the `_PROMPTS_AVAILABLE` flag + the duplicated fallback prompt strings (`:27/29/1248/1559`); prompts/ is permanent. **Test:** flag gone; prompt-dependent paths still work.

### WP-VIZ — migrate inline Plotly to FinancialVizUtils  *(L, ~8h scoped)* [LOC lever — gate to scope]
- **Explicit in-scope set (UA-5) — no fuzzy "where a helper exists":** Wave 3 migrates ONLY the high-frequency repeated chart types — simple **bar**, **line/scatter**, and **gauge** blocks — to `self.viz` helpers (adding `viz_utils.py` helpers for any not yet present). Before execution, produce the concrete in-scope list: grep-classify the 102 blocks (verified count: 102) into {bar, line/scatter, gauge, bespoke}; the in-scope list is the bar/line/gauge subset and is pinned in the gate with a **countable target (migrate ≥ the full bar/line/gauge subset, target ≥40 blocks; if the classified subset is smaller, the target is that exact count)**. Bespoke one-off charts (waterfalls, multi-panel, custom-annotated figures) are explicitly DEFERRED and listed by line number so deferral is auditable. An executor migrating near-zero blocks FAILS the gate.
- **Figure-level regression guard (OBJ-3) — close the detection gap:** "render methods still run" is NOT a regression guard (a dropped trace, swapped axis/color, or changed `use_container_width` passes it). For migrated blocks, add **figure-equivalence assertions**: in the FakeStreamlit stub, `st.plotly_chart` captures the `go.Figure` argument; for each migrated render method, snapshot the PRE-migration figure (trace count, trace types, x/y data per trace, axis titles, `use_container_width` flag) on the rich fixture, then assert the helper produces an equivalent figure POST-migration. New `viz_utils.py` helpers also get standalone unit tests asserting the figure they build (traces, layout) for representative inputs.
- **Test:** the in-scope subset is migrated (count meets the pinned target); per-migrated-method figure-equivalence assertions pass; `viz_utils.py` helper unit tests green; deferred-list documented in the PR.

## 4. Sequencing & parallelism
```
Wave 0 (parallel): WP-TEST (tests/test_insights_page.py) ; WP-B8 (streamlit_app_local.py) ; WP-B9 (app_local.py)   [disjoint files; TEST is the safety net]
Wave 1 (insights_page.py serial): WP-B1 -> WP-B3 -> WP-B6 -> WP-B7 -> WP-B5
Wave 2 (insights_page.py serial, behavior-sensitive): WP-B2 + WP-B4
Wave 3 (insights_page.py + viz_utils.py serial): WP-VIZ (scoped)
Gate after each wave: full suite green (incl. the new test_insights_page.py) + ruff; per-wave commit.
```

## 5. Quality gates (per WP)
1. WP-TEST lands first as the refactor safety net — and per D4–D7 it is THREE layers (FakeStreamlit stub with scripted widget returns + rich guard-passing fixtures + REAL-`st` cache tests), because a bare MagicMock cannot validate cache semantics (OBJ-1) or drive widget-return-gated branches (OBJ-2). 2. Test-first for behavior changes; for pure refactors (B1) the stub render tests on RICH fixtures (assert chart path reached, not just "no raise") are the equivalence guard; for VIZ the equivalence guard is the figure-level snapshot assertions (OBJ-3), NOT "runs without raising". 3. Targeted tests green, then full suite green (no regression vs pinned baseline). 4. `ruff` no new errors (incl. no orphaned `CharlieAnalyzer` F401 from B1). 5. Per-wave commit; pre-commit hook (LOC guard + suite) passes. 6. For B2/B4 the gate records the mutation-after-retrieval audit (OBJ-4) and confirms Refresh busts `@st.cache_data` (UA-4).

## 6. Decision Log
| # | Decision | Rationale |
|---|----------|-----------|
| D1 | test_insights_page.py authored FIRST (Wave 0) | UI refactors (B1/B2/B4/VIZ) need a render-level safety net; none exists today |
| D2 | WP-B2 caching via module-level @st.cache_data fns (not on methods) | `self` is unhashable; Streamlit cache needs hashable args — key on a df digest |
| D3 | WP-VIZ scoped to high-frequency chart types + new shared helpers; bespoke one-offs deferred | 102-block full migration is visual-regression-prone with no Streamlit render harness; gate to confirm |
| D4 (OBJ-1) | Cache-semantics tests (B2/B4) use REAL `streamlit`, not the mocked stub | Under MagicMock `st`, `st.cache_data` is a no-op decorator and CANNOT prove memoization. RESOLVED: WP-TEST layer 3 imports real `streamlit` and spies call-count on the wrapped pure fn. |
| D5 (OBJ-2 / UA-1) | Stub scripts widget RETURN values; fixtures are rich enough to pass data guards; assert chart path reached | A MagicMock return value makes widget-gated branches take the fallback/None path, so "runs without raising" guards near-empty code. RESOLVED: FakeStreamlit returns scripted selections; rich fixture drives methods past `is None`/`>0` guards; acceptance asserts `plotly_chart`/`go.Figure` reached. |
| D6 (UA-2) | FakeStreamlit stub has an explicit contract (sized iterables for `columns`/`tabs`, context managers, dict-like `session_state`); ~+3h estimate | Bare MagicMock can't unpack `c1,c2,c3,c4 = st.columns(4)` (`:7386`) or act as a context manager (`with tab:`). RESOLVED: stub contract specified; WP-TEST re-estimated 4h→7h. |
| D7 (OBJ-3 / UA-5) | VIZ gets figure-level equivalence assertions + an explicit, countable in-scope block list (bar/line/gauge); bespoke deferred-by-line-number | "Render runs" detects no figure regression and lets VIZ pass doing ~zero work. RESOLVED: snapshot trace/axis/flag equivalence per migrated method; pin classified bar/line/gauge subset with a countable target; deferred list auditable. |
| D8 (OBJ-4) | Mandatory mutation-after-retrieval audit before caching `analyze()` output | `@st.cache_data` returns a pickled COPY; today the SAME `AnalysisResults` (custom `__setattr__`, `structured_types.py:186-193`) is reused across tabs — post-retrieval mutation would silently diverge. RESOLVED: audit render methods for post-`analyze()` mutation; if any, cache pure inputs / freeze return. |
| D9 (CG-1 / UA-3) | WP-B7 clears the REAL prefixes `analysis_` + `_wb_` (+ fixed names), NOT the nonexistent `export_cache_` | Export cache key is `analysis_<hash>` (`:1199`), same prefix as per-tab (`:258`); `export_cache_` matches nothing → vacuous test + stale exports. RESOLVED: clear by `analysis_`/`_wb_`; tests pre-seed real keys so assertions are non-vacuous. |
| D10 (UA-4) | Refresh must also call `st.cache_data.clear()` once analysis moves behind module-level cache | Cache lives outside `session_state`; deleting session keys alone leaves stale numbers after Refresh. RESOLVED: WP-B7 + WP-B2/B4 require cache `.clear()` on Refresh; test asserts recompute via spy. |
| D11 (OBJ-5) | WP-B1 verifies each of the 79 sites has `self` in scope before substituting | A `@staticmethod` exists at `:7958` (no `self`); blind `self.analyzer` substitution would break such a site. RESOLVED: per-site `self`-in-scope check; leave bare / refactor signature where absent. |
| D12 (UA-6) | WP-B1 also removes the orphaned local `CharlieAnalyzer` imports it leaves behind | Each local instantiation has a co-located `from financial_analyzer import CharlieAnalyzer, ...` (`:7366`); leaving it triggers ruff F401 and undercuts the LOC gate. RESOLVED: drop `CharlieAnalyzer` from the local import (keep result types still used); ruff F401 gate enforces. |
| _… review agents append …_ | | |

## 7. Exit criteria
- [ ] test_insights_page.py exists + green — 3 layers: FakeStreamlit stub (scripted widget returns) + rich guard-passing fixtures (≥10 render methods, assert chart path reached) + real-`st` cache tests; ≤1 CharlieAnalyzer() site (all replaced sites had `self` in scope; no orphaned `CharlieAnalyzer` imports); robust cache keys; spinner; refresh clears `analysis_*`/`_wb_*` keys AND `st.cache_data.clear()`; empty-states; dead UI + _PROMPTS_AVAILABLE removed; viz migration (scoped, countable target) done with per-method figure-equivalence assertions; B2/B4 mutation-after-retrieval audit recorded
- [ ] Full suite green, no regression vs pinned baseline; ruff clean (no new); insights_page.py LOC reduced
- [ ] Decision Log complete; brainstorming APPROVED; clarity check passed
