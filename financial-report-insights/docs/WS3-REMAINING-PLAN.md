# WS-3 Remaining-Items Execution Plan

**Branch:** `loki/ws3-domain-exports`
**Author (Primary Designer):** Claude Code
**Date:** 2026-05-28
**Source of truth:** verification pass on this branch + direct code reads (not commit messages)
**Status:** FINALIZED — multi-agent-brainstorming **APPROVED** (0 unresolved objections); clarity-gate substantively satisfied (C2 citations re-verified against source by maintainer; C8 silent-cap inventory complete; C9 reconciled). Ready for execution.

---

## 1. Objective

Close out WS-3 ("Domain & exports correctness") so that **every** acceptance
criterion in `LOKI-PRD.md §3 WS-3` is met, tested, and validated. The P0 tier
(P0-6…P0-10) is now committed. This plan covers the **remaining** P1-E tier and
the WS-3-scoped P2 items.

**Definition of done:** all work packages below merged on `loki/ws3-domain-exports`;
full suite green with **no regressions vs. the branch baseline** (capture the
current count via `python3.13 -m pytest tests/ -q` at WP start; do NOT hard-code a
target — MEMORY.md records 4966 at last commit, the actual `loki/ws3-domain-exports`
baseline is unverified, and a fixed numeric gate either trivially passes or blocks
arbitrarily — see D12); each package ships its own regression test(s); PRD WS-3
acceptance line satisfied for PDF unicode and XLSX memory cap; `ruff check .` clean
on touched files. **All five PRD WS-3 P2 "Financial edge cases" sub-items
(`LOKI-PRD.md:155`: NRR cleanup follow-on, CCC label, DCF/DSCR validation,
`or 0` underwriting ×4, ratio threshold docs) are addressed in WP-7 (b–f); none is
silently dropped (CG-2).**

---

## 2. Verified current state (baseline)

| Item | PRD claim | **Verified actual state (2026-05-28)** | Verdict |
|------|-----------|----------------------------------------|---------|
| P1-E1 | Diversification corr scores 0 corr as 20/40 (bug) | `portfolio_analyzer.py:358-361` already maps `[-1,1]→[40,0]`, 0→20, documented | **Already fixed** — test only |
| P1-E2 | XLSX conditional-format threshold 50 vs grade 65 | `export_xlsx.py` has **zero** `conditional_format` calls — feature absent | **Moot / descope** unless color-coding is desired |
| P1-E3 | fpdf2 unicode crash | `export_pdf.py` core "Helvetica" font (latin-1); `_format_value`/labels emit raw strings; no `_sanitize_text`. NOTE: cell display already truncates to 50 chars (`:257`), cover/period to 100 (`:327,:333`) — truncation is **not** sanitization | **Real gap** (acceptance) |
| P1-E4 | PDF table no row-pagination / header redraw | `_add_table` (`export_pdf.py:222`) has no in-loop page-break + no header redraw | **Real gap** |
| P1-E5 | XLSX scenario memory blow-up | `export_xlsx.py` uses `{"in_memory": True}`; no row cap in `export_scenario_comparison` | **Real gap** (acceptance) |
| P1-E6 | Funding `raise_amount` can be negative | `startup_model.py:331` `raise_amount = s.get("raise_amount", 0)` — no guard | **Real gap** (1-line) |
| P1-E7 | SOX score ignores balance-sheet imbalance | `compliance_scorer.py` `sox_compliance` scores off `material_weakness` only; BS imbalance lives in a separate consistency check (`:431`), not fed into SOX risk | **Real gap** |
| P1-E8 | Basel III applied to non-banks | `compliance_scorer.py` already relabels `"Basel III (inspired)"`; no industry gating | **Partial** — decide gate vs relabel-only |

---

## 3. Work packages

Each package is independently shippable. File ownership is disjoint except for
shared-file pairs that must serialize: WP-3/WP-4 (both `export_pdf.py`),
WP-2/WP-8 (both `export_xlsx.py`), WP-1/WP-7 (`startup_model.py`), and
WP-5/WP-7 (`compliance_scorer.py`). See §5 for the exact ownership map.

### WP-1 — P1-E6: guard negative raise amounts  *(XS, ~0.5h)*
- **File:** `startup_model.py` (~line 331, `funding_scenarios`)
- **Change:** `raise_amount = max(0.0, float(s.get("raise_amount", 0) or 0.0))` ONLY. **Do NOT clamp `pre_money_valuation`** (see D5 / OBJ-3): clamping pre_money to 0 silently zeros `implied_arr_mult = safe_divide(pre_money, arr)` (`startup_model.py:349`) and collapses `post_money = pre_money + raise_amount`, changing dilution semantics — an untested behavior change beyond the acceptance line for an "XS, Risk: none" package. Negative pre_money is out of WS-3 scope (P1-E6 names only `raise_amount`).
- **Tests:** negative `raise_amount` → clamped to 0, dilution 0, no negative new-cash. (No pre_money test — pre_money behavior is unchanged.)
- **Risk:** none (scope limited to `raise_amount`). **Acceptance:** funding scenarios never produce negative dilution.

### WP-2 — P1-E5: cap XLSX scenario export memory  *(S, ~1h)*
- **File:** `export_xlsx.py` (`export_scenario_comparison`, workbook ctor)
- **Change:** open workbook with `{"constant_memory": True}` for the scenario path (or expose a flag); add a per-sheet row cap (`_MAX_EXPORT_ROWS = 10_000`) and a truncation note row when exceeded; log the drop (no silent cap).
- **constant_memory ordering (OBJ-4):** under `constant_memory`, only *row-data* writes must be strictly top-to-bottom; `set_column`/`freeze_panes` are tolerated but must still occur in valid positions. Two existing calls need explicit handling:
  - `ws.freeze_panes(row+1, 0)` is called per-scenario mid-stream (`:248`). Under `constant_memory` freeze_panes is metadata, not a row write — keep it, but the **test must assert the workbook reopens** (e.g. via `openpyxl.load_workbook`) to prove no out-of-order failure.
  - `ws.set_column(...)` (`:286-287`) runs AFTER all row writes — already valid; leave as-is.
- **Truncation boundary (OBJ-4):** the `row` counter spans MULTIPLE scenarios in one sheet (`:216-284`). **Decision (D6):** the cap is enforced at a **whole-scenario boundary** — before writing each scenario's header block, if `row + projected_scenario_rows > _MAX_EXPORT_ROWS`, stop and emit ONE truncation note row; never truncate mid-scenario (no partial/corrupt scenario blocks). `projected_scenario_rows = len(all_keys) + header + impact + gap`.
- **Precedence vs WP-8 (UA-5 / D11):** the scenario export path is **EXEMPT** from WP-8's generic 500-row cap. The 10k scenario cap is authoritative for `export_scenario_comparison`; WP-8's 500-row cap applies only to the other tabular paths. Stated here so no engineer applies 500 to the scenario sheet.
- **Tests:** >10k synthetic rows across multiple scenarios → output capped at the last whole scenario that fits + exactly one truncation note present; assert no scenario block is partially written (last written scenario has its full key set); workbook reopens via `openpyxl`.
- **Risk:** `constant_memory` forbids out-of-order *row* writes — covered by reopen assertion above. **Acceptance:** scenario export caps memory at 10k rows.

### WP-3 — P1-E3: PDF unicode safety  *(M, ~2h)*
- **File:** `export_pdf.py`
- **Change:** add `_sanitize_text(s)` that maps common offenders (em/en dash→`-`, `µ`→`u`, smart quotes→ASCII `'`/`"`, `≥`→`>=`, `≤`→`<=`, `…`→`...`) and, as a backstop, `.encode("latin-1","replace").decode("latin-1")`. Route **every** string sink through it. The sinks are enumerated and each MUST be wrapped (OBJ-6): `_add_table` cell render (`:257`, before the 50-char truncation), executive_summary (`:89`), interpretation (`:191`, `:457`), cover title/company/period/generated (`:298-336`), F-score string sub-values (`:399`). Funnel through a single choke point where feasible (e.g. sanitize inside `_add_table` cell loop and `_format_value`) so coverage is structural, not per-call.
- **Tests (OBJ-6 / UA-6 — strengthened):** the "does not raise" check is INSUFFICIENT (fpdf2 with replace-decode won't raise on a missed sink — it silently emits a corrupt glyph). Instead: (1) assert `_sanitize_text` directly on the substitution table (em-dash→`-`, `µ`→`u`, smart-quote→ASCII) — exact output equality; (2) for the rendered PDF, parse the emitted text (via `pypdf`/`pdfplumber` text extraction) and assert the ASCII substitution string appears and no non-latin-1 byte survives — proving the substitution actually occurred, not merely that no exception was raised. Note `_add_table` truncates display to 50 chars, so place sanitization BEFORE truncation and keep the test substring within the first 50 chars.
- **Risk:** double-sanitization harmless (idempotent map). **Acceptance:** PDF exporter handles em-dash, µ, smart-quotes — verified by extracted-text substitution assertions, not "no raise".

### WP-4 — P1-E4: PDF table pagination + header redraw  *(S, ~1h)*
- **File:** `export_pdf.py` (`_add_table`)
- **Change:** extract header-draw into a helper; before each data row, if `pdf.get_y() + row_h > page_break_trigger` then `add_page()` and redraw the header row.
- **auto_page_break interaction (OBJ-7):** `auto_page_break` is set globally in `_create_pdf` (`:204`) and the ratio loop has its OWN manual break check (`if pdf.get_y() > 240`, `:131-133`) that calls `_add_table` per category. To avoid double breaks / blank pages: (a) inside `_add_table`, save the incoming state with `pdf.auto_page_break` / `pdf.b_margin`, call `pdf.set_auto_page_break(False)` on entry, and **restore the original setting on exit** (try/finally) so callers relying on auto-break are unaffected; (b) the caller's pre-`_add_table` `get_y() > 240` check stays as a coarse pre-table guard — it triggers at most one page break *before* the table starts, which is harmless and does not interleave with `_add_table`'s internal per-row breaks. Document that `_add_table` owns breaks only for its own rows.
- **Tests:** 80-row table spans >1 page; assert page count >1 AND header text appears on each page (extract per-page text via `pypdf`); **additionally assert no blank page** — every page has non-empty extracted text — to catch spurious/double breaks (OBJ-7). Assert `pdf.auto_page_break` is back to its original value after `_add_table` returns.
- **Risk:** interaction with global `auto_page_break` — handled by save/disable/restore above. **Acceptance:** multi-page tables repeat the header; no blank/double pages.

### WP-5 — P1-E7: SOX score reflects balance-sheet imbalance  *(S, ~1h)*
- **File:** `compliance_scorer.py` (`sox_compliance`)
- **Tolerance (UA-3 — pinned):** reuse the SAME 1%-of-assets tolerance already used by the `sec_filing_quality` consistency check (`compliance_scorer.py:427`, `safe_divide(bs_diff, total_assets) < 0.01`). That literal is currently inline; **extract it to a module constant `_BS_IMBALANCE_TOLERANCE = 0.01`** and reference it from BOTH the existing consistency check and the new SOX check, so there is one tolerance, not two divergent ones.
- **Double-counting mechanism (OBJ-5 — concrete, not a wish):** `overall_audit_risk` extends `restatement_indicators` with `sox.material_weakness_indicators` (`:637-638`) AND with `sec.red_flags` (`:641-642`). The BS-imbalance signal already exists as a `sec` red_flag (`:430-433`). Therefore:
  - **Do NOT** append the BS-imbalance string to `sox.material_weakness_indicators` (that would re-flow into `overall_audit_risk` at `:638` AND still exist via `sec.red_flags` at `:642` → triple representation).
  - Instead, have `sox_compliance` *read* the same imbalance condition to **adjust only its own `risk_score`** via a dedicated, non-`material_weakness` field (e.g. a `bs_imbalance_penalty` applied to `risk_score` locally), and DO NOT add it to any list consumed by `overall_audit_risk`. Net effect: SOX risk_score reflects imbalance; `overall_audit_risk` still sees it exactly once (via the existing `sec.red_flags` path), unchanged.
- **Data fixture:** balanced fixture = `total_assets=1000, total_liabilities=600, total_equity=400` (diff 0); imbalanced = `total_assets=1000, total_liabilities=600, total_equity=300` (10% diff, well above 1%).
- **Tests:** imbalanced BS → SOX `risk_score` drops + SOX-local penalty present; balanced BS → SOX `risk_score` unchanged; **AND assert `overall_audit_risk` is unaffected** (its `restatement_indicators` count is identical with vs. without the SOX change, proving no double-count).
- **Risk:** double-counting — eliminated by the mechanism above (SOX adjusts its own score only). **Acceptance:** SOX includes balance-sheet imbalance signal; overall_audit_risk unchanged.

### WP-6 — P1-E1: diversification regression test  *(XS, ~0.25h)*
- **File:** `tests/test_portfolio_analyzer.py`
- **Change:** **test only** (code already correct).
- **How to isolate `corr_pts` (OBJ-2):** `corr_pts` is NOT exposed on `DiversificationScore` (only `correlation_penalty=round(avg_corr,4)` is stored, `portfolio_analyzer.py:371`). `total = rev_pts + ast_pts + corr_pts` (`:363`). To assert the correlation component deterministically, **hold `rev_pts` and `ast_pts` constant by using a single-holding / perfectly-concentrated portfolio** so HHI is maximal (`hhi_rev = hhi_ast = 1.0` → `rev_pts = ast_pts = 0` per the `(1 - hhi)/(1 - ideal_hhi)` formulas at `:351-356`). Then `total == corr_pts` and the test reads `corr_pts` directly off `overall_score`:
  - avg_corr = -1 → `overall_score == 40`
  - avg_corr =  0 → `overall_score == 20`
  - avg_corr = +1 → `overall_score == 0`
  Drive `avg_correlation` by constructing the correlation input (or monkeypatching `correlation.avg_correlation`) to the three pinned values. This isolates the corr component without reverse-engineering from variable HHI inputs.
- **Risk:** none. **Acceptance:** regression locks the documented [-1,1]→[40,0] mapping with `corr_pts` isolated via maximal-HHI fixture.

### WP-7 — P1-E8 + P2 financial-edge: regulatory gating + edge cleanups  *(M, ~3.5h)*
> Decisions below are now RESOLVED (not open), so each sub-item is independently shippable (OBJ-9). Split into six atomic commits, each with its own acceptance.
> **PRD coverage (CG-2):** the PRD WS-3 P2 "Financial edge cases" group (`LOKI-PRD.md:155`) names **five** sub-items — NRR cleanup follow-on, CCC label, DCF/DSCR validation, `or 0` underwriting ×4, ratio threshold docs. All five are dispositioned here: (a) Basel III is P1-E8 (not part of the five); (b) `or 0` ×4; (c) CCC label; (d) DCF/DSCR validation; (e) NRR cleanup follow-on; (f) ratio threshold docs. No P2 financial-edge sub-item is left without an in-scope verdict.
- **Files:** `compliance_scorer.py`, `underwriting.py`, `ratio_framework.py`, `startup_model.py`, `financial_analyzer.py` (sub-items (c) CCC label and (d) DCF/DSCR validation live here — see C3/C5). **Core-module note:** `financial_analyzer.py` is the 15K-LOC-guarded core module; WP-7 therefore exercises the §6 item-5 pre-commit LOC guard. The (c)/(d) changes are label/sign-only and small denominator guards (no new methods, negligible LOC delta), so the guard is expected to pass; the engineer must still confirm the LOC count stays under 15K post-edit.
- **(a) P1-E8 Basel III — RESOLVED to relabel-only (CG-1):** gating "by industry field" (option a) is **NOT implementable in WS-3 scope** — `FinancialData` (`financial_analyzer.py:2150`) has NO `industry`/`sector` attribute; `industry` exists only as a string *parameter* to `industry_benchmark` (`:4670`, default `"general"`), never as input state on the analyzed entity. Adding a data field is out of WS-3 scope. **Decision:** keep the existing `"Basel III (inspired)"` relabel; no gating. Acceptance: citation reads "(inspired)"; assert no industry-conditional branch was added.
- **(b) P2 underwriting `or 0` ×4 — RESOLVED:** replace each of the four `... or 0` expressions with explicit None-aware handling (treat missing input as "not evaluable", not as 0, where 0 changes a ratio). Acceptance: per-site test that a `None` input yields `None`/"N/A" rather than a misleading 0-derived value.
- **(c) CCC label — RESOLVED & PINNED (C5):** the CCC *formula* is correct everywhere (DSO + DIO − DPO at `financial_analyzer.py:2692-2705`, `:4364-4367`, `:8299-8305`, `:10004`) — do **NOT** touch the formula. The defect is a label/sign-wording mismatch in the negative-CCC interpretation string at **`financial_analyzer.py:4369`** (inside `working_capital_analysis`, def at `:4302`). It currently reads, verbatim:
  - **BEFORE (`:4369`):** `f"Negative CCC of {ccc:.0f} days: company generates cash before paying suppliers. Excellent."`
  - The wording is mislabeled/ambiguous on sign: a *negative* CCC printed via `{ccc:.0f}` already carries the minus sign, so the literal word "Negative" plus a signed number double-states the sign and reads as a contradiction (e.g. "Negative CCC of -12 days"). Correct the **label/sign wording only** to state the magnitude and meaning unambiguously:
  - **AFTER (`:4369`):** `f"CCC of {ccc:.0f} days (negative): company collects cash before paying suppliers — favorable."`
  - This is wording-only: the stored `ccc` value (`WorkingCapitalResult.ccc`, set at `:4391`) and the `< 0` branch threshold are unchanged.
  - **Test (concrete, provable — replaces "label test"):** build a fixed `FinancialData` fixture that yields a negative CCC (e.g. high `accounts_payable`/`cogs` so DPO > DSO + DIO — e.g. `revenue=1000, accounts_receivable=50, inventory=20, cogs=1000, accounts_payable=400` → DSO≈18, DIO≈7, DPO≈146, CCC≈-121). Then: (1) **positive label assertion** — the returned `insights` list contains the exact AFTER string for that fixture (and does NOT contain the BEFORE string); (2) **numeric snapshot assertion** — `result.ccc == round(dso + dio - dpo, 1)` is unchanged vs. a pre-edit snapshot of the same fixture (label/sign-wording-only, no formula change).
- **(d) DCF/DSCR input validation — RESOLVED & BOUNDED (OBJ-9 / C3):** both functions live in `financial_analyzer.py` (now declared in WP-7's file-ownership list): DCF terminal-value / discount logic at **`financial_analyzer.py:3919-3995`** (`forecast_cashflow`, def at `:3919`, GGM block at `:3985-3994`) and DSCR at **`financial_analyzer.py:12881`** (`debt_service_coverage_analysis`). Scope is exactly: reject/None-return on non-positive or None denominators that would otherwise produce inf/garbage (DCF: discount rate ≤ -100% or terminal-growth ≥ discount-rate — note `:3989` already guards `discount_rate > terminal_growth` for the terminal value; confirm the guard returns `None` rather than `0.0`/inf and extend to discount-rate ≤ -100% if unguarded; DSCR: debt service ≤ 0 → return None, consistent with the P0-6/P0-7 DSCR work already committed). NO new validation beyond these named denominator/edge guards. Acceptance: DCF with growth ≥ discount-rate → terminal value None (not inf); DSCR with zero/negative debt service → None.
- **(e) P2 NRR cleanup follow-on — RESOLVED (follow-on to P0-9, committed 4cf5f2e):** P0-9 already split `gross_revenue_retention` (= 1 − gross_churn) from `net_revenue_retention` and leaves NRR `None` until expansion data exists (`startup_model.py:125-131,154-155`). The P2 *cleanup follow-on* is the surfacing gap: the SaaS interpretation string (`startup_model.py:136-148`) mentions gross churn but **never states that NRR is unavailable**, so a reader cannot tell NRR was intentionally withheld vs. silently dropped. Change: append a "NRR unavailable (requires multi-period expansion data)" clause to the interpretation when `net_revenue_retention is None` (mirroring the existing `mrr_growth_rate` unavailable clause at `:146-147`). NO field/formula change — P0-9's data model stays as-is. Acceptance: when `net_revenue_retention is None`, the interpretation string contains the "NRR unavailable" clause; when a future caller supplies NRR, the clause is absent.
- **(f) P2 ratio threshold docs — RESOLVED:** `RatioDefinition` (`ratio_framework.py:35-50`) carries `scoring_thresholds` with **no provenance field** — the threshold cut-offs are undocumented magic numbers. Change (docs-only, no scoring change): add an optional `threshold_source: str = ""` field to `RatioDefinition` and populate it with a one-line citation/rationale for the catalog entries whose thresholds are currently bare (`RATIO_CATALOG`, `:272+`), and/or document the threshold derivation in the class/catalog docstring. NO threshold *values* change — scores are identical before/after. Acceptance: `threshold_source` (or equivalent docstring) present and non-empty for the documented catalog entries; a test asserts no `scoring_thresholds` tuple value changed (snapshot of computed scores for a fixed `FinancialData` is unchanged).
- **Tests:** per sub-item (six atomic test sets matching a–f above).
- **Risk:** scope creep — bounded to exactly the six resolved items above (the five PRD P2 financial-edge sub-items b–f + P1-E8 relabel a); (d) is explicitly limited to the named denominator guards; (e) and (f) are documentation/surfacing-only with no formula, field-value, or threshold-value changes (snapshot tests lock behavior invariance).

### WP-8 — P2 export hardening  *(XS, ~0.5h — most sub-items already exist)*
Several originally-listed sub-items were found to ALREADY EXIST in a verification re-read (the same stale-baseline error this plan claims to avoid). Revised, de-duplicated scope:
- **`score_to_grade` None guard — DESCOPED / REJECTED (UA-2):** `export_utils.py:24-32` ALREADY raises `TypeError` on non-int/float (incl. `None`) AND already clamps out-of-range via `max(0, min(100, int(score)))` (`:28`). The proposed "None → safe grade" would convert a deliberate fail-fast contract into silently masking a programming error. **Keep the existing raise-on-None contract; no change.** (If any caller passes `None`, fix the caller, not the canonical helper.)
- **`company_name` length constraint — DESCOPED (UA-1):** `api.py:571` ALREADY declares `company_name: str = Field(default="", max_length=200)`, which **rejects** (422) over-long names. "rejected" and "truncated" are contradictory; the reject behavior already exists. **No change.** A regression test asserting a 201-char name → 422 MAY be added to lock the existing behavior, but no code change.
- **PDF cover/value truncation — DESCOPED (OBJ-1 / D13):** `export_pdf.py` ALREADY truncates cell values to 50 chars (`:257`), cover/period text to 100 (`:327,:333`), and ratio entries to 500 (`:113-115`, `_MAX_RATIO_ENTRIES`). **No new PDF cap** — re-implementing would introduce a conflicting second cap. These are pre-existing *silent* truncations (no log, no PDF note). They are knowingly accepted out-of-scope (C8) and are **explicitly excluded** from the §6 item-4 "no NEW silent caps" gate per D13; this plan does not introduce or modify them.
- **Generic 500-row cap on XLSX tabular exports — KEPT (the one real item) — re-verified (C8):** a direct re-read shows the Ratios sheet does NOT lack a cap: `export_xlsx.py:380-382` ALREADY applies a SILENT `_MAX_RATIO_ENTRIES = 500` dict slice (`numeric = dict(list(numeric.items())[:_MAX_RATIO_ENTRIES])`) with NO log and NO truncation note (its only sheet output otherwise is "No ratio data available." when empty — never a truncation note). The earlier "currently lack one" wording was the same stale-baseline error D13/OBJ-1 claim to have eliminated; corrected here.
  - **Disposition (C8):** SURFACE the existing `:380-382` cap rather than adding a second/duplicate cap. WP-8's "generic 500-row cap" work = add a truncation-note row + log the drop on the Ratios sheet's existing `:380-382` slice (do NOT introduce a new cap on that path). Then audit the remaining non-scenario tabular sheets and, for any that genuinely lack ANY cap, add a `_MAX_EXPORT_ROWS`-style 500-row cap WITH log + truncation note. Do not double-cap the Ratios sheet.
  - **Precedence (UA-5 / D11):** the scenario path (WP-2) is EXEMPT and keeps its 10k cap; this 500-row work applies only to the non-scenario tabular sheets.
- **Tests:** (real) >500-entry Ratios sheet → existing `:380-382` cap retained AND now emits one truncation note + logs (no longer silent), reopen OK; (real, if any other uncapped non-scenario sheet exists) oversized table → capped at 500 + truncation note + reopen OK; (lock-only, no code change) 201-char company_name → 422; (lock-only) `score_to_grade(None)` → raises `TypeError`.
- **Risk:** the 500-row cap must not touch the scenario path — see D11 precedence (scenario exempt).

---

## 4. Descope / explicit non-goals
- **P1-E2** (XLSX conditional-format threshold): the feature does not exist; **descoped** unless review decides score cells should be color-coded. If added, threshold must equal the `score_to_grade` D-boundary (65), not 50.
- P0 items (P0-6…P0-10): already committed; out of scope here.
- Non-WS-3 P2/P3 backlog, README/MEMORY (WS-6), other workstreams.

---

## 5. Sequencing & parallelism
File ownership (each WP appears exactly once — UA-4):
- WP-1 → `startup_model.py`
- WP-2 → `export_xlsx.py`  (also WP-8's generic XLSX cap; serialize WP-2 → WP-8 on this file)
- WP-3, WP-4 → `export_pdf.py`  (serialize WP-3 → WP-4)
- WP-5 → `compliance_scorer.py`  (NOT export_xlsx — the old "WP-5(xlsx)" entry was a typo)
- WP-6 → `tests/test_portfolio_analyzer.py`  (test-only)
- WP-7 → `compliance_scorer.py`, `underwriting.py`, `ratio_framework.py`, `startup_model.py`, `financial_analyzer.py` (sub-items (c) CCC label `:4369` (`working_capital_analysis`, def at `:4302`) and (d) DCF `:3919-3995` (`forecast_cashflow`, def at `:3919`) / DSCR `:12881` validation reside in the core module). **No other WP touches `financial_analyzer.py`**, so no new serialization edge is required; but because it is the 15K-LOC-guarded core module, WP-7 exercises the §6 item-5 pre-commit LOC guard (see wave note below).
- WP-8 → `export_xlsx.py` (after WP-2), plus lock-only tests touching `api.py`/`export_utils.py`

```
Wave 1 (parallel, fully disjoint files): WP-1 (startup_model), WP-3 (export_pdf), WP-5 (compliance_scorer), WP-6 (tests)
Wave 2 (serialize on shared files): WP-2 (export_xlsx); WP-4 (export_pdf, after WP-3)
Wave 3 (serialize where files overlap Wave-1/2): WP-7 (touches startup_model after WP-1, compliance_scorer after WP-5); WP-8 (export_xlsx after WP-2)
Gate:  full suite green after each wave; ruff clean; commit per WP
```
Note: WP-7 shares `startup_model.py` with WP-1 and `compliance_scorer.py` with WP-5, so it runs in Wave 3 after both. WP-8's real change shares `export_xlsx.py` with WP-2, so it runs after WP-2. WP-7 also edits `financial_analyzer.py` (the 15K-LOC-guarded core module) for sub-items (c)/(d); **no other WP touches that file**, so it adds no serialization edge, but the §6 item-5 pre-commit LOC guard is exercised on the WP-7 commit(s) — the (c)/(d) deltas are label/sign-wording + small denominator guards (negligible LOC), expected to stay under 15K.

---

## 6. Quality gates (per work package)
1. New regression test(s) authored **first** and failing before the fix.
2. `python3.13 -m pytest tests/<touched>` green, then full suite green.
3. `ruff check <touched files>` clean.
4. No **NEW** silent truncation/caps — every cap introduced or modified in this plan MUST log + surface (truncation note row/line). Pre-existing silent caps are inventoried and dispositioned exhaustively as follows:
   - **Accepted out-of-scope, left silent (D13):** the silent PDF truncations in `export_pdf.py` (`_MAX_RATIO_ENTRIES=500` at `:113-115`; 50-char cell at `:257`; 100-char cover/period at `:327,:333`) are knowingly retained out of scope per WP-8 OBJ-1 / D13 — re-touching them risks a conflicting second cap. They are NOT covered by this gate.
   - **Surfaced by WP-8 (D14):** the silent XLSX Ratios-sheet cap at `export_xlsx.py:380-382` (`_MAX_RATIO_ENTRIES=500`, dict slice, no log/note) IS in scope — WP-8 surfaces it (adds a truncation note row + log) rather than adding a duplicate cap. Once surfaced it satisfies this gate; it must not remain silent.
5. Commit message references the P-ID; pre-commit hook (LOC guard + suite) passes.

---

## 7. Decision Log
| # | Decision | Alternatives | Objections | Resolution |
|---|----------|--------------|------------|------------|
| D1 | Treat P1-E1 as test-only | re-"fix" the formula | — | Code verified already correct; re-fixing risks regression |
| D2 | Descope P1-E2 | implement color-coding now | — | Feature absent; not in acceptance criteria; add only if review insists |
| D3 | P1-E3 via sanitize-map + latin-1 backstop; TTF embed = stretch | TTF-only | TTF adds binary asset + size | Sanitize is zero-dependency and satisfies acceptance |
| OBJ-1 | REVISE (accept) | re-implement PDF cap | major: WP-8 missed existing PDF caps | Stale baseline confirmed: `_MAX_RATIO_ENTRIES=500` (:113-115), 50-char cell (:257), 100-char cover/period (:327,:333) already exist. WP-8 reworded to DESCOPE PDF caps; only a new *non-scenario XLSX* 500-row cap is kept. P1-E3 baseline row annotated that truncation ≠ sanitization. |
| OBJ-2 | REVISE (accept) | reverse-engineer corr_pts | major: corr_pts unverifiable | `corr_pts` not exposed on `DiversificationScore` (only `correlation_penalty`). WP-6 now isolates corr via maximal-HHI (single-holding) fixture so `rev_pts=ast_pts=0` and `overall_score==corr_pts`; reads component directly. |
| OBJ-3 | REVISE (accept) | keep pre_money clamp | major: untested side effect | Dropped the `pre_money_valuation` clamp from WP-1. It silently zeros `implied_arr_mult` (:349) and alters dilution — out-of-scope for P1-E6 (names only `raise_amount`). WP-1 clamps `raise_amount` only. |
| OBJ-4 | REVISE (accept) | undefined truncation boundary | major: mid-scenario corruption | WP-2 now caps at whole-scenario boundary (no mid-scenario truncation, D6); `freeze_panes` kept with reopen assertion; `set_column` already valid post-write; multi-scenario truncation-boundary test added. |
| OBJ-5 | REVISE (accept) | append to SOX material_weakness | major: double/triple counting | Concrete anti-double-count: BS imbalance already flows to `overall_audit_risk` via `sec.red_flags` (:642). SOX adjusts only its OWN `risk_score` (local penalty, NOT in any material_weakness/red_flag list). Test asserts `overall_audit_risk` indicator count unchanged. |
| OBJ-6 | REVISE (accept) | "no raise" test only | minor: partial coverage undetected | WP-3 test strengthened: unit-test `_sanitize_text` substitution table exactly; extract rendered PDF text and assert ASCII substitution present + no non-latin-1 byte survives. All sinks (:89,:191,:257,:298-336,:399,:457) enumerated + funneled through choke points. |
| OBJ-7 | REVISE (accept) | leave auto_page_break global | minor: blank/double pages | WP-4 save/disable/restore `auto_page_break` (try/finally) around `_add_table`; caller's `:131-133` guard documented as pre-table only. Test asserts no blank page (every page has text) + auto_page_break restored. |
| OBJ-8 | REVISE (accept) | fixed ≥5003 gate | minor: unverified gate | DoD no longer hard-codes ≥5003. Capture branch baseline at WP start; require "no regressions vs baseline". MEMORY.md 4966 noted as stale; branch baseline unverified (D12). |
| OBJ-9 | REVISE (accept) | one bundled package w/ open decisions | minor: not independently shippable | WP-7's four items relabeled RESOLVED, each split into an atomic commit w/ own acceptance; (d) DCF/DSCR bounded to named denominator/edge guards — no open-ended validation. |
| CG-1 | REVISE (accept) | gate Basel III by industry | minor: false premise (no field) | WP-7(a) resolved to relabel-only: `FinancialData` (:2150) has no `industry`/`sector` field; `industry` is only an `industry_benchmark` param (:4670, default "general"). Gating needs a new field (out of WS-3 scope). Constraint documented. |
| CG-2 | REVISE (accept) | leave two PRD P2 sub-items unaddressed | major: silent scope drop | Two PRD WS-3 P2 "Financial edge cases" sub-items (`LOKI-PRD.md:155`) — "NRR cleanup follow-on" and "ratio threshold docs" — were absent from every WP and the §4 descope list. Added as in-scope WP-7(e) and WP-7(f): (e) surface NRR-unavailable in the SaaS interpretation string (follow-on to P0-9 4cf5f2e at `startup_model.py:125-148`; docs/surfacing only, no field/formula change); (f) add `threshold_source` provenance to `RatioDefinition` (`ratio_framework.py:35-50`; docs-only, threshold values unchanged, snapshot test locks score invariance). WP-7 now six atomic commits; §1 DoD and WP-7 risk line updated so the count of P2 financial-edge items addressed matches the PRD's five named items, leaving none without an in/out-of-scope verdict. |
| UA-1 | REVISE / DESCOPE (accept) | add new length constraint | major: moot + contradictory | `api.py:571` already has `company_name: Field(max_length=200)` (rejects/422). "rejected vs truncated" resolved → keep reject behavior; descope code change; optional lock-only test. |
| UA-2 | REJECT proposed change (accept objection) | silently grade None | major: contradicts existing contract | `export_utils.py:24-32` already raises `TypeError` on `None` and clamps out-of-range (:28). Silent-grade would mask a programming error. Keep existing raise-on-None contract; lock-only test that `None` raises. |
| UA-3 | REVISE (accept) | new tolerance constant | major: divergent tolerances | WP-5 pins tolerance to the existing 1% (`:427`), extracted to `_BS_IMBALANCE_TOLERANCE=0.01` shared by both checks; concrete balanced/imbalanced fixtures added. |
| UA-4 | REVISE (accept) | leave duplicate WP-5 entry | minor: inconsistent diagram | §5 rewritten: each WP appears once with correct file ownership; erroneous "WP-5(xlsx)" Wave-2 entry removed (WP-5 is compliance_scorer.py); WP-7/WP-8 file overlaps explicitly serialized. |
| UA-5 | REVISE (accept) | leave precedence unstated | minor: silent cap conflict | Precedence stated (D11): scenario export (WP-2) EXEMPT from WP-8's 500-row cap and keeps its 10k cap; 500-row cap applies only to non-scenario tabular sheets. |
| UA-6 | REVISE (accept) | "no raise" test only | minor: no-op passes | Folded into WP-3 strengthened test (see OBJ-6): assert expected ASCII substitution appears in extracted text, not merely "no exception". Sanitization placed before the 50-char truncation. |
| D6 | XLSX cap truncates at whole-scenario boundary | mid-scenario truncation | corrupt partial scenario block | Stop before a scenario that would exceed 10k; emit one note |
| D11 | Scenario path exempt from WP-8 500-row cap | apply 500 globally | would override PRD 10k acceptance | Scenario cap (10k) authoritative; 500 cap = non-scenario only |
| D12 | No hard-coded test-count gate | fixed ≥5003 | unverified target trivially passes or blocks | "No regressions vs. measured branch baseline" |
| D5 | Clamp `raise_amount` only; do NOT clamp `pre_money_valuation` | clamp both | side-effects on `implied_arr_mult`/dilution | Out of P1-E6 scope (PRD names only `raise_amount`); a `pre_money` clamp silently zeros `implied_arr_mult` (`startup_model.py:349`) and alters `post_money`/dilution semantics — an untested behavior change. Same rationale captured by OBJ-3; D5 is the named decision tag referenced on WP-1 line 53. |
| D13 | Narrow §6 item-4 gate to "no NEW silent caps"; accept pre-existing silent PDF truncations as out-of-scope (C8) | (A) surface the existing PDF caps (log + truncation note) within WP-8 | re-touching `export_pdf.py` caps risks a conflicting second cap and is beyond the P1-E/P2 acceptance lines | Pre-existing silent PDF losses (`_MAX_RATIO_ENTRIES=500` at `:113-115`; 50-char cell at `:257`; 100-char cover/period at `:327,:333`) are explicitly accepted out-of-scope per WP-8 OBJ-1. §6 gate reworded to bind only caps introduced/modified in THIS plan (WP-2 10k note, WP-8 500-row note, plus the now-surfaced XLSX `:380-382` cap per D14). Resolves the contradiction between the former absolute gate and the knowingly-retained PDF data loss. |
| C3 | Add `financial_analyzer.py` to WP-7's file-ownership (§3 + §5) | (alt) wrap (d) DCF/DSCR in `underwriting.py` instead | major: hidden file dependency — WP-7(d) edits DCF (`forecast_cashflow`, def at `:3919`, span `:3919-3995`) + DSCR (`:12881`) which live in `financial_analyzer.py`, undeclared in WP-7's ownership list (violates C3 explicit ownership; also a hidden touch of the 15K-LOC-guarded core) | Declared `financial_analyzer.py` in WP-7 ownership (§3 line, §5 map) citing DCF `forecast_cashflow` `:3919-3995` and DSCR `:12881`. Rejected the underwriting-wrapper alternative because the validation belongs at the source functions (DSCR P0-6/P0-7 already committed in-place there; a wrapper would duplicate/diverge). Confirmed no other WP touches `financial_analyzer.py` → no new serialization edge; added §5/wave note that WP-7 exercises the §6 item-5 pre-commit LOC guard. |
| C8 | Accept the three pre-existing SILENT PDF truncations as knowingly out-of-scope for WS-3 | (A) surface them within WP-8 (log + truncation note); (B) remove the caps entirely | major: re-touching `export_pdf.py` caps risks a conflicting second cap and is beyond the P1-E/P2 acceptance lines; removing them risks unbounded PDF size | The three silent PDF losses — `_MAX_RATIO_ENTRIES=500` (`export_pdf.py:113-115`), 50-char cell (`:257`), 100-char cover/period (`:327,:333`) — are inventoried, flagged as silent, and consciously dispositioned OUT-OF-SCOPE for WS-3. They are excluded from the §6 item-4 "no NEW silent caps" gate (the gate binds only caps introduced/modified by this plan). This is the decision the body's "(C8)" references (WP-8 line, D13 row) point to. Tightly coupled with D13, which narrows the gate wording; D13 is the gate-rewording mechanism, C8 is the explicit acceptance of the PDF caps it exempts. |
| D14 | Surface (not duplicate) the pre-existing silent XLSX Ratios-sheet cap at `export_xlsx.py:380-382` as part of WP-8's generic 500-row work | (A) add a second/new 500-row cap to the Ratios sheet (double-cap); (B) accept it silent out-of-scope like the PDF caps (C8) | major: a second cap would double-truncate/conflict; leaving it silent is a C8-style silent data-loss path that this plan's own scope (a non-scenario XLSX tabular cap) directly covers, so out-of-scope is unjustified | Re-read confirmed the Ratios sheet ALREADY has a silent `_MAX_RATIO_ENTRIES=500` dict slice (`:380-382`, no log/note) — the WP-8 "currently lack one" wording was a stale-baseline error and is corrected. WP-8 now SURFACES this existing cap (truncation note row + log) instead of adding a new one; remaining non-scenario sheets are audited and only genuinely uncapped ones get a new logged+noted 500 cap. §6 item-4 inventory updated to list this cap as "surfaced by WP-8" (vs. the PDF caps left silent per C8/D13). Distinct from C8 (PDF, left silent) because this XLSX path is in WS-3 scope and is fixed, not exempted. |
| C5 | Make WP-7(c) CCC test concrete and provable | (alt) leave "label test" placeholder | major: circular placeholder — no file:line, no before/after string, asserts nothing | Pinned the defect to the negative-CCC interpretation string at `financial_analyzer.py:4369`, inside `working_capital_analysis` (def at `:4302`; the non-existent `analyze_working_capital_efficiency` name is NOT used) (formula DSO+DIO−DPO is correct everywhere — confirmed `:2692-2705`, `:4364-4367`, `:8299-8305`, `:10004` — so no formula change). Specified verbatim BEFORE/AFTER label (wording/sign-clarity only, stored `ccc` unchanged) and replaced "label test" with (1) a positive `insights`-contains-AFTER assertion on a fixed negative-CCC `FinancialData` fixture and (2) a numeric snapshot assertion that `result.ccc` is unchanged. Reconciled WP-7 ownership to include `financial_analyzer.py` (same as C3). |

---

## 8. Exit criteria
- [ ] All WP acceptance lines satisfied
- [ ] PRD WS-3 acceptance: PDF unicode ✔, XLSX 10k cap ✔
- [ ] Full suite green, no regressions
- [ ] Decision Log complete; all review objections resolved or explicitly rejected
- [ ] Arbiter disposition = APPROVED
- [ ] clarity-gate = PASS
