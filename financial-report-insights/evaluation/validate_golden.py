"""Validate the golden Q&A set against the live corpus, and the corpus itself.

Why this exists
---------------
The golden set (``evaluation/golden_financial_qa.py``) holds real figures copied
out of private workbooks, so it is git-ignored and cannot be reviewed in a PR.
That makes it silently driftable: on 2026-09-02 one expected value was still the
figure from a workbook that had since been superseded, so the retrieval gate
scored a MISS while retrieval was in fact returning the *newer, correct* row.
The gate blamed the retriever for the golden set being stale.

Two failure classes, both invisible until something checks them:

1. **Stale expectation** -- an ``expected_value`` that no longer appears anywhere
   in the active corpus. The gate reports this as a retrieval miss, which sends
   you debugging the wrong layer entirely.
2. **Unreadable corpus file** -- a document sitting in ``documents/`` that the
   ingestion pipeline cannot parse (a missing optional engine such as ``xlrd``
   for legacy ``.xls``). Ingestion logs a warning and moves on, the index is
   built without it, and every downstream number is then computed over a
   partial corpus.

   This one hides especially well, because the embedding cache masks it. On
   2026-09-02 ``xlrd`` was absent from the active interpreter, so ``pd.ExcelFile``
   could not open either ``.xls`` in this corpus -- yet both were still in the
   index (44 chunks), served from content-hash cache entries written weeks
   earlier by an interpreter that HAD the engine. Nothing was wrong with the
   current index; a fresh ingest on a clean machine would silently have dropped
   both files. Checking readability directly, rather than inferring it from
   chunk counts, is the only way to see that gap before it bites.

Run it standalone, or let ``run_golden_qa.py`` call it as a preflight:

    python3.13 -m evaluation.validate_golden      # exit 0 = consistent
"""

import glob
import sys
from pathlib import Path

import pandas as pd

DOCS = Path("documents")


def _render_sheets(path: str) -> dict:
    """Render every sheet of a workbook the way ingestion does.

    Mirrors ingest_excel: read headerless, detect the real header row beneath any
    banner, re-key, then dense-render. Matching that exactly is the point -- a
    value must be findable in the text the pipeline actually embeds, not in the
    raw cell grid.
    """
    from ingestion_pipeline import _detect_excel_header_row, _df_to_markdown

    out = {}
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=5000)
            if raw.empty:
                continue
            hdr = _detect_excel_header_row(raw)
            cols = raw.iloc[hdr].tolist()
            df = raw.iloc[hdr + 1 :].reset_index(drop=True)
            df.columns = [str(c).strip() if pd.notna(c) and str(c).strip() else f"Col_{i}" for i, c in enumerate(cols)]
            out[sheet] = _df_to_markdown(df, max_rows=5000)
        except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
            out[sheet] = ""
            print(f"    ! sheet {sheet!r} unreadable: {exc}")
    return out


def check_corpus_readable() -> list:
    """Every workbook in documents/ must actually parse. Returns failures."""
    failures = []
    files = sorted(glob.glob(str(DOCS / "*.xls*")))
    print(f"corpus: {len(files)} workbook(s) in {DOCS}/")
    for f in files:
        try:
            pd.ExcelFile(f)
        except Exception as exc:  # noqa: BLE001
            failures.append((f, str(exc)))
            print(f"  UNREADABLE  {Path(f).name}: {exc}")
    if not failures:
        print("  all workbooks parse")
    return failures


def check_golden_values() -> list:
    """Every expected_value must appear in the rendered active corpus."""
    try:
        from evaluation.golden_financial_qa import GOLDEN
    except ImportError:
        print("no golden set present -- nothing to validate")
        return []

    corpus = {}
    for f in sorted(glob.glob(str(DOCS / "*.xls*"))):
        try:
            for sheet, text in _render_sheets(f).items():
                corpus[(Path(f).name, sheet)] = text
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {Path(f).name}: {exc}")

    stale = []
    print(f"\ngolden set: {len(GOLDEN)} entries vs {len(corpus)} rendered sheets")
    for g in GOLDEN:
        value = str(g["expected_value"])
        hits = [k for k, text in corpus.items() if value in text]
        on_named_sheet = [k for k in hits if k[1] == g["sheet"]]

        # The check is SHEET-SCOPED on purpose. An earlier version only failed
        # when a value was absent from the whole corpus, and passed when it
        # turned up on some other sheet -- which is precisely what the real
        # drift looked like: the superseded LARA figure still existed in an
        # unrelated reference workbook, so a corpus-wide search found it and
        # reported OK. A value that is not on the sheet the entry names is
        # stale for this gate's purposes, wherever else it may survive.
        if on_named_sheet:
            continue
        stale.append(g)
        if hits:
            found = ", ".join(f"{wb}/{sh}" for wb, sh in hits[:2])
            print(f"  STALE       [{g['sheet']}] {g['label']} = {value}")
            print(f"              absent from '{g['sheet']}'; only found in {found}")
        else:
            print(f"  STALE       [{g['sheet']}] {g['label']} = {value}")
            print("              not found anywhere in the active corpus")
    if not stale:
        print("  every expected_value is present on the sheet its entry names")
    return stale


def main() -> int:
    unreadable = check_corpus_readable()
    stale = check_golden_values()

    print()
    if unreadable:
        print(f"FAIL: {len(unreadable)} workbook(s) cannot be parsed -- the index is built WITHOUT them.")
        print("      Install the missing engine (legacy .xls needs `xlrd`) and re-ingest.")
    if stale:
        print(f"FAIL: {len(stale)} golden value(s) absent from the corpus -- the golden set is STALE.")
        print("      Regenerate them from the current workbooks; do NOT debug retrieval for these.")
    if not unreadable and not stale:
        print("OK: corpus fully readable and golden set consistent with it.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
