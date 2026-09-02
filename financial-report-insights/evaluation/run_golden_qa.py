"""Run the real-content golden Q&A retrieval eval against the live RAG.

Unlike the synthetic eval harness (which scores filename matches against files
that don't exist in the repo), this gate is grounded in the ACTUAL workbook:
each question's distinctive figure must appear in the chunk that retrieval
returns. Several questions deliberately target rows far into their sheet
(char_offset > 2500), which the old truncate-to-~18% ingestion could not have
surfaced -- so a pass here proves the dense-chunking fix actually improved
coverage, not just that numbers appear somewhere.

Run from the financial-report-insights dir:
    ./.venv/Scripts/python.exe -m evaluation.run_golden_qa
Exit code 0 = gate passed, 1 = failed.
"""

import os
import sys

from app_local import SimpleRAG
from config import settings

try:
    # golden_financial_qa.py holds figures extracted from a private workbook and
    # is intentionally git-ignored; generate it per deployment from real docs.
    from evaluation.golden_financial_qa import GOLDEN
except ImportError:
    # A missing golden set used to exit 0 -- so an environment without one
    # (CI, a fresh clone) reported a PASS it never actually ran, which is the
    # precise shape of failure this gate exists to prevent. Absence is now a
    # hard failure unless a caller opts out EXPLICITLY.
    if os.environ.get("GOLDEN_QA_OPTIONAL") == "1":
        print(
            "SKIP: no evaluation/golden_financial_qa.py, and GOLDEN_QA_OPTIONAL=1. "
            "Retrieval quality is NOT verified by this run."
        )
        sys.exit(0)
    print(
        "GATE FAIL: no evaluation/golden_financial_qa.py found. This gate needs a "
        "golden set built from your own documents. Generate one, or set "
        "GOLDEN_QA_OPTIONAL=1 to skip deliberately -- but a skip verifies nothing.",
        file=sys.stderr,
    )
    sys.exit(1)

# Gate thresholds.
MIN_TOP1 = 8  # >= this many of 12 with the figure in the TOP chunk
MIN_DEEP_FRACTION = 0.5  # >= half the deep questions must hit top-1


def _haystack(doc: dict) -> str:
    """Text a retrieved doc exposes: expanded content + original child text."""
    parts = [str(doc.get("content", ""))]
    if doc.get("_child_content"):
        parts.append(str(doc["_child_content"]))
    return "\n".join(parts)


def main() -> int:
    # Preflight: confirm the golden set still matches the corpus BEFORE scoring
    # retrieval against it. A drifted expectation shows up here as a MISS, which
    # sends you debugging the retriever for a bug in the fixtures -- it did
    # exactly that on 2026-09-02, where a superseded figure made a correct
    # top-1 answer look like a failure. Set GOLDEN_QA_SKIP_VALIDATE=1 to bypass
    # (it re-renders every sheet, so it is not free).
    if os.environ.get("GOLDEN_QA_SKIP_VALIDATE") != "1":
        from evaluation.validate_golden import check_corpus_readable, check_golden_values

        print("preflight: validating golden set against the corpus...")
        unreadable = check_corpus_readable()
        stale = check_golden_values()
        if unreadable or stale:
            print(
                "\nGATE FAIL (preflight): the fixtures disagree with the corpus, so any "
                "retrieval score below would be measuring the wrong thing. Fix the golden "
                "set / corpus first -- see the lines above.",
                file=sys.stderr,
            )
            return 1
        print()

    rag = SimpleRAG(
        docs_folder="./documents",
        embedding_model=settings.embedding_model,
    )
    n_chunks = len(rag.documents)
    print(f"index: {n_chunks} chunks indexed")
    if n_chunks == 0:
        print("GATE FAIL (no chunks indexed)")
        return 1

    top1 = top3 = 0
    deep_top1 = deep_total = 0
    rows = []
    for g in GOLDEN:
        results = rag.retrieve(g["question"], top_k=3) or []
        val = str(g["expected_value"])
        in_top1 = bool(results) and val in _haystack(results[0])
        in_top3 = any(val in _haystack(r) for r in results)
        top1 += in_top1
        top3 += in_top3
        if g.get("deep"):
            deep_total += 1
            deep_top1 += in_top1
        rows.append((g.get("deep", False), in_top1, in_top3, g["sheet"], g["label"], val))

    n = len(GOLDEN)
    for deep, t1, t3, sheet, label, val in rows:
        tag = "DEEP" if deep else "    "
        print(f"  {tag} top1={'Y' if t1 else 'N'} top3={'Y' if t3 else 'N'}  [{sheet}] {label} = {val}")
    print(f"\nTOP1 {top1}/{n}   TOP3 {top3}/{n}   DEEP-TOP1 {deep_top1}/{deep_total}")

    passed = top1 >= MIN_TOP1 and (deep_total == 0 or deep_top1 >= deep_total * MIN_DEEP_FRACTION)
    print(
        "GATE",
        "PASS" if passed else "FAIL",
        f"(need TOP1>={MIN_TOP1} and DEEP-TOP1>={int(deep_total * MIN_DEEP_FRACTION)})",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
