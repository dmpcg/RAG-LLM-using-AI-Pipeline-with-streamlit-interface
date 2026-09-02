"""Sweep RRF fusion parameters against the golden set.

Run from financial-report-insights/ with Ollama up:
    export OLLAMA_HOST=http://localhost:11434
    python3.13 -m evaluation.sweep_fusion

Use this before changing bm25_weight/semantic_weight/rrf_k in config.py. The
current 0.65/0.35 split was chosen from its output on 2026-09-02; that tuning
reflects a golden set of 12 exact-figure lookups, so re-run it if the workload
shifts toward conceptual queries, or after any corpus change large enough to
move rankings.

Builds the index ONCE, then re-runs all 12 questions under each (semantic_weight,
bm25_weight, rrf_k) combination by mutating settings in-process. Reports
TOP1/TOP3/DEEP for each so a weighting is chosen on evidence, not intuition.
"""

import sys

sys.path.insert(0, ".")

from app_local import SimpleRAG  # noqa: E402
from config import settings  # noqa: E402
from evaluation.golden_financial_qa import GOLDEN  # noqa: E402


def haystack(doc: dict) -> str:
    parts = [str(doc.get("content", ""))]
    if doc.get("_child_content"):
        parts.append(str(doc["_child_content"]))
    return "\n".join(parts)


def score(rag) -> tuple:
    top1 = top3 = deep_top1 = deep_total = 0
    misses = []
    for g in GOLDEN:
        results = rag.retrieve(g["question"], top_k=3) or []
        val = str(g["expected_value"])
        in1 = bool(results) and val in haystack(results[0])
        in3 = any(val in haystack(r) for r in results)
        top1 += in1
        top3 += in3
        if g.get("deep"):
            deep_total += 1
            deep_top1 += in1
        if not in1:
            misses.append(f"{g['sheet']}/{g['label'][:26]}" + ("" if in3 else " (NOT-top3)"))
    return top1, top3, deep_top1, deep_total, misses


def main() -> int:
    rag = SimpleRAG(docs_folder="./documents", embedding_model=settings.embedding_model)
    print(f"index: {len(rag.documents)} chunks\n")

    base_sem = settings.semantic_weight
    base_bm = settings.bm25_weight
    base_k = getattr(settings, "rrf_k", 60)
    print(f"BASELINE config: semantic={base_sem} bm25={base_bm} rrf_k={base_k}\n")

    combos = []
    for sem, bm in [(0.6, 0.4), (0.5, 0.5), (0.45, 0.55), (0.4, 0.6), (0.35, 0.65), (0.3, 0.7)]:
        for k in (60, 30, 20, 10):
            combos.append((sem, bm, k))

    rows = []
    for sem, bm, k in combos:
        settings.semantic_weight = sem
        settings.bm25_weight = bm
        if hasattr(settings, "rrf_k"):
            settings.rrf_k = k
        t1, t3, d1, dt, misses = score(rag)
        rows.append((t1, t3, d1, sem, bm, k, misses))
        flag = "  <-- BASELINE" if (sem, bm, k) == (base_sem, base_bm, base_k) else ""
        print(f"sem={sem:<5} bm={bm:<5} k={k:<3} | TOP1 {t1:2}/12  TOP3 {t3:2}/12  DEEP {d1}/{dt}{flag}")
        if misses:
            print(f"      misses: {'; '.join(misses)}")

    settings.semantic_weight, settings.bm25_weight = base_sem, base_bm
    if hasattr(settings, "rrf_k"):
        settings.rrf_k = base_k

    print("\n=== ranked by (TOP1, TOP3, DEEP) ===")
    for t1, t3, d1, sem, bm, k, _ in sorted(rows, key=lambda r: (-r[0], -r[1], -r[2]))[:8]:
        print(f"  TOP1 {t1:2}  TOP3 {t3:2}  DEEP {d1}   sem={sem} bm={bm} k={k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
