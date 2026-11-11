"""Part B (supervisor analysis, 11 Nov): what happens if overlap is removed from the TEST side (training never reduced)?

Rule: a clinical test row is removed if it shares its question, or any passage, with the training data:
    clinical train BEFORE any training-side removal (RAGBench PubMedQA train, RAGBench CovidQA train,
    pqa_artificial's 90/10 train portion) + data/unified_v2/train.jsonl.
Keys come from Prajakta's overlap_keys(row, min_words) in scripts/build_clinical.py (pinned to BUILD_CLINICAL_SHA).
Variants: min_words=0 (every non-empty passage) and min_words=8 (the build default).
RAGTruth Summary rows are left out of QUESTION comparisons (their query is the constant "Summarize the following news").

Reported:
  B1 per clinical test source: rows / positives before and after, prevalence, % removed, question-only / passage-only / both
  B2 which training part each removal touches
  B3 the same rule on the D4 and D5 cases
  B4 train x val overlap (D9 and the rest), reported only, never removed
  B5 hand-check feasibility (RAGBench positives and negatives left vs 25 per class)
  B6 Hanley-McNeil SE of AUROC (at AUROC 0.75) before vs after
  B7 general test rows touching clinical train (G+C)

Run: python -m scripts.analysis.b_test_overlap
Writes: results/analysis_2026-11-11/b_test_overlap.json; prints markdown tables. Writes nothing under data/.
"""

from __future__ import annotations

import math

from scripts.analysis.common import (BUILD_CLINICAL_SHA, CLINICAL_SEED, CLINICAL_VAL_FRAC, SPLITS,
                                     clinical_rows_before_dedup, import_build_clinical, load_unified_v2, md_table,
                                     write_json)

VARIANTS = (0, 8)
HM_AUROC = 0.75
HANDCHECK_N = 25


def is_constant_query(r: dict) -> bool:
    return r["source"] == "ragtruth" and r.get("task") == "summary"


def hanley_mcneil_se(a: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of AUROC a with n_pos positives and n_neg negatives."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(var)


class KeyIndex:
    """Question and passage keys of a set of rows for one min_words value (overlap_keys computed once per row)."""

    def __init__(self, bc, rows: list[dict], min_words: int):
        self.q, self.p = set(), set()
        for r in rows:
            for k in bc.overlap_keys(r, min_words=min_words):
                if k.startswith("p:"):
                    self.p.add(k)
                elif not is_constant_query(r):
                    self.q.add(k)


def one_passage_keys(bc, text: str, min_words: int) -> set[str]:
    """The "p:" key(s) overlap_keys gives one passage (empty if it is skipped at this min_words)."""
    return {k for k in bc.overlap_keys({"query": "_", "context": text}, min_words=min_words) if k.startswith("p:")}


def row_hits(bc, row: dict, index: KeyIndex, min_words: int) -> tuple[bool, bool]:
    keys = bc.overlap_keys(row, min_words=min_words)
    q = (not is_constant_query(row)) and any(k in index.q for k in keys if k.startswith("q:"))
    p = any(k in index.p for k in keys if k.startswith("p:"))
    return q, p


def count_hits(bc, rows: list[dict], index: KeyIndex, mw: int) -> dict:
    hits = [row_hits(bc, r, index, mw) for r in rows]
    removed = [q or p for q, p in hits]
    pos = sum(r["label"] for r in rows)
    pos_removed = sum(r["label"] for r, x in zip(rows, removed) if x)
    gid_removed = {r["group_id"] for r, x in zip(rows, removed) if x}
    return {
        "rows": len(rows), "positives": pos,
        "removed": sum(removed), "positives_removed": pos_removed,
        "q_only": sum(q and not p for q, p in hits), "p_only": sum(p and not q for q, p in hits),
        "both": sum(q and p for q, p in hits),
        "removed_if_whole_groups": sum(r["group_id"] in gid_removed for r in rows),
        "_removed_mask": removed,
    }


def pct(a, b):
    return 100 * a / b if b else float("nan")


def main() -> None:
    bc, sha = import_build_clinical()
    print(f"scripts/build_clinical.py at {BUILD_CLINICAL_SHA} ({sha}); seed {CLINICAL_SEED}, val_frac {CLINICAL_VAL_FRAC}")
    raw = bc.load_raw()
    clin = clinical_rows_before_dedup(bc, raw)
    general = load_unified_v2()
    gen_train, gen_test = general["train"], general["test"]

    train_parts = {
        "ragbench_pubmedqa train": clin["train"]["ragbench_pubmedqa"],
        "ragbench_covidqa train": clin["train"]["ragbench_covidqa"],
        "pqa_artificial train (90%)": clin["train"][bc.MEDHALLU],
        "unified_v2 train": gen_train,
    }
    clin_train_all = [r for k, rows in train_parts.items() if k != "unified_v2 train" for r in rows]
    test_sources = ("ragbench_pubmedqa", "ragbench_covidqa", bc.MEDHALLU)
    result = {"build_clinical_sha": sha, "seed": CLINICAL_SEED, "val_frac": CLINICAL_VAL_FRAC,
              "training_parts": {k: len(v) for k, v in train_parts.items()},
              "hm_auroc": HM_AUROC, "variants": {}}
    print("Training reference (rows): " + ", ".join(f"{k} {len(v)}" for k, v in train_parts.items()))

    for mw in VARIANTS:
        idx_all = KeyIndex(bc, [r for rows in train_parts.values() for r in rows], mw)
        idx_part = {k: KeyIndex(bc, rows, mw) for k, rows in train_parts.items()}
        res = {"main": {}, "by_part": {}, "d4_d5": {}, "train_x_val": {}, "general_test": {}}

        for src in test_sources:
            rows = clin["test"][src]
            c = count_hits(bc, rows, idx_all, mw)
            mask = c.pop("_removed_mask")
            after = [r for r, x in zip(rows, mask) if not x]
            n_pos_a = sum(r["label"] for r in after)
            n_neg_b, n_neg_a = c["rows"] - c["positives"], len(after) - n_pos_a
            c.update({
                "prevalence_before": c["positives"] / c["rows"], "rows_after": len(after), "positives_after": n_pos_a,
                "negatives_after": n_neg_a, "prevalence_after": n_pos_a / len(after) if after else float("nan"),
                "pct_removed": pct(c["removed"], c["rows"]),
                "hm_se_before": hanley_mcneil_se(HM_AUROC, c["positives"], n_neg_b),
                "hm_se_after": hanley_mcneil_se(HM_AUROC, n_pos_a, n_neg_a) if n_pos_a and n_neg_a else float("nan"),
            })
            res["main"][src] = c
            res["by_part"][src] = {}
            for part, idx in idx_part.items():
                pc = count_hits(bc, rows, idx, mw)
                pc.pop("_removed_mask")
                res["by_part"][src][part] = {k: pc[k] for k in ("removed", "q_only", "p_only", "both")}

        # D4 / D5 under the test-side rule (the training side is the protected one)
        idx_art_val = KeyIndex(bc, clin["val"][bc.MEDHALLU], mw)
        cases = {
            "D4a: MedHallu test x RAGBench PubMedQA train (remove from test)":
                (clin["test"][bc.MEDHALLU], idx_part["ragbench_pubmedqa train"]),
            "D4b: RAGBench PubMedQA val x pqa_artificial train (remove from val)":
                (clin["val"]["ragbench_pubmedqa"], idx_part["pqa_artificial train (90%)"]),
            "D5: MedHallu test x pqa_artificial train (remove from test)":
                (clin["test"][bc.MEDHALLU], idx_part["pqa_artificial train (90%)"]),
            "D5 (reported): MedHallu test x pqa_artificial val":
                (clin["test"][bc.MEDHALLU], idx_art_val),
        }
        for name, (rows, idx) in cases.items():
            c = count_hits(bc, rows, idx, mw)
            c.pop("_removed_mask")
            res["d4_d5"][name] = c
        # distinct passages behind D5 (MedHallu test x pqa_artificial train)
        test_p = KeyIndex(bc, clin["test"][bc.MEDHALLU], mw).p
        shared = sorted({p for r in clin["train"][bc.MEDHALLU] for p in bc.split_passages(r["context"])
                         if one_passage_keys(bc, p, mw) & test_p})
        res["d5_shared_passages"] = shared

        # train x val (reported only)
        for src in test_sources:
            vrows = clin["val"][src]
            same = count_hits(bc, vrows, KeyIndex(bc, clin["train"][src], mw), mw); same.pop("_removed_mask")
            allc = count_hits(bc, vrows, idx_all, mw); allc.pop("_removed_mask")
            res["train_x_val"][src] = {"same_source_train": same, "all_training_data": allc}

        # G+C: general test rows touching clinical train
        idx_clin = KeyIndex(bc, clin_train_all, mw)
        for name, pred in (("halueval", lambda r: r["source"] == "halueval"),
                           ("ragtruth qa", lambda r: r["source"] == "ragtruth" and r["task"] == "qa"),
                           ("ragtruth summary", lambda r: r["source"] == "ragtruth" and r["task"] == "summary")):
            c = count_hits(bc, [r for r in gen_test if pred(r)], idx_clin, mw)
            c.pop("_removed_mask")
            res["general_test"][name] = c
        result["variants"][mw] = res

    # -------------------------------------------------------------- print
    for mw in VARIANTS:
        res = result["variants"][mw]
        print(f"\n### B1. Test-side removal, min_words={mw}")
        print(md_table(["test source", "rows", "positives", "prevalence", "removed", "% removed", "question only",
                        "passage only", "both", "rows after", "positives after", "prevalence after",
                        "removed if whole groups"],
                       [[s, c["rows"], c["positives"], f"{c['prevalence_before']:.3f}", c["removed"],
                         f"{c['pct_removed']:.1f}", c["q_only"], c["p_only"], c["both"], c["rows_after"],
                         c["positives_after"], f"{c['prevalence_after']:.3f}", c["removed_if_whole_groups"]]
                        for s, c in res["main"].items()]))
    for mw in VARIANTS:
        res = result["variants"][mw]
        print(f"\n### B2. Test rows touching each training part (any key; q-only / p-only / both), min_words={mw}")
        parts = list(train_parts)
        print(md_table(["test source"] + parts,
                       [[s] + [f"{v[p]['removed']} ({v[p]['q_only']} / {v[p]['p_only']} / {v[p]['both']})" for p in parts]
                        for s, v in res["by_part"].items()]))
    print("\n### B3. D4 and D5 under the test-side rule")
    print(md_table(["case", "min_words", "rows", "positives", "rows touching", "positives touching",
                    "question only", "passage only", "both"],
                   [[name, mw, c["rows"], c["positives"], c["removed"], c["positives_removed"], c["q_only"],
                     c["p_only"], c["both"]]
                    for mw in VARIANTS for name, c in result["variants"][mw]["d4_d5"].items()]))
    for mw in VARIANTS:
        sp = result["variants"][mw]["d5_shared_passages"]
        print(f"\nD5 shared passages (MedHallu test x pqa_artificial train), min_words={mw}: {len(sp)} distinct"
              + "".join(f"\n- {p[:100]!r}" for p in sp[:20]) + ("\n- ..." if len(sp) > 20 else ""))
    print("\n### B4. Validation rows touching training data (reported only, nothing removed)")
    print(md_table(["val source", "min_words", "val rows", "vs same-source train: rows (q-only / p-only / both)",
                    "vs all training data: rows (q-only / p-only / both)"],
                   [[s, mw, v["same_source_train"]["rows"],
                     f"{v['same_source_train']['removed']} ({v['same_source_train']['q_only']} / "
                     f"{v['same_source_train']['p_only']} / {v['same_source_train']['both']})",
                     f"{v['all_training_data']['removed']} ({v['all_training_data']['q_only']} / "
                     f"{v['all_training_data']['p_only']} / {v['all_training_data']['both']})"]
                    for mw in VARIANTS for s, v in result["variants"][mw]["train_x_val"].items()]))
    print(f"\n### B5. Hand-check feasibility: RAGBench test rows left per class (need >= {HANDCHECK_N} each)")
    print(md_table(["test source", "min_words", "positives left", "negatives left", f">= {HANDCHECK_N} per class"],
                   [[s, mw, c["positives_after"], c["negatives_after"],
                     "yes" if min(c["positives_after"], c["negatives_after"]) >= HANDCHECK_N else "NO"]
                    for mw in VARIANTS for s, c in result["variants"][mw]["main"].items() if s.startswith("ragbench")]))
    print(f"\n### B6. Hanley-McNeil SE of AUROC at AUROC {HM_AUROC}")
    rows6 = []
    for s in result["variants"][0]["main"]:
        b = result["variants"][0]["main"][s]
        line = [s, b["positives"], b["rows"] - b["positives"], f"{b['hm_se_before']:.4f}"]
        for mw in VARIANTS:
            c = result["variants"][mw]["main"][s]
            line += [c["positives_after"], c["negatives_after"], f"{c['hm_se_after']:.4f}",
                     f"{c['hm_se_after'] / c['hm_se_before']:.2f}"]
        rows6.append(line)
    hdr = ["test source", "pos before", "neg before", "SE before"]
    for mw in VARIANTS:
        hdr += [f"pos after (mw={mw})", f"neg after (mw={mw})", f"SE after (mw={mw})", f"SE ratio (mw={mw})"]
    print(md_table(hdr, rows6))
    print("\n### B7. General test rows touching clinical train (G+C; reported only)")
    print(md_table(["general test", "min_words", "rows", "positives", "rows touching", "question only", "passage only", "both"],
                   [[s, mw, c["rows"], c["positives"], c["removed"], c["q_only"], c["p_only"], c["both"]]
                    for mw in VARIANTS for s, c in result["variants"][mw]["general_test"].items()]))

    path = write_json(result, "b_test_overlap.json")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
