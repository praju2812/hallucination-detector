"""Part D (supervisor analysis, 11 Nov): does the kept question carry label information?

For HaluEval and RAGTruth QA rows of data/unified_v2 (all splits), single-feature AUROC for the label of
  (i)   question-answer content-word Jaccard        (iii) context-answer content-word Jaccard
  (ii)  question-answer TF-IDF cosine               (iv)  context-answer TF-IDF cosine
  (v)   answer length in words (baseline)
with 95% CIs from a bootstrap over group_id. Then (i)-(iv) again after stratifying by answer-length decile
(within-decile percentile rank, pooled; and the weighted mean of within-decile AUROCs), to separate a question
effect from an answer-length effect. Finally the dropped RAGTruth QA instruction: one constant string, and its token count.

Content words = lowercased tokens matching sklearn's default pattern \\b\\w\\w+\\b, minus sklearn ENGLISH_STOP_WORDS.
TF-IDF = TfidfVectorizer(lowercase=True, stop_words="english"), fitted per source on the unique question, answer and
context texts (all splits); rows are L2-normalised, so cosine = dot product.

Run: python -m scripts.analysis.d_question_cue
Writes: results/analysis_2026-11-11/d_question_cue.json; prints markdown tables.
"""

from __future__ import annotations

import re

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from scripts.analysis.c_summary_length_cue import QA_LINE1, qa_instruction_check
from scripts.analysis.common import (BOOT_SEED, N_BOOT, SPLITS, auroc, group_bootstrap_ci, load_source_info,
                                     load_tokenizer, load_unified_v2, md_table, stratified_auroc, write_json)

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
SOURCES = {"halueval": lambda r: r["source"] == "halueval",
           "ragtruth_qa": lambda r: r["source"] == "ragtruth" and r["task"] == "qa"}
FEATURES = ("q-a Jaccard", "q-a TF-IDF cosine", "ctx-a Jaccard", "ctx-a TF-IDF cosine", "answer words")
STRATIFIED = ("q-a Jaccard", "q-a TF-IDF cosine", "ctx-a Jaccard", "ctx-a TF-IDF cosine")
N_DECILES = 10


def content_words(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def tfidf_cosines(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(question-answer cosine, context-answer cosine) per row, one vectorizer fitted on this source's unique texts."""
    texts = sorted({t for r in rows for t in (r["query"], r["answer"], r["context"])})
    vec = TfidfVectorizer(lowercase=True, stop_words="english").fit(texts)
    q, a, c = (vec.transform([r[k] for r in rows]) for k in ("query", "answer", "context"))
    qa = np.asarray(q.multiply(a).sum(axis=1)).ravel()
    ca = np.asarray(c.multiply(a).sum(axis=1)).ravel()
    return qa, ca


def length_deciles(words: np.ndarray) -> np.ndarray:
    """Decile 0..9 from the average rank of answer length; tied lengths always share a decile."""
    rank = stats.rankdata(words, method="average")
    return np.minimum(N_DECILES - 1, np.floor((rank - 0.5) / len(words) * N_DECILES)).astype(int)


def main() -> None:
    data = load_unified_v2()
    all_rows = [r for s in SPLITS for r in data[s]]
    result = {"bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "group_id"}, "sources": {}}

    for src, pred in SOURCES.items():
        rows = [r for r in all_rows if pred(r)]
        y = np.array([r["label"] for r in rows])
        groups = np.array([r["group_id"] for r in rows])
        qa_cos, ca_cos = tfidf_cosines(rows)
        ans_words = [content_words(r["answer"]) for r in rows]
        feats = {
            "q-a Jaccard": np.array([jaccard(content_words(r["query"]), a) for r, a in zip(rows, ans_words)]),
            "q-a TF-IDF cosine": qa_cos,
            "ctx-a Jaccard": np.array([jaccard(content_words(r["context"]), a) for r, a in zip(rows, ans_words)]),
            "ctx-a TF-IDF cosine": ca_cos,
            "answer words": np.array([len(r["answer"].split()) for r in rows], float),
        }
        res = {"n": len(rows), "positives": int(y.sum()), "groups": len(set(groups)), "features": {}, "stratified": {}}
        for name in FEATURES:
            x = feats[name]
            lo, hi = group_bootstrap_ci(y, x, groups)
            res["features"][name] = {"auroc": auroc(y, x), "ci95": [lo, hi],
                                     "mean_label0": float(x[y == 0].mean()), "mean_label1": float(x[y == 1].mean())}
        dec = length_deciles(feats["answer words"])
        res["length_deciles_nonempty"] = int(len(set(dec)))
        res["decile_sizes"] = {int(d): f"{int(y[dec == d].sum())}/{int((dec == d).sum())}" for d in sorted(set(dec))}
        for name in STRATIFIED:
            weighted, rank = stratified_auroc(y, feats[name], dec)
            lo, hi = group_bootstrap_ci(y, rank, groups)
            res["stratified"][name] = {"pooled_raw": res["features"][name]["auroc"], "weighted_within_decile": weighted,
                                       "pooled_within_decile_rank": auroc(y, rank), "rank_ci95": [lo, hi]}
        result["sources"][src] = res

    for src, res in result["sources"].items():
        print(f"\n### D1. {src}: single-feature AUROC for the label (all splits; n {res['n']}, positives {res['positives']}, "
              f"groups {res['groups']})")
        print(md_table(["feature", "AUROC", "95% CI (group bootstrap)", "mean, label 0", "mean, label 1"],
                       [[f, f"{v['auroc']:.3f}", "{:.3f} to {:.3f}".format(*v["ci95"]),
                         f"{v['mean_label0']:.3f}", f"{v['mean_label1']:.3f}"] for f, v in res["features"].items()]))
    for src, res in result["sources"].items():
        print(f"\n### D2. {src}: AUROC after stratifying by answer-length decile "
              f"({res['length_deciles_nonempty']} non-empty deciles; positives/rows per decile {res['decile_sizes']})")
        print(md_table(["feature", "pooled AUROC", "weighted mean of within-decile AUROCs",
                        "pooled AUROC of within-decile rank", "95% CI (rank version)"],
                       [[f, f"{v['pooled_raw']:.3f}", f"{v['weighted_within_decile']:.3f}",
                         f"{v['pooled_within_decile_rank']:.3f}", "{:.3f} to {:.3f}".format(*v["rank_ci95"])]
                        for f, v in res["stratified"].items()]))

    # D3 dropped QA instruction
    qa = qa_instruction_check(load_source_info(), data)
    if qa["distinct_templates"] != 1 or qa["line1_counts"].get(QA_LINE1) != qa["qa_sources"]:
        raise ValueError(f"QA instruction is not one constant string: {qa['distinct_templates']} templates")
    (template,) = qa["templates"]
    fixed = template.replace("{question}", "").replace("{passages}", "")
    tok = load_tokenizer()
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    result["qa_instruction"] = {"qa_sources": qa["qa_sources"], "distinct_templates": qa["distinct_templates"],
                                "template": template, "line1_tokens": ntok(QA_LINE1), "fixed_text_tokens": ntok(fixed)}
    print("\n### D3. Dropped RAGTruth QA instruction")
    print(md_table(["check", "value"], [
        ["QA sources", qa["qa_sources"]],
        ["distinct templates (prompt minus question and passages)", qa["distinct_templates"]],
        [f"tokens in line 1 '{QA_LINE1}' (ModernBERT, no special tokens)", result["qa_instruction"]["line1_tokens"]],
        ["tokens in the whole fixed template text (question and passages removed)", result["qa_instruction"]["fixed_text_tokens"]],
    ]))

    path = write_json(result, "d_question_cue.json")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
