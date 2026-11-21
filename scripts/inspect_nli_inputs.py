"""Phase 6 (M5): inspect NLI inputs before deciding the hypothesis format (read-only; prints only).

Open decision: does the question go into the hypothesis for QA rows?
This prints what each choice would feed the model, on VALIDATION rows only (labels are never printed):
    per split/source/task: rows, sentences per row, fallback rows, share of QA sentences that start with a
        pronoun or Yes/No (they may not stand alone without the question), the longest sentences;
    with --tokenizer, three token tables per source/task (the NLI checkpoint's tokenizer; method unchanged):
        1. context sentences (newlines, then pysbd; distinct contexts): share longer than 128 tokens, and the max;
        2. hypotheses under both formats: share capped at HYP_CAP, questions shortened or dropped by the cap,
           sentences cut, and the longest hypothesis before the cap;
        3. windows per row (max over the row's sentences) under both formats: median, p95, max, and total pairs;
    seeded QA examples: question, answer, split sentences and both hypothesis variants;
    per source, rows with at least one "preamble-like" sentence, plus 10 seeded matching sentences;
    per source, rows changed by the preamble step (normalize_preambles), plus 10 seeded before/after examples.
    Sentence counts and examples above the last section are BEFORE the preamble step; the token tables are after
    it (they use score_nli.prepare_row).

Run:
    python -m scripts.inspect_nli_inputs
    python -m scripts.inspect_nli_inputs --tokenizer MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
        --revision b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7 [--tokens-only]
    python -m scripts.inspect_nli_inputs --preambles-only
The tokenizer is loaded from its tokenizer files alone (no config.json, no weights): see load_nli_tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.score_nli import HYPOTHESIS_FORMATS, PairBuilder, load_split, n_pairs, prepare_row
from src.nli import (
    HYP_CAP,
    MAX_LENGTH,
    STRIDE,
    build_hypotheses,
    get_segmenter,
    normalize_preambles,
    split_response_sentences,
)

INSPECT_SPLITS = ["ragtruth_val", "clinical_val"]          # validation only: the decision must not look at test
DEPENDENT_START = re.compile(r"^\W*(It|Its|This|These|That|Those|They|Their|He|She|Yes|No)\b")
CTX_SENTENCE_LIMIT = 128                                   # context sentences longer than this are counted (= STRIDE)

# "Preamble-like" sentences : ends with ":" OR starts with one of these
# phrases (anchored to the sentence start, word boundaries, case-insensitive). The seeded samples show what matches.
PREAMBLE_PATTERNS = {
    "based_on": re.compile(r"^\s*based on (the )?(given |provided )?(passages|context|information)\b", re.IGNORECASE),
    "here_is": re.compile(r"^\s*here (is|are)\b", re.IGNORECASE),
}
PREAMBLE_SAMPLES = 10


def _dist(xs) -> str:
    xs = np.asarray(xs)
    return f"median {np.median(xs):.0f}, p95 {np.percentile(xs, 95):.0f}, max {xs.max()}"


def _wrap(text: str, width: int = 110, indent: str = "      ") -> str:
    return textwrap.fill(" ".join(text.split()), width=width, initial_indent=indent, subsequent_indent=indent)


def sentence_stats(rows: list[dict]) -> dict:
    per_row, fallback, dependent, total, longest = [], 0, 0, 0, []
    for r in rows:
        s, fb = split_response_sentences(r["answer"])
        per_row.append(len(s))
        fallback += int(fb)
        total += len(s)
        dependent += sum(bool(DEPENDENT_START.match(x)) for x in s)
        longest.extend((len(x.split()), x) for x in s)
    longest.sort(key=lambda t: -t[0])
    return {"rows": len(rows), "per_row": per_row, "fallback": fallback, "sentences": total,
            "dependent": dependent, "longest": longest[:3]}


def preamble_rules(sentence: str) -> list[str]:
    """Names of the preamble rules a sentence matches ([] = not preamble-like)."""
    rules = ["ends_with_colon"] if sentence.rstrip().endswith(":") else []
    return rules + [name for name, pat in PREAMBLE_PATTERNS.items() if pat.search(sentence)]


def preamble_stats(rows: list[dict]) -> dict:
    """Rows with at least one preamble-like sentence, and every matching (sentence, rules) in row order."""
    rows_with, n_sentences, matches = 0, 0, []
    for r in rows:
        s, _ = split_response_sentences(r["answer"])
        n_sentences += len(s)
        hits = [(x, preamble_rules(x)) for x in s]
        hits = [(x, rules) for x, rules in hits if rules]
        rows_with += int(bool(hits))
        matches.extend(hits)
    return {"rows": len(rows), "rows_with": rows_with, "sentences": n_sentences, "matches": matches}


def normalize_stats(rows: list[dict]) -> dict:
    """What normalize_preambles changes: rows affected (output differs from the split sentences), prefixes
    stripped, colon-line merges, and every affected row's (before, after) sentences in row order."""
    affected, stripped, merged = [], 0, 0
    for r in rows:
        before, _ = split_response_sentences(r["answer"])
        after, n_s, n_m = normalize_preambles(before)
        stripped += n_s
        merged += n_m
        if after != before:
            affected.append((before, after))
    return {"rows": len(rows), "affected": len(affected), "stripped": stripped, "merged": merged,
            "examples": affected}


def print_normalize_section(by_source: dict, seed: int, n_examples: int = PREAMBLE_SAMPLES) -> None:
    """Per source: rows affected by normalize_preambles and n_examples seeded before/after examples."""
    print("== Preamble step (normalize_preambles; validation rows, per source)\n")
    rng = np.random.default_rng(seed)                       # own generator: the sections above do not change
    table, examples = [], []
    for source, rows in sorted(by_source.items()):
        ns = normalize_stats(rows)
        table.append([source, ns["rows"], ns["affected"], _pct(ns["affected"], ns["rows"]), ns["stripped"],
                      ns["merged"]])
        if ns["examples"]:
            pick = rng.choice(len(ns["examples"]), size=min(n_examples, len(ns["examples"])), replace=False)
            examples.append((source, [ns["examples"][int(i)] for i in sorted(pick)]))
    print(md_table(["source", "rows", "rows affected", "% affected", "prefixes stripped", "colon lines merged"],
                   table))
    print()
    for source, exs in examples:
        print(f"-- {source}: {len(exs)} seeded examples")
        for before, after in exs:
            print("   before:")
            for j, x in enumerate(before):
                print(f"     s{j}: {' '.join(x.split())[:200]!r}")
            print("   after:")
            for j, x in enumerate(after):
                print(f"     s{j}: {' '.join(x.split())[:200]!r}")
            print()


# ---- token tables (need --tokenizer) -------------------------------------------------------------

def load_nli_tokenizer(name: str, revision: str | None = None):
    """(tokenizer, revision) from the tokenizer files only.

    transformers 5's AutoTokenizer also reads config.json, which is not downloaded until the model is, so the
    class named in tokenizer_config.json is loaded directly. revision = the snapshot SHA the files came from
    ("local" for a folder).
    """
    import transformers
    if os.path.isdir(name):
        cfg_path, rev = os.path.join(name, "tokenizer_config.json"), "local"
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(name, "tokenizer_config.json", revision=revision)
        rev = Path(cfg_path).parent.name                    # .../snapshots/<sha>/tokenizer_config.json
    with open(cfg_path) as f:
        cls_name = json.load(f)["tokenizer_class"]
    return getattr(transformers, cls_name).from_pretrained(name, revision=revision), rev


def split_context_sentences(text: str) -> list[str]:
    """Inspection only (the method never splits contexts): newlines first, then pysbd per line, as rules 1-2 of
    split_response_sentences. Pieces are stripped; pieces with no letter or digit are dropped; nothing is merged."""
    seg = get_segmenter()
    out = []
    for line in text.split("\n"):
        if line.strip():
            out.extend(p.strip() for p in seg.segment(line) if any(ch.isalnum() for ch in p))
    return out


def context_sentence_stats(rows: list[dict], builder: PairBuilder, limit: int = CTX_SENTENCE_LIMIT) -> dict:
    """Token length (no special tokens) of every sentence of every DISTINCT context: RAGTruth responses share
    contexts, so each context is counted once. Each sentence is tokenised on its own."""
    contexts = list(dict.fromkeys(r["context"] for r in rows))
    lengths = [len(builder.ids(s)) for c in contexts for s in split_context_sentences(c)]
    return {"rows": len(rows), "contexts": len(contexts), "sentences": len(lengths),
            "over": sum(n > limit for n in lengths), "max": max(lengths, default=0)}


def format_stats(rows: list[dict], builder: PairBuilder, include_question: bool, stride: int = STRIDE,
                 hyp_cap: int = HYP_CAP, max_length: int = MAX_LENGTH) -> dict:
    """Hypothesis and window counts for one hypothesis format, from score_nli.prepare_row (the scoring path).

    Per hypothesis: capped (anything removed), question shortened (some of it kept) or dropped (none kept; only
    counted when there was a question), sentence cut (sentence alone over hyp_cap), length before the cap
    (question + sentence tokens). Per row: windows = max over the row's sentences (the budget depends on the
    hypothesis length), pairs = sum over sentences.
    """
    out = {"hypotheses": 0, "capped": 0, "with_question": 0, "q_shortened": 0, "q_dropped": 0, "sentence_cut": 0,
           "max_before_cap": 0, "windows": [], "pairs": 0}
    for r in rows:
        p = prepare_row(r, builder, include_question, stride, max_length, hyp_cap)
        pairs = build_hypotheses(r, p["sentences"], include_question)
        n_capped = 0
        for (question, sentence), hyp in zip(pairs, p["hyps"], strict=True):
            n_q, n_s = (len(builder.ids(question)) if question else 0), len(builder.ids(sentence))
            capped = n_q + n_s > len(hyp)
            n_capped += capped
            out["hypotheses"] += 1
            out["capped"] += capped
            out["sentence_cut"] += n_s > hyp_cap
            out["max_before_cap"] = max(out["max_before_cap"], n_q + n_s)
            if n_q:
                q_kept = len(hyp) - min(n_s, hyp_cap)
                out["with_question"] += 1
                out["q_shortened"] += 0 < q_kept < n_q
                out["q_dropped"] += q_kept == 0
        if n_capped != p["n_capped"]:                       # self-check against cap_hypothesis_ids' own flag
            raise AssertionError(f"capped count {n_capped} != prepare_row's {p['n_capped']}")
        out["windows"].append(max(len(s) for s in p["spans"]))
        out["pairs"] += n_pairs(p)
    return out


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "-"


def _q(xs, q: float) -> str:
    return f"{round(float(np.percentile(xs, q)), 1):g}"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def print_token_tables(groups: dict, builder: PairBuilder, stride: int, hyp_cap: int,
                       ctx_limit: int = CTX_SENTENCE_LIMIT) -> None:
    """groups: {(source, task): rows}. Prints the three token tables."""
    t1, t2, t3 = [], [], []
    for (source, task), rows in sorted(groups.items()):
        cs = context_sentence_stats(rows, builder, ctx_limit)
        t1.append([source, task, cs["rows"], cs["contexts"], cs["sentences"], cs["over"],
                   _pct(cs["over"], cs["sentences"]), cs["max"]])
        for fmt, incl in HYPOTHESIS_FORMATS.items():
            if incl and task != "qa":                       # Summary rows never get a question: same as response
                continue
            fs = format_stats(rows, builder, incl, stride, hyp_cap)
            q = incl                                        # question columns apply only to question_response
            t2.append([source, task, fmt, fs["hypotheses"], fs["capped"], _pct(fs["capped"], fs["hypotheses"]),
                       fs["with_question"] if q else "-", fs["q_shortened"] if q else "-",
                       fs["q_dropped"] if q else "-",
                       _pct(fs["q_shortened"] + fs["q_dropped"], fs["with_question"]) if q else "-",
                       fs["sentence_cut"], fs["max_before_cap"]])
            w = fs["windows"]
            t3.append([source, task, fmt, len(w), _q(w, 50), _q(w, 95), max(w), fs["pairs"]])

    print(f"== 1. Context sentences (newlines, then pysbd; distinct contexts; each sentence tokenised alone)\n")
    print(md_table(["source", "task", "rows", "distinct contexts", "sentences", f"> {ctx_limit} tokens",
                    f"% > {ctx_limit}", "max tokens"], t1))
    print(f"\n== 2. Hypotheses (cap {hyp_cap} tokens; question cut first, sentence cut only if it alone is over)\n")
    print(md_table(["source", "task", "format", "hypotheses", "capped", "% capped", "with question",
                    "question shortened", "question dropped", "% shortened or dropped", "sentence cut",
                    "max tokens before cap"], t2))
    print("   Summary rows never get a question, so question_response = response for them (not repeated).")
    print("   % shortened or dropped is of hypotheses with a question.")
    print(f"\n== 3. Windows per row (max over the row's sentences; {MAX_LENGTH}-token input, stride {stride})\n")
    print(md_table(["source", "task", "format", "rows", "median", "p95", "max", "pairs (total)"], t3))
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--general-dir", default="data/unified_v2")
    ap.add_argument("--clinical-dir", default="data/clinical")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer (id or folder) for the token tables (optional)")
    ap.add_argument("--revision", default=None, help="HF revision of --tokenizer")
    ap.add_argument("--tokens-only", action="store_true", help="print only the token tables (needs --tokenizer)")
    ap.add_argument("--preambles-only", action="store_true",
                    help="print only the preamble-step section (after the token tables, if --tokenizer)")
    ap.add_argument("--examples", type=int, default=5, help="QA examples per source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--hyp-cap", type=int, default=HYP_CAP)
    args = ap.parse_args(argv)
    if args.tokens_only and not args.tokenizer:
        ap.error("--tokens-only needs --tokenizer")

    print("NLI input inspection (validation rows only; labels not shown)\n")
    if args.tokenizer:
        tok, rev = load_nli_tokenizer(args.tokenizer, args.revision)
        builder = PairBuilder(tok)
        builder.verify()
        print(f"tokenizer: {args.tokenizer} @ {rev} ({type(tok).__name__}, fast={getattr(tok, 'is_fast', None)})\n")
        groups = defaultdict(list)
        for name in INSPECT_SPLITS:
            for r in load_split(name, args.general_dir, args.clinical_dir)["rows"]:
                groups[(r["source"], r["task"])].append(r)
        print_token_tables(groups, builder, args.stride, args.hyp_cap)
        if args.tokens_only:
            return 0
    if args.preambles_only:
        by_source = defaultdict(list)
        for name in INSPECT_SPLITS:
            for r in load_split(name, args.general_dir, args.clinical_dir)["rows"]:
                by_source[r["source"]].append(r)
        print_normalize_section(by_source, args.seed)
        return 0

    rng = np.random.default_rng(args.seed)
    for name in INSPECT_SPLITS:
        sp = load_split(name, args.general_dir, args.clinical_dir)
        groups = defaultdict(list)
        for r in sp["rows"]:
            groups[(r["source"], r["task"])].append(r)
        for (source, task), rows in sorted(groups.items()):
            st = sentence_stats(rows)
            print(f"== {name} / {source} / {task}: {st['rows']} rows, {st['sentences']} sentences")
            print(f"   sentences per row: {_dist(st['per_row'])}; fallback rows (nothing left after splitting): {st['fallback']}")
            if task == "qa":
                print(f"   QA sentences starting with a pronoun or Yes/No: {st['dependent']} of {st['sentences']} "
                      f"({100 * st['dependent'] / max(st['sentences'], 1):.1f}%)")
            for n_words, s in st["longest"]:
                print(f"   longest ({n_words} words): {s[:140]!r}")
            print()

    print("== QA examples (seeded)\n")
    for name in INSPECT_SPLITS:
        sp = load_split(name, args.general_dir, args.clinical_dir)
        by_src = defaultdict(list)
        for r in sp["rows"]:
            if r["task"] == "qa":
                by_src[r["source"]].append(r)
        for source, rows in sorted(by_src.items()):
            pick = rng.choice(len(rows), size=min(args.examples, len(rows)), replace=False)
            print(f"-- {name} / {source}")
            for i in sorted(int(x) for x in pick):
                r = rows[i]
                s, fb = split_response_sentences(r["answer"])
                print(f"   [{i}] question: {' '.join(str(r.get('query')).split())[:200]}")
                print(f"   answer ({len(s)} sentence(s){', FALLBACK' if fb else ''}):")
                for j, (q, x) in enumerate(build_hypotheses(r, s, include_question=True)):
                    print(f"     s{j}: response:          {x[:160]!r}")
                    print(f"         question_response: {q!r} + {x[:100]!r}")
                print()

    # Preamble-like sentences, per source (own generator, so the examples above do not change)
    print("== Preamble-like sentences (validation rows, per source)")
    print('   rule: ends with ":", or starts with "based on (the )?(given |provided )?(passages|context|information)" '
          'or "here (is|are)" (whole words, case-insensitive)\n')
    by_source = defaultdict(list)
    for name in INSPECT_SPLITS:
        for r in load_split(name, args.general_dir, args.clinical_dir)["rows"]:
            by_source[r["source"]].append(r)
    prng = np.random.default_rng(args.seed)
    for source, rows in sorted(by_source.items()):
        ps = preamble_stats(rows)
        per_rule = defaultdict(int)
        for _, rules in ps["matches"]:
            for rule in rules:
                per_rule[rule] += 1
        print(f"-- {source}: {ps['rows_with']} of {ps['rows']} rows ({100 * ps['rows_with'] / max(ps['rows'], 1):.1f}%) "
              f"have at least one; {len(ps['matches'])} of {ps['sentences']} sentences match "
              f"({', '.join(f'{k} {v}' for k, v in sorted(per_rule.items())) or 'none'})")
        if ps["matches"]:
            pick = prng.choice(len(ps["matches"]), size=min(PREAMBLE_SAMPLES, len(ps["matches"])), replace=False)
            for i in sorted(int(x) for x in pick):
                x, rules = ps["matches"][i]
                print(f"     [{', '.join(rules)}] {' '.join(x.split())[:200]!r}")
        print()

    print_normalize_section(by_source, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
