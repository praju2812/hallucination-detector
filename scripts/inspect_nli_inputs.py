"""Phase 6 (M5): inspect NLI inputs before deciding the hypothesis format (read-only; prints only).

Open decision (Prajakta): does the question go into the hypothesis for QA rows?
This prints what each choice would feed the model, on VALIDATION rows only (labels are never printed):
    per split/source/task: rows, sentences per row, fallback rows, share of QA sentences that start with a
        pronoun or Yes/No (they may not stand alone without the question), the longest sentences;
    with --tokenizer: pairs, windows per row and capped hypotheses for each format;
    seeded QA examples: question, answer, split sentences and both hypothesis variants;
    per source, rows with at least one "preamble-like" sentence, plus 10 seeded matching sentences.

Run:
    python -m scripts.inspect_nli_inputs
    python -m scripts.inspect_nli_inputs --tokenizer MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from collections import defaultdict

import numpy as np

from scripts.score_nli import HYPOTHESIS_FORMATS, PairBuilder, load_split, n_pairs, prepare_row
from src.nli import HYP_CAP, MAX_LENGTH, STRIDE, build_hypotheses, split_response_sentences

INSPECT_SPLITS = ["ragtruth_val", "clinical_val"]          # validation only: the decision must not look at test
DEPENDENT_START = re.compile(r"^\W*(It|Its|This|These|That|Those|They|Their|He|She|Yes|No)\b")

# "Preamble-like" sentences (Prajakta, 19 Nov; narrowed the same day): ends with ":" OR starts with one of these
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


def token_stats(rows: list[dict], builder: PairBuilder, include_question: bool, stride: int, hyp_cap: int) -> dict:
    prepared = [prepare_row(r, builder, include_question, stride, MAX_LENGTH, hyp_cap) for r in rows]
    return {"pairs": sum(n_pairs(p) for p in prepared),
            "windows": [max(len(s) for s in p["spans"]) for p in prepared],
            "capped": sum(p["n_capped"] for p in prepared),
            "hyp_tokens": [len(h) for p in prepared for h in p["hyps"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--general-dir", default="data/unified_v2")
    ap.add_argument("--clinical-dir", default="data/clinical")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for token-level stats (optional)")
    ap.add_argument("--examples", type=int, default=5, help="QA examples per source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--hyp-cap", type=int, default=HYP_CAP)
    args = ap.parse_args(argv)

    builder = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        builder = PairBuilder(AutoTokenizer.from_pretrained(args.tokenizer))
        builder.verify()

    rng = np.random.default_rng(args.seed)
    print("NLI input inspection (validation rows only; labels not shown)\n")
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
            if builder is not None:
                for fmt, incl in HYPOTHESIS_FORMATS.items():
                    if incl and task != "qa":
                        continue
                    ts = token_stats(rows, builder, incl, args.stride, args.hyp_cap)
                    print(f"   [{fmt}] pairs {ts['pairs']}; windows per row {_dist(ts['windows'])}; "
                          f"hypothesis tokens {_dist(ts['hyp_tokens'])}; capped hypotheses {ts['capped']}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
