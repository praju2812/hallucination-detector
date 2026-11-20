"""Read-only: rows per source and split whose encoding changes from the legacy rule to fit_first (20 Nov 2026).

Both encodings come from Collator.encode_with_response_budget (src/encoding.py) with base.yaml's max_length and
resp_max; only `truncation` differs. A row "changes" when the two id lists differ. Untruncated lengths (for the
"now uncut" column) come from scripts.analysis.common.full_lengths, the 11 Nov helper.

Columns:
    changed:         encodings differ.
    now uncut:       changed rows that fit_first encodes in full (legacy had cut only the response, to resp_max).
    still cut:       changed rows that fit_first still truncates (overflow; the response keeps more than resp_max).
    fit_first raises: rows where fit_first refuses the row (context or response would end with 0 tokens).

Data: data/unified_v2/{train,val,test}.jsonl (general) and data/clinical/{train.jsonl.gz, val, test_full, test}.jsonl
(test.jsonl = test_clean). Prints one table; writes nothing.

Run: python -m scripts.analysis.encoder_change_count [--limit N]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from scripts.analysis.common import full_lengths, load_config, load_tokenizer, md_table
from src.encoding import Collator
from utils.io import read_jsonl

SPLIT_FILES = {
    "general": {"train": "data/unified_v2/train.jsonl", "val": "data/unified_v2/val.jsonl",
                "test": "data/unified_v2/test.jsonl"},
    "clinical": {"train": "data/clinical/train.jsonl.gz", "val": "data/clinical/val.jsonl",
                 "test_full": "data/clinical/test_full.jsonl", "test_clean": "data/clinical/test.jsonl"},
}


def count_changes(rows: list[dict], legacy: Collator, fit: Collator, totals, cols: dict) -> dict:
    out = {"rows": len(rows), "changed": 0, "now_uncut": 0, "still_cut": 0, "raises": 0}
    for r, total in zip(rows, totals):
        q, c, a = r[cols["query_col"]], r[cols["context_col"]], r[cols["response_col"]]
        old = legacy.encode_with_response_budget(q, c, a)
        try:
            new = fit.encode_with_response_budget(q, c, a)
        except ValueError:
            out["raises"] += 1
            continue
        if new != old:
            out["changed"] += 1
            out["now_uncut" if len(new) == total else "still_cut"] += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="first N rows per file (quick look only, not for a report)")
    args = ap.parse_args(argv)

    cfg = load_config()
    m, cols = cfg["model"], cfg["data"]
    tok = load_tokenizer(cfg)
    legacy = Collator(tok, m["max_length"], m["resp_max"], truncation="legacy")
    fit = Collator(tok, m["max_length"], m["resp_max"], truncation="fit_first")
    # full_lengths reads the query / context / answer fields by name
    assert (cols["query_col"], cols["context_col"], cols["response_col"]) == ("query", "context", "answer"), cols

    table = []
    for family, files in SPLIT_FILES.items():
        for split, path in files.items():
            rows = read_jsonl(path)[:args.limit]
            by_source = defaultdict(list)
            for r in rows:
                by_source[r[cols["dataset_col"]]].append(r)
            for source, srows in sorted(by_source.items()):
                totals = full_lengths(tok, srows)["total"]
                c = count_changes(srows, legacy, fit, totals, cols)
                table.append([source, split, c["rows"], c["changed"], f"{100 * c['changed'] / c['rows']:.1f}",
                              c["now_uncut"], c["still_cut"], c["raises"]])

    print(f"Encoding changes, legacy -> fit_first (max_length {m['max_length']}, resp_max {m['resp_max']})"
          + (f"; PARTIAL: first {args.limit} rows per file" if args.limit else ""))
    print(md_table(["source", "split", "rows", "changed", "% changed", "now uncut", "still cut", "fit_first raises"], table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
