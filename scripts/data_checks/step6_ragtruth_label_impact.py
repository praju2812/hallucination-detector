"""Step 6 measurement: how implicit_true / due_to_null spans affect the
response-level label produced by scripts/build_dataset.py (line 125:
`"label": 1 if r["labels"] else 0`). READ-ONLY; does not change any code.

Mirrors build_dataset.load_ragtruth: data2txt is skipped; a response is labelled
1 iff its labels list is non-empty, regardless of span-level implicit_true/due_to_null.
"""
import json, os, sys

RT = "data/raw/RAGTruth/dataset"
resp_path = os.path.join(RT, "response.jsonl")
src_path = os.path.join(RT, "source_info.jsonl")

TASK_MAP = {"QA": "qa", "Summary": "summary", "Data2txt": "data2txt"}
info = {}
for line in open(src_path):
    s = json.loads(line)
    info[s["source_id"]] = TASK_MAP[s["task_type"]]

resps = [json.loads(l) for l in open(resp_path)]

def bucket(task_filter):
    n=pos=0
    only_impl=only_null=only_impl_or_null=0
    for r in resps:
        task = info.get(r["source_id"])
        if task is None: continue
        if task_filter and task not in task_filter: continue
        labs = r.get("labels") or []
        n += 1
        label = 1 if labs else 0        # EXACT rule from build_dataset.py line 125
        pos += label
        if labs:
            impl = [s.get("implicit_true", False) for s in labs]
            null = [s.get("due_to_null", False) for s in labs]
            if all(impl): only_impl += 1
            if all(null): only_null += 1
            if all(impl[i] or null[i] for i in range(len(labs))): only_impl_or_null += 1
    return n, pos, only_impl, only_null, only_impl_or_null

for name, flt in [("ALL tasks", None), ("QA+Summary (as built; data2txt dropped)", {"qa","summary"}), ("QA only", {"qa"}), ("Summary only", {"summary"}), ("Data2txt only", {"data2txt"})]:
    n,pos,oi,on,oio = bucket(flt)
    print(f"[{name}] responses={n}  labelled_hallucinated(label=1)={pos} ({pos/n:.1%})")
    print(f"    of which hallucinated-only-because-of spans that are ALL implicit_true: {oi}")
    print(f"    ... ALL due_to_null: {on}   ... ALL (implicit_true OR due_to_null): {oio}")
