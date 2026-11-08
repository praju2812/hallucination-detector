"""Step 3: MedHallu fields/counts/examples + PubMedQA knowledge-content check.
Matching to PubMedQA is by normalised QUESTION text (MedHallu ships no pubid)."""
import json
from datasets import load_dataset
from common import norm

def short(v, n=200): return (str(v)[:n] + ("…" if len(str(v))>n else ""))

out={}
for cfg in ["pqa_labeled","pqa_artificial"]:
    d=load_dataset("UTAustin-AIHealth/MedHallu",cfg)["train"]
    out[cfg]={"n":len(d),"fields":list(d.features.keys()),
              "examples":[{k:short(d[i][k]) for k in d.features} for i in range(3)]}
print("=== MedHallu fields & counts ===")
for cfg,v in out.items():
    print(f"[{cfg}] n={v['n']} fields={v['fields']}")
print("\n=== 3 example rows per config (truncated) ===")
for cfg,v in out.items():
    print(f"--- {cfg} ---")
    for i,ex in enumerate(v["examples"]):
        print(f" row{i}:")
        for k,val in ex.items(): print(f"    {k}: {val!r}")

# PubMedQA knowledge-content check
print("\n=== Knowledge-content check (MedHallu Knowledge vs PubMedQA LONG_ANSWER vs CONTEXTS) ===")
for cfg in ["pqa_labeled","pqa_artificial"]:
    pq=load_dataset("qiaojin/PubMedQA",cfg)["train"]
    by_q={norm(r["question"]):r for r in pq}
    mh=load_dataset("UTAustin-AIHealth/MedHallu",cfg)["train"]
    matched=contains_la=contains_ctx_all=contains_ctx_any=0
    checked=0
    for r in mh:
        pr=by_q.get(norm(r["Question"]))
        if pr is None: continue
        matched+=1
        k=norm(r["Knowledge"])
        la=norm(pr["long_answer"])
        ctxs=[norm(c) for c in pr["context"]["contexts"]]
        has_la = bool(la) and la in k
        has_ctx_all = all(c in k for c in ctxs) if ctxs else False
        has_ctx_any = any(c in k for c in ctxs) if ctxs else False
        contains_la += has_la; contains_ctx_all += has_ctx_all; contains_ctx_any += has_ctx_any
        checked+=1
    print(f"[{cfg}] MedHallu rows={len(mh)}  matched_by_question={matched}  checked={checked}")
    print(f"    Knowledge contains LONG_ANSWER (conclusion): {contains_la}/{checked}  ({contains_la/checked:.1%})")
    print(f"    Knowledge contains ALL CONTEXTS: {contains_ctx_all}/{checked} ({contains_ctx_all/checked:.1%}); contains ANY context: {contains_ctx_any}/{checked} ({contains_ctx_any/checked:.1%})")
