"""Step 7: ModernBERT-base token-length distributions (p50/p95/p99) for context
and response, and truncation fractions at a 1024-token context window and a
384-token response budget. Token counts use add_special_tokens=False."""
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

tok=AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
def tlen(s): return len(tok(s, add_special_tokens=False)["input_ids"])
def pct(a): a=np.array(a); return (int(np.percentile(a,50)),int(np.percentile(a,95)),int(np.percentile(a,99)),int(a.max()))
def frac_over(a,b): a=np.array(a); return (a>b).mean()

def report(name, contexts, responses):
    c=[tlen(x) for x in contexts]; r=[tlen(x) for x in responses]
    cp=pct(c); rp=pct(r)
    print(f"[{name}] n={len(contexts)}")
    print(f"   context  tokens  p50={cp[0]} p95={cp[1]} p99={cp[2]} max={cp[3]}  | frac > 1024 = {frac_over(c,1024):.1%}")
    print(f"   response tokens  p50={rp[0]} p95={rp[1]} p99={rp[2]} max={rp[3]}  | frac > 384  = {frac_over(r,384):.1%}")

# MedHallu: context=Knowledge (List(string) -> join); response = Hallucinated Answer / Ground Truth
for cfg in ["pqa_labeled","pqa_artificial"]:
    d=load_dataset("UTAustin-AIHealth/MedHallu",cfg)["train"]
    ctx=["\n\n".join(x) if isinstance(x,list) else str(x) for x in d["Knowledge"]]
    report(f"MedHallu/{cfg} (response=Hallucinated Answer)", ctx, [x for x in d["Hallucinated Answer"]])
    report(f"MedHallu/{cfg} (response=Ground Truth)",      ctx, [x for x in d["Ground Truth"]])

# RAGBench: context=documents(joined), response=response (concatenate all splits)
for cfg in ["pubmedqa","covidqa"]:
    rb=load_dataset("rungalileo/ragbench",cfg)
    ctx=[]; resp=[]
    for split in rb:
        for r in rb[split]:
            docs=r["documents"] or []
            ctx.append("\n\n".join(docs) if isinstance(docs,list) else str(docs))
            resp.append(r["response"] or "")
    report(f"RAGBench/{cfg} (all splits)", ctx, resp)
