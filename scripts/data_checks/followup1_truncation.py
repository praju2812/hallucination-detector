"""Follow-up #1: combined context-truncation fraction under the detector's real
input format (src/encoding.py encode_with_response_budget), max_length=1024,
resp_max=384. Context is truncated iff ctx_len > (budget - resp_keep),
budget = 1024 - num_special_tokens_to_add(pair=True),
resp_keep = min(resp_len, resp_max, budget). READ-ONLY; reimplements the logic
(does not import the project's torch module)."""
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MAXLEN, RESP_MAX = 1024, 384
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
N_SPECIAL = tok.num_special_tokens_to_add(pair=True)
BUDGET = MAXLEN - N_SPECIAL
print(f"num_special_tokens_to_add(pair=True) = {N_SPECIAL}; budget = {BUDGET}; resp_max = {RESP_MAX}")

def tlen(s): return len(tok(s, add_special_tokens=False)["input_ids"])

def frac_ctx_truncated(contexts, responses):
    n = len(contexts); trunc = 0
    for c, r in zip(contexts, responses):
        cl, rl = tlen(c), tlen(r)
        resp_keep = min(rl, RESP_MAX, BUDGET)
        ctx_keep = max(0, BUDGET - resp_keep)
        if cl > ctx_keep: trunc += 1
    return trunc, n, trunc / n

def join_list(x): return "\n\n".join(x) if isinstance(x, list) else str(x)

print("\n== MedHallu (context = Knowledge; both answer types) ==")
for cfg in ["pqa_labeled","pqa_artificial"]:
    d = load_dataset("UTAustin-AIHealth/MedHallu", cfg)["train"]
    ctx = [join_list(x) for x in d["Knowledge"]]
    for ans in ["Hallucinated Answer","Ground Truth"]:
        resp = [x for x in d[ans]]
        t,n,f = frac_ctx_truncated(ctx, resp)
        print(f"  MedHallu/{cfg} [train] response={ans}: ctx truncated {t}/{n} = {f:.2%}")

print("\n== RAGBench (context = documents joined; response = response) ==")
for cfg in ["pubmedqa","covidqa"]:
    rb = load_dataset("rungalileo/ragbench", cfg)
    for split in rb:
        ctx = [join_list(x) for x in rb[split]["documents"]]
        resp = [x or "" for x in rb[split]["response"]]
        t,n,f = frac_ctx_truncated(ctx, resp)
        print(f"  RAGBench/{cfg} [{split}]: ctx truncated {t}/{n} = {f:.2%}")
