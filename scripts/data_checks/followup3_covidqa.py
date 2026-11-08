"""Follow-up #3: RAGBench covidqa — unique questions per split and responses per
question; reconcile the 1,765 HF rows with the paper's Table 1 (~3,526). READ-ONLY."""
from datasets import load_dataset
from collections import Counter
from common import norm

rb = load_dataset("rungalileo/ragbench","covidqa")
grand_rows=grand_q=0
for split in rb:
    qs=[norm(q) for q in rb[split]["question"]]
    n=len(qs); uq=len(set(qs))
    per_q=Counter(qs)
    dist=Counter(per_q.values())  # responses-per-question -> count of questions
    grand_rows+=n; grand_q+=uq
    print(f"covidqa/{split}: rows={n}  unique_questions={uq}  responses/question distribution={dict(sorted(dist.items()))}  (mean {n/uq:.2f})")
print(f"TOTAL rows={grand_rows}  (sum unique per split={grand_q})")
# global unique across splits
allq=[norm(q) for split in rb for q in rb[split]["question"]]
print(f"unique questions across ALL splits={len(set(allq))}")
print(f"Paper Table 1 covidqa = 2.5k/534/492 = 3,526; HF total = {grand_rows}; ratio = {3526/grand_rows:.2f}")
