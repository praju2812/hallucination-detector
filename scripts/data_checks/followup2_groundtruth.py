"""Follow-up #2: for question-joined MedHallu rows, is MedHallu 'Ground Truth'
equal to / contained in PubMedQA long_answer (or vice versa)? READ-ONLY."""
from datasets import load_dataset
from common import norm

for cfg in ["pqa_labeled","pqa_artificial"]:
    pq = load_dataset("qiaojin/PubMedQA", cfg)["train"]
    by_q = {norm(r["question"]): r for r in pq}
    mh = load_dataset("UTAustin-AIHealth/MedHallu", cfg)["train"]
    matched=exact=gt_in_la=la_in_gt=neither=0
    examples=[]
    for r in mh:
        pr = by_q.get(norm(r["Question"]))
        if pr is None: continue
        matched+=1
        gt = norm(r["Ground Truth"]); la = norm(pr["long_answer"])
        if gt==la: exact+=1; cat="EXACT"
        elif gt and gt in la: gt_in_la+=1; cat="GT⊂long_answer"
        elif la and la in gt: la_in_gt+=1; cat="long_answer⊂GT"
        else: neither+=1; cat="NEITHER"
        if len(examples)<3 and cfg=="pqa_labeled":
            examples.append((cat, r["Ground Truth"][:220], pr["long_answer"][:220]))
    print(f"[{cfg}] matched={matched}  EXACT={exact} ({exact/matched:.1%})  GT_contained_in_long_answer={gt_in_la} ({gt_in_la/matched:.1%})  long_answer_contained_in_GT={la_in_gt} ({la_in_gt/matched:.1%})  NEITHER={neither} ({neither/matched:.1%})")
    if cfg=="pqa_labeled":
        for i,(cat,gt,la) in enumerate(examples):
            print(f"  ex{i} [{cat}]\n     GT:  {gt!r}\n     LA:  {la!r}")
