"""Step 4: RAGBench PubMedQA & CovidQA — fields, per-split counts, and
response-level hallucination rate computed from adherence_score (False=hallucinated)."""
from datasets import load_dataset
from collections import Counter

for cfg in ["pubmedqa","covidqa"]:
    rb=load_dataset("rungalileo/ragbench",cfg)
    print(f"\n==== RAGBench / {cfg} ====")
    s0=list(rb.keys())[0]
    print("fields:",list(rb[s0].features.keys()))
    tot=totF=0
    for split in rb:
        vals=[str(x) for x in rb[split]["adherence_score"]]
        c=Counter(vals); n=len(vals)
        # hallucinated = adherence False; exclude 'None' from denominator, report separately
        false=c.get("False",0); true=c.get("True",0); none=c.get("None",0)
        denom=false+true
        rate=false/denom if denom else float('nan')
        print(f"  {split}: n={n}  adherent(True)={true}  non-adherent/hallucinated(False)={false}  None={none}  -> hallucination rate={rate:.1%} (of {denom})")
        tot+=denom; totF+=false
    print(f"  OVERALL {cfg}: hallucination rate={totF/tot:.1%} ({totF}/{tot})")
