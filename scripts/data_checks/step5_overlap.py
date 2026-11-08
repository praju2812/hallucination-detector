"""Step 5: overlap between MedHallu (pqa_labeled, pqa_artificial) and
RAGBench pubmedqa (train/validation/test) by (a) PubMed ID, (b) normalised
question text, (c) context-document hash."""
import ast, hashlib, json
from datasets import load_dataset
from common import norm

def h(s): return hashlib.sha1(norm(s).encode()).hexdigest()
def docs_from_medhallu(k):
    # MedHallu 'Knowledge' is a List(string); handle list directly, with a
    # string fallback (older cache) via literal_eval.
    if isinstance(k,(list,tuple)): return [str(x) for x in k]
    try:
        v=ast.literal_eval(k)
        if isinstance(v,(list,tuple)): return [str(x) for x in v]
    except Exception: pass
    return [str(k)]

# PubMedQA question->pubid map across all splits (to recover pubids)
print("building PubMedQA question->pubid map (all splits)...")
q2pub={}
for cfg in ["pqa_labeled","pqa_unlabeled","pqa_artificial"]:
    for r in load_dataset("qiaojin/PubMedQA",cfg)["train"]:
        q2pub.setdefault(norm(r["question"]), str(r["pubid"]))
print("  map size:",len(q2pub))

# Load MedHallu configs
MH={}
for cfg in ["pqa_labeled","pqa_artificial"]:
    d=load_dataset("UTAustin-AIHealth/MedHallu",cfg)["train"]
    rows=[]
    for r in d:
        q=norm(r["Question"])
        rows.append({"q":q,"pubid":q2pub.get(q),"dochashes":set(h(x) for x in docs_from_medhallu(r["Knowledge"]))})
    MH[cfg]=rows
    rec=sum(1 for x in rows if x["pubid"])
    print(f"MedHallu {cfg}: n={len(rows)} pubid_recovered={rec}")

# Load RAGBench pubmedqa splits
RB={}
rb=load_dataset("rungalileo/ragbench","pubmedqa")
for split in rb:
    rows=[]
    for r in rb[split]:
        q=norm(r["question"]); docs=r["documents"] or []
        if isinstance(docs,str): docs=[docs]
        rows.append({"q":q,"pubid":q2pub.get(q),"idsuffix":str(r["id"]).split("_",1)[-1],
                     "dochashes":set(h(x) for x in docs)})
    RB[split]=rows
    rec=sum(1 for x in rows if x["pubid"])
    print(f"RAGBench pubmedqa/{split}: n={len(rows)} pubid_recovered_via_question={rec}")

print("\n=== OVERLAP per split pair ===")
examples=[]
for mcfg,mrows in MH.items():
    m_pub=set(x["pubid"] for x in mrows if x["pubid"])
    m_q=set(x["q"] for x in mrows)
    m_doc=set().union(*[x["dochashes"] for x in mrows]) if mrows else set()
    for split,rrows in RB.items():
        r_pub=set(x["pubid"] for x in rrows if x["pubid"])
        r_q=set(x["q"] for x in rrows)
        r_doc=set().union(*[x["dochashes"] for x in rrows]) if rrows else set()
        ov_pub=m_pub & r_pub; ov_q=m_q & r_q; ov_doc=m_doc & r_doc
        print(f"[MedHallu {mcfg}  vs  RAGBench pubmedqa/{split}]")
        print(f"    by PubMed ID (both recovered via question-join): {len(ov_pub)}")
        print(f"    by normalised question text: {len(ov_q)}")
        print(f"    by context-document hash: {len(ov_doc)} shared document(s)")
        # collect example overlaps (prefer question, then doc)
        for q in list(ov_q)[:10]:
            examples.append((mcfg,split,"question",q[:90]))
        if len(examples)<10:
            for dh in list(ov_doc)[:10-len(examples)]:
                examples.append((mcfg,split,"doc-hash",dh))

print("\n=== up to 10 example overlaps ===")
if not examples: print("  (none found by any method)")
for e in examples[:10]: print("  ",e)
