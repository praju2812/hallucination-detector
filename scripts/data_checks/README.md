# data_checks

- step1: Read-only measurement scripts written (Oct 2026) for verifying numbers.
Independently re-checked by scripts/progress_check.py on 22 Oct 2026 (32 PASS).
- step2: Run from inside this folder (they import common.py): `cd scripts/data_checks && python step5_overlap.py`.Needs the `datasets` package (not yet in requirements.txt).
- step3: MedHallu fields and PubMedQA knowledge check (matched by normalised question; no PubMed ID)
- step4: RAGBench PubMedQA/CovidQA split counts and hallucination rate from adherence_score
- step5: MedHallu x RAGBench overlap by PubMed ID, question text, context hash
- step6: effect of implicit_true / due_to_null spans on the RAGTruth response label
- step7: context and response token lengths (separately, no special tokens)

- followup1: combined truncation under the OLD input format (no query). Superseded by Phase 1 measurements.
- followup2: MedHallu Ground Truth vs PubMedQA long_answer
- followup3: CovidQA row count vs the RAGBench paper
- render_pdf_page.py: utility for reading PDF table cells; not a data check
