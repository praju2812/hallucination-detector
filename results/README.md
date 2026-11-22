# Results

- phase1_truncation.md: truncation at 1,024 on the general data (query field added).
- truncation_clinical.md: truncation at 1,024 on the clinical data.
- truncation_flags_v2.jsonl: per-row truncation flags (identical under fit_first).
- phase2_G.md: G (general detector, fit_first), 3 seeds, test metrics.
- rq1_G.md: RQ1, G zero-shot on the clinical test sets (written by report_rq1).
- analysis_2026-11-11.md, analysis_2026-11-11/: the 11 Nov analysis (window, overlap, length cues). A dated record; not modified by later runs.
- analysis_2026-11-21_G-fit-first/: A3 and C4 re-run on the fit_first G (see its README).
- analysis/encoder_change_count.md: rows whose encoding changed from legacy to fit_first.
- kaggle_logs/: training logs of the fit_first G runs.
- nli/ (after Phase 6): NLI baseline scores and report.
