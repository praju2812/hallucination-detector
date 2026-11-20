Produced by scripts/analysis/encoder_change_count.py at commit 966bc8d (removed with the legacy mode).
Encoding changes, legacy -> fit_first (max_length 1024, resp_max 384)
| source | split | rows | changed | % changed | now uncut | still cut | fit_first raises |
|---|---|---|---|---|---|---|---|
| halueval | train | 16200 | 0 | 0.0 | 0 | 0 | 0 |
| ragtruth | train | 8808 | 67 | 0.8 | 67 | 0 | 0 |
| halueval | val | 1800 | 0 | 0.0 | 0 | 0 | 0 |
| ragtruth | val | 984 | 15 | 1.5 | 15 | 0 | 0 |
| halueval | test | 2000 | 0 | 0.0 | 0 | 0 | 0 |
| ragtruth | test | 1800 | 6 | 0.3 | 6 | 0 | 0 |
| medhallu | train | 16262 | 1 | 0.0 | 1 | 0 | 0 |
| ragbench_covidqa | train | 1252 | 0 | 0.0 | 0 | 0 | 0 |
| ragbench_pubmedqa | train | 19600 | 130 | 0.7 | 126 | 4 | 0 |
| medhallu | val | 1738 | 0 | 0.0 | 0 | 0 | 0 |
| ragbench_covidqa | val | 88 | 0 | 0.0 | 0 | 0 | 0 |
| ragbench_pubmedqa | val | 526 | 3 | 0.6 | 2 | 1 | 0 |
| medhallu | test_full | 2000 | 0 | 0.0 | 0 | 0 | 0 |
| ragbench_covidqa | test_full | 246 | 0 | 0.0 | 0 | 0 | 0 |
| ragbench_pubmedqa | test_full | 2450 | 11 | 0.4 | 10 | 1 | 0 |
| medhallu | test_clean | 1998 | 0 | 0.0 | 0 | 0 | 0 |
| ragbench_covidqa | test_clean | 105 | 0 | 0.0 | 0 | 0 | 0 |
| ragbench_pubmedqa | test_clean | 522 | 3 | 0.6 | 3 | 0 | 0 |
