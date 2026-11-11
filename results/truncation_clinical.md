
config configs/base.yaml: max_length 1024, resp_max 384; data data/clinical
input format: 'Question: {query} Context: {context}' [SEP] response

| split | source | n | ctx truncated | ctx 0 tokens | resp > resp_max | resp overflow cut | query tok p50 / p99 | query head tok p50 / p99 |
|---|---|---|---|---|---|---|---|---|
| train | medhallu | 16262 | 0.0% (1) | 0 | 1 | 0 | 21 / 40 | 25 / 44 |
| train | ragbench_covidqa | 1252 | 0.0% (0) | 0 | 0 | 0 | 12 / 29 | 16 / 33 |
| train | ragbench_pubmedqa | 19600 | 1.0% (193) | 0 | 151 | 0 | 18 / 39 | 22 / 42 |
| val | medhallu | 1738 | 0.0% (0) | 0 | 0 | 0 | 21 / 39 | 25 / 43 |
| val | ragbench_covidqa | 88 | 0.0% (0) | 0 | 0 | 0 | 11 / 25 | 15 / 29 |
| val | ragbench_pubmedqa | 526 | 1.3% (7) | 0 | 3 | 0 | 18 / 37 | 22 / 41 |
| test | medhallu | 1998 | 0.0% (0) | 0 | 0 | 0 | 17 / 35 | 21 / 39 |
| test | ragbench_covidqa | 105 | 0.0% (0) | 0 | 0 | 0 | 10 / 23 | 14 / 27 |
| test | ragbench_pubmedqa | 522 | 0.8% (4) | 0 | 4 | 0 | 18 / 41 | 22 / 44 |
| test_full | medhallu | 2000 | 0.0% (0) | 0 | 0 | 0 | 17 / 35 | 21 / 39 |
| test_full | ragbench_covidqa | 246 | 0.0% (0) | 0 | 0 | 0 | 11 / 28 | 15 / 32 |
| test_full | ragbench_pubmedqa | 2450 | 0.9% (21) | 0 | 12 | 0 | 18 / 39 | 22 / 43 |

[PASS] ctx 0 tokens + resp overflow cut = 0 (must be 0 at max_length 1024)
