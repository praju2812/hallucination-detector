
config configs/base.yaml: max_length 1024, resp_max 384; data data/unified_v2
input format: 'Question: {query} Context: {context}' [SEP] response

| split | source | n | ctx truncated | ctx 0 tokens | resp > resp_max | resp overflow cut | query tok p50 / p99 | query head tok p50 / p99 |
|---|---|---|---|---|---|---|---|---|
| train | halueval | 16200 | 0.0% (0) | 0 | 0 | 0 | 20 / 77 | 24 / 81 |
| train | ragtruth | 8808 | 14.5% (1277) | 0 | 83 | 0 | 6 / 15 | 10 / 19 |
| val | halueval | 1800 | 0.0% (0) | 0 | 0 | 0 | 20 / 75 | 24 / 78 |
| val | ragtruth | 984 | 10.5% (103) | 0 | 17 | 0 | 6 / 16 | 10 / 20 |
| test | halueval | 2000 | 0.0% (0) | 0 | 0 | 0 | 20 / 81 | 24 / 85 |
| test | ragtruth | 1800 | 14.9% (269) | 0 | 8 | 0 | 6 / 16 | 10 / 19 |

[PASS] ctx 0 tokens + resp overflow cut = 0 (must be 0 at max_length 1024)
