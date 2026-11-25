# Phase 7 demo: Gradio app and Hugging Face Space

Research prototype. The page text describing the model is generated from the served run's
`config_snapshot.json` (`src.inference.describe_model`), so swapping the run changes it.

## Run locally (repo root)

```bash
pip install -r requirements-app.txt
```

```bash
python -m app.app --checkpoint ../Hallucination_detector/runs/run_seed43_20261121_000845/best.pt
```

Open http://127.0.0.1:7860. Serve another run (e.g. a G+C run after RQ2) with `--run-dir runs/<run>`
(its folder needs `config_snapshot.json`, `metrics.json` and `best.pt`, or `--checkpoint`).

API call (Gradio's API page is linked at the bottom of the app):

```python
from gradio_client import Client
Client("http://127.0.0.1:7860/").predict("question", "context", "response", False, api_name="/score")
# -> (probability, label text, details dict with prob, label, threshold, model_id[, nli])
```

## Deploy to a Space (Prajakta does the uploads)

1. Model repo (public) with the run's three files:

```bash
huggingface-cli upload <user>/<model-repo> ../Hallucination_detector/runs/run_seed43_20261121_000845/best.pt best.pt
```

```bash
huggingface-cli upload <user>/<model-repo> runs/run_seed43_20261121_000845/config_snapshot.json config_snapshot.json
```

```bash
huggingface-cli upload <user>/<model-repo> runs/run_seed43_20261121_000845/metrics.json metrics.json
```

2. Space folder (Gradio SDK). Only the code the app imports, no data:

```bash
mkdir -p space/src space/scripts space/app space/results/nli/DeBERTa-v3-large-mnli-fever-anli-ling-wanli && cp src/__init__.py src/encoding.py src/inference.py src/metrics.py src/model.py src/nli.py src/rq1.py src/train.py src/data.py space/src/ && cp scripts/score_nli.py space/scripts/ && cp app/__init__.py app/app.py space/app/ && cp app/requirements-space.txt space/requirements.txt && cp results/nli/DeBERTa-v3-large-mnli-fever-anli-ling-wanli/metrics.json space/results/nli/DeBERTa-v3-large-mnli-fever-anli-ling-wanli/ && mkdir -p space/utils && cp utils/__init__.py utils/io.py space/utils/
```

   `space/README.md` must start with this front matter:

```yaml
---
title: Hallucination detector (research prototype)
sdk: gradio
sdk_version: 6.32.0
app_file: app/app.py
python_version: "3.12"
---
```

3. In the Space settings set the variable `HD_HUB_REPO=<user>/<model-repo>`. Optionally `HD_DISABLE_NLI=1`
   (DeBERTa-v3-large on the free CPU is slow and adds ~1.7 GB of RAM on first use).
4. Push `space/` to the Space repo. Start-up time (download + load) is printed in the Space logs.

Hardware: the free Space CPU is not the machine the latency numbers in `results/latency/` were measured on;
do not quote Space timings as RQ3 results.
