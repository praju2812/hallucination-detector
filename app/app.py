"""Phase 7: Gradio demo and API over src.inference.score (research prototype).

Local, one command from the repo root:
    python -m app.app --checkpoint ../Hallucination_detector/runs/run_seed43_20261121_000845/best.pt
Hugging Face Space: see app/README.md (weights, config_snapshot.json and metrics.json come from HD_HUB_REPO).

Which model is served (first match wins):
    --run-dir / HD_RUN_DIR (+ --checkpoint / HD_CHECKPOINT)   a local run folder (G, C or G+C)
    --hub-repo / HD_HUB_REPO                                   a Hub model repo holding the same three files
    default                                                    runs/run_seed43_20261121_000845 (G seed 43)
The disclaimer is generated from the loaded run's snapshot (src.inference.describe_model).
NLI (Phase 6 baseline) is loaded only on the first request that ticks it; HD_DISABLE_NLI=1 hides it.

API: endpoint "/score" (inputs question, context, response, use_nli; outputs probability, label, details).
    from gradio_client import Client
    Client("http://127.0.0.1:7860/").predict("question", "context", "response", False, api_name="/score")
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Callable

if __package__ in (None, ""):                         # `python app/app.py` (how a Space starts it)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src.inference import DEFAULT_RUN_DIR, describe_model, load_detector, load_nli, score

LABEL_TEXT = {1: "hallucinated (unsupported by the context)", 0: "faithful (supported by the context)"}
HUB_FILES = ("config_snapshot.json", "metrics.json", "best.pt")


def resolve_model_source(run_dir: str | None, checkpoint: str | None, hub_repo: str | None) -> dict:
    """CLI argument > environment variable > default. A local run folder wins over a Hub repo."""
    run_dir = run_dir or os.environ.get("HD_RUN_DIR")
    checkpoint = checkpoint or os.environ.get("HD_CHECKPOINT")
    hub_repo = hub_repo or os.environ.get("HD_HUB_REPO")
    if run_dir:
        return {"kind": "local", "run_dir": run_dir, "checkpoint": checkpoint}
    if hub_repo:
        return {"kind": "hub", "repo": hub_repo}
    return {"kind": "local", "run_dir": DEFAULT_RUN_DIR, "checkpoint": checkpoint}


def fetch_hub_run(repo: str, revision: str | None = None) -> str:
    """Download the run's three files from a Hub model repo into one folder; returns that folder (the run_dir)."""
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(repo, f, revision=revision) for f in HUB_FILES]
    return os.path.dirname(paths[0])


def cached_loader(load: Callable[[], object]) -> Callable[[], object]:
    """Call `load` once (thread-safe), then return the same object."""
    lock, box = threading.Lock(), {}

    def get():
        with lock:
            if "v" not in box:
                box["v"] = load()
        return box["v"]
    return get


def make_predict(detector, get_nli: Callable[[], object] | None, score_fn=score):
    """The request handler: (question, context, response, use_nli) -> (probability, label text, details dict)."""

    def predict(question: str, context: str, response: str, use_nli: bool = False):
        if use_nli and get_nli is None:
            raise gr.Error("NLI is disabled on this server.")
        try:
            nli = get_nli() if use_nli else None
            out = score_fn(question or "", context or "", response or "", detector, nli=nli)
        except ValueError as e:                       # e.g. empty context or response (fit_first refuses it)
            raise gr.Error(f"Cannot score this input: {e}") from e
        return out["prob"], LABEL_TEXT[out["label"]], out

    return predict


def build_demo(detector, description: str, get_nli: Callable[[], object] | None, score_fn=score) -> gr.Blocks:
    predict = make_predict(detector, get_nli, score_fn)
    with gr.Blocks(title="Hallucination detector (research prototype)") as demo:
        gr.Markdown("# Hallucination detector for RAG answers")
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(label="Question", lines=2)
                context = gr.Textbox(label="Retrieved context", lines=10)
                response = gr.Textbox(label="Response to check", lines=5)
                use_nli = gr.Checkbox(label="Also score with the NLI baseline (slow on CPU; first use loads it)",
                                      value=False, visible=get_nli is not None)
                button = gr.Button("Score", variant="primary")
            with gr.Column():
                prob = gr.Number(label="P(hallucinated)", precision=4)
                label = gr.Textbox(label="Label at the run's validation threshold")
                details = gr.JSON(label="Details")
        button.click(predict, inputs=[question, context, response, use_nli], outputs=[prob, label, details],
                     api_name="score")
    return demo


def api_names(demo: gr.Blocks) -> list[str]:
    fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
    return [f.api_name for f in fns if getattr(f, "api_name", None)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    ap.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    src = resolve_model_source(args.run_dir, args.checkpoint, args.hub_repo)
    t0 = time.perf_counter()
    if src["kind"] == "hub":
        run_dir, checkpoint = fetch_hub_run(src["repo"]), None
    else:
        run_dir, checkpoint = src["run_dir"], src["checkpoint"]
    detector = load_detector(run_dir, checkpoint=checkpoint, device=args.device)
    print(f"loaded {detector.model_id} ({detector.snapshot['experiment']}) on {detector.device} "
          f"in {time.perf_counter() - t0:.1f}s (start-up, includes any download)", flush=True)

    get_nli = None
    if os.environ.get("HD_DISABLE_NLI") != "1":
        get_nli = cached_loader(lambda: load_nli(device=detector.device))
    demo = build_demo(detector, describe_model(detector), get_nli)
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
