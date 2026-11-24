"""Phase 7 (RQ3): score ONE (question, context, response) request with a trained detector, optionally with NLI.

Used by the Gradio app (app/app.py) and the latency script (scripts/measure_latency.py).
The served detector is any G, C or G+C run (default: G seed 43, the best RAGTruth-validation PR-AUC of the three
G seeds). Changing the served model = pointing `run_dir` at another run folder; nothing else changes.

Serving must equal training: same Collator, same fit_first encoding (src/encoding.py), same sigmoid
(src/metrics.py), same threshold rule (prob >= threshold, src/metrics.thresholded_metrics), and the threshold the
run tuned on its own validation selection rows (its metrics.json), never one tuned on test.

Vocabulary:
    Detector    a loaded run: model (eval mode), collator, threshold, model_id, device, snapshot
    NLIScorer   the Phase 6 NLI baseline loaded once: model, PairBuilder, entailment index, threshold
                (RAGTruth-validation threshold of the Phase 6 run), model_id (model@revision), device

Spec: tests/test_inference.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

from src.encoding import INPUT_FORMAT, Collator, build_tokenizer, truncation_mode
from src.metrics import probs_from_logits   # reuse: the sigmoid training used
from src.train import EXPERIMENTS, get_device, select_sources_from_cfg

DEFAULT_RUN_DIR = "runs/run_seed43_20261121_000845"
NLI_METRICS_PATH = "results/nli/DeBERTa-v3-large-mnli-fever-anli-ling-wanli/metrics.json"

# What a servable run must have been trained with (rule 7 and the fit_first decision of 20 Nov)
SERVE_MAX_LENGTH = 1024
SERVE_RESP_MAX = 384
SERVE_TRUNCATION = "fit_first"

DISCLAIMER = ("Research prototype from an MSc dissertation. Not a medical device and not for clinical use: "
              "a score is a model estimate and can be wrong.")


@dataclass
class Detector:
    model: Any
    collator: Collator
    threshold: float
    model_id: str
    device: str
    snapshot: dict


@dataclass
class NLIScorer:
    model: Any
    builder: Any          # scripts.score_nli.PairBuilder
    ent_idx: int
    threshold: float
    model_id: str
    device: str


# ---- loading (plumbing) -------------------------------------------------------------------------

def check_servable_snapshot(snapshot: dict) -> None:
    """Raise ValueError unless `snapshot` (a run's config_snapshot.json) is a G, C or G+C run in today's format.

    Same idea as src.rq1.check_g_snapshot, which accepts G only; for a G run both accept the same snapshot.
    """
    try:
        experiment = snapshot["experiment"]
        input_format = snapshot["input_format"]
        model_cfg = snapshot["config"]["model"]
        max_length, resp_max = model_cfg["max_length"], model_cfg["resp_max"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"config_snapshot.json is missing {e}; not a servable run") from e
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {experiment!r}")
    if input_format != INPUT_FORMAT:
        raise ValueError(f"input_format must be {INPUT_FORMAT!r}, got {input_format!r}")
    if max_length != SERVE_MAX_LENGTH:
        raise ValueError(f"max_length must be {SERVE_MAX_LENGTH}, got {max_length}")
    if resp_max != SERVE_RESP_MAX:
        raise ValueError(f"resp_max must be {SERVE_RESP_MAX}, got {resp_max}")
    if truncation_mode(model_cfg) != SERVE_TRUNCATION:      # raises ValueError if missing or unknown
        raise ValueError(f"truncation must be {SERVE_TRUNCATION!r}")


def read_run_threshold(run_dir: str) -> float:
    """The run's validation-tuned threshold: metrics.json["threshold"]."""
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing metrics.json in {run_dir} (the threshold comes from it)")
    with open(path) as f:
        metrics = json.load(f)
    thr = metrics.get("threshold")
    if not isinstance(thr, (int, float)) or not 0.0 <= thr <= 1.0:
        raise ValueError(f"{path}: 'threshold' missing or not in [0, 1], got {thr!r}")
    return float(thr)


def _default_build_model(cfg: dict):
    from src.rq1 import _default_build_model as build    # the same builder RQ1 scoring uses
    return build(cfg)


def load_detector(run_dir: str = DEFAULT_RUN_DIR, checkpoint: str | None = None, device: str | None = None,
                  build_model: Callable[[dict], Any] | None = None, tokenizer=None) -> Detector:
    """Load a run for serving.

    run_dir     folder with config_snapshot.json and metrics.json (and best.pt unless `checkpoint` is given)
    checkpoint  path to the weights, e.g. a file downloaded from the Hub; default <run_dir>/best.pt
    device      default cuda > mps > cpu
    build_model / tokenizer: injected by tests so nothing is downloaded; default = the run's own model name.
    The snapshot is checked BEFORE the weights are read; state-dict keys must match exactly.
    """
    snap_path = os.path.join(run_dir, "config_snapshot.json")
    if not os.path.exists(snap_path):
        raise FileNotFoundError(f"missing config_snapshot.json in {run_dir}")
    with open(snap_path) as f:
        snapshot = json.load(f)
    check_servable_snapshot(snapshot)
    threshold = read_run_threshold(run_dir)

    ckpt = checkpoint or os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"missing weights {ckpt}; pass checkpoint= (or --checkpoint / HD_CHECKPOINT)")
    cfg = snapshot["config"]
    device = device or get_device()
    model = (build_model or _default_build_model)(cfg)
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    tok = tokenizer if tokenizer is not None else build_tokenizer(cfg["model"]["name"])
    collator = Collator(tok, cfg["model"]["max_length"], cfg["model"]["resp_max"], truncation_mode(cfg["model"]))
    model_id = f"{os.path.basename(os.path.normpath(run_dir))}@{snapshot.get('git_sha', 'unknown')}"
    return Detector(model=model, collator=collator, threshold=threshold, model_id=model_id, device=device,
                    snapshot=snapshot)


def describe_model(detector: Detector) -> str:
    """Disclaimer + what is loaded, built from the run's snapshot (Markdown). No fixed model description."""
    snap = detector.snapshot
    cfg = snap["config"]
    train_paths = cfg["data"]["train_path"]
    train_paths = train_paths if isinstance(train_paths, list) else [train_paths]
    lines = [
        f"**{DISCLAIMER}**",
        "",
        f"- Model: experiment **{snap['experiment']}**, backbone `{cfg['model']['name']}`, "
        f"seed {snap.get('seed')}, git `{snap.get('git_sha')}`, run `{detector.model_id}`",
        f"- Training data: {', '.join(f'`{p}`' for p in train_paths)}",
    ]
    if cfg.get("train", {}).get("train_sources"):
        lines.append(f"- Training sources kept: {', '.join(cfg['train']['train_sources'])}")
    lines += [
        f"- Threshold {detector.threshold:.2f}, tuned on validation rows of: "
        f"{', '.join(select_sources_from_cfg(cfg))}",
        f"- Input: `{snap['input_format']}` + response, {cfg['model']['max_length']} tokens "
        f"({snap.get('truncation', cfg['model'].get('truncation'))} truncation). Label 1 = hallucinated.",
    ]
    return "\n".join(lines)


def nli_threshold_from_metrics(path: str = NLI_METRICS_PATH) -> float:
    """The Phase 6 NLI threshold tuned on RAGTruth validation (thresholds.ragtruth_val.threshold)."""
    with open(path) as f:
        metrics = json.load(f)
    try:
        thr = metrics["thresholds"]["ragtruth_val"]["threshold"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path}: no thresholds.ragtruth_val.threshold") from e
    return float(thr)


def load_nli(device: str | None = None, model_name: str | None = None, revision: str | None = None,
             metrics_path: str = NLI_METRICS_PATH) -> NLIScorer:
    """Load the Phase 6 NLI model at its pinned revision and verify the pair layout (PairBuilder.verify)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.score_nli import MODEL_REVISION, PairBuilder
    from src.nli import MODEL_ID, entailment_index

    model_name, revision = model_name or MODEL_ID, revision or MODEL_REVISION
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision, dtype=torch.float32)
    model.to(device).eval()
    builder = PairBuilder(tok)
    builder.verify()
    return NLIScorer(model=model, builder=builder, ent_idx=entailment_index(model.config.id2label),
                     threshold=nli_threshold_from_metrics(metrics_path),
                     model_id=f"{model_name}@{revision[:7]}", device=device)


# ---- scoring (TEMPLATES: Prajakta writes the bodies) --------------------------------------------

def score(question: str, context: str, response: str, detector: Detector, nli: NLIScorer | None = None) -> dict:
    """Score one request with the detector (and NLI when `nli` is given).

    Inputs: the three texts; a loaded Detector; optionally a loaded NLIScorer.
    Output: {"prob": float in [0, 1], "label": int 0/1, "threshold": float, "model_id": str}
            plus "nli": score_nli(...) result ONLY when nli is not None.
    Invariants (tests/test_inference.py):
        - the model receives exactly detector.collator.encode_with_response_budget(question, context, response)
          as a batch of one, with an all-ones attention mask, on detector.device.
        - prob = the same sigmoid as training; label 1 = hallucinated; same input -> same prob.
        - label follows the rule training used to compute F1 at the threshold.
    """
    # TODO 1: encode the triple with the detector's collator (the training encoding; do not re-tokenise yourself).
    # TODO 2: build input_ids and attention_mask tensors of shape [1, L] on detector.device.
    # TODO 3: one forward pass with gradients off; turn the logit into a probability with the training helper.
    # TODO 4: label from prob and detector.threshold.
    # TODO 5: build the result dict with plain Python types; add the "nli" entry when nli is given (call score_nli).
    # CHECK: empty / whitespace-only context or response must raise ValueError. Which existing function already does?
    # CHECK: `>=` or `>`? Look at src/metrics.thresholded_metrics.
    # CHECK: the logit comes back as a tensor on the device; the dict must hold a float, not a tensor or numpy type.
    raise NotImplementedError


def score_nli(question: str, context: str, response: str, nli: NLIScorer) -> dict:
    """Score one request with the Phase 6 NLI baseline, exactly as scripts/score_nli.py scores a test row.

    Inputs: the three texts; a loaded NLIScorer.
    Output: {"prob": float, "label": int 0/1, "threshold": float, "model_id": str, "n_pairs": int}
            prob = 1 - row support (higher = more likely hallucinated, same direction as the detector).
    Invariants (tests/test_inference.py):
        - prob equals what scripts/score_nli.score_prepared gives for the same row with the Phase 6 settings
          (response-only hypothesis, src/nli STRIDE, MAX_LENGTH, HYP_CAP).
        - n_pairs = number of (window, sentence) pairs run through the model for this request.
    Reuse prepare_row, n_pairs and score_prepared from scripts/score_nli.py: import them, do not copy them.
    """
    # TODO 1: build the row dict prepare_row expects (which keys does it read?).
    # TODO 2: prepare it with the Phase 6 settings (hypothesis format "response" -> which include_question?).
    # TODO 3: score the one prepared row; take its prob.
    # TODO 4: label against nli.threshold with the same rule as the detector; build the result dict.
    # CHECK: the question is not part of the hypothesis in the Phase 6 run; does the row still need a task key?
    # CHECK: a whitespace-only response must raise ValueError (does split_response_sentences already?).
    raise NotImplementedError
