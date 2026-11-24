"""Phase 7: single-request scoring (src/inference.py).

No model download: a word-level fake tokenizer, a tiny detector and a fake NLI model stand in.
Tests marked "needs body" call the templates score / score_nli, so they pass only once those bodies are written.
The snapshot, loading, threshold and describe_model tests run now.

Slow parity test (opt-in): HD_SLOW=1 and the seed 43 best.pt present (HD_CHECKPOINT or the main checkout path).

Run: python -m pytest tests/test_inference.py
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.score_nli import PairBuilder, n_pairs, prepare_row, score_prepared
from src.encoding import INPUT_FORMAT, Collator
from src.nli import HYP_CAP, MAX_LENGTH, STRIDE
from src.rq1 import check_g_snapshot
import src.inference as inf
from src.inference import (
    Detector,
    NLIScorer,
    check_servable_snapshot,
    describe_model,
    load_detector,
    nli_threshold_from_metrics,
    score,
    score_nli,
)

CLS, SEP, PAD = 1, 2, 0
G_KEYS = {"prob", "label", "threshold", "model_id"}
NLI_KEYS = G_KEYS | {"n_pairs"}


# ---- fakes ----------------------------------------------------------------------------------------

class FakeTokenizer:
    """Word-level: one id per lower-cased word (punctuation stripped). Pair layout [CLS] a [SEP] b [SEP]."""
    cls_token_id, sep_token_id, pad_token_id = CLS, SEP, PAD
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        return [self.vocab.setdefault(w.lower().strip(".,:;!?*-"), len(self.vocab) + 10)
                for w in text.split() if w.strip(".,:;!?*-")]

    def __call__(self, text, text_pair=None, add_special_tokens=True):
        a = self._ids(text)
        if not add_special_tokens:
            ids = a
        elif text_pair is None:
            ids = [CLS] + a + [SEP]
        else:
            ids = [CLS] + a + [SEP] + self._ids(text_pair) + [SEP]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


class TinyDetector(nn.Module):
    """Same call signature as HallucinationDetector: forward(input_ids, attention_mask) -> logits [batch].
    Remembers the last inputs so tests can check what score() fed it."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(2000, 4)
        self.head = nn.Linear(4, 1)
        self.drop = nn.Dropout(0.5)
        self.last = None

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.last = {"input_ids": input_ids.detach().cpu(), "attention_mask": attention_mask.detach().cpu()}
        x = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        pooled = x.sum(1) / attention_mask.sum(1, keepdim=True)
        return self.head(self.drop(pooled)).squeeze(-1)


class FakeNLI:
    """P(entailment) rises with the share of hypothesis words found in the premise window (entailment = label 1)."""
    config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

    def __call__(self, input_ids, attention_mask, token_type_ids=None):
        rows = []
        for ids, m in zip(input_ids.tolist(), attention_mask.tolist()):
            ids = [t for t, k in zip(ids, m) if k]
            p1 = ids.index(SEP)
            p2 = ids.index(SEP, p1 + 1)
            premise, hyp = set(ids[1:p1]), ids[p1 + 1:p2]
            share = sum(t in premise for t in hyp) / len(hyp)
            rows.append([0.0, 8.0 * share - 4.0, 0.0])
        return SimpleNamespace(logits=torch.tensor(rows))


def good_snapshot(experiment="G", train_path="data/unified_v2/train.jsonl", seed=43):
    train = {"seed": seed, "select_on": "pr_auc", "epochs": 2}
    if experiment != "G":
        train["select_sources"] = ["ragbench_pubmedqa", "ragbench_covidqa"]
    return {
        "config": {
            "experiment": experiment,
            "model": {"name": "answerdotai/ModernBERT-base", "max_length": 1024, "dropout": 0.1,
                      "resp_max": 384, "truncation": "fit_first"},
            "data": {"train_path": train_path, "val_path": "data/unified_v2/val.jsonl",
                     "test_path": "data/unified_v2/test.jsonl"},
            "train": train,
        },
        "seed": seed,
        "git_sha": "abc1234",
        "git_dirty": False,
        "experiment": experiment,
        "input_format": INPUT_FORMAT,
        "truncation": "fit_first",
    }


def write_run(tmp_path, snapshot=None, threshold=0.35, state_dict=None, name="run_seed43_x", weights=True):
    run = tmp_path / name
    run.mkdir()
    (run / "config_snapshot.json").write_text(json.dumps(snapshot or good_snapshot()))
    (run / "metrics.json").write_text(json.dumps({"threshold": threshold, "seed": 43}))
    if weights:
        torch.save(state_dict if state_dict is not None else TinyDetector().state_dict(), run / "best.pt")
    return run


def load_tiny(run, **kw):
    return load_detector(str(run), device="cpu", build_model=lambda cfg: TinyDetector(),
                         tokenizer=FakeTokenizer(), **kw)


Q = "where did the cat sit"
C = "the cat sat on the mat in the kitchen"
R = "the cat sat on the mat"


def reference_prob(det: Detector, q=Q, c=C, r=R) -> float:
    ids = det.collator.encode_with_response_budget(q, c, r)
    with torch.inference_mode():
        logit = det.model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))
    return float(1.0 / (1.0 + np.exp(-logit.item())))


# ---- check_servable_snapshot (runs now) -------------------------------------------------------------

@pytest.mark.parametrize("experiment", ["G", "C", "G+C"])
def test_servable_accepts_every_experiment(experiment):
    check_servable_snapshot(good_snapshot(experiment))


def test_servable_agrees_with_rq1_check_for_g():
    snap = good_snapshot("G")
    check_servable_snapshot(snap)
    check_g_snapshot(snap)                              # both accept the same G run


@pytest.mark.parametrize("edit", [
    lambda s: s.update(experiment="X"),
    lambda s: s.pop("experiment"),
    lambda s: s.update(input_format="Context: {context}"),
    lambda s: s.pop("input_format"),
    lambda s: s["config"]["model"].update(max_length=256),
    lambda s: s["config"]["model"].update(resp_max=256),
    lambda s: s["config"]["model"].update(truncation="legacy"),
    lambda s: s["config"]["model"].pop("truncation"),
])
def test_servable_rejects_wrong_runs(edit):
    snap = good_snapshot()
    edit(snap)
    with pytest.raises(ValueError):                     # ValueError, never KeyError
        check_servable_snapshot(snap)


# ---- load_detector (runs now) -------------------------------------------------------------------

def test_load_detector_restores_weights_threshold_and_encoding(tmp_path):
    src = TinyDetector()
    with torch.no_grad():
        src.head.weight.fill_(0.25)
    run = write_run(tmp_path, threshold=0.35, state_dict=src.state_dict())
    det = load_tiny(run)
    assert isinstance(det, Detector)
    assert det.threshold == pytest.approx(0.35)
    assert not det.model.training                       # eval mode: dropout off
    assert torch.allclose(det.model.head.weight, torch.full((1, 4), 0.25))
    assert isinstance(det.collator, Collator)
    assert (det.collator.max_length, det.collator.resp_max, det.collator.truncation) == (1024, 384, "fit_first")
    assert "run_seed43_x" in det.model_id and "abc1234" in det.model_id
    assert det.snapshot["experiment"] == "G"


@pytest.mark.parametrize("experiment", ["C", "G+C"])
def test_load_detector_serves_c_and_gc(tmp_path, experiment):
    run = write_run(tmp_path, snapshot=good_snapshot(experiment), threshold=0.4)
    det = load_tiny(run)
    assert det.snapshot["experiment"] == experiment
    assert det.threshold == pytest.approx(0.4)


def test_load_detector_checkpoint_override(tmp_path):
    run = write_run(tmp_path, weights=False)              # run folder has snapshot + metrics only (as on the Hub)
    other = TinyDetector()
    with torch.no_grad():
        other.head.bias.fill_(1.5)
    ckpt = tmp_path / "elsewhere.pt"
    torch.save(other.state_dict(), ckpt)
    det = load_tiny(run, checkpoint=str(ckpt))
    assert det.model.head.bias.item() == pytest.approx(1.5)


def test_load_detector_missing_files(tmp_path):
    run = write_run(tmp_path, weights=False)
    with pytest.raises(FileNotFoundError):
        load_tiny(run)                                   # no best.pt and no override
    run2 = write_run(tmp_path, name="run_b")
    os.remove(run2 / "metrics.json")
    with pytest.raises(FileNotFoundError):
        load_tiny(run2)


def test_load_detector_rejects_bad_snapshot_before_weights(tmp_path):
    snap = good_snapshot()
    snap["config"]["model"]["max_length"] = 256
    run = write_run(tmp_path, snapshot=snap, state_dict={"garbage": torch.zeros(1)})
    with pytest.raises(ValueError):                      # snapshot check first, so not a state-dict error
        load_tiny(run)


def test_load_detector_strict_keys(tmp_path):
    sd = TinyDetector().state_dict()
    sd["extra.weight"] = torch.zeros(1)
    run = write_run(tmp_path, state_dict=sd)
    with pytest.raises(RuntimeError):
        load_tiny(run)


def test_load_detector_rejects_missing_threshold(tmp_path):
    run = write_run(tmp_path)
    (run / "metrics.json").write_text(json.dumps({"seed": 43}))
    with pytest.raises(ValueError):
        load_tiny(run)


# ---- describe_model (runs now) ------------------------------------------------------------------

def test_describe_model_from_snapshot_g(tmp_path):
    det = load_tiny(write_run(tmp_path))
    text = describe_model(det)
    for needle in ["G", "data/unified_v2/train.jsonl", "43", "abc1234", "ragtruth", "0.35", "not for clinical use"]:
        assert needle in text, needle
    assert "research prototype" in text.lower()


def test_describe_model_lists_every_train_file_for_gc(tmp_path):
    snap = good_snapshot("G+C", train_path=["data/unified_v2/train.jsonl", "data/clinical/train.jsonl.gz"])
    det = load_tiny(write_run(tmp_path, snapshot=snap, threshold=0.25))
    text = describe_model(det)
    for needle in ["G+C", "data/unified_v2/train.jsonl", "data/clinical/train.jsonl.gz",
                   "ragbench_pubmedqa", "ragbench_covidqa", "0.25"]:
        assert needle in text, needle


def test_describe_model_changes_with_snapshot(tmp_path):
    a = describe_model(load_tiny(write_run(tmp_path, name="run_a")))
    b = describe_model(load_tiny(write_run(tmp_path, snapshot=good_snapshot("C", "data/clinical/train.jsonl.gz"),
                                            name="run_b")))
    assert a != b and "data/clinical/train.jsonl.gz" in b and "data/clinical/train.jsonl.gz" not in a


# ---- NLI threshold (runs now) -------------------------------------------------------------------

def test_nli_threshold_read_from_ragtruth_val(tmp_path):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps({"thresholds": {"ragtruth_val": {"threshold": 0.3, "tuned_on": ["ragtruth"]},
                                            "clinical_val": {"medhallu": {"threshold": 0.5}}}}))
    assert nli_threshold_from_metrics(str(p)) == pytest.approx(0.3)
    p.write_text(json.dumps({"thresholds": {}}))
    with pytest.raises(ValueError):
        nli_threshold_from_metrics(str(p))


def test_nli_threshold_of_the_committed_baseline():
    assert nli_threshold_from_metrics(inf.NLI_METRICS_PATH) == pytest.approx(0.3)


# ---- score (needs body) -------------------------------------------------------------------------

@pytest.fixture
def det(tmp_path):
    return load_tiny(write_run(tmp_path, threshold=0.35))


def test_score_result_shape(det):
    out = score(Q, C, R, det)
    assert set(out) == G_KEYS                               # no "nli" key when NLI is off
    assert isinstance(out["prob"], float) and 0.0 <= out["prob"] <= 1.0
    assert out["label"] in (0, 1) and isinstance(out["label"], int)
    assert out["threshold"] == pytest.approx(0.35)
    assert out["model_id"] == det.model_id


def test_score_matches_training_encoding_and_sigmoid(det):
    out = score(Q, C, R, det)
    ids = det.collator.encode_with_response_budget(Q, C, R)
    assert det.model.last["input_ids"].tolist() == [ids]   # exactly the training encoding, batch of one
    assert det.model.last["attention_mask"].tolist() == [[1] * len(ids)]
    assert out["prob"] == pytest.approx(reference_prob(det), abs=1e-6)


def test_score_is_deterministic(det):
    assert score(Q, C, R, det)["prob"] == score(Q, C, R, det)["prob"]


def test_score_label_uses_ge_threshold(det):
    p = score(Q, C, R, det)["prob"]
    det.threshold = p                                       # prob == threshold -> hallucinated (training uses >=)
    assert score(Q, C, R, det)["label"] == 1
    det.threshold = min(1.0, p + 1e-6)
    assert score(Q, C, R, det)["label"] == 0


@pytest.mark.parametrize("q,c,r", [(Q, C, ""), (Q, "", R), (Q, C, "   ")])
def test_score_rejects_empty_parts(det, q, c, r):
    with pytest.raises(ValueError):
        score(q, c, r, det)


def test_score_adds_nli_when_given(det, monkeypatch):
    fake = {"prob": 0.9, "label": 1, "threshold": 0.3, "model_id": "fake-nli", "n_pairs": 2}
    calls = []
    monkeypatch.setattr(inf, "score_nli", lambda q, c, r, nli: calls.append((q, c, r, nli)) or fake)
    out = score(Q, C, R, det, nli="NLI")
    assert out["nli"] == fake and calls == [(Q, C, R, "NLI")]
    assert set(out) == G_KEYS | {"nli"}


# ---- score_nli (needs body) ---------------------------------------------------------------------

@pytest.fixture
def nli():
    tok = FakeTokenizer()
    return NLIScorer(model=FakeNLI(), builder=PairBuilder(tok), ent_idx=1, threshold=0.3,
                     model_id="fake-nli@rev", device="cpu")


def test_score_nli_result_shape(nli):
    out = score_nli(Q, C, R, nli)
    assert set(out) == NLI_KEYS
    assert isinstance(out["prob"], float) and 0.0 <= out["prob"] <= 1.0
    assert out["label"] in (0, 1) and isinstance(out["label"], int)
    assert out["threshold"] == pytest.approx(0.3) and out["model_id"] == "fake-nli@rev"
    assert isinstance(out["n_pairs"], int) and out["n_pairs"] >= 1


def test_score_nli_matches_phase6_pipeline(nli):
    r = "The cat sat on the mat. It was very happy about the dog outside."
    out = score_nli(Q, C, r, nli)
    row = {"query": Q, "context": C, "answer": r, "task": None}
    p = prepare_row(row, nli.builder, include_question=False, stride=STRIDE, max_length=MAX_LENGTH, hyp_cap=HYP_CAP)
    ref, _ = score_prepared([p], nli.model, nli.builder, nli.ent_idx, batch_size=16, device="cpu")
    assert out["prob"] == pytest.approx(ref[0]["prob"], abs=1e-9)
    assert out["n_pairs"] == n_pairs(p)


def test_score_nli_direction(nli):
    supported = score_nli(Q, C, "The cat sat on the mat.", nli)
    unsupported = score_nli(Q, C, "A purple elephant flew over London yesterday.", nli)
    assert supported["prob"] < unsupported["prob"]
    assert supported["label"] == 0 and unsupported["label"] == 1


def test_score_nli_rejects_empty_response(nli):
    with pytest.raises(ValueError):
        score_nli(Q, C, "  ", nli)


# ---- slow parity with Phase 2 predictions (opt-in) -----------------------------------------------

SEED43 = "runs/run_seed43_20261121_000845"
MAIN_CKPT = os.path.join("..", "Hallucination_detector", SEED43, "best.pt")
CKPT = os.environ.get("HD_CHECKPOINT", MAIN_CKPT)


@pytest.mark.skipif(not (os.environ.get("HD_SLOW") and os.path.exists(CKPT)),
                    reason="opt-in: HD_SLOW=1 and the seed 43 best.pt (HD_CHECKPOINT)")
def test_seed43_serving_reproduces_phase2_test_probs():
    from scripts.score_rq1 import PARITY_MAX_ABS          # same tolerance as the RQ1 parity check
    from utils.io import read_jsonl
    det_real = load_detector(SEED43, checkpoint=CKPT, device="cpu")
    rows = read_jsonl("data/unified_v2/test.jsonl")
    preds = [p for p in read_jsonl(os.path.join(SEED43, "predictions.jsonl")) if p["split"] == "test"]
    for i in [0, 1, 900, 1799, 1800]:                      # summary, qa and halueval rows
        r = rows[i]
        assert preds[i]["idx"] == i and preds[i]["label"] == r["label"]
        out = score(r["query"], r["context"], r["answer"], det_real)
        assert abs(out["prob"] - preds[i]["prob"]) <= PARITY_MAX_ABS, (i, out["prob"], preds[i]["prob"])
