"""Phase 7: Gradio app (app/app.py). Runs now: score() is replaced by a fake, so no template body is needed.

Skipped when gradio is not installed (it is only in requirements-app.txt, not the training requirements).

Run: python -m pytest tests/test_app.py
"""

import json

import pytest

gr = pytest.importorskip("gradio")

import app.app as appmod                      # noqa: E402

DESC = "Research prototype. Experiment G, trained on data/unified_v2/train.jsonl."


class FakeDetector:
    model_id = "run_seed43_x@abc1234"
    threshold = 0.35
    snapshot = {"experiment": "G"}


def fake_score(q, c, r, det, nli=None):
    out = {"prob": 0.8, "label": 1, "threshold": det.threshold, "model_id": det.model_id}
    if nli is not None:
        out["nli"] = {"prob": 0.6, "label": 1, "threshold": 0.3, "model_id": "nli", "n_pairs": 4}
    return out


def test_predict_outputs_prob_label_details():
    predict = appmod.make_predict(FakeDetector(), get_nli=None, score_fn=fake_score)
    prob, label, details = predict("q", "ctx", "resp", False)
    assert prob == pytest.approx(0.8)
    assert label == appmod.LABEL_TEXT[1]
    assert details["threshold"] == 0.35 and details["model_id"] == FakeDetector.model_id
    assert "nli" not in details


def test_predict_with_nli_only_when_ticked():
    loads = []

    def get_nli():
        loads.append(1)
        return "NLI"

    predict = appmod.make_predict(FakeDetector(), get_nli=get_nli, score_fn=fake_score)
    _, _, d0 = predict("q", "ctx", "resp", False)
    assert "nli" not in d0 and loads == []                      # NLI never loaded unless asked for
    _, _, d1 = predict("q", "ctx", "resp", True)
    assert d1["nli"]["n_pairs"] == 4 and loads == [1]


def test_nli_loader_caches():
    calls = []
    get = appmod.cached_loader(lambda: calls.append(1) or "NLI")
    assert get() == "NLI" and get() == "NLI" and len(calls) == 1


def test_predict_nli_requested_but_disabled():
    predict = appmod.make_predict(FakeDetector(), get_nli=None, score_fn=fake_score)
    with pytest.raises(gr.Error):
        predict("q", "ctx", "resp", True)


def test_predict_bad_input_is_a_gradio_error():
    def raising(*a, **k):
        raise ValueError("fit_first_lengths needs a non-empty context and response")

    predict = appmod.make_predict(FakeDetector(), get_nli=None, score_fn=raising)
    with pytest.raises(gr.Error):
        predict("q", "", "resp", False)


def test_demo_exposes_score_api_and_disclaimer():
    demo = appmod.build_demo(FakeDetector(), description=DESC, get_nli=None, score_fn=fake_score)
    assert "score" in appmod.api_names(demo)
    assert DESC in json.dumps(demo.config)                      # disclaimer comes from describe_model, passed in


def test_resolve_model_source_precedence(monkeypatch, tmp_path):
    for k in ("HD_RUN_DIR", "HD_CHECKPOINT", "HD_HUB_REPO"):
        monkeypatch.delenv(k, raising=False)
    # CLI beats env; env beats default
    monkeypatch.setenv("HD_RUN_DIR", "env_run")
    monkeypatch.setenv("HD_CHECKPOINT", "env.pt")
    assert appmod.resolve_model_source(run_dir="cli_run", checkpoint=None, hub_repo=None) == \
        {"kind": "local", "run_dir": "cli_run", "checkpoint": "env.pt"}
    monkeypatch.delenv("HD_RUN_DIR")
    monkeypatch.delenv("HD_CHECKPOINT")
    assert appmod.resolve_model_source(None, None, None) == \
        {"kind": "local", "run_dir": appmod.DEFAULT_RUN_DIR, "checkpoint": None}
    monkeypatch.setenv("HD_HUB_REPO", "someone/hd-g")
    assert appmod.resolve_model_source(None, None, None) == {"kind": "hub", "repo": "someone/hd-g"}
