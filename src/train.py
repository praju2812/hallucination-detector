"""Training entry point.

This is the main training script. It loads the configuration, sets up the model and
data, and runs the training loop.

Run:  python -m src.train --config configs/base.yaml
      python -m src.train --config configs/base.yaml --seed 43 --out-dir /kaggle/working/runs   # Kaggle
"""

from __future__ import annotations
import argparse
from datetime import datetime
import os
import random
import json
import subprocess
import time

import numpy as np
import torch
import transformers
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import HallucinationDataset
from src.encoding import INPUT_FORMAT, Collator, build_tokenizer
from src.model import HallucinationDetector, build_loss
from src.metrics import per_source_breakdown, probs_from_logits, ranking_metrics, best_threshold, thresholded_metrics
from utils.io import write_jsonl


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_config(path: str) -> dict:
    """Load the base.yaml configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


# run_dir is the directory where the model checkpoints and logs will be saved.
# It is created based on the current timestamp and the random seed.
def make_run_dir(cfg: dict) -> str:
    seed = cfg["train"]["seed"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["train"]["out_dir"], f"run_seed{seed}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_git_sha() -> str:
    try:
        # subprocess.check_output runs the git command to get the current commit hash (SHA) of the repository.
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_git_dirty() -> bool | None:
    """True if TRACKED files differ from HEAD (untracked files, e.g. run outputs, are ignored).
    None if git is not available. A dirty run was not produced by the commit in git_sha alone."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        )
        return bool(out.strip())
    except Exception:
        return None


def get_device_name() -> str:
    """GPU model (e.g. 'Tesla T4') so runs on different accelerators can be told apart."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return get_device()   # "mps" or "cpu"


# Kaggle sets KAGGLE_KERNEL_RUN_TYPE ("Interactive" or "Batch") in every notebook session.
KAGGLE_ENV_VAR = "KAGGLE_KERNEL_RUN_TYPE"
KAGGLE_WORKING = "/kaggle/working"


def is_kaggle() -> bool:
    return KAGGLE_ENV_VAR in os.environ


def check_out_dir(out_dir: str, on_kaggle: bool) -> None:
    """On Kaggle, out_dir must be an absolute path under /kaggle/working/ (the only persisted area).
    A relative path once silently lost a checkpoint. Off Kaggle, anything is allowed."""
    if not on_kaggle:
        return
    if not os.path.isabs(out_dir):
        raise ValueError(f"on Kaggle out_dir must be absolute under {KAGGLE_WORKING}/, got relative {out_dir!r}; "
                         f"pass --out-dir {KAGGLE_WORKING}/runs")
    norm = os.path.normpath(out_dir)
    if norm != KAGGLE_WORKING and not norm.startswith(KAGGLE_WORKING + os.sep):
        raise ValueError(f"on Kaggle out_dir must be under {KAGGLE_WORKING}/, got {out_dir!r}")


EXPERIMENTS = ("G", "C", "G+C")


def build_run_snapshot(cfg: dict) -> dict:
    """Everything needed to say which data, format and library versions produced a run.
    Written to config_snapshot.json. `experiment` comes from the config's top-level `experiment` key."""
    experiment = cfg.get("experiment")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"config 'experiment' must be one of {EXPERIMENTS}, got {experiment!r}")
    return {
        "config": cfg,                      # includes data paths
        "seed": cfg["train"]["seed"],
        "git_sha": get_git_sha(),
        "git_dirty": get_git_dirty(),
        "experiment": experiment,
        "input_format": INPUT_FORMAT,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_name": get_device_name(),
    }


def prediction_records(split: str, rows: list[dict], labels, probs, cfg_data: dict) -> list[dict]:
    """One record per row for predictions.jsonl: split, idx, source, task, group_id, label, prob.

    `rows` are the loader's dataset rows (val/test loaders are unshuffled and unfiltered, so evaluate()
    returns them in file order). idx = row position in the split file. Raises ValueError if the lengths
    differ or any label disagrees with the file, i.e. if the order assumption is broken."""
    if not (len(rows) == len(labels) == len(probs)):
        raise ValueError(f"{split}: {len(rows)} rows but {len(labels)} labels / {len(probs)} probs")
    label_col, source_col = cfg_data["label_col"], cfg_data.get("dataset_col")
    out = []
    for i, (row, y, p) in enumerate(zip(rows, labels, probs)):
        y_file = int(row[label_col])
        if y_file != int(round(float(y))):
            raise ValueError(f"{split} row {i}: file label {y_file} != evaluated label {y}; row order broken")
        out.append({
            "split": split,
            "idx": i,
            "source": row.get(source_col, "unknown") if source_col else "unknown",
            "task": row.get("task"),
            "group_id": row.get("group_id"),
            "label": y_file,
            "prob": float(p),
        })
    return out


def make_loader(path: str, cfg: dict, tokenizer, shuffle: bool,
                generator: torch.Generator | None = None, sources: list | None = None) -> DataLoader:
    ds = HallucinationDataset(path, cfg["data"], sources=sources)
    collate = Collator(tokenizer, cfg["model"]["max_length"], cfg["model"]["resp_max"])

    #  The DataLoader is a PyTorch utility that provides an iterable over the dataset. 
    # It handles batching, shuffling, and loading data in parallel using multiprocessing workers.

    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
        num_workers=2,#two subprocesses will be used to load the data in parallel
        pin_memory=True,#data loader will copy Tensors into CUDA pinned memory before returning
    )

def build_optimizer_and_scheduler(model, cfg: dict, steps_per_epoch: int):
    """
    Build the optimizer and learning rate scheduler for training.
    """

    # AdamW is a variant of the Adam optimizer that includes weight decay, which helps prevent overfitting by penalizing large weights.
    # It takes the model parameters, learning rate, and weight decay as inputs.
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    total_steps = steps_per_epoch * cfg["train"]["epochs"]

    # schedule learning rate with warmup: the learning rate starts at 0, 
    # increases linearly to the specified learning rate over a warmup period
    # and then decreases linearly for the rest of training.
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(cfg["train"]["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )
    return optim, scheduler
 

# torch.no_grad disables gradient calculation since we don't need to compute gradients when we're just making predictions.
@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model over a loader and return (labels, probs, sources)."""
    model.eval()
    all_labels, all_probs, all_sources = [], [], []

    # For each batch , the dataloader calls __getitem__ on the dataset, which returns a dict with keys 'context', 'response', 'label', and 'source'.
    # The collator then tokenizes the context and response, and returns a dict with keys 'enc' (the tokenized inputs) and 'sources' (the source names).
    for batch in loader:

        # the dict other than labels is passed to the device.
        # v.to(device) moves the tensor to the specified device (CPU or GPU).
        enc = {k: v.to(device) for k, v in batch["enc"].items() if k != "labels"}
        labels = batch["enc"]["labels"]

        # logits returns the raw output of the model (before applying softmax or sigmoid).
        logits = model(**enc)   

        # probs converts the logits to probabilities using the sigmoid function.
        probs  = probs_from_logits(logits.cpu().numpy())

        # extend the three lists with labels, probs, and batch["sources"].
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_sources.extend(batch["sources"])    
    return np.array(all_labels), np.array(all_probs), all_sources

def train_one_epoch(model, loader, loss_fn, optim, scheduler, cfg, device, scaler) -> None:
    """Train the model for one epoch."""
    model.train()
    t0 = time.perf_counter()   # wall time per epoch, read off the step log (decision D1)
    for step, batch in enumerate(loader):

        enc = {k: v.to(device) for k, v in batch["enc"].items() if k != "labels"}
        # labels go to the device here, because the loss is computed on-device.
        labels = batch["enc"]["labels"].to(device)
        
        # optimiser with zero_grad clears the gradients of all optimized tensors.
        # This is necessary because by default, gradients are accumulated in PyTorch.
        optim.zero_grad()

        # autocast runs the forward and loss in fp16 where safe, fp32 where not
        with torch.autocast(device_type="cuda", dtype=torch.float16):
        
            # logits returns the raw output of the model (before applying softmax or sigmoid).
            logits = model(**enc)

            # compute the loss using the logits and the true labels
            loss = loss_fn(logits, labels)

            # this must print autocast=True, logits dtype=torch.float16
            if step == 0:
                print(f"  autocast={torch.is_autocast_enabled('cuda')} logits dtype={logits.dtype} device={device}")

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  step {step}/{len(loader)} loss={loss.item():.4f} elapsed={elapsed:.0f}s")

        # scales the loss to prevent underflow in fp16
        scaler.scale(loss).backward()
        # unscales the gradients of the optimizer's assigned params in-place
        scaler.unscale_(optim)
        # clips the gradients of the model's parameters to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])

        # get the current scale before stepping the optimizer, so we can check if it changed after the step
        scale_before = scaler.get_scale()
        scaler.step(optim)
        scaler.update()

        # only advance the LR schedule if the optimizer actually stepped
        # (scaler lowers the scale when it skips a step due to inf/nan)
        if scaler.get_scale() >= scale_before:
            scheduler.step()
 

def validate_and_save_best_model(cfg, device, val_loader, model, best_score, epoch, run_dir):
    """
    Score the ragtruth validation set
    save the checkpoint if it's the best so far.
    """

    # evaluate the model on the validation set and compute the ranking metrics (PR-AUC and ROC-AUC).

    # extract the labels, probabilities, and sources from the evaluation results.
    labels, probs, sources = evaluate(model, val_loader, device)

    # filter the data to only ragtruth sources since ragtruth is the main source of hallucination data. 
    
    # boolean mask that is True for ragtruth sources and False otherwise.
    ragtruth_mask = np.array([s == "ragtruth" for s in sources])

    if ragtruth_mask.sum() == 0: 
        raise ValueError("no ragtruth rows in validation")

    # apply the mask to the labels and probabilities since index is same for labels, probs and sources.
    labels = labels[ragtruth_mask]
    probs = probs[ragtruth_mask]

    # rank the metrics using the ranking_metrics function, which computes PR-AUC and ROC-AUC.
    val = ranking_metrics(labels, probs)
    print(f"[epoch {epoch}] val pr_auc={val['pr_auc']:.4f} roc_auc={val['roc_auc']:.4f}")

    score = val[cfg["train"]["select_on"]]
    if score > best_score:
        torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        print(f"  saved best ({cfg['train']['select_on']}={score:.4f})")
        return score, val

    return best_score, val

def evaluate_on_test(model, val_loader, test_loader, cfg, device, run_dir) -> None:
    """Reload the best checkpoint, pick a threshold on val, report on test."""

    # get the best checkpoint from the output directory and load it into the model.
    best_path = os.path.join(run_dir, "best.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))

    # evaluate the model on the validation set to get the labels and probabilities
    v_labels, v_probs, v_sources = evaluate(model, val_loader, device)
    # all val rows (every source), before the RAGTruth mask below
    val_records = prediction_records("val", val_loader.dataset.rows, v_labels, v_probs, cfg["data"])

    # filter the data to only ragtruth sources since ragtruth is the main source of hallucination data.
    ragtruth_mask = np.array([s == "ragtruth" for s in v_sources])

    if ragtruth_mask.sum() == 0: 
        raise ValueError("no ragtruth rows in validation")
    
    v_labels = v_labels[ragtruth_mask]
    v_probs = v_probs[ragtruth_mask]

    # find the best threshold on the validation set of ragtruth data that maximizes the F1 score.
    thr = best_threshold(v_labels, v_probs)

    # evaluate the model on the test set to get the labels and probabilities and sources.
    t_labels, t_probs, t_sources = evaluate(model, test_loader, device)

    # per-row predictions for post-hoc analyses (truncated vs untruncated rows)
    test_records = prediction_records("test", test_loader.dataset.rows, t_labels, t_probs, cfg["data"])
    write_jsonl(val_records + test_records, os.path.join(run_dir, "predictions.jsonl"))

    # compute the ranking metrics (PR-AUC and ROC-AUC) and thresholded metrics (precision, recall, F1) at the chosen threshold on the test set.
    test = ranking_metrics(t_labels, t_probs)
    test.update(thresholded_metrics(t_labels, t_probs, thr))
    print("TEST:", {k: round(v, 4) for k, v in test.items()})

    # calling per_source_breakdown on test set to get metrics for each source (e.g., ragtruth, halueval, etc.) at the best threshold.
    per_source = per_source_breakdown(t_labels, t_probs, t_sources, thr)
    
    summary = {
    "threshold": thr,
    "overall": test,
    "per_source": per_source,
    "git_sha": get_git_sha(),
    "seed": cfg["train"]["seed"],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

def train(cfg: dict) -> None:
    """
    Train the hallucination detector model.
    """
    # fail before any work if a Kaggle run would write outside /kaggle/working/
    check_out_dir(cfg["train"]["out_dir"], is_kaggle())

    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"device: {device} ({get_device_name()})")

    # built before the run dir, so a bad `experiment` fails without leaving an empty folder
    snapshot = build_run_snapshot(cfg)
    run_dir = make_run_dir(cfg)
    print(f"run_dir: {run_dir}")

    # snapshot the exact config, git SHA, experiment, input format and versions that produced this run
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=2)

    tokenizer = build_tokenizer(cfg["model"]["name"])
    gen = torch.Generator().manual_seed(cfg["train"]["seed"])  # reproducible shuffle
   
    train_loader = make_loader(cfg["data"]["train_path"], cfg, tokenizer, shuffle=True, generator=gen, sources=cfg["train"].get("train_sources"))
    val_loader = make_loader(cfg["data"]["val_path"], cfg, tokenizer, shuffle=False)
    test_loader = make_loader(cfg["data"]["test_path"], cfg, tokenizer, shuffle=False)
 
    model = HallucinationDetector(cfg["model"]["name"], cfg["model"]["dropout"]).to(device)
    loss_fn = build_loss(cfg["train"]["pos_weight"]).to(device)
    optim, scheduler = build_optimizer_and_scheduler(model, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler()

    best_score = -1.0
    val_history = []
    for epoch in range(cfg["train"]["epochs"]):
        train_one_epoch(model, train_loader, loss_fn, optim, scheduler, cfg, device, scaler)
        best_score, val = validate_and_save_best_model(cfg, device, val_loader, model, best_score, epoch, run_dir)
        val_history.append({"epoch": epoch, **val})

    with open(os.path.join(run_dir, "val_history.json"), "w") as f:
        json.dump(val_history, f, indent=2)
        
    evaluate_on_test(model, val_loader, test_loader, cfg, device, run_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=None,
                    help="overrides train.out_dir; on Kaggle must be absolute under /kaggle/working/")
    return ap.parse_args(argv)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """CLI flags win over the config; done before the snapshot so it records the values actually used."""
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.out_dir is not None:
        cfg["train"]["out_dir"] = args.out_dir


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)
    train(cfg)


if __name__ == "__main__":
    main()
