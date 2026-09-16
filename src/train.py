"""Training entry point.

This is the main training script. It loads the configuration, sets up the model and
data, and runs the training loop.

Run:  python -m src.train --config configs/base.yaml
"""

from __future__ import annotations
import argparse
from datetime import datetime
import os
import random
import json
import subprocess

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import HallucinationDataset
from src.encoding import Collator, build_tokenizer
from src.model import HallucinationDetector, build_loss
from src.metrics import per_source_breakdown, probs_from_logits, ranking_metrics, best_threshold, thresholded_metrics


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

        if step % 50 == 0:
            print(f"  step {step}/{len(loader)} loss={loss.item():.4f}")  

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
    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"device: {device}")

    run_dir = make_run_dir(cfg)
    print(f"run_dir: {run_dir}")

    # snapshot the exact config and git SHA that produced this run
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump({"config": cfg, "git_sha": get_git_sha()}, f, indent=2)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
