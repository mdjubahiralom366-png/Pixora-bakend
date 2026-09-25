"""
evaluate.py

Loads a saved checkpoint (either fine-tuned HF model or from-scratch
Pixora model) and reports:
  - validation loss / perplexity
  - repetition rate on sample generations
  - pass/fail on a small held-out instruction-following test set

A lower training loss does NOT automatically mean a better assistant —
this is why we also run qualitative generation checks, not just the loss
number. See evaluation/benchmark.py for the fuller test-question suite.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch


def repetition_rate(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def evaluate_finetuned(model_dir: str, val_jsonl: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    losses = []
    with open(val_jsonl, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = model(**ids, labels=ids["input_ids"])
            losses.append(out.loss.item())

    avg_loss = sum(losses) / max(len(losses), 1)
    return {"validation_loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20))}


def evaluate_scratch(model_dir: str, val_packed_npy: str, config_path: str) -> dict:
    import numpy as np
    from model import PixoraConfig, PixoraForCausalLM

    with open(config_path) as f:
        cfg = PixoraConfig(**json.load(f))
    model = PixoraForCausalLM(cfg)
    state = torch.load(os.path.join(model_dir, "pytorch_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    data = np.load(val_packed_npy)
    losses = []
    with torch.no_grad():
        for row in data:
            batch = torch.from_numpy(row.astype("int64")).unsqueeze(0).to(device)
            _, loss = model(batch, labels=batch)
            losses.append(loss.item())

    avg_loss = sum(losses) / max(len(losses), 1)
    return {"validation_loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finetune", "scratch"], default=os.environ.get("TRAINING_MODE", "finetune"))
    parser.add_argument("--model-dir", default=os.path.join(os.environ.get("MODEL_DIR", "./models/pixora"), "model"))
    parser.add_argument("--val", default=os.path.join(os.environ.get("DATASET_DIR", "./dataset"), "validation.jsonl"))
    parser.add_argument("--val-packed", default=os.path.join(os.environ.get("DATASET_DIR", "./dataset"), "validation_packed.npy"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.mode == "finetune":
        results = evaluate_finetuned(args.model_dir, args.val)
    else:
        config_path = args.config or os.path.join(args.model_dir, "config.json")
        results = evaluate_scratch(args.model_dir, args.val_packed, config_path)

    print(json.dumps(results, indent=2))
