"""
train.py

Two supported modes, both real gradient-descent training on next-token
prediction — no lookup tables, no canned responses:

  TRAINING_MODE=finetune  (Mode A, recommended first)
      Loads an open-weight causal LM (e.g. Qwen2.5-0.5B) via
      transformers.AutoModelForCausalLM and continues training it on
      dataset/train.jsonl. This is the practical path: a small
      Firebase-sized dataset can meaningfully shift a competent base
      model's behavior; it cannot make a from-scratch model competent.

  TRAINING_MODE=scratch   (Mode B, research)
      Trains training/model.py's PixoraForCausalLM from randomly
      initialized weights on packed token blocks. This needs orders of
      magnitude more data and compute than fine-tuning to reach usable
      quality — treat it as a research track, not a v1 plan.

Usage:
    python train.py                      # reads all config from .env
    python train.py --resume ./models/pixora/checkpoints/step-1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pixora.train")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("No GPU detected — falling back to CPU. Training will be slow.")
    return torch.device("cpu")


def get_amp_dtype(mixed_precision: str, device: torch.device):
    if device.type != "cuda":
        return None
    if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


# ────────────────────────────── Mode A: fine-tuning ──────────────────────────────

def run_finetune(cfg: dict) -> None:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset

    logger.info("Mode A: fine-tuning base model %s", cfg["base_model_name"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model_name"])

    raw = load_dataset(
        "json",
        data_files={"train": f"{cfg['dataset_dir']}/train.jsonl",
                    "validation": f"{cfg['dataset_dir']}/validation.jsonl"},
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=cfg["max_sequence_length"],
        )

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=os.path.join(cfg["model_dir"], "checkpoints"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        warmup_steps=cfg["warmup_steps"],
        eval_strategy="steps",
        eval_steps=cfg["save_every_steps"],
        save_strategy="steps",
        save_steps=cfg["save_every_steps"],
        save_total_limit=3,
        logging_steps=20,
        bf16=(cfg["mixed_precision"] == "bf16"),
        fp16=(cfg["mixed_precision"] == "fp16"),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    resume_ckpt = cfg.get("resume")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = os.path.join(cfg["model_dir"], "model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Fine-tuned model saved to %s", final_dir)


# ────────────────────────────── Mode B: from-scratch ──────────────────────────────

class PackedBlockDataset(Dataset):
    """Loads the .npy produced by dataset_builder.pack_for_scratch_training —
    each row is already a fixed-length block of token ids."""

    def __init__(self, npy_path: str):
        self.data = np.load(npy_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx].astype(np.int64))


def run_scratch(cfg: dict) -> None:
    from model import PixoraConfig, PixoraForCausalLM

    device = get_device()
    amp_dtype = get_amp_dtype(cfg["mixed_precision"], device)

    model_config = PixoraConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        max_sequence_length=cfg["max_sequence_length"],
    )
    model = PixoraForCausalLM(model_config).to(device)
    logger.info("Mode B: from-scratch Pixora model, %.1fM params", model.num_parameters() / 1e6)

    start_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        logger.info("Resumed from step %d", start_step)

    train_ds = PackedBlockDataset(os.path.join(cfg["dataset_dir"], "train_packed.npy"))
    val_ds = PackedBlockDataset(os.path.join(cfg["dataset_dir"], "validation_packed.npy"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    ckpt_dir = os.path.join(cfg["model_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    step = start_step
    model.train()
    for epoch in range(cfg["epochs"]):
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(batch, labels=batch)
                loss = loss / cfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (i + 1) % cfg["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

                if step % 20 == 0:
                    logger.info("epoch=%d step=%d loss=%.4f", epoch, step,
                                loss.item() * cfg["gradient_accumulation_steps"])

                if step % cfg["save_every_steps"] == 0:
                    val_loss = evaluate_scratch(model, val_loader, device, amp_dtype)
                    logger.info("step=%d validation_loss=%.4f perplexity=%.2f",
                                step, val_loss, math.exp(min(val_loss, 20)))
                    torch.save(
                        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                        os.path.join(ckpt_dir, f"step-{step}.pt"),
                    )
                    model.train()

    final_dir = os.path.join(cfg["model_dir"], "model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.pt"))
    with open(os.path.join(final_dir, "config.json"), "w") as f:
        json.dump(model_config.__dict__, f, indent=2)
    logger.info("From-scratch model saved to %s", final_dir)


@torch.no_grad()
def evaluate_scratch(model, val_loader, device, amp_dtype) -> float:
    model.eval()
    losses = []
    for batch in val_loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(batch, labels=batch)
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


# ────────────────────────────── entry point ──────────────────────────────

def load_config(args) -> dict:
    return {
        "training_mode": os.environ.get("TRAINING_MODE", "finetune"),
        "base_model_name": os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-0.5B"),
        "model_dir": os.environ.get("MODEL_DIR", "./models/pixora"),
        "dataset_dir": os.environ.get("DATASET_DIR", "./dataset"),
        "vocab_size": int(os.environ.get("VOCAB_SIZE", 32000)),
        "hidden_size": int(os.environ.get("HIDDEN_SIZE", 768)),
        "num_layers": int(os.environ.get("NUM_LAYERS", 12)),
        "num_attention_heads": int(os.environ.get("NUM_ATTENTION_HEADS", 12)),
        "max_sequence_length": int(os.environ.get("MAX_SEQUENCE_LENGTH", 1024)),
        "batch_size": int(os.environ.get("BATCH_SIZE", 4)),
        "gradient_accumulation_steps": int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 8)),
        "learning_rate": float(os.environ.get("LEARNING_RATE", 2e-5)),
        "epochs": int(os.environ.get("EPOCHS", 3)),
        "warmup_steps": int(os.environ.get("WARMUP_STEPS", 100)),
        "save_every_steps": int(os.environ.get("SAVE_EVERY_STEPS", 500)),
        "mixed_precision": os.environ.get("MIXED_PRECISION", "bf16"),
        "resume": args.resume,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Checkpoint path/dir to resume from")
    args = parser.parse_args()

    cfg = load_config(args)
    if cfg["training_mode"] == "finetune":
        run_finetune(cfg)
    elif cfg["training_mode"] == "scratch":
        run_scratch(cfg)
    else:
        raise ValueError(f"Unknown TRAINING_MODE: {cfg['training_mode']}")
