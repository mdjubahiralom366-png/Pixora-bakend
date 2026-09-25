"""
dataset_builder.py

Orchestrates the full offline pipeline:

    Firebase -> firebase_loader -> cleaner -> formatter -> train.jsonl/validation.jsonl
                                                          -> (optional) packed token shards

Run this whenever you want to build a new dataset version from whatever
is currently authorized in Firebase.

    python dataset_builder.py --version dataset-v1
"""

from __future__ import annotations

import argparse
import json
import os
import logging

import numpy as np

import firebase_loader
import cleaner
import formatter

logger = logging.getLogger("pixora.dataset_builder")


def build_dataset(dataset_dir: str, version: str, require_consent_field: str | None = None) -> dict:
    version_dir = os.path.join(dataset_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    raw = firebase_loader.load_raw_records()
    firebase_loader.save_records_to_local_cache(raw, os.path.join(version_dir, "raw_pull.jsonl"))

    cleaned, report = cleaner.clean_records(raw, require_consent_field=require_consent_field)
    with open(os.path.join(version_dir, "cleaned.jsonl"), "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    examples = formatter.format_records(cleaned)
    train, val = formatter.train_val_split(examples)

    with open(os.path.join(version_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps({"text": ex.text}, ensure_ascii=False) + "\n")
    with open(os.path.join(version_dir, "validation.jsonl"), "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps({"text": ex.text}, ensure_ascii=False) + "\n")

    manifest = {
        "version": version,
        "raw_count": len(raw),
        "cleaned_count": len(cleaned),
        "train_count": len(train),
        "val_count": len(val),
        "clean_report": report.__dict__,
    }
    with open(os.path.join(version_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Convenience symlink-style copies at dataset_dir root so train.py can
    # always point at DATASET_DIR without needing to know the version.
    for name in ("train.jsonl", "validation.jsonl"):
        src = os.path.join(version_dir, name)
        dst = os.path.join(dataset_dir, name)
        with open(src, encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())

    logger.info("Built %s: %s", version, manifest)
    return manifest


def pack_for_scratch_training(
    tokenizer_dir: str, jsonl_path: str, out_path: str, max_sequence_length: int
) -> None:
    """For Mode B (from-scratch): tokenize the whole corpus and pack it
    into fixed-length blocks of token ids, saved as a single .npy file
    of shape (num_blocks, max_sequence_length). This is the standard
    "concatenate + chunk" packing used for causal LM pretraining."""
    from tokenizer import load_tokenizer

    tok = load_tokenizer(tokenizer_dir)
    all_ids: list[int] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            all_ids.extend(tok.encode(text))

    n_blocks = len(all_ids) // max_sequence_length
    all_ids = all_ids[: n_blocks * max_sequence_length]
    arr = np.array(all_ids, dtype=np.uint16).reshape(n_blocks, max_sequence_length)
    np.save(out_path, arr)
    print(f"Packed {n_blocks} blocks of {max_sequence_length} tokens -> {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR", "./dataset"))
    parser.add_argument("--version", required=True, help="e.g. dataset-v1")
    parser.add_argument("--require-consent-field", default=None)
    args = parser.parse_args()
    build_dataset(args.dataset_dir, args.version, args.require_consent_field)
