"""
tokenizer.py

Trains a real subword (byte-level BPE) tokenizer over the formatted
training corpus, and saves it as a HuggingFace-compatible tokenizer.json
so it travels with the model checkpoint. This is deliberately NOT a
character-splitter — subword tokenization is what makes the vocab_size /
sequence-length knobs in the model config meaningful.

Usage:
    python tokenizer.py --train ./dataset/train.jsonl --out ./models/pixora/tokenizer --vocab-size 32000
"""

from __future__ import annotations

import argparse
import json
import os

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from formatter import SPECIAL_TOKENS, BOS, EOS


def iter_training_lines(jsonl_path: str):
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"]


def train_tokenizer(train_jsonl: str, out_dir: str, vocab_size: int = 32000) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Write a flat corpus file — ByteLevelBPETokenizer's trainer wants file paths
    corpus_path = os.path.join(out_dir, "_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in iter_training_lines(train_jsonl):
            f.write(text.replace("\n", " ") + "\n")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>"] + SPECIAL_TOKENS,
    )

    tokenizer.save_model(out_dir)  # vocab.json + merges.txt

    # Wrap as a fast, HF-compatible tokenizer so training/inference code
    # can just use AutoTokenizer.from_pretrained(out_dir) later.
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS,
        eos_token=EOS,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=SPECIAL_TOKENS,
    )
    fast_tokenizer.save_pretrained(out_dir)

    os.remove(corpus_path)
    print(f"Tokenizer trained (vocab_size={vocab_size}) and saved to {out_dir}")


def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./dataset/train.jsonl")
    parser.add_argument("--out", default="./models/pixora/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()
    train_tokenizer(args.train, args.out, args.vocab_size)
