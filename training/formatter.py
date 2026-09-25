"""
formatter.py

Converts CLEANED records (either instruction/output shape or messages
shape) into one standardized internal format:

    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

...and renders that into the plain-text sequence the tokenizer/model will
actually train on, using a simple, explicit chat template (no hidden
magic). Also does the train/validation split.

Downstream (dataset_builder.py) is responsible for tokenizing this text.
"""

from __future__ import annotations

import json
import random
import logging
from dataclasses import dataclass

logger = logging.getLogger("pixora.formatter")

DEFAULT_SYSTEM_PROMPT = (
    "You are Pixora, a helpful AI assistant. Answer clearly and honestly."
)

# Explicit, simple chat template. Special tokens must also be added to the
# tokenizer (see tokenizer.py) so the model can learn turn boundaries.
BOS = "<|bos|>"
EOS = "<|eos|>"
SYS_OPEN, SYS_CLOSE = "<|system|>", "<|/system|>"
USER_OPEN, USER_CLOSE = "<|user|>", "<|/user|>"
ASSIST_OPEN, ASSIST_CLOSE = "<|assistant|>", "<|/assistant|>"

SPECIAL_TOKENS = [BOS, EOS, SYS_OPEN, SYS_CLOSE, USER_OPEN, USER_CLOSE, ASSIST_OPEN, ASSIST_CLOSE]


@dataclass
class FormattedExample:
    text: str                 # full rendered sequence for training
    prompt_text: str          # everything up to (not including) the assistant's answer
    completion_text: str      # the assistant's answer alone
    source_id: str | None = None


def to_standard_messages(record: dict) -> list[dict]:
    """Normalize either shape into a canonical messages list."""
    if "messages" in record:
        return record["messages"]

    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    instruction = record.get("instruction", "").strip()
    extra_input = record.get("input", "").strip()
    user_content = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": record.get("output", "").strip()})
    return messages


def render_messages(messages: list[dict]) -> tuple[str, str, str]:
    """Render a canonical messages list into (full_text, prompt_text, completion_text).
    Only the LAST assistant turn is treated as the "completion" for
    loss-masking purposes; everything before it is "prompt"."""
    if not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages

    last_assistant_idx = None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            last_assistant_idx = i
    if last_assistant_idx is None:
        raise ValueError("No assistant turn found in messages")

    def render_turn(m: dict) -> str:
        role = m["role"]
        content = m["content"]
        if role == "system":
            return f"{SYS_OPEN}{content}{SYS_CLOSE}"
        if role == "user":
            return f"{USER_OPEN}{content}{USER_CLOSE}"
        if role == "assistant":
            return f"{ASSIST_OPEN}{content}{ASSIST_CLOSE}"
        raise ValueError(f"Unknown role: {role}")

    prompt_parts = [render_turn(m) for m in messages[:last_assistant_idx]]
    prompt_parts.append(ASSIST_OPEN)  # cue the model to start generating
    prompt_text = BOS + "".join(prompt_parts)

    completion_text = messages[last_assistant_idx]["content"] + ASSIST_CLOSE + EOS
    full_text = prompt_text + completion_text
    return full_text, prompt_text, completion_text


def format_records(records: list[dict]) -> list[FormattedExample]:
    formatted = []
    for record in records:
        try:
            messages = to_standard_messages(record)
            full, prompt, completion = render_messages(messages)
            formatted.append(FormattedExample(
                text=full, prompt_text=prompt, completion_text=completion,
                source_id=record.get("_id"),
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping record %s: %s", record.get("_id"), exc)
    return formatted


def train_val_split(
    examples: list[FormattedExample], val_ratio: float = 0.05, seed: int = 42
) -> tuple[list[FormattedExample], list[FormattedExample]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 20 else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


if __name__ == "__main__":
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "./dataset/cleaned.jsonl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./dataset"

    records = [json.loads(line) for line in open(in_path, encoding="utf-8")]
    examples = format_records(records)
    train, val = train_val_split(examples)

    with open(f"{out_dir}/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps({"text": ex.text, "prompt": ex.prompt_text,
                                 "completion": ex.completion_text}, ensure_ascii=False) + "\n")

    with open(f"{out_dir}/validation.jsonl", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps({"text": ex.text, "prompt": ex.prompt_text,
                                 "completion": ex.completion_text}, ensure_ascii=False) + "\n")

    print(f"Formatted {len(examples)} examples -> {len(train)} train / {len(val)} val")
