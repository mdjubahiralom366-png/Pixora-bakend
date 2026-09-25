"""
model_server.py

Loads the Pixora model ONCE at process start and exposes a plain
`generate(message, history)` function. There is deliberately no code
path here that calls OpenAI / Gemini / Claude / Grok or any other
external inference API — Pixora only ever talks to its own local
weights, loaded from MODEL_DIR.
"""

from __future__ import annotations

import os
import sys
import logging

import torch

logger = logging.getLogger("pixora.model_server")

TRAINING_MODE = os.environ.get("TRAINING_MODE", "finetune")
MODEL_DIR = os.path.join(os.environ.get("MODEL_DIR", "./models/pixora"), "model")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from formatter import DEFAULT_SYSTEM_PROMPT, SYS_OPEN, SYS_CLOSE, USER_OPEN, USER_CLOSE, ASSIST_OPEN  # noqa: E402


class PixoraEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = TRAINING_MODE
        logger.info("Loading Pixora model (mode=%s, device=%s) from %s", self.mode, self.device, MODEL_DIR)

        if self.mode == "finetune":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(self.device).eval()
        elif self.mode == "scratch":
            import json
            from model import PixoraConfig, PixoraForCausalLM
            from tokenizer import load_tokenizer

            with open(os.path.join(MODEL_DIR, "config.json")) as f:
                cfg = PixoraConfig(**json.load(f))
            self.model = PixoraForCausalLM(cfg).to(self.device)
            state = torch.load(os.path.join(MODEL_DIR, "pytorch_model.pt"), map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.tokenizer = load_tokenizer(os.environ.get("TOKENIZER_DIR", "./models/pixora/tokenizer"))
        else:
            raise ValueError(f"Unknown TRAINING_MODE: {self.mode}")

        logger.info("Pixora model loaded successfully.")

    def _build_prompt(self, message: str, history: list[dict] | None,
                       user_name: str | None, memory: str | None) -> str:
        system_text = DEFAULT_SYSTEM_PROMPT
        if user_name:
            system_text += f" The user's name is {user_name}."
        if memory:
            system_text += f" Remembered context about the user: {memory}"

        parts = [f"{SYS_OPEN}{system_text}{SYS_CLOSE}"]
        for turn in (history or []):
            # Android app sends {"role": "user"|"model", "text": "..."}
            role = turn.get("role")
            content = turn.get("text", turn.get("content", ""))
            if not content:
                continue
            if role == "user":
                parts.append(f"{USER_OPEN}{content}{USER_CLOSE}")
            elif role in ("model", "assistant"):
                parts.append(f"{ASSIST_OPEN}{content}<|/assistant|>")
        parts.append(f"{USER_OPEN}{message}{USER_CLOSE}")
        parts.append(ASSIST_OPEN)
        return "".join(parts)

    def generate(self, message: str, history: list[dict] | None = None,
                 user_name: str | None = None, memory: str | None = None,
                 max_new_tokens: int | None = None, temperature: float | None = None) -> str:
        prompt = self._build_prompt(message, history, user_name, memory)
        max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        temperature = temperature if temperature is not None else TEMPERATURE

        if self.mode == "finetune":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=max(temperature, 1e-3),
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            out_ids = self.model.generate(
                ids, max_new_tokens=max_new_tokens, temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new_ids = out_ids[0][ids.shape[1]:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return text.strip()


_engine: PixoraEngine | None = None


def get_engine() -> PixoraEngine:
    global _engine
    if _engine is None:
        _engine = PixoraEngine()
    return _engine
