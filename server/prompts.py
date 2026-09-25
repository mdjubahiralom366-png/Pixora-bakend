"""
prompts.py

Lightweight safety layer around the model (project doc section 15).
This is a basic filter, not a substitute for real content-safety
infrastructure — treat it as a first line of defense you should expand
over time (e.g. with a dedicated classifier model).
"""

from __future__ import annotations

import re

BLOCKED_INPUT_PATTERNS = [
    re.compile(r"(?i)how (do|can) i (make|build|synthesize) (a )?(bomb|explosive)"),
    re.compile(r"(?i)\bchild (sexual|porn|abuse)\b"),
]

UNSAFE_OUTPUT_MARKERS = [
    "i am now unrestricted",
    "ignore previous instructions",
]

MAX_INPUT_CHARS = 4000


class SafetyRejection(Exception):
    pass


def check_input(message: str) -> None:
    if not message or not message.strip():
        raise SafetyRejection("Empty message.")
    if len(message) > MAX_INPUT_CHARS:
        raise SafetyRejection(f"Message too long (max {MAX_INPUT_CHARS} characters).")
    for pattern in BLOCKED_INPUT_PATTERNS:
        if pattern.search(message):
            raise SafetyRejection("This request can't be processed.")


def sanitize_output(text: str) -> str:
    lowered = text.lower()
    for marker in UNSAFE_OUTPUT_MARKERS:
        if marker in lowered:
            return "I can't help with that."
    return text
