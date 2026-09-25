"""
cleaner.py

Cleans RAW Firebase records before they are turned into a training
format. Responsibilities:
  - drop empty / malformed records
  - deduplicate
  - normalize whitespace/unicode
  - filter out examples that are too short to be useful
  - redact accidental secrets (API keys, tokens, passwords)
  - strip common PII patterns (emails, phone numbers)
  - flag/skip anything without a consent marker, if your app tracks one

This module never trains anything and never talks to Firebase directly —
it's a pure function pipeline over a list[dict], testable in isolation.
"""

from __future__ import annotations

import re
import unicodedata
import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("pixora.cleaner")

MIN_CHARS = 8          # examples shorter than this carry little signal
MAX_CHARS = 20_000      # guard against pasted logs / junk dumps

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),                      # Google API keys
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                          # generic secret-key shape
    re.compile(r"AKIA[0-9A-Z]{16}"),                             # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bpassword\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9\-_.]{20,}"),
]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")


@dataclass
class CleanReport:
    total_in: int = 0
    dropped_empty: int = 0
    dropped_malformed: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_duplicate: int = 0
    dropped_no_consent: int = 0
    redacted_secret_count: int = 0
    redacted_pii_count: int = 0
    kept: int = 0
    notes: list[str] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _redact_secrets(text: str, report: CleanReport) -> str:
    for pattern in _SECRET_PATTERNS:
        text, n = pattern.subn("[REDACTED_SECRET]", text)
        report.redacted_secret_count += n
    return text


def _redact_pii(text: str, report: CleanReport) -> str:
    text, n1 = _EMAIL_RE.subn("[REDACTED_EMAIL]", text)
    text, n2 = _PHONE_RE.subn("[REDACTED_PHONE]", text)
    report.redacted_pii_count += (n1 + n2)
    return text


def _extract_text_fields(record: dict) -> list[str]:
    """Pull out the human-readable text of a record regardless of whether
    it's instruction/output shaped or messages-array shaped."""
    texts = []
    if "instruction" in record or "output" in record:
        texts.append(str(record.get("instruction", "")))
        texts.append(str(record.get("input", "")))
        texts.append(str(record.get("output", "")))
    if "messages" in record and isinstance(record["messages"], list):
        for m in record["messages"]:
            if isinstance(m, dict):
                texts.append(str(m.get("content", "")))
    return texts


def _record_fingerprint(record: dict) -> str:
    joined = "||".join(_extract_text_fields(record)).strip().lower()
    joined = re.sub(r"\s+", " ", joined)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _is_malformed(record: dict) -> bool:
    has_instruction_shape = "instruction" in record and "output" in record
    has_chat_shape = "messages" in record and isinstance(record["messages"], list) and len(record["messages"]) > 0
    return not (has_instruction_shape or has_chat_shape)


def clean_records(
    raw_records: list[dict],
    require_consent_field: str | None = None,
) -> tuple[list[dict], CleanReport]:
    """
    require_consent_field: if your Firebase schema marks records with e.g.
    "consented": true, pass that field name here to enforce it. None
    skips the check (only do this for data you already know is authorized,
    e.g. curated instruction data you wrote yourselves).
    """
    report = CleanReport(total_in=len(raw_records))
    seen_fingerprints: set[str] = set()
    cleaned: list[dict] = []

    for record in raw_records:
        if not record:
            report.dropped_empty += 1
            continue

        if require_consent_field is not None and not record.get(require_consent_field):
            report.dropped_no_consent += 1
            continue

        if _is_malformed(record):
            report.dropped_malformed += 1
            continue

        # Normalize + redact every text field in place
        working = dict(record)
        if "instruction" in working:
            working["instruction"] = _normalize_text(str(working.get("instruction", "")))
            working["input"] = _normalize_text(str(working.get("input", "")))
            working["output"] = _normalize_text(str(working.get("output", "")))
            for key in ("instruction", "input", "output"):
                working[key] = _redact_secrets(working[key], report)
                working[key] = _redact_pii(working[key], report)
            total_len = len(working["instruction"]) + len(working["output"])
        else:
            new_messages = []
            total_len = 0
            for m in working["messages"]:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    continue
                content = _normalize_text(str(m["content"]))
                content = _redact_secrets(content, report)
                content = _redact_pii(content, report)
                new_messages.append({"role": m["role"], "content": content})
                total_len += len(content)
            working["messages"] = new_messages
            if not new_messages:
                report.dropped_malformed += 1
                continue

        if total_len < MIN_CHARS:
            report.dropped_too_short += 1
            continue
        if total_len > MAX_CHARS:
            report.dropped_too_long += 1
            continue

        fingerprint = _record_fingerprint(working)
        if fingerprint in seen_fingerprints:
            report.dropped_duplicate += 1
            continue
        seen_fingerprints.add(fingerprint)

        cleaned.append(working)
        report.kept += 1

    logger.info("Clean report: %s", report)
    return cleaned, report


if __name__ == "__main__":
    import json
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "./dataset/raw_pull.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "./dataset/cleaned.jsonl"

    raw = [json.loads(line) for line in open(in_path, encoding="utf-8")]
    cleaned, rep = clean_records(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(rep.__dict__, indent=2))
    print(f"Wrote {len(cleaned)} cleaned records to {out_path}")
