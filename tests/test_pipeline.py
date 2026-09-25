"""Basic smoke tests. Run with: pytest tests/"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))

from cleaner import clean_records
from formatter import format_records, to_standard_messages, render_messages


def test_cleaner_dedup_and_secret_redaction():
    records = [
        {"instruction": "Say hi", "input": "", "output": "Hello there, my key is AIza" + "X" * 35},
        {"instruction": "Say hi", "input": "", "output": "Hello there, my key is AIza" + "X" * 35},  # dup
        {"instruction": "", "output": ""},  # empty
        {"foo": "bar"},  # malformed
    ]
    cleaned, report = clean_records(records)
    assert report.kept == 1
    assert report.dropped_duplicate == 1
    assert "[REDACTED_SECRET]" in cleaned[0]["output"]


def test_formatter_round_trip():
    record = {"instruction": "What is 2+2?", "input": "", "output": "4"}
    messages = to_standard_messages(record)
    full, prompt, completion = render_messages(messages)
    assert "What is 2+2?" in full
    assert completion.startswith("4")
    assert full.startswith(prompt)


def test_format_records_batch():
    records = [{"instruction": "Hi", "input": "", "output": "Hello!"}]
    examples = format_records(records)
    assert len(examples) == 1
    assert "Hello!" in examples[0].text
