"""
memory.py

Reads/writes per-chat conversation history to Firebase Firestore, so the
Android app's chat_id can carry context across turns. This is
retrieval/storage only — it never generates text itself (see project
doc section 17: keep retrieval and inference separate).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from firebase_loader import _init_app  # reuses the same Admin app init  # noqa: E402
from firebase_admin import firestore

MAX_TURNS_KEPT = 20  # cap context so prompts don't grow unbounded


def _chats_collection():
    _init_app()
    return firestore.client().collection("chats")


def get_history(chat_id: str) -> list[dict]:
    doc = _chats_collection().document(chat_id).get()
    if not doc.exists:
        return []
    data = doc.to_dict() or {}
    return data.get("messages", [])[-MAX_TURNS_KEPT:]


def append_turn(chat_id: str, role: str, content: str) -> None:
    doc_ref = _chats_collection().document(chat_id)
    doc = doc_ref.get()
    messages = (doc.to_dict() or {}).get("messages", []) if doc.exists else []
    messages.append({"role": role, "content": content, "ts": time.time()})
    messages = messages[-MAX_TURNS_KEPT:]
    doc_ref.set({"messages": messages, "updated_at": time.time()}, merge=True)
