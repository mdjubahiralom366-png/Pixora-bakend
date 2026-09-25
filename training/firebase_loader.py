"""
firebase_loader.py

Pulls RAW authorized training records out of Firebase using the Firebase
Admin SDK (server-side, privileged credentials — never ship this key to
the Android app or to GitHub).

This module ONLY retrieves data. It does not clean, format, or judge
quality — that is cleaner.py / formatter.py's job. Keeping retrieval,
cleaning, formatting, and inference in separate modules is intentional
(see project doc section 17 — don't confuse these concepts).

Env vars used (see .env.example):
    FIREBASE_CREDENTIALS_PATH
    FIREBASE_PROJECT_ID
    FIREBASE_DATABASE_URL
    FIREBASE_DATA_BACKEND        "firestore" | "rtdb"
    FIREBASE_TRAINING_COLLECTION
"""

from __future__ import annotations

import os
import json
import logging
from typing import Iterator, Any

import firebase_admin
from firebase_admin import credentials, firestore, db

logger = logging.getLogger("pixora.firebase_loader")

_app: firebase_admin.App | None = None


def _init_app() -> firebase_admin.App:
    """Initialize the Firebase Admin app exactly once, from a service
    account key referenced by an env var — never a hard-coded path or
    a key committed to source control."""
    global _app
    if _app is not None:
        return _app

    cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
    if not cred_path or not os.path.exists(cred_path):
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_PATH is not set or the file does not exist. "
            "Download a service-account key from Firebase Console > Project "
            "Settings > Service Accounts, keep it OUT of git, and point "
            "this env var at it."
        )

    cred = credentials.Certificate(cred_path)
    options = {}
    db_url = os.environ.get("FIREBASE_DATABASE_URL")
    if db_url:
        options["databaseURL"] = db_url
    project_id = os.environ.get("FIREBASE_PROJECT_ID")
    if project_id:
        options["projectId"] = project_id

    _app = firebase_admin.initialize_app(cred, options)
    logger.info("Firebase Admin app initialized for project=%s", project_id)
    return _app


def iter_firestore_records(collection: str, page_size: int = 500) -> Iterator[dict]:
    """Stream documents from a Firestore collection as plain dicts,
    paginated so we never load an unbounded dataset into memory at once."""
    _init_app()
    fs = firestore.client()
    col_ref = fs.collection(collection)

    last_doc = None
    while True:
        query = col_ref.order_by("__name__").limit(page_size)
        if last_doc is not None:
            query = query.start_after(last_doc)
        docs = list(query.stream())
        if not docs:
            break
        for doc in docs:
            record = doc.to_dict() or {}
            record["_id"] = doc.id
            yield record
        last_doc = docs[-1]


def iter_rtdb_records(path: str) -> Iterator[dict]:
    """Stream child records from a Realtime Database path."""
    _init_app()
    ref = db.reference(path)
    snapshot = ref.get()
    if not snapshot:
        return
    # RTDB returns either a dict keyed by push-id, or a raw list
    if isinstance(snapshot, dict):
        for key, value in snapshot.items():
            if isinstance(value, dict):
                value = dict(value)
                value["_id"] = key
                yield value
    elif isinstance(snapshot, list):
        for i, value in enumerate(snapshot):
            if isinstance(value, dict):
                value = dict(value)
                value["_id"] = str(i)
                yield value


def load_raw_records() -> list[dict]:
    """Entry point used by dataset_builder.py. Picks the backend based
    on FIREBASE_DATA_BACKEND and returns a materialized list."""
    backend = os.environ.get("FIREBASE_DATA_BACKEND", "firestore").lower()
    if backend == "firestore":
        collection = os.environ.get("FIREBASE_TRAINING_COLLECTION", "training_examples")
        records = list(iter_firestore_records(collection))
    elif backend == "rtdb":
        path = os.environ.get("FIREBASE_TRAINING_COLLECTION", "training_examples")
        records = list(iter_rtdb_records(path))
    else:
        raise ValueError(f"Unknown FIREBASE_DATA_BACKEND: {backend}")

    logger.info("Loaded %d raw records from Firebase (%s)", len(records), backend)
    return records


def save_records_to_local_cache(records: list[dict], path: str) -> None:
    """Optional: cache raw pulls locally so repeated pipeline runs don't
    hammer Firebase. This cache is gitignored (contains user data)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    recs = load_raw_records()
    out_path = os.path.join(os.environ.get("DATASET_DIR", "./dataset"), "raw_pull.jsonl")
    save_records_to_local_cache(recs, out_path)
    print(f"Wrote {len(recs)} raw records to {out_path}")
