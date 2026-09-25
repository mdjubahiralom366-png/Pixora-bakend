"""
main.py

FastAPI inference server for the Pixora Android app.

Request/response contract matches the EXISTING MainActivity.java callApi()
exactly, so the Android app needs zero code changes — only API_URL has
to be pointed at wherever this server is deployed.

    POST /api/chat
    body:  {"message": "...", "user_name": "...", "history": [...],
            "web_search": false, "memory": "..."}
    reply: {"reply": "...", "memory": "..."}

There is no external AI API call anywhere in this chain — main.py ->
model_server.py -> local Pixora weights only.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixora.server")

from model_server import get_engine  # noqa: E402
from prompts import check_input, sanitize_output, SafetyRejection  # noqa: E402

app = FastAPI(title="Pixora Inference Server")


class ChatRequest(BaseModel):
    message: str
    user_name: Optional[str] = None
    history: list[dict] = []          # [{"role": "user"|"model", "text": "..."}]
    web_search: bool = False
    memory: Optional[str] = ""


class ChatResponse(BaseModel):
    reply: str
    memory: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model": "pixora"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # web_search: Pixora is fully self-hosted with no external API calls,
    # so there is no live web-search backend wired up yet. We accept the
    # flag (so the app doesn't break) but currently answer from the
    # model's own knowledge either way. Wire a real search tool in here
    # later if you add one.
    if req.web_search:
        logger.info("web_search requested but not implemented — answering without it.")

    try:
        check_input(req.message)
    except SafetyRejection as exc:
        return ChatResponse(reply=f"Server Error: {exc}", memory=req.memory or "")

    try:
        engine = get_engine()
        raw_reply = engine.generate(
            req.message,
            history=req.history,
            user_name=req.user_name,
            memory=req.memory,
        )
        reply = sanitize_output(raw_reply)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed")
        return ChatResponse(reply=f"Server Error: {exc}", memory=req.memory or "")

    # NOTE: this does not yet extract new long-term facts from the
    # conversation — it just passes the existing memory string through
    # unchanged. To actually grow `memory` over time, add a second,
    # lightweight generation pass here that asks the model to summarize
    # any new durable facts the user shared, and merge them in.
    return ChatResponse(reply=reply, memory=req.memory or "")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.environ.get("SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVER_PORT", 8000)),
        reload=False,
    )
