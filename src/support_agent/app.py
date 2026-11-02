"""FastAPI application: chat API, tickets API, health check and the web UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import SessionStore, SupportAgent
from .config import Settings
from .db import SupportDB
from .kb import KnowledgeBase, openai_embedder
from .providers import LLMProvider, build_provider
from .tools import ToolExecutor

STATIC_DIR = Path(__file__).parent / "static"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=100)


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]
    summary: str


class SourceOut(BaseModel):
    id: str
    title: str
    section: str
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[ToolCallOut]
    sources: list[SourceOut]
    escalated: bool
    tickets: list[str]


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    embed_fn = (
        openai_embedder(settings.embedding_model, settings.openai_base_url) if settings.embeddings_enabled() else None
    )
    kb = KnowledgeBase.from_directory(settings.kb_dir, embed_fn=embed_fn)
    db = SupportDB(settings.db_path)
    provider = provider or build_provider(settings)
    agent = SupportAgent(provider, ToolExecutor(kb, db), SessionStore(settings.max_history_messages),
                         company=settings.company_name, max_tool_rounds=settings.max_tool_rounds)

    app = FastAPI(title="support-agent-rag", version=__version__,
                  description="AI customer-support agent with RAG, tool calling and human escalation.")
    app.state.agent, app.state.kb, app.state.db = agent, kb, db

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> dict[str, Any]:
        try:
            return agent.chat(req.message.strip(), req.session_id).to_dict()
        except Exception as exc:  # provider outage, bad key, rate limit...
            logging.getLogger(__name__).exception("chat failed")
            raise HTTPException(status_code=502, detail=f"LLM provider error: {type(exc).__name__}") from exc

    @app.delete("/chat/{session_id}", status_code=204)
    def reset(session_id: str) -> None:
        agent.sessions.reset(session_id)

    @app.get("/tickets")
    def tickets() -> list[dict[str, Any]]:
        return db.list_tickets()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": provider.name,
            "model": provider.model,
            "retrieval": kb.mode,
            "kb_documents": len(kb.documents),
            "kb_chunks": len(kb.chunks),
        }

    return app
