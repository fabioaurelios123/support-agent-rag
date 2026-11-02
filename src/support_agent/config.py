"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    llm_provider: str = "auto"  # auto | mock | anthropic | openai | ollama
    anthropic_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_base_url: str = ""
    embeddings: str = "auto"  # auto | none | openai
    embedding_model: str = "text-embedding-3-small"
    kb_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge_base")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "support.db")
    company_name: str = "Northwind Outfitters"
    max_tool_rounds: int = 6
    max_history_messages: int = 40

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        s = cls()
        s.llm_provider = _env("LLM_PROVIDER", s.llm_provider).lower()
        s.anthropic_model = _env("ANTHROPIC_MODEL", s.anthropic_model)
        s.openai_model = _env("OPENAI_MODEL", s.openai_model)
        s.ollama_model = _env("OLLAMA_MODEL", s.ollama_model)
        s.ollama_base_url = _env("OLLAMA_BASE_URL", s.ollama_base_url)
        s.openai_base_url = _env("OPENAI_BASE_URL", s.openai_base_url)
        s.embeddings = _env("EMBEDDINGS", s.embeddings).lower()
        s.embedding_model = _env("EMBEDDING_MODEL", s.embedding_model)
        if kb := _env("KB_DIR"):
            s.kb_dir = Path(kb)
        if db := _env("DB_PATH"):
            s.db_path = Path(db)
        return s

    def resolved_provider(self) -> str:
        """`auto` picks the first provider with credentials, falling back to the mock."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "mock"

    def embeddings_enabled(self) -> bool:
        if self.embeddings == "openai":
            return True
        return self.embeddings == "auto" and bool(os.getenv("OPENAI_API_KEY"))
