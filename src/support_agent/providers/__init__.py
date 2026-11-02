"""LLM provider factory."""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider, LLMResponse, ToolCall
from .mock import MockProvider

__all__ = ["LLMProvider", "LLMResponse", "ToolCall", "MockProvider", "build_provider"]


def build_provider(settings: Settings) -> LLMProvider:
    name = settings.resolved_provider()
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=settings.anthropic_model)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=settings.openai_model, base_url=settings.openai_base_url)
    if name == "ollama":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=settings.ollama_model, base_url=settings.ollama_base_url,
                              api_key="ollama", name="ollama")
    raise ValueError(f"unknown LLM_PROVIDER '{name}' (expected mock, anthropic, openai or ollama)")
