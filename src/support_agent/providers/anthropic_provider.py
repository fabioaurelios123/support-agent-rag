"""Anthropic Claude provider (Messages API with client-side tools)."""

from __future__ import annotations

import os
from typing import Any

from .base import LLMResponse, ToolCall

REFUSAL_TEXT = "I'm not able to help with that request. I can connect you with a member of our team instead."


def to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert neutral history to Anthropic messages.

    Assistant turns replay the provider's raw content blocks when available (this keeps
    thinking blocks intact, which the API requires inside a tool-use loop). Consecutive
    tool results are merged into a single user message, as the API expects.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            if m.get("raw") is not None and m.get("raw_provider") == "anthropic":
                content: Any = m["raw"]
            else:
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            out.append({"role": "assistant", "content": content})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 4096):
        import anthropic  # optional dependency: `uv sync --extra anthropic`

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()
        # Server-side refusal fallback (opt-out with ANTHROPIC_FALLBACKS=off).
        self._fallbacks = os.getenv("ANTHROPIC_FALLBACKS", "default").lower() != "off"

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        extra: dict[str, Any] = {}
        if self._fallbacks:
            extra = {
                "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
                "extra_body": {"fallbacks": "default"},
            }
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=to_anthropic_messages(messages),
            tools=[
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
            # Support chat is latency-sensitive; low effort keeps answers quick and concise.
            output_config={"effort": "low"},
            **extra,
        )
        if resp.stop_reason == "refusal":
            return LLMResponse(text=REFUSAL_TEXT)
        text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls, raw=resp.content)
