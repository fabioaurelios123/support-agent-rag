"""OpenAI and OpenAI-compatible providers (Ollama, vLLM, LM Studio, Together, ...)."""

from __future__ import annotations

import json
from typing import Any

from .base import LLMResponse, ToolCall


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in m["tool_calls"]
                ]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 api_key: str | None = None, name: str = "openai"):
        from openai import OpenAI  # optional dependency: `uv sync --extra openai`

        self.name = name
        self.model = model
        self._client = OpenAI(base_url=base_url or None, api_key=api_key)

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=to_openai_messages(system, messages),
            tools=[{"type": "function", "function": t} for t in tools],
            temperature=0.2,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return LLMResponse(text=(msg.content or "").strip(), tool_calls=calls)
