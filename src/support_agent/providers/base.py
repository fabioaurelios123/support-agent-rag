"""Provider-neutral message format shared by every LLM backend.

History is a list of dicts:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall...], "raw": <provider blocks>}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str (JSON)}

Each provider translates this to its own wire format, so the agent loop, memory and tests
never depend on a specific vendor SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-native content, replayed verbatim on the next request


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...
