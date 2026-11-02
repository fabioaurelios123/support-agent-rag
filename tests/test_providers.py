"""Provider adapters translate the neutral history correctly (no network calls)."""

import json

from support_agent.config import Settings
from support_agent.providers import MockProvider, ToolCall, build_provider
from support_agent.providers.anthropic_provider import to_anthropic_messages
from support_agent.providers.openai_provider import to_openai_messages

HISTORY = [
    {"role": "user", "content": "Where is NW-10482?"},
    {"role": "assistant", "content": "Let me check.", "tool_calls": [
        ToolCall("t1", "lookup_order", {"order_id": "NW-10482"}),
        ToolCall("t2", "search_knowledge_base", {"query": "tracking"}),
    ]},
    {"role": "tool", "tool_call_id": "t1", "name": "lookup_order", "content": json.dumps({"found": True})},
    {"role": "tool", "tool_call_id": "t2", "name": "search_knowledge_base", "content": "{}"},
]


def test_anthropic_merges_parallel_tool_results_into_one_user_message():
    msgs = to_anthropic_messages(HISTORY)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["t1", "t2"]


def test_anthropic_replays_raw_content_blocks():
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "hi"}]
    msgs = to_anthropic_messages([{"role": "user", "content": "x"},
                                  {"role": "assistant", "content": "hi", "tool_calls": [], "raw": raw,
                                   "raw_provider": "anthropic"}])
    assert msgs[1]["content"] is raw


def test_openai_format():
    msgs = to_openai_messages("sys", HISTORY)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "lookup_order", "arguments": '{"order_id": "NW-10482"}'}
    assert [m["role"] for m in msgs[3:]] == ["tool", "tool"]


def test_auto_provider_falls_back_to_mock_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(build_provider(Settings(llm_provider="auto")), MockProvider)


def test_anthropic_provider_parses_tool_use(monkeypatch):
    from types import SimpleNamespace

    from support_agent.providers.anthropic_provider import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider()
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop_reason="tool_use", content=[
            SimpleNamespace(type="text", text="Checking your order."),
            SimpleNamespace(type="tool_use", id="toolu_1", name="lookup_order", input={"order_id": "NW-10482"}),
        ])

    monkeypatch.setattr(provider._client.messages, "create", fake_create)
    resp = provider.complete("sys", [{"role": "user", "content": "Where is NW-10482?"}],
                             [{"name": "lookup_order", "description": "d", "parameters": {"type": "object"}}])
    assert captured["model"] == "claude-sonnet-5-5"
    assert captured["tools"][0]["input_schema"] == {"type": "object"}
    assert resp.text == "Checking your order."
    assert resp.tool_calls[0].arguments == {"order_id": "NW-10482"}
    assert resp.raw is not None
