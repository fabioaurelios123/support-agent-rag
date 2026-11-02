"""Tools the agent can call. Schemas are provider-neutral JSON Schema."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .db import SupportDB
from .kb import KnowledgeBase

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ORDER_RE = re.compile(r"\bNW-?\d{5}\b", re.I)

# The best passage must cover at least 40% of the (IDF-weighted) query terms to count as a
# confident match. Coverage is far more stable than raw BM25 scores, which depend on query length.
MIN_COVERAGE = 0.4

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the store's help-center articles (shipping, returns, exchanges, warranty, "
            "sizing, payments, order changes, product care, rewards and gift cards). Use it for "
            "any policy or product question. Returns passages with a source id to cite."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query in English."},
                "top_k": {"type": "integer", "description": "Number of passages (1-6).", "default": 4},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "Look up an order by its order number (format NW-12345). Returns status, items, "
            "carrier, tracking number and estimated delivery."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order number, e.g. NW-10482."}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Open a ticket for the support team when the customer needs an action the assistant "
            "cannot perform (warranty claim, return label, refund, address change, lost package). "
            "Requires the customer's email address."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email address."},
                "summary": {"type": "string", "description": "One or two sentences describing the request."},
                "order_id": {"type": "string", "description": "Related order number, if any."},
                "priority": {"type": "string", "enum": ["low", "normal", "high"], "default": "normal"},
            },
            "required": ["email", "summary"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the conversation to a human agent. Use when the customer asks for a person, is "
            "upset, or when the knowledge base does not answer the question with confidence. "
            "Never guess a policy: escalate instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why a human is needed."},
                "email": {"type": "string", "description": "Customer email, if known."},
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolContext:
    session_id: str | None = None


class ToolExecutor:
    def __init__(self, kb: KnowledgeBase, db: SupportDB):
        self.kb = kb
        self.db = db

    def run(self, name: str, args: dict[str, Any], ctx: ToolContext | None = None) -> dict[str, Any]:
        ctx = ctx or ToolContext()
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return handler(ctx=ctx, **args)
        except TypeError as exc:  # bad / missing arguments from the model
            return {"error": f"invalid arguments for {name}: {exc}"}

    def _tool_search_knowledge_base(self, ctx: ToolContext, query: str, top_k: int = 4) -> dict[str, Any]:
        results = self.kb.search(query, top_k=max(1, min(int(top_k), 6)))
        confident = bool(results) and results[0].coverage >= MIN_COVERAGE
        return {
            "query": query,
            "confidence": "high" if confident else "low",
            "note": None if confident else (
                "No passage matches strongly. Do not guess: ask a clarifying question or escalate."
            ),
            "results": [
                {
                    "source_id": r.chunk.id,
                    "title": r.chunk.title,
                    "section": r.chunk.section,
                    "score": r.raw_score,
                    "text": r.chunk.text,
                }
                for r in results
            ],
        }

    def _tool_lookup_order(self, ctx: ToolContext, order_id: str) -> dict[str, Any]:
        normalized = order_id.strip().upper()
        if re.fullmatch(r"NW\d{5}", normalized):
            normalized = f"NW-{normalized[2:]}"
        order = self.db.get_order(normalized)
        if order is None:
            return {"found": False, "order_id": normalized,
                    "message": "No order with this number. Ask the customer to double-check it."}
        # Never leak the full email back to the chat; a mask is enough to confirm identity.
        user, _, domain = order["email"].partition("@")
        order["email"] = f"{user[:2]}***@{domain}"
        return {"found": True, "order": order}

    def _tool_create_support_ticket(
        self,
        ctx: ToolContext,
        email: str,
        summary: str,
        order_id: str | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        if not EMAIL_RE.fullmatch(email.strip()):
            return {"created": False, "error": "A valid email address is required to open a ticket."}
        if priority not in {"low", "normal", "high"}:
            priority = "normal"
        ticket = self.db.create_ticket(
            email.strip(), summary.strip(), priority=priority, order_id=order_id, session_id=ctx.session_id
        )
        return {"created": True, "ticket_id": ticket["ticket_id"], "priority": priority,
                "sla": "A support specialist replies within 1 business day."}

    def _tool_escalate_to_human(self, ctx: ToolContext, reason: str, email: str | None = None) -> dict[str, Any]:
        contact = email.strip() if email and EMAIL_RE.fullmatch(email.strip()) else "unknown"
        ticket = self.db.create_ticket(
            contact, f"[Escalation] {reason.strip()}", priority="high", kind="escalation",
            session_id=ctx.session_id,
        )
        return {"escalated": True, "ticket_id": ticket["ticket_id"],
                "message": "A human agent has been notified and will join this conversation.",
                "needs_email": contact == "unknown"}
