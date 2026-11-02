"""Deterministic, rule-based provider.

It speaks the same protocol as the real LLM providers (it returns tool calls, receives tool
results, then writes a final answer), so the whole agent loop, the API and the UI can be
demoed and tested with no API key and no network. It is intentionally simple: intent
detection with keyword rules and extractive answers from the retrieved passages.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from ..kb import tokenize
from ..tools import EMAIL_RE, ORDER_RE
from .base import LLMResponse, ToolCall

HUMAN_RE = re.compile(r"\b(human|real person|someone|representative|manager|supervisor|live agent|talk to an agent|speak to an agent)\b", re.I)
TICKET_RE = re.compile(
    r"\b(ticket|claim|replace|replacement|refund me|broken|broke|defect\w*|damaged|torn|ripped|leak\w*|"
    r"faulty|lost|missing|never arrived|wrong item|return label|open a case)\b", re.I)
POLICY_RE = re.compile(
    r"\b(return\w*|exchang\w*|refund\w*|warrant\w*|size|sizing|fit|cancel\w*|change|address|pay\w*|klarna|"
    r"price|promo\w*|code|gift|reward\w*|points|wash\w*|care|clean\w*|zipper|repair\w*|tax|international|"
    r"canada|uk|europe|duties|po box|military|boots?|jacket|inseam|chest)\b", re.I)
DEFECT_RE = re.compile(r"\b(broke\w*|defect\w*|faulty|ripped|torn|leak\w*|zipper|seams?|buckle|stitching|membrane)\b", re.I)
SHIPPING_RE = re.compile(r"\b(ship\w*|deliver\w*|arriv\w*|tracking|carrier|overnight|expedited)\b", re.I)


def _fmt_date(value: str | None) -> str:
    if not value:
        return "TBD"
    try:
        d = date.fromisoformat(value[:10])
    except ValueError:
        return value
    return d.strftime("%b %-d, %Y")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


class MockProvider:
    name = "mock"
    model = "rule-based"

    # ---------------------------------------------------------------- protocol
    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user_text = messages[last_user]["content"]
        turn = messages[last_user + 1 :]
        calls_made = [tc for m in turn if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
        results = {m["tool_call_id"]: json.loads(m["content"]) for m in turn if m["role"] == "tool"}

        plan = self._plan(user_text, messages[:last_user], calls_made, results)
        if len(plan) > len(calls_made):
            name, args = plan[len(calls_made)]
            return LLMResponse(text="", tool_calls=[ToolCall(f"mock_{len(calls_made) + 1}", name, args)])
        return LLMResponse(text=self._compose(user_text, messages[:last_user], calls_made, results))

    # ---------------------------------------------------------------- planning
    def _plan(self, text: str, history: list[dict[str, Any]], calls_made: list[ToolCall],
              results: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        email = self._email(text, history)
        order_ids = [self._norm_order(o) for o in ORDER_RE.findall(text)]
        wants_human = bool(HUMAN_RE.search(text))
        wants_ticket = bool(TICKET_RE.search(text))

        plan: list[tuple[str, dict[str, Any]]] = [("lookup_order", {"order_id": o}) for o in order_ids]

        query = EMAIL_RE.sub(" ", ORDER_RE.sub(" ", text)).strip()
        about_policy = bool(POLICY_RE.search(text)) or (bool(SHIPPING_RE.search(text)) and not order_ids)
        needs_order_id = bool(re.search(r"\border\b", text, re.I)) and not order_ids and not about_policy
        if not wants_human and not needs_order_id and tokenize(query) and (
            about_policy or not (order_ids or wants_ticket)
        ):
            # Query rewriting, as an LLM would do: a defect report is a warranty question.
            defect = DEFECT_RE.findall(text)
            if wants_ticket and defect:
                query = "warranty claim defect " + " ".join(dict.fromkeys(w.lower() for w in defect))
            plan.append(("search_knowledge_base", {"query": query}))

        # Follow-up turn that only supplies the email we asked for: act on the previous request.
        subject = text
        if not wants_ticket and EMAIL_RE.search(text) and not tokenize(query):
            prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            if TICKET_RE.search(prev):
                wants_ticket, subject = True, prev

        if wants_ticket and email and not wants_human:
            order = order_ids[0] if order_ids else self._order_from_history(history)
            args = {"email": email, "summary": self._summary(subject), "priority": "normal"}
            if order:
                args["order_id"] = order
            plan.append(("create_support_ticket", args))

        if wants_human:
            plan.append(("escalate_to_human", {"reason": self._summary(text), **({"email": email} if email else {})}))

        # Escalate when retrieval came back weak and nothing else handled the request.
        if len(calls_made) >= len(plan) and not wants_human and not (wants_ticket and email):
            for tc in calls_made:
                r = results.get(tc.id, {})
                if tc.name == "search_knowledge_base" and r.get("confidence") == "low" and not order_ids:
                    reason = f"Assistant could not answer confidently: {self._summary(text)}"
                    plan.append(("escalate_to_human", {"reason": reason, **({"email": email} if email else {})}))
                    break
        return plan

    # ---------------------------------------------------------------- answering
    def _compose(self, text: str, history: list[dict[str, Any]], calls: list[ToolCall], results: dict[str, Any]) -> str:
        parts: list[str] = []
        escalated = any(tc.name == "escalate_to_human" for tc in calls)
        ticket_made = any(tc.name == "create_support_ticket" and results[tc.id].get("created") for tc in calls)
        for tc in calls:
            r = results.get(tc.id, {})
            if tc.name == "lookup_order":
                parts.append(self._describe_order(r))
            elif tc.name == "search_knowledge_base":
                if r.get("confidence") == "high":
                    parts.append(self._extract_answer(r))
                elif not escalated:
                    parts.append("I couldn't find a reliable answer to that in our help center.")
            elif tc.name == "create_support_ticket":
                if r.get("created"):
                    parts.append(f"I've opened ticket **{r['ticket_id']}** for our support team. {r['sla']}")
                else:
                    parts.append(r.get("error", "I couldn't open a ticket."))
            elif tc.name == "escalate_to_human":
                lead = ("Of course. I've" if HUMAN_RE.search(text)
                        else "I don't want to guess on this one, so I've")
                msg = f"{lead} passed the conversation to a human agent (reference **{r.get('ticket_id')}**)."
                if r.get("needs_email"):
                    msg += " What email address can they use to follow up?"
                parts.append(msg)

        if TICKET_RE.search(text) and not ticket_made and not escalated and not self._email(text, history):
            parts.append("If you'd like me to open a ticket for this, just share the email address used on the order.")
        if not parts:
            if "order" in text.lower() and not ORDER_RE.search(text):
                return "Happy to check on that. What's your order number? It looks like NW-12345."
            return ("Hi! I'm the Northwind Outfitters assistant. I can answer questions about shipping, returns, "
                    "warranty and sizing, look up an order (NW-12345), or open a support ticket for you.")
        return "\n\n".join(parts)

    def _extract_answer(self, result: dict[str, Any]) -> str:
        """Extractive answer: the sentences that best overlap the query, grouped by source."""
        q = set(tokenize(result["query"]))
        hits = [h for h in result["results"][:3] if h["score"] >= 0.6 * result["results"][0]["score"]]
        cands = []  # (overlap, passage_rank, position, sentence)
        for rank, hit in enumerate(hits):
            for pos, sent in enumerate(_sentences(hit["text"])):
                cands.append((len(q & set(tokenize(sent))), rank, pos, sent))
        best = max((c[0] for c in cands), default=0)
        if best == 0:
            return f"{_sentences(hits[0]['text'])[0]} [{hits[0]['source_id']}]"
        chosen = sorted((c for c in cands if c[0] == best), key=lambda c: (c[1], c[2]))
        if len(chosen) < 2:  # add one supporting sentence from the same passage for context
            extra = [c for c in cands if c[1] == chosen[0][1] and c not in chosen and c[0] >= max(1, best - 1)]
            chosen += extra[:1]
        per_passage: dict[int, list[tuple[int, str]]] = {}
        limit = 3 if best > 1 else 2  # weak lexical overlap: stay short rather than ramble
        for _, rank, pos, sent in chosen:
            if len(per_passage.get(rank, [])) < 2 and sum(map(len, per_passage.values())) < limit:
                per_passage.setdefault(rank, []).append((pos, sent))
        return " ".join(
            " ".join(sent for _, sent in sorted(sents)) + f" [{hits[rank]['source_id']}]"
            for rank, sents in sorted(per_passage.items())
        )

    @staticmethod
    def _describe_order(r: dict[str, Any]) -> str:
        if not r.get("found"):
            return (f"I couldn't find an order with number **{r.get('order_id')}**. "
                    "Could you double-check it? It should look like NW-12345.")
        o = r["order"]
        items = ", ".join(f"{i['name']} ({i['size']})" + (f" x{i['qty']}" if i["qty"] > 1 else "") for i in o["items"])
        oid, status = o["order_id"], o["status"]
        if status == "shipped":
            return (f"Order **{oid}** ({items}) shipped on {_fmt_date(o['shipped_at'])} via {o['carrier']}, "
                    f"tracking number {o['tracking_number']}. Estimated delivery: **{_fmt_date(o['estimated_delivery'])}**.")
        if status == "delivered":
            return f"Order **{oid}** ({items}) was delivered on {_fmt_date(o['estimated_delivery'])} by {o['carrier']}."
        if status == "processing":
            return (f"Order **{oid}** ({items}) is being prepared at our warehouse and should ship soon. "
                    f"Estimated delivery: **{_fmt_date(o['estimated_delivery'])}**.")
        if status == "backordered":
            return (f"Order **{oid}** ({items}) is backordered. The expected ship date puts delivery around "
                    f"**{_fmt_date(o['estimated_delivery'])}**, and you can cancel a backordered item at any time.")
        return f"Order **{oid}** ({items}) is **{status}**."

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _email(text: str, history: list[dict[str, Any]]) -> str | None:
        for content in [text] + [m["content"] for m in reversed(history) if m["role"] == "user"]:
            if match := EMAIL_RE.search(content):
                return match.group(0)
        return None

    def _order_from_history(self, history: list[dict[str, Any]]) -> str | None:
        for m in reversed(history):
            if m["role"] == "user" and (match := ORDER_RE.search(m["content"])):
                return self._norm_order(match.group(0))
        return None

    @staticmethod
    def _norm_order(value: str) -> str:
        digits = re.sub(r"\D", "", value)
        return f"NW-{digits}"

    @staticmethod
    def _summary(text: str) -> str:
        clean = re.sub(r"\s+", " ", EMAIL_RE.sub("", text)).strip(" .,")
        return clean[:200]
