"""End-to-end tests through the HTTP API with the deterministic mock provider."""


def chat(client, message, session_id=None):
    res = client.post("/chat", json={"message": message, "session_id": session_id})
    assert res.status_code == 200, res.text
    return res.json()


def tool_names(data):
    return [t["name"] for t in data["tool_calls"]]


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["provider"] == "mock" and h["retrieval"] == "bm25"
    assert h["kb_documents"] == 9


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "Northwind" in res.text


def test_policy_question_is_answered_with_citation(client):
    data = chat(client, "How long does standard shipping take to Canada?")
    assert tool_names(data) == ["search_knowledge_base"]
    assert "5 to 8 business days" in data["reply"]
    assert [s["id"] for s in data["sources"]] == ["shipping#international-shipping"]
    assert "[shipping#international-shipping]" in data["reply"]
    assert data["escalated"] is False


def test_order_status_uses_lookup_tool(client):
    data = chat(client, "Where is my order NW-10482?")
    assert tool_names(data) == ["lookup_order"]
    assert data["tool_calls"][0]["arguments"] == {"order_id": "NW-10482"}
    assert "1Z999AA10123456784" in data["reply"]


def test_unknown_order(client):
    data = chat(client, "status of NW-00001 please")
    assert "couldn't find" in data["reply"]


def test_asks_for_order_number_when_missing(client):
    data = chat(client, "Where is my order?")
    assert data["tool_calls"] == []
    assert "order number" in data["reply"]


def test_defect_report_searches_warranty_and_opens_ticket(client):
    data = chat(client, "The zipper on my jacket broke. My email is ana.ruiz@example.com")
    assert tool_names(data) == ["search_knowledge_base", "create_support_ticket"]
    assert data["sources"][0]["id"].startswith("warranty#")
    assert data["tickets"] == ["TCK-0001"]
    tickets = client.get("/tickets").json()
    assert tickets[0]["ticket_id"] == "TCK-0001" and tickets[0]["email"] == "ana.ruiz@example.com"


def test_conversation_memory_across_turns(client):
    first = chat(client, "Where is my order NW-10482?")
    sid = first["session_id"]
    second = chat(client, "The zipper broke on day two, can I get a replacement?", sid)
    assert "create_support_ticket" not in tool_names(second)
    assert "email" in second["reply"]
    # The email arrives in a separate turn; the agent remembers the request and the order.
    third = chat(client, "ana.ruiz@example.com", sid)
    assert tool_names(third) == ["create_support_ticket"]
    ticket = client.get("/tickets").json()[0]
    assert ticket["order_id"] == "NW-10482" and "zipper" in ticket["summary"].lower()
    assert ticket["session_id"] == sid


def test_sessions_are_isolated(client):
    chat(client, "my email is ana.ruiz@example.com", "a")
    other = chat(client, "My jacket zipper broke, please open a ticket", "b")
    assert "create_support_ticket" not in tool_names(other)


def test_explicit_request_for_human_escalates(client):
    data = chat(client, "This is ridiculous, I want to talk to a human")
    assert tool_names(data) == ["escalate_to_human"]
    assert data["escalated"] is True
    assert client.get("/tickets").json()[0]["kind"] == "escalation"


def test_unanswerable_question_escalates_instead_of_guessing(client):
    data = chat(client, "Do you ship to Brazil?")
    assert tool_names(data) == ["search_knowledge_base", "escalate_to_human"]
    assert data["escalated"] is True and data["sources"] == []


def test_reset_session(client):
    data = chat(client, "my email is ana.ruiz@example.com")
    assert client.delete(f"/chat/{data['session_id']}").status_code == 204


def test_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
