def test_lookup_existing_order(tools):
    out = tools.run("lookup_order", {"order_id": "nw-10482"})
    assert out["found"] is True
    order = out["order"]
    assert order["status"] == "shipped"
    assert order["carrier"] == "UPS"
    assert order["email"].startswith("an***@"), "customer email must be masked"


def test_lookup_accepts_order_number_without_dash(tools):
    assert tools.run("lookup_order", {"order_id": "NW10533"})["order"]["status"] == "processing"


def test_lookup_missing_order(tools):
    out = tools.run("lookup_order", {"order_id": "NW-99999"})
    assert out["found"] is False


def test_create_ticket_persists(tools, db):
    out = tools.run("create_support_ticket", {"email": "sam@example.com", "summary": "Lost package", "order_id": "NW-10590"})
    assert out["created"] is True and out["ticket_id"] == "TCK-0001"
    [ticket] = db.list_tickets()
    assert ticket["email"] == "sam@example.com" and ticket["order_id"] == "NW-10590"
    assert ticket["kind"] == "support" and ticket["status"] == "open"


def test_create_ticket_rejects_invalid_email(tools, db):
    out = tools.run("create_support_ticket", {"email": "not-an-email", "summary": "x"})
    assert out["created"] is False
    assert db.list_tickets() == []


def test_escalation_creates_high_priority_ticket(tools, db):
    out = tools.run("escalate_to_human", {"reason": "Customer is upset"})
    assert out["escalated"] is True and out["needs_email"] is True
    assert db.list_tickets()[0]["priority"] == "high"
    assert db.list_tickets()[0]["kind"] == "escalation"


def test_unknown_tool_and_bad_arguments_do_not_raise(tools):
    assert "error" in tools.run("delete_database", {})
    assert "error" in tools.run("lookup_order", {"wrong": 1})
