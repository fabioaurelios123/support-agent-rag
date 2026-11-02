"""SQLite store for (fake, seeded) orders and the support tickets the agent creates."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id           TEXT PRIMARY KEY,
    email              TEXT NOT NULL,
    customer_name      TEXT NOT NULL,
    status             TEXT NOT NULL,
    items              TEXT NOT NULL,   -- JSON list of {sku, name, size, qty, price}
    total              REAL NOT NULL,
    placed_at          TEXT NOT NULL,
    shipped_at         TEXT,
    carrier            TEXT,
    tracking_number    TEXT,
    estimated_delivery TEXT
);
CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT NOT NULL,
    summary     TEXT NOT NULL,
    priority    TEXT NOT NULL DEFAULT 'normal',
    kind        TEXT NOT NULL DEFAULT 'support',   -- support | escalation
    status      TEXT NOT NULL DEFAULT 'open',
    order_id    TEXT,
    session_id  TEXT,
    created_at  TEXT NOT NULL
);
"""

SEED_ORDERS: list[dict[str, Any]] = [
    {
        "order_id": "NW-10482", "email": "ana.ruiz@example.com", "customer_name": "Ana Ruiz",
        "status": "shipped", "total": 214.00, "placed_at": "2026-10-27",
        "shipped_at": "2026-10-28", "carrier": "UPS", "tracking_number": "1Z999AA10123456784",
        "estimated_delivery": "2026-11-03",
        "items": [{"sku": "JKT-ALP-M", "name": "Alpine Shell Jacket", "size": "M", "qty": 1, "price": 189.00},
                  {"sku": "SCK-MER-L", "name": "Merino Hiking Socks", "size": "L", "qty": 1, "price": 25.00}],
    },
    {
        "order_id": "NW-10511", "email": "ana.ruiz@example.com", "customer_name": "Ana Ruiz",
        "status": "delivered", "total": 129.00, "placed_at": "2026-09-30",
        "shipped_at": "2026-10-01", "carrier": "USPS", "tracking_number": "9400111899223344556677",
        "estimated_delivery": "2026-10-04",
        "items": [{"sku": "PCK-TRL-28", "name": "Trailhead 28L Daypack", "size": "One size", "qty": 1, "price": 129.00}],
    },
    {
        "order_id": "NW-10533", "email": "marcus.lee@example.com", "customer_name": "Marcus Lee",
        "status": "processing", "total": 74.50, "placed_at": "2026-11-01",
        "shipped_at": None, "carrier": None, "tracking_number": None, "estimated_delivery": "2026-11-07",
        "items": [{"sku": "TEE-MER-L", "name": "Merino Base Layer Tee", "size": "L", "qty": 1, "price": 74.50}],
    },
    {
        "order_id": "NW-10547", "email": "priya.n@example.com", "customer_name": "Priya Natarajan",
        "status": "backordered", "total": 248.00, "placed_at": "2026-10-29",
        "shipped_at": None, "carrier": None, "tracking_number": None, "estimated_delivery": "2026-11-14",
        "items": [{"sku": "BOT-RDG-8W", "name": "Ridgeline Hiking Boot (Wide)", "size": "US 8 2E", "qty": 1, "price": 248.00}],
    },
    {
        "order_id": "NW-10562", "email": "tom.becker@example.com", "customer_name": "Tom Becker",
        "status": "delivered", "total": 329.00, "placed_at": "2026-10-12",
        "shipped_at": "2026-10-13", "carrier": "DHL Express", "tracking_number": "JD014600006281234567",
        "estimated_delivery": "2026-10-22",
        "items": [{"sku": "DWN-SUM-L", "name": "Summit Down Parka", "size": "L", "qty": 1, "price": 329.00}],
    },
    {
        "order_id": "NW-10578", "email": "jin.park@example.com", "customer_name": "Jin Park",
        "status": "cancelled", "total": 59.00, "placed_at": "2026-10-20",
        "shipped_at": None, "carrier": None, "tracking_number": None, "estimated_delivery": None,
        "items": [{"sku": "HAT-WOL-OS", "name": "Wool Trail Beanie", "size": "One size", "qty": 2, "price": 29.50}],
    },
    {
        "order_id": "NW-10590", "email": "lucia.costa@example.com", "customer_name": "Lucia Costa",
        "status": "shipped", "total": 162.00, "placed_at": "2026-10-30",
        "shipped_at": "2026-10-31", "carrier": "DHL Express", "tracking_number": "JD014600006289876543",
        "estimated_delivery": "2026-11-08",
        "items": [{"sku": "PNT-HKR-32", "name": "Switchback Hiking Pants", "size": "30 / Regular", "qty": 2, "price": 81.00}],
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SupportDB:
    """Thin, thread-safe wrapper. In production this is where a Shopify/ERP/helpdesk client goes."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
        self.seed()

    def seed(self) -> None:
        with self._lock, self._conn:
            for o in SEED_ORDERS:
                self._conn.execute(
                    """INSERT OR IGNORE INTO orders VALUES
                       (:order_id, :email, :customer_name, :status, :items, :total,
                        :placed_at, :shipped_at, :carrier, :tracking_number, :estimated_delivery)""",
                    {**o, "items": json.dumps(o["items"])},
                )

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id.strip().upper(),)
            ).fetchone()
        if row is None:
            return None
        order = dict(row)
        order["items"] = json.loads(order["items"])
        return order

    def create_ticket(
        self,
        email: str,
        summary: str,
        *,
        priority: str = "normal",
        kind: str = "support",
        order_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """INSERT INTO tickets (email, summary, priority, kind, order_id, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (email, summary, priority, kind, order_id, session_id, _now()),
            )
            row = self._conn.execute("SELECT * FROM tickets WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _ticket(row)

    def list_tickets(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM tickets ORDER BY id DESC").fetchall()
        return [_ticket(r) for r in rows]


def _ticket(row: sqlite3.Row) -> dict[str, Any]:
    t = dict(row)
    t["ticket_id"] = f"TCK-{t['id']:04d}"
    return t
