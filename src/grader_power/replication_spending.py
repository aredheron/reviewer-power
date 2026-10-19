"""Transactional flat request ledger for the concurrent replication pipeline."""

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from grader_power.spending import SpendingLimit


class RequestLedger:
    """Preserve every reservation/settlement without rewriting growing JSON logs.

    SQLite transactions serialize admissions, including across processes. WAL
    with FULL synchronous commits makes paid-request intent durable before send.
    Unknown charges keep their reservations. No deletion/release API exists.
    """

    def __init__(self, path: Path, *, cap: int, scope: str) -> None:
        if type(cap) is not int or cap <= 0 or not scope:
            raise ValueError("Positive cap and frozen scope required")
        self.path, self.cap, self.scope = path, cap, scope

    def _change(self, action: str, amounts: Mapping[str, int]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS scope (id INTEGER PRIMARY KEY CHECK(id=1), cap INTEGER, identity TEXT, overrun INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries (identity TEXT PRIMARY KEY, reserved INTEGER NOT NULL, settled INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, at TEXT DEFAULT CURRENT_TIMESTAMP, action TEXT, amounts TEXT)"
            )
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR IGNORE INTO scope VALUES (1, ?, ?, 0)",
                (self.cap, self.scope),
            )
            cap, scope, overrun = db.execute(
                "SELECT cap, identity, overrun FROM scope"
            ).fetchone()
            if (cap, scope) != (self.cap, self.scope):
                raise ValueError("Spending scope/ceiling differs from frozen ledger")
            for identity, amount in amounts.items():
                if not identity or type(amount) is not int or amount < 0:
                    raise ValueError(
                        "Stable identity and integer nonnegative amount required"
                    )
                previous = db.execute(
                    "SELECT reserved, settled FROM entries WHERE identity=?",
                    (identity,),
                ).fetchone()
                if action == "reserve":
                    if previous is not None and previous[0] != amount:
                        raise ValueError("Reservation changed for existing request")
                    db.execute(
                        "INSERT OR IGNORE INTO entries VALUES (?, ?, NULL)",
                        (identity, amount),
                    )
                elif action == "settle":
                    if previous is None or (
                        previous[1] is not None and previous[1] != amount
                    ):
                        raise ValueError("Unreserved or revised settlement")
                    db.execute(
                        "UPDATE entries SET settled=? WHERE identity=?",
                        (amount, identity),
                    )
                    overrun = overrun or amount > previous[0]
                else:
                    raise ValueError("Unknown ledger action")
            spent, outstanding = db.execute(
                "SELECT COALESCE(SUM(settled),0), COALESCE(SUM(CASE WHEN settled IS NULL THEN reserved ELSE 0 END),0) FROM entries"
            ).fetchone()
            if action == "reserve" and (overrun or spent + outstanding > cap):
                raise SpendingLimit(
                    "Recorded plus outstanding plus new reservation exceeds ceiling"
                )
            db.execute("UPDATE scope SET overrun=? WHERE id=1", (int(overrun),))
            if amounts:
                db.execute(
                    "INSERT INTO events(action, amounts) VALUES (?, ?)",
                    (action, json.dumps(dict(amounts), sort_keys=True)),
                )
            return {
                "cap": cap,
                "scope": scope,
                "spent": spent,
                "outstanding": outstanding,
                "committed": spent + outstanding,
                "overrun": bool(overrun),
            }

    def reserve(self, amounts: Mapping[str, int]) -> dict[str, Any]:
        """Atomically reserve requests before intent or HTTP dispatch."""
        return self._change("reserve", amounts)

    def settle(self, amounts: Mapping[str, int]) -> dict[str, Any]:
        """Reconcile complete usage once; retain any overrun as an admission block."""
        return self._change("settle", amounts)

    def snapshot(self) -> dict[str, Any]:
        """Return exact totals; all individual entries and events remain in SQLite."""
        return self._change("settle", {})
