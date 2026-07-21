"""Tiny SQLite persistence for applicants.

Replaces the in-memory dict so applicants survive a restart and can be
listed. Uses the stdlib sqlite3 (no extra dependency) behind a small
function API that the rest of the app calls -- easy to swap for Postgres
later. Profiles are stored as JSON.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models import ApplicantProfile

DB_PATH = Path(__file__).resolve().parent.parent / "rentready.db"


class SlotConflict(Exception):
    """Raised when an atomic booking loses a race for a slot (agent+start
    already booked). The API turns this into a 409."""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applicants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                chunks_indexed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        # Production telemetry: one row per served request (online monitoring).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prod_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                applicant_id TEXT,
                latency_ms REAL,
                source TEXT,
                faithfulness_violations INTEGER DEFAULT 0,
                meta_json TEXT
            )
            """
        )
        # User feedback (thumbs up/down) on a served output.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                applicant_id TEXT,
                target TEXT NOT NULL,
                item_id TEXT,
                rating TEXT NOT NULL,
                comment TEXT
            )
            """
        )
        # Reviewer decisions on an applicant (approve/decline/waitlist/info).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                applicant_id TEXT NOT NULL,
                action TEXT NOT NULL,
                note TEXT,
                reviewer TEXT
            )
            """
        )
        # Tour Scheduler: leasing agents who run tours. `areas` is JSON text
        # (list of neighborhood names); [] means the agent covers all areas.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tour_agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                areas_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Recurring weekly availability windows (weekday 0=Mon..6=Sun, local HH:MM).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS availability_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                weekday INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL
            )
            """
        )
        # Booked tours. A UNIQUE index on (agent_id, start) for status='booked'
        # is the last line of defense against a double-book race.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tour_bookings (
                id TEXT PRIMARY KEY,
                property_id TEXT NOT NULL,
                property_name TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL DEFAULT 30,
                agent_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                prospect_name TEXT NOT NULL,
                prospect_email TEXT DEFAULT '',
                prospect_phone TEXT DEFAULT '',
                applicant_id TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'booked',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_booked_agent_start "
            "ON tour_bookings (agent_id, start) WHERE status = 'booked'"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(
    endpoint: str,
    applicant_id: str = None,
    latency_ms: float = None,
    source: str = None,
    faithfulness_violations: int = 0,
    meta: dict = None,
) -> None:
    """Record one served request for online monitoring."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO prod_events (ts, endpoint, applicant_id, latency_ms, "
            "source, faithfulness_violations, meta_json) VALUES (?,?,?,?,?,?,?)",
            (
                _now(),
                endpoint,
                applicant_id,
                latency_ms,
                source,
                int(faithfulness_violations or 0),
                json.dumps(meta or {}),
            ),
        )


def recent_events(limit: int = 500) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, endpoint, applicant_id, latency_ms, source, "
            "faithfulness_violations, meta_json FROM prod_events "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["meta"] = json.loads(d.pop("meta_json") or "{}")
        out.append(d)
    return out


def save_feedback(
    applicant_id: str,
    target: str,
    rating: str,
    item_id: str = None,
    comment: str = None,
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO feedback (ts, applicant_id, target, item_id, rating, "
            "comment) VALUES (?,?,?,?,?,?)",
            (_now(), applicant_id, target, item_id, rating, comment),
        )


def recent_feedback(limit: int = 500) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, applicant_id, target, item_id, rating, comment "
            "FROM feedback ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_decision(
    applicant_id: str, action: str, note: str = None, reviewer: str = None
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO decisions (ts, applicant_id, action, note, reviewer) "
            "VALUES (?,?,?,?,?)",
            (_now(), applicant_id, action, note, reviewer),
        )


def decisions_for(applicant_id: str) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, action, note, reviewer FROM decisions "
            "WHERE applicant_id = ? ORDER BY id DESC",
            (applicant_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def latest_statuses() -> dict:
    """Map applicant_id -> most recent decision action (its current status)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT applicant_id, action FROM decisions d "
            "WHERE id = (SELECT MAX(id) FROM decisions WHERE applicant_id = d.applicant_id)"
        ).fetchall()
    return {r["applicant_id"]: r["action"] for r in rows}


def save_applicant(
    applicant_id: str, profile: ApplicantProfile, chunks_indexed: int
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO applicants "
            "(id, name, profile_json, chunks_indexed, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                applicant_id,
                profile.name,
                profile.model_dump_json(),
                chunks_indexed,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_profile(applicant_id: str):
    with _conn() as conn:
        row = conn.execute(
            "SELECT profile_json FROM applicants WHERE id = ?", (applicant_id,)
        ).fetchone()
    if row is None:
        return None
    return ApplicantProfile.model_validate_json(row["profile_json"])


def list_applicants() -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, name, chunks_indexed, created_at "
            "FROM applicants ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_applicant(applicant_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM applicants WHERE id = ?", (applicant_id,)
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Tour Scheduler persistence
# ---------------------------------------------------------------------------


def save_agent(agent: dict) -> None:
    """Insert/replace a tour agent. ``areas`` stored as JSON text."""
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tour_agents (id, name, role, areas_json, "
            "created_at) VALUES (?,?,?,?,?)",
            (
                agent["id"],
                agent["name"],
                agent.get("role", "Leasing Consultant"),
                json.dumps(agent.get("areas") or []),
                _now(),
            ),
        )


def list_agents() -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, name, role, areas_json FROM tour_agents ORDER BY id"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["areas"] = json.loads(d.pop("areas_json") or "[]")
        out.append(d)
    return out


def save_window(agent_id: str, weekday: int, start: str, end: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO availability_windows (agent_id, weekday, start_time, "
            "end_time) VALUES (?,?,?,?)",
            (agent_id, int(weekday), start, end),
        )


def list_windows(agent_id: str = None) -> list:
    """Recurring weekly windows, normalized to the engine's dict shape
    ({agent_id, weekday, start, end})."""
    with _conn() as conn:
        if agent_id is not None:
            rows = conn.execute(
                "SELECT agent_id, weekday, start_time, end_time FROM "
                "availability_windows WHERE agent_id = ? ORDER BY weekday, start_time",
                (agent_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT agent_id, weekday, start_time, end_time FROM "
                "availability_windows ORDER BY agent_id, weekday, start_time"
            ).fetchall()
    return [
        {
            "agent_id": r["agent_id"],
            "weekday": r["weekday"],
            "start": r["start_time"],
            "end": r["end_time"],
        }
        for r in rows
    ]


def _booking_row(r: sqlite3.Row) -> dict:
    return dict(r)


def list_bookings(
    status: str = None,
    property_id: str = None,
    email: str = None,
    applicant_id: str = None,
    agent_id: str = None,
) -> list:
    """Bookings matching the given filters, soonest first."""
    clauses = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if property_id:
        clauses.append("property_id = ?")
        params.append(property_id)
    if email:
        clauses.append("prospect_email = ?")
        params.append(email)
    if applicant_id:
        clauses.append("applicant_id = ?")
        params.append(applicant_id)
    if agent_id:
        clauses.append("agent_id = ?")
        params.append(agent_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM tour_bookings {where} ORDER BY start ASC", params
        ).fetchall()
    return [_booking_row(r) for r in rows]


def get_booking(booking_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM tour_bookings WHERE id = ?", (booking_id,)
        ).fetchone()
    return _booking_row(row) if row else None


def book_tour(
    property_id: str,
    property_name: str,
    agent_id: str,
    agent_name: str,
    start: str,
    end: str,
    prospect_name: str,
    duration_minutes: int = 30,
    prospect_email: str = "",
    prospect_phone: str = "",
    applicant_id: str = "",
    notes: str = "",
) -> dict:
    """Atomically book a tour. Re-checks the conflict inside a transaction
    (BEGIN IMMEDIATE) and raises SlotConflict if the slot was taken in the
    meantime, so two concurrent callers can never double-book one agent+start.
    """
    booking_id = "TOUR-" + uuid.uuid4().hex[:12]
    created_at = _now()
    conn = sqlite3.connect(DB_PATH, isolation_level=None)  # manual transaction
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        taken = conn.execute(
            "SELECT 1 FROM tour_bookings WHERE agent_id = ? AND start = ? "
            "AND status = 'booked'",
            (agent_id, start),
        ).fetchone()
        if taken:
            conn.execute("ROLLBACK")
            raise SlotConflict(f"{agent_id} is already booked at {start}")
        conn.execute(
            "INSERT INTO tour_bookings (id, property_id, property_name, start, "
            "end, duration_minutes, agent_id, agent_name, prospect_name, "
            "prospect_email, prospect_phone, applicant_id, notes, status, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                booking_id,
                property_id,
                property_name,
                start,
                end,
                int(duration_minutes),
                agent_id,
                agent_name,
                prospect_name,
                prospect_email,
                prospect_phone,
                applicant_id,
                notes,
                "booked",
                created_at,
            ),
        )
        conn.execute("COMMIT")
    except sqlite3.IntegrityError as exc:
        # The partial UNIQUE index fired: another writer won the race.
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise SlotConflict(f"{agent_id} is already booked at {start}") from exc
    finally:
        conn.close()
    return get_booking(booking_id)


def cancel_booking(booking_id: str) -> bool:
    """Mark a booking cancelled (frees the slot). False if unknown."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE tour_bookings SET status = 'cancelled' WHERE id = ?",
            (booking_id,),
        )
        return cur.rowcount > 0
