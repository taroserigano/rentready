"""Persistence for applicants + the Tour Scheduler.

Two backends behind ONE function API, so every caller (tours.py, tours_api.py,
tours_chat.py, apply_api.py, dashboard_api.py, ...) is unaffected by which is
active:

  - SQLite (default -- every local run and every test): the stdlib sqlite3
    driver pointed at ``DB_PATH``, a single file. Zero setup, zero network.
  - Postgres (when ``settings.database_url`` is set, e.g. a shared Neon
    project used by other apps too): SQLAlchemy Core over psycopg2, with
    every table of THIS app isolated under a dedicated ``rentready`` schema
    so it can never collide with another project's tables in the same
    database.

SQLAlchemy Core (not the ORM) is the shared layer for both backends so
queries don't need hand-translated placeholder syntax (sqlite's ``?`` vs
psycopg2's ``%s``) or exception types (``sqlite3.IntegrityError`` vs
``psycopg2.errors.UniqueViolation``) -- SQLAlchemy compiles named ``:param``
binds per-dialect and always raises its own ``IntegrityError`` regardless of
driver. Only the DDL (run once, at ``init_db()``) is written twice, since the
autoincrement/serial and schema-qualification differences are small and
one-time.

Tests force ``settings.database_url`` empty (see tests/conftest.py) and
monkeypatch ``DB_PATH`` per-fixture for isolation, exactly as before this file
grew a second backend.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from models import ApplicantProfile
from settings import settings

# The SQLite file (used whenever settings.database_url is empty). Overridable
# via ``RENTREADY_DB`` so e2e/integration runs can point at a throwaway copy
# instead of mutating the real ``rentready.db``.
DB_PATH = Path(os.environ.get("RENTREADY_DB") or (Path(__file__).resolve().parent.parent / "rentready.db"))

# All of this app's Postgres tables live under this schema -- Neon projects in
# practice get shared across several small apps, and a dedicated schema keeps
# our tables from ever colliding with (or being confused for) theirs.
_PG_SCHEMA = "rentready"


class SlotConflict(Exception):
    """Raised when an atomic booking loses a race for a slot (agent+start
    already booked). The API turns this into a 409."""


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def _is_pg() -> bool:
    return bool(settings.database_url)


def _pg_url() -> str:
    url = settings.database_url
    # SQLAlchemy 1.4+/2.0 only recognizes the "postgresql://" scheme; Neon (and
    # most managed Postgres providers) hand out the shorter Heroku-style
    # "postgres://" -- normalize rather than ask the user to edit their .env.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


@lru_cache(maxsize=128)
def _engine_for(url: str):
    return create_engine(url, future=True, pool_pre_ping=True)


def _engine():
    """The active engine, re-resolved on every call so tests/scripts that
    monkeypatch ``DB_PATH`` (or force ``database_url`` empty) take effect
    immediately -- never a stale engine from a previous config."""
    url = _pg_url() if _is_pg() else f"sqlite:///{DB_PATH}"
    return _engine_for(url)


def _t(name: str) -> str:
    """Schema-qualify a table name on Postgres; bare on SQLite."""
    return f"{_PG_SCHEMA}.{name}" if _is_pg() else name


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def _sqlite_ddl() -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS applicants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            chunks_indexed INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """,
        # Production telemetry: one row per served request (online monitoring).
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
        """,
        # User feedback (thumbs up/down) on a served output.
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
        """,
        # Reviewer decisions on an applicant (approve/decline/waitlist/info).
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            applicant_id TEXT NOT NULL,
            action TEXT NOT NULL,
            note TEXT,
            reviewer TEXT
        )
        """,
        # Tour Scheduler: leasing agents who run tours. `areas` is JSON text
        # (list of property ids the agent is dedicated to, normally exactly
        # one); [] means the agent covers all properties.
        """
        CREATE TABLE IF NOT EXISTS tour_agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            areas_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        # Recurring weekly availability windows (weekday 0=Mon..6=Sun, local HH:MM).
        """
        CREATE TABLE IF NOT EXISTS availability_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            weekday INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL
        )
        """,
        # Booked tours. A UNIQUE index on (agent_id, start) for status='booked'
        # is the last line of defense against a double-book race.
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
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_booked_agent_start "
        "ON tour_bookings (agent_id, start) WHERE status = 'booked'",
    ]


def _pg_ddl() -> list[str]:
    s = _PG_SCHEMA
    return [
        f"CREATE SCHEMA IF NOT EXISTS {s}",
        f"""
        CREATE TABLE IF NOT EXISTS {s}.applicants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            chunks_indexed INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {s}.prod_events (
            id SERIAL PRIMARY KEY,
            ts TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            applicant_id TEXT,
            latency_ms REAL,
            source TEXT,
            faithfulness_violations INTEGER DEFAULT 0,
            meta_json TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {s}.feedback (
            id SERIAL PRIMARY KEY,
            ts TEXT NOT NULL,
            applicant_id TEXT,
            target TEXT NOT NULL,
            item_id TEXT,
            rating TEXT NOT NULL,
            comment TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {s}.decisions (
            id SERIAL PRIMARY KEY,
            ts TEXT NOT NULL,
            applicant_id TEXT NOT NULL,
            action TEXT NOT NULL,
            note TEXT,
            reviewer TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {s}.tour_agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            areas_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {s}.availability_windows (
            id SERIAL PRIMARY KEY,
            agent_id TEXT NOT NULL,
            weekday INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {s}.tour_bookings (
            id TEXT PRIMARY KEY,
            property_id TEXT NOT NULL,
            property_name TEXT NOT NULL,
            start TEXT NOT NULL,
            "end" TEXT NOT NULL,
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
        """,
        f"CREATE UNIQUE INDEX IF NOT EXISTS ux_booked_agent_start "
        f"ON {s}.tour_bookings (agent_id, start) WHERE status = 'booked'",
    ]


def init_db() -> None:
    ddl = _pg_ddl() if _is_pg() else _sqlite_ddl()
    with _engine().begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(result) -> list[dict]:
    return [dict(r) for r in result.mappings().all()]


def _row(result) -> dict | None:
    r = result.mappings().first()
    return dict(r) if r is not None else None


# ---------------------------------------------------------------------------
# Applicant telemetry / feedback / decisions
# ---------------------------------------------------------------------------
def log_event(
    endpoint: str,
    applicant_id: str = None,
    latency_ms: float = None,
    source: str = None,
    faithfulness_violations: int = 0,
    meta: dict = None,
) -> None:
    """Record one served request for online monitoring."""
    with _engine().begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {_t('prod_events')} (ts, endpoint, applicant_id, "
                "latency_ms, source, faithfulness_violations, meta_json) "
                "VALUES (:ts, :endpoint, :applicant_id, :latency_ms, :source, "
                ":violations, :meta_json)"
            ),
            {
                "ts": _now(),
                "endpoint": endpoint,
                "applicant_id": applicant_id,
                "latency_ms": latency_ms,
                "source": source,
                "violations": int(faithfulness_violations or 0),
                "meta_json": json.dumps(meta or {}),
            },
        )


def recent_events(limit: int = 500) -> list:
    with _engine().begin() as conn:
        rows = _rows(
            conn.execute(
                text(
                    f"SELECT ts, endpoint, applicant_id, latency_ms, source, "
                    "faithfulness_violations, meta_json FROM "
                    f"{_t('prod_events')} ORDER BY id DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
        )
    out = []
    for d in rows:
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
    with _engine().begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {_t('feedback')} (ts, applicant_id, target, "
                "item_id, rating, comment) VALUES (:ts, :applicant_id, :target, "
                ":item_id, :rating, :comment)"
            ),
            {
                "ts": _now(),
                "applicant_id": applicant_id,
                "target": target,
                "item_id": item_id,
                "rating": rating,
                "comment": comment,
            },
        )


def recent_feedback(limit: int = 500) -> list:
    with _engine().begin() as conn:
        return _rows(
            conn.execute(
                text(
                    f"SELECT ts, applicant_id, target, item_id, rating, comment "
                    f"FROM {_t('feedback')} ORDER BY id DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
        )


def save_decision(
    applicant_id: str, action: str, note: str = None, reviewer: str = None
) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {_t('decisions')} (ts, applicant_id, action, "
                "note, reviewer) VALUES (:ts, :applicant_id, :action, :note, "
                ":reviewer)"
            ),
            {
                "ts": _now(),
                "applicant_id": applicant_id,
                "action": action,
                "note": note,
                "reviewer": reviewer,
            },
        )


def decisions_for(applicant_id: str) -> list:
    with _engine().begin() as conn:
        return _rows(
            conn.execute(
                text(
                    f"SELECT ts, action, note, reviewer FROM {_t('decisions')} "
                    "WHERE applicant_id = :applicant_id ORDER BY id DESC"
                ),
                {"applicant_id": applicant_id},
            )
        )


def latest_statuses() -> dict:
    """Map applicant_id -> most recent decision action (its current status)."""
    with _engine().begin() as conn:
        rows = _rows(
            conn.execute(
                text(
                    f"SELECT applicant_id, action FROM {_t('decisions')} d "
                    "WHERE id = (SELECT MAX(id) FROM "
                    f"{_t('decisions')} WHERE applicant_id = d.applicant_id)"
                )
            )
        )
    return {r["applicant_id"]: r["action"] for r in rows}


def save_applicant(
    applicant_id: str, profile: ApplicantProfile, chunks_indexed: int
) -> None:
    with _engine().begin() as conn:
        # DELETE-then-INSERT is a dialect-agnostic upsert (no ON CONFLICT
        # syntax differences between sqlite/postgres to worry about).
        conn.execute(
            text(f"DELETE FROM {_t('applicants')} WHERE id = :id"),
            {"id": applicant_id},
        )
        conn.execute(
            text(
                f"INSERT INTO {_t('applicants')} (id, name, profile_json, "
                "chunks_indexed, created_at) VALUES (:id, :name, :profile_json, "
                ":chunks_indexed, :created_at)"
            ),
            {
                "id": applicant_id,
                "name": profile.name,
                "profile_json": profile.model_dump_json(),
                "chunks_indexed": chunks_indexed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def get_profile(applicant_id: str):
    with _engine().begin() as conn:
        row = _row(
            conn.execute(
                text(f"SELECT profile_json FROM {_t('applicants')} WHERE id = :id"),
                {"id": applicant_id},
            )
        )
    if row is None:
        return None
    return ApplicantProfile.model_validate_json(row["profile_json"])


def list_applicants() -> list:
    with _engine().begin() as conn:
        return _rows(
            conn.execute(
                text(
                    f"SELECT id, name, chunks_indexed, created_at "
                    f"FROM {_t('applicants')} ORDER BY created_at DESC"
                )
            )
        )


def delete_applicant(applicant_id: str) -> bool:
    """Delete the applicant row plus what's clearly THEIR data (reviewer
    decisions about them). ``tour_bookings`` are a leasing agent's real
    calendar entries, not applicant-owned — the booking itself stays, only the
    now-dangling applicant link is cleared so it stops 404ing when looked up
    by applicant id. ``prod_events``/``feedback`` are aggregate telemetry (the
    monitoring dashboard's historical charts), so they're deliberately left
    alone rather than corrupting that history.

    The uploaded PDF and RAG index for this applicant are cleaned up by the
    caller (see main.py) — this function only owns the SQL rows."""
    with _engine().begin() as conn:
        result = conn.execute(
            text(f"DELETE FROM {_t('applicants')} WHERE id = :id"),
            {"id": applicant_id},
        )
        deleted = result.rowcount > 0
        if deleted:
            conn.execute(
                text(f"DELETE FROM {_t('decisions')} WHERE applicant_id = :id"),
                {"id": applicant_id},
            )
            conn.execute(
                text(
                    f"UPDATE {_t('tour_bookings')} SET applicant_id = '' "
                    "WHERE applicant_id = :id"
                ),
                {"id": applicant_id},
            )
        return deleted


# ---------------------------------------------------------------------------
# Tour Scheduler persistence
# ---------------------------------------------------------------------------
def save_agent(agent: dict) -> None:
    """Insert/replace a tour agent. ``areas`` stored as JSON text."""
    with _engine().begin() as conn:
        conn.execute(
            text(f"DELETE FROM {_t('tour_agents')} WHERE id = :id"),
            {"id": agent["id"]},
        )
        conn.execute(
            text(
                f"INSERT INTO {_t('tour_agents')} (id, name, role, areas_json, "
                "created_at) VALUES (:id, :name, :role, :areas_json, :created_at)"
            ),
            {
                "id": agent["id"],
                "name": agent["name"],
                "role": agent.get("role", "Leasing Consultant"),
                "areas_json": json.dumps(agent.get("areas") or []),
                "created_at": _now(),
            },
        )


def list_agents() -> list:
    with _engine().begin() as conn:
        rows = _rows(
            conn.execute(
                text(f"SELECT id, name, role, areas_json FROM {_t('tour_agents')} ORDER BY id")
            )
        )
    out = []
    for d in rows:
        d["areas"] = json.loads(d.pop("areas_json") or "[]")
        out.append(d)
    return out


def save_window(agent_id: str, weekday: int, start: str, end: str) -> None:
    with _engine().begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {_t('availability_windows')} (agent_id, weekday, "
                "start_time, end_time) VALUES (:agent_id, :weekday, :start, :end)"
            ),
            {"agent_id": agent_id, "weekday": int(weekday), "start": start, "end": end},
        )


def list_windows(agent_id: str = None) -> list:
    """Recurring weekly windows, normalized to the engine's dict shape
    ({agent_id, weekday, start, end})."""
    with _engine().begin() as conn:
        if agent_id is not None:
            rows = _rows(
                conn.execute(
                    text(
                        "SELECT agent_id, weekday, start_time, end_time FROM "
                        f"{_t('availability_windows')} WHERE agent_id = :agent_id "
                        "ORDER BY weekday, start_time"
                    ),
                    {"agent_id": agent_id},
                )
            )
        else:
            rows = _rows(
                conn.execute(
                    text(
                        "SELECT agent_id, weekday, start_time, end_time FROM "
                        f"{_t('availability_windows')} ORDER BY agent_id, weekday, start_time"
                    )
                )
            )
    return [
        {
            "agent_id": r["agent_id"],
            "weekday": r["weekday"],
            "start": r["start_time"],
            "end": r["end_time"],
        }
        for r in rows
    ]


def list_bookings(
    status: str = None,
    property_id: str = None,
    email: str = None,
    applicant_id: str = None,
    agent_id: str = None,
) -> list:
    """Bookings matching the given filters, soonest first."""
    clauses = []
    params: dict = {}
    if status:
        clauses.append("status = :status")
        params["status"] = status
    if property_id:
        clauses.append("property_id = :property_id")
        params["property_id"] = property_id
    if email:
        clauses.append("prospect_email = :email")
        params["email"] = email
    if applicant_id:
        clauses.append("applicant_id = :applicant_id")
        params["applicant_id"] = applicant_id
    if agent_id:
        clauses.append("agent_id = :agent_id")
        params["agent_id"] = agent_id
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _engine().begin() as conn:
        return _rows(
            conn.execute(
                text(f"SELECT * FROM {_t('tour_bookings')} {where} ORDER BY start ASC"),
                params,
            )
        )


def get_booking(booking_id: str) -> dict | None:
    with _engine().begin() as conn:
        return _row(
            conn.execute(
                text(f"SELECT * FROM {_t('tour_bookings')} WHERE id = :id"),
                {"id": booking_id},
            )
        )


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
    """Atomically book a tour. The partial UNIQUE index (agent_id, start) WHERE
    status='booked' is enforced by the database itself, so two concurrent
    callers can never double-book one agent+start: whichever transaction's
    INSERT commits second gets an IntegrityError, translated to SlotConflict.
    """
    booking_id = "TOUR-" + uuid.uuid4().hex[:12]
    created_at = _now()
    try:
        with _engine().begin() as conn:
            taken = _row(
                conn.execute(
                    text(
                        f"SELECT 1 AS one FROM {_t('tour_bookings')} WHERE "
                        "agent_id = :agent_id AND start = :start AND status = 'booked'"
                    ),
                    {"agent_id": agent_id, "start": start},
                )
            )
            if taken:
                raise SlotConflict(f"{agent_id} is already booked at {start}")
            conn.execute(
                text(
                    f"INSERT INTO {_t('tour_bookings')} (id, property_id, "
                    'property_name, start, "end", duration_minutes, agent_id, '
                    "agent_name, prospect_name, prospect_email, prospect_phone, "
                    "applicant_id, notes, status, created_at) VALUES (:id, "
                    ":property_id, :property_name, :start, :end, "
                    ":duration_minutes, :agent_id, :agent_name, :prospect_name, "
                    ":prospect_email, :prospect_phone, :applicant_id, :notes, "
                    "'booked', :created_at)"
                ),
                {
                    "id": booking_id,
                    "property_id": property_id,
                    "property_name": property_name,
                    "start": start,
                    "end": end,
                    "duration_minutes": int(duration_minutes),
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "prospect_name": prospect_name,
                    "prospect_email": prospect_email,
                    "prospect_phone": prospect_phone,
                    "applicant_id": applicant_id,
                    "notes": notes,
                    "created_at": created_at,
                },
            )
    except IntegrityError as exc:
        # The partial UNIQUE index fired: another writer won the race.
        raise SlotConflict(f"{agent_id} is already booked at {start}") from exc
    return get_booking(booking_id)


def cancel_booking(booking_id: str) -> bool:
    """Mark a booking cancelled (frees the slot). False if unknown."""
    with _engine().begin() as conn:
        result = conn.execute(
            text(f"UPDATE {_t('tour_bookings')} SET status = 'cancelled' WHERE id = :id"),
            {"id": booking_id},
        )
        return result.rowcount > 0
