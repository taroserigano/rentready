"""Pure slot engine for the Tour Scheduler + seed data.

No FastAPI, no LLM, no I/O beyond the store (only ``seed_tours`` touches the
store). Everything here is deterministic and unit-testable: every function
that depends on "now"/"today" takes an injected reference date/datetime so
tests can pin the clock to Monday 2026-07-20.

Slot model:
  - A slot is a 30-minute interval assigned to exactly one agent.
  - Stable id = f"{agent_id}|{start_iso}" e.g. "AGENT-01|2026-07-22T14:00:00".
  - Multiple eligible agents may be free at the same time; we assign ONE
    (the least-loaded) via ``assign_agent`` so the prospect sees one slot per
    time, and load stays balanced across staff.

time_of_day buckets: morning < 12:00, afternoon 12:00-17:00, evening >= 17:00.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

DEFAULT_DURATION_MIN = 30
DEFAULT_STEP_MIN = 30
DEFAULT_HORIZON_DAYS = 7

# Tours are stored as naive local wall-clock in Austin's zone (see tours_api's
# _TZID docstring). bare `datetime.now()` returns naive time in whatever zone
# the SERVER PROCESS happens to run in (UTC in most containers) — comparing
# that against Central-wall-clock slot times would shift every "is this slot
# in the past" check by the offset between the two. This gives the current
# wall-clock time AS IT WOULD READ in Central, regardless of server TZ.
_LOCAL_TZ = ZoneInfo("America/Chicago")


def _local_now() -> datetime:
    return datetime.now(_LOCAL_TZ).replace(tzinfo=None)


_WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@dataclass
class OpenSlot:
    """An open slot produced by the engine (naive local datetimes)."""

    property_id: str
    start: datetime
    end: datetime
    agent_id: str
    agent_name: str

    @property
    def slot_id(self) -> str:
        return make_slot_id(self.agent_id, self.start)

    @property
    def label(self) -> str:
        return label(self.start)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_hhmm(s: str) -> int:
    """"HH:MM" -> minutes since midnight."""
    h, m = str(s).strip().split(":")
    return int(h) * 60 + int(m)


def label(dt: datetime) -> str:
    """Human label, e.g. "Tue Jul 22, 2:00 PM" (no leading zeros, no %-flags
    that break on Windows)."""
    wd = _WEEKDAY_ABBR[dt.weekday()]
    mon = _MONTH_ABBR[dt.month - 1]
    hour24 = dt.hour
    ampm = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    return f"{wd} {mon} {dt.day}, {hour12}:{dt.minute:02d} {ampm}"


def make_slot_id(agent_id: str, start: datetime) -> str:
    return f"{agent_id}|{start.isoformat(timespec='seconds')}"


def parse_slot_id(slot_id: str) -> tuple[str, datetime]:
    """"AGENT-01|2026-07-22T14:00:00" -> ("AGENT-01", datetime(...))."""
    agent_id, _, start_iso = str(slot_id).partition("|")
    if not agent_id or not start_iso:
        raise ValueError(f"Malformed slot_id: {slot_id!r}")
    return agent_id, datetime.fromisoformat(start_iso)


def time_of_day_of(dt: datetime) -> str:
    minutes = dt.hour * 60 + dt.minute
    if minutes < 12 * 60:
        return "morning"
    if minutes < 17 * 60:
        return "afternoon"
    return "evening"


# ---------------------------------------------------------------------------
# Windows / intervals
# ---------------------------------------------------------------------------


def window_applies(window: dict, day: date) -> bool:
    """True if a recurring weekly window applies on the given calendar day."""
    return int(window["weekday"]) == day.weekday()


def chunk_interval(
    start: datetime,
    end: datetime,
    duration_min: int = DEFAULT_DURATION_MIN,
    step_min: int = DEFAULT_STEP_MIN,
) -> list[tuple[datetime, datetime]]:
    """Slice [start, end) into fixed-length slots stepping every ``step_min``.

    A slot is only emitted if it fits entirely inside the interval.
    """
    out: list[tuple[datetime, datetime]] = []
    cur = start
    dur = timedelta(minutes=duration_min)
    step = timedelta(minutes=step_min)
    while cur + dur <= end:
        out.append((cur, cur + dur))
        cur = cur + step
    return out


def expand_windows(
    windows: list[dict],
    day: date,
    duration_min: int = DEFAULT_DURATION_MIN,
    step_min: int = DEFAULT_STEP_MIN,
) -> list[tuple[datetime, datetime]]:
    """All (start, end) slot intervals produced by the given windows on ``day``."""
    slots: list[tuple[datetime, datetime]] = []
    for w in windows:
        if not window_applies(w, day):
            continue
        w_start = datetime.combine(day, datetime.min.time()) + timedelta(
            minutes=parse_hhmm(w["start"])
        )
        w_end = datetime.combine(day, datetime.min.time()) + timedelta(
            minutes=parse_hhmm(w["end"])
        )
        slots.extend(chunk_interval(w_start, w_end, duration_min, step_min))
    return slots


def overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    """True if [a_start,a_end) and [b_start,b_end) overlap."""
    return a_start < b_end and b_start < a_end


def has_conflict(
    agent_id: str, start: datetime, end: datetime, bookings: list[dict]
) -> bool:
    """True if ``agent_id`` already has a booked tour overlapping [start,end)."""
    for b in bookings:
        if b.get("status") != "booked":
            continue
        if b.get("agent_id") != agent_id:
            continue
        b_start = datetime.fromisoformat(b["start"])
        b_end = datetime.fromisoformat(b["end"])
        if overlaps(start, end, b_start, b_end):
            return True
    return False


def agent_free(
    agent_id: str, start: datetime, end: datetime, bookings: list[dict]
) -> bool:
    return not has_conflict(agent_id, start, end, bookings)


def agent_covers(agent: dict, area: str) -> bool:
    """An agent covers a property's neighborhood if its areas list is empty
    (covers all) or the area is listed."""
    areas = agent.get("areas") or []
    if not areas:
        return True
    return area in areas


def assign_agent(agents: list[dict], load_by_agent: dict[str, int]) -> dict:
    """Pick the least-loaded agent (fewest bookings/assignments so far),
    tie-broken by agent id for determinism."""
    return min(agents, key=lambda a: (load_by_agent.get(a["id"], 0), a["id"]))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def open_slots(
    property_id: str,
    area: str,
    agents: list[dict],
    windows: list[dict],
    bookings: list[dict],
    date_from: date,
    date_to: date,
    time_of_day: str = "any",
    now: datetime | None = None,
    duration_min: int = DEFAULT_DURATION_MIN,
    step_min: int = DEFAULT_STEP_MIN,
    after_min: int | None = None,
    before_min: int | None = None,
    at_min: int | None = None,
) -> list[OpenSlot]:
    """All open (unbooked, non-past) tour slots for a property in a date range.

    Only agents whose ``areas`` cover ``area`` (or areas==[]) are considered.
    For each distinct start time, exactly one agent is assigned - the least
    loaded eligible+free one - so the caller sees one slot per time and load
    is balanced across staff.
    """
    if now is None:
        now = _local_now()

    eligible = [a for a in agents if agent_covers(a, area)]
    windows_by_agent: dict[str, list[dict]] = {}
    for w in windows:
        windows_by_agent.setdefault(w["agent_id"], []).append(w)

    # Seed each agent's load with its existing booked tours so a busy agent
    # naturally gets fewer new assignments.
    load_by_agent: dict[str, int] = {a["id"]: 0 for a in eligible}
    for b in bookings:
        if b.get("status") == "booked" and b.get("agent_id") in load_by_agent:
            load_by_agent[b["agent_id"]] += 1

    # Gather candidate (start, end) times across all eligible agents.
    candidate_starts: dict[datetime, datetime] = {}
    for a in eligible:
        aw = windows_by_agent.get(a["id"], [])
        day = date_from
        while day <= date_to:
            for s, e in expand_windows(aw, day, duration_min, step_min):
                candidate_starts.setdefault(s, e)
            day += timedelta(days=1)

    # Tracks intervals already assigned to each agent BY THIS CALL, in
    # addition to `bookings` (real, already-persisted bookings). Only needed
    # when duration_min > step_min: candidate slots for one agent can then
    # overlap each other (e.g. 60-min tours on a 30-min step), and
    # `agent_free` alone -- checking only `bookings` -- would let the same
    # agent be assigned to two overlapping proposed slots in one response.
    # A no-op with the default equal duration/step, where slots never overlap.
    assigned_here: dict[str, list[tuple[datetime, datetime]]] = {
        a["id"]: [] for a in eligible
    }

    results: list[OpenSlot] = []
    for start in sorted(candidate_starts):
        end = candidate_starts[start]
        if start <= now:  # exclude past
            continue
        if time_of_day and time_of_day != "any" and time_of_day_of(start) != time_of_day:
            continue
        start_min = start.hour * 60 + start.minute
        if after_min is not None and start_min < after_min:
            continue
        if before_min is not None and start_min >= before_min:
            continue
        if at_min is not None and start_min != at_min:
            continue
        # Which eligible agents both have a window covering this slot AND are free?
        free_here = [
            a
            for a in eligible
            if _agent_window_covers(windows_by_agent.get(a["id"], []), start, end)
            and agent_free(a["id"], start, end, bookings)
            and not any(overlaps(start, end, s, e) for s, e in assigned_here[a["id"]])
        ]
        if not free_here:
            continue
        chosen = assign_agent(free_here, load_by_agent)
        load_by_agent[chosen["id"]] += 1
        assigned_here[chosen["id"]].append((start, end))
        results.append(
            OpenSlot(
                property_id=property_id,
                start=start,
                end=end,
                agent_id=chosen["id"],
                agent_name=chosen["name"],
            )
        )
    return results


def _agent_window_covers(windows: list[dict], start: datetime, end: datetime) -> bool:
    """True if one of the agent's recurring windows contains [start, end)."""
    day = start.date()
    for w in windows:
        if not window_applies(w, day):
            continue
        w_start = datetime.combine(day, datetime.min.time()) + timedelta(
            minutes=parse_hhmm(w["start"])
        )
        w_end = datetime.combine(day, datetime.min.time()) + timedelta(
            minutes=parse_hhmm(w["end"])
        )
        if w_start <= start and end <= w_end:
            return True
    return False


def slot_is_structurally_valid(
    agent_id: str,
    start: datetime,
    agents: list[dict],
    windows: list[dict],
    now: datetime | None = None,
    duration_min: int = DEFAULT_DURATION_MIN,
    step_min: int = DEFAULT_STEP_MIN,
) -> bool:
    """True if a slot could ever be booked (ignoring current bookings): the
    agent exists, the slot is not in the past, it is aligned to the step, and
    it falls inside one of the agent's recurring windows."""
    if now is None:
        now = _local_now()
    if not any(a["id"] == agent_id for a in agents):
        return False
    if start <= now:
        return False
    end = start + timedelta(minutes=duration_min)
    agent_windows = [w for w in windows if w["agent_id"] == agent_id]
    return _agent_window_covers(agent_windows, start, end)


# ---------------------------------------------------------------------------
# Seed data (idempotent)
# ---------------------------------------------------------------------------

MON, TUE, WED, THU, FRI, SAT, SUN = range(7)

# Every property gets its own dedicated agent -- eligibility is keyed to the
# property's OWN id (via `agent["areas"] == [property_id]`), not its
# neighborhood, so no two properties ever share the same assigned tour agent.
# (`tours_api._area_of` feeds that property id into `agent_covers` under the
# same "area" parameter the engine already understands -- one agent, one
# property, no overlap.)
_AGENT_NAMES = [
    "Maria Lopez", "James Chen", "Priya Patel", "Deshawn Brown", "Sofia Ramirez",
    "Marcus Webb", "Elena Volkov", "Tyrese Jackson", "Grace Kim", "Omar Haddad",
    "Lucia Fernandez", "Noah Bennett", "Aisha Rahman", "Connor Walsh", "Mei Lin",
    "Diego Alvarez", "Fatima Al-Sayed", "Jackson Reid", "Naomi Cohen", "Trevor Boyd",
    "Yuki Tanaka", "Andre Dupont", "Simone Baptiste", "Wesley Foster", "Ingrid Solberg",
    "Rashid Malik", "Camille Girard", "Devon Marsh", "Anika Sharma", "Miguel Torres",
    "Clara Jensen", "Bilal Ahmed", "Renee Dubois", "Julian Ortiz", "Keisha Wright",
    "Henrik Larsen", "Zara Malik", "Cole Ramsey", "Isabela Souza", "Franklin Osei",
    "Petra Novak", "Damon Ellis", "Nadia Hassan", "Colton Briggs", "Yara Nasser",
    "Preston Cole", "Amara Nwosu", "Silas Grant", "Leilani Kahale", "Emmett Hale",
]
_ROLE_CYCLE = ["Leasing Consultant", "Senior Leasing Agent", "Tour Specialist"]


def _build_seed_agents() -> list[dict]:
    """One dedicated agent per property, in property order, so assignment is
    stable and deterministic run to run."""
    import graph

    agents = []
    for i, prop in enumerate(graph.load_properties()):
        agents.append(
            {
                "id": f"AGENT-{i + 1:03d}",
                "name": _AGENT_NAMES[i % len(_AGENT_NAMES)],
                "role": _ROLE_CYCLE[i % len(_ROLE_CYCLE)],
                "areas": [prop["id"]],
                "days": [MON, TUE, WED, THU, FRI, SAT],
                "start": "09:00",
                "end": "19:00",
            }
        )
    return agents


_SEED_AGENTS = _build_seed_agents()


def seed_tours() -> dict:
    """Populate agents + availability windows and 1-2 demo bookings.

    Idempotent: if the agents table is already populated, does nothing.
    """
    import store

    if store.list_agents():
        return {"seeded": False, "agents": len(store.list_agents())}

    for spec in _SEED_AGENTS:
        store.save_agent(
            {
                "id": spec["id"],
                "name": spec["name"],
                "role": spec["role"],
                "areas": spec["areas"],
            }
        )
        if "windows" in spec:
            for weekday, start, end in spec["windows"]:
                store.save_window(spec["id"], weekday, start, end)
        else:
            for weekday in spec["days"]:
                store.save_window(spec["id"], weekday, spec["start"], spec["end"])

    # One demo booking so the UI is not empty: PROP-002 Riverside Lofts,
    # 2026-07-22 (Wed) 10:00 with its dedicated agent.
    demo_agent = next(
        (spec for spec in _SEED_AGENTS if "PROP-002" in spec["areas"]),
        _SEED_AGENTS[0],
    )
    demo_start = datetime(2026, 7, 22, 10, 0, 0)
    try:
        store.book_tour(
            property_id="PROP-002",
            property_name="Riverside Lofts",
            agent_id=demo_agent["id"],
            agent_name=demo_agent["name"],
            start=demo_start.isoformat(timespec="seconds"),
            end=(demo_start + timedelta(minutes=30)).isoformat(timespec="seconds"),
            duration_minutes=30,
            prospect_name="Jordan Rivera",
            prospect_email="jordan.rivera@example.com",
        )
    except Exception as exc:  # noqa: BLE001 - demo data is best-effort
        print(f"seed_tours demo booking skipped: {type(exc).__name__}: {exc}")

    return {"seeded": True, "agents": len(store.list_agents())}
