"""Tests for the Tour Scheduler.

Everything is anchored to Monday 2026-07-20 via injected datetimes, so the
slot engine, assignment, and parser are fully deterministic. Store tests run
against a throwaway SQLite file. The LLM is disabled so chat source=="rules".
"""

from datetime import date, datetime, timedelta

import pytest

import store
import tours
import tours_chat
from models import ChatMessage, ChatState, TourChatRequest

# Monday 2026-07-20, 08:00 local — before any 09:00 window opens.
MON = datetime(2026, 7, 20, 8, 0, 0)
MON_DATE = date(2026, 7, 20)


# ---------------------------------------------------------------------------
# Pure engine
# ---------------------------------------------------------------------------


def test_label_no_leading_zeros():
    assert tours.label(datetime(2026, 7, 22, 14, 0)) == "Wed Jul 22, 2:00 PM"
    assert tours.label(datetime(2026, 7, 22, 9, 30)) == "Wed Jul 22, 9:30 AM"
    assert tours.label(datetime(2026, 7, 22, 0, 0)) == "Wed Jul 22, 12:00 AM"


def test_slot_id_roundtrip():
    start = datetime(2026, 7, 22, 14, 0, 0)
    sid = tours.make_slot_id("AGENT-01", start)
    assert sid == "AGENT-01|2026-07-22T14:00:00"
    agent_id, parsed = tours.parse_slot_id(sid)
    assert agent_id == "AGENT-01"
    assert parsed == start


def test_parse_slot_id_rejects_garbage():
    with pytest.raises(ValueError):
        tours.parse_slot_id("not-a-slot")


def test_chunk_interval_fits_only_whole_slots():
    start = datetime(2026, 7, 20, 9, 0)
    end = datetime(2026, 7, 20, 11, 0)
    chunks = tours.chunk_interval(start, end, 30, 30)
    assert len(chunks) == 4  # 9:00, 9:30, 10:00, 10:30
    assert chunks[0] == (start, start + timedelta(minutes=30))
    assert chunks[-1][1] == end


def test_expand_windows_resolves_weekday():
    windows = [{"agent_id": "A", "weekday": 0, "start": "09:00", "end": "10:00"}]
    # Monday 2026-07-20
    assert len(tours.expand_windows(windows, MON_DATE)) == 2
    # Tuesday -> the Monday window does not apply
    assert tours.expand_windows(windows, MON_DATE + timedelta(days=1)) == []


def test_overlaps_and_conflict():
    s = datetime(2026, 7, 20, 9, 0)
    e = datetime(2026, 7, 20, 9, 30)
    assert tours.overlaps(s, e, s, e)
    assert not tours.overlaps(s, e, e, e + timedelta(minutes=30))
    bookings = [{"agent_id": "A", "start": s.isoformat(), "end": e.isoformat(),
                 "status": "booked"}]
    assert tours.has_conflict("A", s, e, bookings)
    assert not tours.has_conflict("B", s, e, bookings)
    # A cancelled booking never conflicts.
    bookings[0]["status"] = "cancelled"
    assert not tours.has_conflict("A", s, e, bookings)


AGENTS = [
    {"id": "AGENT-01", "name": "Maria", "areas": ["Downtown"]},
    {"id": "AGENT-04", "name": "Deshawn", "areas": []},  # all areas
]
WINDOWS = [
    {"agent_id": "AGENT-01", "weekday": 0, "start": "09:00", "end": "11:00"},
    {"agent_id": "AGENT-04", "weekday": 0, "start": "12:00", "end": "14:00"},
]


def _open(area="Downtown", agents=AGENTS, windows=WINDOWS, bookings=None,
          time_of_day="any", now=MON, **kw):
    return tours.open_slots(
        property_id="PROP-002", area=area, agents=agents, windows=windows,
        bookings=bookings or [], date_from=MON_DATE, date_to=MON_DATE,
        time_of_day=time_of_day, now=now, **kw,
    )


def test_open_slots_basic_and_assignment():
    slots = _open()
    # 4 from AGENT-01 (9-11) + 4 from AGENT-04 (12-14)
    assert len(slots) == 8
    by_agent = {}
    for s in slots:
        by_agent.setdefault(s.agent_id, []).append(s)
    assert len(by_agent["AGENT-01"]) == 4
    assert len(by_agent["AGENT-04"]) == 4
    assert all(s.start > MON for s in slots)


def test_open_slots_excludes_past():
    # now = 9:45 -> 9:00 and 9:30 are in the past; 10:00 is future.
    slots = _open(now=datetime(2026, 7, 20, 9, 45))
    starts = [s.start.strftime("%H:%M") for s in slots if s.agent_id == "AGENT-01"]
    assert "09:00" not in starts and "09:30" not in starts
    assert "10:00" in starts


def test_open_slots_area_filter():
    # South Congress: AGENT-01 (Downtown only) excluded; AGENT-04 (all) stays.
    slots = _open(area="South Congress")
    assert {s.agent_id for s in slots} == {"AGENT-04"}


def test_open_slots_time_of_day():
    morning = _open(time_of_day="morning")
    assert morning and all(s.start.hour < 12 for s in morning)
    afternoon = _open(time_of_day="afternoon")
    assert afternoon and all(12 <= s.start.hour < 17 for s in afternoon)


def test_open_slots_excludes_booked_conflict():
    booked = [{"agent_id": "AGENT-01", "start": "2026-07-20T09:00:00",
               "end": "2026-07-20T09:30:00", "status": "booked"}]
    slots = _open(area="Downtown", agents=[AGENTS[0]], windows=[WINDOWS[0]],
                  bookings=booked)
    starts = [s.start.strftime("%H:%M") for s in slots]
    assert "09:00" not in starts
    assert "09:30" in starts


def test_load_balanced_assignment_favours_idle_agent():
    agents = [
        {"id": "AGENT-01", "name": "Maria", "areas": ["Downtown"]},
        {"id": "AGENT-02", "name": "James", "areas": ["Downtown"]},
    ]
    windows = [
        {"agent_id": "AGENT-01", "weekday": 0, "start": "09:00", "end": "11:00"},
        {"agent_id": "AGENT-02", "weekday": 0, "start": "09:00", "end": "11:00"},
    ]
    # AGENT-02 already has one booking (outside these windows -> no conflict),
    # so it should receive fewer of the 4 new slots.
    bookings = [{"agent_id": "AGENT-02", "start": "2026-07-20T15:00:00",
                 "end": "2026-07-20T15:30:00", "status": "booked"}]
    slots = tours.open_slots(
        "PROP-002", "Downtown", agents, windows, bookings,
        MON_DATE, MON_DATE, now=MON,
    )
    counts = {"AGENT-01": 0, "AGENT-02": 0}
    for s in slots:
        counts[s.agent_id] += 1
    assert len(slots) == 4
    assert counts["AGENT-01"] > counts["AGENT-02"]


def test_slot_structurally_valid():
    valid_start = datetime(2026, 7, 20, 9, 0)  # inside AGENT-01 Mon 09-11
    assert tours.slot_is_structurally_valid(
        "AGENT-01", valid_start, AGENTS, WINDOWS, now=MON
    )
    # Outside the window
    assert not tours.slot_is_structurally_valid(
        "AGENT-01", datetime(2026, 7, 20, 20, 0), AGENTS, WINDOWS, now=MON
    )
    # In the past
    assert not tours.slot_is_structurally_valid(
        "AGENT-01", valid_start, AGENTS, WINDOWS, now=datetime(2026, 7, 21, 9, 0)
    )


# ---------------------------------------------------------------------------
# Store: seed, atomic booking, cancel/rebook
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test_tours.db")
    store.init_db()
    tours.seed_tours()
    # Disable the LLM so chat stays deterministic (source == "rules").
    import llm
    monkeypatch.setattr(llm, "get_langchain_llm", lambda: None)
    yield


def test_seed_tours_idempotent(db):
    assert len(store.list_agents()) == len(tours._SEED_AGENTS)
    again = tours.seed_tours()
    assert again["seeded"] is False
    assert len(store.list_agents()) == len(tours._SEED_AGENTS)
    # 4+4+... windows exist
    assert len(store.list_windows()) >= 4


def _book(agent="AGENT-01", start="2026-07-27T09:00:00"):
    return store.book_tour(
        property_id="PROP-002", property_name="Riverside Lofts",
        agent_id=agent, agent_name="Maria Lopez", start=start,
        end="2026-07-27T09:30:00", prospect_name="Test Prospect",
    )


def test_local_now_is_naive_central_wall_clock_not_server_tz():
    """Regression: open_slots()/slot_is_structurally_valid() defaulted `now`
    to bare datetime.now() — the SERVER PROCESS's OS timezone (UTC in most
    containers) — while every stored slot/booking time is naive Central
    wall-clock (see tours_api._TZID). _local_now() must return the current
    time AS IT READS in Central, regardless of what timezone the machine
    running the tests/server is actually set to."""
    from zoneinfo import ZoneInfo

    got = tours._local_now()
    assert got.tzinfo is None  # naive, matching the stored-slot convention

    expected = datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None)
    assert abs((got - expected).total_seconds()) < 5


def test_atomic_double_book_rejected(db):
    b1 = _book()
    assert b1["id"].startswith("TOUR-")
    assert b1["status"] == "booked"
    with pytest.raises(store.SlotConflict):
        _book()  # same agent + start


def test_delete_applicant_cascades_decisions_and_unlinks_bookings(db):
    """Regression: deleting an applicant used to leave their decisions and
    tour bookings pointing at a now-nonexistent id (list_tours(applicant_id=)
    would still return the booking, 404ing when the UI tried to open it)."""
    from models import ApplicantProfile

    aid = "APP-DELETE-TEST"
    store.save_applicant(aid, ApplicantProfile(
        name="Cascade Test", monthly_income=5000, desired_rent=1500,
    ), chunks_indexed=0)
    store.save_decision(aid, action="approve", reviewer="Taro")
    booking = store.book_tour(
        property_id="PROP-002", property_name="Riverside Lofts",
        agent_id="AGENT-02", agent_name="Jordan Reyes",
        start="2026-07-28T10:00:00", end="2026-07-28T10:30:00",
        prospect_name="Cascade Test", applicant_id=aid,
    )

    assert store.decisions_for(aid)
    assert store.get_booking(booking["id"])["applicant_id"] == aid

    assert store.delete_applicant(aid) is True

    assert store.get_profile(aid) is None
    # The decision (reviewer's note ABOUT this applicant) is gone...
    assert store.decisions_for(aid) == []
    # ...but the booking itself (a real agent calendar slot) is NOT deleted —
    # only the dangling applicant_id link is cleared.
    kept = store.get_booking(booking["id"])
    assert kept is not None
    assert kept["applicant_id"] == ""
    assert kept["status"] == "booked"


def test_cancel_then_rebook(db):
    b1 = _book()
    assert store.cancel_booking(b1["id"]) is True
    assert store.get_booking(b1["id"])["status"] == "cancelled"
    # The slot is free again after cancellation.
    b2 = _book()
    assert b2["id"] != b1["id"]
    assert b2["status"] == "booked"


def test_list_bookings_filters(db):
    _book()
    booked = store.list_bookings(status="booked", property_id="PROP-002")
    assert any(b["agent_id"] == "AGENT-01" for b in booked)


# ---------------------------------------------------------------------------
# Clock-time token parser
# ---------------------------------------------------------------------------


def test_parse_time_token_noon_and_afternoon_together():
    # Regression: a broken `.replace(..., count=0)` no-op used to make this
    # silently fail to find noon whenever "afternoon" also appeared.
    assert tours_chat._parse_time_token("lets meet in the afternoon around noon") == 12 * 60
    assert tours_chat._parse_time_token("noon works") == 12 * 60
    assert tours_chat._parse_time_token("afternoon works") is None
    assert tours_chat._parse_time_token("midnight") == 0
    assert tours_chat._parse_time_token("5pm") == 17 * 60


# ---------------------------------------------------------------------------
# NL timing parser
# ---------------------------------------------------------------------------


def test_parse_this_weekend_afternoon():
    r = tours_chat.parse_timing("this weekend afternoon", MON)
    assert r["date_from"] == date(2026, 7, 25)  # Sat
    assert r["date_to"] == date(2026, 7, 26)  # Sun
    assert r["time_of_day"] == "afternoon"


def test_parse_tuesday_morning():
    r = tours_chat.parse_timing("Tuesday morning", MON)
    assert r["date_from"] == date(2026, 7, 21)
    assert r["date_to"] == date(2026, 7, 21)
    assert r["time_of_day"] == "morning"


def test_parse_after_5pm_next_week():
    r = tours_chat.parse_timing("after 5pm next week", MON)
    assert r["date_from"] == date(2026, 7, 27)  # next Monday
    assert r["date_to"] == date(2026, 8, 2)  # Sunday
    assert r["after_min"] == 17 * 60


def test_parse_tomorrow_and_today():
    assert tours_chat.parse_timing("today", MON)["date_from"] == MON_DATE
    r = tours_chat.parse_timing("tomorrow", MON)
    assert r["date_from"] == r["date_to"] == date(2026, 7, 21)


def test_parse_default_when_nothing_parses():
    r = tours_chat.parse_timing("hello there", MON)
    assert r["date_from"] == MON_DATE
    assert r["date_to"] == MON_DATE + timedelta(days=7)
    assert r["time_of_day"] == "any"
    assert r["after_min"] is None and r["before_min"] is None


def test_parse_before_and_at():
    assert tours_chat.parse_timing("before noon", MON)["before_min"] == 12 * 60
    assert tours_chat.parse_timing("at 3pm", MON)["at_min"] == 15 * 60


# ---------------------------------------------------------------------------
# Selection / name detectors
# ---------------------------------------------------------------------------


def _slots_for_detector():
    return [
        {"slot_id": "AGENT-01|2026-07-21T10:00:00", "start": "2026-07-21T10:00:00",
         "label": "x"},
        {"slot_id": "AGENT-01|2026-07-21T14:00:00", "start": "2026-07-21T14:00:00",
         "label": "y"},
    ]


def test_detect_selection_ordinal_and_time():
    slots = _slots_for_detector()
    assert tours_chat.detect_selection("the first one", slots) == slots[0]["slot_id"]
    assert tours_chat.detect_selection("last", slots) == slots[-1]["slot_id"]
    assert tours_chat.detect_selection("2pm works", slots) == slots[1]["slot_id"]
    assert tours_chat.detect_selection("maybe later", slots) is None


def test_extract_name():
    assert tours_chat.extract_name("my name is John Smith") == "John Smith"
    assert tours_chat.extract_name("I'm Priya") == "Priya"
    assert tours_chat.extract_name("Deshawn Brown") == "Deshawn Brown"
    assert tours_chat.extract_name("yes") == ""
    assert tours_chat.extract_name("tomorrow afternoon") == ""


def test_extract_phone():
    assert tours_chat.extract_phone("555-123-4567") == "555-123-4567"
    assert tours_chat.extract_phone("(555) 123-4567") == "(555) 123-4567"
    assert tours_chat.extract_phone("call me at 5551234567") == "5551234567"
    assert tours_chat.extract_phone("1-555-123-4567") == "1-555-123-4567"
    assert tours_chat.extract_phone("no phone here") == ""
    assert tours_chat.extract_phone("12345") == ""


def test_extract_email():
    assert tours_chat.extract_email("alex.kim@example.com") == "alex.kim@example.com"
    assert (
        tours_chat.extract_email("reach me at alex+tours@sub.example.co")
        == "alex+tours@sub.example.co"
    )
    assert tours_chat.extract_email("no email here") == ""


# ---------------------------------------------------------------------------
# Chat state machine (end to end against the DB)
# ---------------------------------------------------------------------------


def test_chat_propose_then_book(db):
    req = TourChatRequest(
        messages=[ChatMessage(role="user", content="this week afternoon")],
        property_id="PROP-002",
    )
    r1 = tours_chat.handle(req, now=MON)
    assert r1.source == "rules"
    assert r1.state.phase == "proposing"
    assert r1.proposed_slots
    assert r1.booking is None

    # Click the first slot -> asked for a name.
    slot_id = r1.proposed_slots[0].slot_id
    r2 = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(role="user", content="that one")],
            property_id="PROP-002", state=r1.state, selected_slot_id=slot_id,
        ),
        now=MON,
    )
    assert r2.state.phase == "awaiting_name"
    assert r2.state.pending_slot_id == slot_id

    # Give a name -> asked for a phone number.
    r3 = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(role="user", content="Alex Kim")],
            property_id="PROP-002", state=r2.state,
        ),
        now=MON,
    )
    assert r3.state.phase == "awaiting_phone"
    assert r3.state.prospect_name == "Alex Kim"
    assert r3.booking is None

    # Give a phone number -> asked for an email.
    r4 = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(role="user", content="555-123-4567")],
            property_id="PROP-002", state=r3.state,
        ),
        now=MON,
    )
    assert r4.state.phase == "awaiting_email"
    assert r4.state.prospect_phone == "555-123-4567"
    assert r4.booking is None

    # Give an email -> booked.
    r5 = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(role="user", content="alex.kim@example.com")],
            property_id="PROP-002", state=r4.state,
        ),
        now=MON,
    )
    assert r5.state.phase == "booked"
    assert r5.booking is not None
    assert r5.booking.prospect_name == "Alex Kim"
    assert r5.booking.prospect_phone == "555-123-4567"
    assert r5.booking.prospect_email == "alex.kim@example.com"


def test_chat_slot_taken_reproposes(db):
    req = TourChatRequest(
        messages=[ChatMessage(role="user", content="this week afternoon")],
        property_id="PROP-002",
    )
    r1 = tours_chat.handle(req, now=MON)
    slot_id = r1.proposed_slots[0].slot_id
    # Book that slot out of band so the chat booking loses the race.
    agent_id, start = tours.parse_slot_id(slot_id)
    store.book_tour(
        property_id="PROP-002", property_name="Riverside Lofts", agent_id=agent_id,
        agent_name="x", start=start.isoformat(timespec="seconds"),
        end=(start + timedelta(minutes=30)).isoformat(timespec="seconds"),
        prospect_name="Someone Else",
    )
    state = r1.state.model_copy(update={
        "prospect_name": "Alex Kim",
        "prospect_phone": "555-123-4567",
        "prospect_email": "alex.kim@example.com",
    })
    r2 = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(role="user", content="book it")],
            property_id="PROP-002", state=state, selected_slot_id=slot_id,
        ),
        now=MON,
    )
    assert r2.booking is None
    assert r2.state.phase == "proposing"
    assert "just booked" in r2.reply.lower()


def test_chat_cancel_existing_booking(db):
    booking = store.book_tour(
        property_id="PROP-002", property_name="Riverside Lofts",
        agent_id="AGENT-01", agent_name="Maria Lopez",
        start="2026-07-27T09:00:00", end="2026-07-27T09:30:00",
        prospect_name="Alex Kim", prospect_email="alex.kim@example.com",
    )
    r1 = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(role="user", content="cancel my tour booking")],
            property_id="PROP-002",
        ),
        now=MON,
    )
    # Must NOT fall through to proposing new slots (the original bug).
    assert r1.state.phase == "awaiting_cancel_email"
    assert not r1.proposed_slots
    assert "email" in r1.reply.lower()

    r2 = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(role="user", content="alex.kim@example.com")],
            property_id="PROP-002", state=r1.state,
        ),
        now=MON,
    )
    assert r2.state.phase == "greeting"
    assert "cancelled" in r2.reply.lower()
    assert store.get_booking(booking["id"])["status"] == "cancelled"


def test_chat_cancel_with_email_in_same_message(db):
    booking = store.book_tour(
        property_id="PROP-002", property_name="Riverside Lofts",
        agent_id="AGENT-01", agent_name="Maria Lopez",
        start="2026-07-27T09:00:00", end="2026-07-27T09:30:00",
        prospect_name="Alex Kim", prospect_email="alex.kim@example.com",
    )
    r = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(
                role="user",
                content="please cancel my reservation, my email is alex.kim@example.com",
            )],
            property_id="PROP-002",
        ),
        now=MON,
    )
    assert r.state.phase == "greeting"
    assert "cancelled" in r.reply.lower()
    assert store.get_booking(booking["id"])["status"] == "cancelled"


def test_chat_cancel_no_matching_booking(db):
    r = tours_chat.handle(
        TourChatRequest(
            messages=[ChatMessage(
                role="user", content="cancel my tour, email nobody@nowhere.com",
            )],
            property_id="PROP-002",
        ),
        now=MON,
    )
    assert r.state.phase == "greeting"
    assert "couldn't find" in r.reply.lower()


def test_chat_never_raises_on_bad_input(db):
    r = tours_chat.handle(
        TourChatRequest(messages=[], property_id="NOPE-999"), now=MON
    )
    assert r.source == "rules"
    assert r.booking is None
