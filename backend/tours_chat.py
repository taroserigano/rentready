"""The Tour Scheduler reasoning brain.

``handle(req) -> TourChatResponse`` runs a deterministic conversation for
booking a property tour:

  greeting -> proposing -> awaiting_name -> booked
                  ^                            |
                  +---- SlotTaken re-propose --+

The deterministic path is PRIMARY and always works offline. An optional
grounded LLM pass may only rewrite the wording and narrow the proposed slots
to a subset of the candidates we computed; on ANY error (the Anthropic key is
rate-limited, so ``.invoke()`` raises) we keep the templated copy and set
``source="rules"``. ``handle`` NEVER raises: the outermost try/except returns
a safe templated response.

Everything that depends on "now" takes an injected datetime so tests are
deterministic.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta

import graph
import store
import tours
from models import ChatState, Slot, TourBooking, TourChatRequest, TourChatResponse

MAX_PROPOSE = 6

_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_ORDINALS = {
    "first": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2,
    "fourth": 3, "4th": 3, "four": 3,
    "last": -1,
}

_AFFIRM = re.compile(
    r"\b(yes|yeah|yep|yup|ok|okay|sure|sounds good|perfect|great|book it|"
    r"confirm|confirmed|that works|works for me|lets do it|let's do it|"
    r"go ahead|please do)\b",
    re.IGNORECASE,
)
_REJECT = re.compile(
    r"\b(no|nope|nah|not that|different|another|something else|other time|"
    r"other times|anything else|never mind|nevermind|change)\b",
    re.IGNORECASE,
)

# Words that signal the user is (re)stating a time preference, not giving a name.
_TIMING_WORDS = re.compile(
    r"\b(today|tomorrow|weekend|week|morning|afternoon|evening|night|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thurs|fri|sat|sun|"
    r"after|before|noon|midnight|am|pm|\d{1,2}(:\d{2})?)\b",
    re.IGNORECASE,
)

_NAME_STOP = {
    "yes", "no", "ok", "okay", "sure", "the", "a", "an", "please", "thanks",
    "thank", "you", "hi", "hello", "hey", "book", "tour", "want", "need",
    "would", "like", "to", "for", "me", "my", "name", "is", "im", "i", "am",
    "that", "works", "sounds", "good", "great", "perfect", "confirm",
}


def _now_dt() -> datetime:
    """Anchor for "now"; overridable in tests via monkeypatch or the ``now``
    argument to ``handle``."""
    return datetime.now()


# ---------------------------------------------------------------------------
# Small parsers
# ---------------------------------------------------------------------------


def _parse_time_token(text: str) -> int | None:
    """Find a clock time in ``text`` -> minutes since midnight, or None.

    Handles "5pm", "5 pm", "10am", "2:30pm", "17:00", "noon", "midnight".
    """
    t = text.lower()
    if "noon" in t and "afternoon" not in t.replace("noon", "afternoon", 0):
        # plain "noon"
        if re.search(r"\bnoon\b", t):
            return 12 * 60
    if re.search(r"\bmidnight\b", t):
        return 0
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        hour = int(m.group(1)) % 12
        minute = int(m.group(2) or 0)
        if m.group(3) == "pm":
            hour += 12
        return hour * 60 + minute
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)  # 24h "17:00"
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def parse_timing(text: str, now: datetime) -> dict:
    """Deterministic NL timing parser.

    Returns {date_from: date, date_to: date, time_of_day: str,
             after_min/before_min/at_min: int|None}. Falls back to the next
    7 days, any time, when nothing parses.
    """
    t = (text or "").lower()
    today = now.date()
    date_from = today
    date_to = today + timedelta(days=tours.DEFAULT_HORIZON_DAYS)
    tod = "any"
    after_min = before_min = at_min = None

    if "morning" in t:
        tod = "morning"
    elif "afternoon" in t:
        tod = "afternoon"
    elif "evening" in t or "tonight" in t or re.search(r"\bnight\b", t):
        tod = "evening"

    matched_range = False
    if "day after tomorrow" in t:
        date_from = date_to = today + timedelta(days=2)
        matched_range = True
    elif "tomorrow" in t:
        date_from = date_to = today + timedelta(days=1)
        matched_range = True
    elif "today" in t:
        date_from = date_to = today
        matched_range = True
    elif "weekend" in t:
        # upcoming Saturday + Sunday
        days_to_sat = (5 - today.weekday()) % 7
        sat = today + timedelta(days=days_to_sat)
        date_from, date_to = sat, sat + timedelta(days=1)
        matched_range = True
    elif "next week" in t:
        days_to_next_mon = (7 - today.weekday()) % 7 or 7
        nm = today + timedelta(days=days_to_next_mon)
        date_from, date_to = nm, nm + timedelta(days=6)
        matched_range = True
    elif "this week" in t:
        date_from = today
        date_to = today + timedelta(days=(6 - today.weekday()))
        matched_range = True

    if not matched_range:
        for name, idx in _WEEKDAYS.items():
            if re.search(rf"\b{name}\b", t):
                delta = (idx - today.weekday()) % 7
                target = today + timedelta(days=delta)
                date_from = date_to = target
                matched_range = True
                break

    # after/before/at <time>
    after_m = re.search(r"\b(after|from|starting)\b", t)
    before_m = re.search(r"\b(before|by|until|til|till)\b", t)
    at_m = re.search(r"\b(at|around)\b", t)
    tv = _parse_time_token(t)
    if tv is not None:
        if after_m:
            after_min = tv
        elif before_m:
            before_min = tv
        elif at_m:
            at_min = tv

    return {
        "date_from": date_from,
        "date_to": date_to,
        "time_of_day": tod,
        "after_min": after_min,
        "before_min": before_min,
        "at_min": at_min,
    }


def has_timing(text: str) -> bool:
    return bool(_TIMING_WORDS.search(text or ""))


def is_affirmation(text: str) -> bool:
    return bool(_AFFIRM.search(text or ""))


def is_rejection(text: str) -> bool:
    return bool(_REJECT.search(text or ""))


def detect_selection(text: str, proposed: list) -> str | None:
    """Return the slot_id the user is selecting from ``proposed`` (list of
    Slot or dict), via ordinal ("first"/"last"), or a time echo ("2pm"). None
    if no clear selection."""
    if not proposed:
        return None
    slots = [_slot_dict(s) for s in proposed]

    t = (text or "").lower()
    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            try:
                return slots[idx]["slot_id"]
            except IndexError:
                return None

    tv = _parse_time_token(t)
    if tv is not None:
        matches = []
        for s in slots:
            start = datetime.fromisoformat(s["start"])
            if start.hour * 60 + start.minute == tv:
                matches.append(s["slot_id"])
        if len(matches) == 1:
            return matches[0]
    return None


def extract_name(text: str) -> str:
    """Pull a person's name out of a message. Explicit patterns first, then a
    bare short capitalized/word phrase. Returns "" if none found."""
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(
        r"\b(?:my name is|i am|i'm|im|this is|it's|its|name's|name is|call me)\s+"
        r"([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})",
        t,
        re.IGNORECASE,
    )
    if m:
        return _titlecase(m.group(1))

    if has_timing(t):  # "tomorrow afternoon" is a time preference, not a name
        return ""
    cleaned = re.sub(r"[.,!?]", " ", t).strip()
    tokens = cleaned.split()
    if 1 <= len(tokens) <= 3 and all(
        re.fullmatch(r"[A-Za-z][A-Za-z'\-]+", tok) for tok in tokens
    ):
        if not any(tok.lower() in _NAME_STOP for tok in tokens):
            return _titlecase(cleaned)
    return ""


def _titlecase(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------


def _slot_dict(s) -> dict:
    if isinstance(s, Slot):
        return s.model_dump()
    if hasattr(s, "model_dump"):
        return s.model_dump()
    return dict(s)


def _open_to_slot(o: tours.OpenSlot) -> Slot:
    return Slot(
        slot_id=o.slot_id,
        property_id=o.property_id,
        start=o.start.isoformat(timespec="seconds"),
        end=o.end.isoformat(timespec="seconds"),
        agent_id=o.agent_id,
        agent_name=o.agent_name,
        label=o.label,
    )


def _property(property_id: str) -> dict | None:
    for p in graph.load_properties():
        if p.get("id") == property_id:
            return p
    return None


def _propose_slots(property_id: str, area: str, timing: dict, now: datetime) -> list[Slot]:
    agents = store.list_agents()
    windows = store.list_windows()
    bookings = store.list_bookings(status="booked")
    opens = tours.open_slots(
        property_id=property_id,
        area=area,
        agents=agents,
        windows=windows,
        bookings=bookings,
        date_from=timing["date_from"],
        date_to=timing["date_to"],
        time_of_day=timing["time_of_day"],
        now=now,
        after_min=timing.get("after_min"),
        before_min=timing.get("before_min"),
        at_min=timing.get("at_min"),
    )
    return [_open_to_slot(o) for o in _diversify(opens)]


def _diversify(opens: list, limit: int = MAX_PROPOSE, max_per_day: int = 2) -> list:
    """Spread proposed slots ACROSS DAYS instead of returning the first N
    consecutive slots of the earliest day.

    ``opens`` arrives sorted by start, so a naive ``opens[:limit]`` yields six
    back-to-back half-hours on one day — useless for someone choosing when to
    visit. We group by day, sample a spread of times within each day (early /
    late / midday first), then round-robin across days so as many distinct days
    as possible appear before we ever show a second time on the same day.
    """
    if not opens:
        return []

    by_day: dict = {}
    order: list = []
    for o in opens:  # already sorted by start
        d = o.start.date()
        if d not in by_day:
            by_day[d] = []
            order.append(d)
        by_day[d].append(o)

    def sample(day_slots: list) -> list:
        n = len(day_slots)
        if n <= 1:
            return list(day_slots)
        picks, seen = [], set()
        for i in (0, n - 1, n // 2, n // 4, (3 * n) // 4):
            if 0 <= i < n and i not in seen:
                seen.add(i)
                picks.append(day_slots[i])
        for i in range(n):  # fill the rest in natural order
            if i not in seen:
                picks.append(day_slots[i])
        return picks

    sampled = {d: sample(by_day[d]) for d in order}
    result: list = []
    for rnd in range(max_per_day):  # one time per day per round
        for d in order:
            if len(result) >= limit:
                break
            if rnd < len(sampled[d]):
                result.append(sampled[d][rnd])
        if len(result) >= limit:
            break
    result.sort(key=lambda o: o.start)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handle(req: TourChatRequest, now: datetime | None = None) -> TourChatResponse:
    """Never raises. Returns a safe templated response on any internal error."""
    state = req.state or ChatState()
    if now is None:
        now = _now_dt()
    try:
        return _handle(req, state, now)
    except Exception as exc:  # noqa: BLE001 - the whole point: degrade, don't 500
        print(f"tours_chat.handle failed ({type(exc).__name__}: {exc}); safe reply.")
        return TourChatResponse(
            reply="Sorry, I hit a snag scheduling that. Could you tell me a day "
            "or time that works and I'll pull up open tour slots?",
            proposed_slots=[],
            booking=None,
            state=ChatState(phase="greeting", prospect_name=state.prospect_name),
            source="rules",
        )


def _last_user_message(req: TourChatRequest) -> str:
    for msg in reversed(req.messages or []):
        if msg.role == "user":
            return msg.content or ""
    return ""


def _handle(req: TourChatRequest, state: ChatState, now: datetime) -> TourChatResponse:
    last_user = _last_user_message(req)
    property_id = req.property_id

    if not property_id:
        return TourChatResponse(
            reply="Happy to set up a tour! Which property would you like to see?",
            proposed_slots=[],
            booking=None,
            state=ChatState(phase="greeting", prospect_name=state.prospect_name),
            source="rules",
        )

    prop = _property(property_id)
    if prop is None:
        return TourChatResponse(
            reply="I couldn't find that property. Could you pick one from the "
            "listings and I'll find open tour times?",
            proposed_slots=[],
            booking=None,
            state=ChatState(phase="greeting", prospect_name=state.prospect_name),
            source="rules",
        )
    area = (prop.get("neighborhood") or {}).get("name", "")
    property_name = prop.get("name", property_id)

    # ---- Resolve any slot the user is selecting -------------------------
    selected_slot_id = req.selected_slot_id or None
    if not selected_slot_id and state.last_proposed:
        selected_slot_id = detect_selection(last_user, state.last_proposed)
        if not selected_slot_id and is_affirmation(last_user) and not is_rejection(last_user):
            # Bare "yes" -> confirm the single proposed slot, or a pending one.
            if state.pending_slot_id:
                selected_slot_id = state.pending_slot_id
            elif len(state.last_proposed) == 1:
                selected_slot_id = _slot_dict(state.last_proposed[0])["slot_id"]

    # ---- awaiting_name phase -------------------------------------------
    if state.phase == "awaiting_name" and not selected_slot_id:
        if is_rejection(last_user) or has_timing(last_user):
            return _do_propose(property_id, area, property_name, last_user, now, state)
        name = extract_name(last_user)
        if name and state.pending_slot_id:
            return _do_book(
                state.pending_slot_id, property_id, property_name, name, area, now, state
            )
        return TourChatResponse(
            reply="Almost there — what name should I put the tour under?",
            proposed_slots=list(state.last_proposed),
            booking=None,
            state=ChatState(
                phase="awaiting_name",
                prospect_name=state.prospect_name,
                last_proposed=list(state.last_proposed),
                pending_slot_id=state.pending_slot_id,
            ),
            source="rules",
        )

    # ---- A slot was selected -> book it (or ask for a name) -------------
    if selected_slot_id:
        name = state.prospect_name or extract_name(last_user)
        if not name:
            slot = _find_slot(state.last_proposed, selected_slot_id)
            label = _slot_dict(slot)["label"] if slot else "that time"
            return TourChatResponse(
                reply=f"Great pick — {label}. What name should I put the tour under?",
                proposed_slots=list(state.last_proposed),
                booking=None,
                state=ChatState(
                    phase="awaiting_name",
                    prospect_name=state.prospect_name,
                    last_proposed=list(state.last_proposed),
                    pending_slot_id=selected_slot_id,
                ),
                source="rules",
            )
        return _do_book(selected_slot_id, property_id, property_name, name, area, now, state)

    # ---- Otherwise: (re)propose slots for the parsed timing -------------
    return _do_propose(property_id, area, property_name, last_user, now, state)


def _find_slot(proposed: list, slot_id: str):
    for s in proposed or []:
        if _slot_dict(s)["slot_id"] == slot_id:
            return s
    return None


def _do_book(
    slot_id: str,
    property_id: str,
    property_name: str,
    name: str,
    area: str,
    now: datetime,
    state: ChatState,
) -> TourChatResponse:
    try:
        agent_id, start = tours.parse_slot_id(slot_id)
    except ValueError:
        return _do_propose(property_id, area, property_name, "", now, state,
                           lead="Let me pull up open tour times.")

    agents = {a["id"]: a for a in store.list_agents()}
    agent = agents.get(agent_id)
    if agent is None:
        return _do_propose(property_id, area, property_name, "", now, state,
                           lead="That agent is no longer available. Here are open times:")

    end = start + timedelta(minutes=tours.DEFAULT_DURATION_MIN)
    try:
        row = store.book_tour(
            property_id=property_id,
            property_name=property_name,
            agent_id=agent_id,
            agent_name=agent["name"],
            start=start.isoformat(timespec="seconds"),
            end=end.isoformat(timespec="seconds"),
            prospect_name=name,
            duration_minutes=tours.DEFAULT_DURATION_MIN,
        )
    except store.SlotConflict:
        # SlotTaken -> re-propose fresh slots.
        resp = _do_propose(
            property_id, area, property_name, "", now, state,
            lead="Ah, that time was just booked by someone else. Here are the "
            "next open times:",
        )
        return resp

    booking = TourBooking(**row)
    return TourChatResponse(
        reply=f"You're all set, {name}! I booked your tour of {property_name} "
        f"on {tours.label(start)} with {agent['name']}. See you then!",
        proposed_slots=[],
        booking=booking,
        state=ChatState(phase="booked", prospect_name=name),
        source="rules",
    )


def _do_propose(
    property_id: str,
    area: str,
    property_name: str,
    last_user: str,
    now: datetime,
    state: ChatState,
    lead: str | None = None,
) -> TourChatResponse:
    timing = parse_timing(last_user, now)
    slots = _propose_slots(property_id, area, timing, now)

    if not slots:
        return TourChatResponse(
            reply=f"I don't see any open tour times for {property_name} in that "
            "window. Want me to check a different day or time?",
            proposed_slots=[],
            booking=None,
            state=ChatState(
                phase="no_availability", prospect_name=state.prospect_name
            ),
            source="rules",
        )

    base_lead = lead or (
        f"Here are some open tour times for {property_name} — tap one to grab it:"
    )
    reply = base_lead
    source = "rules"

    # Optional grounded LLM rewrite of the PROSE only. We keep the deterministic
    # day-spread slots for display (never let the model re-collapse them to one
    # day); it just supplies warmer wording. Degrades to the template on error.
    enhanced = _enhance_with_llm(property_name, base_lead, slots)
    if enhanced is not None:
        reply = enhanced[0]
        source = "anthropic"

    return TourChatResponse(
        reply=reply,
        proposed_slots=slots,
        booking=None,
        state=ChatState(
            phase="proposing",
            prospect_name=state.prospect_name,
            last_proposed=slots,
        ),
        source=source,
    )


# ---------------------------------------------------------------------------
# Optional LLM enhancement (grounded; never authoritative)
# ---------------------------------------------------------------------------

_ENHANCE_SYSTEM = (
    "You are RentReady's friendly leasing scheduler. You help a prospect pick "
    "a tour time. You are given a fixed list of CANDIDATE slots (each with a "
    "slot_id and a human label) that span one or more days. RULES:\n"
    "1. You may ONLY refer to times from the candidates. Never invent a time.\n"
    "2. Your `slot_ids` MUST be a subset of the candidate slot_ids.\n"
    "3. Keep the reply to one or two warm, plain sentences. Do NOT list "
    "specific times or name a single day in the prose (the UI renders the "
    "times as buttons). If the candidates cover several days, invite them to "
    "pick whichever day and time works — e.g. 'Here are some open times across "
    "the next few days.'\n"
    'Return ONLY JSON: {"reply": "...", "slot_ids": ["...", "..."]}'
)


def _enhance_with_llm(property_name: str, base_lead: str, slots: list[Slot]):
    """Return (reply, slots_subset) or None to keep the template."""
    try:
        from llm import get_langchain_llm

        llm = get_langchain_llm()
        if llm is None:
            return None
        candidates = [{"slot_id": s.slot_id, "label": s.label} for s in slots]
        raw = llm.invoke(
            [
                ("system", _ENHANCE_SYSTEM),
                (
                    "human",
                    f"Property: {property_name}\nCandidate slots (JSON):\n"
                    f"{json.dumps(candidates)}\n\nReturn the JSON now.",
                ),
            ]
        ).content
        if isinstance(raw, list):
            raw = "".join(str(b) for b in raw)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        reply = str(data.get("reply") or "").strip() or base_lead
        wanted = data.get("slot_ids") or []
        by_id = {s.slot_id: s for s in slots}
        subset = [by_id[sid] for sid in wanted if sid in by_id]
        if not subset:  # never let the LLM empty the list
            subset = slots
        return reply, subset
    except Exception as exc:  # noqa: BLE001
        print(f"tours_chat LLM enhance failed ({type(exc).__name__}: {exc}); rules.")
        return None
