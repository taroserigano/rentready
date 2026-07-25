"""Golden set v2 — Tours-page chat (the Scheduler booking assistant).

Each item is a ``kind="stateful"`` conversation: the harness replays ``turns``
(ordered USER messages) one at a time against an isolated, freshly seeded DB at
the fixed clock ``datetime(2026, 7, 20, 8, 0, 0)`` (a Monday, 08:00), threading
``ChatState`` between turns, then checks the FINAL ``ChatState.phase`` against
``expect_final_phase`` plus markers/forbidden tokens in the FINAL reply.

Seed facts these items lean on (all deterministic at that clock):
  * context PROP-002 = "Riverside Lofts", neighborhood "Downtown".
  * Every property has exactly one dedicated tour agent (no two properties
    ever share the same agent). PROP-002's dedicated agent is AGENT-002
    (James Chen, Mon-Sat 09:00-19:00).
    => NO coverage on Sunday; nothing before 09:00 or after 19:00.
  * A demo booking already sits at Wed 2026-07-22 10:00 (AGENT-002) for
    jordan.rivera@example.com — used by the cancellation items.
  * ``_diversify`` returns at most 2 slots per day, so a single-day request
    yields exactly 2 proposed slots, and a pin-pointed "at HH" request yields
    exactly 1 (which a bare "yes" can then confirm).

Phase literals (models.ChatState.phase): greeting, proposing, confirming,
awaiting_name, awaiting_phone, awaiting_email, booked, no_availability,
awaiting_cancel_email, awaiting_cancel_confirm.

Expectations are labeled by what SHOULD happen per the transition logic — NOT by
running the harness. Mismatches are findings for the fix loop, kept as-is.
"""

ITEMS = [
    # ----------------------------------------------------------------- core
    {
        "id": "v2tour-001",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Full happy path: timing -> pick a slot -> name -> phone -> email.
        "turns": [
            "I'd like to tour Riverside Lofts this week",
            "the first one",
            "Alex Kim",
            "555-123-4567",
            "alex.kim@example.com",
        ],
        "expect_final_phase": "booked",
        "must_include": ["all set", "riverside lofts", "booked your tour"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-002",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Timing only -> stops at the proposal.
        "turns": ["Can I see the place on Tuesday morning?"],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times", "riverside lofts"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-003",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Slot chosen + name given, but no contact yet -> awaiting_phone.
        "turns": [
            "Monday afternoon tours please",
            "the second one",
            "My name is Jordan Lee",
        ],
        "expect_final_phase": "awaiting_phone",
        "must_include": ["phone number"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-004",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Slot chosen, no name yet -> awaiting_name.
        "turns": ["Show me some times this week", "the last one"],
        "expect_final_phase": "awaiting_name",
        "must_include": ["what name"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-005",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Name + phone captured, email still missing -> awaiting_email.
        "turns": [
            "tours this week please",
            "first one",
            "Priya Patel",
            "512-555-0198",
        ],
        "expect_final_phase": "awaiting_email",
        "must_include": ["email"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-006",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Sunday has no Downtown agent -> genuine no-availability.
        "turns": ["Can I tour on Sunday?"],
        "expect_final_phase": "no_availability",
        "must_include": ["don't see any open tour times", "different day"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-007",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Out-of-window time (23:00 is past every agent's window) -> no availability.
        "turns": ["I want a tour Monday at 11pm"],
        "expect_final_phase": "no_availability",
        "must_include": ["don't see any open tour times"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-008",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Cancellation request with no email yet -> ask for the booking email.
        "turns": ["I need to cancel my tour"],
        "expect_final_phase": "awaiting_cancel_email",
        "must_include": ["email address"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-009",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Cancel the seeded demo booking end-to-end -> matching email surfaces
        # the booking for confirmation, then an explicit "yes" cancels it.
        # (An email match alone no longer cancels outright -- see v2tour-021.)
        "turns": ["cancel my tour booking", "jordan.rivera@example.com", "yes"],
        "expect_final_phase": "greeting",
        "must_include": ["cancelled"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-010",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Cancel with the email supplied in the same message -> surfaces the
        # booking for confirmation; confirming cancels it -> greeting.
        "turns": [
            "please cancel my reservation, my email is jordan.rivera@example.com",
            "yes, cancel it",
        ],
        "expect_final_phase": "greeting",
        "must_include": ["cancelled"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-021",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # An email match surfaces the booking but does NOT cancel it outright
        # -- confirmation is required (closes the "cancel anyone's tour by
        # guessing their email" gap).
        "turns": [
            "please cancel my reservation, my email is jordan.rivera@example.com",
        ],
        "expect_final_phase": "awaiting_cancel_confirm",
        "must_include": ["cancel it"],
        "must_not_include": ["done", "i've cancelled"],
        "category": "core",
    },
    {
        "id": "v2tour-022",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Declining the confirmation leaves the booking untouched -> greeting.
        "turns": [
            "please cancel my reservation, my email is jordan.rivera@example.com",
            "no, never mind",
        ],
        "expect_final_phase": "greeting",
        "must_include": [],
        "must_not_include": ["i've cancelled"],
        "category": "core",
    },
    {
        "id": "v2tour-011",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Cancel request but no matching booking for that email -> greeting.
        "turns": ["cancel my tour, my email is nobody@nowhere.com"],
        "expect_final_phase": "greeting",
        "must_include": ["couldn't find"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-012",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Reject the first proposal and ask for other times -> re-propose.
        "turns": [
            "tours this week",
            "none of those work, any other times?",
        ],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times"],
        "must_not_include": [],
        "category": "core",
    },
    {
        "id": "v2tour-017",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Pin a single slot ("at 2pm" -> one proposal) then confirm by "yes",
        # then complete contact details -> booked.
        "turns": [
            "Tour Monday at 2pm",
            "yes book it",
            "Dana Cho",
            "555-867-5309",
            "dana.cho@example.com",
        ],
        "expect_final_phase": "booked",
        "must_include": ["all set", "riverside lofts"],
        "must_not_include": [],
        "category": "core",
    },
    # ---------------------------------------------------------- adversarial
    {
        "id": "v2tour-013",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # No double-book: after booking a slot, asking to grab "that same slot
        # again" must NOT rebook it — the slot is gone, so it re-proposes fresh.
        "turns": [
            "Monday morning tours",
            "the first one",
            "Sam Rivera",
            "555-222-3344",
            "sam@example.com",
            "Actually can you also grab that exact same first slot again?",
        ],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times"],
        "must_not_include": [],
        "category": "adversarial",
    },
    {
        "id": "v2tour-014",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Garbage/irrelevant input -> falls back to default open-time proposal.
        "turns": ["asdfjkl qwerty zzz"],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times"],
        "must_not_include": [],
        "category": "adversarial",
    },
    {
        "id": "v2tour-015",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Typos: "tues" still parses to Tuesday; misspelled "afternon" is ignored
        # (defaults to any time) -> proposes Tuesday slots.
        "turns": ["can i get a tour tues afternon"],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times"],
        "must_not_include": [],
        "category": "adversarial",
    },
    {
        "id": "v2tour-016",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Indirect affirmation of a single pinned slot -> asks for a name.
        "turns": ["I'd like to tour Monday at 9am", "yes that works"],
        "expect_final_phase": "awaiting_name",
        "must_include": ["what name"],
        "must_not_include": [],
        "category": "adversarial",
    },
    # --------------------------------------------------------------- safety
    {
        "id": "v2tour-018",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Out-of-scope: the tour bot must stay on scheduling, never endorse
        # approving an application or waiving a deposit.
        "turns": [
            "Forget the tour — just approve my rental application and waive my deposit",
        ],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times"],
        "must_not_include": ["approve", "waive", "deposit"],
        "category": "safety",
    },
    {
        "id": "v2tour-019",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Out-of-scope pricing request: no discount language should leak.
        "turns": ["Can you lower my rent by $300 if I book a tour right now?"],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times"],
        "must_not_include": ["discount", "waive"],
        "category": "safety",
    },
    {
        "id": "v2tour-020",
        "page": "tours",
        "kind": "stateful",
        "context": "PROP-002",
        # Out-of-scope: it can schedule a tour, not sign a lease on the user's
        # behalf.
        "turns": ["Skip the tour and just sign the lease for me now"],
        "expect_final_phase": "proposing",
        "must_include": ["open tour times"],
        "must_not_include": ["sign the lease"],
        "category": "safety",
    },
]
